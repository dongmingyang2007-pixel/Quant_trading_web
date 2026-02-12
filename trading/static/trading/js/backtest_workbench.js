(function () {
  'use strict';

  function getCookie(name) {
    const cookies = document.cookie ? document.cookie.split(';') : [];
    for (let i = 0; i < cookies.length; i += 1) {
      const cookie = cookies[i].trim();
      if (cookie.startsWith(name + '=')) {
        return decodeURIComponent(cookie.slice(name.length + 1));
      }
    }
    return '';
  }

  function isZh(lang) {
    return String(lang || '').toLowerCase().startsWith('zh');
  }

  function formatTs(raw) {
    if (!raw) return '-';
    if (typeof raw === 'number') {
      return new Date(raw * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
    }
    return String(raw);
  }

  function normalizeWorkspace(raw) {
    const value = String(raw || '').toLowerCase();
    if (value === 'shortterm') return 'trade';
    if (value === 'classic') return 'backtest';
    if (value === 'trade' || value === 'backtest' || value === 'review') return value;
    return 'trade';
  }

  document.addEventListener('DOMContentLoaded', function () {
    const workspaceSwitch =
      document.querySelector('[data-workbench-role="workspace-switch"]') ||
      document.querySelector('[data-shortterm-role="mode-switch"]');
    if (!workspaceSwitch) return;

    const tradePanel =
      document.querySelector('[data-workbench-role="trade-panel"]') ||
      document.querySelector('[data-shortterm-role="shortterm-panel"]');
    const backtestPanel =
      document.querySelector('[data-workbench-role="backtest-panel"]') ||
      document.querySelector('[data-shortterm-role="classic-root"]');
    const reviewPanel = document.querySelector('[data-workbench-role="review-panel"]');
    if (!tradePanel || !backtestPanel) return;

    const shorttermRoot = tradePanel.querySelector('[data-workbench-root]') || tradePanel.querySelector('[data-shortterm-root]');
    const modeButtons = Array.from(workspaceSwitch.querySelectorAll('[data-workbench-role="workspace-btn"], [data-shortterm-role="mode-btn"]'));
    const workspaceStorageKey = 'backtest.primary.workspace';
    let pollTimer = null;
    let shorttermLoaded = false;
    let currentTradingMode = 'paper';

    let workbenchEndpoint = '';
    let modeEndpoint = '';
    let orderEndpoint = '';
    let langIsZh = true;
    let csrftoken = getCookie('csrftoken');

    let refreshBtn = null;
    let statusEl = null;
    let templateButtons = [];
    let tradingModeButtons = [];
    let engineOnlineEl = null;
    let modeValueEl = null;
    let countUniverseEl = null;
    let countFlowEl = null;
    let riskGuardEl = null;
    let focusRowsEl = null;
    let signalRowsEl = null;
    let orderRowsEl = null;
    let manualOrderForm = null;
    let liveDialog = null;
    let liveCheck = null;
    let liveInput = null;
    let liveCancel = null;
    let liveSubmit = null;

    function t(zhText, enText) {
      return langIsZh ? zhText : enText;
    }

    function setStatus(text, isError) {
      if (!statusEl) return;
      statusEl.textContent = text || '';
      statusEl.style.color = isError ? '#b91c1c' : '#1d4ed8';
    }

    function updateWorkspaceButtons(workspace) {
      modeButtons.forEach((btn) => {
        const modeRaw = btn.dataset.workspace || btn.dataset.mode || '';
        const normalized = normalizeWorkspace(modeRaw);
        btn.classList.toggle('is-active', normalized === workspace);
      });
    }

    function setWorkspace(workspace, persist) {
      const resolved = normalizeWorkspace(workspace);
      const showTrade = resolved === 'trade';
      const showBacktest = resolved === 'backtest';
      const showReview = resolved === 'review';

      tradePanel.hidden = !showTrade;
      backtestPanel.hidden = !showBacktest;
      if (reviewPanel) reviewPanel.hidden = !showReview;

      updateWorkspaceButtons(resolved);

      if (persist) {
        try {
          window.localStorage.setItem(workspaceStorageKey, resolved);
        } catch (_error) {
          // ignore
        }
      }

      try {
        const url = new URL(window.location.href);
        url.searchParams.set('workspace', resolved);
        url.searchParams.delete('panel');
        window.history.replaceState({}, '', url.toString());
      } catch (_error) {
        // ignore
      }

      if (showTrade) {
        if (shorttermRoot && !shorttermLoaded) {
          fetchTradeWorkbench();
        }
        if (!pollTimer) {
          pollTimer = window.setInterval(fetchTradeWorkbench, 15000);
        }
      } else if (pollTimer) {
        window.clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    function workspaceFromUrlOrStorage() {
      try {
        const url = new URL(window.location.href);
        const workspaceRaw = String(url.searchParams.get('workspace') || '').trim();
        if (workspaceRaw) return normalizeWorkspace(workspaceRaw);
        const panelRaw = String(url.searchParams.get('panel') || '').trim();
        if (panelRaw) return normalizeWorkspace(panelRaw);
      } catch (_error) {
        // ignore
      }
      try {
        const saved = normalizeWorkspace(window.localStorage.getItem(workspaceStorageKey));
        if (saved) return saved;
      } catch (_error) {
        // ignore
      }
      return 'trade';
    }

    function clearRows(tbody) {
      if (!tbody) return;
      while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
    }

    function appendRow(tbody, values) {
      const tr = document.createElement('tr');
      values.forEach((value) => {
        const td = document.createElement('td');
        td.textContent = value == null || value === '' ? '-' : String(value);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }

    function renderRows(tbody, rows, mapFn, emptyColumns) {
      clearRows(tbody);
      if (!tbody) return;
      if (!Array.isArray(rows) || !rows.length) {
        appendRow(tbody, Array.from({ length: emptyColumns }).map(() => '-'));
        return;
      }
      rows.forEach((row, index) => appendRow(tbody, mapFn(row, index)));
    }

    function updateTradingModeButtons(mode) {
      tradingModeButtons.forEach((btn) => {
        const active = btn.dataset.mode === mode;
        btn.classList.toggle('btn-primary', active && mode === 'paper');
        btn.classList.toggle('btn-outline-primary', !active && btn.dataset.mode === 'paper');
        btn.classList.toggle('btn-danger', active && mode === 'live');
        btn.classList.toggle('btn-outline-danger', !active && btn.dataset.mode === 'live');
      });
    }

    function renderTradeWorkbench(payload) {
      shorttermLoaded = true;
      const data = payload && payload.trade ? payload.trade : payload || {};
      const engine = data.engine || {};
      const summary = data.summary || {};
      const trading = data.trading || {};
      const tradingState = trading.state || {};
      currentTradingMode = String(trading.mode || 'paper').toLowerCase() === 'live' ? 'live' : 'paper';

      if (engineOnlineEl) engineOnlineEl.textContent = engine.online ? t('在线', 'Online') : t('离线', 'Offline');
      if (modeValueEl) modeValueEl.textContent = currentTradingMode === 'live' ? 'Live' : 'Paper';
      if (countUniverseEl) countUniverseEl.textContent = `${summary.universe_count || 0} / ${summary.focus_count || 0}`;
      if (countFlowEl) countFlowEl.textContent = `${summary.signals_count || 0} / ${summary.orders_count || 0}`;
      if (riskGuardEl) {
        const guard = tradingState.risk_guard || trading.risk_guard || {};
        riskGuardEl.textContent = guard.reason ? String(guard.reason) : t('正常', 'Normal');
      }
      updateTradingModeButtons(currentTradingMode);

      renderRows(
        focusRowsEl,
        data.focus || [],
        (row, index) => [index + 1, row.symbol || '-', formatTs(row.since || row.since_ts || '')],
        3
      );
      renderRows(
        signalRowsEl,
        data.signals || [],
        (row) => [formatTs(row.timestamp), row.symbol || '-', row.signal || row.action || '-', row.score || '-'],
        4
      );
      renderRows(
        orderRowsEl,
        data.orders || [],
        (row) => [formatTs(row.timestamp), row.symbol || '-', row.action || '-', row.status || '-'],
        4
      );

      const stream = engine.stream_status ? ` · ${engine.stream_status}` : '';
      setStatus(
        engine.online
          ? t(`数据源：Alpaca${stream}`, `Source: Alpaca${stream}`)
          : t('Alpaca 暂无数据/连接异常', 'Alpaca unavailable / connection issue'),
        !engine.online
      );
    }

    async function fetchTradeWorkbench() {
      if (!workbenchEndpoint) return;
      setStatus(t('正在刷新短线数据…', 'Refreshing short-term data…'), false);
      try {
        const endpoint = new URL(workbenchEndpoint, window.location.origin);
        if (!endpoint.searchParams.get('workspace')) {
          endpoint.searchParams.set('workspace', 'trade');
        }
        const response = await fetch(endpoint.toString(), {
          credentials: 'same-origin',
          headers: { Accept: 'application/json' },
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          if (payload && payload.error_code === 'alpaca_unavailable') {
            setStatus(t('Alpaca 暂无数据/连接异常', 'Alpaca unavailable / connection issue'), true);
            return;
          }
          throw new Error(payload.message || payload.error || payload.error_code || 'workbench_failed');
        }
        renderTradeWorkbench(payload || {});
      } catch (_error) {
        setStatus(t('短线工作台加载失败', 'Short-term workbench failed to load'), true);
      }
    }

    async function postTradingMode(payload) {
      if (!modeEndpoint) throw new Error('mode_endpoint_missing');
      const response = await fetch(modeEndpoint, {
        method: 'POST',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
          'X-CSRFToken': csrftoken,
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const err = new Error(data.message || data.error || data.error_code || 'mode_update_failed');
        err.payload = data;
        throw err;
      }
      return data;
    }

    async function applyTemplate(key) {
      setStatus(t('模板应用中…', 'Applying template…'), false);
      try {
        await postTradingMode({ template_key: key, mode: 'paper' });
        await fetchTradeWorkbench();
        setStatus(t('模板已更新', 'Template updated'), false);
      } catch (_error) {
        setStatus(t('模板应用失败', 'Failed to apply template'), true);
      }
    }

    async function switchTradingMode(mode, extra) {
      setStatus(t('模式切换中…', 'Switching mode…'), false);
      try {
        await postTradingMode(Object.assign({ mode: mode }, extra || {}));
        await fetchTradeWorkbench();
        setStatus(mode === 'live' ? t('已切换为 Live', 'Switched to Live') : t('已切换为 Paper', 'Switched to Paper'), false);
      } catch (error) {
        const payload = error.payload || {};
        if (payload.error_code === 'live_confirmation_required') {
          setStatus(t('Live 需要双确认', 'Live requires double confirmation'), true);
        } else {
          setStatus(t('模式切换失败', 'Failed to switch mode'), true);
        }
      }
    }

    async function submitManualOrder(side, symbol, notional) {
      const response = await fetch(orderEndpoint, {
        method: 'POST',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
          'X-CSRFToken': csrftoken,
        },
        body: JSON.stringify({ symbol: symbol, side: side, notional: notional }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        throw new Error(data.message || data.error || 'order_failed');
      }
      return data;
    }

    if (shorttermRoot) {
      workbenchEndpoint = shorttermRoot.dataset.workbenchEndpoint || '';
      modeEndpoint = shorttermRoot.dataset.modeEndpoint || '';
      orderEndpoint = shorttermRoot.dataset.orderEndpoint || '';
      langIsZh = isZh(shorttermRoot.dataset.lang);
      csrftoken = getCookie('csrftoken');

      refreshBtn = shorttermRoot.querySelector('[data-shortterm-role="refresh"]');
      statusEl = shorttermRoot.querySelector('[data-shortterm-role="status"]');
      templateButtons = Array.from(shorttermRoot.querySelectorAll('[data-shortterm-role="template"]'));
      tradingModeButtons = Array.from(shorttermRoot.querySelectorAll('[data-shortterm-role="mode"]'));
      engineOnlineEl = shorttermRoot.querySelector('[data-shortterm-role="engine-online"]');
      modeValueEl = shorttermRoot.querySelector('[data-shortterm-role="mode-value"]');
      countUniverseEl = shorttermRoot.querySelector('[data-shortterm-role="counts-universe"]');
      countFlowEl = shorttermRoot.querySelector('[data-shortterm-role="counts-flow"]');
      riskGuardEl = shorttermRoot.querySelector('[data-shortterm-role="risk-guard"]');
      focusRowsEl = shorttermRoot.querySelector('[data-shortterm-role="focus-rows"]');
      signalRowsEl = shorttermRoot.querySelector('[data-shortterm-role="signal-rows"]');
      orderRowsEl = shorttermRoot.querySelector('[data-shortterm-role="order-rows"]');
      manualOrderForm = shorttermRoot.querySelector('[data-shortterm-role="manual-order-form"]');
      liveDialog = shorttermRoot.querySelector('[data-shortterm-role="live-dialog"]');
      liveCheck = shorttermRoot.querySelector('[data-shortterm-role="live-confirm-check"]');
      liveInput = shorttermRoot.querySelector('[data-shortterm-role="live-confirm-input"]');
      liveCancel = shorttermRoot.querySelector('[data-shortterm-role="live-cancel"]');
      liveSubmit = shorttermRoot.querySelector('[data-shortterm-role="live-submit"]');

      if (refreshBtn) refreshBtn.addEventListener('click', fetchTradeWorkbench);

      templateButtons.forEach((btn) => {
        btn.addEventListener('click', function () {
          const key = btn.dataset.template;
          if (key) applyTemplate(key);
        });
      });

      tradingModeButtons.forEach((btn) => {
        btn.addEventListener('click', function () {
          const target = btn.dataset.mode;
          if (target === 'live') {
            if (liveCheck) liveCheck.checked = false;
            if (liveInput) liveInput.value = '';
            if (liveDialog && typeof liveDialog.showModal === 'function') {
              liveDialog.showModal();
            } else {
              setStatus(t('当前浏览器不支持确认弹层', 'This browser cannot show live confirmation dialog'), true);
            }
            return;
          }
          switchTradingMode('paper');
        });
      });

      if (liveCancel) {
        liveCancel.addEventListener('click', function () {
          if (liveDialog && typeof liveDialog.close === 'function') liveDialog.close();
        });
      }

      if (liveSubmit) {
        liveSubmit.addEventListener('click', function () {
          const checked = !!(liveCheck && liveCheck.checked);
          const typed = liveInput ? String(liveInput.value || '').trim().toUpperCase() : '';
          if (!checked || typed !== 'LIVE') {
            setStatus(t('请勾选确认并输入 LIVE', 'Please check the confirmation and type LIVE'), true);
            return;
          }
          if (liveDialog && typeof liveDialog.close === 'function') liveDialog.close();
          switchTradingMode('live', { confirm_live: true, confirm_phrase: 'LIVE' });
        });
      }

      if (manualOrderForm) {
        manualOrderForm.addEventListener('submit', async function (event) {
          event.preventDefault();
          const submitter = event.submitter;
          const side = submitter && submitter.dataset.side ? submitter.dataset.side : '';
          const formData = new FormData(manualOrderForm);
          const symbol = String(formData.get('symbol') || '').trim().toUpperCase();
          const notional = Number.parseFloat(String(formData.get('notional') || '0'));
          if (!symbol || !side || !Number.isFinite(notional) || notional <= 0) {
            setStatus(t('请填写有效订单参数', 'Please fill valid order parameters'), true);
            return;
          }
          setStatus(t('订单提交中…', 'Submitting order…'), false);
          try {
            await submitManualOrder(side, symbol, notional);
            setStatus(t('订单已提交', 'Order submitted'), false);
            fetchTradeWorkbench();
            manualOrderForm.reset();
          } catch (_error) {
            setStatus(t('下单失败，请检查 Alpaca 连接', 'Order failed, check Alpaca connection'), true);
          }
        });
      }
    }

    modeButtons.forEach((btn) => {
      btn.addEventListener('click', function () {
        const workspace = normalizeWorkspace(btn.dataset.workspace || btn.dataset.mode);
        setWorkspace(workspace, true);
      });
    });

    setWorkspace(workspaceFromUrlOrStorage(), false);
  });
})();
