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

  document.addEventListener('DOMContentLoaded', function () {
    const modeSwitch = document.querySelector('[data-shortterm-role="mode-switch"]');
    const shorttermPanel = document.querySelector('[data-shortterm-role="shortterm-panel"]');
    const classicRoot = document.querySelector('[data-shortterm-role="classic-root"]');
    if (!modeSwitch || !shorttermPanel || !classicRoot) return;

    const shorttermRoot = shorttermPanel.querySelector('[data-shortterm-root]');
    if (!shorttermRoot) return;

    const workbenchEndpoint = shorttermRoot.dataset.workbenchEndpoint;
    const modeEndpoint = shorttermRoot.dataset.modeEndpoint;
    const orderEndpoint = shorttermRoot.dataset.orderEndpoint;
    const langIsZh = isZh(shorttermRoot.dataset.lang);
    const csrftoken = getCookie('csrftoken');

    const modeButtons = Array.from(modeSwitch.querySelectorAll('[data-shortterm-role="mode-btn"]'));
    const refreshBtn = shorttermRoot.querySelector('[data-shortterm-role="refresh"]');
    const statusEl = shorttermRoot.querySelector('[data-shortterm-role="status"]');
    const templateButtons = Array.from(shorttermRoot.querySelectorAll('[data-shortterm-role="template"]'));
    const tradingModeButtons = Array.from(shorttermRoot.querySelectorAll('[data-shortterm-role="mode"]'));

    const engineOnlineEl = shorttermRoot.querySelector('[data-shortterm-role="engine-online"]');
    const modeValueEl = shorttermRoot.querySelector('[data-shortterm-role="mode-value"]');
    const countUniverseEl = shorttermRoot.querySelector('[data-shortterm-role="counts-universe"]');
    const countFlowEl = shorttermRoot.querySelector('[data-shortterm-role="counts-flow"]');
    const riskGuardEl = shorttermRoot.querySelector('[data-shortterm-role="risk-guard"]');

    const focusRowsEl = shorttermRoot.querySelector('[data-shortterm-role="focus-rows"]');
    const signalRowsEl = shorttermRoot.querySelector('[data-shortterm-role="signal-rows"]');
    const orderRowsEl = shorttermRoot.querySelector('[data-shortterm-role="order-rows"]');

    const manualOrderForm = shorttermRoot.querySelector('[data-shortterm-role="manual-order-form"]');

    const liveDialog = shorttermRoot.querySelector('[data-shortterm-role="live-dialog"]');
    const liveCheck = shorttermRoot.querySelector('[data-shortterm-role="live-confirm-check"]');
    const liveInput = shorttermRoot.querySelector('[data-shortterm-role="live-confirm-input"]');
    const liveCancel = shorttermRoot.querySelector('[data-shortterm-role="live-cancel"]');
    const liveSubmit = shorttermRoot.querySelector('[data-shortterm-role="live-submit"]');

    const PANEL_STORAGE_KEY = 'backtest.primary.panel';
    let currentMode = 'paper';
    let shorttermLoaded = false;
    let pollTimer = null;

    function t(zhText, enText) {
      return langIsZh ? zhText : enText;
    }

    function setStatus(text, isError) {
      if (!statusEl) return;
      statusEl.textContent = text || '';
      statusEl.style.color = isError ? '#b91c1c' : '#1d4ed8';
    }

    function setPanelMode(mode, persist) {
      const resolved = mode === 'classic' ? 'classic' : 'shortterm';
      const showShortterm = resolved === 'shortterm';
      shorttermPanel.hidden = !showShortterm;
      classicRoot.hidden = showShortterm;
      modeButtons.forEach((btn) => {
        const active = btn.dataset.mode === resolved;
        btn.classList.toggle('is-active', active);
      });
      if (persist) {
        try {
          window.localStorage.setItem(PANEL_STORAGE_KEY, resolved);
        } catch (error) {
          // ignore
        }
      }
      try {
        const url = new URL(window.location.href);
        url.searchParams.set('panel', resolved);
        window.history.replaceState({}, '', url.toString());
      } catch (error) {
        // ignore
      }
      if (showShortterm) {
        if (!shorttermLoaded) {
          fetchWorkbench();
        }
        if (!pollTimer) {
          pollTimer = window.setInterval(fetchWorkbench, 15000);
        }
      } else if (pollTimer) {
        window.clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    function modeFromUrlOrStorage() {
      const url = new URL(window.location.href);
      const panel = (url.searchParams.get('panel') || '').toLowerCase();
      if (panel === 'shortterm' || panel === 'classic') return panel;
      try {
        const saved = window.localStorage.getItem(PANEL_STORAGE_KEY);
        if (saved === 'shortterm' || saved === 'classic') return saved;
      } catch (error) {
        // ignore
      }
      return 'shortterm';
    }

    function clearRows(tbody) {
      if (!tbody) return;
      while (tbody.firstChild) {
        tbody.removeChild(tbody.firstChild);
      }
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
      rows.forEach((row, index) => {
        appendRow(tbody, mapFn(row, index));
      });
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

    function renderWorkbench(payload) {
      shorttermLoaded = true;
      const engine = payload.engine || {};
      const summary = payload.summary || {};
      const trading = payload.trading || {};
      const tradingState = trading.state || {};
      currentMode = String(trading.mode || 'paper').toLowerCase() === 'live' ? 'live' : 'paper';

      if (engineOnlineEl) {
        engineOnlineEl.textContent = engine.online ? t('在线', 'Online') : t('离线', 'Offline');
      }
      if (modeValueEl) {
        modeValueEl.textContent = currentMode === 'live' ? 'Live' : 'Paper';
      }
      if (countUniverseEl) {
        countUniverseEl.textContent = `${summary.universe_count || 0} / ${summary.focus_count || 0}`;
      }
      if (countFlowEl) {
        countFlowEl.textContent = `${summary.signals_count || 0} / ${summary.orders_count || 0}`;
      }
      if (riskGuardEl) {
        const guard = tradingState.risk_guard || {};
        riskGuardEl.textContent = guard.reason ? String(guard.reason) : t('正常', 'Normal');
      }
      updateTradingModeButtons(currentMode);

      renderRows(
        focusRowsEl,
        payload.focus || [],
        (row, index) => [index + 1, row.symbol || '-', formatTs(row.since || row.since_ts || '')],
        3
      );
      renderRows(
        signalRowsEl,
        payload.signals || [],
        (row) => [formatTs(row.timestamp), row.symbol || '-', row.signal || row.action || '-', row.score || '-'],
        4
      );
      renderRows(
        orderRowsEl,
        payload.orders || [],
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

    async function fetchWorkbench() {
      if (!workbenchEndpoint) return;
      setStatus(t('正在刷新短线数据…', 'Refreshing short-term data…'), false);
      try {
        const response = await fetch(workbenchEndpoint, {
          credentials: 'same-origin',
          headers: { Accept: 'application/json' },
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.message || payload.error || 'workbench_failed');
        }
        renderWorkbench(payload || {});
      } catch (error) {
        setStatus(t('短线工作台加载失败', 'Short-term workbench failed to load'), true);
      }
    }

    async function postMode(payload) {
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
      const data = await response.json();
      if (!response.ok) {
        const err = new Error(data.message || data.error || 'mode_update_failed');
        err.payload = data;
        throw err;
      }
      return data;
    }

    async function applyTemplate(key) {
      if (!modeEndpoint) return;
      setStatus(t('模板应用中…', 'Applying template…'), false);
      try {
        await postMode({ template_key: key, mode: 'paper' });
        await fetchWorkbench();
        setStatus(t('模板已更新', 'Template updated'), false);
      } catch (error) {
        setStatus(t('模板应用失败', 'Failed to apply template'), true);
      }
    }

    async function switchMode(mode, extra) {
      if (!modeEndpoint) return;
      setStatus(t('模式切换中…', 'Switching mode…'), false);
      try {
        await postMode(Object.assign({ mode: mode }, extra || {}));
        await fetchWorkbench();
        setStatus(
          mode === 'live' ? t('已切换为 Live', 'Switched to Live') : t('已切换为 Paper', 'Switched to Paper'),
          false
        );
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
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.message || data.error || 'order_failed');
      }
      return data;
    }

    modeButtons.forEach((btn) => {
      btn.addEventListener('click', function () {
        const mode = btn.dataset.mode === 'classic' ? 'classic' : 'shortterm';
        setPanelMode(mode, true);
      });
    });

    if (refreshBtn) {
      refreshBtn.addEventListener('click', fetchWorkbench);
    }

    templateButtons.forEach((btn) => {
      btn.addEventListener('click', function () {
        const key = btn.dataset.template;
        if (!key) return;
        applyTemplate(key);
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
        switchMode('paper');
      });
    });

    if (liveCancel) {
      liveCancel.addEventListener('click', function () {
        if (liveDialog && typeof liveDialog.close === 'function') {
          liveDialog.close();
        }
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
        if (liveDialog && typeof liveDialog.close === 'function') {
          liveDialog.close();
        }
        switchMode('live', { confirm_live: true, confirm_phrase: 'LIVE' });
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
          fetchWorkbench();
          manualOrderForm.reset();
        } catch (error) {
          setStatus(t('下单失败，请检查 Alpaca 连接', 'Order failed, check Alpaca connection'), true);
        }
      });
    }

    setPanelMode(modeFromUrlOrStorage(), false);
  });
})();
