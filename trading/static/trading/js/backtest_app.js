(function () {
  'use strict';

  function isZh(lang) {
    return String(lang || '').toLowerCase().startsWith('zh');
  }

  function normalizeWorkspace(raw) {
    const value = String(raw || '').toLowerCase();
    if (value === 'shortterm') return 'trade';
    if (value === 'classic') return 'backtest';
    if (value === 'trade' || value === 'backtest' || value === 'review') return value;
    return 'trade';
  }

  function workspaceFromUrlOrStorage(storageKey) {
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
      const saved = window.localStorage.getItem(storageKey);
      if (saved) return normalizeWorkspace(saved);
    } catch (_error) {
      // ignore
    }
    return 'trade';
  }

  function setStatus(root, text, isError) {
    const statusEl = root ? root.querySelector('[data-shortterm-role="status"]') : null;
    if (!statusEl) return;
    statusEl.textContent = text || '';
    statusEl.style.color = isError ? '#ef4444' : '#60a5fa';
  }

  function normalizeStreamStatus(rawStatus, rawDetail, langIsZh) {
    const status = String(rawStatus || '').trim().toLowerCase();
    const detail = String(rawDetail || '').trim();
    const detailLower = detail.toLowerCase();
    if (!status && !detail) {
      return { suffix: '', isError: false };
    }

    const hasErrorHint = ['error', 'fail', 'forbidden', 'unauthorized', 'denied', 'invalid', 'disconnect', 'timeout']
      .some(function (token) {
        return status.indexOf(token) >= 0 || detailLower.indexOf(token) >= 0;
      });
    if (hasErrorHint || status === 'disconnected') {
      return {
        suffix: langIsZh ? ' · 连接异常' : ' · Stream error',
        isError: true,
      };
    }

    if (status === 'fallback') {
      return {
        suffix: langIsZh ? ' · 已切换 IEX' : ' · IEX fallback',
        isError: false,
      };
    }

    if (status === 'connecting' || status === 'authorizing' || status === 'authenticating') {
      return {
        suffix: langIsZh ? ' · 连接中' : ' · Connecting',
        isError: false,
      };
    }

    if (status === 'connected' || status === 'success' || status === 'listening' || status === 'subscription') {
      return { suffix: '', isError: false };
    }

    return {
      suffix: detail ? (langIsZh ? ' · 状态更新' : ' · Status updated') : '',
      isError: false,
    };
  }

  document.addEventListener('DOMContentLoaded', function () {
    const shorttermRoot = document.querySelector('[data-workbench-root]');
    if (!shorttermRoot) return;

    const store = window.BacktestState.createStore({ workspace: 'trade' });
    const langIsZh = isZh(shorttermRoot.dataset.lang);
    const workspaceSwitch = document.querySelector('[data-workbench-role="workspace-switch"]');
    const workspaceButtons = workspaceSwitch
      ? Array.from(workspaceSwitch.querySelectorAll('[data-workbench-role="workspace-btn"]'))
      : [];

    const tradePanel = document.querySelector('[data-workbench-role="trade-panel"]');
    const backtestPanel = document.querySelector('[data-workbench-role="backtest-panel"]');
    const reviewPanel = document.querySelector('[data-workbench-role="review-panel"]');

    const apiClient = window.BacktestApi.createClient({
      workbenchEndpoint: shorttermRoot.dataset.workbenchEndpoint,
      modeEndpoint: shorttermRoot.dataset.modeEndpoint,
      orderEndpoint: shorttermRoot.dataset.orderEndpoint,
    });

    const storageKey = 'backtest.primary.workspace';
    let pollTimer = null;

    function renderWorkspaceVisibility(workspace) {
      if (tradePanel) tradePanel.hidden = workspace !== 'trade';
      if (backtestPanel) backtestPanel.hidden = workspace !== 'backtest';
      if (reviewPanel) reviewPanel.hidden = workspace !== 'review';
      workspaceButtons.forEach(function (btn) {
        btn.classList.toggle('is-active', btn.dataset.workspace === workspace);
      });
    }

    function persistWorkspace(workspace) {
      try {
        window.localStorage.setItem(storageKey, workspace);
      } catch (_error) {
        // ignore
      }
      try {
        const url = new URL(window.location.href);
        url.searchParams.set('workspace', workspace);
        url.searchParams.delete('panel');
        window.history.replaceState({}, '', url.toString());
      } catch (_error) {
        // ignore
      }
    }

    function syncTradeStatus(payload) {
      const rendered = window.BacktestWorkspaceTrade.render(shorttermRoot, payload, langIsZh) || {};
      const engine = payload && payload.trade && payload.trade.engine ? payload.trade.engine : {};
      const streamState = normalizeStreamStatus(engine.stream_status, engine.stream_detail, langIsZh);
      const marketSource = String(
        (payload && (payload.market_data_source || payload.source))
          || (payload && payload.trade && (payload.trade.market_data_source || payload.trade.source))
          || 'unknown'
      ).toUpperCase();
      const executionSource = String(
        (payload && payload.execution_source)
          || (payload && payload.trade && payload.trade.execution_source)
          || 'alpaca'
      ).toUpperCase();
      const sourceLabel = langIsZh
        ? ('数据源：' + marketSource + ' · 执行：' + executionSource)
        : ('Source: ' + marketSource + ' · Execution: ' + executionSource);
      if (streamState.isError) {
        setStatus(shorttermRoot, sourceLabel + streamState.suffix, true);
        return;
      }
      if (rendered.engineOnline) {
        setStatus(shorttermRoot, sourceLabel + streamState.suffix, false);
      } else {
        setStatus(
          shorttermRoot,
          langIsZh ? (marketSource + ' 暂无数据/连接异常') : (marketSource + ' unavailable / connection issue'),
          true
        );
      }
    }

    function syncPanels(payload) {
      if (backtestPanel) {
        window.BacktestWorkspaceBacktest.render(backtestPanel, payload);
      }
      if (reviewPanel) {
        window.BacktestWorkspaceReview.render(reviewPanel, payload);
      }
    }

    function loadWorkbench(workspace) {
      const target = normalizeWorkspace(workspace);
      if (target === 'trade') {
        setStatus(shorttermRoot, langIsZh ? '正在刷新短线数据…' : 'Refreshing short-term data…', false);
      }
      return apiClient
        .getWorkbench(target)
        .then(function (payload) {
          store.setState({ trade: payload.trade, backtest: payload.backtest, review: payload.review });
          syncPanels(payload);
          if (target === 'trade') {
            syncTradeStatus(payload);
          }
        })
        .catch(function (error) {
          if (target === 'trade') {
            if (error && (error.errorCode === 'market_data_unavailable' || error.errorCode === 'alpaca_unavailable')) {
              setStatus(shorttermRoot, langIsZh ? '当前行情数据源不可用' : 'Market data source unavailable', true);
            } else {
              setStatus(shorttermRoot, error && error.message ? error.message : (langIsZh ? '工作台加载失败' : 'Failed to load workbench'), true);
            }
          }
          if (window.BacktestUiToast) {
            window.BacktestUiToast.error(error && error.message ? error.message : 'Workbench request failed');
          }
        });
    }

    function setWorkspace(workspace, persist) {
      const target = normalizeWorkspace(workspace);
      store.setState({ workspace: target });
      renderWorkspaceVisibility(target);
      if (persist) persistWorkspace(target);
      loadWorkbench(target);
      if (target === 'trade') {
        if (pollTimer) window.clearInterval(pollTimer);
        pollTimer = window.setInterval(function () {
          loadWorkbench('trade');
        }, 15000);
      } else if (pollTimer) {
        window.clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    workspaceButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        setWorkspace(btn.dataset.workspace || 'trade', true);
      });
    });

    const refreshBtn = shorttermRoot.querySelector('[data-shortterm-role="refresh"]');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', function () {
        loadWorkbench(store.getState().workspace || 'trade');
      });
    }

    Array.from(shorttermRoot.querySelectorAll('[data-shortterm-role="template"]')).forEach(function (btn) {
      btn.addEventListener('click', function () {
        const templateKey = btn.dataset.template;
        if (!templateKey) return;
        setStatus(shorttermRoot, langIsZh ? '模板应用中…' : 'Applying template…', false);
        apiClient
          .setTradingMode({ template_key: templateKey, mode: 'paper' })
          .then(function () {
            loadWorkbench('trade');
            if (window.BacktestUiToast) {
              window.BacktestUiToast.success(langIsZh ? '模板已更新' : 'Template updated');
            }
          })
          .catch(function (error) {
            setStatus(shorttermRoot, error.message || (langIsZh ? '模板应用失败' : 'Template update failed'), true);
          });
      });
    });

    const liveDialog = shorttermRoot.querySelector('[data-shortterm-role="live-dialog"]');
    const liveCheck = shorttermRoot.querySelector('[data-shortterm-role="live-confirm-check"]');
    const liveInput = shorttermRoot.querySelector('[data-shortterm-role="live-confirm-input"]');

    Array.from(shorttermRoot.querySelectorAll('[data-shortterm-role="mode"]')).forEach(function (btn) {
      btn.addEventListener('click', function () {
        const mode = String(btn.dataset.mode || '').toLowerCase();
        if (mode === 'live') {
          if (liveCheck) liveCheck.checked = false;
          if (liveInput) liveInput.value = '';
          if (liveDialog && typeof liveDialog.showModal === 'function') {
            liveDialog.showModal();
          }
          return;
        }
        apiClient
          .setTradingMode({ mode: 'paper' })
          .then(function () {
            loadWorkbench('trade');
          })
          .catch(function (error) {
            setStatus(shorttermRoot, error.message || (langIsZh ? '模式切换失败' : 'Failed to switch mode'), true);
          });
      });
    });

    const liveCancel = shorttermRoot.querySelector('[data-shortterm-role="live-cancel"]');
    if (liveCancel) {
      liveCancel.addEventListener('click', function () {
        if (liveDialog && typeof liveDialog.close === 'function') {
          liveDialog.close();
        }
      });
    }

    const liveSubmit = shorttermRoot.querySelector('[data-shortterm-role="live-submit"]');
    if (liveSubmit) {
      liveSubmit.addEventListener('click', function () {
        const checked = !!(liveCheck && liveCheck.checked);
        const phrase = liveInput ? String(liveInput.value || '').trim().toUpperCase() : '';
        if (!checked || phrase !== 'LIVE') {
          setStatus(shorttermRoot, langIsZh ? '请勾选确认并输入 LIVE' : 'Please confirm and type LIVE', true);
          return;
        }
        apiClient
          .setTradingMode({ mode: 'live', confirm_live: true, confirm_phrase: 'LIVE' })
          .then(function () {
            if (liveDialog && typeof liveDialog.close === 'function') {
              liveDialog.close();
            }
            loadWorkbench('trade');
          })
          .catch(function (error) {
            setStatus(shorttermRoot, error.message || (langIsZh ? '模式切换失败' : 'Failed to switch mode'), true);
          });
      });
    }

    const manualOrderForm = shorttermRoot.querySelector('[data-shortterm-role="manual-order-form"]');
    if (manualOrderForm) {
      manualOrderForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const submitter = event.submitter;
        const side = submitter && submitter.dataset.side ? submitter.dataset.side : '';
        const formData = new FormData(manualOrderForm);
        const symbol = String(formData.get('symbol') || '').trim().toUpperCase();
        const notional = Number.parseFloat(String(formData.get('notional') || '0'));
        if (!symbol || !side || !Number.isFinite(notional) || notional <= 0) {
          setStatus(shorttermRoot, langIsZh ? '请填写有效订单参数' : 'Please fill valid order parameters', true);
          return;
        }

        apiClient
          .submitManualOrder({ symbol: symbol, side: side, notional: notional })
          .then(function () {
            manualOrderForm.reset();
            loadWorkbench('trade');
            if (window.BacktestUiToast) {
              window.BacktestUiToast.success(langIsZh ? '订单已提交' : 'Order submitted');
            }
          })
          .catch(function (error) {
            setStatus(shorttermRoot, error.message || (langIsZh ? '下单失败，请检查 Alpaca 连接' : 'Order failed, check Alpaca connection'), true);
          });
      });
    }

    apiClient
      .getTradingModeInfo()
      .then(function (payload) {
        store.setState({ tradingModeInfo: payload });
      })
      .catch(function () {
        // non-blocking
      });

    setWorkspace(workspaceFromUrlOrStorage(storageKey), false);
  });
})();
