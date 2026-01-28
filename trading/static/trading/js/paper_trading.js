(() => {
  const panel = document.querySelector('[data-tab-panel="paper"]') || document.querySelector('.paper-page');
  if (!panel) return;

  const lang = (window.langPrefix || document.documentElement.lang || 'zh').toLowerCase();
  const isZh = lang.startsWith('zh');
  const TEXT = {
    envPaper: isZh ? '模拟' : 'PAPER',
    envLive: isZh ? '实盘' : 'LIVE',
    connecting: isZh ? '连接中…' : 'Connecting…',
    connected: isZh ? '已连接' : 'Connected',
    refreshing: isZh ? '刷新中…' : 'Refreshing…',
    error: isZh ? '连接失败' : 'Error',
    updatedAt: isZh ? '更新时间' : 'Updated',
    freshness: (s) => (isZh ? `距今 ${s}s` : `${s}s ago`),
    noAccount: isZh ? '暂无 Alpaca 账户数据，请检查 API Key。' : 'No Alpaca data. Check API keys.',
    noPositions: isZh ? '暂无持仓。' : 'No positions.',
    emptyCta: isZh ? '去市场页添加标的' : 'Go to Market',
    previewTitle: isZh ? '调仓预览' : 'Rebalance preview',
    confirmTitle: isZh ? '二次确认' : 'Confirm action',
    confirmLive: isZh ? '输入 LIVE 以确认' : 'Type LIVE to confirm',
    confirmSwitchLive: isZh ? '切换到实盘将使用真实资金，下单不可撤销。继续？' : 'Switching to LIVE uses real capital. Continue?',
    confirmLiquidate: isZh ? '确认清仓当前账户全部持仓？' : 'Liquidate all positions?',
    confirmExecute: isZh ? '确认执行本次调仓？' : 'Execute this rebalance?',
    previewEmpty: isZh ? '没有需要执行的订单。' : 'No orders to execute.',
    previewWarnings: isZh ? '校验提示' : 'Validation',
    previewOrders: isZh ? '预览订单' : 'Planned orders',
    previewSummary: isZh ? '目标权重合计' : 'Target sum',
    previewCash: isZh ? '现金剩余' : 'Cash remainder',
    executeSuccess: isZh ? '调仓指令已提交' : 'Rebalance submitted',
    executeFailed: isZh ? '调仓失败' : 'Rebalance failed',
    liveRequired: isZh ? '请在输入框中输入 LIVE 以继续。' : 'Type LIVE to continue.',
    retry: isZh ? '重试' : 'Retry',
    statusOk: isZh ? '正常' : 'OK',
    statusBlocked: isZh ? '受限' : 'Blocked',
    statusUnknown: isZh ? '未知' : 'Unknown',
    sumLabel: isZh ? '目标合计' : 'Target sum',
    cashLabel: isZh ? '现金剩余' : 'Cash',
  };

  const alpacaModeButtons = Array.prototype.slice.call(panel.querySelectorAll('[data-role="alpaca-mode-btn"]'));
  const alpacaSummaryEl = panel.querySelector('[data-role="alpaca-summary"]');
  const alpacaPositionsEl = panel.querySelector('[data-role="alpaca-positions"]');
  const alpacaRefreshBtn = panel.querySelector('[data-role="alpaca-refresh"]');
  const alpacaUpdatedEl = panel.querySelector('[data-role="alpaca-updated"]');
  const alpacaErrorEl = panel.querySelector('[data-role="alpaca-error"]');
  const alpacaStatusPill = panel.querySelector('[data-role="alpaca-status-pill"]');
  const alpacaPreviewBtn = panel.querySelector('[data-role="alpaca-preview"]');
  const alpacaResetBtn = panel.querySelector('[data-role="alpaca-reset"]');
  const alpacaLiquidateBtn = panel.querySelector('[data-role="alpaca-liquidate"]');
  const alpacaLiquidateUnlisted = panel.querySelector('[data-role="alpaca-liquidate-unlisted"]');
  const alpacaNormalizeBtn = panel.querySelector('[data-role="alpaca-normalize"]');
  const alpacaRebalanceSummary = panel.querySelector('[data-role="alpaca-rebalance-summary"]');
  const alpacaRebalanceStatus = panel.querySelector('[data-role="alpaca-rebalance-status"]');

  const envModePill = panel.querySelector('[data-role="env-mode-pill"]');
  const envConnection = panel.querySelector('[data-role="env-connection"]');
  const envUpdated = panel.querySelector('[data-role="env-updated"]');
  const envFreshness = panel.querySelector('[data-role="env-freshness"]');
  const envAutoToggle = panel.querySelector('[data-role="alpaca-auto-toggle"]');

  const modal = panel.querySelector('[data-role="paper-modal"]');
  const modalBody = panel.querySelector('[data-role="paper-modal-body"]');
  const modalTitle = panel.querySelector('[data-role="paper-modal-title"]');
  const modalInputWrap = panel.querySelector('[data-role="paper-modal-input"]');
  const modalInput = panel.querySelector('#paper-modal-confirm');
  const modalConfirmBtn = panel.querySelector('[data-role="paper-modal-confirm"]');
  const modalCancelBtn = panel.querySelector('[data-role="paper-modal-cancel"]');
  const modalClosers = Array.prototype.slice.call(panel.querySelectorAll('[data-role="paper-modal-close"]'));

  const alpacaAccountEndpoint = '/api/paper/alpaca/account/';
  const alpacaRebalanceEndpoint = '/api/paper/alpaca/rebalance/';
  const DEFAULT_AUTO_REFRESH_MS = 0;
  const AUTO_REFRESH_OPTIONS = [0, 15000, 30000, 60000];

  const alpacaState = {
    mode: 'paper',
    account: null,
    positions: [],
    lastUpdated: null,
    loading: false,
    error: '',
    autoRefreshMs: DEFAULT_AUTO_REFRESH_MS,
    autoTimer: null,
    freshnessTimer: null,
  };

  const getCsrfToken = () => {
    const match = document.cookie.match(/csrftoken=([^;]+)/);
    if (match) return match[1];
    const input = document.querySelector('input[name=csrfmiddlewaretoken]');
    return input ? input.value : '';
  };

  const formatMoney = (val) => {
    if (val === null || val === undefined || Number.isNaN(val)) return '--';
    return new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(val);
  };

  const formatPct = (val) => {
    if (val === null || val === undefined || Number.isNaN(val)) return '--';
    return `${(Number(val) * 100).toFixed(2)}%`;
  };

  const formatTs = (ts) => {
    if (!ts) return '--';
    try {
      return new Date(ts).toLocaleString();
    } catch (e) {
      return ts;
    }
  };

  const setConnectionState = (state) => {
    if (!envConnection) return;
    if (state === 'loading') {
      envConnection.textContent = TEXT.refreshing;
      return;
    }
    if (state === 'error') {
      envConnection.textContent = TEXT.error;
      return;
    }
    envConnection.textContent = TEXT.connected;
  };

  const updateEnvPill = () => {
    if (!envModePill) return;
    const isLive = alpacaState.mode === 'live';
    envModePill.textContent = isLive ? TEXT.envLive : TEXT.envPaper;
    envModePill.classList.toggle('is-live', isLive);
  };

  const updateModeButtons = () => {
    alpacaModeButtons.forEach((btn) => {
      const mode = btn.dataset.mode;
      const active = mode === alpacaState.mode;
      btn.classList.toggle('is-active', active);
      btn.classList.toggle('is-live', active && mode === 'live');
    });
    updateEnvPill();
  };

  const updateAutoToggle = () => {
    if (!envAutoToggle) return;
    envAutoToggle.querySelectorAll('button[data-interval]').forEach((btn) => {
      const val = Number(btn.dataset.interval);
      const active = val === alpacaState.autoRefreshMs;
      btn.classList.toggle('is-active', active);
    });
  };

  const setLoading = (loading) => {
    alpacaState.loading = loading;
    if (alpacaRefreshBtn) {
      alpacaRefreshBtn.disabled = loading;
      alpacaRefreshBtn.textContent = loading ? (isZh ? '刷新中…' : 'Refreshing…') : (isZh ? '刷新' : 'Refresh');
    }
    setConnectionState(loading ? 'loading' : alpacaState.error ? 'error' : 'ok');
  };

  const setError = (message) => {
    alpacaState.error = message || '';
    if (alpacaErrorEl) {
      alpacaErrorEl.textContent = alpacaState.error;
    }
    setConnectionState(alpacaState.error ? 'error' : 'ok');
  };

  const updateFreshness = () => {
    if (!envFreshness) return;
    if (!alpacaState.lastUpdated) {
      envFreshness.textContent = '';
      return;
    }
    const diff = Math.max(0, Math.floor((Date.now() - new Date(alpacaState.lastUpdated).getTime()) / 1000));
    envFreshness.textContent = TEXT.freshness(diff);
  };

  const updateUpdatedTime = () => {
    const label = alpacaState.lastUpdated ? formatTs(alpacaState.lastUpdated) : '--';
    if (envUpdated) {
      envUpdated.textContent = `${TEXT.updatedAt}: ${label}`;
    }
    if (alpacaUpdatedEl) {
      alpacaUpdatedEl.textContent = label;
    }
    updateFreshness();
  };

  const renderSummary = () => {
    if (!alpacaSummaryEl) return;
    if (alpacaState.loading) {
      alpacaSummaryEl.innerHTML = `<p class="text-muted small mb-0">${isZh ? '加载中…' : 'Loading…'}</p>`;
      return;
    }
    if (!alpacaState.account) {
      alpacaSummaryEl.innerHTML = `<p class="text-muted small mb-0">${TEXT.noAccount}</p>`;
      if (alpacaStatusPill) {
        alpacaStatusPill.textContent = TEXT.statusUnknown;
        alpacaStatusPill.classList.remove('is-error');
        alpacaStatusPill.classList.toggle('is-live', alpacaState.mode === 'live');
      }
      return;
    }
    const acc = alpacaState.account;
    const metrics = [
      { label: isZh ? '权益' : 'Equity', value: `$${formatMoney(acc.equity)}` },
      { label: isZh ? '现金' : 'Cash', value: `$${formatMoney(acc.cash)}` },
      { label: isZh ? '购买力' : 'Buying Power', value: `$${formatMoney(acc.buying_power)}` },
      { label: isZh ? '净值' : 'Net Value', value: `$${formatMoney(acc.portfolio_value)}` },
      { label: isZh ? '初始保证金' : 'Initial Margin', value: `$${formatMoney(acc.initial_margin)}` },
      { label: isZh ? '维持保证金' : 'Maintenance', value: `$${formatMoney(acc.maintenance_margin)}` },
    ];
    alpacaSummaryEl.innerHTML = `
      <div class="paper-alpaca-metrics">
        ${metrics
          .map(
            (item) => `
            <div>
              <div class="label">${item.label}</div>
              <div class="value">${item.value}</div>
            </div>`
          )
          .join('')}
      </div>
    `;
    if (alpacaStatusPill) {
      const status = acc.status || '';
      alpacaStatusPill.textContent = status || TEXT.statusUnknown;
      alpacaStatusPill.classList.toggle('is-error', status && status.toLowerCase() === 'blocked');
      alpacaStatusPill.classList.toggle('is-live', alpacaState.mode === 'live');
    }
  };

  const getEquity = () => {
    if (!alpacaState.account) return 0;
    return Number(alpacaState.account.equity || alpacaState.account.portfolio_value || 0);
  };

  const computeWeight = (marketValue, equity) => {
    if (!equity) return 0;
    return (marketValue / equity) * 100;
  };

  const renderPositions = () => {
    if (!alpacaPositionsEl) return;
    if (alpacaState.loading) {
      alpacaPositionsEl.innerHTML = `<p class="text-muted small mb-0">${isZh ? '加载持仓中…' : 'Loading positions…'}</p>`;
      return;
    }
    const positions = alpacaState.positions || [];
    if (!positions.length) {
      alpacaPositionsEl.innerHTML = `
        <div class="paper-empty-state">
          <p class="mb-0">${TEXT.noPositions}</p>
          <a class="btn btn-outline-primary btn-sm" href="/market/">${TEXT.emptyCta}</a>
        </div>
      `;
      return;
    }
    const equity = getEquity();
    const headers = [
      isZh ? '标的' : 'Symbol',
      isZh ? '数量' : 'Qty',
      isZh ? '成本' : 'Avg Cost',
      isZh ? '最新' : 'Last',
      isZh ? '市值' : 'Market Value',
      isZh ? '浮盈' : 'Unrealized P/L',
      isZh ? '当前权重' : 'Weight',
      isZh ? '目标权重' : 'Target',
      isZh ? '偏差' : 'Delta',
      isZh ? '操作' : 'Actions',
    ];
    const rows = positions
      .map((pos) => {
        const marketValue = Number(pos.market_value || 0);
        const currentWeight = computeWeight(marketValue, equity);
        const targetValue = currentWeight.toFixed(2);
        const unrealizedPct = Number(pos.unrealized_plpc || 0);
        const delta = 0;
        const deltaClass = delta >= 0 ? 'pill-positive' : 'pill-negative';
        return `
          <tr data-symbol="${pos.symbol || ''}">
            <td><strong>${pos.symbol || '--'}</strong></td>
            <td class="text-end">${pos.qty ? Number(pos.qty).toFixed(4) : '--'}</td>
            <td class="text-end">$${formatMoney(pos.avg_entry_price)}</td>
            <td class="text-end">$${formatMoney(pos.current_price)}</td>
            <td class="text-end">$${formatMoney(marketValue)}</td>
            <td class="text-end ${unrealizedPct >= 0 ? 'pill-positive' : 'pill-negative'}">${formatPct(unrealizedPct)}</td>
            <td class="text-end" data-role="alpaca-current" data-current="${currentWeight.toFixed(4)}">${currentWeight.toFixed(2)}%</td>
            <td class="text-end">
              <input class="paper-alpaca-target" data-role="alpaca-target" data-symbol="${pos.symbol || ''}" data-current="${currentWeight.toFixed(4)}" value="${targetValue}" />
            </td>
            <td class="text-end" data-role="alpaca-delta" data-symbol="${pos.symbol || ''}">
              <span class="${deltaClass}">0.00%</span>
            </td>
            <td class="text-end">
              <div class="paper-alpaca-actions">
                <button type="button" class="paper-alpaca-action-btn" data-role="alpaca-clear" data-symbol="${pos.symbol || ''}">${isZh ? '清零' : 'Clear'}</button>
                <button type="button" class="paper-alpaca-action-btn" data-role="alpaca-use" data-symbol="${pos.symbol || ''}">${isZh ? '用现值' : 'Use'}</button>
              </div>
            </td>
          </tr>
        `;
      })
      .join('');
    alpacaPositionsEl.innerHTML = `
      <table class="paper-alpaca-table">
        <thead><tr>${headers.map((h) => `<th>${h}</th>`).join('')}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
    panel.querySelectorAll('[data-role="alpaca-target"]').forEach((input) => {
      updateRowDelta(input.dataset.symbol || '');
    });
    updateRebalanceSummary();
  };

  const updateRebalanceSummary = () => {
    if (!alpacaRebalanceSummary) return;
    const { totalWeight, cashRemainder } = computeTargetSummary();
    alpacaRebalanceSummary.textContent = `${TEXT.sumLabel}: ${totalWeight.toFixed(2)}% · ${TEXT.cashLabel}: $${formatMoney(cashRemainder)}`;
  };

  const computeTargetSummary = () => {
    const equity = getEquity();
    let totalWeight = 0;
    panel.querySelectorAll('[data-role="alpaca-target"]').forEach((input) => {
      const val = Number(input.value);
      if (Number.isNaN(val)) return;
      totalWeight += val;
    });
    const remainderPct = Math.max(0, 100 - totalWeight);
    const cashRemainder = equity * (remainderPct / 100);
    return { totalWeight, cashRemainder };
  };

  const updateRowDelta = (symbol) => {
    const row = panel.querySelector(`tr[data-symbol="${symbol}"]`);
    if (!row) return;
    const currentCell = row.querySelector('[data-role="alpaca-current"]');
    const targetInput = row.querySelector('[data-role="alpaca-target"]');
    const deltaCell = row.querySelector('[data-role="alpaca-delta"]');
    if (!currentCell || !targetInput || !deltaCell) return;
    const current = Number(currentCell.dataset.current || 0);
    const target = Number(targetInput.value || 0);
    const delta = target - current;
    const span = deltaCell.querySelector('span') || deltaCell;
    span.textContent = `${delta.toFixed(2)}%`;
    span.classList.toggle('pill-positive', delta >= 0);
    span.classList.toggle('pill-negative', delta < 0);
  };

  const collectTargets = () => {
    const targets = [];
    panel.querySelectorAll('[data-role="alpaca-target"]').forEach((input) => {
      const symbol = input.dataset.symbol;
      const val = Number(input.value);
      if (!symbol || Number.isNaN(val)) return;
      const weight = Math.max(0, val);
      targets.push({ symbol, target_weight: weight });
    });
    return targets;
  };

  const normalizeTargets = () => {
    const inputs = Array.from(panel.querySelectorAll('[data-role="alpaca-target"]'));
    let sum = 0;
    inputs.forEach((input) => {
      const val = Number(input.value);
      if (!Number.isNaN(val)) sum += val;
    });
    if (!sum) return;
    inputs.forEach((input) => {
      const val = Number(input.value);
      if (Number.isNaN(val)) return;
      const normalized = (val / sum) * 100;
      input.value = normalized.toFixed(2);
      updateRowDelta(input.dataset.symbol || '');
    });
    updateRebalanceSummary();
  };

  const resetTargets = () => {
    panel.querySelectorAll('[data-role="alpaca-target"]').forEach((input) => {
      if (input.dataset.current) {
        input.value = Number(input.dataset.current).toFixed(2);
        updateRowDelta(input.dataset.symbol || '');
      }
    });
    updateRebalanceSummary();
  };

  const buildPlan = ({ liquidateAll = false } = {}) => {
    const equity = getEquity();
    const positions = alpacaState.positions || [];
    const posMap = new Map();
    positions.forEach((pos) => {
      const symbol = (pos.symbol || '').toUpperCase();
      if (symbol) posMap.set(symbol, pos);
    });
    const warnings = [];
    const orders = [];
    const minNotional = 10;
    if (!equity) {
      warnings.push(isZh ? '账户权益为 0，无法计算目标仓位。' : 'Equity is 0.');
    }
    if (liquidateAll) {
      positions.forEach((pos) => {
        const qty = Number(pos.qty || 0);
        if (!qty) return;
        orders.push({ symbol: pos.symbol, action: qty > 0 ? 'sell' : 'buy', qty: Math.abs(qty) });
      });
      return { orders, warnings };
    }
    const targets = collectTargets();
    const targetSymbols = new Set();
    targets.forEach((t) => {
      const raw = Number(t.target_weight || 0);
      const weight = raw > 1 ? raw / 100 : raw;
      const symbol = (t.symbol || '').toUpperCase();
      if (!symbol) return;
      targetSymbols.add(symbol);
      const targetNotional = equity * weight;
      const pos = posMap.get(symbol) || {};
      const currentNotional = Number(pos.market_value || 0);
      const diff = targetNotional - currentNotional;
      if (Math.abs(diff) < minNotional) return;
      if (diff > 0) {
        orders.push({ symbol, action: 'buy', notional: diff });
      } else {
        const price = Number(pos.current_price || 0);
        if (!price) {
          warnings.push(isZh ? `${symbol} 缺少价格，无法计算卖出数量。` : `${symbol} missing price for sell.`);
        } else {
          orders.push({ symbol, action: 'sell', qty: Math.abs(diff) / price });
        }
      }
    });
    if (alpacaLiquidateUnlisted && alpacaLiquidateUnlisted.checked) {
      positions.forEach((pos) => {
        const symbol = (pos.symbol || '').toUpperCase();
        if (!symbol || targetSymbols.has(symbol)) return;
        const qty = Number(pos.qty || 0);
        if (!qty) return;
        orders.push({ symbol, action: qty > 0 ? 'sell' : 'buy', qty: Math.abs(qty), reason: 'unlisted' });
      });
    }
    return { orders, warnings };
  };

  const openModal = ({ title, bodyHtml, requireInput = false, onConfirm, confirmText, cancelText, showCancel = true }) => {
    if (!modal || !modalTitle || !modalBody || !modalConfirmBtn || !modalCancelBtn) return;
    modalTitle.textContent = title || '';
    modalBody.innerHTML = bodyHtml || '';
    if (modalInputWrap) {
      modalInputWrap.hidden = !requireInput;
    }
    if (modalInput) {
      modalInput.value = '';
    }
    if (modalConfirmBtn) {
      modalConfirmBtn.textContent = confirmText || (isZh ? '确认' : 'Confirm');
    }
    if (modalCancelBtn) {
      modalCancelBtn.textContent = cancelText || (isZh ? '取消' : 'Cancel');
      modalCancelBtn.style.display = showCancel ? '' : 'none';
    }
    modal.hidden = false;
    const confirmHandler = () => {
      if (requireInput && modalInput && modalInput.value.trim().toUpperCase() !== 'LIVE') {
        modalBody.insertAdjacentHTML('afterbegin', `<div class="alert alert-warning">${TEXT.liveRequired}</div>`);
        return;
      }
      if (onConfirm) onConfirm();
      closeModal();
    };
    modalConfirmBtn.onclick = confirmHandler;
    modalCancelBtn.onclick = closeModal;
  };

  const closeModal = () => {
    if (!modal) return;
    modal.hidden = true;
    if (modalBody) modalBody.innerHTML = '';
  };

  modalClosers.forEach((btn) => btn.addEventListener('click', closeModal));

  const renderPreview = () => {
    const plan = buildPlan({ liquidateAll: false });
    const { totalWeight, cashRemainder } = computeTargetSummary();
    const warnings = plan.warnings || [];
    if (totalWeight > 100.01) {
      warnings.push(isZh ? '目标权重合计超过 100%。' : 'Target weights exceed 100%.');
    }
    if (totalWeight < 99.0) {
      warnings.push(isZh ? '目标权重未达到 100%，剩余将保留现金。' : 'Target weights below 100%, remainder kept as cash.');
    }
    const orders = plan.orders || [];
    const warningsHtml = warnings.length
      ? `<div><strong>${TEXT.previewWarnings}</strong><ul>${warnings.map((w) => `<li>${w}</li>`).join('')}</ul></div>`
      : '';
    const ordersHtml = orders.length
      ? `
        <table class="paper-preview-table">
          <thead><tr><th>${isZh ? '标的' : 'Symbol'}</th><th>${isZh ? '动作' : 'Side'}</th><th>${isZh ? '数量/金额' : 'Qty/Notional'}</th></tr></thead>
          <tbody>
            ${orders
              .map((o) => {
                const qty = o.qty ? Number(o.qty).toFixed(4) : '--';
                const notional = o.notional ? `$${formatMoney(o.notional)}` : '--';
                return `<tr><td>${o.symbol}</td><td>${o.action}</td><td>${o.qty ? qty : notional}</td></tr>`;
              })
              .join('')}
          </tbody>
        </table>`
      : `<p>${TEXT.previewEmpty}</p>`;

    const summaryHtml = `
      <div><strong>${TEXT.previewSummary}:</strong> ${totalWeight.toFixed(2)}%</div>
      <div><strong>${TEXT.previewCash}:</strong> $${formatMoney(cashRemainder)}</div>
    `;

    openModal({
      title: TEXT.previewTitle,
      bodyHtml: `${warningsHtml}${summaryHtml}${ordersHtml}`,
      requireInput: alpacaState.mode === 'live',
      onConfirm: () => {
        if (!orders.length) return;
        executeRebalance({ liquidateAll: false });
      },
    });
  };

  const executeRebalance = async ({ liquidateAll = false } = {}) => {
    const payload = {
      mode: alpacaState.mode,
      targets: liquidateAll ? [] : collectTargets(),
      liquidate_unlisted: alpacaLiquidateUnlisted ? alpacaLiquidateUnlisted.checked : false,
      liquidate_all: liquidateAll,
    };
    try {
      setLoading(true);
      const resp = await fetch(alpacaRebalanceEndpoint, {
        method: 'POST',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        body: JSON.stringify(payload),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || !data.ok) {
        setError(data.error || TEXT.executeFailed);
        setStatus(TEXT.executeFailed);
        return;
      }
      setStatus(TEXT.executeSuccess);
      if (data.orders && Array.isArray(data.orders) && data.orders.length) {
        const rows = data.orders
          .map(
            (order) =>
              `<li>${order.symbol || '--'} · ${order.side || '--'} · ${order.qty || order.notional || '--'} · ${
                order.status || '--'
              }</li>`
          )
          .join('');
        openModal({
          title: isZh ? '执行结果' : 'Execution result',
          bodyHtml: `<div class=\"paper-preview-log\"><ul>${rows}</ul></div>`,
          confirmText: isZh ? '关闭' : 'Close',
          showCancel: false,
        });
      }
      await loadAlpacaAccount();
    } catch (err) {
      setStatus(isZh ? '请求失败，请稍后重试。' : 'Request failed.');
      setError(TEXT.executeFailed);
    } finally {
      setLoading(false);
    }
  };

  const setStatus = (message) => {
    if (!alpacaRebalanceStatus) return;
    alpacaRebalanceStatus.textContent = message || '';
  };

  const loadAlpacaAccount = async () => {
    try {
      setLoading(true);
      setError('');
      const resp = await fetch(`${alpacaAccountEndpoint}?mode=${alpacaState.mode}`, { credentials: 'same-origin' });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || !data.ok) {
        alpacaState.account = null;
        alpacaState.positions = [];
        alpacaState.lastUpdated = null;
        setError(isZh ? '无法读取 Alpaca 账户，请检查 API Key。' : 'Failed to load Alpaca account.');
        setLoading(false);
        renderSummary();
        renderPositions();
        return;
      }
      alpacaState.account = data.account || null;
      alpacaState.positions = Array.isArray(data.positions) ? data.positions : [];
      alpacaState.lastUpdated = data.updated_at || null;
      setLoading(false);
      renderSummary();
      renderPositions();
      updateUpdatedTime();
      setError('');
    } catch (err) {
      setError(isZh ? '请求失败，请稍后重试。' : 'Request failed.');
      setLoading(false);
    } finally {
      setConnectionState(alpacaState.error ? 'error' : 'ok');
    }
  };

  const startAutoRefresh = () => {
    if (alpacaState.autoTimer) {
      clearInterval(alpacaState.autoTimer);
      alpacaState.autoTimer = null;
    }
    if (!alpacaState.autoRefreshMs) return;
    alpacaState.autoTimer = setInterval(() => {
      if (!alpacaState.loading) loadAlpacaAccount();
    }, alpacaState.autoRefreshMs);
  };

  if (envAutoToggle) {
    envAutoToggle.addEventListener('click', (event) => {
      const btn = event.target.closest('button[data-interval]');
      if (!btn) return;
      const next = Number(btn.dataset.interval);
      if (!AUTO_REFRESH_OPTIONS.includes(next)) return;
      alpacaState.autoRefreshMs = next;
      updateAutoToggle();
      startAutoRefresh();
    });
  }

  if (alpacaModeButtons.length) {
    alpacaModeButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const nextMode = btn.dataset.mode || 'paper';
        if (nextMode === alpacaState.mode) return;
        if (nextMode === 'live') {
          openModal({
            title: TEXT.confirmTitle,
            bodyHtml: `<p>${TEXT.confirmSwitchLive}</p>`,
            requireInput: true,
            onConfirm: () => {
              alpacaState.mode = 'live';
              updateModeButtons();
              loadAlpacaAccount();
            },
          });
          return;
        }
        alpacaState.mode = nextMode;
        updateModeButtons();
        loadAlpacaAccount();
      });
    });
  }

  if (alpacaRefreshBtn) {
    alpacaRefreshBtn.addEventListener('click', () => {
      loadAlpacaAccount();
    });
  }

  if (alpacaResetBtn) {
    alpacaResetBtn.addEventListener('click', () => {
      resetTargets();
    });
  }

  if (alpacaNormalizeBtn) {
    alpacaNormalizeBtn.addEventListener('click', () => {
      normalizeTargets();
    });
  }

  if (alpacaLiquidateBtn) {
    alpacaLiquidateBtn.addEventListener('click', () => {
      openModal({
        title: TEXT.confirmTitle,
        bodyHtml: `<p>${TEXT.confirmLiquidate}</p>`,
        requireInput: alpacaState.mode === 'live',
        onConfirm: () => executeRebalance({ liquidateAll: true }),
      });
    });
  }

  if (alpacaPreviewBtn) {
    alpacaPreviewBtn.addEventListener('click', () => {
      renderPreview();
    });
  }

  panel.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.matches('[data-role="alpaca-target"]')) return;
    updateRowDelta(target.dataset.symbol || '');
    updateRebalanceSummary();
  });

  panel.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.matches('[data-role="alpaca-clear"]')) {
      const symbol = target.dataset.symbol || '';
      const input = panel.querySelector(`[data-role="alpaca-target"][data-symbol="${symbol}"]`);
      if (input) {
        input.value = '0.00';
        updateRowDelta(symbol);
        updateRebalanceSummary();
      }
    }
    if (target.matches('[data-role="alpaca-use"]')) {
      const symbol = target.dataset.symbol || '';
      const input = panel.querySelector(`[data-role="alpaca-target"][data-symbol="${symbol}"]`);
      if (input && input.dataset.current) {
        input.value = Number(input.dataset.current).toFixed(2);
        updateRowDelta(symbol);
        updateRebalanceSummary();
      }
    }
  });

  alpacaState.freshnessTimer = setInterval(updateFreshness, 1000);
  updateModeButtons();
  updateAutoToggle();
  setConnectionState('loading');
  loadAlpacaAccount();
})();
