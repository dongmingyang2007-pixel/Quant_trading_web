(() => {
  const panel = document.querySelector('[data-tab-panel="paper"]') || document.querySelector('.paper-page');
  if (!panel) return;

  const listEl = panel.querySelector('[data-role="paper-session-list"]');
  const detailEl = panel.querySelector('[data-role="paper-detail"]');
  const form = panel.querySelector('[data-role="paper-form"]');
  const alertEl = panel.querySelector('[data-role="paper-form-alert"]');

  const getCsrfToken = () => {
    const match = document.cookie.match(/csrftoken=([^;]+)/);
    if (match) return match[1];
    const input = document.querySelector("input[name=csrfmiddlewaretoken]");
    return input ? input.value : "";
  };

  const formatMoney = (val) => {
    if (val === null || val === undefined || Number.isNaN(val)) return "--";
    return new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(val);
  };

  const formatTs = (ts) => {
    if (!ts) return "--";
    try {
      return new Date(ts).toLocaleString();
    } catch (e) {
      return ts;
    }
  };

  const renderDetail = (session) => {
    if (!detailEl) return;
    if (!session) {
      const lang = (window.langPrefix || document.documentElement.lang || "zh").toLowerCase();
      detailEl.innerHTML = `<p class="text-muted mb-0">${lang.startsWith("zh") ? "暂无会话，先创建一个吧。" : "No session yet. Create one to get started."}</p>`;
      return;
    }
    const trades = session.recent_trades || [];
    const curve = session.equity_curve || [];
    const latestEquity = session.last_equity || 0;
    const pnlPct = session.pnl_pct !== null && session.pnl_pct !== undefined ? `${(session.pnl_pct * 100).toFixed(2)}%` : "--";
    const lang = (window.langPrefix || document.documentElement.lang || "zh").toLowerCase();
    const cfg = session.config || {};
    const slippageBps = cfg.slippage_bps ?? 5;
    const commissionBps = cfg.transaction_cost_bps ?? 8;
    const confidence = session.signal_confidence;
    const riskGuard = session.risk_guard;
    const lastRunTs = session.last_run_at ? new Date(session.last_run_at).getTime() : null;
    const intervalMs = (session.interval_seconds || 300) * 1000;
    const nowTs = Date.now();
    let healthState = "ok";
    let healthLabel = lang.startsWith("zh") ? "调度正常" : "On schedule";
    if (!lastRunTs) {
      healthState = "warn";
      healthLabel = lang.startsWith("zh") ? "尚未运行" : "Not run yet";
    } else if (nowTs - lastRunTs > intervalMs * 3) {
      healthState = "error";
      healthLabel = lang.startsWith("zh") ? "严重延迟" : "Severely delayed";
    } else if (nowTs - lastRunTs > intervalMs * 1.5) {
      healthState = "warn";
      healthLabel = lang.startsWith("zh") ? "轻微延迟" : "Running behind";
    }
    const labels = lang.startsWith("zh")
      ? { status: "状态", equity: "权益", cash: "现金", pnl: "收益", positions: "持仓", trades: "成交明细", curve: "权益曲线（最近）", none: "暂无数据", noTrades: "暂无成交。", noPos: "暂无持仓", heartbeat: "调仓间隔", lastRun: "最近运行" }
      : { status: "Status", equity: "Equity", cash: "Cash", pnl: "Return", positions: "Positions", trades: "Trades", curve: "Equity (latest)", none: "No data", noTrades: "No trades yet.", noPos: "No positions", heartbeat: "Rebalance interval", lastRun: "Last run" };
    const signalSource = session.signal_source || "unknown";
    const signalLabels = lang.startsWith("zh")
      ? { fresh: "新信号", fallback_cached: "回退信号", light_cached: "快速缓存", failure: "失败", unknown: "未知" }
      : { fresh: "Fresh", fallback_cached: "Fallback", light_cached: "Light cached", failure: "Failure", unknown: "Unknown" };
    const lastSkip = session.last_skip || null;
    const skipLabels = lang.startsWith("zh")
      ? { illiquid: "因流动性跳过", quote_unavailable: "行情缺失跳过", other: "已跳过" }
      : { illiquid: "Skipped (illiquid)", quote_unavailable: "Skipped (no quote)", other: "Skipped" };

    const positions = session.positions || {};
    const posList = Object.keys(positions).length
      ? Object.entries(positions).map(([sym, qty]) => `<div class="paper-detail-box"><div class="fw-semibold">${sym}</div><div class="text-muted small">${lang.startsWith("zh") ? "数量" : "Qty"}: ${Number(qty).toFixed(3)}</div></div>`).join("")
      : `<p class="text-muted mb-0">${labels.noPos}</p>`;

    const tradesList = trades.length
      ? trades.map(tr => `<li class="list-group-item d-flex justify-content-between align-items-start">
            <div>
              <div class="fw-semibold">${tr.side === "buy" ? (lang.startsWith("zh") ? "买入" : "Buy") : (lang.startsWith("zh") ? "卖出" : "Sell")} ${tr.symbol}</div>
              <div class="text-muted small">${formatTs(tr.executed_at)}</div>
            </div>
            <div class="text-end">
              <div>${Number(tr.quantity).toFixed(4)} @ $${formatMoney(tr.price)}</div>
              <div class="text-muted small">${lang.startsWith("zh") ? "金额" : "Notional"} $${formatMoney(tr.notional)}</div>
            </div>
          </li>`).join("")
      : `<li class="list-group-item text-muted">${labels.noTrades}</li>`;

    detailEl.innerHTML = `
      <div class="paper-detail-grid">
        <div>
          <div class="d-flex align-items-center gap-2 flex-wrap">
            <p class="mb-1"><strong>${session.name || session.ticker}</strong></p>
            <span class="health-pill state-${healthState}">${healthLabel}</span>
          </div>
          <p class="text-muted mb-1">${session.status.toUpperCase()} · ${session.ticker}${session.benchmark ? ` / ${session.benchmark}` : ""}</p>
          <p class="mb-0">${labels.heartbeat}: ${session.interval_seconds}s · ${labels.lastRun}: ${formatTs(session.last_run_at)}</p>
        </div>
        <div class="paper-metrics">
          <div><span class="label">${labels.equity}</span><span class="value">$${formatMoney(latestEquity)}</span></div>
          <div><span class="label">${labels.cash}</span><span class="value">$${formatMoney(session.current_cash)}</span></div>
          <div><span class="label">${labels.pnl}</span><span class="value">${pnlPct}</span></div>
        </div>
      </div>

      <div class="paper-detail-summary">
        <div class="paper-detail-box">
          <div class="fw-semibold">${lang.startsWith("zh") ? "账户成立" : "Created"}</div>
          <div class="text-muted small">${formatTs(session.created_at)}</div>
        </div>
        <div class="paper-detail-box">
          <div class="fw-semibold">${lang.startsWith("zh") ? "下次运行" : "Next run"}</div>
          <div class="text-muted small">${formatTs(session.next_run_at)}</div>
        </div>
        <div class="paper-detail-box">
          <div class="fw-semibold">${lang.startsWith("zh") ? "最近成交" : "Last trade"}</div>
          <div class="text-muted small">${formatTs(session.last_trade_at)}</div>
        </div>
        <div class="paper-detail-box">
          <div class="fw-semibold">${lang.startsWith("zh") ? "最近跳过" : "Last skip"}</div>
          <div class="text-muted small">
            ${lastSkip ? `${skipLabels[lastSkip.reason] || skipLabels.other} · ${formatTs(lastSkip.at)}` : (lang.startsWith("zh") ? "暂无" : "None")}
          </div>
        </div>
        <div class="paper-detail-box">
          <div class="fw-semibold">${lang.startsWith("zh") ? "信号来源" : "Signal source"}</div>
          <div class="text-muted small d-flex align-items-center gap-2 flex-wrap">
            <span class="signal-pill source-${signalSource}">${signalLabels[signalSource] || signalLabels.unknown}</span>
            ${confidence !== null && confidence !== undefined ? `<span class="confidence-pill">${lang.startsWith("zh") ? "置信度" : "Confidence"} ${(confidence * 100).toFixed(1)}%</span>` : ""}
            <span>${formatTs(session.last_signal_at)}</span>
            ${riskGuard ? `<span class="badge bg-warning-subtle text-warning-emphasis">${lang.startsWith("zh") ? "降风险" : "De-risked"} ×${riskGuard.factor ?? 1}</span>` : ""}
          </div>
        </div>
        <div class="paper-detail-box">
          <div class="fw-semibold">${lang.startsWith("zh") ? "成本假设" : "Cost model"}</div>
          <div class="text-muted small">${lang.startsWith("zh") ? "滑点" : "Slippage"} ${slippageBps} bps · ${lang.startsWith("zh") ? "佣金" : "Commission"} ${commissionBps} bps</div>
        </div>
      </div>

      <div class="paper-detail-section">
        <p class="fw-semibold mb-2">${labels.positions}</p>
        <div class="paper-list-grid">
          ${posList}
        </div>
      </div>

      <div class="paper-detail-section">
        <p class="fw-semibold mb-2">${labels.curve}</p>
        ${curve && curve.length ? `
          <ul class="paper-list small">
            ${curve.slice(-12).map(pt => `<li><span>${formatTs(pt.ts)}</span><span>$${formatMoney(pt.equity)}</span></li>`).join("")}
          </ul>
        ` : `<p class="text-muted mb-0">${labels.none}</p>`}
      </div>

      <div class="paper-detail-section">
        <p class="fw-semibold mb-2">${labels.trades}</p>
        <ul class="list-group list-group-flush">
          ${tradesList}
        </ul>
      </div>
    `;
  };

  const renderSessions = (sessions) => {
    if (!listEl) return;
    const lang = (window.langPrefix || document.documentElement.lang || "zh").toLowerCase();
    if (!sessions || !sessions.length) {
      const emptyText = lang.startsWith("zh") ? "暂无会话，创建你的第一个模拟盘吧。" : "No sessions yet. Create your first paper session.";
      listEl.innerHTML = `<div class="text-muted small">${emptyText}</div>`;
      return;
    }
    const statusLabel = (st) => {
      if (lang.startsWith("zh")) {
        return { running: "运行中", paused: "已暂停", stopped: "已停止", error: "异常", draft: "草稿" }[st] || st;
      }
      return { running: "Running", paused: "Paused", stopped: "Stopped", error: "Error", draft: "Draft" }[st] || st;
    };
    const signalLabels = lang.startsWith("zh")
      ? { fresh: "新信号", fallback_cached: "回退", light_cached: "缓存", failure: "失败", unknown: "未知" }
      : { fresh: "Fresh", fallback_cached: "Fallback", light_cached: "Cached", failure: "Failure", unknown: "Unknown" };
    listEl.innerHTML = sessions.map((s) => `
      <article class="paper-card paper-card--peach mb-3" data-session-id="${s.session_id}">
        <header class="d-flex justify-content-between align-items-start">
          <div>
            <p class="paper-name mb-0 fw-semibold">${s.name || s.ticker}</p>
            <p class="text-muted small mb-0">${s.ticker}${s.benchmark ? ` / ${s.benchmark}` : ""}</p>
          </div>
          <span class="paper-badge status-${s.status}">${statusLabel(s.status)}</span>
        </header>
        <div class="paper-meta mt-2">
          <span>${lang.startsWith("zh") ? "权益" : "Equity"} <strong>$${formatMoney(s.last_equity)}</strong></span>
          <span>${lang.startsWith("zh") ? "现金" : "Cash"} <strong>$${formatMoney(s.current_cash)}</strong></span>
          <span>${lang.startsWith("zh") ? "收益" : "Return"} <strong>${s.pnl_pct !== null && s.pnl_pct !== undefined ? (s.pnl_pct * 100).toFixed(2) + "%" : "--"}</strong></span>
          <span>${lang.startsWith("zh") ? "频率" : "Interval"} <strong>${s.interval_seconds}s</strong></span>
          <span>${lang.startsWith("zh") ? "下次运行" : "Next run"} <strong>${formatTs(s.next_run_at)}</strong></span>
          <span>${lang.startsWith("zh") ? "信号" : "Signal"} <strong><span class="signal-pill source-${s.signal_source || "unknown"}">${signalLabels[s.signal_source || "unknown"] || signalLabels.unknown}</span></strong></span>
        </div>
        <div class="paper-actions mt-2">
          <button type="button" class="btn btn-outline-secondary btn-sm" data-action="detail">${lang.startsWith("zh") ? "详情" : "Details"}</button>
          <button type="button" class="btn btn-outline-primary btn-sm" data-action="${s.status === "paused" ? "resume" : "pause"}">
            ${s.status === "paused" ? (lang.startsWith("zh") ? "恢复" : "Resume") : (lang.startsWith("zh") ? "暂停" : "Pause")}
          </button>
          <button type="button" class="btn btn-outline-danger btn-sm" data-action="stop">${lang.startsWith("zh") ? "停止" : "Stop"}</button>
          <button type="button" class="btn btn-outline-dark btn-sm" data-action="delete">${lang.startsWith("zh") ? "删除" : "Delete"}</button>
        </div>
      </article>
    `).join("");
    listEl.querySelectorAll(".paper-card").forEach((card) => {
      const sessionId = card.dataset.sessionId;
      card.addEventListener("click", (evt) => {
        const actionBtn = evt.target.closest("[data-action]");
        if (!actionBtn) return loadDetail(sessionId);
        evt.stopPropagation();
        const action = actionBtn.dataset.action;
        if (action === "detail") return loadDetail(sessionId);
        if (action === "delete") return deleteSession(sessionId);
        mutateSession(sessionId, action);
      });
    });
  };

  const handleError = (msg) => {
    if (!alertEl) return;
    alertEl.textContent = msg;
    alertEl.classList.remove("d-none");
  };

  const clearError = () => {
    if (!alertEl) return;
    alertEl.classList.add("d-none");
    alertEl.textContent = "";
  };

  const loadSessions = async () => {
    try {
      const resp = await fetch("/api/v1/paper/sessions/", {
        credentials: "include",
        headers: { "X-Requested-With": "XMLHttpRequest" },
      });
      const text = await resp.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch (e) {
        data = null;
      }
      if (!resp.ok) {
        console.error("Load sessions failed", resp.status, text);
        handleError(`加载会话失败（${resp.status}），请稍后重试。`);
        return;
      }
      renderSessions((data && data.sessions) || []);
    } catch (err) {
      console.error("Load sessions failed", err);
      handleError("加载模拟盘列表失败，请稍后重试。");
    }
  };

  const loadDetail = async (sessionId) => {
    try {
      const resp = await fetch(`/api/v1/paper/sessions/${sessionId}/`, {
        credentials: "include",
        headers: { "X-Requested-With": "XMLHttpRequest" },
      });
      if (!resp.ok) throw new Error("failed");
      const data = await resp.json();
      renderDetail(data);
    } catch (err) {
      handleError("加载会话详情失败。");
    }
  };

  const deleteSession = async (sessionId) => {
    try {
      const resp = await fetch(`/api/v1/paper/sessions/${sessionId}/`, {
        method: "DELETE",
        credentials: "include",
        headers: {
          "X-Requested-With": "XMLHttpRequest",
          "X-CSRFToken": getCsrfToken(),
        },
      });
      if (!resp.ok) {
        handleError("删除失败，请稍后重试。");
        return;
      }
      await loadSessions();
      renderDetail(null);
    } catch (err) {
      handleError("删除失败，请稍后重试。");
    }
  };

  const mutateSession = async (sessionId, action) => {
    try {
      const resp = await fetch(`/api/v1/paper/sessions/${sessionId}/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCsrfToken(),
          "X-Requested-With": "XMLHttpRequest",
        },
        credentials: "include",
        body: JSON.stringify({ action }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "error");
      await loadSessions();
      renderDetail(data);
    } catch (err) {
      handleError("操作失败，请稍后重试。");
    }
  };

  const submitForm = async (evt) => {
    evt.preventDefault();
    clearError();
    const fd = new FormData(form);
    const today = new Date().toISOString().slice(0, 10);
    let start = fd.get("start_date");
    let end = fd.get("end_date") || today;
    // Clamp future dates to today
    if (end > today) end = today;
    if (start && start > today) start = today;
    // Ensure start < end (fallback to one year back)
    if (!start) {
        const d = new Date(end);
        d.setDate(d.getDate() - 365);
        start = d.toISOString().slice(0, 10);
    }
    if (start >= end) {
        const d = new Date(end);
        d.setDate(d.getDate() - 10);
        start = d.toISOString().slice(0, 10);
    }
    const payload = {
      name: fd.get("name") || "",
      initial_cash: fd.get("initial_cash"),
      interval_seconds: fd.get("interval_seconds"),
      params: {
        ticker: fd.get("ticker"),
        benchmark_ticker: fd.get("benchmark") || "",
        start_date: start,
        end_date: end,
        ml_mode: "light",
        capital: fd.get("capital"),
      },
    };
    try {
      const resp = await fetch("/api/v1/paper/sessions/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCsrfToken(),
          "X-Requested-With": "XMLHttpRequest",
        },
        credentials: "include",
        body: JSON.stringify(payload),
      });
      const text = await resp.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch (e) {
        data = { error: text || "Unknown error" };
      }
      if (!resp.ok) {
        let msg = data.error || "创建失败，请检查表单。";
        if (data.details) {
          try {
            msg += " " + JSON.stringify(data.details);
          } catch (e) {
            /* noop */
          }
        }
        handleError(msg);
        return;
      }
      form.reset();
      await loadSessions();
      renderDetail(data);
    } catch (err) {
      handleError("创建模拟盘失败，请稍后再试。");
    }
  };

  if (form) {
    form.addEventListener("submit", submitForm);
  }

  loadSessions();
})();
