(function () {
  'use strict';

  function formatTs(raw) {
    if (!raw) return '-';
    if (typeof raw === 'number') {
      return new Date(raw * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
    }
    return String(raw);
  }

  function textOrDash(value) {
    return value == null || value === '' ? '-' : String(value);
  }

  function parseNumeric(value) {
    if (value == null || value === '') return null;
    const parsed = Number.parseFloat(String(value));
    return Number.isFinite(parsed) ? parsed : null;
  }

  function normalizeScore(row) {
    if (!row || typeof row !== 'object') return null;
    const direct = parseNumeric(row.score);
    if (direct != null) {
      return direct > 1 ? direct / 100 : direct;
    }
    const confidence = parseNumeric(row.confidence);
    if (confidence != null) {
      return confidence > 1 ? confidence / 100 : confidence;
    }
    const prob = parseNumeric(row.probability);
    if (prob != null) {
      return prob > 1 ? prob / 100 : prob;
    }
    return null;
  }

  function mapSignalLabel(raw, langIsZh) {
    const value = String(raw || '').trim();
    const token = value.toLowerCase();
    if (!value) {
      return langIsZh ? '观望' : 'Watch';
    }
    if (token.includes('buy') || token.includes('long') || token.includes('bull')) {
      return langIsZh ? '买入' : 'Buy';
    }
    if (token.includes('sell') || token.includes('short') || token.includes('bear')) {
      return langIsZh ? '卖出' : 'Sell';
    }
    if (token.includes('hold') || token.includes('neutral') || token.includes('wait') || token.includes('watch')) {
      return langIsZh ? '观望' : 'Watch';
    }
    return value;
  }

  function mapSignalStrength(row, langIsZh) {
    const score = normalizeScore(row);
    if (score == null) {
      return langIsZh ? '中' : 'Medium';
    }
    if (score >= 0.72) {
      return langIsZh ? '强' : 'Strong';
    }
    if (score >= 0.55) {
      return langIsZh ? '中' : 'Medium';
    }
    return langIsZh ? '弱' : 'Weak';
  }

  function mapOrderStatus(raw, langIsZh) {
    const value = String(raw || '').trim();
    const token = value.toLowerCase();
    if (!value) return '-';
    if (token.includes('filled') && token.includes('partial')) return langIsZh ? '部分成交' : 'Partially filled';
    if (token.includes('filled')) return langIsZh ? '已成交' : 'Filled';
    if (token.includes('submitted') || token.includes('accepted') || token === 'new') return langIsZh ? '已提交' : 'Submitted';
    if (token.includes('cancel')) return langIsZh ? '已撤单' : 'Canceled';
    if (token.includes('risk_blocked') || token.includes('blocked')) return langIsZh ? '风控拦截' : 'Risk blocked';
    if (token.includes('throttled')) return langIsZh ? '触发节流' : 'Throttled';
    if (token.includes('disabled')) return langIsZh ? '执行已关闭' : 'Execution off';
    if (token.includes('rejected') || token.includes('error') || token.includes('failed')) return langIsZh ? '失败' : 'Failed';
    return value;
  }

  function mapRiskGuard(rawReason, langIsZh) {
    const reason = String(rawReason || '').trim();
    if (!reason) return langIsZh ? '正常' : 'Normal';
    const token = reason.toLowerCase();
    if (token.includes('max_daily_loss') || token.includes('daily_loss') || token.includes('daily loss')) {
      return langIsZh ? '已达日内风险上限' : 'Daily loss limit reached';
    }
    if (token.includes('kill_switch') || token.includes('kill switch')) {
      return langIsZh ? '风控暂停中' : 'Risk pause active';
    }
    if (token.includes('throttle')) {
      return langIsZh ? '交易节流中' : 'Trade throttled';
    }
    return langIsZh ? '已触发保护' : 'Protection triggered';
  }

  function renderTableRows(tbody, rows, mapper, emptyColumns) {
    if (!tbody) return;
    tbody.innerHTML = '';
    if (!Array.isArray(rows) || !rows.length) {
      const tr = document.createElement('tr');
      for (let i = 0; i < emptyColumns; i += 1) {
        const td = document.createElement('td');
        td.textContent = '-';
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
      return;
    }

    rows.forEach(function (row, index) {
      const tr = document.createElement('tr');
      mapper(row, index).forEach(function (value) {
        const td = document.createElement('td');
        td.textContent = textOrDash(value);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  function updateModeButtons(root, mode) {
    const modeButtons = Array.from(root.querySelectorAll('[data-shortterm-role="mode"]'));
    modeButtons.forEach(function (btn) {
      const isActive = btn.dataset.mode === mode;
      btn.classList.toggle('btn-primary', isActive && mode === 'paper');
      btn.classList.toggle('btn-outline-primary', !isActive && btn.dataset.mode === 'paper');
      btn.classList.toggle('btn-danger', isActive && mode === 'live');
      btn.classList.toggle('btn-outline-danger', !isActive && btn.dataset.mode === 'live');
    });
  }

  function render(root, payload, langIsZh) {
    if (!root || !payload) return;

    const data = payload.trade || payload;
    const engine = data.engine || {};
    const summary = data.summary || {};
    const trading = data.trading || {};
    const riskGuard = (trading.state && trading.state.risk_guard) || trading.risk_guard || {};
    const mode = String(trading.mode || 'paper').toLowerCase() === 'live' ? 'live' : 'paper';

    const engineOnlineEl = root.querySelector('[data-shortterm-role="engine-online"]');
    const modeValueEl = root.querySelector('[data-shortterm-role="mode-value"]');
    const countUniverseEl = root.querySelector('[data-shortterm-role="counts-universe"]');
    const countFlowEl = root.querySelector('[data-shortterm-role="counts-flow"]');
    const riskGuardEl = root.querySelector('[data-shortterm-role="risk-guard"]');

    if (engineOnlineEl) engineOnlineEl.textContent = engine.online ? (langIsZh ? '在线' : 'Online') : (langIsZh ? '离线' : 'Offline');
    if (modeValueEl) modeValueEl.textContent = mode === 'live' ? 'Live' : 'Paper';
    if (countUniverseEl) {
      const focusCount = Number(summary.focus_count || 0);
      countUniverseEl.textContent = langIsZh ? `${focusCount} 支` : `${focusCount} symbols`;
    }
    if (countFlowEl) {
      const signalCount = Number(summary.signals_count || 0);
      const orderCount = Number(summary.orders_count || 0);
      countFlowEl.textContent = langIsZh ? `${signalCount} 条 / ${orderCount} 笔` : `${signalCount} sig / ${orderCount} ord`;
    }
    if (riskGuardEl) riskGuardEl.textContent = mapRiskGuard(riskGuard.reason, langIsZh);

    updateModeButtons(root, mode);

    renderTableRows(
      root.querySelector('[data-shortterm-role="focus-rows"]'),
      data.focus || [],
      function (row, index) {
        return [index + 1, row.symbol || '-', formatTs(row.since || row.since_ts || '')];
      },
      3
    );
    renderTableRows(
      root.querySelector('[data-shortterm-role="signal-rows"]'),
      data.signals || [],
      function (row) {
        return [
          formatTs(row.timestamp),
          row.symbol || '-',
          mapSignalLabel(row.signal || row.action || '', langIsZh),
          mapSignalStrength(row, langIsZh),
        ];
      },
      4
    );
    renderTableRows(
      root.querySelector('[data-shortterm-role="order-rows"]'),
      data.orders || [],
      function (row) {
        return [
          formatTs(row.timestamp),
          row.symbol || '-',
          mapSignalLabel(row.action || row.signal || '', langIsZh),
          mapOrderStatus(row.status, langIsZh),
        ];
      },
      4
    );

    return {
      mode: mode,
      engineOnline: !!engine.online,
      streamStatus: engine.stream_status || '',
    };
  }

  window.BacktestWorkspaceTrade = {
    render: render,
  };
})();
