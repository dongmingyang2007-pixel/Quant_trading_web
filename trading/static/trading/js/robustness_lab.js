(() => {
  const form = document.getElementById('analysis-form');
  if (!form) return;
  const endpoint = form.dataset.robustnessEndpoint || '';
  const statusTemplate = form.dataset.taskStatusTemplate || '';
  if (!endpoint || !statusTemplate) return;

  const runBtn = document.getElementById('robustness-run');
  const statusEl = document.getElementById('robustness-status');
  const heatmapsEl = document.getElementById('robustness-heatmaps');
  const recommendEl = document.getElementById('robustness-recommend');
  const metricToggle = document.getElementById('robustness-metric-toggle');
  if (!runBtn || !statusEl || !heatmapsEl || !metricToggle || !recommendEl) return;

  const langIsZh = (document.documentElement.lang || '').toLowerCase().startsWith('zh');
  const metricLabels = {
    sharpe: 'Sharpe',
    max_drawdown: langIsZh ? '最大回撤' : 'Max DD',
    avg_coverage: langIsZh ? '成交覆盖' : 'Coverage',
  };

  let currentMetric = 'sharpe';
  let currentGrid = null;

  const getCsrfToken = () => {
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? decodeURIComponent(match[1]) : '';
  };

  const serializeFormParams = () => {
    const data = new FormData(form);
    const params = {};
    data.forEach((value, key) => {
      if (key === 'csrfmiddlewaretoken') return;
      params[key] = value;
    });
    ['start_date', 'end_date'].forEach((key) => {
      if (params[key]) params[key] = String(params[key]).replace(/\//g, '-');
    });
    const todayIso = new Date().toISOString().slice(0, 10);
    if (!params.end_date) params.end_date = todayIso;
    if (!params.start_date) params.start_date = todayIso;
    return params;
  };

  const updateStatus = (text, isError = false) => {
    statusEl.textContent = text;
    statusEl.classList.toggle('text-danger', isError);
  };

  const formatValue = (metric, value) => {
    if (!Number.isFinite(value)) return '-';
    if (metric === 'avg_coverage') return `${(value * 100).toFixed(1)}%`;
    if (metric === 'max_drawdown') return `${(value * 100).toFixed(1)}%`;
    return value.toFixed(2);
  };

  const metricTransform = (metric, value) => {
    if (!Number.isFinite(value)) return null;
    if (metric === 'max_drawdown') return Math.abs(value);
    return value;
  };

  const colorFor = (value, min, max, invert = false) => {
    if (!Number.isFinite(value)) return '#f8fafc';
    const span = max - min || 1;
    let ratio = (value - min) / span;
    ratio = Math.max(0, Math.min(1, ratio));
    if (invert) ratio = 1 - ratio;
    const hue = 120 * ratio;
    return `hsl(${hue}, 65%, 85%)`;
  };

  const buildCellMap = (cells) => {
    const map = new Map();
    cells.forEach((cell) => {
      if (!cell || !cell.ok) return;
      const cost = Number(cell.cost_rate).toFixed(6);
      const adv = Number(cell.adv_participation).toFixed(4);
      const thr = Number(cell.entry_threshold).toFixed(4);
      map.set(`${cost}:${adv}:${thr}`, cell);
    });
    return map;
  };

  const renderHeatmaps = (grid) => {
    if (!grid || !grid.cells || !grid.cost_rates || !grid.adv_participation || !grid.thresholds) return;
    heatmapsEl.innerHTML = '';
    const cells = Array.isArray(grid.cells) ? grid.cells : [];
    const cellMap = buildCellMap(cells);
    const metricValues = cells
      .map((cell) => metricTransform(currentMetric, Number(cell.metrics && cell.metrics[currentMetric])))
      .filter((val) => Number.isFinite(val));
    const minVal = metricValues.length ? Math.min(...metricValues) : 0;
    const maxVal = metricValues.length ? Math.max(...metricValues) : 1;
    const invert = currentMetric === 'max_drawdown';
    const best = grid.best || null;
    const bestKey = best
      ? `${Number(best.cost_rate).toFixed(6)}:${Number(best.adv_participation).toFixed(4)}:${Number(best.entry_threshold).toFixed(4)}`
      : null;

    grid.cost_rates.forEach((costRate) => {
      const group = document.createElement('div');
      group.className = 'robustness-heatmap-group';
      const bps = (Number(costRate) * 10000).toFixed(1);
      group.innerHTML = `<div class="robustness-heatmap-title">${langIsZh ? '成本' : 'Cost'} ${bps} bps</div>`;
      const table = document.createElement('table');
      table.className = 'robustness-heatmap-table';
      const thead = document.createElement('thead');
      const headRow = document.createElement('tr');
      headRow.innerHTML = `<th>${langIsZh ? 'ADV 参与率' : 'ADV'}</th>` + grid.thresholds.map((thr) => `<th>${Number(thr).toFixed(2)}</th>`).join('');
      thead.appendChild(headRow);
      table.appendChild(thead);
      const tbody = document.createElement('tbody');

      grid.adv_participation.forEach((adv) => {
        const row = document.createElement('tr');
        const advLabel = `${(Number(adv) * 100).toFixed(0)}%`;
        row.innerHTML = `<th>${advLabel}</th>`;
        grid.thresholds.forEach((thr) => {
          const key = `${Number(costRate).toFixed(6)}:${Number(adv).toFixed(4)}:${Number(thr).toFixed(4)}`;
          const cell = cellMap.get(key);
          const rawValue = cell ? Number(cell.metrics && cell.metrics[currentMetric]) : NaN;
          const transformed = metricTransform(currentMetric, rawValue);
          const color = colorFor(transformed, minVal, maxVal, invert);
          const td = document.createElement('td');
          td.className = 'robustness-heatmap-cell';
          td.style.background = cell ? color : '#f8fafc';
          td.style.border = bestKey && bestKey === key ? '2px solid #2563eb' : '1px solid #e2e8f0';
          td.innerHTML = cell ? `${formatValue(currentMetric, rawValue)}<small>${formatValue('avg_coverage', Number(cell.metrics && cell.metrics.avg_coverage))}</small>` : '-';
          row.appendChild(td);
        });
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      group.appendChild(table);
      heatmapsEl.appendChild(group);
    });
  };

  const renderRecommendations = (grid) => {
    if (!grid || !grid.recommendations) return;
    const rec = grid.recommendations;
    const cost = Array.isArray(rec.cost_rate_range) && rec.cost_rate_range.length === 2
      ? `${(rec.cost_rate_range[0] * 10000).toFixed(1)} - ${(rec.cost_rate_range[1] * 10000).toFixed(1)} bps`
      : null;
    const adv = Array.isArray(rec.adv_participation_range) && rec.adv_participation_range.length === 2
      ? `${(rec.adv_participation_range[0] * 100).toFixed(0)}% - ${(rec.adv_participation_range[1] * 100).toFixed(0)}%`
      : null;
    const thr = Array.isArray(rec.threshold_range) && rec.threshold_range.length === 2
      ? `${rec.threshold_range[0].toFixed(2)} - ${rec.threshold_range[1].toFixed(2)}`
      : null;
    if (!cost && !adv && !thr) {
      recommendEl.hidden = true;
      return;
    }
    recommendEl.hidden = false;
    recommendEl.textContent = langIsZh
      ? `推荐区间：成本 ${cost || '-'}，ADV ${adv || '-'}，阈值 ${thr || '-'}`
      : `Suggested ranges: cost ${cost || '-'}, ADV ${adv || '-'}, threshold ${thr || '-'}`;
  };

  const render = (grid) => {
    currentGrid = grid;
    renderHeatmaps(grid);
    renderRecommendations(grid);
  };

  metricToggle.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-metric]');
    if (!button) return;
    const metric = button.dataset.metric;
    if (!metric || metric === currentMetric) return;
    currentMetric = metric;
    metricToggle.querySelectorAll('button').forEach((node) => {
      node.classList.toggle('active', node.dataset.metric === metric);
      node.textContent = metricLabels[node.dataset.metric] || node.textContent;
    });
    if (currentGrid) renderHeatmaps(currentGrid);
  });

  const pollStatus = (taskId) => {
    const statusUrl = statusTemplate.replace('TASK_ID_PLACEHOLDER', encodeURIComponent(taskId));
    const tick = () => {
      fetch(statusUrl, { headers: { 'X-CSRFToken': getCsrfToken() } })
        .then((response) => response.json().catch(() => ({})))
        .then((data) => {
          const state = data.state || '';
          if (state === 'SUCCESS') {
            updateStatus(langIsZh ? '稳健性评估完成。' : 'Robustness evaluation complete.');
            const grid = data.result && data.result.grid ? data.result.grid : null;
            if (grid) {
              render(grid);
            }
            return;
          }
          if (state === 'FAILURE' || state === 'REVOKED') {
            updateStatus(langIsZh ? '稳健性评估失败。' : 'Robustness evaluation failed.', true);
            return;
          }
          const progress = data.meta && data.meta.progress ? data.meta.progress : 0;
          updateStatus(
            langIsZh ? `稳健性评估进行中… ${progress}%` : `Robustness running… ${progress}%`
          );
          window.setTimeout(tick, 4000);
        })
        .catch(() => {
          updateStatus(langIsZh ? '稳健性评估状态获取失败。' : 'Failed to fetch robustness status.', true);
        });
    };
    tick();
  };

  runBtn.addEventListener('click', () => {
    const params = serializeFormParams();
    params.robustness = { max_runs: 12 };
    updateStatus(langIsZh ? '正在提交稳健性评估…' : 'Submitting robustness run…');
    fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfToken(),
      },
      body: JSON.stringify(params),
    })
      .then((response) =>
        response
          .json()
          .catch(() => ({}))
          .then((data) => ({ ok: response.ok, data }))
      )
      .then(({ ok, data }) => {
        if (!ok) {
          const msg = data && (data.error || data.detail) ? data.error || data.detail : 'HTTP error';
          updateStatus(langIsZh ? `提交失败：${msg}` : `Submit failed: ${msg}`, true);
          return;
        }
        if (data && data.result && data.result.grid) {
          updateStatus(langIsZh ? '稳健性评估完成。' : 'Robustness evaluation complete.');
          render(data.result.grid);
          return;
        }
        if (data && data.task_id) {
          updateStatus(langIsZh ? '稳健性评估已排队…' : 'Robustness queued…');
          pollStatus(data.task_id);
          return;
        }
        updateStatus(langIsZh ? '未获取任务 ID。' : 'Task id missing.', true);
      })
      .catch((error) => {
        updateStatus(
          langIsZh ? `稳健性评估提交失败：${error.message}` : `Submit failed: ${error.message}`,
          true
        );
      });
  });

  metricToggle.querySelectorAll('button').forEach((button) => {
    if (button.dataset.metric && metricLabels[button.dataset.metric]) {
      button.textContent = metricLabels[button.dataset.metric];
    }
  });
})();
