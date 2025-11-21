(() => {
  const mountEl = document.querySelector('[data-portfolio-builder]');
  const optionsNode = document.getElementById('portfolio-history-options');
  if (!mountEl || !optionsNode) return;
  const endpoint = mountEl.dataset.endpoint;
  if (!endpoint) return;

  let historyOptions = [];
  try {
    historyOptions = JSON.parse(optionsNode.textContent || '[]');
  } catch (error) {
    console.error('Failed to parse history options', error);
  }
  if (!window.Vue) {
    console.warn('Vue runtime is missing; portfolio builder will not initialize.');
    return;
  }
  const { createApp } = window.Vue;
  const docLang = (document.documentElement.lang || navigator.language || '').toLowerCase();
  const langIsZh = docLang.startsWith('zh');
  const TEXT = langIsZh
    ? {
        select: '选择策略',
        addRow: '添加策略',
        compute: '计算组合',
        loading: '组合计算中…',
        tip: '提示：挑选 2-3 条风格不同的策略，可直观看到分散效果。',
        insufficient: '至少选择一条策略并设置正权重。',
        error: '组合计算失败，请稍后重试。',
        updated: '组合已更新。',
      }
    : {
        select: 'Select strategy',
        addRow: 'Add strategy',
        compute: 'Compute portfolio',
        loading: 'Calculating portfolio…',
        tip: 'Tip: Mix 2–3 runs with distinct regimes to inspect diversification.',
        insufficient: 'Pick at least one strategy with a positive weight.',
        error: 'Unable to build portfolio right now.',
        updated: 'Portfolio updated.',
      };

  const getCsrfToken = () => {
    const input = document.querySelector('input[name=csrfmiddlewaretoken]');
    if (input && input.value) return input.value;
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? decodeURIComponent(match[1]) : '';
  };

  const palette = ['#2563eb', '#10b981', '#f97316', '#8b5cf6'];

  const app = createApp({
    data() {
      return {
        options: historyOptions,
        rows: [],
        cashWeight: 20,
        message: TEXT.tip,
        messageState: 'info',
        loading: false,
        results: null,
        chartInstance: null,
        chartObserver: null,
      };
    },
    computed: {
      selectLabel() {
        return TEXT.select;
      },
      addLabel() {
        return TEXT.addRow;
      },
      computeLabel() {
        return TEXT.compute;
      },
      loadingLabel() {
        return TEXT.loading;
      },
      canSubmit() {
        return this.rows.some((row) => row.recordId && row.weight > 0);
      },
      weightsList() {
        const weights = this.results?.weights || {};
        return Object.entries(weights).map(([label, value]) => ({
          label,
          value: (Number(value) * 100).toFixed(1),
        }));
      },
      metricRows() {
        const metrics = this.results?.metrics || {};
        const defs = [
          { key: 'total_return', label: langIsZh ? '累计收益' : 'Total return', percent: true },
          { key: 'sharpe', label: 'Sharpe', percent: false },
          { key: 'volatility', label: langIsZh ? '波动率' : 'Volatility', percent: true },
          { key: 'max_drawdown', label: langIsZh ? '最大回撤' : 'Max drawdown', percent: true },
          { key: 'cvar_95', label: 'CVaR', percent: true },
          { key: 'sortino', label: 'Sortino', percent: false },
        ];
        return defs.map((def) => {
          const raw = metrics[def.key];
          let formatted = '—';
          if (typeof raw === 'number' && Number.isFinite(raw)) {
            formatted = def.percent ? `${(raw * 100).toFixed(2)}%` : raw.toFixed(2);
          }
          return { key: def.key, label: def.label, value: formatted };
        });
      },
      correlations() {
        if (!Array.isArray(this.results?.correlation)) return [];
        return this.results.correlation.map((pair) => ({
          label: `${pair.a} ↔ ${pair.b}`,
          value: (Number(pair.value) * 100).toFixed(1),
        }));
      },
    },
    methods: {
      addRow() {
        const nextId = `row-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        this.rows.push({ id: nextId, recordId: '', weight: 40 });
      },
      removeRow(index) {
        if (this.rows.length <= 1) {
          this.rows[0].recordId = '';
          this.rows[0].weight = 40;
        } else {
          this.rows.splice(index, 1);
        }
      },
      setMessage(text, state = 'info') {
        this.message = text;
        this.messageState = state;
      },
      payloadComponents() {
        return this.rows
          .filter((row) => row.recordId && row.weight > 0)
          .map((row) => ({ record_id: row.recordId, weight: row.weight }));
      },
      resetChart() {
        if (this.chartObserver) {
          this.chartObserver.disconnect();
          this.chartObserver = null;
        }
        if (this.chartInstance) {
          this.chartInstance.remove();
          this.chartInstance = null;
        }
        if (this.$refs.chart) {
          this.$refs.chart.innerHTML = '';
        }
      },
      renderChart() {
        if (!this.results || !this.$refs.chart || typeof LightweightCharts === 'undefined') {
          return;
        }
        this.resetChart();
        const container = this.$refs.chart;
        const chart = LightweightCharts.createChart(container, {
          layout: { background: { color: 'transparent' }, textColor: '#0f172a' },
          rightPriceScale: { borderVisible: false },
          timeScale: { borderVisible: false },
          grid: {
            vertLines: { visible: false },
            horzLines: { color: 'rgba(15,23,42,0.08)' },
          },
        });
        const series = chart.addLineSeries({ color: '#111827', lineWidth: 3 });
        series.setData((this.results.curve || []).map((point) => ({ time: point.time, value: point.value })));
        (this.results.components || []).forEach((component, index) => {
          if (!Array.isArray(component.points)) return;
          const child = chart.addLineSeries({
            color: palette[index % palette.length],
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dotted,
          });
          child.setData(component.points.map((point) => ({ time: point.time, value: point.value })));
        });
        const resize = () => {
          chart.applyOptions({ width: container.clientWidth, height: 320 });
          chart.timeScale().fitContent();
        };
        resize();
        if (typeof ResizeObserver === 'function') {
          this.chartObserver = new ResizeObserver(resize);
          this.chartObserver.observe(container);
        }
        this.chartInstance = chart;
      },
      async submitPortfolio() {
        const components = this.payloadComponents();
        if (!components.length) {
          this.setMessage(TEXT.insufficient, 'error');
          return;
        }
        this.loading = true;
        this.setMessage(TEXT.loading, 'info');
        try {
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-CSRFToken': getCsrfToken(),
              'X-Requested-With': 'XMLHttpRequest',
            },
            credentials: 'same-origin',
            body: JSON.stringify({ components, cash_weight: this.cashWeight }),
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload && payload.error ? payload.error : TEXT.error);
          }
          this.results = payload;
          this.setMessage(TEXT.updated, 'info');
          this.$nextTick(() => this.renderChart());
        } catch (error) {
          console.error('Portfolio builder error', error);
          this.setMessage(error && error.message ? error.message : TEXT.error, 'error');
        } finally {
          this.loading = false;
        }
      },
    },
    mounted() {
      this.addRow();
      this.addRow();
    },
  });

  app.config.compilerOptions.delimiters = ['[[', ']]'];
  app.mount(mountEl);
})();
