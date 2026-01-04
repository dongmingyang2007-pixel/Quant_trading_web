(() => {
  const parsePayload = (node) => {
    if (!node) return null;
    try {
      return JSON.parse(node.textContent || '{}');
    } catch (_error) {
      return null;
    }
  };

  const toLineData = (rows, key) => {
    if (!Array.isArray(rows)) return [];
    return rows
      .map((row) => {
        if (!row || typeof row !== 'object') return null;
        const value = Number(row[key]);
        if (!Number.isFinite(value)) return null;
        return { time: row.date, value };
      })
      .filter(Boolean);
  };

  const pickFirstSeries = (candidates) => candidates.find((series) => Array.isArray(series) && series.length);

  const buildMarkers = (events, langIsZh) => {
    if (!Array.isArray(events)) return [];
    const fallbackLabels = {
      entry: langIsZh ? '进场' : 'Entry',
      exit: langIsZh ? '出场' : 'Exit',
      halt: langIsZh ? '停牌' : 'Halt',
      limit: langIsZh ? '涨跌停' : 'Limit',
      adv: langIsZh ? 'ADV 限制' : 'ADV cap',
    };
    const styles = {
      entry: { position: 'belowBar', color: '#16a34a', shape: 'arrowUp' },
      exit: { position: 'aboveBar', color: '#dc2626', shape: 'arrowDown' },
      halt: { position: 'belowBar', color: '#f59e0b', shape: 'circle' },
      limit: { position: 'aboveBar', color: '#f97316', shape: 'circle' },
      adv: { position: 'aboveBar', color: '#7c3aed', shape: 'circle' },
    };
    return events
      .map((event) => {
        if (!event || !event.date) return null;
        const kind = event.type || 'event';
        const style = styles[kind] || styles.adv;
        const label = event.label || fallbackLabels[kind] || kind;
        return {
          time: event.date,
          text: label,
          position: style.position,
          color: style.color,
          shape: style.shape,
        };
      })
      .filter(Boolean);
  };

  const renderChart = () => {
    const dataNode = document.getElementById('overview-series-data');
    const payload = parsePayload(dataNode);
    const returnNode = document.getElementById('overview-return-series-data');
    const returnPayload = parsePayload(returnNode);

    const root = document.getElementById('overview-signal-chart');
    if (!root) return;
    if (typeof LightweightCharts === 'undefined') {
      console.warn('LightweightCharts is unavailable.');
      return;
    }

    const seriesRows = payload && Array.isArray(payload.series) ? payload.series : [];
    const returnRows = Array.isArray(returnPayload) ? returnPayload : [];
    const probability = toLineData(seriesRows, 'probability');
    const position = toLineData(seriesRows, 'position');
    const fallback = toLineData(returnRows, 'cum_strategy');
    if (!probability.length && !position.length && !fallback.length) return;

    const chart = LightweightCharts.createChart(root, {
      layout: { background: { type: 'solid', color: '#f8fafc' }, textColor: '#0f172a' },
      rightPriceScale: { borderColor: '#e2e8f0', visible: true },
      leftPriceScale: { borderColor: '#e2e8f0', visible: true },
      timeScale: { borderColor: '#e2e8f0', timeVisible: true, secondsVisible: false },
      grid: {
        horzLines: { color: '#e2e8f0' },
        vertLines: { color: '#e2e8f0' },
      },
      localization: {
        locale: document.documentElement.lang || navigator.language,
        dateFormat: 'yyyy-MM-dd',
      },
    });

    const resizeChart = () => {
      const rect = root.getBoundingClientRect();
      if (rect.width && rect.height) {
        chart.resize(rect.width, rect.height);
        chart.timeScale().fitContent();
      }
    };
    resizeChart();
    window.addEventListener('resize', () => resizeChart());

    const docLang = (document.documentElement.getAttribute('lang') || navigator.language || '').toLowerCase();
    const langIsZh = docLang.startsWith('zh');

    let markerSeries = null;
    if (probability.length) {
      const probSeries = chart.addLineSeries({
        color: '#2563eb',
        lineWidth: 2,
        priceScaleId: 'right',
        priceFormat: { type: 'price', precision: 3, minMove: 0.001 },
      });
      probSeries.setData(probability);
      markerSeries = probSeries;
      const thresholds = payload && typeof payload === 'object' ? payload.thresholds || {} : {};
      const entry = Number(thresholds.entry);
      const exit = Number(thresholds.exit);
      if (Number.isFinite(entry)) {
        probSeries.createPriceLine({
          price: entry,
          color: '#16a34a',
          lineStyle: 2,
          lineWidth: 1,
          title: langIsZh ? '入场阈值' : 'Entry',
        });
      }
      if (Number.isFinite(exit)) {
        probSeries.createPriceLine({
          price: exit,
          color: '#dc2626',
          lineStyle: 2,
          lineWidth: 1,
          title: langIsZh ? '离场阈值' : 'Exit',
        });
      }
    }

    if (position.length) {
      const posSeries = chart.addLineSeries({
        color: '#0f766e',
        lineWidth: 2,
        priceScaleId: 'left',
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      });
      posSeries.setData(position);
      markerSeries = markerSeries || posSeries;
    }

    if (!probability.length && !position.length && fallback.length) {
      const equitySeries = chart.addLineSeries({
        color: '#6366f1',
        lineWidth: 2,
        priceScaleId: 'right',
      });
      equitySeries.setData(fallback);
      markerSeries = markerSeries || equitySeries;
    }

    const markers = buildMarkers(payload && typeof payload === 'object' ? payload.events || [] : [], langIsZh);
    if (markerSeries && markers.length) {
      markerSeries.setMarkers(markers);
    }

    const overviewPane = document.getElementById('overview-pane');
    if (overviewPane && typeof MutationObserver !== 'undefined') {
      const observer = new MutationObserver(() => {
        if (overviewPane.classList.contains('active')) {
          resizeChart();
        }
      });
      observer.observe(overviewPane, { attributes: true, attributeFilter: ['class'] });
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderChart);
  } else {
    renderChart();
  }
})();
