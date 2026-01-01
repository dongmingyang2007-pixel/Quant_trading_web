(() => {
  function detectDarkTheme() {
    const html = document.documentElement;
    return html && html.getAttribute('data-theme') === 'dark';
  }

  const container = document.getElementById('arena-chart');
  const payloadEl = document.getElementById('arena-series-data');
  if (!container || !payloadEl) return;

  let payload = [];
  try {
    payload = JSON.parse(payloadEl.textContent || '[]');
  } catch (error) {
    console.error('Invalid compare payload', error);
    return;
  }
  if (!payload.length || typeof LightweightCharts === 'undefined') return;

  const isDark = detectDarkTheme();
  const chart = LightweightCharts.createChart(container, {
    layout: {
      background: { color: isDark ? '#03050c' : '#ffffff' },
      textColor: isDark ? '#e2e8f0' : '#0f172a',
    },
    rightPriceScale: {
      visible: true,
      borderVisible: false,
    },
    timeScale: {
      borderVisible: false,
    },
    grid: {
      vertLines: { visible: false },
      horzLines: { color: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(15,23,42,0.05)' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
    localization: {
      locale: document.documentElement.lang || 'en',
    },
  });

  payload.forEach((series, index) => {
    if (!Array.isArray(series.points) || !series.points.length) return;
    const line = chart.addLineSeries({
      color: series.color || ['#2563eb', '#10b981', '#f97316', '#8b5cf6', '#ef4444'][index % 5],
      lineWidth: 3,
      title: series.label || `Strategy ${index + 1}`,
    });
    line.setData(series.points);
  });

  if (typeof ResizeObserver === 'function') {
    const observer = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth, height: 420 });
    });
    observer.observe(container);
  }
  chart.applyOptions({ width: container.clientWidth, height: 420 });
})();
