(() => {
  const viewport = document.querySelector('[data-role="watchlist-viewport"]');
  if (!viewport) return;

  const itemsEl = viewport.querySelector('[data-role="watchlist-items"]');
  const spacer = viewport.querySelector('[data-role="watchlist-spacer"]');
  const statusEl = document.querySelector('[data-role="watchlist-status"]');
  const statusDot = statusEl && statusEl.querySelector('.status-dot');
  const statusText = statusEl && statusEl.querySelector('.status-text');

  const apiUrl = '/api/market/watchlist/';
  const rowHeight = 56;
  const overscan = 4;
  const lang = (document.documentElement.getAttribute('lang') || 'zh').toLowerCase();
  const TEXT = lang.startsWith('zh')
    ? {
        connected: '已连接',
        reconnecting: '重连中',
        offline: '离线',
        empty: '暂无自选股',
      }
    : {
        connected: 'Connected',
        reconnecting: 'Reconnecting',
        offline: 'Offline',
        empty: 'No watchlist symbols',
      };

  let items = [];
  let renderRaf = null;
  let socket = null;
  let reconnectTimer = null;
  let reconnectAttempts = 0;

  const itemMap = new Map();
  const flashTimers = new Map();

  function normalizeSymbol(symbol) {
    return (symbol || '').toString().trim().toUpperCase();
  }

  function setStatus(state) {
    if (!statusText || !statusDot) return;
    statusText.textContent = TEXT[state] || TEXT.offline;
    statusDot.classList.remove('is-connected', 'is-reconnecting', 'is-offline');
    if (state === 'connected') {
      statusDot.classList.add('is-connected');
    } else if (state === 'reconnecting') {
      statusDot.classList.add('is-reconnecting');
    } else {
      statusDot.classList.add('is-offline');
    }
  }

  function scheduleRender() {
    if (renderRaf) return;
    renderRaf = window.requestAnimationFrame(renderVirtual);
  }

  function setItems(payloadItems) {
    items = Array.isArray(payloadItems) ? payloadItems : [];
    itemMap.clear();
    items.forEach((item) => {
      const symbol = normalizeSymbol(item.symbol);
      if (!symbol) return;
      item.symbol = symbol;
      item.series = Array.isArray(item.series) ? item.series.map(Number).filter((val) => Number.isFinite(val)) : [];
      item.price = Number.isFinite(item.price) ? item.price : Number.parseFloat(item.price);
      if (!Number.isFinite(item.price)) {
        item.price = item.series.length ? item.series[item.series.length - 1] : null;
      }
      item.change_pct = Number.isFinite(item.change_pct) ? item.change_pct : Number.parseFloat(item.change_pct);
      itemMap.set(symbol, item);
    });
    if (spacer) {
      spacer.style.height = `${items.length * rowHeight}px`;
    }
    scheduleRender();
  }

  function formatPrice(value) {
    if (!Number.isFinite(value)) return '--';
    return value.toFixed(2);
  }

  function formatChange(value) {
    if (!Number.isFinite(value)) return '--';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  }

  function buildSparkline(series) {
    const width = 48;
    const height = 24;
    if (!Array.isArray(series) || series.length < 2) {
      return { path: `M0 ${height / 2} L${width} ${height / 2}`, trend: 'flat' };
    }
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    series.forEach((value) => {
      if (!Number.isFinite(value)) return;
      min = Math.min(min, value);
      max = Math.max(max, value);
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return { path: `M0 ${height / 2} L${width} ${height / 2}`, trend: 'flat' };
    }
    const range = max - min || 1;
    const step = width / (series.length - 1);
    const points = series.map((value, index) => {
      const normalized = (value - min) / range;
      const x = index * step;
      const y = height - normalized * height;
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`;
    });
    const trend = series[series.length - 1] >= series[0] ? 'up' : 'down';
    return { path: points.join(' '), trend };
  }

  function renderVirtual() {
    renderRaf = null;
    if (!itemsEl || !viewport) return;
    if (!items.length) {
      itemsEl.innerHTML = `<div class="watchlist-empty">${TEXT.empty}</div>`;
      return;
    }
    const scrollTop = viewport.scrollTop;
    const height = viewport.clientHeight || 0;
    const startIndex = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
    const endIndex = Math.min(items.length, Math.ceil((scrollTop + height) / rowHeight) + overscan);

    itemsEl.innerHTML = '';
    const fragment = document.createDocumentFragment();

    for (let i = startIndex; i < endIndex; i += 1) {
      const item = items[i];
      const row = document.createElement('div');
      row.className = 'watchlist-row';
      row.style.transform = `translateY(${i * rowHeight}px)`;
      row.dataset.symbol = item.symbol;
      row.dataset.index = i.toString();

      const spark = buildSparkline(item.series || []);
      const changeClass = Number.isFinite(item.change_pct)
        ? item.change_pct > 0
          ? 'is-up'
          : item.change_pct < 0
            ? 'is-down'
            : 'is-flat'
        : 'is-flat';

      row.innerHTML = `
        <div class="watchlist-left">
          <span class="watchlist-symbol">${item.symbol || ''}</span>
          <svg class="watchlist-spark ${spark.trend}" viewBox="0 0 48 24" aria-hidden="true">
            <path d="${spark.path}"></path>
          </svg>
        </div>
        <div class="watchlist-right">
          <span class="watchlist-price">${formatPrice(item.price)}</span>
          <span class="watchlist-change ${changeClass}">${formatChange(item.change_pct)}</span>
        </div>
      `;
      fragment.appendChild(row);
    }

    itemsEl.appendChild(fragment);
  }

  function applyFlash(symbol, direction) {
    if (!itemsEl) return;
    const row = itemsEl.querySelector(`.watchlist-row[data-symbol="${symbol}"]`);
    if (!row) return;
    row.classList.remove('flash-up', 'flash-down');
    row.classList.add(direction === 'up' ? 'flash-up' : 'flash-down');
    if (flashTimers.has(symbol)) {
      clearTimeout(flashTimers.get(symbol));
    }
    const timer = setTimeout(() => {
      row.classList.remove('flash-up', 'flash-down');
      flashTimers.delete(symbol);
    }, 520);
    flashTimers.set(symbol, timer);
  }

  function updateItem(payload) {
    const symbol = normalizeSymbol(payload.symbol || '');
    if (!symbol || !itemMap.has(symbol)) return;
    const item = itemMap.get(symbol);
    const price = Number.parseFloat(payload.price);
    const changePct = Number.parseFloat(payload.change_pct);
    const prevPrice = Number.isFinite(item.price) ? item.price : null;

    if (Number.isFinite(price)) {
      item.price = price;
      if (Array.isArray(item.series)) {
        item.series.push(price);
        if (item.series.length > 20) {
          item.series = item.series.slice(-20);
        }
      }
    }
    if (Number.isFinite(changePct)) {
      item.change_pct = changePct;
    } else if (Array.isArray(item.series) && item.series.length >= 2) {
      const last = item.series[item.series.length - 1];
      const prev = item.series[item.series.length - 2];
      if (prev) {
        item.change_pct = (last / prev - 1) * 100;
      }
    }

    scheduleRender();
    if (Number.isFinite(price) && Number.isFinite(prevPrice)) {
      const direction = price > prevPrice ? 'up' : price < prevPrice ? 'down' : null;
      if (direction) {
        window.requestAnimationFrame(() => applyFlash(symbol, direction));
      }
    }
  }

  function connectSocket() {
    if (!window.WebSocket) return;
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;

    setStatus('reconnecting');
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/market/`);

    socket.onopen = () => {
      reconnectAttempts = 0;
      setStatus('connected');
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const eventType = payload && (payload.type || payload.event);
        if (payload && (eventType === 'market_update' || eventType === 'market.update' || payload.symbol)) {
          updateItem(payload);
        }
      } catch (error) {
        return;
      }
    };

    socket.onclose = () => {
      setStatus('offline');
      scheduleReconnect();
    };

    socket.onerror = () => {
      setStatus('offline');
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectAttempts += 1;
    const delay = Math.min(12000, 1000 * Math.pow(1.4, reconnectAttempts));
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectSocket();
    }, delay);
  }

  async function fetchWatchlist() {
    try {
      const response = await fetch(`${apiUrl}?limit=20&interval=1d`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload && payload.error ? payload.error : 'Failed');
      }
      setItems(payload.items || []);
    } catch (error) {
      setItems([]);
    }
  }

  viewport.addEventListener('scroll', () => {
    scheduleRender();
  });

  window.addEventListener('resize', () => {
    scheduleRender();
  });

  document.addEventListener('watchlist:refresh', () => {
    fetchWatchlist();
  });

  setStatus('offline');
  fetchWatchlist();
  connectSocket();

  window.watchlistRealtime = {
    refresh: fetchWatchlist,
  };
})();
