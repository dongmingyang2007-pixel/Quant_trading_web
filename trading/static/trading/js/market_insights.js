(() => {
  const apiMeta = document.querySelector('meta[name="market-api"]');
  const apiUrl = apiMeta ? apiMeta.getAttribute('content') : window.MARKET_API_URL || '/market/api/';
  const langMeta = document.querySelector('meta[name="market-lang"]');
  const backtestMeta = document.querySelector('meta[name="backtest-base"]');
  const backtestBase = backtestMeta ? backtestMeta.getAttribute('content') : '/backtest/';
  const docLang = document.documentElement.getAttribute('lang');
  const langPrefix = ((langMeta && langMeta.getAttribute('content')) || docLang || navigator.language || 'zh')
    .toLowerCase()
    .slice(0, 2);

  const listContainer = document.querySelector('[data-role="ranking-list"]');
  const rankTabs = Array.prototype.slice.call(document.querySelectorAll('[data-role="rank-tab"]'));
  const rankDesc = document.querySelector('[data-role="rank-desc"]');
  const statusText = document.querySelector('[data-role="status-text"]');
  const sourceText = document.querySelector('[data-role="source-text"]');
  const statusSection = document.querySelector('.market-status');
  const timeframeButtons = Array.prototype.slice.call(document.querySelectorAll('.market-timeframe'));
  const searchForm = document.getElementById('market-search-form');
  const searchInput = document.getElementById('market-search-input');
  const cardTemplate = document.getElementById('market-card-template');
  const skeletonTemplate = document.getElementById('market-card-skeleton');
  const suggestionList = document.getElementById('market-suggestions-list');
  const recentChips = document.querySelector('[data-role="recent-chips"]');
  const recentCount = document.querySelector('[data-role="recent-count"]');
  const watchlistChips = document.querySelector('[data-role="watchlist-chips"]');
  const watchCount = document.querySelector('[data-role="watch-count"]');
  const watchAddBtn = document.getElementById('market-watch-add');
  const typeaheadPanel = document.querySelector('[data-role="typeahead-panel"]');
  const typeaheadList = typeaheadPanel && typeaheadPanel.querySelector('[data-role="typeahead-list"]');
  const typeaheadHint = typeaheadPanel && typeaheadPanel.querySelector('[data-role="typeahead-hint"]');
  const detailModalEl = document.getElementById('marketDetailModal');
  const detailTitle = detailModalEl && detailModalEl.querySelector('[data-role="detail-title"]');
  const detailSource = detailModalEl && detailModalEl.querySelector('[data-role="detail-source"]');
  const detailUpdated = detailModalEl && detailModalEl.querySelector('[data-role="detail-updated"]');
  const detailStatus = detailModalEl && detailModalEl.querySelector('[data-role="detail-status"]');
  const detailChartEl = document.getElementById('market-detail-chart');
  const detailIndicatorEl = document.getElementById('market-detail-indicator');
  const detailOverlaySelect = detailModalEl && detailModalEl.querySelector('#detail-overlay-select');
  const detailIndicatorSelect = detailModalEl && detailModalEl.querySelector('#detail-indicator-select');
  const detailDrawButtons = detailModalEl
    ? Array.prototype.slice.call(detailModalEl.querySelectorAll('.draw-btn'))
    : [];
  const detailTimeframes = detailModalEl
    ? Array.prototype.slice.call(detailModalEl.querySelectorAll('.detail-timeframe'))
    : [];
  const marketSocketUrl = (() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}/ws/market/`;
  })();

  const initialBtn = document.querySelector('.market-timeframe.is-active');
  let currentTimeframe = (initialBtn && initialBtn.getAttribute('data-timeframe')) || '1mo';
  let currentListType = 'gainers';
  let suggestionPool = [];
  let recentPool = [];
  let watchPool = [];
  let hideTypeaheadTimer = null;
  let typeaheadOptions = [];
  let typeaheadActiveIndex = -1;
  let retryTimer = null;
  let lastRequest = { query: '', options: {} };
  let marketSocket = null;
  let socketRetryTimer = null;
  let detailModal = null;
  let detailManager = null;
  let detailSymbol = '';
  let detailRange = '1d';
  let detailRetryTimer = null;

  const TEXT = langPrefix === 'zh'
    ? {
        timeframes: { '1d': '近1日', '5d': '近5日', '1mo': '近1月', '6mo': '近6月' },
        loading: '正在加载',
        dataSuffix: '数据…',
        updated: '数据已更新',
        justNow: '刚刚',
        retrying: '请求过快，正在重试',
        emptySymbol: '暂无可展示的标的。',
        emptyList: '暂无数据',
        statusError: '加载失败，请稍后再试。',
        genericError: '加载失败',
        updatedLabel: '更新：',
        emptyChips: '暂无推荐',
        emptyWatchlist: '还没有自选股。',
        statusNeedSymbol: '请先输入股票代码。',
        watchAdded: (symbol) => `已加入关注：${symbol}`,
        watchRemoved: (symbol) => `已移除关注：${symbol}`,
        typeaheadTrending: '热门推荐',
        typeaheadRecent: '最近检索',
        typeaheadWatch: '自选股',
        typeaheadAdd: '加入自选',
        typeaheadRemove: '移除',
        typeaheadEmpty: '没有匹配的股票代码。',
        typeaheadHint: '↑↓ 选择，Enter 跳转或加入自选列表',
        historyCleared: '最近检索已清空',
        historyDeleted: (symbol) => `已删除 ${symbol}`,
        detailLoading: '正在加载K线…',
        detailEmpty: '暂无行情数据',
        detailError: '加载失败',
        detailFallback: (requested, used) => `无 ${requested} 数据，已显示 ${used} 历史`,
        volumeLabel: '成交量',
        sourcePrefix: '数据来源：',
        sourceLabels: {
          alpaca: 'Alpaca',
          yfinance: 'Yahoo Finance',
          cache: '缓存',
          unknown: '未知',
        },
      }
    : {
        timeframes: { '1d': '1D', '5d': '5D', '1mo': '1M', '6mo': '6M' },
        loading: 'Loading',
        dataSuffix: 'data…',
        updated: 'Data refreshed',
        justNow: 'just now',
        retrying: 'Rate limited, retrying',
        emptySymbol: 'No symbols to display.',
        emptyList: 'No data',
        statusError: 'Failed to load, please try again later.',
        genericError: 'Failed to load',
        updatedLabel: 'Updated:',
        emptyChips: 'No suggestions available.',
        emptyWatchlist: 'Watchlist is empty.',
        statusNeedSymbol: 'Enter a ticker before performing this action.',
        watchAdded: (symbol) => `Added ${symbol} to watchlist`,
        watchRemoved: (symbol) => `Removed ${symbol} from watchlist`,
        typeaheadTrending: 'Trending',
        typeaheadRecent: 'Recent',
        typeaheadWatch: 'Watchlist',
        typeaheadAdd: 'Add to watchlist',
        typeaheadRemove: 'Remove',
        typeaheadEmpty: 'No matching tickers yet.',
        typeaheadHint: 'Use ↑↓ to browse, Enter to open or add to watchlist',
        historyCleared: 'Recent searches cleared',
        historyDeleted: (symbol) => `Removed ${symbol}`,
        detailLoading: 'Loading candles…',
        detailEmpty: 'No data available',
        detailError: 'Failed to load',
        detailFallback: (requested, used) => `No ${requested} data, showing ${used} history`,
        volumeLabel: 'Volume',
        sourcePrefix: 'Data source: ',
        sourceLabels: {
          alpaca: 'Alpaca',
          yfinance: 'Yahoo Finance',
          cache: 'Cache',
          unknown: 'Unknown',
        },
      };

  class ChartManager {
    static registry = [];

    constructor({ container, indicatorContainer, langPrefix, onStatus }) {
      this.container = container;
      this.indicatorContainer = indicatorContainer;
      this.langPrefix = langPrefix || 'zh';
      this.onStatus = onStatus;
      this.chart = null;
      this.candleSeries = null;
      this.overlaySeries = [];
      this.indicatorChart = null;
      this.indicatorSeries = [];
      this.overlayMode = 'none';
      this.indicatorMode = 'none';
      this.ohlcData = [];
      this.overlayCanvas = null;
      this.overlayCtx = null;
      this.overlayRatio = 1;
      this.drawings = [];
      this.activeDrawing = null;
      this.drawMode = 'none';
      this._syncing = false;
      this._syncingIndicator = false;
      this._linkedCharts = new Set();
      this._syncTargets = new Set();
      this._resizeObserver = null;
    }

    init() {
      const chartLib = window.LightweightCharts;
      if (!chartLib || typeof chartLib.createChart !== 'function') {
        if (this.onStatus) {
          this.onStatus(TEXT.detailError, true);
        }
        return false;
      }
      if (!this.container) return false;
      const baseOptions = {
        layout: { background: { color: '#ffffff' }, textColor: '#0f172a' },
        grid: { vertLines: { color: 'rgba(148, 163, 184, 0.3)' }, horzLines: { color: 'rgba(148, 163, 184, 0.3)' } },
        rightPriceScale: { borderColor: 'rgba(148, 163, 184, 0.4)' },
        timeScale: { borderColor: 'rgba(148, 163, 184, 0.4)' },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          axisPressedMouseMove: true,
          mouseWheel: true,
          pinch: true,
        },
      };
      this.chart = chartLib.createChart(this.container, {
        ...baseOptions,
        width: this.container.clientWidth || 680,
        height: this.container.clientHeight || 360,
      });
      this.candleSeries = this.chart.addCandlestickSeries({
        upColor: '#16a34a',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#16a34a',
        wickDownColor: '#ef4444',
      });

      if (this.indicatorContainer) {
        const indicatorWidth = this.indicatorContainer.clientWidth || this.container.clientWidth || 680;
        const indicatorHeight = this.indicatorContainer.clientHeight || 160;
        this.indicatorChart = chartLib.createChart(this.indicatorContainer, {
          ...baseOptions,
          width: indicatorWidth,
          height: indicatorHeight,
          timeScale: { borderColor: 'rgba(148, 163, 184, 0.4)', visible: false },
        });
      }

      this._initOverlay();
      this._bindResize();
      this._bindTimeSync();
      ChartManager.register(this);
      return true;
    }

    static register(instance) {
      ChartManager.registry.forEach((other) => {
        if (other !== instance) {
          instance.linkWith(other);
        }
      });
      ChartManager.registry.push(instance);
    }

    linkWith(other) {
      if (!other || other === this || this._linkedCharts.has(other)) return;
      this._linkedCharts.add(other);
      other._linkedCharts.add(this);
      this._subscribeSync(other);
      other._subscribeSync(this);
    }

    _subscribeSync(target) {
      if (this._syncTargets.has(target)) return;
      this._syncTargets.add(target);
      const syncRange = (range) => {
        if (!range || this._syncing || !target.chart) return;
        this._syncing = true;
        target._syncing = true;
        target.chart.timeScale().setVisibleRange(range);
        target._syncing = false;
        this._syncing = false;
      };
      this.chart.timeScale().subscribeVisibleTimeRangeChange(syncRange);
    }

    setData(bars) {
      this.ohlcData = Array.isArray(bars) ? bars : [];
      if (this.candleSeries) {
        this.candleSeries.setData(this.ohlcData);
      }
      this.updateOverlay();
      this.updateIndicator();
      if (this.chart) {
        this.chart.timeScale().fitContent();
      }
      this.renderOverlay();
    }

    setOverlay(mode) {
      this.overlayMode = mode || 'none';
      this.updateOverlay();
    }

    setIndicator(mode) {
      this.indicatorMode = mode || 'none';
      this.updateIndicator();
    }

    setDrawMode(mode) {
      if (!mode) return;
      this.drawMode = mode;
    }

    clearDrawings() {
      this.drawings = [];
      this.activeDrawing = null;
      this.renderOverlay();
    }

    resize() {
      if (this.chart && this.container) {
        this.chart.applyOptions({
          width: this.container.clientWidth || 0,
          height: this.container.clientHeight || 0,
        });
      }
      if (this.indicatorChart && this.indicatorContainer) {
        this.indicatorChart.applyOptions({
          width: this.indicatorContainer.clientWidth || 0,
          height: this.indicatorContainer.clientHeight || 0,
        });
      }
      this._resizeOverlay();
      this.renderOverlay();
    }

    updateOverlay() {
      if (!this.chart || !this.candleSeries) return;
      this.overlaySeries.forEach((series) => this.chart.removeSeries(series));
      this.overlaySeries = [];
      if (!this.ohlcData.length) return;
      if (this.overlayMode === 'none') return;
      const indicatorLib = window.technicalindicators;
      if (!indicatorLib) return;
      const closes = this._getCloses();
      if (!closes.length) return;

      if (this.overlayMode === 'sma') {
        const values = indicatorLib.SMA.calculate({ period: 20, values: closes });
        const series = this.chart.addLineSeries({ color: '#0ea5e9', lineWidth: 2 });
        series.setData(this._mapSeries(values, 20));
        this.overlaySeries.push(series);
      } else if (this.overlayMode === 'ema') {
        const values = indicatorLib.EMA.calculate({ period: 20, values: closes });
        const series = this.chart.addLineSeries({ color: '#f59e0b', lineWidth: 2 });
        series.setData(this._mapSeries(values, 20));
        this.overlaySeries.push(series);
      } else if (this.overlayMode === 'bbands') {
        const values = indicatorLib.BollingerBands.calculate({ period: 20, values: closes, stdDev: 2 });
        const upper = this.chart.addLineSeries({ color: 'rgba(59, 130, 246, 0.8)', lineWidth: 1 });
        const middle = this.chart.addLineSeries({ color: 'rgba(14, 165, 233, 0.9)', lineWidth: 1 });
        const lower = this.chart.addLineSeries({ color: 'rgba(59, 130, 246, 0.8)', lineWidth: 1 });
        upper.setData(this._mapBand(values, 20, 'upper'));
        middle.setData(this._mapBand(values, 20, 'middle'));
        lower.setData(this._mapBand(values, 20, 'lower'));
        this.overlaySeries.push(upper, middle, lower);
      }
    }

    updateIndicator() {
      if (!this.indicatorChart || !this.indicatorContainer) return;
      this.indicatorSeries.forEach((series) => this.indicatorChart.removeSeries(series));
      this.indicatorSeries = [];
      if (!this.ohlcData.length || this.indicatorMode === 'none') {
        this.indicatorContainer.hidden = true;
        return;
      }
      this.indicatorContainer.hidden = false;
      this.resize();
      const indicatorLib = window.technicalindicators;
      if (!indicatorLib) return;
      const closes = this._getCloses();
      if (!closes.length) return;

      if (this.indicatorMode === 'rsi') {
        const values = indicatorLib.RSI.calculate({ period: 14, values: closes });
        const line = this.indicatorChart.addLineSeries({ color: '#6366f1', lineWidth: 2 });
        line.setData(this._mapSeries(values, 14));
        this.indicatorSeries.push(line);
      } else if (this.indicatorMode === 'macd') {
        const values = indicatorLib.MACD.calculate({
          values: closes,
          fastPeriod: 12,
          slowPeriod: 26,
          signalPeriod: 9,
          SimpleMAOscillator: false,
          SimpleMASignal: false,
        });
        const startIndex = this.ohlcData.length - values.length;
        const histogram = this.indicatorChart.addHistogramSeries({ color: '#94a3b8' });
        const macdLine = this.indicatorChart.addLineSeries({ color: '#0ea5e9', lineWidth: 2 });
        const signalLine = this.indicatorChart.addLineSeries({ color: '#f97316', lineWidth: 2 });
        const histogramData = values.map((item, index) => {
          const time = this.ohlcData[startIndex + index].time;
          const value = typeof item.histogram === 'number' ? item.histogram : 0;
          return {
            time,
            value,
            color: value >= 0 ? 'rgba(34, 197, 94, 0.65)' : 'rgba(239, 68, 68, 0.65)',
          };
        });
        const macdData = values.map((item, index) => ({
          time: this.ohlcData[startIndex + index].time,
          value: item.MACD,
        }));
        const signalData = values.map((item, index) => ({
          time: this.ohlcData[startIndex + index].time,
          value: item.signal,
        }));
        histogram.setData(histogramData);
        macdLine.setData(macdData);
        signalLine.setData(signalData);
        this.indicatorSeries.push(histogram, macdLine, signalLine);
      }
      this.indicatorChart.timeScale().fitContent();
    }

    _getCloses() {
      return this.ohlcData.map((bar) => bar.close).filter((val) => typeof val === 'number');
    }

    _mapSeries(values, period) {
      const offset = Math.max(0, period - 1);
      return values.map((value, index) => ({
        time: this.ohlcData[index + offset].time,
        value,
      }));
    }

    _mapBand(values, period, key) {
      const offset = Math.max(0, period - 1);
      return values.map((value, index) => ({
        time: this.ohlcData[index + offset].time,
        value: value[key],
      }));
    }

    _initOverlay() {
      if (!this.container) return;
      this.overlayCanvas = document.createElement('canvas');
      this.overlayCanvas.className = 'market-draw-layer';
      this.container.appendChild(this.overlayCanvas);
      this.overlayCtx = this.overlayCanvas.getContext('2d');
      this._resizeOverlay();
      this.container.addEventListener('pointerdown', (event) => this._handlePointerDown(event));
      window.addEventListener('pointermove', (event) => this._handlePointerMove(event));
      window.addEventListener('pointerup', (event) => this._handlePointerUp(event));
    }

    _bindResize() {
      if (typeof ResizeObserver === 'undefined') return;
      if (this._resizeObserver) return;
      this._resizeObserver = new ResizeObserver(() => this.resize());
      if (this.container) this._resizeObserver.observe(this.container);
      if (this.indicatorContainer) this._resizeObserver.observe(this.indicatorContainer);
    }

    _bindTimeSync() {
      if (!this.chart) return;
      const timeScale = this.chart.timeScale();
      timeScale.subscribeVisibleTimeRangeChange(() => this.renderOverlay());
      const priceScale = this.chart.priceScale && this.chart.priceScale('right');
      if (priceScale && typeof priceScale.subscribeVisibleLogicalRangeChange === 'function') {
        priceScale.subscribeVisibleLogicalRangeChange(() => this.renderOverlay());
      }
      if (this.indicatorChart) {
        const indicatorScale = this.indicatorChart.timeScale();
        timeScale.subscribeVisibleTimeRangeChange((range) => {
          if (this._syncingIndicator || !range) return;
          this._syncingIndicator = true;
          indicatorScale.setVisibleRange(range);
          this._syncingIndicator = false;
        });
        indicatorScale.subscribeVisibleTimeRangeChange((range) => {
          if (this._syncingIndicator || !range) return;
          this._syncingIndicator = true;
          timeScale.setVisibleRange(range);
          this._syncingIndicator = false;
        });
      }
    }

    _resizeOverlay() {
      if (!this.overlayCanvas || !this.container) return;
      const ratio = window.devicePixelRatio || 1;
      this.overlayRatio = ratio;
      const width = this.container.clientWidth || 0;
      const height = this.container.clientHeight || 0;
      this.overlayCanvas.width = Math.floor(width * ratio);
      this.overlayCanvas.height = Math.floor(height * ratio);
      this.overlayCanvas.style.width = `${width}px`;
      this.overlayCanvas.style.height = `${height}px`;
    }

    _handlePointerDown(event) {
      if (!this.chart || !this.candleSeries) return;
      if (event.button !== 0) return;
      if (!['line', 'rect'].includes(this.drawMode)) return;
      event.preventDefault();
      event.stopPropagation();
      const point = this._eventToPoint(event);
      if (!point) return;
      this.activeDrawing = { type: this.drawMode, start: point, end: point };
    }

    _handlePointerMove(event) {
      if (!this.activeDrawing) return;
      const point = this._eventToPoint(event);
      if (!point) return;
      this.activeDrawing.end = point;
      this.renderOverlay();
    }

    _handlePointerUp() {
      if (!this.activeDrawing) return;
      this.drawings.push(this.activeDrawing);
      this.activeDrawing = null;
      this.renderOverlay();
    }

    _eventToPoint(event) {
      if (!this.container || !this.chart || !this.candleSeries) return null;
      const rect = this.container.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const time = this.chart.timeScale().coordinateToTime(x);
      const price = this.candleSeries.coordinateToPrice(y);
      if (time === null || price === null) return null;
      return { time, price };
    }

    renderOverlay() {
      if (!this.overlayCanvas || !this.overlayCtx || !this.chart || !this.candleSeries) return;
      const ctx = this.overlayCtx;
      ctx.setTransform(this.overlayRatio, 0, 0, this.overlayRatio, 0, 0);
      ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)';
      ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';

      const drawShape = (shape) => {
        const x1 = this.chart.timeScale().timeToCoordinate(shape.start.time);
        const x2 = this.chart.timeScale().timeToCoordinate(shape.end.time);
        const y1 = this.candleSeries.priceToCoordinate(shape.start.price);
        const y2 = this.candleSeries.priceToCoordinate(shape.end.price);
        if (x1 === null || x2 === null || y1 === null || y2 === null) return;
        if (shape.type === 'line') {
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        } else if (shape.type === 'rect') {
          const left = Math.min(x1, x2);
          const top = Math.min(y1, y2);
          const width = Math.abs(x2 - x1);
          const height = Math.abs(y2 - y1);
          ctx.fillRect(left, top, width, height);
          ctx.strokeRect(left, top, width, height);
        }
      };

      this.drawings.forEach(drawShape);
      if (this.activeDrawing) {
        drawShape(this.activeDrawing);
      }
    }
  }

  if (typeaheadHint) {
    typeaheadHint.textContent = TEXT.typeaheadHint;
  }

  function getCsrfToken() {
    const formInput = document.querySelector('input[name="csrfmiddlewaretoken"]');
    if (formInput && formInput.value) {
      return formInput.value;
    }
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta && meta.getAttribute('content')) {
      return meta.getAttribute('content');
    }
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    if (match) {
      return decodeURIComponent(match[1]);
    }
    return '';
  }

  function requestRecentAction(action, symbol) {
    const normalized = normalizeSymbol(symbol);
    const options = { recentAction: action };
    if (normalized) {
      options.recentTarget = normalized;
    }
    loadData('', options);
  }

  timeframeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.classList.contains('is-active')) return;
      timeframeButtons.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      currentTimeframe = btn.getAttribute('data-timeframe') || '1mo';
      if (searchInput) {
        searchInput.value = '';
      }
      loadData();
    });
  });

  if (searchForm) {
    searchForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const value = (searchInput && searchInput.value.trim()) || '';
      loadData(value);
    });
  }

  if (watchAddBtn) {
    watchAddBtn.addEventListener('click', () => {
      if (!searchInput) return;
      const symbol = (searchInput.value || '').trim().toUpperCase();
      if (!symbol) {
        setStatus(TEXT.statusNeedSymbol);
        searchInput.focus();
        return;
      }
      loadData(symbol, { watchAction: 'add' });
    });
  }

  function setStatus(text) {
    if (statusText) {
      statusText.textContent = text;
    }
  }

  function updateRankDescription(type) {
    if (!rankDesc) return;
    const descMap = {
      gainers: rankDesc.dataset.descGainers,
      losers: rankDesc.dataset.descLosers,
      most_active: rankDesc.dataset.descMostActive,
    };
    const nextText = descMap[type] || rankDesc.dataset.descGainers || '';
    if (nextText) {
      rankDesc.textContent = nextText;
    }
  }

  function setActiveListType(type) {
    currentListType = type;
    rankTabs.forEach((tab) => {
      const tabType = tab.dataset.list || '';
      tab.classList.toggle('is-active', tabType === type);
    });
    updateRankDescription(type);
  }

  function setRetryingState(retryAfterSeconds) {
    const delaySeconds = Math.min(Math.max(parseInt(retryAfterSeconds, 10) || 3, 2), 30);
    if (statusSection) {
      statusSection.classList.add('is-retrying');
    }
    setStatus(TEXT.retrying);
    if (retryTimer) {
      clearTimeout(retryTimer);
    }
    retryTimer = setTimeout(() => {
      if (statusSection) {
        statusSection.classList.remove('is-retrying');
      }
      loadData(lastRequest.query || '', lastRequest.options || {});
    }, delaySeconds * 1000);
  }

  function setSource(sourceKey) {
    if (!sourceText) return;
    const labels = TEXT.sourceLabels || {};
    const normalizedKey = sourceKey && labels[sourceKey] ? sourceKey : 'unknown';
    const label = labels[normalizedKey] || labels.unknown || '';
    sourceText.textContent = `${TEXT.sourcePrefix || ''}${label}`;
  }

  function applyLiveUpdate(update) {
    if (!update || typeof update !== 'object') return;
    const symbol = normalizeSymbol(update.symbol || '');
    if (!symbol) return;
    const price = typeof update.price === 'number' ? update.price : Number.parseFloat(update.price);
    const changePct = typeof update.change_pct === 'number' ? update.change_pct : Number.parseFloat(update.change_pct);
    document.querySelectorAll(`.market-card[data-symbol="${symbol}"]`).forEach((card) => {
      const priceEl = card.querySelector('[data-role="price"]');
      const dayEl = card.querySelector('[data-role="day-change"]');
      if (priceEl && Number.isFinite(price)) {
        priceEl.textContent = price.toFixed(2);
      }
      if (dayEl && Number.isFinite(changePct)) {
        dayEl.textContent = formatChange(changePct);
        applyChangeState(dayEl, changePct, card.dataset.invert === '1', true);
      }
    });
  }

  function setDetailStatus(message, isError) {
    if (!detailStatus) return;
    detailStatus.textContent = message || '';
    detailStatus.hidden = !message;
    detailStatus.classList.toggle('is-error', Boolean(isError));
  }

  function ensureDetailModal() {
    if (!detailModalEl || typeof bootstrap === 'undefined') return null;
    if (!detailModal) {
      detailModal = new bootstrap.Modal(detailModalEl);
    }
    return detailModal;
  }

  function resizeDetailChart() {
    if (!detailManager) return;
    detailManager.resize();
  }

  function ensureDetailChart() {
    if (!detailChartEl) return false;
    if (detailManager) return true;
    detailManager = new ChartManager({
      container: detailChartEl,
      indicatorContainer: detailIndicatorEl,
      langPrefix,
      onStatus: setDetailStatus,
    });
    const ready = detailManager.init();
    if (ready && detailOverlaySelect) {
      detailManager.setOverlay(detailOverlaySelect.value);
    }
    if (ready && detailIndicatorSelect) {
      detailManager.setIndicator(detailIndicatorSelect.value);
    }
    if (ready && detailDrawButtons.length) {
      const activeDraw = detailDrawButtons.find((btn) => btn.classList.contains('is-active'));
      if (activeDraw && activeDraw.dataset.draw && activeDraw.dataset.draw !== 'clear') {
        detailManager.setDrawMode(activeDraw.dataset.draw);
      }
    }
    return ready;
  }

  async function loadDetailData(symbol, rangeKey) {
    if (!symbol) return;
    if (detailRetryTimer) {
      clearTimeout(detailRetryTimer);
      detailRetryTimer = null;
    }
    if (detailChartEl) {
      detailChartEl.classList.add('is-loading');
    }
    setDetailStatus(TEXT.detailLoading);
    if (detailSource) {
      detailSource.textContent = '';
    }
    if (detailUpdated) {
      detailUpdated.textContent = '';
    }

    const endpointBase = apiUrl || '/api/market/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({
      detail: '1',
      symbol,
      range: rangeKey,
    });
    try {
      const response = await fetch(`${endpoint}?${params.toString()}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      const payload = await response.json();
      if (response.status === 429 || payload.rate_limited) {
        const retryAfter = Math.min(Math.max(parseInt(payload.retry_after_seconds, 10) || 3, 2), 30);
        setDetailStatus(TEXT.retrying);
        detailRetryTimer = setTimeout(() => loadDetailData(symbol, rangeKey), retryAfter * 1000);
        return;
      }
      if (!response.ok) {
        setDetailStatus(payload.error || TEXT.detailError, true);
        return;
      }
      const bars = Array.isArray(payload.bars) ? payload.bars : [];
      if (!bars.length) {
        setDetailStatus(TEXT.detailEmpty);
        return;
      }
      if (!ensureDetailChart() || !detailManager) {
        return;
      }
      detailManager.setData(bars);
      if (detailSource) {
        const sourceLabel = TEXT.sourceLabels[payload.data_source] || TEXT.sourceLabels.unknown || '';
        detailSource.textContent = sourceLabel ? `${TEXT.sourcePrefix || ''}${sourceLabel}` : '';
      }
      if (detailUpdated) {
        detailUpdated.textContent = payload.generated_at ? `${TEXT.updatedLabel} ${payload.generated_at}` : '';
      }
      const tfLabel =
        payload.timeframe && (langPrefix === 'zh' ? payload.timeframe.label : payload.timeframe.label_en);
      if (detailTitle) {
        detailTitle.textContent = tfLabel ? `${symbol} · ${tfLabel}` : symbol;
      }
      const requested = payload.requested_timeframe
        ? langPrefix === 'zh'
          ? payload.requested_timeframe.label
          : payload.requested_timeframe.label_en
        : '';
      const used = tfLabel || '';
      if (requested && used && requested !== used) {
        setDetailStatus(TEXT.detailFallback(requested, used));
      } else {
        setDetailStatus('');
      }
    } catch (error) {
      setDetailStatus(TEXT.detailError, true);
    } finally {
      if (detailChartEl) {
        detailChartEl.classList.remove('is-loading');
      }
    }
  }

  function setDetailRange(rangeKey) {
    detailRange = rangeKey;
    detailTimeframes.forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.range === rangeKey);
    });
  }

  function openDetailModal(symbol) {
    detailSymbol = symbol;
    const modal = ensureDetailModal();
    if (detailTitle) {
      detailTitle.textContent = symbol;
    }
    setDetailRange(detailRange);
    if (modal) {
      modal.show();
    }
    loadDetailData(symbol, detailRange);
  }

  function connectMarketSocket() {
    if (!window.WebSocket) return;
    if (socketRetryTimer) {
      clearTimeout(socketRetryTimer);
      socketRetryTimer = null;
    }
    if (marketSocket && (marketSocket.readyState === WebSocket.OPEN || marketSocket.readyState === WebSocket.CONNECTING)) {
      return;
    }
    marketSocket = new WebSocket(marketSocketUrl);
    marketSocket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        applyLiveUpdate(payload);
      } catch (error) {
        return;
      }
    };
    marketSocket.onclose = () => {
      socketRetryTimer = setTimeout(connectMarketSocket, 3000);
    };
  }

  function clearListState(container) {
    if (!container) return;
    container.removeAttribute('data-loading');
    container.classList.remove('is-loading');
    container.innerHTML = '';
  }

  function setListLoading(container) {
    if (!container) return;
    if (!skeletonTemplate) {
      clearListState(container);
      return;
    }
    const count = Number(container.getAttribute('data-skeleton-count')) || 3;
    container.setAttribute('data-loading', 'true');
    container.classList.add('is-loading');
    container.innerHTML = '';
    for (let i = 0; i < count; i += 1) {
      container.appendChild(skeletonTemplate.content.cloneNode(true));
    }
  }

  function renderEmpty(container, message) {
    if (!container) return;
    clearListState(container);
    const div = document.createElement('div');
    div.className = 'market-list-empty';
    div.textContent = message;
    container.appendChild(div);
  }

  function renderError(container, message) {
    if (!container) return;
    clearListState(container);
    const div = document.createElement('div');
    div.className = 'market-error';
    div.textContent = message || TEXT.genericError;
    container.appendChild(div);
  }

  function formatChange(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      return '--';
    }
    const prefix = value > 0 ? '+' : '';
    return `${prefix}${value.toFixed(2)}%`;
  }

  function formatCompactNumber(value) {
    const parsed = typeof value === 'number' ? value : Number.parseFloat(value);
    if (!Number.isFinite(parsed)) {
      return '--';
    }
    const abs = Math.abs(parsed);
    if (abs >= 1e12) return `${(parsed / 1e12).toFixed(1)}T`;
    if (abs >= 1e9) return `${(parsed / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `${(parsed / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(parsed / 1e3).toFixed(1)}K`;
    return Math.round(parsed).toString();
  }

  function applyChangeState(el, value, invert, subtle) {
    if (!el) return;
    el.classList.remove('is-up', 'is-down');
    if (typeof value !== 'number' || Number.isNaN(value)) return;
    const positive = invert ? value < 0 : value > 0;
    el.classList.add(positive ? 'is-up' : 'is-down');
    if (subtle) {
      el.classList.add('subtle');
    }
  }

  function drawSparkline(canvas, series, invert) {
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (!Array.isArray(series) || series.length < 2) {
      ctx.strokeStyle = '#cbd5f5';
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    const rawValues = series.map((value) => {
      const parsed = typeof value === 'number' ? value : Number.parseFloat(value);
      if (!Number.isFinite(parsed)) {
        return null;
      }
      min = Math.min(min, parsed);
      max = Math.max(max, parsed);
      return parsed;
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      ctx.strokeStyle = '#cbd5f5';
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }
    const shouldNormalize = min < 0 || max > 1;
    const range = max - min;
    const points = rawValues.map((value) => {
      if (!Number.isFinite(value)) return 0;
      if (shouldNormalize && range === 0) {
        return 0.5;
      }
      const normalized = shouldNormalize ? (value - min) / range : value;
      return Math.min(Math.max(normalized, 0), 1);
    });

    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    if (invert) {
      gradient.addColorStop(0, 'rgba(244, 63, 94, 0.3)');
      gradient.addColorStop(1, 'rgba(244, 63, 94, 0)');
    } else {
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
    }

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(0, height - points[0] * height);
    const step = width / (points.length - 1);
    points.forEach((point, index) => {
      ctx.lineTo(step * index, height - point * height);
    });
    ctx.strokeStyle = invert ? '#f43f5e' : '#3b82f6';
    ctx.stroke();
    ctx.fillStyle = gradient;
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fill();
  }

  function updateSuggestionList(items) {
    if (!suggestionList) return;
    suggestionList.innerHTML = '';
    (items || []).slice(0, 12).forEach((symbol) => {
      if (!symbol) return;
      const option = document.createElement('option');
      option.value = symbol;
      suggestionList.appendChild(option);
    });
  }

  function renderChipGroup(container, items, options = {}) {
    if (!container) return;
    const pool = Array.isArray(items) ? items.filter(Boolean) : [];
    hideChipSkeleton(container);
    container.innerHTML = '';
    if (options.countTarget) {
      options.countTarget.textContent = pool.length;
    }
    if (!pool.length) {
      const empty = document.createElement('span');
      empty.className = 'market-chip-empty text-muted';
      empty.textContent = options.emptyText || TEXT.emptyList;
      container.appendChild(empty);
      return;
    }
    pool.forEach((symbol) => {
      if (!symbol) return;
      const chip = document.createElement('div');
      chip.className = 'market-chip';
      chip.dataset.role = 'symbol-chip';
      chip.dataset.symbol = symbol;
      chip.textContent = symbol;
      chip.tabIndex = 0;
      if (options.watch || options.allowRemove) {
        const removeBtn = document.createElement('span');
        removeBtn.className = 'market-chip-remove';
        removeBtn.setAttribute('data-role', 'chip-remove');
        removeBtn.setAttribute('data-symbol', symbol);
        removeBtn.setAttribute(
          'aria-label',
          langPrefix === 'zh' ? `移除 ${symbol}` : `Remove ${symbol}`
        );
        removeBtn.textContent = '×';
        removeBtn.tabIndex = 0;
        chip.appendChild(removeBtn);
      }
      container.appendChild(chip);
    });
  }

  function showChipSkeleton(container, count = 3) {
    if (!container) return;
    if (container.dataset.loading === 'true') return;
    container.dataset.loading = 'true';
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < count; i += 1) {
      const placeholder = document.createElement('span');
      placeholder.className = 'skeleton-chip';
      placeholder.setAttribute('aria-hidden', 'true');
      fragment.appendChild(placeholder);
    }
    container.appendChild(fragment);
  }

  function hideChipSkeleton(container) {
    if (!container) return;
    if (container.dataset.loading !== 'true') return;
    container.removeAttribute('data-loading');
    container.querySelectorAll('.skeleton-chip').forEach((node) => node.remove());
  }

  function attachChipHandler(container, options = {}) {
    if (!container) return;
    const removeHandler = typeof options.onRemove === 'function' ? options.onRemove : null;
    container.addEventListener('click', (event) => {
      const removeTarget = options.allowRemove && event.target.closest('[data-role="chip-remove"]');
      if (removeTarget) {
        event.stopPropagation();
        const symbol = removeTarget.dataset.symbol;
        if (symbol) {
          if (removeHandler) {
            removeHandler(symbol);
          } else if (options.watch) {
            if (searchInput) searchInput.value = symbol;
            loadData(symbol, { watchAction: 'remove' });
          }
        }
        return;
      }
      const chip = event.target.closest('[data-role="symbol-chip"]');
      if (!chip) return;
      const symbol = chip.dataset.symbol;
      if (!symbol) return;
      if (searchInput) searchInput.value = symbol;
      loadData(symbol);
    });
    container.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const removeTarget = options.allowRemove && event.target.closest('[data-role="chip-remove"]');
      if (removeTarget) {
        event.preventDefault();
        const symbol = removeTarget.dataset.symbol;
        if (symbol) {
          if (removeHandler) {
            removeHandler(symbol);
          } else if (options.watch) {
            if (searchInput) searchInput.value = symbol;
            loadData(symbol, { watchAction: 'remove' });
          }
        }
        return;
      }
      const chip = event.target.closest('[data-role="symbol-chip"]');
      if (!chip) return;
      event.preventDefault();
      const symbol = chip.dataset.symbol;
      if (!symbol) return;
      if (searchInput) searchInput.value = symbol;
      loadData(symbol);
    });
  }

  const hasTypeaheadUi = Boolean(searchInput && typeaheadPanel && typeaheadList);
  const TYPEAHEAD_LIMIT = 9;

  function normalizeSymbol(value) {
    return (value || '').toString().trim().toUpperCase();
  }

  function normalizeListType(value) {
    const text = (value || '').toString().trim().toLowerCase();
    if (text === 'gainers' || text === 'losers' || text === 'most_active') {
      return text;
    }
    return 'gainers';
  }

  function normalizeList(items) {
    if (!Array.isArray(items)) return [];
    const seen = new Set();
    const normalized = [];
    items.forEach((value) => {
      const symbol = normalizeSymbol(value);
      if (symbol && !seen.has(symbol)) {
        seen.add(symbol);
        normalized.push(symbol);
      }
    });
    return normalized;
  }

  function buildTypeaheadOptions(filterValue) {
    if (!hasTypeaheadUi) return [];
    const trimmed = normalizeSymbol(filterValue);
    const results = [];
    const seen = new Set();
    const buckets = [
      { list: watchPool, label: TEXT.typeaheadWatch, action: 'remove' },
      { list: recentPool, label: TEXT.typeaheadRecent, action: 'add' },
      { list: suggestionPool, label: TEXT.typeaheadTrending, action: 'add' },
    ];
    buckets.forEach((bucket) => {
      normalizeList(bucket.list).forEach((symbol) => {
        if (seen.has(symbol)) return;
        if (trimmed && !symbol.includes(trimmed)) return;
        results.push({
          symbol,
          sourceLabel: bucket.label,
          action: bucket.action,
        });
        seen.add(symbol);
      });
    });
    return results.slice(0, TYPEAHEAD_LIMIT);
  }

  function highlightTypeaheadOption(index) {
    if (!hasTypeaheadUi) return;
    if (!typeaheadList) return;
    const optionNodes = Array.from(typeaheadList.querySelectorAll('[data-role="typeahead-option"]'));
    optionNodes.forEach((node, idx) => {
      node.classList.toggle('is-active', idx === index);
    });
  }

  function setTypeaheadVisibility(visible) {
    if (!hasTypeaheadUi || !typeaheadPanel) return;
    if (visible) {
      typeaheadPanel.hidden = false;
      typeaheadPanel.setAttribute('aria-expanded', 'true');
    } else {
      typeaheadPanel.hidden = true;
      typeaheadPanel.setAttribute('aria-expanded', 'false');
      typeaheadActiveIndex = -1;
      highlightTypeaheadOption(-1);
    }
  }

  function selectTypeaheadSymbol(symbol, options = {}) {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return;
    if (searchInput) {
      searchInput.value = normalized;
      if (typeof searchInput.setSelectionRange === 'function') {
        const len = normalized.length;
        searchInput.setSelectionRange(len, len);
      }
      searchInput.focus();
    }
    setTypeaheadVisibility(false);
    if (options.watchAction) {
      loadData(normalized, { watchAction: options.watchAction });
    } else {
      loadData(normalized);
    }
  }

  function renderTypeahead(keyword = '') {
    if (!hasTypeaheadUi || !typeaheadPanel || !typeaheadList) return;
    const options = buildTypeaheadOptions(keyword);
    typeaheadOptions = options;
    typeaheadActiveIndex = -1;
    typeaheadList.innerHTML = '';
    const hasFilter = !!normalizeSymbol(keyword);
    if (!options.length) {
      if (!hasFilter) {
        setTypeaheadVisibility(false);
        return;
      }
      const empty = document.createElement('div');
      empty.className = 'typeahead-empty text-muted';
      empty.textContent = TEXT.typeaheadEmpty;
      typeaheadList.appendChild(empty);
      setTypeaheadVisibility(true);
      return;
    }
    options.forEach((option, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'typeahead-option';
      button.dataset.symbol = option.symbol;
      button.dataset.index = String(index);
      button.setAttribute('data-role', 'typeahead-option');

      const symbolSpan = document.createElement('span');
      symbolSpan.className = 'option-symbol';
      symbolSpan.textContent = option.symbol;

      const meta = document.createElement('div');
      meta.className = 'option-meta';

      const sourceSpan = document.createElement('span');
      sourceSpan.className = 'option-source';
      sourceSpan.textContent = option.sourceLabel;
      meta.appendChild(sourceSpan);

      if (option.action) {
        const watchWrap = document.createElement('span');
        watchWrap.className = 'typeahead-watch';
        const watchBtn = document.createElement('button');
        watchBtn.type = 'button';
        watchBtn.setAttribute('data-role', 'typeahead-watch');
        watchBtn.dataset.symbol = option.symbol;
        watchBtn.dataset.action = option.action;
        watchBtn.textContent = option.action === 'remove' ? TEXT.typeaheadRemove : TEXT.typeaheadAdd;
        watchWrap.appendChild(watchBtn);
        meta.appendChild(watchWrap);
      }

      button.appendChild(symbolSpan);
      button.appendChild(meta);
      typeaheadList.appendChild(button);
    });
    setTypeaheadVisibility(true);
  }

  function cancelHideTypeahead() {
    if (hideTypeaheadTimer) {
      window.clearTimeout(hideTypeaheadTimer);
      hideTypeaheadTimer = null;
    }
  }

  function scheduleHideTypeahead() {
    cancelHideTypeahead();
    hideTypeaheadTimer = window.setTimeout(() => {
      setTypeaheadVisibility(false);
    }, 160);
  }

  function syncTypeaheadPools(payload) {
    if (!hasTypeaheadUi || !payload) return;
    suggestionPool = normalizeList(payload.suggestions || []);
    recentPool = normalizeList(payload.recent_queries || []);
    watchPool = normalizeList(payload.watchlist || []);
    if (searchInput && document.activeElement === searchInput && !typeaheadPanel.hidden) {
      renderTypeahead(searchInput.value);
    }
  }

  function handleTypeaheadKeydown(event) {
    if (!hasTypeaheadUi || !searchInput) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      if (!typeaheadOptions.length) {
        renderTypeahead(searchInput.value);
      }
      if (!typeaheadOptions.length) return;
      event.preventDefault();
      const delta = event.key === 'ArrowDown' ? 1 : -1;
      const nextIndex = (typeaheadActiveIndex + delta + typeaheadOptions.length) % typeaheadOptions.length;
      highlightTypeaheadOption(nextIndex);
      setTypeaheadVisibility(true);
      typeaheadActiveIndex = nextIndex;
      return;
    }
    if (event.key === 'Enter' && typeaheadActiveIndex >= 0 && typeaheadOptions[typeaheadActiveIndex]) {
      event.preventDefault();
      selectTypeaheadSymbol(typeaheadOptions[typeaheadActiveIndex].symbol);
      return;
    }
    if (event.key === 'Escape' && typeaheadPanel && !typeaheadPanel.hidden) {
      setTypeaheadVisibility(false);
    }
  }

  function handleTypeaheadClick(event) {
    if (!hasTypeaheadUi) return;
    const watchButton = event.target.closest('[data-role="typeahead-watch"]');
    if (watchButton) {
      event.preventDefault();
      const symbol = watchButton.dataset.symbol;
      const action = watchButton.dataset.action;
      selectTypeaheadSymbol(symbol, { watchAction: action });
      return;
    }
    const option = event.target.closest('[data-role="typeahead-option"]');
    if (!option) return;
    event.preventDefault();
    const symbol = option.dataset.symbol;
    if (symbol) {
      selectTypeaheadSymbol(symbol);
    }
  }

  if (hasTypeaheadUi && searchInput && typeaheadPanel && typeaheadList) {
    searchInput.addEventListener('input', () => renderTypeahead(searchInput.value));
    searchInput.addEventListener('focus', () => {
      cancelHideTypeahead();
      renderTypeahead(searchInput.value);
    });
    searchInput.addEventListener('blur', scheduleHideTypeahead);
    searchInput.addEventListener('keydown', handleTypeaheadKeydown);
    typeaheadPanel.addEventListener('mouseenter', cancelHideTypeahead);
    typeaheadPanel.addEventListener('mouseleave', scheduleHideTypeahead);
    typeaheadPanel.addEventListener('pointerdown', cancelHideTypeahead);
    typeaheadList.addEventListener('mousedown', (event) => event.preventDefault());
    typeaheadList.addEventListener('click', handleTypeaheadClick);
  }

  async function loadData(query = '', options = {}) {
    const normalizedQuery = normalizeSymbol(query);
    const activeListType = normalizeListType(options.listType || currentListType);
    lastRequest = { query: normalizedQuery || '', options: { ...options, listType: activeListType } };
    const requestPayload = {
      timeframe: currentTimeframe,
      list: activeListType,
    };
    if (normalizedQuery) {
      requestPayload.query = normalizedQuery;
    }
    if (options.watchAction && normalizedQuery) {
      requestPayload.watch = options.watchAction;
    }
    if (options.recentAction) {
      requestPayload.recent = options.recentAction;
      if (options.recentTarget) {
        const normalizedTarget = normalizeSymbol(options.recentTarget);
        if (normalizedTarget) {
          requestPayload.recent_target = normalizedTarget;
        }
      }
    }
    if (options.limit) {
      requestPayload.limit = options.limit;
    }

    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (statusSection) {
      statusSection.classList.remove('is-retrying');
    }
    setStatus(`${TEXT.loading} ${TEXT.timeframes[currentTimeframe] || currentTimeframe} ${TEXT.dataSuffix}`);
    setListLoading(listContainer);
    showChipSkeleton(recentChips, 3);
    showChipSkeleton(watchlistChips, 4);

    const shouldPost =
      Boolean(options.watchAction) || Boolean(options.recentAction) || Boolean(normalizedQuery);

    try {
      const endpointBase = apiUrl || '/api/market/';
      const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
      let response;
      if (shouldPost) {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken(),
            'X-Requested-With': 'XMLHttpRequest',
          },
          credentials: 'same-origin',
          body: JSON.stringify(requestPayload),
        });
      } else {
        const params = new URLSearchParams({ timeframe: currentTimeframe, list: activeListType });
        if (options.limit) {
          params.set('limit', options.limit);
        }
        response = await fetch(`${endpoint}?${params.toString()}`, {
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
          credentials: 'same-origin',
        });
      }
      const payload = await response.json();
      if (response.status === 429 || payload.rate_limited) {
        setRetryingState(payload.retry_after_seconds);
        return;
      }
      if (!response.ok) {
        throw new Error(payload.error || TEXT.genericError);
      }
      const responseListType = normalizeListType(payload.list_type || activeListType);
      if (payload.list_type && responseListType !== currentListType) {
        setActiveListType(responseListType);
      }
      let items = Array.isArray(payload.items) ? payload.items : [];
      if (!items.length) {
        if (responseListType === 'losers') {
          items = payload.losers || [];
        } else if (responseListType === 'most_active') {
          items = payload.most_actives || [];
        } else {
          items = payload.gainers || [];
        }
      }
      renderList(listContainer, items, payload.timeframe, responseListType);
      if (!items.length) {
        renderEmpty(listContainer, TEXT.emptySymbol);
      }
      updateSuggestionList(payload.suggestions || []);
      renderChipGroup(recentChips, payload.recent_queries || [], {
        emptyText: TEXT.emptyChips,
        countTarget: recentCount,
        allowRemove: true,
      });
      renderChipGroup(watchlistChips, payload.watchlist || [], {
        emptyText: TEXT.emptyWatchlist,
        watch: true,
        countTarget: watchCount,
        allowRemove: true,
      });
      syncTypeaheadPools(payload);
      const tfKey = payload.timeframe && payload.timeframe.key;
      const tfLabel = payload.timeframe && (langPrefix === 'zh' ? payload.timeframe.label : payload.timeframe.label_en);
      let statusMessage = `${TEXT.timeframes[tfKey] || tfLabel || ''} ${TEXT.updated}（${
        payload.generated_at || TEXT.justNow
      }）`;
      const normalizedSymbol = normalizedQuery || '';
      const actionSymbol = (options.recentTarget || '').toUpperCase();
      if (options.watchAction === 'add' && normalizedSymbol) {
        statusMessage += ` · ${TEXT.watchAdded(normalizedSymbol)}`;
      } else if (options.watchAction === 'remove' && normalizedSymbol) {
        statusMessage += ` · ${TEXT.watchRemoved(normalizedSymbol)}`;
      }
      if (options.recentAction === 'clear') {
        statusMessage += ` · ${TEXT.historyCleared || ''}`;
      } else if (options.recentAction === 'delete' && actionSymbol) {
        const deletedText =
          typeof TEXT.historyDeleted === 'function' ? TEXT.historyDeleted(actionSymbol) : TEXT.historyDeleted;
        statusMessage += ` · ${deletedText}`;
      }
      setStatus(statusMessage);
      setSource(payload.data_source);
    } catch (error) {
      renderError(listContainer, error && error.message);
      setStatus(TEXT.statusError);
      setSource('unknown');
      hideChipSkeleton(recentChips);
      hideChipSkeleton(watchlistChips);
    }
  }

  function renderList(container, items, timeframe, listType) {
    if (!container) return;
    clearListState(container);
    if (!items.length) {
      renderEmpty(container, TEXT.emptyList);
      return;
    }
    const isMostActive = listType === 'most_active';
    const invert = listType === 'losers';
    items.forEach((item) => {
      if (!cardTemplate) return;
      const fragment = cardTemplate.content.cloneNode(true);
      const symbolLink = fragment.querySelector('.market-symbol');
      const priceEl = fragment.querySelector('[data-role="price"]');
      const primaryLabelEl = fragment.querySelector('[data-role="primary-label"]');
      const primaryEl = fragment.querySelector('[data-role="primary-change"]');
      const dayEl = fragment.querySelector('[data-role="day-change"]');
      const canvas = fragment.querySelector('canvas');
      const windowLabel = fragment.querySelector('[data-role="window-label"]');
      const updatedEl = fragment.querySelector('[data-role="updated"]');
      const backtestLink = fragment.querySelector('[data-role="backtest-link"]');

      if (symbolLink) {
        symbolLink.textContent = item.symbol || '';
        symbolLink.href = item.url || `https://www.tradingview.com/symbols/${item.symbol || ''}/`;
      }
      if (priceEl) {
        priceEl.textContent = typeof item.price === 'number' ? item.price.toFixed(2) : '--';
      }
      if (primaryLabelEl) {
        if (isMostActive) {
          primaryLabelEl.textContent = item.period_label || item.period_label_en || TEXT.volumeLabel;
        } else {
          const fallback = TEXT.timeframes[currentTimeframe] || (langPrefix === 'zh' ? '近1月' : '1M');
          const tfLabel = timeframe ? (langPrefix === 'zh' ? timeframe.label : timeframe.label_en) : '';
          const itemLabel = langPrefix === 'zh' ? item.period_label : item.period_label_en;
          primaryLabelEl.textContent = itemLabel || tfLabel || fallback;
        }
      }
      if (primaryEl) {
        primaryEl.classList.remove('is-neutral');
        if (isMostActive) {
          primaryEl.textContent = formatCompactNumber(item.volume);
          primaryEl.classList.add('is-neutral');
        } else {
          primaryEl.textContent = formatChange(item.change_pct_period);
          applyChangeState(primaryEl, item.change_pct_period, invert);
        }
      }
      if (dayEl) {
        dayEl.textContent = formatChange(item.change_pct_day);
        applyChangeState(dayEl, item.change_pct_day, invert, true);
      }
      if (windowLabel) {
        if (isMostActive) {
          windowLabel.textContent = item.period_label || item.period_label_en || TEXT.volumeLabel;
        } else {
          const fallback = TEXT.timeframes[currentTimeframe] || (langPrefix === 'zh' ? '近1月' : '1M');
          const tfWindow = timeframe ? (langPrefix === 'zh' ? timeframe.label : timeframe.label_en) : '';
          const itemWindow = langPrefix === 'zh' ? item.period_label : item.period_label_en;
          windowLabel.textContent = itemWindow || tfWindow || fallback;
        }
      }
      if (updatedEl) {
        const timestamps = Array.isArray(item.timestamps) ? item.timestamps : [];
        const stamp = timestamps.length ? timestamps[timestamps.length - 1] : '';
        updatedEl.textContent = stamp ? `${TEXT.updatedLabel} ${stamp}` : '';
      }
      if (backtestLink) {
        const symbol = item.symbol || '';
        backtestLink.href = `${backtestBase}?ticker=${encodeURIComponent(symbol)}`;
      }
      if (canvas) {
        drawSparkline(canvas, item.series || [], invert);
      }
      const card = fragment.querySelector('.market-card');
      if (card) {
        card.dataset.symbol = normalizeSymbol(item.symbol || '');
        card.dataset.invert = invert ? '1' : '0';
        card.dataset.listType = listType || '';
      }
      container.appendChild(fragment);
    });
  }

  function switchList(type) {
    const nextType = normalizeListType(type);
    if (nextType === currentListType) return;
    setActiveListType(nextType);
    loadData('', { listType: nextType });
  }

  function handleCardClick(event) {
    if (event.target.closest('a')) {
      return;
    }
    const card = event.target.closest('.market-card');
    if (!card) return;
    const symbol = card.dataset.symbol;
    if (!symbol) return;
    openDetailModal(symbol);
  }

  attachChipHandler(recentChips, { allowRemove: true, onRemove: (symbol) => requestRecentAction('delete', symbol) });
  attachChipHandler(watchlistChips, { allowRemove: true, watch: true });

  if (rankTabs.length) {
    const activeTab = rankTabs.find((tab) => tab.classList.contains('is-active')) || rankTabs[0];
    if (activeTab) {
      setActiveListType(normalizeListType(activeTab.dataset.list));
    }
    rankTabs.forEach((tab) => {
      tab.addEventListener('click', () => switchList(tab.dataset.list));
    });
  }

  if (listContainer) {
    listContainer.addEventListener('click', handleCardClick);
  }

  if (detailTimeframes.length) {
    const activeRange = detailTimeframes.find((btn) => btn.classList.contains('is-active'));
    if (activeRange && activeRange.dataset.range) {
      detailRange = activeRange.dataset.range;
    }
    detailTimeframes.forEach((btn) => {
      btn.addEventListener('click', () => {
        const rangeKey = btn.dataset.range || '1d';
        if (rangeKey === detailRange) return;
        setDetailRange(rangeKey);
        if (detailSymbol) {
          loadDetailData(detailSymbol, rangeKey);
        }
      });
    });
  }

  if (detailOverlaySelect) {
    detailOverlaySelect.addEventListener('change', () => {
      if (!ensureDetailChart() || !detailManager) return;
      detailManager.setOverlay(detailOverlaySelect.value);
    });
  }

  if (detailIndicatorSelect) {
    detailIndicatorSelect.addEventListener('change', () => {
      if (!ensureDetailChart() || !detailManager) return;
      detailManager.setIndicator(detailIndicatorSelect.value);
    });
  }

  if (detailDrawButtons.length) {
    detailDrawButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.draw;
        if (!mode) return;
        if (!ensureDetailChart() || !detailManager) return;
        if (mode === 'clear') {
          detailManager.clearDrawings();
          return;
        }
        if (btn.classList.contains('is-active')) {
          btn.classList.remove('is-active');
          detailManager.setDrawMode('none');
          return;
        }
        detailDrawButtons.forEach((other) => {
          other.classList.toggle('is-active', other === btn);
        });
        detailManager.setDrawMode(mode);
      });
    });
  }

  if (detailModalEl) {
    detailModalEl.addEventListener('shown.bs.modal', () => resizeDetailChart());
    detailModalEl.addEventListener('hidden.bs.modal', () => {
      detailSymbol = '';
      setDetailStatus('');
    });
  }

  loadData();
  connectMarketSocket();
})();
