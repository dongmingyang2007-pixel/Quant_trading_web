(() => {
  const apiMeta = document.querySelector('meta[name="market-api"]');
  const apiUrl = apiMeta ? apiMeta.getAttribute('content') : window.MARKET_API_URL || '/market/api/';
  const assetsMeta = document.querySelector('meta[name="market-assets"]');
  const assetsUrl = assetsMeta ? assetsMeta.getAttribute('content') : '/api/market/assets/';
  const langMeta = document.querySelector('meta[name="market-lang"]');
  const backtestMeta = document.querySelector('meta[name="backtest-base"]');
  const backtestBase = backtestMeta ? backtestMeta.getAttribute('content') : '/backtest/';
  const docLang = document.documentElement.getAttribute('lang');
  const langPrefix = ((langMeta && langMeta.getAttribute('content')) || docLang || navigator.language || 'zh')
    .toLowerCase()
    .slice(0, 2);

  const listContainer = document.querySelector('[data-role="ranking-list"]');
  const rankingChangeHeader = document.querySelector('[data-role="ranking-change-header"]');
  const rankingsSection = document.querySelector('[data-role="rankings-section"]');
  const allStocksSection = document.querySelector('[data-role="all-stocks"]');
  const allStocksLetters = document.querySelector('[data-role="all-stocks-letters"]');
  const allStocksBody = document.querySelector('[data-role="all-stocks-body"]');
  const allStocksPagination = document.querySelector('[data-role="all-stocks-pagination"]');
  const allStocksCount = document.querySelector('[data-role="all-stocks-count"]');
  const allStocksBack = document.querySelector('[data-role="all-stocks-back"]');
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
  const detailRoot = document.querySelector('[data-role="market-detail"]') || document;
  const detailTitle = detailRoot.querySelector('[data-role="detail-title"]');
  const detailSubtitle = detailRoot.querySelector('[data-role="detail-subtitle"]');
  const detailSource = detailRoot.querySelector('[data-role="detail-source"]');
  const detailUpdated = detailRoot.querySelector('[data-role="detail-updated"]');
  const detailStatus = detailRoot.querySelector('[data-role="detail-status"]');
  const detailChartEl = detailRoot.querySelector('#market-detail-chart');
  const detailIndicatorEl = detailRoot.querySelector('#market-detail-indicator');
  const detailOverlaySelect = detailRoot.querySelector('#detail-overlay-select');
  const detailIndicatorSelect = detailRoot.querySelector('#detail-indicator-select');
  const detailDrawButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('.draw-btn'));
  const detailTimeframes = Array.prototype.slice.call(detailRoot.querySelectorAll('.detail-timeframe'));
  const detailTimeframeTrigger = detailRoot.querySelector('[data-role="detail-timeframe-trigger"]');
  const detailTimeframeMenu = detailRoot.querySelector('[data-role="detail-timeframe-menu"]');
  const detailTimeframeCurrent = detailRoot.querySelector('[data-role="detail-timeframe-current"]');
  const profileSummary = document.querySelector('[data-role="profile-summary"]');
  const profileMetrics = document.querySelector('[data-role="profile-metrics"]');
  const aiSummary = document.querySelector('[data-role="ai-summary"]');
  const newsList = document.querySelector('[data-role="news-list"]');
  const viewList = document.querySelector('[data-view="list"]');
  const viewDetail = document.querySelector('[data-view="detail"]');
  const viewChart = document.querySelector('[data-view="chart"]');
  const viewBackButtons = Array.prototype.slice.call(document.querySelectorAll('[data-view-back]'));
  const viewChartButton = document.querySelector('[data-view-chart]');
  const detailSymbolEl = document.querySelector('[data-role="detail-symbol"]');
  const detailNameEl = document.querySelector('[data-role="detail-name"]');
  const detailPriceEl = document.querySelector('[data-role="detail-price"]');
  const detailChangeEl = document.querySelector('[data-role="detail-change"]');
  const detailMetaEl = document.querySelector('[data-role="detail-meta"]');
  const marketSocketUrl = (() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}/ws/market/`;
  })();

  const initialBtn = document.querySelector('.market-timeframe.is-active');
  let currentTimeframe = (initialBtn && initialBtn.getAttribute('data-timeframe')) || '1mo';
  let currentListType = 'gainers';
  let lastRankingType = 'gainers';
  let allStocksLetter = 'ALL';
  let allStocksPage = 1;
  let allStocksSize = 50;
  let allStocksQuery = '';
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
  let detailManager = null;
  let detailSymbol = '';
  let detailRange = '1d';
  let detailBarIntervalSec = null;
  let detailRetryTimer = null;
  let currentView = 'list';
  let autoRefreshTimer = null;
  let isListLoading = false;
  let liveWaitTimer = null;
  let lastLiveUpdateAt = 0;
  let liveQuoteTimer = null;
  const AUTO_REFRESH_MS = 60 * 1000;
  const DEFAULT_LIST_LIMIT = 200;
  const LIVE_WAIT_MS = 10000;
  const LIVE_QUOTE_POLL_MS = 15000;
  let detailCache = {
    symbol: '',
    range: '',
    bars: [],
    payload: null,
  };

  const TEXT = langPrefix === 'zh'
    ? {
        timeframes: { '1d': '近1日', '5d': '近5日', '1mo': '近1月', '6mo': '近6月' },
        loading: '正在加载',
        dataSuffix: '数据…',
        updated: '数据已更新',
        closedFallback: '闭市，已展示上一交易日榜单',
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
        detailPrompt: '请选择标的后展示图表与指标。',
        detailError: '加载失败',
        detailFallback: (requested, used) => `无 ${requested} 数据，已显示 ${used} 历史`,
        volumeLabel: '成交量',
        detailLive: '实时流更新中…',
        detailLiveWaiting: '实时流暂未收到更新，请确认实时引擎已启动。',
        allStocksLoading: '正在加载全部股票…',
        allStocksEmpty: '暂无股票数据',
        allStocksCount: (total, page, totalPages) => `共 ${total} 只 · 第 ${page}/${totalPages} 页`,
        pagePrev: '上一页',
        pageNext: '下一页',
        profileEmpty: '暂无公司信息。',
        newsEmpty: '暂无相关新闻。',
        aiPlaceholder: 'AI 摘要准备中。',
        profileLabels: {
          sector: '板块',
          industry: '行业',
          market_cap: '市值',
          pe: 'PE',
          pb: 'PB',
          beta: 'Beta',
          employees: '员工数',
          dividend_yield: '股息率',
        },
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
        closedFallback: 'Market closed, showing previous session movers',
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
        detailPrompt: 'Select a symbol to render chart and indicators.',
        detailError: 'Failed to load',
        detailFallback: (requested, used) => `No ${requested} data, showing ${used} history`,
        volumeLabel: 'Volume',
        detailLive: 'Live stream updating…',
        detailLiveWaiting: 'No live ticks yet. Check the realtime engine.',
        allStocksLoading: 'Loading all stocks…',
        allStocksEmpty: 'No stocks found.',
        allStocksCount: (total, page, totalPages) => `${total} symbols · Page ${page}/${totalPages}`,
        pagePrev: 'Prev',
        pageNext: 'Next',
        profileEmpty: 'No company profile available.',
        newsEmpty: 'No related headlines yet.',
        aiPlaceholder: 'AI summary is preparing.',
        profileLabels: {
          sector: 'Sector',
          industry: 'Industry',
          market_cap: 'Market Cap',
          pe: 'P/E',
          pb: 'P/B',
          beta: 'Beta',
          employees: 'Employees',
          dividend_yield: 'Dividend Yield',
        },
        sourcePrefix: 'Data source: ',
        sourceLabels: {
          alpaca: 'Alpaca',
          yfinance: 'Yahoo Finance',
          cache: 'Cache',
          unknown: 'Unknown',
        },
      };

  function getIndicatorLib() {
    if (window.technicalindicators) return window.technicalindicators;
    if (window.SMA && window.EMA && window.RSI) return window;
    return null;
  }

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
      this.priceLine = null;
      this.lastLivePrice = null;
      this.liveMode = null;
      this.liveIntervalSec = null;
      this.liveBar = null;
      this.liveTickBase = null;
      this.liveTickIndex = 0;
      this.liveMaxBars = 600;
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
        timeScale: { borderColor: 'rgba(148, 163, 184, 0.4)', timeVisible: true, secondsVisible: false },
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
      this.clearLiveMode();
      this.lastLivePrice = null;
      this.ohlcData = this._sanitizeBars(bars);
      if (this.candleSeries) {
        this.candleSeries.setData(this.ohlcData);
      }
      this.updateOverlay();
      this.updateIndicator();
      if (this.chart) {
        this.chart.timeScale().fitContent();
      }
      if (this.ohlcData.length) {
        const last = this.ohlcData[this.ohlcData.length - 1];
        if (last && typeof last.close === 'number') {
          this.updatePriceLine(last.close);
        }
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

    updatePriceLine(price) {
      if (!this.candleSeries || typeof price !== 'number' || Number.isNaN(price)) return;
      const color = this.lastLivePrice !== null && price < this.lastLivePrice ? '#ef4444' : '#16a34a';
      if (!this.priceLine) {
        this.priceLine = this.candleSeries.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle: 2,
          axisLabelVisible: true,
          title: 'Last',
        });
      } else {
        this.priceLine.applyOptions({ price, color });
      }
      this.lastLivePrice = price;
    }

    setLiveMode(mode, intervalSec) {
      if (!mode) {
        this.clearLiveMode();
        return;
      }
      this.liveMode = mode;
      this.liveIntervalSec = Math.max(1, Number.parseInt(intervalSec, 10) || 1);
      this.liveBar = null;
      this.liveTickBase = null;
      this.liveTickIndex = 0;
      this.ohlcData = [];
      if (this.candleSeries) {
        this.candleSeries.setData([]);
      }
      if (this.indicatorContainer) {
        this.indicatorContainer.hidden = true;
      }
    }

    clearLiveMode() {
      this.liveMode = null;
      this.liveIntervalSec = null;
      this.liveBar = null;
      this.liveTickBase = null;
      this.liveTickIndex = 0;
    }

    updateLivePrice(price, ts) {
      if (typeof price !== 'number' || Number.isNaN(price)) return;
      const timestamp = Number.isFinite(ts) ? ts : Date.now() / 1000;
      this.updatePriceLine(price);
      if (!this.liveMode || !this.candleSeries) return;

      if (this.liveMode === 'tick') {
        if (this.liveTickBase === null) {
          this.liveTickBase = Math.floor(timestamp);
        }
        const time = this.liveTickBase + this.liveTickIndex;
        this.liveTickIndex += 1;
        const bar = { time, open: price, high: price, low: price, close: price };
        this._pushLiveBar(bar);
        return;
      }

      const interval = this.liveIntervalSec || 1;
      const bucket = Math.floor(timestamp / interval) * interval;
      if (!this.liveBar || this.liveBar.time !== bucket) {
        const bar = { time: bucket, open: price, high: price, low: price, close: price };
        this.liveBar = bar;
        this._pushLiveBar(bar);
      } else {
        this.liveBar.high = Math.max(this.liveBar.high, price);
        this.liveBar.low = Math.min(this.liveBar.low, price);
        this.liveBar.close = price;
        this.candleSeries.update({ ...this.liveBar });
      }
    }

    updateCurrentBar(price, ts, intervalSec) {
      if (this.liveMode || !this.candleSeries) return;
      if (!Array.isArray(this.ohlcData) || !this.ohlcData.length) return;
      if (typeof price !== 'number' || Number.isNaN(price)) return;
      const interval = Math.max(1, Number.parseInt(intervalSec, 10) || 0);
      if (!interval) return;
      const timestamp = Number.isFinite(ts) ? ts : Date.now() / 1000;
      const bucket = Math.floor(timestamp / interval) * interval;
      const lastIndex = this.ohlcData.length - 1;
      const last = this.ohlcData[lastIndex];
      if (!last || typeof last.time !== 'number') return;
      if (bucket < last.time) return;
      if (bucket === last.time) {
        const updated = { ...last };
        const nextOpen = typeof updated.open === 'number' ? updated.open : price;
        const nextHigh = typeof updated.high === 'number' ? Math.max(updated.high, price) : price;
        const nextLow = typeof updated.low === 'number' ? Math.min(updated.low, price) : price;
        updated.open = nextOpen;
        updated.high = nextHigh;
        updated.low = nextLow;
        updated.close = price;
        this.ohlcData[lastIndex] = updated;
        this.candleSeries.update(updated);
        return;
      }
      const bar = { time: bucket, open: price, high: price, low: price, close: price };
      this.ohlcData.push(bar);
      this.candleSeries.update(bar);
    }

    _pushLiveBar(bar) {
      this.ohlcData.push(bar);
      if (this.ohlcData.length > this.liveMaxBars) {
        this.ohlcData = this.ohlcData.slice(-this.liveMaxBars);
        if (this.candleSeries) {
          this.candleSeries.setData(this.ohlcData);
          return;
        }
      }
      if (this.candleSeries) {
        this.candleSeries.update(bar);
      }
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
      const indicatorLib = getIndicatorLib();
      if (!indicatorLib || !indicatorLib.SMA) return;
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
      const indicatorLib = getIndicatorLib();
      if (!indicatorLib || !indicatorLib.RSI) return;
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
        if (startIndex < 0) {
          return;
        }
        const histogram = this.indicatorChart.addHistogramSeries({ color: '#94a3b8' });
        const macdLine = this.indicatorChart.addLineSeries({ color: '#0ea5e9', lineWidth: 2 });
        const signalLine = this.indicatorChart.addLineSeries({ color: '#f97316', lineWidth: 2 });
        const histogramData = [];
        const macdData = [];
        const signalData = [];
        values.forEach((item, index) => {
          const bar = this.ohlcData[startIndex + index];
          if (!bar || !Number.isFinite(bar.time)) return;
          const histValue =
            typeof item.histogram === 'number' ? item.histogram : Number.parseFloat(item.histogram);
          if (Number.isFinite(histValue)) {
            histogramData.push({
              time: bar.time,
              value: histValue,
              color: histValue >= 0 ? 'rgba(34, 197, 94, 0.65)' : 'rgba(239, 68, 68, 0.65)',
            });
          }
          const macdValue = typeof item.MACD === 'number' ? item.MACD : Number.parseFloat(item.MACD);
          if (Number.isFinite(macdValue)) {
            macdData.push({ time: bar.time, value: macdValue });
          }
          const signalValue = typeof item.signal === 'number' ? item.signal : Number.parseFloat(item.signal);
          if (Number.isFinite(signalValue)) {
            signalData.push({ time: bar.time, value: signalValue });
          }
        });
        histogram.setData(histogramData);
        macdLine.setData(macdData);
        signalLine.setData(signalData);
        this.indicatorSeries.push(histogram, macdLine, signalLine);
      }
      this.indicatorChart.timeScale().fitContent();
    }

    _getCloses() {
      return this.ohlcData.map((bar) => bar.close).filter((val) => Number.isFinite(val));
    }

    _mapSeries(values, period) {
      const offset = Math.max(0, period - 1);
      const points = [];
      values.forEach((value, index) => {
        const bar = this.ohlcData[index + offset];
        if (!bar || !Number.isFinite(bar.time) || !Number.isFinite(value)) return;
        points.push({ time: bar.time, value });
      });
      return points;
    }

    _mapBand(values, period, key) {
      const offset = Math.max(0, period - 1);
      const points = [];
      values.forEach((value, index) => {
        const bar = this.ohlcData[index + offset];
        const bandValue = value ? value[key] : null;
        if (!bar || !Number.isFinite(bar.time) || !Number.isFinite(bandValue)) return;
        points.push({ time: bar.time, value: bandValue });
      });
      return points;
    }

    _sanitizeBars(bars) {
      if (!Array.isArray(bars)) return [];
      const cleaned = [];
      bars.forEach((bar) => {
        if (!bar) return;
        const time = Number.isFinite(bar.time) ? Math.floor(bar.time) : Number.parseInt(bar.time, 10);
        const open = Number.isFinite(bar.open) ? bar.open : Number.parseFloat(bar.open);
        const high = Number.isFinite(bar.high) ? bar.high : Number.parseFloat(bar.high);
        const low = Number.isFinite(bar.low) ? bar.low : Number.parseFloat(bar.low);
        const close = Number.isFinite(bar.close) ? bar.close : Number.parseFloat(bar.close);
        if (!Number.isFinite(time) || !Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) {
          return;
        }
        cleaned.push({ time, open, high, low, close });
      });
      return cleaned;
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
    const isAllMode = currentListType === 'all';
    const options = {
      recentAction: action,
      listType: isAllMode ? 'gainers' : currentListType,
      skipListRender: isAllMode,
      keepListType: isAllMode,
      openDetail: !isAllMode,
    };
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
      if (currentListType !== 'all') {
        if (searchInput) {
          searchInput.value = '';
        }
        loadData();
      }
    });
  });

  if (searchForm) {
    searchForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const value = (searchInput && searchInput.value.trim()) || '';
      if (currentListType === 'all') {
        loadAllStocks({ query: value, page: 1 });
      } else {
        loadData(value);
      }
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
      const isAllMode = currentListType === 'all';
      loadData(symbol, {
        watchAction: 'add',
        listType: isAllMode ? 'gainers' : currentListType,
        skipListRender: isAllMode,
        keepListType: isAllMode,
        openDetail: !isAllMode,
      });
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
      all: rankDesc.dataset.descAll,
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
    if (rankingsSection && allStocksSection) {
      const showAll = type === 'all';
      rankingsSection.hidden = showAll;
      allStocksSection.hidden = !showAll;
    }
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
    const serverTs =
      typeof update.server_ts === 'number' ? update.server_ts : Number.parseFloat(update.server_ts);
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
    const tableRows = document.querySelectorAll(`tr[data-symbol="${symbol}"]`);
    tableRows.forEach((row) => {
      const cells = row.querySelectorAll('td');
      if (cells.length < 5) return;
      const priceCell = cells[3];
      const changeCell = cells[4];
      if (priceCell && Number.isFinite(price)) {
        priceCell.textContent = price.toFixed(2);
      }
      if (!changeCell || !Number.isFinite(changePct)) return;
      const isAllStocksRow = Boolean(row.closest('[data-role="all-stocks-body"]'));
      const isRankingRow = Boolean(row.closest('[data-role="ranking-list"]'));
      if (isAllStocksRow) {
        changeCell.textContent = formatChange(changePct);
        applyChangeState(changeCell, changePct);
        return;
      }
      if (isRankingRow && currentListType !== 'most_active' && currentTimeframe === '1d') {
        changeCell.textContent = formatChange(changePct);
        applyChangeState(changeCell, changePct, currentListType === 'losers');
      }
    });
    if (detailSymbol && detailSymbol === symbol) {
      if (detailPriceEl && Number.isFinite(price)) {
        detailPriceEl.textContent = price.toFixed(2);
      }
      if (detailChangeEl && Number.isFinite(changePct)) {
        detailChangeEl.textContent = formatChange(changePct);
        detailChangeEl.classList.remove('is-up', 'is-down');
        applyChangeState(detailChangeEl, changePct);
      }
      if (detailManager && Number.isFinite(price)) {
        detailManager.updateLivePrice(price, serverTs);
        if (detailBarIntervalSec) {
          detailManager.updateCurrentBar(price, serverTs, detailBarIntervalSec);
        }
      }
      lastLiveUpdateAt = Date.now();
      if (liveWaitTimer) {
        clearTimeout(liveWaitTimer);
        liveWaitTimer = null;
      }
      if (resolveLiveRange(detailRange)) {
        setDetailStatus(TEXT.detailLive);
      }
    }
  }

  function clearLiveWait() {
    if (liveWaitTimer) {
      clearTimeout(liveWaitTimer);
      liveWaitTimer = null;
    }
  }

  function scheduleLiveWait() {
    clearLiveWait();
    liveWaitTimer = setTimeout(() => {
      if (!resolveLiveRange(detailRange)) return;
      if (lastLiveUpdateAt) return;
      setDetailStatus(TEXT.detailLiveWaiting);
    }, LIVE_WAIT_MS);
  }

  function stopLiveQuotePolling() {
    if (liveQuoteTimer) {
      clearInterval(liveQuoteTimer);
      liveQuoteTimer = null;
    }
  }

  async function fetchLiveQuote(symbol) {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return;
    const endpointBase = apiUrl || '/api/market/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({ quote: '1', symbol: normalized });
    try {
      const response = await fetch(`${endpoint}?${params.toString()}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      const payload = await response.json();
      if (!response.ok) return;
      applyLiveUpdate(payload);
    } catch (error) {
      return;
    }
  }

  function startLiveQuotePolling(symbol) {
    stopLiveQuotePolling();
    fetchLiveQuote(symbol);
    liveQuoteTimer = setInterval(() => fetchLiveQuote(symbol), LIVE_QUOTE_POLL_MS);
  }

  function requestLiveSymbol(symbol) {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return;
    const endpointBase = apiUrl || '/api/market/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({ subscribe: normalized, subscribe_only: '1' });
    fetch(`${endpoint}?${params.toString()}`, {
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      credentials: 'same-origin',
    }).catch(() => {});
  }

  function requestLiveSymbols(symbols) {
    if (!Array.isArray(symbols) || !symbols.length) return;
    const cleaned = [];
    const seen = new Set();
    symbols.forEach((raw) => {
      const sym = normalizeSymbol(raw);
      if (!sym || seen.has(sym)) return;
      seen.add(sym);
      cleaned.push(sym);
    });
    if (!cleaned.length) return;
    const endpointBase = apiUrl || '/api/market/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({
      subscribe_symbols: cleaned.join(','),
      subscribe_only: '1',
    });
    fetch(`${endpoint}?${params.toString()}`, {
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      credentials: 'same-origin',
    }).catch(() => {});
  }

  function setDetailStatus(message, isError) {
    if (!detailStatus) return;
    detailStatus.textContent = message || '';
    detailStatus.hidden = !message;
    detailStatus.classList.toggle('is-error', Boolean(isError));
  }

  function setView(mode) {
    currentView = mode || 'list';
    if (viewList) viewList.classList.toggle('is-active', currentView === 'list');
    if (viewDetail) viewDetail.classList.toggle('is-active', currentView === 'detail');
    if (viewChart) viewChart.classList.toggle('is-active', currentView === 'chart');
    if (currentView === 'chart') {
      resizeDetailChart();
    }
    if (currentView === 'list') {
      scheduleAutoRefresh();
    }
    if (currentView !== 'chart') {
      clearLiveWait();
      stopLiveQuotePolling();
    }
  }

  function updateDetailHero(symbol, payload, bars) {
    if (detailSymbolEl) {
      detailSymbolEl.textContent = symbol || '--';
    }
    if (detailNameEl) {
      const profile = payload && payload.profile;
      const displayName = (profile && (profile.name || profile.shortName)) || symbol || '--';
      detailNameEl.textContent = displayName;
    }
    const hasBars = Array.isArray(bars) && bars.length > 0;
    if (detailPriceEl) {
      detailPriceEl.textContent = hasBars ? bars[bars.length - 1].close.toFixed(2) : '--';
    }
    if (detailChangeEl) {
      detailChangeEl.classList.remove('is-up', 'is-down');
      if (hasBars && bars.length > 1) {
        const latest = bars[bars.length - 1].close;
        const prev = bars[bars.length - 2].close;
        const changePct = prev ? ((latest / prev) - 1) * 100 : null;
        detailChangeEl.textContent = typeof changePct === 'number' ? formatChange(changePct) : '--';
        applyChangeState(detailChangeEl, changePct, false);
      } else {
        detailChangeEl.textContent = '--';
      }
    }
    if (detailMetaEl) {
      detailMetaEl.textContent = payload && payload.generated_at ? `${TEXT.updatedLabel} ${payload.generated_at}` : '';
    }
  }

  function hasCachedDetail(symbol, rangeKey) {
    return (
      detailCache &&
      detailCache.symbol === symbol &&
      detailCache.range === rangeKey &&
      Array.isArray(detailCache.bars) &&
      detailCache.bars.length > 0
    );
  }

  function storeDetailCache(symbol, rangeKey, bars, payload) {
    detailCache = {
      symbol,
      range: rangeKey,
      bars: Array.isArray(bars) ? bars : [],
      payload: payload && typeof payload === 'object' ? payload : null,
    };
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

  function renderProfilePanel(profile) {
    if (!profileSummary || !profileMetrics) return;
    const meta = profile && typeof profile === 'object' ? profile : {};
    const rawSummary = meta.summary || meta.name || '';
    const summary =
      rawSummary && rawSummary.length > 180 ? `${rawSummary.slice(0, 180)}…` : rawSummary;
    profileSummary.textContent = summary || TEXT.profileEmpty;
    const labels = TEXT.profileLabels || {};
    const formatValue = (key, value) => {
      if (typeof value === 'number') {
        if (key === 'employees') {
          return value.toLocaleString();
        }
        return value.toFixed(2);
      }
      return value;
    };
    const metrics = [
      ['sector', meta.sector],
      ['industry', meta.industry],
      ['market_cap', meta.market_cap],
      ['pe', meta.pe],
      ['pb', meta.pb],
      ['beta', meta.beta],
      ['employees', meta.employees],
      ['dividend_yield', meta.dividend_yield],
    ]
      .filter(([, value]) => value || value === 0)
      .map(([key, value]) => ({
        label: labels[key] || key,
        value: formatValue(key, value),
      }));

    profileMetrics.innerHTML = '';
    if (!metrics.length) {
      const item = document.createElement('li');
      item.textContent = TEXT.profileEmpty;
      profileMetrics.appendChild(item);
      return;
    }
    metrics.forEach((metric) => {
      const item = document.createElement('li');
      item.appendChild(document.createTextNode(`${metric.label}: `));
      const strong = document.createElement('strong');
      strong.textContent = metric.value;
      item.appendChild(strong);
      profileMetrics.appendChild(item);
    });
  }

  function renderNewsPanel(items) {
    if (!newsList) return;
    const newsItems = Array.isArray(items) ? items : [];
    newsList.innerHTML = '';
    if (!newsItems.length) {
      const item = document.createElement('li');
      item.textContent = TEXT.newsEmpty;
      newsList.appendChild(item);
      return;
    }
    newsItems.slice(0, 6).forEach((entry) => {
      const title = entry.title || '';
      const url = entry.url || '#';
      const source = entry.source || '';
      const time = entry.time || '';
      const snippet = entry.summary || '';
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = url || '#';
      link.target = '_blank';
      link.rel = 'noopener';
      link.textContent = title || '—';
      const metaEl = document.createElement('span');
      metaEl.className = 'news-meta';
      metaEl.textContent = [source, time].filter(Boolean).join(' · ');
      item.appendChild(link);
      item.appendChild(metaEl);
      if (snippet) {
        const snippetEl = document.createElement('span');
        snippetEl.className = 'news-snippet';
        snippetEl.textContent = snippet.length > 120 ? `${snippet.slice(0, 120)}…` : snippet;
        item.appendChild(snippetEl);
      }
      newsList.appendChild(item);
    });
  }

  function renderAiPanel(summary) {
    if (!aiSummary) return;
    const text = summary || TEXT.aiPlaceholder;
    aiSummary.textContent = text;
  }

  function updateInsightPanels(payload) {
    if (!payload || typeof payload !== 'object') return;
    renderProfilePanel(payload.profile);
    renderNewsPanel(payload.news);
    renderAiPanel(payload.ai_summary);
  }

  function applyDetailPayload(symbol, rangeKey, payload, bars, renderChart) {
    updateDetailHero(symbol, payload, bars);
    updateInsightPanels(payload || {});
    const effectiveKey =
      payload && payload.timeframe && payload.timeframe.key ? payload.timeframe.key : rangeKey;
    detailBarIntervalSec = resolveBarIntervalSeconds(effectiveKey) || inferBarIntervalSeconds(bars);
    updateDetailTimeScale(effectiveKey);
    if (!renderChart) return;
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
    if (detailSubtitle) {
      detailSubtitle.textContent =
        langPrefix === 'zh'
          ? `更新时间：${payload.generated_at || TEXT.justNow}`
          : `Updated: ${payload.generated_at || TEXT.justNow}`;
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
  }

  async function loadDetailData(symbol, rangeKey, options = {}) {
    if (!symbol) return;
    const { renderChart = true, allowCache = true } = options;
    const liveSpec = resolveLiveRange(rangeKey);
    if (detailRetryTimer) {
      clearTimeout(detailRetryTimer);
      detailRetryTimer = null;
    }
    clearLiveWait();
    stopLiveQuotePolling();
    if (liveSpec) {
      if (renderChart && ensureDetailChart() && detailManager) {
        detailManager.setLiveMode(liveSpec.mode, liveSpec.interval);
        setDetailStatus(TEXT.detailLive);
      }
      requestLiveSymbol(symbol);
      startLiveQuotePolling(symbol);
      lastLiveUpdateAt = 0;
      scheduleLiveWait();
      detailBarIntervalSec = null;
      if (detailChartEl) {
        detailChartEl.classList.remove('is-loading');
      }
      if (detailTitle) {
        detailTitle.textContent = `${symbol} · ${(rangeKey || '').toUpperCase()}`;
      }
      if (detailSubtitle) {
        detailSubtitle.textContent = TEXT.detailLive;
      }
      return;
    }
    if (detailManager) {
      detailManager.clearLiveMode();
    }
    if (renderChart && detailChartEl) {
      detailChartEl.classList.add('is-loading');
    }
    if (renderChart) {
      setDetailStatus(TEXT.detailLoading);
    }
    if (detailSource) {
      detailSource.textContent = '';
    }
    if (detailUpdated) {
      detailUpdated.textContent = '';
    }

    if (allowCache && hasCachedDetail(symbol, rangeKey)) {
      const payload = detailCache.payload || {};
      const bars = detailCache.bars || [];
      applyDetailPayload(symbol, rangeKey, payload, bars, renderChart);
      if (renderChart && detailChartEl) {
        detailChartEl.classList.remove('is-loading');
      }
      return;
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
        detailRetryTimer = setTimeout(
          () => loadDetailData(symbol, rangeKey, { renderChart, allowCache: false }),
          retryAfter * 1000
        );
        return;
      }
      if (!response.ok) {
        setDetailStatus(payload.error || TEXT.detailError, true);
        updateInsightPanels({});
        return;
      }
      const bars = Array.isArray(payload.bars) ? payload.bars : [];
      if (!bars.length) {
        setDetailStatus(TEXT.detailEmpty);
        updateInsightPanels(payload);
        return;
      }
      storeDetailCache(symbol, rangeKey, bars, payload);
      applyDetailPayload(symbol, rangeKey, payload, bars, renderChart);
    } catch (error) {
      setDetailStatus(TEXT.detailError, true);
      updateInsightPanels({});
    } finally {
      if (renderChart && detailChartEl) {
        detailChartEl.classList.remove('is-loading');
      }
    }
  }

  function setDetailRange(rangeKey) {
    detailRange = rangeKey;
    detailTimeframes.forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.range === rangeKey);
    });
    if (detailTimeframeCurrent) {
      const active = detailTimeframes.find((btn) => btn.dataset.range === rangeKey);
      if (active) {
        detailTimeframeCurrent.textContent = active.textContent || rangeKey.toUpperCase();
      }
    }
    if (detailTimeframeMenu) {
      detailTimeframeMenu.hidden = true;
    }
    updateDetailTimeScale(rangeKey);
  }

  function updateDetailTimeScale(rangeKey) {
    if (!detailManager || !detailManager.chart) return;
    const key = (rangeKey || '').toString().trim().toLowerCase();
    const liveSpec = resolveLiveRange(key);
    const isIntraday =
      Boolean(liveSpec) || /^(?:\d+)(?:s|m|h)$/.test(key) || /t$/.test(key) || key === '1d' || key === '5d';
    const showSeconds = Boolean(liveSpec && (liveSpec.mode === 'second' || liveSpec.mode === 'tick'));
    detailManager.chart.timeScale().applyOptions({
      timeVisible: isIntraday,
      secondsVisible: showSeconds,
    });
  }

  function openDetailPanel(symbol) {
    detailSymbol = symbol;
    setView('detail');
    if (detailSymbolEl) {
      detailSymbolEl.textContent = symbol;
    }
    if (detailNameEl) {
      detailNameEl.textContent = symbol;
    }
    if (detailPriceEl) {
      detailPriceEl.textContent = '--';
    }
    if (detailChangeEl) {
      detailChangeEl.textContent = '--';
      detailChangeEl.classList.remove('is-up', 'is-down');
    }
    if (detailMetaEl) {
      detailMetaEl.textContent = '';
    }
    setDetailRange(detailRange);
    loadDetailData(symbol, detailRange, { renderChart: false, allowCache: true });
  }

  function openChartView() {
    if (!detailSymbol) {
      setStatus(TEXT.statusNeedSymbol);
      setView('list');
      return;
    }
    setView('chart');
    setDetailRange(detailRange);
    loadDetailData(detailSymbol, detailRange, { renderChart: true, allowCache: true });
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
    const isTableBody = container.tagName === 'TBODY';
    if (isTableBody) {
      clearListState(container);
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 5;
      cell.textContent = TEXT.loading;
      row.appendChild(cell);
      container.appendChild(row);
      return;
    }
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
    if (container.tagName === 'TBODY') {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 5;
      cell.className = 'market-list-empty';
      cell.textContent = message;
      row.appendChild(cell);
      container.appendChild(row);
      return;
    }
    const div = document.createElement('div');
    div.className = 'market-list-empty';
    div.textContent = message;
    container.appendChild(div);
  }

  function renderError(container, message) {
    if (!container) return;
    clearListState(container);
    if (container.tagName === 'TBODY') {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 5;
      cell.className = 'market-error';
      cell.textContent = message || TEXT.genericError;
      row.appendChild(cell);
      container.appendChild(row);
      return;
    }
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
            const isAllMode = currentListType === 'all';
            loadData(symbol, {
              watchAction: 'remove',
              listType: isAllMode ? 'gainers' : currentListType,
              skipListRender: isAllMode,
              keepListType: isAllMode,
              openDetail: !isAllMode,
            });
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
            const isAllMode = currentListType === 'all';
            loadData(symbol, {
              watchAction: 'remove',
              listType: isAllMode ? 'gainers' : currentListType,
              skipListRender: isAllMode,
              keepListType: isAllMode,
              openDetail: !isAllMode,
            });
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
      if (currentListType === 'all') {
        loadData(symbol, {
          listType: 'gainers',
          skipListRender: true,
          keepListType: true,
          openDetail: true,
        });
      } else {
        loadData(symbol);
      }
    });
  }

  const hasTypeaheadUi = Boolean(searchInput && typeaheadPanel && typeaheadList);
  const TYPEAHEAD_LIMIT = 9;

  function normalizeSymbol(value) {
    return (value || '').toString().trim().toUpperCase();
  }

  function normalizeListType(value) {
    const text = (value || '').toString().trim().toLowerCase();
    if (text === 'gainers' || text === 'losers' || text === 'most_active' || text === 'all') {
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
    const isAllMode = currentListType === 'all';
    if (options.watchAction) {
      loadData(normalized, {
        watchAction: options.watchAction,
        listType: isAllMode ? 'gainers' : currentListType,
        skipListRender: isAllMode,
        keepListType: isAllMode,
        openDetail: !isAllMode,
      });
    } else if (isAllMode) {
      loadData(normalized, {
        listType: 'gainers',
        skipListRender: true,
        keepListType: true,
        openDetail: true,
      });
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
      const entry = document.createElement('div');
      entry.className = 'typeahead-option';
      entry.dataset.symbol = option.symbol;
      entry.dataset.index = String(index);
      entry.setAttribute('data-role', 'typeahead-option');
      entry.setAttribute('role', 'button');
      entry.tabIndex = -1;

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

      entry.appendChild(symbolSpan);
      entry.appendChild(meta);
      typeaheadList.appendChild(entry);
    });
    setTypeaheadVisibility(true);
  }

  function resolveLiveRange(rangeKey) {
    const raw = (rangeKey || '').toString().trim().toLowerCase();
    const tickMatch = raw.match(/^(\d+)?t$/);
    if (tickMatch) {
      return { mode: 'tick', interval: 1 };
    }
    const secMatch = raw.match(/^(\d+)(s)$/);
    if (secMatch) {
      const interval = Math.max(1, parseInt(secMatch[1], 10) || 1);
      return { mode: 'second', interval };
    }
    return null;
  }

  function resolveBarIntervalSeconds(rangeKey) {
    const key = (rangeKey || '').toString().trim().toLowerCase();
    const map = {
      '1m': 60,
      '2m': 120,
      '3m': 180,
      '5m': 300,
      '10m': 600,
      '15m': 900,
      '30m': 1800,
      '45m': 2700,
      '1h': 3600,
      '2h': 7200,
      '4h': 14400,
      '1d': 60,
      '5d': 1800,
      '1mo': 86400,
      '6mo': 86400,
    };
    return map[key] || null;
  }

  function inferBarIntervalSeconds(bars) {
    if (!Array.isArray(bars) || bars.length < 2) return null;
    const last = bars[bars.length - 1];
    const prev = bars[bars.length - 2];
    if (!last || !prev) return null;
    if (typeof last.time !== 'number' || typeof prev.time !== 'number') return null;
    const delta = last.time - prev.time;
    return delta > 0 ? delta : null;
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

  function scheduleAutoRefresh() {
    if (autoRefreshTimer) {
      clearTimeout(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    if (currentView !== 'list' || currentListType === 'all') {
      return;
    }
    autoRefreshTimer = window.setTimeout(() => {
      if (currentView !== 'list' || currentListType === 'all') return;
      if (isListLoading) {
        scheduleAutoRefresh();
        return;
      }
      const nextQuery = lastRequest.query || '';
      loadData(nextQuery, { listType: currentListType, keepListType: true, openDetail: false });
    }, AUTO_REFRESH_MS);
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
    const rawQuery = (query || '').toString().trim();
    const normalizedQuery = normalizeSymbol(rawQuery);
    const activeListType = normalizeListType(options.listType || currentListType);
    const skipListRender = Boolean(options.skipListRender);
    const keepListType = Boolean(options.keepListType);
    const openDetail = options.openDetail !== false;
    if (activeListType === 'all') {
      await loadAllStocks({ query: rawQuery, page: 1 });
      return;
    }
    lastRequest = { query: normalizedQuery || '', options: { ...options, listType: activeListType } };
    const requestPayload = {
      timeframe: currentTimeframe,
      list: activeListType,
    };
    if (!normalizedQuery && !options.limit) {
      requestPayload.limit = DEFAULT_LIST_LIMIT;
    }
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
    if (!skipListRender) {
      setListLoading(listContainer);
    }
    showChipSkeleton(recentChips, 3);
    showChipSkeleton(watchlistChips, 4);
    isListLoading = true;

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
        if (!normalizedQuery && !options.limit) {
          params.set('limit', String(DEFAULT_LIST_LIMIT));
        } else if (options.limit) {
          params.set('limit', String(options.limit));
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
      if (!keepListType && payload.list_type && responseListType !== currentListType) {
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
      if (!skipListRender) {
        renderList(listContainer, items, payload.timeframe, responseListType);
        if (!items.length) {
          renderEmpty(listContainer, TEXT.emptySymbol);
        } else {
          const liveSymbols = items
            .map((item) => item && item.symbol)
            .filter(Boolean)
            .slice(0, 40);
          if (detailSymbol) {
            liveSymbols.unshift(detailSymbol);
          }
          requestLiveSymbols(liveSymbols);
        }
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
      const rankingTimeframe = payload.ranking_timeframe;
      if (
        rankingTimeframe &&
        rankingTimeframe.key &&
        tfKey &&
        rankingTimeframe.key !== tfKey &&
        TEXT.closedFallback
      ) {
        statusMessage += ` · ${TEXT.closedFallback}`;
      }
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
      if (normalizedQuery && openDetail) {
        openDetailPanel(normalizedQuery);
      }
    } catch (error) {
      renderError(listContainer, error && error.message);
      setStatus(TEXT.statusError);
      setSource('unknown');
      hideChipSkeleton(recentChips);
      hideChipSkeleton(watchlistChips);
    } finally {
      isListLoading = false;
      scheduleAutoRefresh();
    }
  }

  function renderList(container, items, timeframe, listType) {
    if (!container) return;
    clearListState(container);
    if (!items.length) {
      renderEmpty(container, TEXT.emptyList);
      return;
    }
    const isTableBody = container.tagName === 'TBODY';
    const isMostActive = listType === 'most_active';
    const invert = listType === 'losers';

    if (rankingChangeHeader) {
      rankingChangeHeader.textContent = isMostActive
        ? TEXT.volumeLabel
        : langPrefix === 'zh'
          ? '涨跌幅'
          : 'Chg%';
    }

    if (isTableBody) {
      items.forEach((item) => {
        const symbol = (item.symbol || '').toString().toUpperCase();
        if (!symbol) return;
        const row = document.createElement('tr');
        row.dataset.symbol = normalizeSymbol(symbol);
        const symbolCell = document.createElement('td');
        symbolCell.textContent = symbol;
        const nameCell = document.createElement('td');
        nameCell.textContent = item.name || '--';
        const exchangeCell = document.createElement('td');
        exchangeCell.textContent = item.exchange || '--';
        const priceCell = document.createElement('td');
        const priceValue =
          typeof item.price === 'number'
            ? item.price
            : typeof item.last === 'number'
              ? item.last
              : Number.parseFloat(item.price);
        priceCell.textContent = Number.isFinite(priceValue) ? priceValue.toFixed(2) : '--';
        const changeCell = document.createElement('td');
        if (isMostActive) {
          changeCell.textContent = formatCompactNumber(item.volume);
          changeCell.classList.add('is-neutral');
        } else {
          const changeValue =
            typeof item.change_pct_period === 'number'
              ? item.change_pct_period
              : typeof item.change_pct_day === 'number'
                ? item.change_pct_day
                : Number.parseFloat(item.change_pct_period);
          changeCell.textContent = formatChange(changeValue);
          applyChangeState(changeCell, changeValue, invert);
        }
        row.appendChild(symbolCell);
        row.appendChild(nameCell);
        row.appendChild(exchangeCell);
        row.appendChild(priceCell);
        row.appendChild(changeCell);
        container.appendChild(row);
      });
      return;
    }

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

  function renderAllStocksMessage(message) {
    if (!allStocksBody) return;
    allStocksBody.innerHTML = '';
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.textContent = message;
    row.appendChild(cell);
    allStocksBody.appendChild(row);
  }

  function renderAllStocks(items) {
    if (!allStocksBody) return;
    allStocksBody.innerHTML = '';
    if (!Array.isArray(items) || !items.length) {
      renderAllStocksMessage(TEXT.allStocksEmpty);
      return;
    }
    items.forEach((item) => {
      const row = document.createElement('tr');
      const symbol = normalizeSymbol(item.symbol || '');
      row.dataset.symbol = symbol;

      const symbolCell = document.createElement('td');
      symbolCell.textContent = symbol || '--';

      const nameCell = document.createElement('td');
      nameCell.textContent = item.name || '--';

      const exchangeCell = document.createElement('td');
      exchangeCell.textContent = item.exchange || '--';

      const lastCell = document.createElement('td');
      if (typeof item.last === 'number' && Number.isFinite(item.last)) {
        lastCell.textContent = item.last.toFixed(2);
      } else {
        lastCell.textContent = '--';
      }

      const changeCell = document.createElement('td');
      const changeValue =
        typeof item.change_pct === 'number' && Number.isFinite(item.change_pct) ? item.change_pct : null;
      changeCell.textContent = changeValue === null ? '--' : formatChange(changeValue);
      applyChangeState(changeCell, changeValue);

      row.appendChild(symbolCell);
      row.appendChild(nameCell);
      row.appendChild(exchangeCell);
      row.appendChild(lastCell);
      row.appendChild(changeCell);
      allStocksBody.appendChild(row);
    });
  }

  function renderAllStocksPagination(page, totalPages) {
    if (!allStocksPagination) return;
    allStocksPagination.innerHTML = '';
    if (!totalPages || totalPages <= 1) {
      return;
    }
    const makeButton = (label, targetPage, disabled, isActive) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = label;
      btn.disabled = disabled;
      if (isActive) {
        btn.classList.add('is-active');
      }
      btn.dataset.page = targetPage;
      return btn;
    };

    allStocksPagination.appendChild(
      makeButton(TEXT.pagePrev, Math.max(1, page - 1), page <= 1, false),
    );

    const start = Math.max(1, page - 2);
    const end = Math.min(totalPages, page + 2);
    for (let p = start; p <= end; p += 1) {
      allStocksPagination.appendChild(makeButton(String(p), p, false, p === page));
    }

    allStocksPagination.appendChild(
      makeButton(TEXT.pageNext, Math.min(totalPages, page + 1), page >= totalPages, false),
    );
  }

  function updateAllStocksCount(total, page, totalPages) {
    if (!allStocksCount) return;
    const safeTotal = Number.isFinite(total) ? total : 0;
    const safePage = Number.isFinite(page) ? page : 1;
    const safeTotalPages = Number.isFinite(totalPages) ? totalPages : 1;
    if (typeof TEXT.allStocksCount === 'function') {
      allStocksCount.textContent = TEXT.allStocksCount(safeTotal, safePage, safeTotalPages);
    } else {
      allStocksCount.textContent = `${safeTotal}`;
    }
  }

  function setAllStocksLetter(letter) {
    const next = (letter || '').toString().trim().toUpperCase() || 'ALL';
    allStocksLetter = next;
    if (!allStocksLetters) return;
    Array.from(allStocksLetters.querySelectorAll('[data-letter]')).forEach((btn) => {
      const btnLetter = (btn.dataset.letter || '').toString().trim().toUpperCase();
      btn.classList.toggle('is-active', btnLetter === next);
    });
  }

  async function loadAllStocks(options = {}) {
    const nextLetter = options.letter ? options.letter.toString().toUpperCase() : allStocksLetter;
    const nextQuery = typeof options.query === 'string' ? options.query.trim() : allStocksQuery;
    const nextPage = Number.isFinite(options.page) ? options.page : allStocksPage;
    const nextSize = typeof options.size === 'number' && Number.isFinite(options.size) ? options.size : allStocksSize;
    allStocksLetter = nextLetter || 'ALL';
    allStocksQuery = nextQuery;
    allStocksPage = Math.max(1, nextPage || 1);
    allStocksSize = Math.min(200, Math.max(20, nextSize || 50));
    setAllStocksLetter(allStocksLetter);
    renderAllStocksMessage(TEXT.allStocksLoading);
    if (allStocksPagination) {
      allStocksPagination.innerHTML = '';
    }

    try {
      const endpointBase = assetsUrl || '/api/market/assets/';
      const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
      const params = new URLSearchParams({
        page: String(allStocksPage),
        size: String(allStocksSize),
      });
      if (allStocksLetter && allStocksLetter !== 'ALL') {
        params.set('letter', allStocksLetter);
      }
      if (allStocksQuery) {
        params.set('query', allStocksQuery);
      }
      const response = await fetch(`${endpoint}?${params.toString()}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || TEXT.genericError);
      }
      const items = Array.isArray(payload.items) ? payload.items : [];
      renderAllStocks(items);
      const total = Number(payload.total) || 0;
      const page = Number(payload.page) || 1;
      const totalPages = Number(payload.total_pages) || 1;
      updateAllStocksCount(total, page, totalPages);
      renderAllStocksPagination(page, totalPages);
    } catch (error) {
      renderAllStocksMessage(error && error.message ? error.message : TEXT.genericError);
    }
  }

  function switchList(type) {
    const nextType = normalizeListType(type);
    if (nextType === currentListType) return;
    if (nextType === 'all' && currentListType !== 'all') {
      lastRankingType = currentListType;
    }
    if (nextType !== 'all') {
      lastRankingType = nextType;
    }
    setActiveListType(nextType);
    if (nextType === 'all') {
      const query = (searchInput && searchInput.value.trim()) || '';
      loadAllStocks({ query, page: 1, letter: allStocksLetter });
    } else {
      loadData('', { listType: nextType });
    }
  }

  function handleCardClick(event) {
    if (event.target.closest('a')) {
      return;
    }
    const card = event.target.closest('.market-card');
    if (card) {
      const symbol = card.dataset.symbol;
      if (!symbol) return;
      openDetailPanel(symbol);
      return;
    }
    const row = event.target.closest('tr[data-symbol]');
    if (!row) return;
    const symbol = row.dataset.symbol;
    if (!symbol) return;
    openDetailPanel(symbol);
  }

  attachChipHandler(recentChips, { allowRemove: true, onRemove: (symbol) => requestRecentAction('delete', symbol) });
  attachChipHandler(watchlistChips, { allowRemove: true, watch: true });

  if (rankTabs.length) {
    const activeTab = rankTabs.find((tab) => tab.classList.contains('is-active')) || rankTabs[0];
    if (activeTab) {
      const activeType = normalizeListType(activeTab.dataset.list);
      setActiveListType(activeType);
      if (activeType !== 'all') {
        lastRankingType = activeType;
      }
    }
    rankTabs.forEach((tab) => {
      tab.addEventListener('click', () => switchList(tab.dataset.list));
    });
  }

  if (listContainer) {
    listContainer.addEventListener('click', handleCardClick);
  }

  if (allStocksLetters) {
    allStocksLetters.addEventListener('click', (event) => {
      const button = event.target.closest('[data-letter]');
      if (!button) return;
      const letter = button.dataset.letter || 'ALL';
      setAllStocksLetter(letter);
      loadAllStocks({ letter: allStocksLetter, page: 1, query: allStocksQuery });
    });
  }

  if (allStocksBack) {
    allStocksBack.addEventListener('click', () => {
      switchList(lastRankingType || 'gainers');
    });
  }

  if (allStocksPagination) {
    allStocksPagination.addEventListener('click', (event) => {
      const button = event.target.closest('button');
      if (!button || button.disabled) return;
      const nextPage = parseInt(button.dataset.page, 10);
      if (!Number.isFinite(nextPage)) return;
      loadAllStocks({ page: nextPage, letter: allStocksLetter, query: allStocksQuery });
    });
  }

  if (allStocksBody) {
    allStocksBody.addEventListener('click', (event) => {
      const row = event.target.closest('tr');
      if (!row) return;
      const symbol = row.dataset.symbol;
      if (!symbol) return;
      openDetailPanel(symbol);
    });
  }

  if (viewBackButtons.length) {
    viewBackButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.viewBack;
        if (!target) return;
        setView(target);
      });
    });
  }

  if (viewChartButton) {
    viewChartButton.addEventListener('click', () => {
      openChartView();
    });
  }

  if (detailTimeframes.length) {
    const activeRange = detailTimeframes.find((btn) => btn.classList.contains('is-active'));
    if (activeRange && activeRange.dataset.range) {
      detailRange = activeRange.dataset.range;
    }
    setDetailRange(detailRange);
    detailTimeframes.forEach((btn) => {
      btn.addEventListener('click', () => {
        if (btn.disabled || btn.classList.contains('is-disabled')) return;
        const rangeKey = btn.dataset.range || '1d';
        if (rangeKey === detailRange) return;
        setDetailRange(rangeKey);
        if (detailSymbol) {
          loadDetailData(detailSymbol, rangeKey, { renderChart: true, allowCache: false });
        }
      });
    });
  }

  if (detailTimeframeTrigger && detailTimeframeMenu) {
    detailTimeframeTrigger.addEventListener('click', (event) => {
      event.stopPropagation();
      detailTimeframeMenu.hidden = !detailTimeframeMenu.hidden;
    });
    document.addEventListener('click', (event) => {
      if (detailTimeframeMenu.hidden) return;
      const target = event.target;
      if (detailTimeframeMenu.contains(target) || detailTimeframeTrigger.contains(target)) {
        return;
      }
      detailTimeframeMenu.hidden = true;
    });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (!detailTimeframeMenu.hidden) {
        detailTimeframeMenu.hidden = true;
      }
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

  window.addEventListener('resize', resizeDetailChart);
  window.addEventListener('load', () => {
    if (currentView === 'list') {
      setView('list');
    }
    resizeDetailChart();
  });

  if (detailTitle && !detailSymbol) {
    setDetailStatus(TEXT.detailPrompt);
  }

  updateInsightPanels({});

  loadData();
  connectMarketSocket();
})();
