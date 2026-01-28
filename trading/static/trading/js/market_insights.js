(() => {
  const apiMeta = document.querySelector('meta[name="market-api"]');
  const apiUrl = apiMeta ? apiMeta.getAttribute('content') : window.MARKET_API_URL || '/market/api/';
  const chartApiMeta = document.querySelector('meta[name="market-chart-api"]');
  const chartApiUrl = chartApiMeta ? chartApiMeta.getAttribute('content') : '/api/market/chart/';
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
  const rankingChangeLabel = document.querySelector('[data-role="ranking-change-label"]');
  const rankingSortButton = document.querySelector('[data-role="ranking-sort"]');
  const rankingSortIcon = document.querySelector('[data-role="ranking-sort-icon"]');
  const rankingsSection = document.querySelector('[data-role="rankings-section"]');
  const allStocksSection = document.querySelector('[data-role="all-stocks"]');
  const allStocksLetters = document.querySelector('[data-role="all-stocks-letters"]');
  const allStocksBody = document.querySelector('[data-role="all-stocks-body"]');
  const allStocksPagination = document.querySelector('[data-role="all-stocks-pagination"]');
  const allStocksCount = document.querySelector('[data-role="all-stocks-count"]');
  const allStocksBack = document.querySelector('[data-role="all-stocks-back"]');
  const rankTabs = Array.prototype.slice.call(document.querySelectorAll('[data-role="rank-tab"]'));
  const rankDesc = document.querySelector('[data-role="rank-desc"]');
  const rankContext = document.querySelector('[data-role="rank-context"]');
  const rankSortSummaries = Array.prototype.slice.call(document.querySelectorAll('[data-role="rank-sort-summary"]'));
  const rankSentinel = document.querySelector('[data-role="ranking-sentinel"]');
  const statusState = document.querySelector('[data-role="status-state"]');
  const statusText = document.querySelector('[data-role="status-text"]');
  const statusContext = document.querySelector('[data-role="status-context"]');
  const statusUpdated = document.querySelector('[data-role="status-updated"]');
  const statusExpanded = document.querySelector('[data-role="status-expanded"]');
  let snapshotText = document.querySelector('[data-role="snapshot-text"]');
  const snapshotProgress = document.querySelector('[data-role="snapshot-progress"]');
  const snapshotProgressFill = document.querySelector('[data-role="snapshot-progress-fill"]');
  const snapshotWarning = document.querySelector('[data-role="snapshot-warning"]');
  const snapshotWarningText = document.querySelector('[data-role="snapshot-warning-text"]');
  const snapshotDetails = document.querySelector('[data-role="snapshot-details"]');
  const snapshotDetailsPanel = document.querySelector('[data-role="snapshot-details-panel"]');
  const snapshotRetry = document.querySelector('[data-role="snapshot-retry"]');
  const priceBasisText = document.querySelector('[data-role="price-basis"]');
  const sourceText = document.querySelector('[data-role="source-text"]');
  const statusRefresh = document.querySelector('[data-role="status-refresh"]');
  const statusSection = document.querySelector('.market-status');
  const timezoneToggle = document.querySelector('[data-role="timezone-toggle"]');
  const autoRefreshToggle = document.querySelector('[data-role="auto-refresh-toggle"]');
  const autoRefreshPause = document.querySelector('[data-role="auto-refresh-pause"]');
  const autoRefreshCountdown = document.querySelector('[data-role="auto-refresh-countdown"]');
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
  const quickTabs = Array.prototype.slice.call(document.querySelectorAll('[data-role="quick-tab"]'));
  const quickPanels = Array.prototype.slice.call(document.querySelectorAll('[data-role="quick-panel"]'));
  const recentClear = document.querySelector('[data-role="recent-clear"]');
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
  const detailRangeButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('.detail-timeframe--quick'));
  const intervalTrigger = detailRoot.querySelector('[data-role="detail-interval-trigger"]');
  const intervalMenu = detailRoot.querySelector('[data-role="detail-interval-menu"]');
  const intervalCurrent = detailRoot.querySelector('[data-role="detail-interval-current"]');
  const intervalSelectButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('[data-role="detail-interval-select"]'));
  const intervalFavButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('[data-role="detail-interval-fav"]'));
  const intervalFavoritesWrap = detailRoot.querySelector('[data-role="detail-interval-favorites"]');
  const intervalFavoritesList = detailRoot.querySelector('[data-role="detail-interval-favorites-list"]');
  const intervalCustomToggle = detailRoot.querySelector('[data-role="detail-interval-custom-toggle"]');
  const intervalCustomPanel = detailRoot.querySelector('[data-role="detail-interval-custom-panel"]');
  const intervalCustomValue = detailRoot.querySelector('[data-role="detail-interval-custom-value"]');
  const intervalCustomUnit = detailRoot.querySelector('[data-role="detail-interval-custom-unit"]');
  const intervalCustomApply = detailRoot.querySelector('[data-role="detail-interval-custom-apply"]');
  const detailAdvancedToggle = detailRoot.querySelector('[data-role="detail-advanced-toggle"]');
  const detailAdvancedPanel = detailRoot.querySelector('[data-role="detail-advanced-panel"]');
  const profileSummary = document.querySelector('[data-role="profile-summary"]');
  const profileMetrics = document.querySelector('[data-role="profile-metrics"]');
  const aiSummary = document.querySelector('[data-role="ai-summary"]');
  const newsList = document.querySelector('[data-role="news-list"]');
  const viewList = document.querySelector('[data-view="list"]');
  const viewDetail = document.querySelector('[data-view="detail"]');
  const viewChart = document.querySelector('[data-view="chart"]');
  const viewBackButtons = Array.prototype.slice.call(document.querySelectorAll('[data-view-back]'));
  const viewChartButton = document.querySelector('[data-view-chart]');
  const paneTabs = Array.prototype.slice.call(document.querySelectorAll('[data-role="pane-tab"]'));
  const detailSymbolEl = document.querySelector('[data-role="detail-symbol"]');
  const detailNameEl = document.querySelector('[data-role="detail-name"]');
  const detailPriceEl = document.querySelector('[data-role="detail-price"]');
  const detailChangeEl = document.querySelector('[data-role="detail-change"]');
  const detailMetaEl = document.querySelector('[data-role="detail-meta"]');
  const tradeSymbolEl = document.querySelector('[data-role="trade-symbol"]');
  const tradeQtyInput = document.querySelector('[data-role="trade-qty"]');
  const tradeBuyBtn = document.querySelector('[data-role="trade-buy"]');
  const tradeSellBtn = document.querySelector('[data-role="trade-sell"]');
  const tradeStatusEl = document.querySelector('[data-role="trade-status"]');
  const tradeModeButtons = Array.prototype.slice.call(document.querySelectorAll('[data-role="trade-mode-btn"]'));
  const tradeModeStatus = document.querySelector('[data-role="trade-mode-status"]');
  const rankingTable = document.querySelector('[data-role="ranking-table"]');
  const marketPage = document.querySelector('.market-page');
  const marketSocketUrl = (() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}/ws/market/`;
  })();
  const chartSocketUrl = (() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}/ws/market-chart/`;
  })();

  const stateUtils = window.MarketInsightsState || {};
  const mergeRankItems =
    stateUtils.mergeRankItems ||
    ((existing, incoming) => ({
      merged: Array.isArray(existing) ? existing.concat(Array.isArray(incoming) ? incoming : []) : Array.isArray(incoming) ? incoming : [],
      appended: Array.isArray(incoming) ? incoming : [],
    }));
  const toggleSortState =
    stateUtils.toggleSortState ||
    ((state) => (state === 'default' ? 'desc' : state === 'desc' ? 'asc' : 'default'));
  const formatTimestamp = stateUtils.formatTimestamp || ((value) => (value || '').toString());
  const loadPreference = stateUtils.loadPreference || ((key, fallback) => fallback);
  const savePreference = stateUtils.savePreference || (() => {});
  const locale = langPrefix === 'zh' ? 'zh-CN' : 'en-US';
  const orderEndpoint = '/api/market/order/';
  const tradeModeEndpoint = '/api/market/trading/';

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
  let detailInterval = '1m';
  let detailBarIntervalSec = null;
  let detailRetryTimer = null;
  let currentView = 'list';
  let autoRefreshTimer = null;
  let autoRefreshCountdownTimer = null;
  let autoRefreshNextAt = 0;
  let autoRefreshRemainingMs = 0;
  let autoRefreshPaused = false;
  let autoRefreshSuspended = false;
  let statusMessageOverride = '';
  let currentStatusState = 'ready';
  let snapshotState = null;
  let isListLoading = false;
  let liveWaitTimer = null;
  let lastLiveUpdateAt = 0;
  let liveQuoteTimer = null;
  let chartInitTimer = null;
  let pendingChartRender = null;
  let chartSocket = null;
  let chartSocketRetryTimer = null;
  let chartSocketReady = false;
  let chartSocketSymbol = '';
  const DEFAULT_AUTO_REFRESH_MS = 60 * 1000;
  const AUTO_REFRESH_OPTIONS = [0, 15000, 30000, 60000];
  const PREF_TIMEZONE_KEY = 'market.timezone';
  const PREF_AUTO_REFRESH_KEY = 'market.autoRefreshMs';
  const PREF_INTERVAL_KEY = 'market.chart.interval';
  const PREF_INTERVAL_FAVORITES_KEY = 'market.chart.intervalFavorites';
  const PREF_INTERVAL_CUSTOM_KEY = 'market.chart.intervalCustom';
  let timezoneMode = loadPreference(PREF_TIMEZONE_KEY, 'utc');
  if (timezoneMode !== 'utc' && timezoneMode !== 'local') {
    timezoneMode = 'utc';
  }
  let autoRefreshMs = parseInt(loadPreference(PREF_AUTO_REFRESH_KEY, `${DEFAULT_AUTO_REFRESH_MS}`), 10);
  if (!Number.isFinite(autoRefreshMs) || !AUTO_REFRESH_OPTIONS.includes(autoRefreshMs)) {
    autoRefreshMs = DEFAULT_AUTO_REFRESH_MS;
  }
  let intervalFavorites = [];
  let customIntervals = [];
  const LIVE_WAIT_MS = 10000;
  const LIVE_QUOTE_POLL_MS = 15000;
  const RANK_PAGE_SIZE = 20;
  const RANKING_COLUMNS =
    rankingTable && rankingTable.querySelectorAll('th').length
      ? rankingTable.querySelectorAll('th').length
      : 6;
  const ALL_STOCKS_COLUMNS = 5;
  let rankOffset = 0;
  let rankPageSize = RANK_PAGE_SIZE;
  let rankNextOffset = null;
  let rankHasMore = false;
  let rankIsLoadingMore = false;
  let rankObserver = null;
  let rankItemsBase = [];
  let rankItems = [];
  let rankSort = 'default';
  let chartVisible = false;
  let lastRankingTimeframe = null;
  let lastRankingListType = null;
  let lastStatusGeneratedAt = '';
  let lastSnapshotMeta = null;
  let lastSnapshotKey = '';
  let detailInfoCache = {
    symbol: '',
    range: '',
    payload: null,
  };
  let detailChartCache = {
    symbol: '',
    range: '',
    interval: '',
    bars: [],
    payload: null,
  };
  let tradeMode = 'paper';
  let tradeExecutionEnabled = false;
  let tradeModeBusy = false;
  let tradeModeError = '';

  const TEXT_ZH = {
        timeframes: { '1d': '实时榜', '5d': '近5日', '1mo': '近1月', '6mo': '近6月' },
        listLabels: {
          gainers: '涨幅榜',
          losers: '跌幅榜',
          most_active: '活跃榜',
          top_turnover: '成交额',
          all: '全部股票',
        },
        statusLabels: {
          ready: '已就绪',
          refreshing: '刷新中',
          snapshot_building: '生成快照',
          partial_ready: '部分就绪',
          stale: '数据过期',
        },
        statusMessages: {
          ready: '数据已更新',
          refreshing: '正在刷新榜单…',
          snapshot_building: '正在生成快照…',
          partial_ready: '快照已完成，部分标的仍在补齐/校验',
          stale: '数据可能已过期',
        },
        loading: '正在加载',
        dataSuffix: '数据…',
        updated: '数据已更新',
        sortedBy: '排序：',
        sortDefault: '默认',
        sortAsc: '升序',
        sortDesc: '降序',
        loadingMore: '正在加载更多…',
        closedFallback: '闭市，已展示上一交易日榜单',
        justNow: '刚刚',
        retrying: '请求过快，正在重试',
        refreshSkipped: '刷新中，已忽略重复请求',
        countdownPrefix: '下次刷新',
        paused: '已暂停',
        pause: '暂停',
        resume: '继续',
        emptySymbol: '暂无可展示的标的。',
        emptyList: '暂无数据',
        emptyHint: '试试切换时间范围或搜索股票。',
        statusError: '加载失败，请稍后再试。',
        genericError: '加载失败',
        errorHint: '请检查网络或稍后再试。',
        retryAction: '重试',
        updatedLabel: '更新：',
        partialData: '部分数据尚未就绪',
        partialDetail: '快照已完成，但部分标的仍在后台补齐或校验。',
        emptyChips: '暂无推荐',
        emptyWatchlist: '还没有自选股。',
        statusNeedSymbol: '请先输入股票代码。',
        watchAdded: (symbol) => `已加入自选：${symbol}`,
        watchRemoved: (symbol) => `已移除自选：${symbol}`,
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
        detailWindowLimited: '高频数据仅展示最近窗口。',
        detailDowngraded: (message) => message || '已自动降级为分钟级行情。',
        intervalInvalid: '请输入有效的时间粒度。',
        tradeSubmitting: '提交中…',
        tradeSuccess: '已提交订单',
        tradeFailed: '下单失败',
        tradeMissingSymbol: '请先选择标的',
        tradeMissingQty: '请输入下单数量',
        tradeModePaper: '模拟下单',
        tradeModeLive: '实盘下单',
        tradeModeExecutionOff: '执行未开启',
        tradeModeUpdating: '切换中…',
        tradeModeFailed: '切换失败',
        tradeModeConfirm: '确认切换到实盘？这会使用 Live 账户。',
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
        snapshotPrefix: '快照',
        snapshotUnknown: '—',
        snapshotUpdated: '已更新',
        snapshotRunning: '刷新中',
        snapshotError: '刷新失败',
        sourcePrefix: '数据来源：',
        sourceLabels: {
          alpaca: 'Alpaca',
          cache: '缓存',
          unknown: '未知',
        },
      };

  const TEXT_EN = {
        timeframes: { '1d': 'Realtime', '5d': '5D', '1mo': '1M', '6mo': '6M' },
        listLabels: {
          gainers: 'Gainers',
          losers: 'Losers',
          most_active: 'Most Active',
          top_turnover: 'Turnover',
          all: 'All Stocks',
        },
        statusLabels: {
          ready: 'Up to date',
          refreshing: 'Refreshing',
          snapshot_building: 'Building',
          partial_ready: 'Partial',
          stale: 'Stale',
        },
        statusMessages: {
          ready: 'Data is up to date.',
          refreshing: 'Refreshing leaderboard…',
          snapshot_building: 'Snapshot is building…',
          partial_ready: 'Snapshot complete; some symbols are still backfilling.',
          stale: 'Data may be stale.',
        },
        loading: 'Loading',
        dataSuffix: 'data…',
        updated: 'Data refreshed',
        sortedBy: 'Sorted by:',
        sortDefault: 'default',
        sortAsc: 'asc',
        sortDesc: 'desc',
        loadingMore: 'Loading more…',
        closedFallback: 'Market closed, showing previous session movers',
        justNow: 'just now',
        retrying: 'Rate limited, retrying',
        refreshSkipped: 'Refresh already in progress',
        countdownPrefix: 'Next refresh',
        paused: 'Paused',
        pause: 'Pause',
        resume: 'Resume',
        emptySymbol: 'No symbols to display.',
        emptyList: 'No data',
        emptyHint: 'Try a different timeframe or search for a ticker.',
        statusError: 'Failed to load, please try again later.',
        genericError: 'Failed to load',
        errorHint: 'Check your network connection and try again.',
        retryAction: 'Retry',
        updatedLabel: 'Updated:',
        partialData: 'Partial data pending',
        partialDetail: 'Snapshot is complete, but some instruments are still being backfilled or validated.',
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
        detailWindowLimited: 'High-frequency data is window-limited.',
        detailDowngraded: (message) => message || 'Interval auto-downgraded to minute bars.',
        intervalInvalid: 'Enter a valid interval value.',
        tradeSubmitting: 'Submitting…',
        tradeSuccess: 'Order submitted',
        tradeFailed: 'Order failed',
        tradeMissingSymbol: 'Select a symbol first',
        tradeMissingQty: 'Enter a quantity',
        tradeModePaper: 'Paper trading',
        tradeModeLive: 'Live trading',
        tradeModeExecutionOff: 'Execution disabled',
        tradeModeUpdating: 'Switching…',
        tradeModeFailed: 'Switch failed',
        tradeModeConfirm: 'Switch to live trading? This will use your Live account.',
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
        snapshotPrefix: 'Snapshot',
        snapshotUnknown: '—',
        snapshotUpdated: 'Updated',
        snapshotRunning: 'Refreshing',
        snapshotError: 'Failed',
        sourcePrefix: 'Data source: ',
        sourceLabels: {
          alpaca: 'Alpaca',
          cache: 'Cache',
          unknown: 'Unknown',
        },
      };

  const TEXT = langPrefix === 'zh' ? TEXT_ZH : TEXT_EN;

  const ORDER_ERROR_TRANSLATIONS = [
    { pattern: /not\s+fractionable/i, zh: '该标的不支持小数股交易' },
    { pattern: /insufficient\s+buying\s+power/i, zh: '可用资金不足' },
    { pattern: /insufficient\s+qty/i, zh: '可卖数量不足' },
    { pattern: /invalid\s+symbol/i, zh: '标的无效' },
    { pattern: /market\s+is\s+closed/i, zh: '市场已休市' },
    { pattern: /missing_credentials/i, zh: '缺少 API 密钥' },
    { pattern: /notional\s+must\s+be/i, zh: '下单金额不合法' },
    { pattern: /qty\s+must\s+be/i, zh: '下单数量不合法' },
  ];

  function buildOrderErrorMessage(rawMessage) {
    const raw = (rawMessage || '').toString().trim();
    if (!raw) {
      return TEXT.tradeFailed;
    }
    let detail = raw;
    const splitIndex = raw.indexOf(':');
    if (splitIndex !== -1) {
      detail = raw.slice(splitIndex + 1).trim();
    }
    let zhDetail = '';
    for (const item of ORDER_ERROR_TRANSLATIONS) {
      if (item.pattern.test(detail)) {
        zhDetail = item.zh;
        break;
      }
    }
    if (langPrefix === 'zh') {
      return zhDetail ? `订单提交失败：${zhDetail}` : `订单提交失败：${detail || raw}`;
    }
    return detail ? `Order failed: ${detail}` : `Order failed: ${raw}`;
  }

  function ensureSnapshotNode() {
    if (snapshotText || !statusSection || !sourceText) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'market-status-meta';
    snapshotText = document.createElement('span');
    snapshotText.className = 'market-snapshot';
    snapshotText.dataset.role = 'snapshot-text';
    snapshotText.textContent = `${TEXT.snapshotPrefix}：${TEXT.snapshotUnknown}`;
    wrapper.appendChild(snapshotText);
    wrapper.appendChild(sourceText);
    statusSection.appendChild(wrapper);
  }

  ensureSnapshotNode();

  function getIndicatorLib() {
    if (window.technicalindicators) return window.technicalindicators;
    if (window.SMA && window.EMA && window.RSI) return window;
    return null;
  }

  function formatAxisTime(epochSeconds, { timezoneMode: mode = 'utc', showSeconds = false } = {}) {
    if (!Number.isFinite(epochSeconds)) return '';
    const ms = epochSeconds * 1000;
    const date = new Date(ms);
    const options = {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    };
    if (showSeconds) {
      options.second = '2-digit';
    }
    if (mode === 'utc') {
      options.timeZone = 'UTC';
    }
    const formatted = new Intl.DateTimeFormat(locale || undefined, options).format(date);
    if (!showSeconds) {
      return formatted;
    }
    const fractional = ms % 1000;
    if (!fractional) {
      return formatted;
    }
    const frac = String(Math.floor(fractional)).padStart(3, '0');
    return `${formatted}.${frac}`;
  }

  class ChartManager {
    static registry = [];

    constructor({ container, indicatorContainer, langPrefix, onStatus, timezoneMode }) {
      this.container = container;
      this.indicatorContainer = indicatorContainer;
      this.langPrefix = langPrefix || 'zh';
      this.onStatus = onStatus;
      this.timezoneMode = timezoneMode || 'utc';
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
      this.intervalSpec = null;
      this.axisShowSeconds = false;
      this.liveBar = null;
      this.liveBucket = null;
      this.liveTickCount = 0;
      this.liveTickTarget = 0;
      this.liveMaxBars = 800;
      this.lastLiveTime = null;
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
        localization: {
          locale,
          timeFormatter: (timestamp) =>
            formatAxisTime(timestamp, { timezoneMode: this.timezoneMode, showSeconds: this.axisShowSeconds }),
        },
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

    setData(bars, options = {}) {
      const { intervalSpec = null } = options || {};
      this._resetLiveState();
      this.lastLivePrice = null;
      this.ohlcData = this._sanitizeBars(bars);
      if (intervalSpec) {
        this.setIntervalSpec(intervalSpec, { preserveData: true });
      }
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

    setIntervalSpec(intervalSpec, { preserveData = false } = {}) {
      this.intervalSpec = intervalSpec;
      this._resetLiveState();
      if (intervalSpec && preserveData) {
        this._syncLiveStateFromData(intervalSpec);
      }
      const showSeconds = Boolean(intervalSpec && (intervalSpec.unit === 'tick' || intervalSpec.unit === 'second'));
      const timeVisible = Boolean(intervalSpec && intervalSpec.unit !== 'day');
      this.setAxisOptions({ showSeconds, timeVisible });
    }

    setAxisOptions({ showSeconds = false, timeVisible = true } = {}) {
      this.axisShowSeconds = Boolean(showSeconds);
      if (this.chart) {
        this.chart.timeScale().applyOptions({
          timeVisible: Boolean(timeVisible),
          secondsVisible: this.axisShowSeconds,
        });
        this._applyLocalization(this.chart);
      }
      if (this.indicatorChart) {
        this._applyLocalization(this.indicatorChart);
      }
    }

    setTimezone(mode) {
      this.timezoneMode = mode || 'utc';
      if (this.chart) {
        this._applyLocalization(this.chart);
      }
      if (this.indicatorChart) {
        this._applyLocalization(this.indicatorChart);
      }
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

    updateLivePrice(price, ts, size = 0) {
      if (typeof price !== 'number' || Number.isNaN(price)) return;
      const timestamp = Number.isFinite(ts) ? ts : Date.now() / 1000;
      this.applyTradeUpdate({ price, size, ts: timestamp });
    }

    applyTradeUpdate(trade) {
      if (!trade || typeof trade.price !== 'number' || Number.isNaN(trade.price)) return;
      if (!this.candleSeries) return;
      const timestamp = Number.isFinite(trade.ts) ? trade.ts : Date.now() / 1000;
      const price = trade.price;
      const size = Number.isFinite(trade.size) ? trade.size : 0;
      this.updatePriceLine(price);
      const interval = this.intervalSpec;
      if (!interval) return;
      const time = this._ensureMonotonicTime(timestamp);
      if (interval.unit === 'tick') {
        this._applyTickTrade({ time, price, size }, interval.value);
        return;
      }
      const intervalSeconds = interval.seconds || 1;
      this._applyTimeTrade({ time, price, size }, intervalSeconds);
    }

    updateCurrentBar(price, ts, intervalSec) {
      if (!this.candleSeries) return;
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

    _applyLocalization(chartInstance) {
      if (!chartInstance) return;
      chartInstance.applyOptions({
        localization: {
          locale,
          timeFormatter: (timestamp) =>
            formatAxisTime(timestamp, { timezoneMode: this.timezoneMode, showSeconds: this.axisShowSeconds }),
        },
      });
    }

    _resetLiveState() {
      this.liveBar = null;
      this.liveBucket = null;
      this.liveTickCount = 0;
      this.liveTickTarget = 0;
      this.lastLiveTime = null;
    }

    _syncLiveStateFromData(intervalSpec) {
      if (!intervalSpec || !Array.isArray(this.ohlcData) || !this.ohlcData.length) return;
      const last = this.ohlcData[this.ohlcData.length - 1];
      if (!last || !Number.isFinite(last.time)) return;
      if (intervalSpec.unit === 'tick') {
        const count = Number.isFinite(last.trade_count) ? Number(last.trade_count) : 0;
        if (count > 0 && count < intervalSpec.value) {
          this.liveBar = { ...last };
          this.liveTickCount = count;
          this.liveTickTarget = intervalSpec.value;
        }
      } else {
        const intervalSeconds = intervalSpec.seconds || 1;
        const bucket = Math.floor(Number(last.time) / intervalSeconds) * intervalSeconds;
        this.liveBucket = bucket;
        this.liveBar = { ...last };
      }
      this.lastLiveTime = Number(last.time);
    }

    _ensureMonotonicTime(time) {
      if (!Number.isFinite(time)) return Date.now() / 1000;
      if (this.lastLiveTime === null || time > this.lastLiveTime) {
        this.lastLiveTime = time;
        return time;
      }
      const adjusted = this.lastLiveTime + 1e-3;
      this.lastLiveTime = adjusted;
      return adjusted;
    }

    _applyTickTrade({ time, price, size }, ticksPerBar) {
      const target = Math.max(1, Number.parseInt(ticksPerBar, 10) || 1);
      if (!this.liveBar || this.liveTickCount >= target) {
        const bar = {
          time,
          open: price,
          high: price,
          low: price,
          close: price,
          volume: size,
          trade_count: 1,
        };
        this.liveBar = bar;
        this.liveTickCount = 1;
        this._appendOrUpdate(bar, true);
      } else {
        this.liveBar.high = Math.max(this.liveBar.high, price);
        this.liveBar.low = Math.min(this.liveBar.low, price);
        this.liveBar.close = price;
        this.liveBar.time = time;
        this.liveBar.volume = (this.liveBar.volume || 0) + size;
        this.liveBar.trade_count = (this.liveBar.trade_count || 0) + 1;
        this.liveTickCount += 1;
        this._appendOrUpdate(this.liveBar, false);
      }
      if (this.liveTickCount >= target) {
        this.liveTickCount = 0;
        this.liveBar = null;
      }
    }

    _applyTimeTrade({ time, price, size }, intervalSeconds) {
      const bucket = Math.floor(time / intervalSeconds) * intervalSeconds;
      if (!this.liveBar || this.liveBucket !== bucket) {
        const bar = {
          time: bucket,
          open: price,
          high: price,
          low: price,
          close: price,
          volume: size,
          trade_count: 1,
        };
        this.liveBar = bar;
        this.liveBucket = bucket;
        this._appendOrUpdate(bar, true);
      } else {
        this.liveBar.high = Math.max(this.liveBar.high, price);
        this.liveBar.low = Math.min(this.liveBar.low, price);
        this.liveBar.close = price;
        this.liveBar.volume = (this.liveBar.volume || 0) + size;
        this.liveBar.trade_count = (this.liveBar.trade_count || 0) + 1;
        this._appendOrUpdate(this.liveBar, false);
      }
    }

    _appendOrUpdate(bar, isNew) {
      if (!this.candleSeries) return;
      if (!Array.isArray(this.ohlcData)) {
        this.ohlcData = [];
      }
      const lastIndex = this.ohlcData.length - 1;
      const last = this.ohlcData[lastIndex];
      if (isNew || !last) {
        this.ohlcData.push({ ...bar });
        this._trimLiveBars();
        this.candleSeries.update({ ...bar });
        return;
      }
      this.ohlcData[lastIndex] = { ...bar };
      this.candleSeries.update({ ...bar });
    }

    _trimLiveBars() {
      if (this.ohlcData.length <= this.liveMaxBars) return;
      this.ohlcData = this.ohlcData.slice(-this.liveMaxBars);
      if (this.candleSeries) {
        this.candleSeries.setData(this.ohlcData);
      }
    }

    resize() {
      if (this.chart && this.container) {
        const width = this.container.clientWidth || 0;
        const height = this.container.clientHeight || 0;
        if (width > 0 && height > 0) {
          this.chart.applyOptions({ width, height });
        }
      }
      if (this.indicatorChart && this.indicatorContainer) {
        const width = this.indicatorContainer.clientWidth || 0;
        const height = this.indicatorContainer.clientHeight || 0;
        if (width > 0 && height > 0) {
          this.indicatorChart.applyOptions({ width, height });
        }
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
        const time = Number.isFinite(bar.time) ? Number(bar.time) : Number.parseFloat(bar.time);
        const open = Number.isFinite(bar.open) ? bar.open : Number.parseFloat(bar.open);
        const high = Number.isFinite(bar.high) ? bar.high : Number.parseFloat(bar.high);
        const low = Number.isFinite(bar.low) ? bar.low : Number.parseFloat(bar.low);
        const close = Number.isFinite(bar.close) ? bar.close : Number.parseFloat(bar.close);
        if (!Number.isFinite(time) || !Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) {
          return;
        }
        const volume = Number.isFinite(bar.volume) ? bar.volume : Number.parseFloat(bar.volume);
        const tradeCount = Number.isFinite(bar.trade_count) ? bar.trade_count : Number.parseInt(bar.trade_count, 10);
        const entry = { time, open, high, low, close };
        if (Number.isFinite(volume)) {
          entry.volume = volume;
        }
        if (Number.isFinite(tradeCount)) {
          entry.trade_count = tradeCount;
        }
        cleaned.push(entry);
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
          if (this._syncingIndicator || !range || !this.indicatorSeries.length) return;
          this._syncingIndicator = true;
          indicatorScale.setVisibleRange(range);
          this._syncingIndicator = false;
        });
        indicatorScale.subscribeVisibleTimeRangeChange((range) => {
          if (this._syncingIndicator || !range || !this.indicatorSeries.length) return;
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

  function setTradeSymbol(symbol) {
    if (tradeSymbolEl) {
      tradeSymbolEl.textContent = symbol || '--';
    }
  }

  function setTradeStatus(message, tone = 'neutral') {
    if (!tradeStatusEl) return;
    tradeStatusEl.textContent = message || '';
    tradeStatusEl.classList.remove('is-success', 'is-error');
    if (tone === 'success') {
      tradeStatusEl.classList.add('is-success');
    } else if (tone === 'error') {
      tradeStatusEl.classList.add('is-error');
    }
  }


  function applyTradeModeState(payload) {
    if (!payload || typeof payload !== 'object') return;
    if (typeof payload.mode === 'string') {
      tradeMode = payload.mode.toLowerCase();
    }
    if (typeof payload.execution_enabled === 'boolean') {
      tradeExecutionEnabled = payload.execution_enabled;
    }
    tradeModeError = '';
    updateTradeModeUI();
  }

  function updateTradeModeUI() {
    if (tradeModeError && tradeModeStatus) {
      tradeModeStatus.textContent = tradeModeError;
      tradeModeStatus.classList.remove('is-live');
      return;
    }
    if (tradeModeButtons.length) {
      tradeModeButtons.forEach((btn) => {
        const btnMode = btn.dataset.mode;
        const isActive = btnMode === tradeMode;
        btn.classList.toggle('is-active', isActive);
        btn.classList.toggle('is-live', isActive && tradeMode === 'live');
        btn.disabled = tradeModeBusy;
      });
    }
    if (!tradeModeStatus) return;
    if (!tradeExecutionEnabled) {
      tradeModeStatus.textContent = TEXT.tradeModeExecutionOff;
      tradeModeStatus.classList.remove('is-live');
      return;
    }
    const isLive = tradeMode === 'live';
    tradeModeStatus.textContent = isLive ? TEXT.tradeModeLive : TEXT.tradeModePaper;
    tradeModeStatus.classList.toggle('is-live', isLive);
  }

  async function fetchTradeMode() {
    if (!tradeModeEndpoint || (!tradeModeStatus && !tradeModeButtons.length)) return;
    try {
      const response = await fetch(tradeModeEndpoint, {
        headers: { Accept: 'application/json' },
        credentials: 'same-origin',
      });
      if (!response.ok) return;
      const payload = await response.json();
      if (payload && payload.ok) {
        applyTradeModeState(payload);
      }
    } catch (err) {
      // silent
    }
  }

  async function setTradeMode(mode) {
    if (!mode || tradeModeBusy) return;
    const normalized = mode.toLowerCase();
    if (normalized === tradeMode) return;
    if (normalized === 'live' && !window.confirm(TEXT.tradeModeConfirm)) {
      return;
    }
    tradeModeBusy = true;
    tradeModeError = '';
    if (tradeModeStatus) {
      tradeModeStatus.textContent = TEXT.tradeModeUpdating;
    }
    updateTradeModeUI();
    try {
      const response = await fetch(tradeModeEndpoint, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        credentials: 'same-origin',
        body: JSON.stringify({ mode: normalized }),
      });
      let payload = {};
      try {
        payload = await response.json();
      } catch (parseErr) {
        payload = { message: response.statusText || TEXT.tradeModeFailed };
      }
      if (!response.ok || !payload.ok) {
        throw new Error(payload.message || TEXT.tradeModeFailed);
      }
      applyTradeModeState(payload);
    } catch (err) {
      tradeModeError = err && err.message ? err.message : TEXT.tradeModeFailed;
    } finally {
      tradeModeBusy = false;
      updateTradeModeUI();
    }
  }

  async function submitTrade(side) {
    const symbol = (detailSymbol || (detailSymbolEl && detailSymbolEl.textContent) || '').trim();
    if (!symbol || symbol === '--') {
      setTradeStatus(TEXT.tradeMissingSymbol, 'error');
      return;
    }
    const qtyVal = tradeQtyInput ? Number(tradeQtyInput.value) : 0;
    if (!qtyVal || qtyVal <= 0) {
      setTradeStatus(TEXT.tradeMissingQty, 'error');
      return;
    }
    setTradeStatus(TEXT.tradeSubmitting);
    try {
      const response = await fetch(orderEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        body: JSON.stringify({
          symbol,
          side,
          qty: qtyVal,
        }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        setTradeStatus(buildOrderErrorMessage(payload && payload.message ? payload.message : ''), 'error');
        return;
      }
      setTradeStatus(TEXT.tradeSuccess, 'success');
    } catch (err) {
      setTradeStatus(TEXT.tradeFailed, 'error');
    }
  }

  if (tradeBuyBtn) {
    tradeBuyBtn.addEventListener('click', () => submitTrade('buy'));
  }
  if (tradeSellBtn) {
    tradeSellBtn.addEventListener('click', () => submitTrade('sell'));
  }
  if (tradeModeButtons.length) {
    tradeModeButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        if (!mode) return;
        setTradeMode(mode);
      });
    });
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
      rankSort = 'default';
      rankItemsBase = [];
      rankItems = [];
      updateSortIndicator();
      updateStatusContext();
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

  if (statusRefresh) {
    statusRefresh.addEventListener('click', () => {
      triggerRefresh();
    });
  }

  if (timezoneToggle) {
    timezoneToggle.addEventListener('click', (event) => {
      const button = event.target.closest('button[data-tz]');
      if (!button) return;
      const nextMode = button.dataset.tz;
      setTimezoneMode(nextMode);
    });
  }

  if (autoRefreshToggle) {
    autoRefreshToggle.addEventListener('click', (event) => {
      const button = event.target.closest('button[data-interval]');
      if (!button) return;
      const nextValue = parseInt(button.dataset.interval, 10);
      if (!Number.isFinite(nextValue)) return;
      setAutoRefreshInterval(nextValue);
    });
  }

  if (autoRefreshPause) {
    autoRefreshPause.addEventListener('click', () => {
      if (!autoRefreshMs) return;
      if (autoRefreshPaused) {
        resumeAutoRefresh({ manual: true });
      } else {
        pauseAutoRefresh({ manual: true });
      }
    });
  }

  if (snapshotDetails && snapshotDetailsPanel) {
    snapshotDetails.addEventListener('click', () => {
      const next = !snapshotDetailsPanel.hidden;
      snapshotDetailsPanel.hidden = next;
    });
  }

  if (snapshotRetry) {
    snapshotRetry.addEventListener('click', () => {
      triggerRefresh();
    });
  }

  function getStatusLabel(state) {
    const labels = TEXT.statusLabels || {};
    return labels[state] || labels.ready || '';
  }

  function getStatusMessage(state) {
    const messages = TEXT.statusMessages || {};
    return messages[state] || messages.ready || '';
  }

  function applyStatusState(nextState, { forceMessage = false } = {}) {
    currentStatusState = nextState;
    if (statusSection) {
      statusSection.dataset.state = nextState || '';
    }
    if (statusState) {
      statusState.textContent = getStatusLabel(nextState);
    }
    if (statusText && (forceMessage || !statusMessageOverride)) {
      statusText.textContent = getStatusMessage(nextState);
    }
  }

  function deriveStatusState() {
    if (isListLoading) return 'refreshing';
    if (snapshotState && snapshotState.state === 'snapshot_building') return 'snapshot_building';
    if (snapshotState && snapshotState.state === 'partial_ready') return 'partial_ready';
    if (!lastStatusGeneratedAt) return 'stale';
    return 'ready';
  }

  function refreshStatusState({ forceMessage = false } = {}) {
    const derived = deriveStatusState();
    const force =
      forceMessage ||
      derived === 'partial_ready' ||
      derived === 'snapshot_building' ||
      derived === 'stale';
    applyStatusState(derived, { forceMessage: force });
  }

  function setStatus(text, { forceState = null, forceMessage = false } = {}) {
    statusMessageOverride = text || '';
    if (statusText && text) {
      statusText.textContent = text;
    }
    if (forceState) {
      applyStatusState(forceState, { forceMessage });
      return;
    }
    refreshStatusState();
  }

  function formatDisplayTime(value) {
    return formatTimestamp(value, timezoneMode, locale);
  }

  function setStatusUpdated(value) {
    lastStatusGeneratedAt = value || '';
    if (!statusUpdated) return;
    const formatted = value ? formatDisplayTime(value) : '';
    statusUpdated.textContent = formatted || '';
    if (statusUpdated.tagName === 'TIME') {
      statusUpdated.setAttribute('datetime', value || '');
    }
  }

  function updateTimezoneToggle() {
    if (!timezoneToggle) return;
    timezoneToggle.querySelectorAll('button[data-tz]').forEach((btn) => {
      const isActive = btn.dataset.tz === timezoneMode;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  }

  function updateAutoRefreshToggle() {
    if (!autoRefreshToggle) return;
    autoRefreshToggle.querySelectorAll('button[data-interval]').forEach((btn) => {
      const value = parseInt(btn.dataset.interval, 10);
      const isActive = Number.isFinite(value) && value === autoRefreshMs;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
    const enabled = Boolean(autoRefreshMs);
    if (autoRefreshPause) {
      autoRefreshPause.disabled = !enabled;
      autoRefreshPause.classList.toggle('is-disabled', !enabled);
      autoRefreshPause.textContent = autoRefreshPaused || autoRefreshSuspended ? TEXT.resume : TEXT.pause;
    }
    updateAutoRefreshCountdown();
  }

  function updateAutoRefreshCountdown() {
    if (!autoRefreshCountdown) return;
    if (!autoRefreshMs) {
      autoRefreshCountdown.textContent = `${TEXT.countdownPrefix}: —`;
      return;
    }
    if (autoRefreshPaused || autoRefreshSuspended) {
      autoRefreshCountdown.textContent = `${TEXT.countdownPrefix}: ${TEXT.paused}`;
      return;
    }
    if (!autoRefreshNextAt) {
      autoRefreshCountdown.textContent = `${TEXT.countdownPrefix}: —`;
      return;
    }
    const remaining = Math.max(autoRefreshNextAt - Date.now(), 0);
    const seconds = Math.max(Math.ceil(remaining / 1000), 0);
    autoRefreshCountdown.textContent = `${TEXT.countdownPrefix}: ${seconds}s`;
  }

  function refreshTimeDisplays() {
    if (lastStatusGeneratedAt) {
      setStatusUpdated(lastStatusGeneratedAt);
    }
    if (lastSnapshotMeta) {
      setSnapshotStatus(lastSnapshotMeta, lastSnapshotKey);
    }
    if (detailChartCache && detailChartCache.payload) {
      updateDetailTimes(detailChartCache.payload);
    }
  }

  function setTimezoneMode(nextMode, { persist = true } = {}) {
    if (nextMode !== 'utc' && nextMode !== 'local') return;
    timezoneMode = nextMode;
    if (persist) {
      savePreference(PREF_TIMEZONE_KEY, nextMode);
    }
    updateTimezoneToggle();
    refreshTimeDisplays();
    if (detailManager) {
      detailManager.setTimezone(timezoneMode);
    }
  }

  function setAutoRefreshInterval(nextInterval, { persist = true } = {}) {
    if (!AUTO_REFRESH_OPTIONS.includes(nextInterval)) return;
    autoRefreshMs = nextInterval;
    autoRefreshPaused = false;
    autoRefreshSuspended = false;
    autoRefreshRemainingMs = 0;
    autoRefreshNextAt = 0;
    if (persist) {
      savePreference(PREF_AUTO_REFRESH_KEY, nextInterval);
    }
    updateAutoRefreshToggle();
    scheduleAutoRefresh();
  }

  function triggerRefresh() {
    if (isListLoading) {
      setStatus(TEXT.refreshSkipped, { forceState: 'refreshing', forceMessage: true });
      return;
    }
    const value = (searchInput && searchInput.value.trim()) || '';
    if (currentListType === 'all') {
      loadAllStocks({ query: value, page: 1, letter: allStocksLetter });
    } else {
      loadData(value, { listType: currentListType, keepListType: true, openDetail: false });
    }
  }

  function getListLabelText(listType) {
    if (TEXT.listLabels && TEXT.listLabels[listType]) {
      return TEXT.listLabels[listType];
    }
    return getListLabel(listType);
  }

  function buildContextLabel(listType, timeframeKey) {
    const listLabel = getListLabelText(listType);
    if (listType === 'all') {
      return listLabel;
    }
    const tfLabel = TEXT.timeframes[timeframeKey] || timeframeKey;
    return `${listLabel} · ${tfLabel}`;
  }

  function updateStatusContext() {
    const label = buildContextLabel(currentListType, currentTimeframe);
    if (statusContext) {
      statusContext.textContent = label;
    }
    if (rankContext) {
      if (currentListType === 'all') {
        rankContext.textContent = label;
      } else {
        const tfLabel = TEXT.timeframes[currentTimeframe] || currentTimeframe;
        rankContext.textContent = `${getListLabelText(currentListType)}（${tfLabel}）`;
      }
    }
    updateSortSummary();
  }

  function isListVisible() {
    return currentView === 'list';
  }

  function isChartContainerReady() {
    if (!detailChartEl) return false;
    if (detailChartEl.offsetParent === null) return false;
    const width = detailChartEl.clientWidth || 0;
    const height = detailChartEl.clientHeight || 0;
    return width > 24 && height > 24;
  }

  function scheduleChartInit() {
    if (chartInitTimer || !detailChartEl) return;
    let attempts = 0;
    const tryInit = () => {
      attempts += 1;
      if (ensureDetailChart()) {
        chartInitTimer = null;
        if (pendingChartRender) {
          const { symbol, rangeKey, payload, bars, chartPayload } = pendingChartRender;
          pendingChartRender = null;
          applyDetailPayload(symbol, rangeKey, payload, chartPayload || { bars }, true);
        }
        return;
      }
      if (attempts < 10) {
        chartInitTimer = setTimeout(tryInit, 120);
      } else {
        chartInitTimer = null;
      }
    };
    chartInitTimer = setTimeout(tryInit, 0);
  }

  function buildSortSummary() {
    const header = getListHeader(currentListType);
    const stateLabel =
      rankSort === 'asc' ? TEXT.sortAsc : rankSort === 'desc' ? TEXT.sortDesc : TEXT.sortDefault;
    return `${TEXT.sortedBy} ${header} (${stateLabel})`;
  }

  function updateSortSummary() {
    if (!rankSortSummaries.length) return;
    const summary = buildSortSummary();
    rankSortSummaries.forEach((el) => {
      if (el) el.textContent = summary;
    });
  }

  function updateSortIndicator() {
    if (rankingSortButton) {
      const isActive = rankSort !== 'default';
      rankingSortButton.classList.toggle('is-active', isActive);
      if (rankingChangeHeader) {
        const sortState = rankSort === 'asc' ? 'ascending' : rankSort === 'desc' ? 'descending' : 'none';
        rankingChangeHeader.setAttribute('aria-sort', sortState);
      }
      if (rankingSortIcon) {
        if (rankSort === 'asc') {
          rankingSortIcon.textContent = '↑';
        } else if (rankSort === 'desc') {
          rankingSortIcon.textContent = '↓';
        } else {
          rankingSortIcon.textContent = '↕';
        }
      }
    }
    updateSortSummary();
  }

  function updateRankDescription(type) {
    if (!rankDesc) return;
    const descMap = {
      gainers: rankDesc.dataset.descGainers,
      losers: rankDesc.dataset.descLosers,
      most_active: rankDesc.dataset.descMostActive,
      top_turnover: rankDesc.dataset.descTopTurnover,
      all: rankDesc.dataset.descAll,
    };
    const nextText = descMap[type] || rankDesc.dataset.descGainers || '';
    if (nextText) {
      rankDesc.textContent = nextText;
    }
  }

  function setActiveListType(type) {
    currentListType = type;
    rankSort = 'default';
    rankItemsBase = [];
    rankItems = [];
    updateSortIndicator();
    rankTabs.forEach((tab) => {
      const tabType = tab.dataset.list || '';
      const isActive = tabType === type;
      tab.classList.toggle('is-active', isActive);
      tab.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
    updateRankDescription(type);
    updateStatusContext();
    if (rankingsSection && allStocksSection) {
      const showAll = type === 'all';
      rankingsSection.hidden = showAll;
      allStocksSection.hidden = !showAll;
    }
  }

  function resetRankPaging(pageSize) {
    rankOffset = 0;
    rankNextOffset = null;
    rankHasMore = false;
    rankIsLoadingMore = false;
    if (typeof pageSize === 'number' && Number.isFinite(pageSize)) {
      rankPageSize = pageSize;
    }
    if (rankSentinel) {
      rankSentinel.hidden = true;
    }
  }

  function updateRankPaging(payload, items, offset, pageSize) {
    const responseOffset = Number(payload && payload.offset);
    const responseLimit = Number(payload && payload.limit);
    const nextOffset =
      payload && Number.isFinite(payload.next_offset)
        ? Number(payload.next_offset)
        : (Number.isFinite(responseOffset) ? responseOffset : offset) + (items ? items.length : 0);
    const hasMore = Boolean(payload && payload.has_more);
    rankOffset = Number.isFinite(responseOffset) ? responseOffset : offset;
    rankNextOffset = hasMore ? nextOffset : null;
    rankHasMore = hasMore;
    if (Number.isFinite(responseLimit)) {
      rankPageSize = responseLimit;
    } else if (Number.isFinite(pageSize)) {
      rankPageSize = pageSize;
    }
    if (rankSentinel) {
      rankSentinel.hidden = !rankHasMore;
    }
  }

  function applyRankSort(items) {
    if (!Array.isArray(items)) return [];
    if (rankSort === 'default') return items.slice();
    const direction = rankSort === 'asc' ? 1 : -1;
    return items
      .slice()
      .sort((left, right) => {
        const leftValue = resolveMetricValue(left, currentListType);
        const rightValue = resolveMetricValue(right, currentListType);
        if (leftValue === null && rightValue === null) return 0;
        if (leftValue === null) return 1;
        if (rightValue === null) return -1;
        return (leftValue - rightValue) * direction;
      });
  }

  function updateRankItems(items, isAppend) {
    const incoming = Array.isArray(items) ? items : [];
    const mergedResult = mergeRankItems(isAppend ? rankItemsBase : [], incoming);
    rankItemsBase = mergedResult.merged;
    rankItems = applyRankSort(rankItemsBase);
    return { merged: rankItems, appended: mergedResult.appended };
  }

  function maybeLoadMoreRankings() {
    if (rankIsLoadingMore || !rankHasMore) return;
    if (!isListVisible() || currentListType === 'all') return;
    if (!rankNextOffset && rankNextOffset !== 0) return;
    loadData('', {
      listType: currentListType,
      keepListType: true,
      openDetail: false,
      append: true,
      offset: rankNextOffset,
      pageSize: rankPageSize,
    });
  }

  function setupRankObserver() {
    if (!rankSentinel) return;
    if (rankObserver) {
      rankObserver.disconnect();
      rankObserver = null;
    }
    rankObserver = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        maybeLoadMoreRankings();
      },
      { rootMargin: '200px' },
    );
    rankObserver.observe(rankSentinel);
  }

  function setRetryingState(retryAfterSeconds) {
    const delaySeconds = Math.min(Math.max(parseInt(retryAfterSeconds, 10) || 3, 2), 30);
    if (statusSection) {
      statusSection.classList.add('is-retrying');
    }
    setStatus(TEXT.retrying, { forceState: 'refreshing', forceMessage: true });
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

  function buildSnapshotPrefix(timeframeKey) {
    const key = typeof timeframeKey === 'string' ? timeframeKey : '';
    const label = key && key !== '1d' ? (TEXT.timeframes && TEXT.timeframes[key]) || key : '';
    return label ? `${TEXT.snapshotPrefix}(${label})` : TEXT.snapshotPrefix;
  }

  function formatSnapshotStatus(meta, timeframeKey) {
    const prefix = buildSnapshotPrefix(timeframeKey);
    const result = {
      state: 'ready',
      text: `${prefix}：${TEXT.snapshotUnknown}`,
      warning: false,
      warningText: TEXT.partialData,
      warningDetail: TEXT.partialDetail,
      progressValue: 0,
      showProgress: false,
    };
    if (!meta || typeof meta !== 'object') {
      result.state = 'stale';
      return result;
    }
    const progress = meta.progress;
    if (progress && progress.status === 'running') {
      const completed = Number(progress.chunks_completed);
      const total = Number(progress.total_chunks);
      const totalSymbols = Number(progress.total_symbols);
      const chunkSize = Number(progress.chunk_size);
      let progressText = '';
      let progressValue = 0;
      if (Number.isFinite(completed) && Number.isFinite(chunkSize) && Number.isFinite(totalSymbols) && totalSymbols > 0) {
        const completedSymbols = Math.min(totalSymbols, completed * chunkSize);
        progressValue = Math.round((completedSymbols / totalSymbols) * 100);
        progressText = ` ${completedSymbols}/${totalSymbols} (${progressValue}%)`;
      } else if (Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
        progressValue = Math.round((completed / total) * 100);
        progressText = ` ${completed}/${total} (${progressValue}%)`;
      } else if (Number.isFinite(completed)) {
        progressText = ` ${completed}`;
      }
      result.text = `${prefix}：${TEXT.snapshotRunning}${progressText}`;
      result.progressValue = progressValue;
      result.showProgress = true;
      if (progressValue >= 100) {
        result.state = 'partial_ready';
        result.warning = true;
      } else {
        result.state = 'snapshot_building';
      }
      return result;
    }
    if (meta.error) {
      const errorMsg = meta.error && meta.error.error ? String(meta.error.error) : '';
      const suffix = errorMsg ? ` · ${errorMsg}` : '';
      result.text = `${prefix}：${TEXT.snapshotError}${suffix}`;
      result.state = 'partial_ready';
      result.warning = true;
      result.progressValue = 0;
      return result;
    }
    const timeframes = meta.timeframes && typeof meta.timeframes === 'object' ? meta.timeframes : null;
    let latest = null;
    if (timeframeKey && timeframeKey !== '1d' && timeframes && timeframes[timeframeKey]) {
      latest = timeframes[timeframeKey];
      if (latest && latest.status && latest.status !== 'complete') {
        result.text = `${prefix}：${TEXT.snapshotError}`;
        result.state = 'partial_ready';
        result.warning = true;
        return result;
      }
    }
    if (!latest && meta.latest) {
      latest = meta.latest;
    }
    const generatedAt = latest && latest.generated_at ? String(latest.generated_at) : '';
    if (generatedAt) {
      result.text = `${prefix}：${TEXT.snapshotUpdated} ${formatDisplayTime(generatedAt)}`;
      result.progressValue = 100;
      result.showProgress = false;
      result.state = 'ready';
      return result;
    }
    result.state = 'stale';
    return result;
  }

  function setSnapshotStatus(meta, timeframeKey) {
    if (!snapshotText) return;
    lastSnapshotMeta = meta;
    lastSnapshotKey = timeframeKey;
    const result = formatSnapshotStatus(meta, timeframeKey);
    snapshotText.textContent = result.text;
    snapshotText.classList.toggle('is-error', result.state === 'partial_ready');
    if (snapshotProgressFill) {
      const progressBar = snapshotProgressFill.closest('.market-progress-bar');
      const clamped = Math.min(Math.max(Number(result.progressValue) || 0, 0), 100);
      snapshotProgressFill.style.width = `${clamped}%`;
      if (progressBar) {
        progressBar.setAttribute('aria-valuenow', String(clamped));
      }
    }
    if (snapshotProgress) {
      snapshotProgress.hidden = !result.showProgress;
    }
    if (snapshotWarning) {
      const warningText = result.warningText || TEXT.partialData;
      if (snapshotWarningText) {
        snapshotWarningText.textContent = warningText;
      } else {
        snapshotWarning.textContent = warningText;
      }
      if (snapshotDetailsPanel) {
        snapshotDetailsPanel.hidden = true;
        snapshotDetailsPanel.textContent = result.warningDetail || TEXT.partialDetail;
      }
      snapshotWarning.hidden = !result.warning;
    }
    if (statusExpanded) {
      statusExpanded.hidden = !(result.showProgress || result.warning);
    }
    snapshotState = result;
    refreshStatusState();
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
      if (isRankingRow && (currentListType === 'gainers' || currentListType === 'losers') && currentTimeframe === '1d') {
        changeCell.textContent = formatChange(changePct);
        const meta = getListMeta(currentListType);
        applyChangeState(changeCell, changePct, meta && meta.invert);
      }
    });
    const chartRealtimeActive = isChartRealtimeActive();
    if (detailSymbol && detailSymbol === symbol && !chartRealtimeActive) {
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
      if (!isChartRealtimeActive()) return;
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
    const targetView = mode || 'list';
    chartVisible = targetView === 'chart';
    currentView = targetView;
    if (viewList) viewList.classList.toggle('is-active', currentView === 'list');
    if (viewDetail) viewDetail.classList.toggle('is-active', currentView === 'detail');
    if (viewChart) viewChart.classList.toggle('is-active', currentView === 'chart');
    if (marketPage) {
      marketPage.classList.remove('view-list', 'view-detail', 'view-chart');
      marketPage.classList.add(`view-${currentView}`);
    }

    if (paneTabs.length) {
      paneTabs.forEach((tab) => {
        const matches = tab.dataset.view === currentView;
        tab.classList.toggle('is-active', matches);
        tab.setAttribute('aria-selected', matches ? 'true' : 'false');
        tab.setAttribute('tabindex', matches ? '0' : '-1');
      });
    }

    if (chartVisible) {
      resizeDetailChart();
      connectChartSocket();
      if (detailSymbol) {
        setChartSocketSymbol(detailSymbol);
      }
    }
    if (isListVisible()) {
      scheduleAutoRefresh();
    }
    if (!chartVisible) {
      clearLiveWait();
      stopLiveQuotePolling();
      disconnectChartSocket();
    }
  }

  function updateDetailHero(symbol, payload, bars) {
    if (detailSymbolEl) {
      detailSymbolEl.textContent = symbol || '--';
    }
    setTradeSymbol(symbol || '--');
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
    updateDetailTimes(payload);
  }

  function updateDetailTimes(payload, options = {}) {
    const { updateSubtitle = true } = options;
    const raw = payload && payload.generated_at ? payload.generated_at : '';
    const display = raw ? formatDisplayTime(raw) : '';
    if (detailMetaEl) {
      detailMetaEl.textContent = display ? `${TEXT.updatedLabel} ${display}` : '';
    }
    if (detailUpdated) {
      detailUpdated.textContent = display ? `${TEXT.updatedLabel} ${display}` : '';
    }
    if (detailSubtitle && updateSubtitle) {
      if (display) {
        detailSubtitle.textContent = langPrefix === 'zh' ? `更新时间：${display}` : `Updated: ${display}`;
      } else {
        detailSubtitle.textContent = '';
      }
    }
  }

  function hasCachedDetailInfo(symbol, rangeKey) {
    return (
      detailInfoCache &&
      detailInfoCache.symbol === symbol &&
      detailInfoCache.range === rangeKey &&
      detailInfoCache.payload &&
      typeof detailInfoCache.payload === 'object'
    );
  }

  function hasCachedChart(symbol, rangeKey, intervalKey) {
    return (
      detailChartCache &&
      detailChartCache.symbol === symbol &&
      detailChartCache.range === rangeKey &&
      detailChartCache.interval === intervalKey &&
      Array.isArray(detailChartCache.bars) &&
      detailChartCache.bars.length > 0
    );
  }

  function storeDetailInfoCache(symbol, rangeKey, payload) {
    detailInfoCache = {
      symbol,
      range: rangeKey,
      payload: payload && typeof payload === 'object' ? payload : null,
    };
  }

  function storeDetailChartCache(symbol, rangeKey, intervalKey, bars, payload) {
    detailChartCache = {
      symbol,
      range: rangeKey,
      interval: intervalKey,
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
    if (!isChartContainerReady()) return false;
    try {
      detailManager = new ChartManager({
        container: detailChartEl,
        indicatorContainer: detailIndicatorEl,
        langPrefix,
        timezoneMode,
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
      if (ready) {
        const spec = resolveIntervalSpec(detailInterval);
        if (spec) {
          detailManager.setIntervalSpec(spec, { preserveData: true });
        }
        detailManager.setTimezone(timezoneMode);
      }
      return ready;
    } catch (error) {
      if (detailManager) {
        detailManager = null;
      }
      setDetailStatus(TEXT.detailError, true);
      return false;
    }
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

  function applyDetailPayload(symbol, rangeKey, infoPayload, chartPayload, renderChart) {
    const payload = infoPayload || {};
    const chartData = chartPayload || {};
    const bars = Array.isArray(chartData.bars) ? chartData.bars : [];
    updateDetailHero(symbol, payload, bars);
    updateInsightPanels(payload || {});
    const intervalKey =
      (chartData.interval && chartData.interval.key) ||
      (chartData.interval_key ? chartData.interval_key : detailInterval);
    detailBarIntervalSec = resolveBarIntervalSeconds(intervalKey) || inferBarIntervalSeconds(bars);
    updateDetailTimeScale(intervalKey);
    if (!renderChart) return;
    if (!ensureDetailChart() || !detailManager) {
      pendingChartRender = { symbol, rangeKey, payload, bars, chartPayload: chartData };
      scheduleChartInit();
      return;
    }
    const intervalSpec = resolveIntervalSpec(intervalKey);
    detailManager.setData(bars, { intervalSpec });
    if (detailSource) {
      const sourceLabel = TEXT.sourceLabels[chartData.data_source] || TEXT.sourceLabels.unknown || '';
      const sourceText = sourceLabel ? `${TEXT.sourcePrefix || ''}${sourceLabel}` : '';
      const rawHint =
        intervalSpec && (intervalSpec.unit === 'tick' || intervalSpec.unit === 'second')
          ? langPrefix === 'zh'
            ? ' · 成交明细为原始数据'
            : ' · Trades are raw'
          : '';
      detailSource.textContent = `${sourceText}${rawHint}`;
    }
    updateDetailTimes(chartData);
    if (detailTitle) {
      const intervalLabel = getIntervalLabel(intervalKey);
      detailTitle.textContent = intervalLabel ? `${symbol} · ${intervalLabel}` : symbol;
    }
    if (chartData.downgrade_message) {
      setDetailStatus(TEXT.detailDowngraded(chartData.downgrade_message));
    } else if (chartData.window_limited) {
      setDetailStatus(TEXT.detailWindowLimited);
    } else {
      setDetailStatus('');
    }
  }

  async function fetchChartBars(symbol, rangeKey, intervalKey) {
    const endpointBase = chartApiUrl || '/api/market/chart/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({
      symbol,
      range: rangeKey,
      interval: intervalKey,
    });
    const response = await fetch(`${endpoint}?${params.toString()}`, {
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      credentials: 'same-origin',
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || TEXT.detailError);
    }
    return payload;
  }

  async function loadDetailData(symbol, rangeKey, options = {}) {
    if (!symbol) return;
    const { renderChart = true, allowCache = true } = options;
    const intervalKey = normalizeIntervalKey(detailInterval) || resolveDefaultInterval(rangeKey);
    if (!intervalKey) {
      setDetailStatus(TEXT.intervalInvalid, true);
      return;
    }
    if (detailRetryTimer) {
      clearTimeout(detailRetryTimer);
      detailRetryTimer = null;
    }
    clearLiveWait();
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

    let infoPayload = null;
    let chartPayload = null;

    if (allowCache && hasCachedDetailInfo(symbol, rangeKey)) {
      infoPayload = detailInfoCache.payload || null;
    }
    if (allowCache && hasCachedChart(symbol, rangeKey, intervalKey)) {
      chartPayload = detailChartCache.payload || null;
    }

    const infoPromise = infoPayload
      ? Promise.resolve(infoPayload)
      : (async () => {
          const endpointBase = apiUrl || '/api/market/';
          const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
          const params = new URLSearchParams({
            detail: '1',
            symbol,
            range: rangeKey,
          });
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
            throw new Error('rate_limited');
          }
          if (!response.ok) {
            throw new Error(payload.error || TEXT.detailError);
          }
          storeDetailInfoCache(symbol, rangeKey, payload);
          return payload;
        })();

    const chartPromise = chartPayload
      ? Promise.resolve(chartPayload)
      : fetchChartBars(symbol, rangeKey, intervalKey)
          .then((payload) => {
            if (payload && payload.downgrade_to) {
              const normalized = normalizeIntervalKey(payload.downgrade_to);
              if (normalized && normalized !== detailInterval) {
                setDetailInterval(normalized, { persist: true, skipReload: true });
              }
            }
            storeDetailChartCache(symbol, rangeKey, intervalKey, payload.bars || [], payload);
            return payload;
          });

    try {
      const [infoResult, chartResult] = await Promise.allSettled([infoPromise, chartPromise]);
      if (infoResult.status === 'fulfilled') {
        infoPayload = infoResult.value;
      } else if (infoResult.reason && infoResult.reason.message !== 'rate_limited') {
        setDetailStatus(infoResult.reason.message || TEXT.detailError, true);
        updateInsightPanels({});
      }
      if (chartResult.status === 'fulfilled') {
        chartPayload = chartResult.value;
      } else if (chartResult.reason) {
        setDetailStatus(chartResult.reason.message || TEXT.detailError, true);
      }
      if (!chartPayload || !Array.isArray(chartPayload.bars) || !chartPayload.bars.length) {
        setDetailStatus(TEXT.detailEmpty);
        updateInsightPanels(infoPayload || {});
        return;
      }
      applyDetailPayload(symbol, rangeKey, infoPayload || {}, chartPayload, renderChart);
      if (chartVisible) {
        setChartSocketSymbol(symbol);
      }
    } catch (error) {
      if (error && error.message === 'rate_limited') return;
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
    detailRangeButtons.forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.range === rangeKey);
    });
    updateDetailTimeScale(detailInterval);
  }

  function updateDetailTimeScale(intervalKey) {
    if (!detailManager || !detailManager.chart) return;
    const spec = resolveIntervalSpec(intervalKey);
    const isIntraday = Boolean(spec && spec.unit !== 'day');
    const showSeconds = Boolean(spec && (spec.unit === 'second' || spec.unit === 'tick'));
    detailManager.setAxisOptions({ timeVisible: isIntraday, showSeconds });
  }

  const intervalLabelMap = new Map();

  function buildIntervalLabelMap() {
    intervalLabelMap.clear();
    intervalSelectButtons.forEach((btn) => {
      const key = normalizeIntervalKey(btn.dataset.interval);
      if (!key) return;
      const label = (btn.textContent || '').trim();
      if (label) {
        intervalLabelMap.set(key, label);
      }
    });
  }

  function formatIntervalLabel(intervalKey) {
    const spec = resolveIntervalSpec(intervalKey);
    if (!spec) return (intervalKey || '').toUpperCase();
    if (langPrefix === 'zh') {
      const unitLabel = {
        tick: '笔',
        second: '秒',
        minute: '分',
        hour: '小时',
        day: '天',
      }[spec.unit];
      return `${spec.value}${unitLabel || ''}`;
    }
    const suffix = {
      tick: 't',
      second: 's',
      minute: 'm',
      hour: 'h',
      day: 'd',
    }[spec.unit];
    return `${spec.value}${suffix || ''}`;
  }

  function getIntervalLabel(intervalKey) {
    const key = normalizeIntervalKey(intervalKey);
    if (!key) return formatIntervalLabel(intervalKey);
    return intervalLabelMap.get(key) || formatIntervalLabel(key);
  }

  function loadIntervalPrefs() {
    const rawFav = loadPreference(PREF_INTERVAL_FAVORITES_KEY, '[]');
    const rawCustom = loadPreference(PREF_INTERVAL_CUSTOM_KEY, '[]');
    try {
      const parsed = JSON.parse(rawFav);
      intervalFavorites = Array.isArray(parsed) ? parsed.map(normalizeIntervalKey).filter(Boolean) : [];
    } catch (error) {
      intervalFavorites = [];
    }
    try {
      const parsed = JSON.parse(rawCustom);
      customIntervals = Array.isArray(parsed) ? parsed.map(normalizeIntervalKey).filter(Boolean) : [];
    } catch (error) {
      customIntervals = [];
    }
    intervalFavorites = Array.from(new Set(intervalFavorites));
    customIntervals = Array.from(new Set(customIntervals));
  }

  function saveIntervalPrefs() {
    savePreference(PREF_INTERVAL_FAVORITES_KEY, JSON.stringify(intervalFavorites));
    savePreference(PREF_INTERVAL_CUSTOM_KEY, JSON.stringify(customIntervals));
  }

  function renderIntervalFavorites() {
    if (!intervalFavoritesList) return;
    const merged = Array.from(new Set([...(intervalFavorites || []), ...(customIntervals || [])]));
    intervalFavoritesList.innerHTML = '';
    if (!merged.length) {
      if (intervalFavoritesWrap) {
        intervalFavoritesWrap.hidden = true;
      }
      return;
    }
    if (intervalFavoritesWrap) {
      intervalFavoritesWrap.hidden = false;
    }
    merged.forEach((key) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'detail-interval-select';
      btn.dataset.role = 'detail-interval-select';
      btn.dataset.interval = key;
      btn.textContent = getIntervalLabel(key);
      intervalFavoritesList.appendChild(btn);
    });
  }

  function updateIntervalFavButtons() {
    const favoriteSet = new Set(intervalFavorites || []);
    intervalFavButtons.forEach((btn) => {
      const key = normalizeIntervalKey(btn.dataset.interval);
      const isActive = key && favoriteSet.has(key);
      btn.classList.toggle('is-active', Boolean(isActive));
      btn.textContent = isActive ? '★' : '☆';
    });
  }

  function setDetailInterval(nextKey, { persist = true, skipReload = false } = {}) {
    const normalized = normalizeIntervalKey(nextKey);
    if (!normalized) {
      setDetailStatus(TEXT.intervalInvalid, true);
      return;
    }
    detailInterval = normalized;
    if (persist) {
      savePreference(PREF_INTERVAL_KEY, normalized);
    }
    if (intervalCurrent) {
      intervalCurrent.textContent = getIntervalLabel(normalized);
    }
    intervalSelectButtons.forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.interval === normalized);
    });
    renderIntervalFavorites();
    updateIntervalFavButtons();
    updateDetailTimeScale(normalized);
    if (detailManager) {
      const spec = resolveIntervalSpec(normalized);
      if (spec) {
        detailManager.setIntervalSpec(spec, { preserveData: true });
      }
    }
    if (!skipReload && detailSymbol) {
      loadDetailData(detailSymbol, detailRange, { renderChart: true, allowCache: false });
    }
  }

  function toggleIntervalFavorite(key) {
    const normalized = normalizeIntervalKey(key);
    if (!normalized) return;
    if (intervalFavorites.includes(normalized)) {
      intervalFavorites = intervalFavorites.filter((item) => item !== normalized);
    } else {
      intervalFavorites.push(normalized);
    }
    intervalFavorites = Array.from(new Set(intervalFavorites));
    saveIntervalPrefs();
    renderIntervalFavorites();
    updateIntervalFavButtons();
  }

  function addCustomInterval(key) {
    const normalized = normalizeIntervalKey(key);
    if (!normalized) return false;
    if (!customIntervals.includes(normalized)) {
      customIntervals.push(normalized);
    }
    customIntervals = Array.from(new Set(customIntervals));
    saveIntervalPrefs();
    renderIntervalFavorites();
    updateIntervalFavButtons();
    return true;
  }

  let intervalMenuOpen = false;
  let intervalMenuIndex = -1;

  function getIntervalMenuOptions() {
    if (!intervalMenu) return [];
    return Array.prototype.slice.call(intervalMenu.querySelectorAll('[data-role="detail-interval-select"]'));
  }

  function focusIntervalOption(index) {
    const options = getIntervalMenuOptions();
    if (!options.length) return;
    const nextIndex = Math.max(0, Math.min(options.length - 1, index));
    intervalMenuIndex = nextIndex;
    options[nextIndex].focus({ preventScroll: true });
  }

  function openIntervalMenu() {
    if (!intervalMenu || !intervalTrigger) return;
    intervalMenu.hidden = false;
    intervalMenuOpen = true;
    intervalTrigger.setAttribute('aria-expanded', 'true');
    const options = getIntervalMenuOptions();
    if (options.length) {
      const activeIndex = options.findIndex((btn) => btn.dataset.interval === detailInterval);
      focusIntervalOption(activeIndex >= 0 ? activeIndex : 0);
    }
  }

  function closeIntervalMenu() {
    if (!intervalMenu || !intervalTrigger) return;
    intervalMenu.hidden = true;
    intervalMenuOpen = false;
    intervalMenuIndex = -1;
    intervalTrigger.setAttribute('aria-expanded', 'false');
  }

  function toggleIntervalMenu() {
    if (!intervalMenuOpen) {
      openIntervalMenu();
    } else {
      closeIntervalMenu();
    }
  }

  function highlightSelectedRows(symbol) {
    const normalized = normalizeSymbol(symbol);
    if (listContainer) {
      listContainer.querySelectorAll('tr[data-symbol]').forEach((row) => {
        const isActive = row.dataset.symbol === normalized;
        row.classList.toggle('is-selected', isActive);
        row.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });
    }
    if (allStocksBody) {
      allStocksBody.querySelectorAll('tr[data-symbol]').forEach((row) => {
        const isActive = row.dataset.symbol === normalized;
        row.classList.toggle('is-selected', isActive);
        row.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });
    }
  }

  function openDetailPanel(symbol) {
    detailSymbol = symbol;
    setView('detail');
    highlightSelectedRows(symbol);
    if (detailSymbolEl) {
      detailSymbolEl.textContent = symbol;
    }
    setTradeSymbol(symbol);
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
    if (!normalizeIntervalKey(detailInterval)) {
      setDetailInterval(resolveDefaultInterval(detailRange), { persist: true, skipReload: true });
    }
    if (!isChartContainerReady()) {
      scheduleChartInit();
      return;
    }
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

  function isChartRealtimeActive() {
    return Boolean(chartVisible && detailSymbol && chartSocketReady);
  }

  function connectChartSocket() {
    if (!window.WebSocket) return;
    if (chartSocketRetryTimer) {
      clearTimeout(chartSocketRetryTimer);
      chartSocketRetryTimer = null;
    }
    if (chartSocket && (chartSocket.readyState === WebSocket.OPEN || chartSocket.readyState === WebSocket.CONNECTING)) {
      return;
    }
    chartSocket = new WebSocket(chartSocketUrl);
    chartSocket.onopen = () => {
      chartSocketReady = true;
      if (chartSocketSymbol) {
        chartSocket.send(JSON.stringify({ action: 'subscribe', symbol: chartSocketSymbol }));
        scheduleLiveWait();
      }
    };
    chartSocket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        applyChartTradeUpdate(payload);
      } catch (error) {
        return;
      }
    };
    chartSocket.onclose = () => {
      chartSocketReady = false;
      if (chartVisible && detailSymbol) {
        chartSocketRetryTimer = setTimeout(connectChartSocket, 2500);
      }
    };
  }

  function disconnectChartSocket() {
    if (chartSocket) {
      try {
        chartSocket.close();
      } catch (error) {
        // noop
      }
    }
    chartSocket = null;
    chartSocketReady = false;
  }

  function setChartSocketSymbol(symbol) {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return;
    if (chartSocketSymbol && chartSocketSymbol !== normalized && chartSocketReady && chartSocket) {
      chartSocket.send(JSON.stringify({ action: 'unsubscribe', symbol: chartSocketSymbol }));
    }
    chartSocketSymbol = normalized;
    if (!chartSocketReady) {
      connectChartSocket();
      return;
    }
    chartSocket.send(JSON.stringify({ action: 'subscribe', symbol: chartSocketSymbol }));
    lastLiveUpdateAt = 0;
    scheduleLiveWait();
  }

  function applyChartTradeUpdate(update) {
    if (!update || typeof update !== 'object') return;
    const symbol = normalizeSymbol(update.symbol || '');
    if (!symbol || symbol !== detailSymbol) return;
    const price = typeof update.price === 'number' ? update.price : Number.parseFloat(update.price);
    const size = typeof update.size === 'number' ? update.size : Number.parseFloat(update.size);
    const ts = typeof update.ts === 'number' ? update.ts : Number.parseFloat(update.ts);
    if (!Number.isFinite(price)) return;
    if (detailManager) {
      detailManager.applyTradeUpdate({ price, size, ts });
      if (detailPriceEl) {
        detailPriceEl.textContent = price.toFixed(2);
      }
      if (detailChangeEl && detailManager.ohlcData.length > 1) {
        const last = detailManager.ohlcData[detailManager.ohlcData.length - 1];
        const prev = detailManager.ohlcData[detailManager.ohlcData.length - 2];
        const changePct = prev && prev.close ? ((last.close / prev.close) - 1) * 100 : null;
        detailChangeEl.textContent = typeof changePct === 'number' ? formatChange(changePct) : '--';
        detailChangeEl.classList.remove('is-up', 'is-down');
        applyChangeState(detailChangeEl, changePct, false);
      }
    }
    lastLiveUpdateAt = Date.now();
    if (detailStatus && detailStatus.textContent === TEXT.detailLiveWaiting) {
      setDetailStatus(TEXT.detailLive);
    }
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
      cell.colSpan = container === listContainer ? RANKING_COLUMNS : ALL_STOCKS_COLUMNS;
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

  function setRankingLoadingMore(isLoading) {
    if (!listContainer || listContainer.tagName !== 'TBODY') return;
    const existing = listContainer.querySelector('[data-role="ranking-loader"]');
    if (!isLoading) {
      if (existing) {
        existing.remove();
      }
      return;
    }
    if (existing) return;
    const row = document.createElement('tr');
    row.dataset.role = 'ranking-loader';
    const cell = document.createElement('td');
    cell.colSpan = RANKING_COLUMNS;
    cell.className = 'ranking-loading';
    cell.textContent = TEXT.loadingMore || TEXT.loading;
    row.appendChild(cell);
    listContainer.appendChild(row);
  }

  function renderEmpty(container, message) {
    if (!container) return;
    clearListState(container);
    if (container.tagName === 'TBODY') {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = container === listContainer ? RANKING_COLUMNS : ALL_STOCKS_COLUMNS;
      const wrapper = document.createElement('div');
      wrapper.className = 'market-list-empty';
      const title = document.createElement('p');
      title.className = 'market-state-title';
      title.textContent = message;
      const hint = document.createElement('p');
      hint.className = 'market-state-hint';
      hint.textContent = TEXT.emptyHint || '';
      const actions = document.createElement('div');
      actions.className = 'market-state-actions';
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button';
      retryBtn.className = 'market-state-btn';
      retryBtn.dataset.action = 'retry';
      retryBtn.textContent = TEXT.retryAction || 'Retry';
      actions.appendChild(retryBtn);
      wrapper.appendChild(title);
      wrapper.appendChild(hint);
      wrapper.appendChild(actions);
      cell.appendChild(wrapper);
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
      cell.colSpan = container === listContainer ? RANKING_COLUMNS : ALL_STOCKS_COLUMNS;
      const wrapper = document.createElement('div');
      wrapper.className = 'market-error';
      const title = document.createElement('p');
      title.className = 'market-state-title';
      title.textContent = message || TEXT.genericError;
      const hint = document.createElement('p');
      hint.className = 'market-state-hint';
      hint.textContent = TEXT.errorHint || '';
      const actions = document.createElement('div');
      actions.className = 'market-state-actions';
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button';
      retryBtn.className = 'market-state-btn';
      retryBtn.dataset.action = 'retry';
      retryBtn.textContent = TEXT.retryAction || 'Retry';
      actions.appendChild(retryBtn);
      wrapper.appendChild(title);
      wrapper.appendChild(hint);
      wrapper.appendChild(actions);
      cell.appendChild(wrapper);
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

  function formatCompactCurrency(value) {
    const compact = formatCompactNumber(value);
    return compact === '--' ? compact : `$${compact}`;
  }

  function formatMultiple(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return '--';
    }
    if (value >= 10) {
      return `${value.toFixed(1)}x`;
    }
    return `${value.toFixed(2)}x`;
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

  function syncWatchButtons() {
    if (!listContainer) return;
    const buttons = listContainer.querySelectorAll('[data-role="watch-toggle"]');
    if (!buttons.length) return;
    buttons.forEach((btn) => {
      const symbol = (btn.dataset.symbol || '').toUpperCase();
      if (!symbol) return;
      const isWatched = watchPool.includes(symbol);
      btn.classList.toggle('is-active', isWatched);
      btn.setAttribute('aria-pressed', isWatched ? 'true' : 'false');
      btn.setAttribute(
        'aria-label',
        isWatched ? `${TEXT.typeaheadRemove} ${symbol}` : `${TEXT.typeaheadAdd} ${symbol}`,
      );
      btn.textContent = isWatched ? '★' : '☆';
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

  function setQuickTab(target) {
    quickTabs.forEach((tab) => {
      const isActive = tab.dataset.target === target;
      tab.classList.toggle('is-active', isActive);
      tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
      tab.setAttribute('tabindex', isActive ? '0' : '-1');
    });
    quickPanels.forEach((panel) => {
      const isActive = panel.dataset.panel === target;
      panel.classList.toggle('is-active', isActive);
      panel.hidden = !isActive;
    });
  }

  const hasTypeaheadUi = Boolean(searchInput && typeaheadPanel && typeaheadList);
  const TYPEAHEAD_LIMIT = 9;

  function normalizeSymbol(value) {
    return (value || '').toString().trim().toUpperCase();
  }

  function normalizeListType(value) {
    const text = (value || '').toString().trim().toLowerCase();
    if (
      text === 'gainers' ||
      text === 'losers' ||
      text === 'most_active' ||
      text === 'top_turnover' ||
      text === 'all'
    ) {
      return text;
    }
    return 'gainers';
  }

  const LIST_TYPE_META = {
    gainers: {
      header: { zh: '涨跌幅', en: 'Chg%' },
      label: { zh: '涨跌幅', en: 'Chg%' },
      metricKey: 'change_pct_period',
      metricType: 'percent',
      invert: false,
      useMetricLabel: false,
    },
    losers: {
      header: { zh: '涨跌幅', en: 'Chg%' },
      label: { zh: '涨跌幅', en: 'Chg%' },
      metricKey: 'change_pct_period',
      metricType: 'percent',
      invert: false,
      useMetricLabel: false,
    },
    most_active: {
      header: { zh: '成交量', en: 'Volume' },
      label: { zh: '成交量', en: 'Volume' },
      metricKey: 'volume',
      metricType: 'volume',
      invert: false,
      useMetricLabel: true,
    },
    top_turnover: {
      header: { zh: '成交额', en: 'Turnover' },
      label: { zh: '成交额', en: 'Turnover' },
      metricKey: 'dollar_volume',
      metricType: 'turnover',
      invert: false,
      neutral: true,
      useMetricLabel: true,
    },
    all: {
      header: { zh: '涨跌幅', en: 'Chg%' },
      label: { zh: '全部股票', en: 'All Stocks' },
      metricKey: 'change_pct',
      metricType: 'percent',
      invert: false,
      useMetricLabel: true,
    },
  };

  function getListMeta(listType) {
    return LIST_TYPE_META[listType] || LIST_TYPE_META.gainers;
  }

  function coerceNumber(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function resolveMetricValue(item, listType) {
    if (!item || typeof item !== 'object') return null;
    if (listType === 'gainers' || listType === 'losers') {
      return coerceNumber(item.change_pct_period) ?? coerceNumber(item.change_pct_day);
    }
    if (listType === 'most_active') return coerceNumber(item.volume);
    if (listType === 'top_turnover') return coerceNumber(item.dollar_volume);
    if (listType === 'all') return coerceNumber(item.change_pct);
    return null;
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

  function normalizeIntervalKey(value) {
    const raw = (value || '').toString().trim().toLowerCase();
    const match = raw.match(/^(\d+)(t|s|m|h|d)$/);
    if (!match) return null;
    const amount = Math.max(1, parseInt(match[1], 10) || 1);
    const unit = match[2];
    return `${amount}${unit}`;
  }

  function resolveIntervalSpec(value) {
    const key = normalizeIntervalKey(value);
    if (!key) return null;
    const match = key.match(/^(\d+)(t|s|m|h|d)$/);
    if (!match) return null;
    const amount = Math.max(1, parseInt(match[1], 10) || 1);
    const unitKey = match[2];
    const unitMap = {
      t: 'tick',
      s: 'second',
      m: 'minute',
      h: 'hour',
      d: 'day',
    };
    const unit = unitMap[unitKey];
    const secondsMap = { t: null, s: 1, m: 60, h: 3600, d: 86400 };
    const secondsBase = secondsMap[unitKey];
    const seconds = secondsBase ? amount * secondsBase : null;
    return {
      key,
      unit,
      value: amount,
      seconds,
    };
  }

  function resolveDefaultInterval(rangeKey) {
    const key = (rangeKey || '').toString().trim().toLowerCase();
    if (key === '1d' || key === '5d') return '1m';
    if (key === '1mo' || key === '6mo') return '1d';
    return '1m';
  }

  function resolveBarIntervalSeconds(intervalKey) {
    const spec = resolveIntervalSpec(intervalKey);
    if (!spec) return null;
    return spec.seconds || null;
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

  function clearAutoRefreshTimers() {
    if (autoRefreshTimer) {
      clearTimeout(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    if (autoRefreshCountdownTimer) {
      clearInterval(autoRefreshCountdownTimer);
      autoRefreshCountdownTimer = null;
    }
  }

  function pauseAutoRefresh({ manual = false } = {}) {
    if (!autoRefreshMs) return;
    if (manual) {
      autoRefreshPaused = true;
    } else {
      autoRefreshSuspended = true;
    }
    autoRefreshRemainingMs = Math.max(autoRefreshNextAt - Date.now(), autoRefreshMs);
    autoRefreshNextAt = 0;
    clearAutoRefreshTimers();
    updateAutoRefreshToggle();
  }

  function resumeAutoRefresh({ manual = false } = {}) {
    if (!autoRefreshMs) return;
    if (manual) {
      autoRefreshPaused = false;
    } else {
      autoRefreshSuspended = false;
    }
    const delay = autoRefreshRemainingMs || autoRefreshMs;
    autoRefreshRemainingMs = 0;
    scheduleAutoRefresh({ delayMs: delay });
    updateAutoRefreshToggle();
  }

  function scheduleAutoRefresh({ delayMs = null } = {}) {
    clearAutoRefreshTimers();
    if (!isListVisible() || currentListType === 'all') {
      autoRefreshNextAt = 0;
      updateAutoRefreshCountdown();
      return;
    }
    if (!autoRefreshMs) {
      autoRefreshNextAt = 0;
      updateAutoRefreshCountdown();
      return;
    }
    if (autoRefreshPaused || autoRefreshSuspended) {
      autoRefreshNextAt = 0;
      updateAutoRefreshCountdown();
      return;
    }
    const delay = typeof delayMs === 'number' && delayMs >= 0 ? delayMs : autoRefreshMs;
    autoRefreshNextAt = Date.now() + delay;
    updateAutoRefreshCountdown();
    autoRefreshCountdownTimer = window.setInterval(updateAutoRefreshCountdown, 1000);
    autoRefreshTimer = window.setTimeout(() => {
      if (!isListVisible() || currentListType === 'all') return;
      if (isListLoading) {
        scheduleAutoRefresh();
        return;
      }
      const nextQuery = lastRequest.query || '';
      loadData(nextQuery, { listType: currentListType, keepListType: true, openDetail: false });
    }, delay);
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
    const isAppend = Boolean(options.append);
    const pageSizeCandidate = Number.parseInt(options.pageSize ?? options.limit, 10);
    const pageSize = Number.isFinite(pageSizeCandidate) ? pageSizeCandidate : RANK_PAGE_SIZE;
    const offset = Number.isFinite(options.offset) ? Number(options.offset) : 0;
    if (activeListType === 'all') {
      await loadAllStocks({ query: rawQuery, page: 1 });
      return;
    }
    if (!isAppend) {
      resetRankPaging(pageSize);
      lastRequest = { query: normalizedQuery || '', options: { ...options, listType: activeListType } };
    }
    const requestPayload = {
      timeframe: currentTimeframe,
      list: activeListType,
    };
    if (!normalizedQuery) {
      requestPayload.limit = pageSize;
    }
    if (isAppend || offset) {
      requestPayload.offset = isAppend ? offset : 0;
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
      requestPayload.limit = pageSize;
    }

    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (statusSection) {
      statusSection.classList.remove('is-retrying');
    }
    if (!isAppend) {
      updateStatusContext();
      setStatus(`${TEXT.loading} ${TEXT.timeframes[currentTimeframe] || currentTimeframe} ${TEXT.dataSuffix}`, {
        forceState: 'refreshing',
      });
      if (!skipListRender) {
        setListLoading(listContainer);
      }
      showChipSkeleton(recentChips, 3);
      showChipSkeleton(watchlistChips, 4);
      isListLoading = true;
    } else {
      rankIsLoadingMore = true;
      setRankingLoadingMore(true);
    }

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
        if (!normalizedQuery) {
          params.set('limit', String(pageSize));
        }
        if (isAppend || offset) {
          params.set('offset', String(isAppend ? offset : 0));
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
      const incomingItems = Array.isArray(items) ? items.slice() : [];
      const isRankingList = responseListType !== 'all' && listContainer && listContainer.tagName === 'TBODY';
      const shouldUpdateRanking = !skipListRender;
      let rankUpdate = { merged: incomingItems, appended: incomingItems };
      if (shouldUpdateRanking && isRankingList) {
        rankUpdate = updateRankItems(incomingItems, isAppend);
        items = rankUpdate.merged;
        lastRankingTimeframe = payload.timeframe || null;
        lastRankingListType = responseListType;
      }
      if (shouldUpdateRanking) {
        updateRankPaging(payload, incomingItems, isAppend ? offset : 0, pageSize);
      }
      if (!skipListRender) {
        if (isAppend && rankSort === 'default') {
          appendList(listContainer, rankUpdate.appended, payload.timeframe, responseListType);
        } else {
          renderList(listContainer, items, payload.timeframe, responseListType);
          if (!items.length) {
            renderEmpty(listContainer, TEXT.emptySymbol);
          }
        }
        if (items.length) {
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
      if (!isAppend) {
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
        syncWatchButtons();
        syncTypeaheadPools(payload);
        const tfKey = payload.timeframe && payload.timeframe.key;
        const tfLabel = payload.timeframe && (langPrefix === 'zh' ? payload.timeframe.label : payload.timeframe.label_en);
        const statusNotes = [];
        const rankingTimeframe = payload.ranking_timeframe;
        if (
          rankingTimeframe &&
          rankingTimeframe.key &&
          tfKey &&
          rankingTimeframe.key !== tfKey &&
          TEXT.closedFallback
        ) {
          statusNotes.push(TEXT.closedFallback);
        }
        const normalizedSymbol = normalizedQuery || '';
        const actionSymbol = (options.recentTarget || '').toUpperCase();
        if (options.watchAction === 'add' && normalizedSymbol) {
          statusNotes.push(TEXT.watchAdded(normalizedSymbol));
        } else if (options.watchAction === 'remove' && normalizedSymbol) {
          statusNotes.push(TEXT.watchRemoved(normalizedSymbol));
        }
        if (options.recentAction === 'clear') {
          statusNotes.push(TEXT.historyCleared || '');
        } else if (options.recentAction === 'delete' && actionSymbol) {
          const deletedText =
            typeof TEXT.historyDeleted === 'function' ? TEXT.historyDeleted(actionSymbol) : TEXT.historyDeleted;
          statusNotes.push(deletedText);
        }
        setStatus(statusNotes.length ? `${TEXT.updated} · ${statusNotes.join(' · ')}` : TEXT.updated);
        setStatusUpdated(payload.generated_at || TEXT.justNow);
        updateStatusContext();
        const snapshotKey =
          (payload.ranking_timeframe && payload.ranking_timeframe.key) ||
          (payload.timeframe && payload.timeframe.key) ||
          currentTimeframe;
        setSnapshotStatus(payload.snapshot_refresh, snapshotKey);
        setSource(payload.data_source);
        if (normalizedQuery && openDetail) {
          openDetailPanel(normalizedQuery);
        }
      }
    } catch (error) {
      if (isAppend) {
        setRankingLoadingMore(false);
        return;
      }
      renderError(listContainer, error && error.message);
      setStatus(TEXT.statusError, { forceState: 'stale', forceMessage: true });
      setSource('unknown');
      hideChipSkeleton(recentChips);
      hideChipSkeleton(watchlistChips);
    } finally {
      if (isAppend) {
        rankIsLoadingMore = false;
        setRankingLoadingMore(false);
      } else {
        isListLoading = false;
        scheduleAutoRefresh();
        refreshStatusState();
      }
    }
  }

  function getListHeader(listType) {
    const meta = getListMeta(listType);
    return langPrefix === 'zh' ? meta.header.zh : meta.header.en;
  }

  function getListLabel(listType) {
    const meta = getListMeta(listType);
    return langPrefix === 'zh' ? meta.label.zh : meta.label.en;
  }

  function formatMetricValue(value, meta) {
    if (!meta) return '--';
    if (meta.metricType === 'percent') {
      return formatChange(value);
    }
    if (meta.metricType === 'multiple') {
      return formatMultiple(value);
    }
    if (meta.metricType === 'turnover') {
      return formatCompactCurrency(value);
    }
    return formatCompactNumber(value);
  }

  function applyMetricStyle(cell, value, meta) {
    if (!cell || !meta) return;
    cell.classList.remove('is-neutral');
    if (meta.metricType === 'percent' && !meta.neutral) {
      applyChangeState(cell, value, meta.invert);
      return;
    }
    cell.classList.add('is-neutral');
  }

  function buildRankingRow(item, listType) {
    const symbol = (item.symbol || '').toString().toUpperCase();
    if (!symbol) return null;
    const meta = getListMeta(listType);
    const row = document.createElement('tr');
    row.dataset.symbol = normalizeSymbol(symbol);
    row.setAttribute('aria-selected', 'false');
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
    const metricValue = resolveMetricValue(item, listType);
    changeCell.textContent = formatMetricValue(metricValue, meta);
    applyMetricStyle(changeCell, metricValue, meta);
    const watchCell = document.createElement('td');
    watchCell.className = 'col-watch';
    const watchButton = document.createElement('button');
    watchButton.type = 'button';
    watchButton.className = 'watch-toggle';
    watchButton.dataset.role = 'watch-toggle';
    watchButton.dataset.symbol = symbol;
    const isWatched = watchPool.includes(symbol);
    watchButton.classList.toggle('is-active', isWatched);
    watchButton.setAttribute('aria-pressed', isWatched ? 'true' : 'false');
    watchButton.setAttribute(
      'aria-label',
      isWatched ? `${TEXT.typeaheadRemove} ${symbol}` : `${TEXT.typeaheadAdd} ${symbol}`,
    );
    watchButton.textContent = isWatched ? '★' : '☆';
    watchCell.appendChild(watchButton);
    row.appendChild(symbolCell);
    row.appendChild(nameCell);
    row.appendChild(exchangeCell);
    row.appendChild(priceCell);
    row.appendChild(changeCell);
    row.appendChild(watchCell);
    return row;
  }

  function renderList(container, items, timeframe, listType) {
    if (!container) return;
    clearListState(container);
    if (!items.length) {
      renderEmpty(container, TEXT.emptyList);
      return;
    }
    const isTableBody = container.tagName === 'TBODY';
    const meta = getListMeta(listType);

    if (rankingChangeLabel) {
      rankingChangeLabel.textContent = getListHeader(listType);
    } else if (rankingChangeHeader) {
      rankingChangeHeader.textContent = getListHeader(listType);
    }

    if (isTableBody) {
      items.forEach((item) => {
        const row = buildRankingRow(item, listType);
        if (row) {
          container.appendChild(row);
        }
      });
      if (detailSymbol) {
        highlightSelectedRows(detailSymbol);
      }
      syncWatchButtons();
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
        if (meta.useMetricLabel) {
          primaryLabelEl.textContent = getListLabel(listType);
        } else {
          const fallback = TEXT.timeframes[currentTimeframe] || (langPrefix === 'zh' ? '近1月' : '1M');
          const tfLabel = timeframe ? (langPrefix === 'zh' ? timeframe.label : timeframe.label_en) : '';
          const itemLabel = langPrefix === 'zh' ? item.period_label : item.period_label_en;
          primaryLabelEl.textContent = itemLabel || tfLabel || fallback;
        }
      }
      if (primaryEl) {
        primaryEl.classList.remove('is-neutral');
        const metricValue = resolveMetricValue(item, listType);
        if (meta.useMetricLabel) {
          primaryEl.textContent = formatMetricValue(metricValue, meta);
          applyMetricStyle(primaryEl, metricValue, meta);
        } else {
          primaryEl.textContent = formatChange(item.change_pct_period);
          applyChangeState(primaryEl, item.change_pct_period, meta.invert);
        }
      }
      if (dayEl) {
        dayEl.textContent = formatChange(item.change_pct_day);
        applyChangeState(dayEl, item.change_pct_day, meta.invert, true);
      }
      if (windowLabel) {
        if (meta.useMetricLabel) {
          windowLabel.textContent = getListLabel(listType);
        } else {
          const fallback = TEXT.timeframes[currentTimeframe] || (langPrefix === 'zh' ? '近1月' : '1M');
          const tfWindow = timeframe ? (langPrefix === 'zh' ? timeframe.label : timeframe.label_en) : '';
          const itemWindow = langPrefix === 'zh' ? item.period_label : item.period_label_en;
          windowLabel.textContent = itemWindow || tfWindow || fallback;
        }
      }
      if (updatedEl) {
        const timestamps = Array.isArray(item.timestamps) ? item.timestamps : [];
        const stamp = timestamps.length ? formatDisplayTime(timestamps[timestamps.length - 1]) : '';
        updatedEl.textContent = stamp ? `${TEXT.updatedLabel} ${stamp}` : '';
      }
      if (backtestLink) {
        const symbol = item.symbol || '';
        backtestLink.href = `${backtestBase}?ticker=${encodeURIComponent(symbol)}`;
      }
      if (canvas) {
        drawSparkline(canvas, item.series || [], meta.invert);
      }
      const card = fragment.querySelector('.market-card');
      if (card) {
        card.dataset.symbol = normalizeSymbol(item.symbol || '');
        card.dataset.invert = meta.invert ? '1' : '0';
        card.dataset.listType = listType || '';
      }
      container.appendChild(fragment);
    });
  }

  function appendList(container, items, timeframe, listType) {
    if (!container) return;
    if (container.tagName !== 'TBODY') {
      renderList(container, items, timeframe, listType);
      return;
    }
    setRankingLoadingMore(false);
    items.forEach((item) => {
      const row = buildRankingRow(item, listType);
      if (row) {
        container.appendChild(row);
      }
    });
    if (detailSymbol) {
      highlightSelectedRows(detailSymbol);
    }
    syncWatchButtons();
  }

  function renderAllStocksMessage(message) {
    if (!allStocksBody) return;
    allStocksBody.innerHTML = '';
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = ALL_STOCKS_COLUMNS;
    cell.textContent = message;
    row.appendChild(cell);
    allStocksBody.appendChild(row);
  }

  function renderAllStocks(items) {
    if (!allStocksBody) return;
    allStocksBody.innerHTML = '';
    if (!Array.isArray(items) || !items.length) {
      renderEmpty(allStocksBody, TEXT.allStocksEmpty);
      return;
    }
    const seen = new Set();
    items.forEach((item) => {
      const row = document.createElement('tr');
      const symbol = normalizeSymbol(item.symbol || '');
      if (!symbol || seen.has(symbol)) {
        return;
      }
      seen.add(symbol);
      row.dataset.symbol = symbol;
      row.setAttribute('aria-selected', 'false');

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
    if (detailSymbol) {
      highlightSelectedRows(detailSymbol);
    }
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
    updateStatusContext();
    setStatus(TEXT.allStocksLoading, { forceState: 'refreshing' });
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
      setStatus(TEXT.updated);
      setStatusUpdated(TEXT.justNow);
    } catch (error) {
      renderError(allStocksBody, error && error.message ? error.message : TEXT.genericError);
      setStatus(TEXT.statusError, { forceState: 'stale', forceMessage: true });
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
    const retryButton = event.target.closest('[data-action="retry"]');
    if (retryButton) {
      const value = (searchInput && searchInput.value.trim()) || '';
      if (currentListType === 'all') {
        loadAllStocks({ query: value, page: 1, letter: allStocksLetter });
      } else {
        loadData(value, { listType: currentListType, keepListType: true, openDetail: false });
      }
      return;
    }
    const watchButton = event.target.closest('[data-role="watch-toggle"]');
    if (watchButton) {
      event.stopPropagation();
      const symbol = watchButton.dataset.symbol;
      if (!symbol) return;
      const isWatched = watchPool.includes(symbol);
      const action = isWatched ? 'remove' : 'add';
      loadData(symbol, {
        watchAction: action,
        listType: currentListType,
        keepListType: true,
        skipListRender: true,
        openDetail: false,
      });
      return;
    }
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

  if (quickTabs.length) {
    const activeTab = quickTabs.find((tab) => tab.classList.contains('is-active')) || quickTabs[0];
    if (activeTab) {
      setQuickTab(activeTab.dataset.target);
    }
    quickTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        setQuickTab(tab.dataset.target);
      });
    });
  }

  if (recentClear) {
    recentClear.addEventListener('click', () => {
      requestRecentAction('clear');
    });
  }

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

  if (rankingSortButton) {
    rankingSortButton.addEventListener('click', () => {
      if (currentListType === 'all') return;
      if (!rankItemsBase.length) return;
      rankSort = toggleSortState(rankSort);
      updateSortIndicator();
      rankItems = applyRankSort(rankItemsBase);
      if (listContainer && lastRankingTimeframe && lastRankingListType) {
        renderList(listContainer, rankItems, lastRankingTimeframe, lastRankingListType);
      }
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
      const retryButton = event.target.closest('[data-action="retry"]');
      if (retryButton) {
        loadAllStocks({ query: allStocksQuery, page: 1, letter: allStocksLetter });
        return;
      }
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

  if (paneTabs.length) {
    paneTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.view || 'detail';
        if (target === 'list') {
          setView('list');
          return;
        }
        if (target === 'chart') {
          openChartView();
        } else {
          setView('detail');
        }
      });
    });
  }

  if (viewChartButton) {
    viewChartButton.addEventListener('click', () => {
      openChartView();
    });
  }

  if (detailRangeButtons.length) {
    const activeRange = detailRangeButtons.find((btn) => btn.classList.contains('is-active'));
    if (activeRange && activeRange.dataset.range) {
      detailRange = activeRange.dataset.range;
    }
    setDetailRange(detailRange);
    detailRangeButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const rangeKey = btn.dataset.range || '1d';
        if (rangeKey === detailRange) return;
        setDetailRange(rangeKey);
        if (!detailInterval || !normalizeIntervalKey(detailInterval)) {
          setDetailInterval(resolveDefaultInterval(rangeKey), { persist: true, skipReload: true });
        }
        if (detailSymbol) {
          loadDetailData(detailSymbol, rangeKey, { renderChart: true, allowCache: false });
        }
      });
    });
  }

  buildIntervalLabelMap();
  loadIntervalPrefs();
  const savedInterval = normalizeIntervalKey(loadPreference(PREF_INTERVAL_KEY, '')) || resolveDefaultInterval(detailRange);
  setDetailInterval(savedInterval, { persist: false, skipReload: true });

  if (intervalTrigger && intervalMenu) {
    intervalTrigger.addEventListener('click', (event) => {
      event.stopPropagation();
      toggleIntervalMenu();
    });
    document.addEventListener('click', (event) => {
      if (!intervalMenuOpen) return;
      const target = event.target;
      if (intervalMenu.contains(target) || intervalTrigger.contains(target)) {
        return;
      }
      closeIntervalMenu();
    });
    document.addEventListener('keydown', (event) => {
      if (!intervalMenuOpen) return;
      if (event.key === 'Escape') {
        closeIntervalMenu();
        intervalTrigger.focus({ preventScroll: true });
        return;
      }
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        focusIntervalOption(intervalMenuIndex + 1);
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault();
        focusIntervalOption(intervalMenuIndex - 1);
      }
      if (event.key === 'Enter') {
        const options = getIntervalMenuOptions();
        if (intervalMenuIndex >= 0 && options[intervalMenuIndex]) {
          options[intervalMenuIndex].click();
        }
      }
    });
  }

  if (intervalMenu) {
    intervalMenu.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (target.matches('[data-role="detail-interval-select"]')) {
        const intervalKey = target.dataset.interval;
        if (!intervalKey) return;
        setDetailInterval(intervalKey);
        closeIntervalMenu();
      }
      if (target.matches('[data-role="detail-interval-fav"]')) {
        event.stopPropagation();
        toggleIntervalFavorite(target.dataset.interval);
      }
    });
  }

  if (intervalCustomToggle && intervalCustomPanel) {
    intervalCustomToggle.addEventListener('click', () => {
      intervalCustomPanel.hidden = !intervalCustomPanel.hidden;
    });
  }

  if (intervalCustomApply && intervalCustomValue && intervalCustomUnit) {
    intervalCustomApply.addEventListener('click', () => {
      const value = parseInt(intervalCustomValue.value, 10);
      const unit = intervalCustomUnit.value;
      if (!Number.isFinite(value) || value <= 0) {
        setDetailStatus(TEXT.intervalInvalid, true);
        return;
      }
      const key = `${value}${unit}`;
      if (!addCustomInterval(key)) {
        setDetailStatus(TEXT.intervalInvalid, true);
        return;
      }
      setDetailInterval(key);
      intervalCustomPanel.hidden = true;
      intervalCustomValue.value = '';
    });
  }

  if (detailAdvancedToggle && detailAdvancedPanel) {
    detailAdvancedToggle.addEventListener('click', () => {
      const willOpen = detailAdvancedPanel.hidden;
      detailAdvancedPanel.hidden = !willOpen;
      detailAdvancedToggle.classList.toggle('is-active', willOpen);
      detailAdvancedToggle.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
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

  window.addEventListener('resize', () => {
    setView(currentView || 'list');
    resizeDetailChart();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (!autoRefreshPaused) {
        pauseAutoRefresh({ manual: false });
      }
      return;
    }
    if (autoRefreshSuspended && !autoRefreshPaused) {
      resumeAutoRefresh({ manual: false });
    }
  });
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
  updateStatusContext();
  updateSortIndicator();
  updateTimezoneToggle();
  updateAutoRefreshToggle();
  updateTradeModeUI();
  setView(currentView || 'list');

  setupRankObserver();
  loadData();
  fetchTradeMode();
  connectMarketSocket();
})();
