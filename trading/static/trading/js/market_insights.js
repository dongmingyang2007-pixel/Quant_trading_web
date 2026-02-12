(() => {
  const apiMeta = document.querySelector('meta[name="market-api"]');
  const apiUrl = apiMeta ? apiMeta.getAttribute('content') : window.MARKET_API_URL || '/market/api/';
  const chartApiMeta = document.querySelector('meta[name="market-chart-api"]');
  const chartApiUrl = chartApiMeta ? chartApiMeta.getAttribute('content') : '/api/market/chart/';
  const chartAnalyzeMeta = document.querySelector('meta[name="market-chart-analyze-api"]');
  const chartAnalyzeApiUrl = chartAnalyzeMeta ? chartAnalyzeMeta.getAttribute('content') : '/api/market/chart/analyze/';
  const chartAnalyzeSampleMeta = document.querySelector('meta[name="market-chart-analyze-sample-api"]');
  const chartAnalyzeSampleApiUrl = chartAnalyzeSampleMeta
    ? chartAnalyzeSampleMeta.getAttribute('content')
    : '/api/market/chart/analyze/sample/';
  const chartAnalyzeTrainMeta = document.querySelector('meta[name="market-chart-analyze-train-api"]');
  const chartAnalyzeTrainApiUrl = chartAnalyzeTrainMeta
    ? chartAnalyzeTrainMeta.getAttribute('content')
    : '/api/market/chart/analyze/train/';
  const chartAnalyzeMetaMeta = document.querySelector('meta[name="market-chart-analyze-meta-api"]');
  const chartAnalyzeMetaApiUrl = chartAnalyzeMetaMeta
    ? chartAnalyzeMetaMeta.getAttribute('content')
    : '/api/market/chart/analyze/meta/';
  const assetsMeta = document.querySelector('meta[name="market-assets"]');
  const assetsUrl = assetsMeta ? assetsMeta.getAttribute('content') : '/api/market/assets/';
  const newsSentimentApiUrl = '/api/market/news/sentiment/';
  const langMeta = document.querySelector('meta[name="market-lang"]');
  const backtestMeta = document.querySelector('meta[name="backtest-base"]');
  const backtestBase = backtestMeta ? backtestMeta.getAttribute('content') : '/backtest/';
  const docLang = document.documentElement.getAttribute('lang');
  const langPrefix = ((langMeta && langMeta.getAttribute('content')) || docLang || navigator.language || 'zh')
    .toLowerCase()
    .slice(0, 2);
  const formatPrice4 = (value) => {
    const numeric = typeof value === 'number' ? value : Number.parseFloat(value);
    return Number.isFinite(numeric) ? numeric.toFixed(4) : '--';
  };
  const CHART_PRICE_FORMAT = { type: 'price', precision: 4, minMove: 0.0001 };

  const listContainer = document.querySelector('[data-role="ranking-list"]');
  const rankingChangeHeader = document.querySelector('[data-role="ranking-change-header"]');
  const rankingChangeLabel = document.querySelector('[data-role="ranking-change-label"]');
  const rankingSortButton = document.querySelector('[data-role="ranking-sort"]');
  const rankingSortIcon = document.querySelector('[data-role="ranking-sort-icon"]');
  const rankingsSection = document.querySelector('[data-role="rankings-section"]');
  const allStocksSection = document.querySelector('[data-role="all-stocks"]');
  const allStocksFilter = document.querySelector('[data-role="all-stocks-filter"]');
  const allStocksLetters = document.querySelector('[data-role="all-stocks-letters"]');
  const allStocksBody = document.querySelector('[data-role="all-stocks-body"]');
  const allStocksPagination = document.querySelector('[data-role="all-stocks-pagination"]');
  const allStocksCount = document.querySelector('[data-role="all-stocks-count"]');
  const allStocksBack = document.querySelector('[data-role="all-stocks-back"]');
  const rankTabs = Array.prototype.slice.call(document.querySelectorAll('[data-role="rank-tab"]'));
  const rankGroupsToggle = document.querySelector('[data-role="rank-groups-toggle"]');
  const rankGroupsAdvanced = document.querySelector('[data-role="rank-groups-advanced"]');
  const rankDesc = document.querySelector('[data-role="rank-desc"]');
  const rankContext = document.querySelector('[data-role="rank-context"]');
  const denseList = document.querySelector('[data-role="dense-list"]');
  const denseTimeframe = document.querySelector('[data-role="dense-timeframe"]');
  const denseCount = document.querySelector('[data-role="dense-count"]');
  const denseLeaderSymbol = document.querySelector('[data-role="dense-leader-symbol"]');
  const denseLeaderMetric = document.querySelector('[data-role="dense-leader-metric"]');
  const denseLaggardSymbol = document.querySelector('[data-role="dense-laggard-symbol"]');
  const denseLaggardMetric = document.querySelector('[data-role="dense-laggard-metric"]');
  const denseAuto = document.querySelector('[data-role="dense-auto"]');
  const denseUpdated = document.querySelector('[data-role="dense-updated"]');
  const denseSource = document.querySelector('[data-role="dense-source"]');
  const rankSortSummaries = Array.prototype.slice.call(document.querySelectorAll('[data-role="rank-sort-summary"]'));
  const rankSentinel = document.querySelector('[data-role="ranking-sentinel"]');
  const statusState = document.querySelector('[data-role="status-state"]');
  const statusText = document.querySelector('[data-role="status-text"]');
  const statusContext = document.querySelector('[data-role="status-context"]');
  const statusUpdated = document.querySelector('[data-role="status-updated"]');
  const statusExpanded = document.querySelector('[data-role="status-expanded"]');
  const statusTrackShell = document.querySelector('[data-role="status-track-shell"]');
  const statusTrackFill = document.querySelector('[data-role="status-track-fill"]');
  const statusTrackText = document.querySelector('[data-role="status-track-text"]');
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
  const statusControlsToggle = document.querySelector('[data-role="status-controls-toggle"]');
  const statusControlsAdvanced = document.querySelector('[data-role="status-controls-advanced"]');
  const controlsPopoverBackdrop = document.querySelector('[data-role="controls-popover-backdrop"]');
  const controlsPopoverClose = document.querySelector('[data-role="controls-popover-close"]');
  const autoRefreshSummary = document.querySelector('[data-role="auto-refresh-summary"]');
  const timezoneSummary = document.querySelector('[data-role="timezone-summary"]');
  const timezoneToggles = Array.prototype.slice.call(document.querySelectorAll('[data-role="timezone-toggle"]'));
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
  const quickRail = document.querySelector('[data-role="quick-rail"]');
  const quickRailContent = quickRail ? quickRail.closest('.mi-hub-v2-content') : null;
  const quickPanelCollapse = document.querySelector('[data-role="quick-panel-collapse"]');
  const recentClear = document.querySelector('[data-role="recent-clear"]');
  const typeaheadPanel = document.querySelector('[data-role="typeahead-panel"]');
  const typeaheadList = typeaheadPanel && typeaheadPanel.querySelector('[data-role="typeahead-list"]');
  const typeaheadHint = typeaheadPanel && typeaheadPanel.querySelector('[data-role="typeahead-hint"]');
  const detailRoot = document.querySelector('[data-role="market-detail"]') || document;
  const detailTitle = detailRoot.querySelector('[data-role="detail-title"]');
  const detailSubtitle = detailRoot.querySelector('[data-role="detail-subtitle"]');
  const detailSource = detailRoot.querySelector('[data-role="detail-source"]');
  const detailUpdated = detailRoot.querySelector('[data-role="detail-updated"]');
  const detailUpdatedCompact = document.querySelector('[data-role="detail-updated-compact"]');
  const detailLatency = detailRoot.querySelector('[data-role="detail-latency"]');
  const detailStatus = detailRoot.querySelector('[data-role="detail-status"]');
  const detailStatusCardText = detailRoot.querySelector('[data-role="detail-status-card-text"]');
  const detailStatusCardMeta = detailRoot.querySelector('[data-role="detail-status-card-meta"]');
  const detailStatusCardCta = detailRoot.querySelector('[data-role="detail-status-card-cta"]');
  const detailStatusReconnect = detailRoot.querySelector('[data-role="detail-status-reconnect"]');
  const detailLazy = detailRoot.querySelector('[data-role="detail-lazy"]');
  const waveMigratedNotice = detailRoot.querySelector('[data-role="wave-migrated-notice"]');
  const waveStatus = detailRoot.querySelector('[data-role="wave-status"]');
  const wavePatternKey = detailRoot.querySelector('[data-role="wave-pattern-key"]');
  const wavePatternBias = detailRoot.querySelector('[data-role="wave-pattern-bias"]');
  const wavePatternConfidence = detailRoot.querySelector('[data-role="wave-pattern-confidence"]');
  const wavePatternHint = detailRoot.querySelector('[data-role="wave-pattern-hint"]');
  const waveWaveKey = detailRoot.querySelector('[data-role="wave-wave-key"]');
  const waveWaveStage = detailRoot.querySelector('[data-role="wave-wave-stage"]');
  const waveWaveDirection = detailRoot.querySelector('[data-role="wave-wave-direction"]');
  const waveWaveConfidence = detailRoot.querySelector('[data-role="wave-wave-confidence"]');
  const waveProbUp = detailRoot.querySelector('[data-role="wave-prob-up"]');
  const waveProbDown = detailRoot.querySelector('[data-role="wave-prob-down"]');
  const waveProbNeutral = detailRoot.querySelector('[data-role="wave-prob-neutral"]');
  const waveProbUpFill = detailRoot.querySelector('[data-role="wave-prob-up-fill"]');
  const waveProbDownFill = detailRoot.querySelector('[data-role="wave-prob-down-fill"]');
  const waveProbNeutralFill = detailRoot.querySelector('[data-role="wave-prob-neutral-fill"]');
  const waveSetupDirection = detailRoot.querySelector('[data-role="wave-setup-direction"]');
  const waveSetupConfidence = detailRoot.querySelector('[data-role="wave-setup-confidence"]');
  const waveSetupNote = detailRoot.querySelector('[data-role="wave-setup-note"]');
  const waveDiagnostics = detailRoot.querySelector('[data-role="wave-diagnostics"]');
  const waveOverlayToggle = detailRoot.querySelector('[data-role="wave-overlay-toggle"]');
  const waveSeriesModeSelect = detailRoot.querySelector('[data-role="wave-series-mode"]');
  const waveSmoothingWindowSelect = detailRoot.querySelector('[data-role="wave-smoothing-window"]');
  const waveAnalyzeRun = detailRoot.querySelector('[data-role="wave-analyze-run"]');
  const waveSampleLabel = detailRoot.querySelector('[data-role="wave-sample-label"]');
  const waveSampleSave = detailRoot.querySelector('[data-role="wave-sample-save"]');
  const waveTrainRun = detailRoot.querySelector('[data-role="wave-train-run"]');
  const waveTrainMeta = detailRoot.querySelector('[data-role="wave-train-meta"]');
  const detailChartEl = detailRoot.querySelector('#market-detail-chart');
  const detailIndicatorEl = detailRoot.querySelector('#market-detail-indicator');
  const detailIndicatorSecondaryEl = detailRoot.querySelector('#market-detail-indicator-secondary');
  const detailOverlaySelect = detailRoot.querySelector('#detail-overlay-select');
  const detailIndicatorSelect = detailRoot.querySelector('#detail-indicator-select');
  const detailDrawButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('.draw-btn'));
  const detailResetZoom = detailRoot.querySelector('[data-role="detail-reset-zoom"]');
  const chartRefreshBtn = detailRoot.querySelector('[data-role="chart-refresh"]');
  const detailRangeButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('.detail-timeframe--quick'));
  const intervalTrigger = detailRoot.querySelector('[data-role="detail-interval-trigger"]');
  const intervalMenu = detailRoot.querySelector('[data-role="detail-interval-menu"]');
  const intervalCurrent = detailRoot.querySelector('[data-role="detail-interval-current"]');
  const intervalSelectButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('[data-role="detail-interval-select"]'));
  const intervalFavButtons = Array.prototype.slice.call(detailRoot.querySelectorAll('[data-role="detail-interval-fav"]'));
  const intervalGroupToggles = Array.prototype.slice.call(detailRoot.querySelectorAll('[data-role="detail-interval-group-toggle"]'));
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
  const profileOverview = document.querySelector('[data-role="profile-overview"]');
  const profileLogo = document.querySelector('[data-role="profile-logo"]');
  const profileGrid = document.querySelector('[data-role="profile-grid"]');
  const profileSector = document.querySelector('[data-role="profile-sector"]');
  const profileCeo = document.querySelector('[data-role="profile-ceo"]');
  const profileHq = document.querySelector('[data-role="profile-hq"]');
  const profileIndustry = document.querySelector('[data-role="profile-industry"]');
  const profileSkeleton = document.querySelector('[data-role="profile-skeleton"]');
  const profileNote = document.querySelector('[data-role="profile-note"]');
  const aiSummary = document.querySelector('[data-role="ai-summary"]');
  const aiSummaryStatus = document.querySelector('[data-role="ai-status"]');
  const aiSkeleton = document.querySelector('[data-role="ai-skeleton"]');
  const aiSummaryBody = document.querySelector('[data-role="ai-body"]');
  const aiSummaryEvent = document.querySelector('[data-role="ai-summary-event"]');
  const aiSummaryImpact = document.querySelector('[data-role="ai-summary-impact"]');
  const aiSummaryImplication = document.querySelector('[data-role="ai-summary-implication"]');
  const aiRefreshBtn = document.querySelector('[data-role="ai-refresh"]');
  const aiUpdated = document.querySelector('[data-role="ai-updated"]');
  const newsList = document.querySelector('[data-role="news-list"]');
  const newsScrollRoot = document.querySelector('[data-role="news-scroll-root"]');
  const newsLoadMore = document.querySelector('[data-role="news-load-more"]');
  const newsSentinel = document.querySelector('[data-role="news-sentinel"]');
  const newsFilterButtons = Array.prototype.slice.call(document.querySelectorAll('[data-role="news-filter"]'));
  const newsSubtitle = document.querySelector('[data-role="news-subtitle"]');
  const keyStatsCard = document.querySelector('[data-role="key-stats-card"]');
  const keyStatsEmpty = document.querySelector('[data-role="key-stats-empty"]');
  const keyStatsNote = document.querySelector('[data-role="key-stats-note"]');
  const keyStatsSkeleton = document.querySelector('[data-role="key-stats-skeleton"]');
  const keyStatsError = document.querySelector('[data-role="key-stats-error"]');
  const keyStatsChartBtn = document.querySelector('[data-role="key-stats-chart"]');
  const keyStatsNewsBtn = document.querySelector('[data-role="key-stats-news"]');
  const statOpen = document.querySelector('[data-role="stat-open"]');
  const statHigh = document.querySelector('[data-role="stat-high"]');
  const statLow = document.querySelector('[data-role="stat-low"]');
  const statPrevClose = document.querySelector('[data-role="stat-prev-close"]');
  const statVolume = document.querySelector('[data-role="stat-volume"]');
  const statVwap = document.querySelector('[data-role="stat-vwap"]');
  const stat52w = document.querySelector('[data-role="stat-52w"]');
  const statAtr = document.querySelector('[data-role="stat-atr"]');
  const statAvgVol = document.querySelector('[data-role="stat-avg-vol"]');
  const statMarketCap = document.querySelector('[data-role="stat-market-cap"]');
  const statSector = document.querySelector('[data-role="stat-sector"]');
  const statIndustry = document.querySelector('[data-role="stat-industry"]');
  const contextTabs = Array.prototype.slice.call(document.querySelectorAll('[data-role="context-tab"]'));
  const contextPanels = Array.prototype.slice.call(document.querySelectorAll('[data-role="context-panel"]'));
  const contextCard = document.querySelector('[data-role="context-card"]');
  const viewList = document.querySelector('[data-view="list"]');
  const viewDetail = document.querySelector('[data-view="detail"]');
  const viewChart = document.querySelector('[data-view="chart"]');
  const viewBackButtons = Array.prototype.slice.call(document.querySelectorAll('[data-view-back]'));
  const viewChartButton = document.querySelector('[data-view-chart]');
  const paneTabs = Array.prototype.slice.call(document.querySelectorAll('[data-role="pane-tab"]'));
  const workbenchPanels = Array.prototype.slice.call(document.querySelectorAll('[data-role="workbench-panel"]'));
  const instrumentBackBtn = document.querySelector('[data-role="instrument-back"]');
  const detailWatchToggle = document.querySelector('[data-role="detail-watch-toggle"]');
  const headerStatusChip = document.querySelector('[data-role="detail-status-chip"]');
  const headerStatusText = document.querySelector('[data-role="detail-status-text"]');
  const headerStatusCta = document.querySelector('[data-role="detail-status-cta"]');
  const chartStatusChip = detailRoot.querySelector('[data-role="chart-status-chip"]');
  const chartStatusText = detailRoot.querySelector('[data-role="chart-status-text"]');
  const chartStatusMeta = detailRoot.querySelector('[data-role="chart-status-meta"]');
  const detailSymbolEl = document.querySelector('[data-role="detail-symbol"]');
  const detailNameEl = document.querySelector('[data-role="detail-name"]');
  const detailPriceEl = document.querySelector('[data-role="detail-price"]');
  const detailChangeEl = document.querySelector('[data-role="detail-change"]');
  const detailMetaEl = document.querySelector('[data-role="detail-meta"]');
  const quickOpenEl = document.querySelector('[data-role="quick-open"]');
  const quickHighEl = document.querySelector('[data-role="quick-high"]');
  const quickLowEl = document.querySelector('[data-role="quick-low"]');
  const quickPrevEl = document.querySelector('[data-role="quick-prev"]');
  const quickVolumeEl = document.querySelector('[data-role="quick-volume"]');
  const quick52wEl = document.querySelector('[data-role="quick-52w"]');
  const tradeSymbolEl = document.querySelector('[data-role="trade-symbol"]');
  const tradeQtyInput = document.querySelector('[data-role="trade-qty"]');
  const tradeNotionalInput = document.querySelector('[data-role="trade-notional"]');
  const tradeUnitButtons = Array.prototype.slice.call(document.querySelectorAll('[data-role="trade-unit-btn"]'));
  const tradeInputAddon = document.querySelector('[data-role="trade-input-addon"]');
  const tradeInputHint = document.querySelector('[data-role="trade-input-hint"]');
  const tradeOrderTypeButtons = Array.prototype.slice.call(document.querySelectorAll('[data-role="order-type-btn"]'));
  const tradeOrderMoreToggle = document.querySelector('[data-role="order-type-more"]');
  const tradeOrderMorePanel = document.querySelector('[data-role="order-more"]');
  const tradeEstimateNotional = document.querySelector('[data-role="trade-estimate-notional"]');
  const tradeBuyingPowerEl = document.querySelector('[data-role="trade-buying-power"]');
  const tradeRemainingCashEl = document.querySelector('[data-role="trade-remaining-cash"]');
  const tradeBuyBtn = document.querySelector('[data-role="trade-buy"]');
  const tradeSellBtn = document.querySelector('[data-role="trade-sell"]');
  const tradeStatusEl = document.querySelector('[data-role="trade-status"]');
  const tradeRetryBtn = document.querySelector('[data-role="trade-retry"]');
  const tradeToastEl = document.querySelector('[data-role="trade-toast"]');
  const tradeModePill = document.querySelector('[data-role="trade-mode-pill"]');
  const tradeRecentBody = document.querySelector('[data-role="trade-recent-body"]');
  const accountCard = document.querySelector('[data-role="account-card"]');
  const accountModeEl = document.querySelector('[data-role="account-mode"]');
  const accountEquityEl = document.querySelector('[data-role="account-equity"]');
  const accountCashEl = document.querySelector('[data-role="account-cash"]');
  const accountBuyingPowerEl = document.querySelector('[data-role="account-buying-power"]');
  const accountPortfolioEl = document.querySelector('[data-role="account-portfolio"]');
  const accountExecutionEl = document.querySelector('[data-role="account-execution"]');
  const accountUpdatedEl = document.querySelector('[data-role="account-updated"]');
  const accountStatusEl = document.querySelector('[data-role="account-status"]');
  const accountRefreshBtn = document.querySelector('[data-role="account-refresh"]');
  const tradeModeButtons = Array.prototype.slice.call(document.querySelectorAll('[data-role="trade-mode-btn"]'));
  const tradeModeStatus = document.querySelector('[data-role="trade-mode-status"]');
  const rankingTable = document.querySelector('[data-role="ranking-table"]');
  const rankingScrollRoot = rankingTable ? rankingTable.closest('.ranking-table-wrap') : null;
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
  const accountEndpoint = '/api/paper/alpaca/account/';

  const initialBtn = document.querySelector('.market-timeframe.is-active');
  let currentTimeframe = (initialBtn && initialBtn.getAttribute('data-timeframe')) || '1mo';
  let currentListType = 'gainers';
  let lastRankingType = 'gainers';
  let allStocksLetter = 'ALL';
  let allStocksPage = 0;
  let allStocksSize = 20;
  let allStocksHasMore = false;
  let allStocksIsLoadingMore = false;
  let allStocksTotal = 0;
  let allStocksQuery = '';
  let suggestionPool = [];
  let recentPool = [];
  let watchPool = [];
  let hideTypeaheadTimer = null;
  let typeaheadOptions = [];
  let typeaheadActiveIndex = -1;
  let retryTimer = null;
  let lastRequest = { query: '', options: {} };
  let rankingRequestController = null;
  let rankingRequestKey = '';
  let rankingRequestNonce = 0;
  let marketSocket = null;
  let socketRetryTimer = null;
  let detailManager = null;
  let detailSymbol = '';
  let detailCompanyName = '';
  let detailRange = '1d';
  let detailInterval = '1m';
  let detailBarIntervalSec = null;
  let detailRetryTimer = null;
  let detailRequestSeq = 0;
  let activeDetailRequest = 0;
  let currentView = 'list';
  let activeWorkbenchPanel = 'overview';
  let autoRefreshTimer = null;
  let autoRefreshCountdownTimer = null;
  let autoRefreshNextAt = 0;
  let autoRefreshRemainingMs = 0;
  let autoRefreshPaused = false;
  let autoRefreshSuspended = false;
  let quickRailCollapsed = false;
  let controlsExpanded = false;
  let rankGroupsExpanded = false;
  let denseItems = [];
  let denseListType = 'gainers';
  let denseTimeframeKey = '1mo';
  let denseSourceKey = 'unknown';
  let statusMessageOverride = '';
  let currentStatusState = 'ready';
  let snapshotState = null;
  let isListLoading = false;
  let liveWaitTimer = null;
  let lastLiveUpdateAt = 0;
  let lastChartSocketUpdateAt = 0;
  let lastFallbackUpdateAt = 0;
  let activeContextTab = null;
  let liveQuoteTimer = null;
  let chartInitTimer = null;
  let pendingChartRender = null;
  let chartSocket = null;
  let chartSocketRetryTimer = null;
  let chartSocketRetryAttempts = 0;
  let chartSocketRetryDelay = 0;
  let chartSocketReady = false;
  let chartSocketSymbol = '';
  let chartSocketLastClose = null;
  let chartHeartbeatTimer = null;
  let chartHeartbeatLastPong = 0;
  let chartTradeBuffer = [];
  let chartTradeFlushHandle = null;
  let chartPollTimer = null;
  let chartPollInFlight = false;
  let chartPollIntervalMs = 1000;
  let newsSentimentToken = 0;
  let newsSentimentInFlight = false;
  let chartTickPollTimer = null;
  let chartTickPollInFlight = false;
  let chartTickCursor = null;
  let chartDataContextVersion = 0;
  const MAX_CHART_TRADE_BUFFER = 2000;
  const DEFAULT_AUTO_REFRESH_MS = 60 * 1000;
  const AUTO_REFRESH_OPTIONS = [0, 15000, 30000, 60000];
  const PREF_TIMEZONE_KEY = 'market.timezone';
  const PREF_AUTO_REFRESH_KEY = 'market.autoRefreshMs';
  const PREF_QUICK_RAIL_COLLAPSED_KEY = 'market.quickRailCollapsed';
  const PREF_INTERVAL_KEY = 'market.chart.interval';
  const PREF_INTERVAL_FAVORITES_KEY = 'market.chart.intervalFavorites';
  const PREF_INTERVAL_CUSTOM_KEY = 'market.chart.intervalCustom';
  const PREF_INTERVAL_GROUPS_KEY = 'market.chart.intervalGroups';
  const PREF_CONTEXT_TAB_KEY = 'market.contextTab';
  const GLOBAL_QUICK_WATCH_KEY = 'market.globalQuick.watchlist';
  const GLOBAL_QUICK_RECENT_KEY = 'market.globalQuick.recent';
  const GLOBAL_QUICK_LAUNCH_KEY = 'market.globalQuick.launchSymbol';
  let timezoneMode = loadPreference(PREF_TIMEZONE_KEY, 'utc');
  if (timezoneMode !== 'utc' && timezoneMode !== 'local') {
    timezoneMode = 'utc';
  }
  let autoRefreshMs = parseInt(loadPreference(PREF_AUTO_REFRESH_KEY, `${DEFAULT_AUTO_REFRESH_MS}`), 10);
  if (!Number.isFinite(autoRefreshMs) || !AUTO_REFRESH_OPTIONS.includes(autoRefreshMs)) {
    autoRefreshMs = DEFAULT_AUTO_REFRESH_MS;
  }
  const quickRailPrefRaw = (loadPreference(PREF_QUICK_RAIL_COLLAPSED_KEY, '1') || '').toString().toLowerCase();
  quickRailCollapsed = quickRailPrefRaw === '1' || quickRailPrefRaw === 'true';
  let intervalFavorites = [];
  let customIntervals = [];
  const LIVE_WAIT_MS = 2000;
  const LIVE_QUOTE_POLL_MS = 5000;
  const LIVE_CHART_STALE_MS = 15000;
  const CHART_SOCKET_RETRY_BASE_MS = 2000;
  const CHART_SOCKET_RETRY_MAX_MS = 20000;
  const CHART_POLL_MS = 1000;
  const CHART_POLL_MS_SECOND = 1000;
  const CHART_POLL_MS_TICK = 250;
  const CHART_POLL_MS_MINUTE = 5000;
  const CHART_POLL_MS_HOUR = 15000;
  const CHART_POLL_MS_DAY = 30000;
  const CHART_TICK_FALLBACK_MS = 1000;
  const CHART_TICK_FALLBACK_WINDOW_SEC = 180;
  const HIGH_FREQ_HISTORY_MAX_BARS = 30000;
  const STANDARD_HISTORY_MAX_BARS = 20000;
  const PRICE_SCALE_TOP_BASE = 0.1;
  const PRICE_SCALE_BOTTOM_BASE = 0.26;
  const PRICE_SCALE_MIN_WIDTH = 64;
  const PRICE_SCALE_WHEEL_SPEED = 0.0022;
  const PRICE_SCALE_HIT_WIDTH = 88;
  const CHART_PULSE_PERIOD_MS = 1800;
  const CHART_PULSE_MIN_RADIUS = 2.2;
  const CHART_PULSE_MAX_RADIUS = 8.5;
  const CHART_PULSE_ALPHA = 0.32;
  const CHART_PULSE_DOT_RADIUS = 2.2;
  const CHART_PULSE_MARKER_SIZE = 1;
  const CHART_PULSE_MARKER_ENABLED = false;
  const CHART_PULSE_MAX_FPS = 30;
  const MARKET_TIMEZONE = 'America/New_York';
  const DIAG_QUERY_KEY = 'diag';
  const AI_REFRESH_COOLDOWN_MS = 20000;
  const diagState = {
    enabled: null,
    booted: false,
    last: new Map(),
  };

  function isDiagEnabled() {
    if (diagState.enabled !== null) return diagState.enabled;
    try {
      const params = new URLSearchParams(window.location.search || '');
      const raw = (params.get(DIAG_QUERY_KEY) || '').toLowerCase();
      diagState.enabled = raw === '1' || raw === 'true' || raw === 'yes';
    } catch (error) {
      diagState.enabled = false;
    }
    return diagState.enabled;
  }

  function diagOnce() {
    if (!isDiagEnabled() || diagState.booted) return;
    diagState.booted = true;
    console.info('[market diag] enabled (append ?diag=1 to URL)');
    console.info('[market diag] recommended: hard refresh + disable cache to reproduce');
  }

  function summarizeBars(bars) {
    if (!Array.isArray(bars)) return null;
    let invalid = 0;
    let duplicates = 0;
    let firstTime = null;
    let lastTime = null;
    const seen = new Set();
    bars.forEach((bar) => {
      if (!bar) {
        invalid += 1;
        return;
      }
      const time = normalizeEpochSeconds(bar.time);
      const open = Number.isFinite(bar.open) ? bar.open : Number.parseFloat(bar.open);
      const high = Number.isFinite(bar.high) ? bar.high : Number.parseFloat(bar.high);
      const low = Number.isFinite(bar.low) ? bar.low : Number.parseFloat(bar.low);
      const close = Number.isFinite(bar.close) ? bar.close : Number.parseFloat(bar.close);
      if (!Number.isFinite(time) || !Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) {
        invalid += 1;
        return;
      }
      const key = `${time.toFixed(6)}`;
      if (seen.has(key)) {
        duplicates += 1;
      } else {
        seen.add(key);
      }
      if (firstTime === null) firstTime = time;
      lastTime = time;
    });
    return {
      count: bars.length,
      firstTime,
      lastTime,
      invalid,
      duplicates,
    };
  }

  function summarizePayload(data) {
    if (Array.isArray(data)) {
      const sample = [];
      if (data.length > 0) sample.push(data[0]);
      if (data.length > 1) sample.push(data[1]);
      if (data.length > 2) sample.push(data[data.length - 1]);
      return { type: 'array', summary: summarizeBars(data), sample };
    }
    if (data && typeof data === 'object') {
      return {
        type: 'object',
        sample: {
          time: data.time,
          open: data.open,
          high: data.high,
          low: data.low,
          close: data.close,
          value: data.value,
          volume: data.volume,
        },
      };
    }
    return { type: typeof data, value: data };
  }

  function captureDiagContext(manager, extra = {}) {
    const lastBar =
      manager && Array.isArray(manager.ohlcData) && manager.ohlcData.length
        ? manager.ohlcData[manager.ohlcData.length - 1]
        : null;
    return {
      symbol: detailSymbol,
      interval: detailInterval,
      range: detailRange,
      chartVisible,
      chartRealtime: typeof isChartRealtimeActive === 'function' ? isChartRealtimeActive() : null,
      chartPollInFlight,
      axisMode: manager ? manager.axisMode : null,
      intervalSpec: manager ? manager.intervalSpec : null,
      ohlcLen: manager && Array.isArray(manager.ohlcData) ? manager.ohlcData.length : 0,
      lastBar,
      liveBucket: manager ? manager.liveBucket : null,
      lastLiveTime: manager ? manager.lastLiveTime : null,
      ...extra,
    };
  }

  function diagLog(type, payload) {
    if (!isDiagEnabled()) return;
    const now = Date.now();
    const last = diagState.last.get(type) || 0;
    if (now - last < 1200) return;
    diagState.last.set(type, now);
    console.groupCollapsed(`[market diag] ${type}`);
    console.log(payload);
    if (console.trace) {
      console.trace();
    }
    console.groupEnd();
  }

  diagOnce();
  const SESSION_COLORS = {
    pre: 'rgba(59, 130, 246, 0.10)',
    post: 'rgba(249, 115, 22, 0.12)',
  };
  const RANK_PAGE_SIZE = 20;
  const ALL_MODE_PAGE_SIZE = 20;
  const NEWS_PAGE_SIZE = 10;
  const NEWS_SENTIMENT_BATCH_SIZE = 8;
  const NEWS_LOAD_MORE_OBSERVER_MARGIN = '240px';
  const RANKING_COLUMNS =
    rankingTable && rankingTable.querySelectorAll('th').length
      ? rankingTable.querySelectorAll('th').length
      : 6;
  const ALL_STOCKS_COLUMNS = 5;
  const CHART_LAZY_EMPTY_RETRY_ATTEMPTS = 4;
  const CHART_LAZY_MAX_EMPTY_STREAK = 4;
  const CHART_LAZY_MIN_JUMP_SECONDS = 3600;
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
  let chartLazyLoading = false;
  let chartLazyExhausted = false;
  let chartLazyCursor = null;
  let chartLazyEmptyStreak = 0;
  let detailLastRangeInteractionAt = 0;
  let detailUserPanned = false;
  let detailAutoRangeLock = false;
  let detailLastChartKey = '';
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
  let lastTradeAgeSeconds = null;
  let lastTradeStaleLabel = '';
  let detailStatusOverride = '';
  let detailStatusIsError = false;
  let newsFilterMode = 'all';
  let latestNewsItems = [];
  let newsListSymbol = '';
  let newsSentimentSymbol = '';
  let newsSentimentCache = new Map();
  let newsSentimentQueue = [];
  let newsSentimentQueuedKeys = new Set();
  let newsPageSize = NEWS_PAGE_SIZE;
  let newsNextOffset = 0;
  let newsHasMore = false;
  let newsLoadingMore = false;
  let newsLoadMoreError = '';
  let newsScrollArmed = false;
  let newsObserver = null;
  let newsRequestSeq = 0;
  let aiRefreshCooldownUntil = 0;
  let aiRefreshCooldownTimer = null;
  let intervalGroupState = {};
  let tradeMode = 'paper';
  let tradeEnabled = true;
  let tradeExecutionEnabled = false;
  let tradeModeBusy = false;
  let tradeModeError = '';
  let tradeUnit = 'shares';
  let tradeOrderType = 'market';
  let tradeOrderMoreOpen = false;
  let tradeBuyingPower = null;
  let tradeLastAttempt = null;
  let tradeToastTimer = null;
  let waveAnalyzeTimer = null;
  let waveAnalyzeInFlight = false;
  let waveAnalyzeQueuedTask = null;
  let waveLastFingerprint = '';
  let waveLastTask = null;
  let waveLastResult = null;
  let waveAnalyzeGeneration = 0;
  let waveAutoEnabled = true;
  let waveOverlayEnabled = true;
  let waveMetaLoaded = false;
  let waveToolPreferred = false;
  let waveToolLaunchPending = false;
  const WAVE_ANALYZE_DEBOUNCE_MS = 800;
  const WAVE_ANALYZE_MAX_POINTS = 360;
  const WAVE_ANALYZE_MIN_POINTS = 24;
  const WAVE_SERIES_MODES = new Set(['close', 'hlc3', 'ohlc4']);
  const WAVE_SMOOTHING_WINDOWS = new Set([1, 3, 5, 7, 9, 11]);

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
        keepPreviousOnRate: '请求受限，沿用上一版数据。',
        keepPreviousOnError: '加载异常，沿用上一版数据。',
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
        latencyLabel: '延迟',
        latencyUnit: 'ms',
        chartTooltipTime: '时间',
        chartTooltipTick: 'T',
        chartTooltipOpen: '开',
        chartTooltipHigh: '高',
        chartTooltipLow: '低',
        chartTooltipClose: '收',
        chartTooltipVolume: '量',
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
        typeaheadHint: '↑↓ 选择，回车跳转或加入自选列表',
        historyCleared: '最近检索已清空',
        historyDeleted: (symbol) => `已删除 ${symbol}`,
        detailLoading: '正在加载K线…',
        detailEmpty: '暂无行情数据',
        detailPrompt: '请选择标的后展示图表与指标。',
        detailError: '加载失败',
        detailFallback: (requested, used) => `无 ${requested} 数据，已显示 ${used} 历史`,
        volumeLabel: '成交量',
        detailLive: '实时流更新中…',
        detailLiveWaiting: '实时流暂未收到更新，请确认短线引擎服务已启动。',
        detailStatusLabels: {
          live: '实时',
          delayed: '延迟',
          stale: '过期',
          disconnected: '断开',
        },
        detailStatusMessages: {
          live: '实时流更新中。',
          delayed: '行情略有延迟。',
          stale: '行情可能已过期。',
          disconnected: '实时连接已断开。',
        },
        detailStatusMeta: (age) => (age ? `最后更新：${age}前` : ''),
        detailStatusMonitor: '打开短线工作台',
        detailStatusReconnect: '重连',
        detailStatusReconnectAttempt: (count) => (count ? `重连中（第 ${count} 次）` : ''),
        detailStatusCloseCode: (code, reason) =>
          code ? `断开码 ${code}${reason ? `：${reason}` : ''}` : '',
        detailStale: (age) => (age ? `行情已延迟（${age}前）` : '行情已延迟'),
        detailWindowLimited: '高频数据仅展示最近窗口。',
        detailDowngraded: (message) => message || '已自动降级为分钟级行情。',
        detailRangeGuard: (label) => `高频粒度与长周期冲突，已切换为 ${label}。`,
        detailLazyLoading: '懒加载中…',
        intervalInvalid: '请输入有效的时间粒度。',
        intervalGroupCollapse: '收起',
        intervalGroupExpand: '展开',
        chartRefresh: '刷新',
        newsFilterAll: '全部',
        newsFilterStrong: '强相关',
        newsStrongEmpty: '暂无强相关新闻',
        newsLoadingMore: '正在加载更多新闻…',
        newsWaitAiBeforeLoadMore: '请等待当前新闻 AI 分析完成后继续下滑加载。',
        newsLoadMoreFailed: '加载更多新闻失败，继续下滑可重试。',
        aiRefresh: '重新生成',
        aiRefreshCooldown: (seconds) => `稍后可刷新（${seconds}s）`,
        keyStatsError: '关键指标加载失败，请稍后重试。',
        tradeSubmitting: '提交中…',
        tradeSuccess: '已提交订单',
        tradeFailed: '下单失败',
        tradeMissingSymbol: '请先选择标的',
        tradeMissingQty: '请输入下单数量',
        tradeMissingNotional: '请输入下单金额',
        tradeMinQty: (min) => `最小下单数量为 ${min}`,
        tradeMinNotional: (min) => `最小下单金额为 ${min}`,
        tradeInsufficientBuyingPower: '可用资金不足',
        tradeBuyingPowerUnknown: '未接入',
        tradeUnitShares: '股数',
        tradeUnitNotional: '金额',
        tradeUnitAddonShares: '股',
        tradeUnitAddonNotional: '美元',
        tradeModePaper: '模拟下单',
        tradeModeLive: '实盘下单',
        tradeModeExecutionOff: '执行未开启',
        tradeModeUpdating: '切换中…',
        tradeModeFailed: '切换失败',
        tradeModeConfirm: '确认切换到实盘？这会使用实盘账户。',
        tradeSellConfirm: '确认卖出？',
        tradeSellConfirmLive: '确认卖出？该操作将使用实盘账户。',
        tradeRecentEmpty: '暂无订单记录',
        accountLoading: '账户加载中…',
        accountMissing: '缺少交易账户或 API 密钥',
        accountError: '账户信息加载失败',
        accountExecutionOn: '执行：开启',
        accountExecutionOff: '执行：关闭',
        accountUpdatedPrefix: '更新',
        allStocksLoading: '正在加载全部股票…',
        allStocksEmpty: '暂无股票数据',
        allStocksCount: (total, page, totalPages) => `共 ${total} 只 · 第 ${page}/${totalPages} 页`,
        timeframeNotSupported: (timeframeLabel) => `该榜单暂不支持 ${timeframeLabel} 区间`,
        pagePrev: '上一页',
        pageNext: '下一页',
        profileEmpty: '暂无公司信息。',
        newsEmpty: '暂无相关新闻。',
        sentimentLoading: 'AI 判断中',
        aiPlaceholder: 'AI 摘要准备中。',
        aiStatus: {
          pending: 'AI 摘要生成中…',
          missing_key: 'AI 未配置百炼 API Key，当前显示新闻摘要。',
          error: 'AI 暂时不可用，已显示回退摘要。',
          fallback: 'AI 暂无输出，已显示回退摘要。',
        },
        keyStatsEmpty: '暂无可用数据，部分指标可能由数据源提供。',
        keyStatsFundamentalNote: '基础面数据由数据源提供，当前暂不可用。',
        profileFundamentalNote: '基础面字段缺失，数据源未提供。',
        profileLabels: {
          sector: '板块',
          industry: '行业',
          market_cap: '市值',
          pe: '市盈率（PE）',
          pb: '市净率（PB）',
          beta: 'β',
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
          massive: 'Massive',
          cache: '缓存',
          unknown: '未知',
        },
        controlsExpand: '更多控制',
        controlsCollapse: '收起控制',
        autoSummaryLabel: '自动',
        autoSummaryOff: '关闭',
        timezoneSummaryLabel: '时区',
        timezoneLocalLabel: '本地',
        rankListsExpand: '更多榜单',
        rankListsCollapse: '收起榜单',
        waveStatusWaiting: '等待图表数据后自动分析。',
        waveStatusAnalyzing: '正在进行波型分析…',
        waveStatusReady: '分析完成。',
        waveStatusError: '分析失败，请稍后重试。',
        waveStatusInsufficient: (required, got) => `可分析数据不足（需要 ${required}，当前 ${got}）。`,
        waveStatusLabelRequired: '请先选择样本标签。',
        waveStatusSaving: '正在保存样本…',
        waveStatusTraining: '正在训练模型…',
        waveStatusSaved: (total) => `样本已保存（累计 ${total}）。`,
        waveStatusTrained: (accuracy) =>
          Number.isFinite(accuracy) ? `模型训练完成（准确率 ${(accuracy * 100).toFixed(1)}%）。` : '模型训练完成。',
        waveMeta: (total, classes, threshold) => `样本 ${total} · 类别 ${classes} · 阈值 ${threshold}`,
        waveBiasLabels: {
          bullish: '偏多',
          bearish: '偏空',
          neutral: '中性',
        },
        waveDirectionLabels: {
          up: '上行',
          down: '下行',
          neutral: '中性',
        },
        waveStageLabels: {
          impulse: '推动浪',
          correction: '调整浪',
          unknown: '未知',
        },
        waveSetupLabels: {
          bullish: '偏多跟随',
          bearish: '偏空防守',
          neutral: '等待确认',
        },
        waveSeriesModeLabels: {
          close: '收盘',
          hlc3: 'HLC3',
          ohlc4: 'OHLC4',
        },
        waveSmoothLabel: (window) => (window > 1 ? `平滑 ${window}` : '未平滑'),
        missingReasons: {
          source_not_provided: {
            short: '源缺失',
            full: '数据源未提供该字段。',
          },
          not_applicable_fund: {
            short: '不适用',
            full: '该字段对基金/ETF 等标的通常不适用。',
          },
          insufficient_window: {
            short: '窗口不足',
            full: '当前时间窗口不足，暂无法计算该字段。',
          },
          timeframe_snapshot_pending: {
            short: '计算中',
            full: '所选时间区间快照尚在生成，当前字段暂不可用。',
          },
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
        keepPreviousOnRate: 'Rate limited, keeping previous data.',
        keepPreviousOnError: 'Request failed, keeping previous data.',
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
        latencyLabel: 'Latency',
        latencyUnit: 'ms',
        chartTooltipTime: 'Time',
        chartTooltipTick: 'T',
        chartTooltipOpen: 'O',
        chartTooltipHigh: 'H',
        chartTooltipLow: 'L',
        chartTooltipClose: 'C',
        chartTooltipVolume: 'Vol',
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
        detailLiveWaiting: 'No live ticks yet. Check the short-term engine service.',
        detailStatusLabels: {
          live: 'Live',
          delayed: 'Delayed',
          stale: 'Stale',
          disconnected: 'Disconnected',
        },
        detailStatusMessages: {
          live: 'Live stream is updating.',
          delayed: 'Data is slightly delayed.',
          stale: 'Data may be stale.',
          disconnected: 'Realtime connection is offline.',
        },
        detailStatusMeta: (age) => (age ? `Last update: ${age} ago` : ''),
        detailStatusMonitor: 'Open short-term workbench',
        detailStatusReconnect: 'Reconnect',
        detailStatusReconnectAttempt: (count) => (count ? `Reconnecting (${count})` : ''),
        detailStatusCloseCode: (code, reason) =>
          code ? `Close ${code}${reason ? `: ${reason}` : ''}` : '',
        detailStale: (age) => (age ? `Stale data (${age} ago)` : 'Stale data'),
        detailWindowLimited: 'High-frequency data is window-limited.',
        detailDowngraded: (message) => message || 'Interval auto-downgraded to minute bars.',
        detailRangeGuard: (label) => `High-frequency interval conflicts with long range; switched to ${label}.`,
        detailLazyLoading: 'Loading older data…',
        intervalInvalid: 'Enter a valid interval value.',
        intervalGroupCollapse: 'Collapse',
        intervalGroupExpand: 'Expand',
        chartRefresh: 'Refresh',
        newsFilterAll: 'All',
        newsFilterStrong: 'Strong',
        newsStrongEmpty: 'No strong-related headlines.',
        newsLoadingMore: 'Loading more headlines…',
        newsWaitAiBeforeLoadMore: 'Please wait until AI finishes analyzing current headlines before loading more.',
        newsLoadMoreFailed: 'Failed to load more headlines. Scroll again to retry.',
        aiRefresh: 'Regenerate',
        aiRefreshCooldown: (seconds) => `Refresh in ${seconds}s`,
        keyStatsError: 'Key stats failed to load. Please retry.',
        tradeSubmitting: 'Submitting…',
        tradeSuccess: 'Order submitted',
        tradeFailed: 'Order failed',
        tradeMissingSymbol: 'Select a symbol first',
        tradeMissingQty: 'Enter a quantity',
        tradeMissingNotional: 'Enter a notional amount',
        tradeMinQty: (min) => `Minimum quantity is ${min}`,
        tradeMinNotional: (min) => `Minimum notional is ${min}`,
        tradeInsufficientBuyingPower: 'Insufficient buying power',
        tradeBuyingPowerUnknown: 'Not available',
        tradeUnitShares: 'Shares',
        tradeUnitNotional: 'Notional',
        tradeUnitAddonShares: 'Shares',
        tradeUnitAddonNotional: 'USD',
        tradeModePaper: 'Paper trading',
        tradeModeLive: 'Live trading',
        tradeModeExecutionOff: 'Execution disabled',
        tradeModeUpdating: 'Switching…',
        tradeModeFailed: 'Switch failed',
        tradeModeConfirm: 'Switch to live trading? This will use your Live account.',
        tradeSellConfirm: 'Confirm sell order?',
        tradeSellConfirmLive: 'Confirm sell order? This uses your Live account.',
        tradeRecentEmpty: 'No recent orders yet',
        accountLoading: 'Loading account…',
        accountMissing: 'Missing trading account or API key',
        accountError: 'Failed to load account',
        accountExecutionOn: 'Execution: On',
        accountExecutionOff: 'Execution: Off',
        accountUpdatedPrefix: 'Updated',
        allStocksLoading: 'Loading all stocks…',
        allStocksEmpty: 'No stocks found.',
        allStocksCount: (total, page, totalPages) => `${total} symbols · Page ${page}/${totalPages}`,
        timeframeNotSupported: (timeframeLabel) => `This leaderboard does not support ${timeframeLabel}`,
        pagePrev: 'Prev',
        pageNext: 'Next',
        profileEmpty: 'No company profile available.',
        newsEmpty: 'No related headlines yet.',
        sentimentLoading: 'AI analyzing',
        aiPlaceholder: 'AI summary is preparing.',
        aiStatus: {
          pending: 'Generating AI summary…',
          missing_key: 'AI API key is missing; showing fallback summary.',
          error: 'AI is unavailable right now; showing fallback summary.',
          fallback: 'AI returned no output; showing fallback summary.',
        },
        keyStatsEmpty: 'No stats yet. Some fields depend on data provider availability.',
        keyStatsFundamentalNote: 'Fundamentals are unavailable from the current data source.',
        profileFundamentalNote: 'Fundamentals are missing from the current data source.',
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
          massive: 'Massive',
          cache: 'Cache',
          unknown: 'Unknown',
        },
        controlsExpand: 'More controls',
        controlsCollapse: 'Hide controls',
        autoSummaryLabel: 'Auto',
        autoSummaryOff: 'Off',
        timezoneSummaryLabel: 'TZ',
        timezoneLocalLabel: 'Local',
        rankListsExpand: 'More lists',
        rankListsCollapse: 'Hide lists',
        waveStatusWaiting: 'Waiting for chart data to run analysis.',
        waveStatusAnalyzing: 'Running pattern and wave analysis…',
        waveStatusReady: 'Analysis complete.',
        waveStatusError: 'Analysis failed. Please retry.',
        waveStatusInsufficient: (required, got) => `Not enough points (${got}/${required}).`,
        waveStatusLabelRequired: 'Select a sample label first.',
        waveStatusSaving: 'Saving sample…',
        waveStatusTraining: 'Training model…',
        waveStatusSaved: (total) => `Sample saved (${total} total).`,
        waveStatusTrained: (accuracy) =>
          Number.isFinite(accuracy) ? `Model trained (accuracy ${(accuracy * 100).toFixed(1)}%).` : 'Model trained.',
        waveMeta: (total, classes, threshold) => `${total} samples · ${classes} classes · threshold ${threshold}`,
        waveBiasLabels: {
          bullish: 'Bullish',
          bearish: 'Bearish',
          neutral: 'Neutral',
        },
        waveDirectionLabels: {
          up: 'Up',
          down: 'Down',
          neutral: 'Neutral',
        },
        waveStageLabels: {
          impulse: 'Impulse',
          correction: 'Correction',
          unknown: 'Unknown',
        },
        waveSetupLabels: {
          bullish: 'Bullish follow-through',
          bearish: 'Bearish risk-off',
          neutral: 'Await confirmation',
        },
        waveSeriesModeLabels: {
          close: 'Close',
          hlc3: 'HLC3',
          ohlc4: 'OHLC4',
        },
        waveSmoothLabel: (window) => (window > 1 ? `Smooth ${window}` : 'Raw'),
        missingReasons: {
          source_not_provided: {
            short: 'Source missing',
            full: 'The data source did not provide this field.',
          },
          not_applicable_fund: {
            short: 'N/A for fund',
            full: 'This field is usually not applicable to fund/ETF instruments.',
          },
          insufficient_window: {
            short: 'Window short',
            full: 'Not enough bars in the selected window to calculate this field.',
          },
          timeframe_snapshot_pending: {
            short: 'Computing',
            full: 'The selected timeframe snapshot is still building.',
          },
        },
      };

  const TEXT = langPrefix === 'zh' ? TEXT_ZH : TEXT_EN;
  const TRADE_MIN_QTY = 0.0001;
  const TRADE_MIN_NOTIONAL = 1;
  const currencyFormatter = (() => {
    try {
      return new Intl.NumberFormat(locale, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
    } catch (err) {
      return null;
    }
  })();

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

  function formatCurrency(value) {
    if (!Number.isFinite(value)) return '—';
    if (currencyFormatter) {
      return currencyFormatter.format(value);
    }
    return `$${value.toFixed(2)}`;
  }

  function parseNumberFromInput(inputEl) {
    if (!inputEl) return null;
    const raw = Number(inputEl.value);
    return Number.isFinite(raw) ? raw : null;
  }

  function getDetailPriceValue() {
    if (!detailPriceEl) return null;
    const raw = (detailPriceEl.textContent || '').replace(/[^0-9.\-]/g, '');
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }

  function ensureSnapshotNode() {
    return;
  }

  ensureSnapshotNode();

  function syncGlobalQuickStorage() {
    try {
      const watch = JSON.stringify(Array.isArray(watchPool) ? watchPool : []);
      const recent = JSON.stringify(Array.isArray(recentPool) ? recentPool : []);
      window.localStorage.setItem(GLOBAL_QUICK_WATCH_KEY, watch);
      window.localStorage.setItem(GLOBAL_QUICK_RECENT_KEY, recent);
    } catch (error) {
      return;
    }
    try {
      window.dispatchEvent(
        new CustomEvent('market:global-quick-updated', {
          detail: {
            watchlist: Array.isArray(watchPool) ? watchPool.slice() : [],
            recent: Array.isArray(recentPool) ? recentPool.slice() : [],
          },
        })
      );
    } catch (error) {
      // ignore
    }
  }

  function consumeGlobalQuickLaunchSymbol() {
    try {
      const raw = window.localStorage.getItem(GLOBAL_QUICK_LAUNCH_KEY);
      if (!raw) return '';
      window.localStorage.removeItem(GLOBAL_QUICK_LAUNCH_KEY);
      return normalizeSymbol(raw);
    } catch (error) {
      return '';
    }
  }

  function getIndicatorLib() {
    if (window.technicalindicators) return window.technicalindicators;
    if (window.SMA && window.EMA && window.RSI) return window;
    return null;
  }

  function parsePageLaunchState() {
    try {
      const params = new URLSearchParams(window.location.search || '');
      const view = (params.get('view') || '').trim().toLowerCase();
      const tool = (params.get('tool') || '').trim().toLowerCase();
      return { view, tool };
    } catch (_error) {
      return { view: '', tool: '' };
    }
  }

  const launchState = parsePageLaunchState();
  waveToolPreferred = launchState.tool === 'wave';
  waveToolLaunchPending = waveToolPreferred;
  if (launchState.view === 'chart') {
    currentView = 'chart';
  }
  if (waveMigratedNotice) {
    waveMigratedNotice.hidden = !waveToolPreferred;
  }

  function formatWavePercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    return `${(number * 100).toFixed(1)}%`;
  }

  function humanizeWaveKey(key) {
    const raw = (key || '').toString().trim();
    if (!raw) return '—';
    return raw.replace(/_/g, ' ');
  }

  function setWaveStatusText(text, tone = '') {
    if (!waveStatus) return;
    waveStatus.textContent = text || '';
    waveStatus.classList.remove('is-error', 'is-success');
    if (tone === 'error') {
      waveStatus.classList.add('is-error');
    } else if (tone === 'success') {
      waveStatus.classList.add('is-success');
    }
  }

  function resetWavePanel() {
    if (wavePatternKey) {
      wavePatternKey.textContent = '—';
      wavePatternKey.classList.remove('is-bullish', 'is-bearish', 'is-neutral');
    }
    if (wavePatternBias) wavePatternBias.textContent = '—';
    if (wavePatternConfidence) wavePatternConfidence.textContent = '—';
    if (wavePatternHint) wavePatternHint.textContent = '—';
    if (waveWaveKey) waveWaveKey.textContent = '—';
    if (waveWaveStage) waveWaveStage.textContent = '—';
    if (waveWaveDirection) waveWaveDirection.textContent = '—';
    if (waveWaveConfidence) waveWaveConfidence.textContent = '—';
    if (waveSetupDirection) {
      waveSetupDirection.textContent = '—';
      waveSetupDirection.classList.remove('is-bullish', 'is-bearish', 'is-neutral');
    }
    if (waveSetupConfidence) waveSetupConfidence.textContent = '—';
    if (waveSetupNote) waveSetupNote.textContent = '—';
    if (waveDiagnostics) waveDiagnostics.textContent = '—';
    if (waveProbUp) waveProbUp.textContent = '—';
    if (waveProbDown) waveProbDown.textContent = '—';
    if (waveProbNeutral) waveProbNeutral.textContent = '—';
    if (waveProbUpFill) waveProbUpFill.style.width = '0%';
    if (waveProbDownFill) waveProbDownFill.style.width = '0%';
    if (waveProbNeutralFill) waveProbNeutralFill.style.width = '0%';
  }

  function updateWaveButtons() {
    const disabled = waveAnalyzeInFlight;
    if (waveAnalyzeRun) waveAnalyzeRun.disabled = disabled;
    if (waveSampleSave) waveSampleSave.disabled = disabled;
    if (waveTrainRun) waveTrainRun.disabled = disabled;
  }

  function normalizeWaveSeriesMode(value) {
    const mode = (value || '').toString().trim().toLowerCase();
    return WAVE_SERIES_MODES.has(mode) ? mode : 'close';
  }

  function normalizeWaveSmoothingWindow(value) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return 5;
    if (WAVE_SMOOTHING_WINDOWS.has(parsed)) return parsed;
    if (parsed <= 1) return 1;
    if (parsed < 3) return 3;
    if (parsed > 11) return 11;
    return parsed % 2 === 0 ? parsed - 1 : parsed;
  }

  function getWaveSeriesSettings() {
    const seriesMode = normalizeWaveSeriesMode(waveSeriesModeSelect ? waveSeriesModeSelect.value : 'close');
    const smoothingWindow = normalizeWaveSmoothingWindow(
      waveSmoothingWindowSelect ? waveSmoothingWindowSelect.value : '5'
    );
    if (waveSeriesModeSelect && waveSeriesModeSelect.value !== seriesMode) {
      waveSeriesModeSelect.value = seriesMode;
    }
    if (waveSmoothingWindowSelect && waveSmoothingWindowSelect.value !== String(smoothingWindow)) {
      waveSmoothingWindowSelect.value = String(smoothingWindow);
    }
    return { seriesMode, smoothingWindow };
  }

  function resolveWaveBarValue(bar, seriesMode) {
    if (!bar || typeof bar !== 'object') return null;
    const open = Number.isFinite(bar.open) ? bar.open : Number.parseFloat(bar.open);
    const high = Number.isFinite(bar.high) ? bar.high : Number.parseFloat(bar.high);
    const low = Number.isFinite(bar.low) ? bar.low : Number.parseFloat(bar.low);
    const close = Number.isFinite(bar.close) ? bar.close : Number.parseFloat(bar.close);
    if (!Number.isFinite(close)) return null;
    if (seriesMode === 'hlc3') {
      const hv = Number.isFinite(high) ? high : close;
      const lv = Number.isFinite(low) ? low : close;
      return (hv + lv + close) / 3;
    }
    if (seriesMode === 'ohlc4') {
      const ov = Number.isFinite(open) ? open : close;
      const hv = Number.isFinite(high) ? high : close;
      const lv = Number.isFinite(low) ? low : close;
      return (ov + hv + lv + close) / 4;
    }
    return close;
  }

  function smoothWaveSeries(values, windowSize) {
    if (!Array.isArray(values) || values.length <= 2) return values.slice();
    const normalizedWindow = normalizeWaveSmoothingWindow(windowSize);
    if (normalizedWindow <= 1 || values.length < normalizedWindow) return values.slice();
    const half = Math.floor(normalizedWindow / 2);
    const smoothed = [];
    for (let idx = 0; idx < values.length; idx += 1) {
      let sum = 0;
      let count = 0;
      for (let offset = -half; offset <= half; offset += 1) {
        const candidateIndex = Math.min(values.length - 1, Math.max(0, idx + offset));
        const candidate = values[candidateIndex];
        if (!Number.isFinite(candidate)) continue;
        sum += candidate;
        count += 1;
      }
      smoothed.push(count ? sum / count : values[idx]);
    }
    return smoothed;
  }

  function buildWaveAnalyzeTask() {
    if (!detailManager || !Array.isArray(detailManager.ohlcData) || !detailManager.ohlcData.length) {
      return null;
    }
    const bars = detailManager.ohlcData;
    const { seriesMode, smoothingWindow } = getWaveSeriesSettings();
    const seriesValues = [];
    const sourceIndexMap = [];
    bars.forEach((bar, idx) => {
      const value = resolveWaveBarValue(bar, seriesMode);
      if (!Number.isFinite(value)) return;
      seriesValues.push(value);
      sourceIndexMap.push(idx);
    });
    if (!seriesValues.length) return null;
    const workingSeries = smoothWaveSeries(seriesValues, smoothingWindow);
    let sampledSeries = workingSeries;
    let sampledIndexMap = sourceIndexMap;
    if (workingSeries.length > WAVE_ANALYZE_MAX_POINTS) {
      const sampled = [];
      const sampledIndices = [];
      const step = (workingSeries.length - 1) / (WAVE_ANALYZE_MAX_POINTS - 1);
      let previous = -1;
      for (let i = 0; i < WAVE_ANALYZE_MAX_POINTS; i += 1) {
        let nextIndex = Math.round(i * step);
        if (nextIndex <= previous) nextIndex = previous + 1;
        if (nextIndex >= workingSeries.length) nextIndex = workingSeries.length - 1;
        sampled.push(workingSeries[nextIndex]);
        sampledIndices.push(sourceIndexMap[nextIndex]);
        previous = nextIndex;
      }
      sampledSeries = sampled;
      sampledIndexMap = sampledIndices;
    }
    const firstBar = bars[0];
    const lastBar = bars[bars.length - 1];
    const firstTime = firstBar ? normalizeEpochSeconds(firstBar.time) : null;
    const lastTime = lastBar ? normalizeEpochSeconds(lastBar.time) : null;
    const fingerprint = [
      detailSymbol,
      detailRange,
      detailInterval,
      seriesMode,
      smoothingWindow,
      sampledSeries.length,
      Number.isFinite(firstTime) ? firstTime : 'na',
      Number.isFinite(lastTime) ? lastTime : 'na',
    ].join('|');
    return {
      symbol: detailSymbol,
      range: detailRange,
      interval: detailInterval,
      seriesMode,
      smoothingWindow,
      series: sampledSeries,
      indexMap: sampledIndexMap,
      fingerprint,
    };
  }

  function setWaveOverlayEnabled(enabled) {
    waveOverlayEnabled = Boolean(enabled);
    if (waveOverlayToggle) {
      waveOverlayToggle.checked = waveOverlayEnabled;
    }
    if (detailManager && typeof detailManager.setAnalysisOverlayVisible === 'function') {
      detailManager.setAnalysisOverlayVisible(waveOverlayEnabled);
    }
  }

  function resetWaveAnalysisState({ keepStatus = false } = {}) {
    waveAnalyzeGeneration += 1;
    if (waveAnalyzeTimer) {
      window.clearTimeout(waveAnalyzeTimer);
      waveAnalyzeTimer = null;
    }
    waveAnalyzeQueuedTask = null;
    waveLastFingerprint = '';
    waveLastTask = null;
    waveLastResult = null;
    resetWavePanel();
    if (!keepStatus) {
      setWaveStatusText(TEXT.waveStatusWaiting);
    }
    if (detailManager && typeof detailManager.clearAnalysisOverlay === 'function') {
      detailManager.clearAnalysisOverlay();
    }
  }

  function buildWaveOverlayPayload(result, task, indexMapOverride = null) {
    if (!result || !task || !detailManager || !Array.isArray(detailManager.ohlcData)) return null;
    const wave = result.wave;
    if (!wave || typeof wave !== 'object' || !Array.isArray(wave.pivots) || !wave.pivots.length) {
      return null;
    }
    const countLabels = new Map();
    if (Array.isArray(wave.count)) {
      wave.count.forEach((entry) => {
        if (!entry || typeof entry !== 'object') return;
        const pivotIndex = Number.parseInt(entry.pivot_index, 10);
        if (!Number.isFinite(pivotIndex)) return;
        const label = entry.label != null ? String(entry.label) : '';
        if (!label) return;
        countLabels.set(pivotIndex, label);
      });
    }
    const bars = detailManager.ohlcData;
    const resolvedIndexMap = Array.isArray(indexMapOverride) && indexMapOverride.length ? indexMapOverride : task.indexMap;
    const points = [];
    wave.pivots.forEach((pivot) => {
      if (!pivot || typeof pivot !== 'object') return;
      const pivotIdx = Number.parseInt(pivot.idx, 10);
      if (!Number.isFinite(pivotIdx)) return;
      const sourceIndex = resolvedIndexMap[pivotIdx];
      if (!Number.isFinite(sourceIndex) || sourceIndex < 0 || sourceIndex >= bars.length) return;
      const sourceBar = bars[sourceIndex];
      if (!sourceBar) return;
      const time = normalizeEpochSeconds(sourceBar.time);
      if (!Number.isFinite(time)) return;
      const pivotValue = resolveWaveBarValue(sourceBar, task.seriesMode || 'close');
      if (!Number.isFinite(pivotValue)) return;
      points.push({
        time,
        price: pivotValue,
        kind: pivot.kind || '',
        label: countLabels.get(pivotIdx) || '',
      });
    });
    if (!points.length) return null;
    return {
      direction: wave.direction || 'neutral',
      points,
    };
  }

  function setWaveToneClass(el, tone) {
    if (!el || !el.classList) return;
    el.classList.remove('is-bullish', 'is-bearish', 'is-neutral');
    if (tone === 'bullish' || tone === 'up') {
      el.classList.add('is-bullish');
    } else if (tone === 'bearish' || tone === 'down') {
      el.classList.add('is-bearish');
    } else {
      el.classList.add('is-neutral');
    }
  }

  function applyWaveAnalysisResult(result, task) {
    waveLastResult = result;
    waveLastTask = task;
    const biasLabels = TEXT.waveBiasLabels || {};
    const directionLabels = TEXT.waveDirectionLabels || {};
    const stageLabels = TEXT.waveStageLabels || {};
    const setupLabels = TEXT.waveSetupLabels || {};
    const modeLabels = TEXT.waveSeriesModeLabels || {};
    const patternKey = result && result.pattern_key ? String(result.pattern_key) : '';
    const biasKey = result && result.bias ? String(result.bias) : 'neutral';
    const confidenceVal = result && Number.isFinite(result.confidence) ? Number(result.confidence) : null;
    const wave = result && result.wave && typeof result.wave === 'object' ? result.wave : null;
    const diagnostics = result && result.diagnostics && typeof result.diagnostics === 'object' ? result.diagnostics : {};
    const seriesMode = normalizeWaveSeriesMode(diagnostics.series_mode || task.seriesMode || 'close');
    const smoothingWindow = normalizeWaveSmoothingWindow(
      diagnostics.smoothing_window || task.smoothingWindow || 1
    );

    if (wavePatternKey) wavePatternKey.textContent = humanizeWaveKey(patternKey);
    if (wavePatternBias) wavePatternBias.textContent = biasLabels[biasKey] || biasKey || '—';
    if (wavePatternConfidence) wavePatternConfidence.textContent = confidenceVal === null ? '—' : formatWavePercent(confidenceVal);
    setWaveToneClass(wavePatternKey, biasKey);
    if (wavePatternHint) {
      const baseHint = biasLabels[biasKey] || biasKey || 'neutral';
      const modeLabel = modeLabels[seriesMode] || seriesMode.toUpperCase();
      const smoothLabel = TEXT.waveSmoothLabel ? TEXT.waveSmoothLabel(smoothingWindow) : `smooth ${smoothingWindow}`;
      wavePatternHint.textContent =
        confidenceVal === null
          ? `${baseHint} · ${modeLabel} · ${smoothLabel}`
          : `${baseHint} · ${formatWavePercent(confidenceVal)} · ${modeLabel} · ${smoothLabel}`;
    }

    if (waveWaveKey) waveWaveKey.textContent = humanizeWaveKey(wave && wave.wave_key);
    if (waveWaveStage) {
      const stageKey = wave && wave.stage ? String(wave.stage) : 'unknown';
      waveWaveStage.textContent = stageLabels[stageKey] || stageKey || '—';
    }
    if (waveWaveDirection) {
      const directionKey = wave && wave.direction ? String(wave.direction) : 'neutral';
      waveWaveDirection.textContent = directionLabels[directionKey] || directionKey || '—';
    }
    if (waveWaveConfidence) {
      const waveConfidence = wave && Number.isFinite(wave.confidence) ? Number(wave.confidence) : null;
      waveWaveConfidence.textContent = waveConfidence === null ? '—' : formatWavePercent(waveConfidence);
    }

    const probabilities =
      (result && result.fused_probabilities && typeof result.fused_probabilities === 'object'
        ? result.fused_probabilities
        : null) ||
      (result && result.probabilities && typeof result.probabilities === 'object' ? result.probabilities : null) ||
      {};
    const upProb = Number.isFinite(probabilities.up) ? Number(probabilities.up) : null;
    const downProb = Number.isFinite(probabilities.down) ? Number(probabilities.down) : null;
    const neutralProb = Number.isFinite(probabilities.neutral) ? Number(probabilities.neutral) : null;
    if (waveProbUp) waveProbUp.textContent = formatWavePercent(upProb);
    if (waveProbDown) waveProbDown.textContent = formatWavePercent(downProb);
    if (waveProbNeutral) waveProbNeutral.textContent = formatWavePercent(neutralProb);
    if (waveProbUpFill) waveProbUpFill.style.width = Number.isFinite(upProb) ? `${Math.max(0, Math.min(100, upProb * 100))}%` : '0%';
    if (waveProbDownFill) waveProbDownFill.style.width = Number.isFinite(downProb) ? `${Math.max(0, Math.min(100, downProb * 100))}%` : '0%';
    if (waveProbNeutralFill) {
      waveProbNeutralFill.style.width = Number.isFinite(neutralProb)
        ? `${Math.max(0, Math.min(100, neutralProb * 100))}%`
        : '0%';
    }

    let setupKey = 'neutral';
    let setupConfidence = neutralProb;
    const upScore = Number.isFinite(upProb) ? upProb : 0;
    const downScore = Number.isFinite(downProb) ? downProb : 0;
    const neutralScore = Number.isFinite(neutralProb) ? neutralProb : 0;
    if (upScore >= downScore && upScore >= neutralScore) {
      setupKey = 'bullish';
      setupConfidence = upScore;
    } else if (downScore >= upScore && downScore >= neutralScore) {
      setupKey = 'bearish';
      setupConfidence = downScore;
    }
    if (waveSetupDirection) {
      waveSetupDirection.textContent = setupLabels[setupKey] || setupKey;
      setWaveToneClass(waveSetupDirection, setupKey);
    }
    if (waveSetupConfidence) {
      waveSetupConfidence.textContent = formatWavePercent(setupConfidence);
    }
    if (waveSetupNote) {
      const suggestedIntervalMs = Number(result && result.suggested_interval_ms);
      if (Number.isFinite(suggestedIntervalMs) && suggestedIntervalMs > 0) {
        const seconds = Math.max(1, Math.round(suggestedIntervalMs / 1000));
        waveSetupNote.textContent =
          langPrefix === 'zh' ? `建议分析间隔 ${seconds}s` : `Suggested cadence ${seconds}s`;
      } else {
        waveSetupNote.textContent =
          langPrefix === 'zh' ? '结合波浪方向与融合概率生成。' : 'Derived from wave direction and fused probabilities.';
      }
    }

    if (waveDiagnostics) {
      const pivotCount = wave && Array.isArray(wave.pivots) ? wave.pivots.length : 0;
      const samplePoints = Number(diagnostics.sample_points || task.series.length || 0) || 0;
      const originalPoints = Number(diagnostics.series_original_length || task.series.length || 0) || 0;
      const modelUsed =
        langPrefix === 'zh'
          ? diagnostics.model_used
            ? '模型:开'
            : '模型:关'
          : diagnostics.model_used
          ? 'model:on'
          : 'model:off';
      const modeLabel = modeLabels[seriesMode] || seriesMode.toUpperCase();
      const smoothLabel = TEXT.waveSmoothLabel ? TEXT.waveSmoothLabel(smoothingWindow) : `smooth ${smoothingWindow}`;
      waveDiagnostics.textContent =
        langPrefix === 'zh'
          ? `样本 ${samplePoints}/${originalPoints} · pivots ${pivotCount} · ${modeLabel} · ${smoothLabel} · ${modelUsed}`
          : `samples ${samplePoints}/${originalPoints} · pivots ${pivotCount} · ${modeLabel} · ${smoothLabel} · ${modelUsed}`;
    }

    if (detailManager) {
      const resultIndexMap =
        diagnostics && Array.isArray(diagnostics.sample_index_map) ? diagnostics.sample_index_map : null;
      const overlayPayload = buildWaveOverlayPayload(result, task, resultIndexMap);
      if (overlayPayload && typeof detailManager.setAnalysisOverlay === 'function') {
        detailManager.setAnalysisOverlay(overlayPayload);
        if (typeof detailManager.setAnalysisOverlayVisible === 'function') {
          detailManager.setAnalysisOverlayVisible(waveOverlayEnabled);
        }
      } else if (typeof detailManager.clearAnalysisOverlay === 'function') {
        detailManager.clearAnalysisOverlay();
      }
    }
  }

  async function runWaveAnalyze(task, { manual = false } = {}) {
    if (!task || !Array.isArray(task.series) || !task.series.length) return;
    const runGeneration = waveAnalyzeGeneration;
    if (!manual && task.fingerprint === waveLastFingerprint) return;
    if (task.series.length < WAVE_ANALYZE_MIN_POINTS) {
      setWaveStatusText(TEXT.waveStatusInsufficient(WAVE_ANALYZE_MIN_POINTS, task.series.length), 'error');
      return;
    }
    if (waveAnalyzeInFlight) {
      waveAnalyzeQueuedTask = { ...task, manual };
      return;
    }
    waveAnalyzeInFlight = true;
    updateWaveButtons();
    setWaveStatusText(TEXT.waveStatusAnalyzing);
    const endpointBase = chartAnalyzeApiUrl || '/api/market/chart/analyze/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
          'X-Requested-With': 'XMLHttpRequest',
        },
        credentials: 'same-origin',
        body: JSON.stringify({
          symbol: task.symbol,
          range: task.range,
          interval: task.interval,
          series: task.series,
          series_mode: task.seriesMode,
          smoothing_window: task.smoothingWindow,
          include_fusion: true,
        }),
      });
      const parsed = await parseApiResponse(response);
      const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
      if (runGeneration !== waveAnalyzeGeneration) {
        return;
      }
      if (!response.ok) {
        if (payload.error_code === 'series_insufficient') {
          const required = Number(payload.required_points || WAVE_ANALYZE_MIN_POINTS) || WAVE_ANALYZE_MIN_POINTS;
          const got = Number(payload.points || task.series.length) || task.series.length;
          setWaveStatusText(TEXT.waveStatusInsufficient(required, got), 'error');
        } else {
          setWaveStatusText(payload.message || payload.error || TEXT.waveStatusError, 'error');
        }
        return;
      }
      waveLastFingerprint = task.fingerprint;
      applyWaveAnalysisResult(payload, task);
      setWaveStatusText(TEXT.waveStatusReady, 'success');
    } catch (_error) {
      if (runGeneration !== waveAnalyzeGeneration) {
        return;
      }
      setWaveStatusText(TEXT.waveStatusError, 'error');
    } finally {
      waveAnalyzeInFlight = false;
      updateWaveButtons();
      if (waveAnalyzeQueuedTask) {
        const queued = waveAnalyzeQueuedTask;
        waveAnalyzeQueuedTask = null;
        runWaveAnalyze(queued, { manual: Boolean(queued.manual) });
      }
    }
  }

  function scheduleWaveAutoAnalyze() {
    if (!waveAutoEnabled || !chartVisible || !detailSymbol) return;
    const task = buildWaveAnalyzeTask();
    if (!task) return;
    if (task.fingerprint === waveLastFingerprint) return;
    waveAnalyzeQueuedTask = task;
    if (waveAnalyzeTimer) {
      window.clearTimeout(waveAnalyzeTimer);
    }
    waveAnalyzeTimer = window.setTimeout(() => {
      waveAnalyzeTimer = null;
      if (!waveAnalyzeQueuedTask) return;
      const queuedTask = waveAnalyzeQueuedTask;
      waveAnalyzeQueuedTask = null;
      runWaveAnalyze(queuedTask, { manual: false });
    }, WAVE_ANALYZE_DEBOUNCE_MS);
  }

  function triggerWaveManualAnalyze() {
    const task = buildWaveAnalyzeTask();
    if (!task) {
      setWaveStatusText(TEXT.waveStatusWaiting);
      return;
    }
    runWaveAnalyze(task, { manual: true });
  }

  async function refreshWaveMeta() {
    const endpointBase = chartAnalyzeMetaApiUrl || '/api/market/chart/analyze/meta/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    try {
      const response = await fetch(endpoint, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      const parsed = await parseApiResponse(response);
      const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
      if (!response.ok) return;
      const total = Number(payload.total_samples || 0) || 0;
      const classesObj = payload.classes && typeof payload.classes === 'object' ? payload.classes : {};
      const classCount = Object.keys(classesObj).length;
      const thresholdVal = Number(payload.override_threshold);
      const threshold = Number.isFinite(thresholdVal) ? thresholdVal.toFixed(2) : '—';
      if (waveTrainMeta) {
        waveTrainMeta.textContent = TEXT.waveMeta(total, classCount, threshold);
      }
      waveMetaLoaded = true;
    } catch (_error) {
      // ignore meta failures
    }
  }

  async function saveWaveSample() {
    if (!waveSampleLabel) return;
    const label = (waveSampleLabel.value || '').trim();
    if (!label) {
      setWaveStatusText(TEXT.waveStatusLabelRequired, 'error');
      return;
    }
    const task = waveLastTask || buildWaveAnalyzeTask();
    if (!task || !Array.isArray(task.series) || !task.series.length) {
      setWaveStatusText(TEXT.waveStatusWaiting, 'error');
      return;
    }
    setWaveStatusText(TEXT.waveStatusSaving);
    const endpointBase = chartAnalyzeSampleApiUrl || '/api/market/chart/analyze/sample/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
          'X-Requested-With': 'XMLHttpRequest',
        },
        credentials: 'same-origin',
        body: JSON.stringify({
          symbol: task.symbol,
          range: task.range,
          interval: task.interval,
          series: task.series,
          series_mode: task.seriesMode,
          smoothing_window: task.smoothingWindow,
          label,
        }),
      });
      const parsed = await parseApiResponse(response);
      const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
      if (!response.ok) {
        setWaveStatusText(payload.message || payload.error || TEXT.waveStatusError, 'error');
        return;
      }
      const total = Number(payload.total_samples || 0) || 0;
      setWaveStatusText(TEXT.waveStatusSaved(total), 'success');
      await refreshWaveMeta();
    } catch (_error) {
      setWaveStatusText(TEXT.waveStatusError, 'error');
    }
  }

  async function trainWaveModel() {
    setWaveStatusText(TEXT.waveStatusTraining);
    const endpointBase = chartAnalyzeTrainApiUrl || '/api/market/chart/analyze/train/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
          'X-Requested-With': 'XMLHttpRequest',
        },
        credentials: 'same-origin',
        body: JSON.stringify({}),
      });
      const parsed = await parseApiResponse(response);
      const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
      if (!response.ok) {
        setWaveStatusText(payload.message || payload.error || TEXT.waveStatusError, 'error');
        return;
      }
      const accuracy = Number(payload.accuracy);
      setWaveStatusText(TEXT.waveStatusTrained(accuracy), 'success');
      await refreshWaveMeta();
    } catch (_error) {
      setWaveStatusText(TEXT.waveStatusError, 'error');
    }
  }

  function formatAxisTime(
    epochSeconds,
    {
      timezoneMode: mode = 'utc',
      showSeconds = false,
      includeDate = true,
      fullDate = false,
      dateOnly = false,
    } = {}
  ) {
    const normalized = normalizeEpochSeconds(epochSeconds);
    if (!Number.isFinite(normalized)) return '';
    if (fullDate) {
      const parts = getDateParts(normalized, mode);
      if (!parts) return '';
      if (langPrefix === 'zh') {
        const dateText = `${parts.year}/${String(parts.month).padStart(2, '0')}/${String(parts.day).padStart(2, '0')}`;
        const timeText = `${String(parts.hour).padStart(2, '0')}:${String(parts.minute).padStart(2, '0')}${
          showSeconds ? `:${String(parts.second).padStart(2, '0')}` : ''
        }`;
        return `${dateText} ${timeText}`;
      }
      const weekdayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const weekday = weekdayNames[parts.weekday] || '';
      const month = monthNames[parts.month - 1] || '';
      const yearShort = String(parts.year).slice(-2);
      const timeText = `${String(parts.hour).padStart(2, '0')}:${String(parts.minute).padStart(2, '0')}${
        showSeconds ? `:${String(parts.second).padStart(2, '0')}` : ''
      }`;
      return `${weekday} ${String(parts.day).padStart(2, '0')} ${month} '${yearShort} ${timeText}`.trim();
    }
    const ms = normalized * 1000;
    const date = new Date(ms);
    const options = dateOnly
      ? {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
        }
      : includeDate
        ? {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
          }
        : {
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

  function getDateParts(epochSeconds, mode) {
    const ms = epochSeconds * 1000;
    const date = new Date(ms);
    if (Number.isNaN(date.getTime())) return null;
    if (mode === 'utc') {
      return {
        year: date.getUTCFullYear(),
        month: date.getUTCMonth() + 1,
        day: date.getUTCDate(),
        weekday: date.getUTCDay(),
        hour: date.getUTCHours(),
        minute: date.getUTCMinutes(),
        second: date.getUTCSeconds(),
      };
    }
    return {
      year: date.getFullYear(),
      month: date.getMonth() + 1,
      day: date.getDate(),
      weekday: date.getDay(),
      hour: date.getHours(),
      minute: date.getMinutes(),
      second: date.getSeconds(),
    };
  }

  function normalizeEpochSeconds(value) {
    if (value == null) return null;
    if (value instanceof Date) {
      const ms = value.getTime();
      return Number.isFinite(ms) ? ms / 1000 : null;
    }
    if (typeof value === 'object') {
      const year = Number(value.year);
      const month = Number(value.month);
      const day = Number(value.day);
      if (Number.isFinite(year) && Number.isFinite(month) && Number.isFinite(day)) {
        const ts = Date.UTC(year, month - 1, day) / 1000;
        return Number.isFinite(ts) ? ts : null;
      }
    }
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) return null;
      if (value > 1e15) return value / 1e9;
      if (value > 1e12) return value / 1e3;
      return value;
    }
    if (typeof value === 'string') {
      const raw = value.trim();
      if (!raw) return null;
      if (/^\d+(\.\d+)?$/.test(raw)) {
        return normalizeEpochSeconds(Number.parseFloat(raw));
      }
      let normalized = raw;
      if (normalized.endsWith('UTC')) {
        normalized = normalized.replace(' UTC', 'Z');
      }
      if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}/.test(normalized) && !normalized.includes('T')) {
        normalized = normalized.replace(' ', 'T');
      }
      if (/^\d{4}\/\d{2}\/\d{2}/.test(normalized)) {
        normalized = normalized.replace(/\//g, '-');
      }
      if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(normalized)) {
        normalized = `${normalized}Z`;
      }
      const parsed = Date.parse(normalized);
      if (!Number.isNaN(parsed)) {
        return parsed / 1000;
      }
      const fallback = Number.parseFloat(raw);
      return Number.isFinite(fallback) ? normalizeEpochSeconds(fallback) : null;
    }
    const numeric = Number.parseFloat(value);
    if (!Number.isFinite(numeric)) return null;
    if (numeric > 1e15) return numeric / 1e9;
    if (numeric > 1e12) return numeric / 1e3;
    return numeric;
  }

  const TZ_PARTS_FORMATTER = new Intl.DateTimeFormat('en-US', {
    timeZone: MARKET_TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });

  function getTimeZoneParts(timestampMs, formatter) {
    if (!Number.isFinite(timestampMs)) return null;
    const parts = (formatter || TZ_PARTS_FORMATTER).formatToParts(new Date(timestampMs));
    const out = {};
    parts.forEach((part) => {
      if (part.type !== 'literal') {
        out[part.type] = part.value;
      }
    });
    return out;
  }

  function getTimeZoneOffsetMs(timestampMs, timeZone) {
    try {
      const formatter =
        timeZone && timeZone !== MARKET_TIMEZONE
          ? new Intl.DateTimeFormat('en-US', {
              timeZone,
              year: 'numeric',
              month: '2-digit',
              day: '2-digit',
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              hour12: false,
            })
          : TZ_PARTS_FORMATTER;
      const parts = getTimeZoneParts(timestampMs, formatter);
      if (!parts) return 0;
      const asUtc = Date.UTC(
        Number(parts.year),
        Number(parts.month) - 1,
        Number(parts.day),
        Number(parts.hour),
        Number(parts.minute),
        Number(parts.second)
      );
      return asUtc - timestampMs;
    } catch (error) {
      return 0;
    }
  }

  function toUtcTimestampInZone(year, month, day, hour, minute, timeZone) {
    const guess = Date.UTC(year, month - 1, day, hour, minute, 0);
    const offset = getTimeZoneOffsetMs(guess, timeZone || MARKET_TIMEZONE);
    return (guess - offset) / 1000;
  }

  function getEtDateKey(epochSeconds) {
    const ts = normalizeEpochSeconds(epochSeconds);
    if (!Number.isFinite(ts)) return null;
    const parts = getTimeZoneParts(ts * 1000, TZ_PARTS_FORMATTER);
    if (parts && parts.year) {
      return `${parts.year}-${parts.month}-${parts.day}`;
    }
    const ms = ts * 1000;
    const offset = getTimeZoneOffsetMs(ms, MARKET_TIMEZONE);
    const etDate = new Date(ms + offset);
    const year = etDate.getUTCFullYear();
    const month = etDate.getUTCMonth() + 1;
    const day = etDate.getUTCDate();
    if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) return null;
    const pad2 = (value) => String(value).padStart(2, '0');
    return `${year}-${pad2(month)}-${pad2(day)}`;
  }

  function getEtMinutes(epochSeconds) {
    const ts = normalizeEpochSeconds(epochSeconds);
    if (!Number.isFinite(ts)) return null;
    const parts = getTimeZoneParts(ts * 1000, TZ_PARTS_FORMATTER);
    if (parts && parts.hour != null && parts.minute != null) {
      const hour = Number(parts.hour);
      const minute = Number(parts.minute);
      if (Number.isFinite(hour) && Number.isFinite(minute)) {
        return hour * 60 + minute;
      }
    }
    const ms = ts * 1000;
    const offset = getTimeZoneOffsetMs(ms, MARKET_TIMEZONE);
    const etDate = new Date(ms + offset);
    const hour = etDate.getUTCHours();
    const minute = etDate.getUTCMinutes();
    if (!Number.isFinite(hour) || !Number.isFinite(minute)) return null;
    return hour * 60 + minute;
  }

  function classifyEtSession(minutes) {
    if (!Number.isFinite(minutes)) return 'regular';
    if (minutes < 9 * 60 + 30) return 'pre';
    if (minutes >= 16 * 60 && minutes <= 20 * 60) return 'post';
    return 'regular';
  }

  function formatLatencyValue(ms) {
    if (!Number.isFinite(ms)) return '';
    const absMs = Math.max(0, ms);
    if (absMs < 1000) {
      return `${Math.round(absMs)}ms`;
    }
    const seconds = absMs / 1000;
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    }
    const minutes = seconds / 60;
    if (minutes < 60) {
      return langPrefix === 'zh' ? `${minutes.toFixed(1)}分` : `${minutes.toFixed(1)}m`;
    }
    const hours = minutes / 60;
    return langPrefix === 'zh' ? `${hours.toFixed(1)}小时` : `${hours.toFixed(1)}h`;
  }

  function formatAge(seconds) {
    if (!Number.isFinite(seconds)) return '';
    const abs = Math.max(0, seconds);
    if (abs < 60) {
      return langPrefix === 'zh' ? `${Math.round(abs)}秒` : `${Math.round(abs)}s`;
    }
    const minutes = abs / 60;
    if (minutes < 60) {
      return langPrefix === 'zh' ? `${minutes.toFixed(1)}分` : `${minutes.toFixed(1)}m`;
    }
    const hours = minutes / 60;
    if (hours < 24) {
      return langPrefix === 'zh' ? `${hours.toFixed(1)}小时` : `${hours.toFixed(1)}h`;
    }
    const days = hours / 24;
    return langPrefix === 'zh' ? `${days.toFixed(1)}天` : `${days.toFixed(1)}d`;
  }

  function withAutoRangeLock(fn) {
    detailAutoRangeLock = true;
    try {
      fn();
    } finally {
      window.setTimeout(() => {
        detailAutoRangeLock = false;
      }, 0);
    }
  }

  class ChartManager {
    static registry = [];

    constructor({ container, indicatorContainer, indicatorSecondaryContainer, langPrefix, onStatus, timezoneMode }) {
      this.container = container;
      this.indicatorContainer = indicatorContainer;
      this.indicatorSecondaryContainer = indicatorSecondaryContainer;
      this.langPrefix = langPrefix || 'zh';
      this.onStatus = onStatus;
      this.timezoneMode = timezoneMode || 'utc';
      this.chart = null;
      this.candleSeries = null;
      this.lineSeries = null;
      this.overlaySeries = [];
      this.indicatorChart = null;
      this.indicatorSecondaryChart = null;
      this.indicatorSeries = [];
      this.indicatorSecondarySeries = [];
      this.volumeSeries = null;
      this.volumeScaleId = 'volume';
      this.overlayMode = 'none';
      this.indicatorMode = 'none';
      this.priceSeriesMode = 'candlestick';
      this.ohlcData = [];
      this.overlayCanvas = null;
      this.overlayCtx = null;
      this.overlayRatio = 1;
      this.drawings = [];
      this.activeDrawing = null;
      this.drawMode = 'none';
      this.analysisOverlay = null;
      this.analysisOverlayVisible = true;
      this.priceLine = null;
      this.tooltipEl = null;
      this.lastLivePrice = null;
      this.intervalSpec = null;
      this.axisShowSeconds = false;
      this.axisIncludeDate = true;
      this.axisFullDate = false;
      this.axisMode = 'time';
      this.tickEpochBase = 0;
      this.tickIndex = 0;
      this.tickIndexMap = new Map();
      this.tickTimeMap = new Map();
      this.tickIndexCounter = 0;
      this.liveBar = null;
      this.liveBucket = null;
      this.liveTickCount = 0;
      this.liveTickTarget = 0;
      this.liveMaxBars = 800;
      this.historyMaxBars = STANDARD_HISTORY_MAX_BARS;
      this.lastLiveTime = null;
      this.sessionBands = [];
      this._syncing = false;
      this._syncingIndicator = false;
      this._linkedCharts = new Set();
      this._syncTargets = new Set();
      this._resizeObserver = null;
      this.pulseEnabled = true;
      this._pulseFrame = null;
      this._pulseLastRender = 0;
      this._pulseBase = typeof performance !== 'undefined' ? performance.now() : Date.now();
      this.lockPriceScale = true;
      this._priceScaleLocked = false;
      this.manualPriceRange = null;
      this._manualAutoscaleProvider = null;
      this._priceScaleWheelBound = false;
      this._pulseMarkerTime = null;
      this._pulseMarkerSeries = null;
      this._panBound = false;
      this._panActive = false;
      this._panPointerId = null;
      this._panStart = null;
      this._panStartLogicalRange = null;
      this._panStartPriceRange = null;
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
        rightPriceScale: {
          borderColor: 'rgba(148, 163, 184, 0.4)',
          minimumWidth: PRICE_SCALE_MIN_WIDTH,
        },
        timeScale: {
          borderColor: 'rgba(148, 163, 184, 0.4)',
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (timestamp) => this._formatAxisTime(timestamp, { context: 'axis' }),
        },
        localization: {
          locale,
          timeFormatter: (timestamp) =>
            this._formatAxisTime(timestamp, {
              context: 'tooltip',
              includeDate: true,
              fullDate: true,
            }),
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
        crosshair: {
          mode: 1,
          vertLine: {
            color: 'rgba(59, 130, 246, 0.35)',
            width: 1,
            style: 1,
            labelBackgroundColor: '#1e293b',
          },
          horzLine: {
            color: 'rgba(59, 130, 246, 0.35)',
            width: 1,
            style: 1,
            labelBackgroundColor: '#1e293b',
          },
        },
      };
      this.chart = chartLib.createChart(this.container, {
        ...baseOptions,
        width: this.container.clientWidth || 680,
        height: this.container.clientHeight || 360,
      });
      this._applyPriceScaleMargins();
      this.candleSeries = this.chart.addCandlestickSeries({
        upColor: '#16a34a',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#16a34a',
        wickDownColor: '#ef4444',
        priceFormat: CHART_PRICE_FORMAT,
      });
      this.lineSeries = this.chart.addLineSeries({
        color: '#2563eb',
        lineWidth: 2,
        priceFormat: CHART_PRICE_FORMAT,
      });
      if (this.lineSeries) {
        this.lineSeries.applyOptions({ visible: false });
      }
      this._wrapSeriesDiagnostics(this.candleSeries, 'candle');
      this._wrapSeriesDiagnostics(this.lineSeries, 'line');
      this._applyManualAutoscaleProvider();

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
      if (this.indicatorSecondaryContainer) {
        const indicatorWidth = this.indicatorSecondaryContainer.clientWidth || this.container.clientWidth || 680;
        const indicatorHeight = this.indicatorSecondaryContainer.clientHeight || 140;
        this.indicatorSecondaryChart = chartLib.createChart(this.indicatorSecondaryContainer, {
          ...baseOptions,
          width: indicatorWidth,
          height: indicatorHeight,
          timeScale: { borderColor: 'rgba(148, 163, 184, 0.4)', visible: false },
        });
      }

      this._initOverlay();
      this._initTooltip();
      this._bindResize();
      this._bindTimeSync();
      this._bindPriceScaleWheel();
      this._bindPanDrag();
      this._wrapTimeScaleDiagnostics();
      ChartManager.register(this);
      return true;
    }

    _wrapSeriesDiagnostics(series, label) {
      if (!series || !isDiagEnabled() || series.__diagWrapped) return;
      const wrap = (method) => {
        const original = series[method];
        if (typeof original !== 'function') return;
        series[method] = (...args) => {
          try {
            return original.apply(series, args);
          } catch (error) {
            diagLog(`${label}.${method}`, {
              error: String(error),
              payload: summarizePayload(args[0]),
              context: captureDiagContext(this, { label, method }),
            });
            throw error;
          }
        };
      };
      wrap('update');
      wrap('setData');
      series.__diagWrapped = true;
    }

    _wrapTimeScaleDiagnostics() {
      if (!isDiagEnabled() || !this.chart || !this.chart.timeScale) return;
      const timeScale = this.chart.timeScale();
      if (!timeScale || timeScale.__diagWrapped) return;
      const wrap = (method) => {
        const original = timeScale[method];
        if (typeof original !== 'function') return;
        timeScale[method] = (...args) => {
          try {
            return original.apply(timeScale, args);
          } catch (error) {
            diagLog(`timeScale.${method}`, {
              error: String(error),
              range: args[0],
              ohlcSummary: summarizeBars(this.ohlcData),
              context: captureDiagContext(this, { method }),
            });
            throw error;
          }
        };
      };
      wrap('setVisibleRange');
      wrap('setVisibleLogicalRange');
      wrap('fitContent');
      timeScale.__diagWrapped = true;
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
        if (!Number.isFinite(range.from) || !Number.isFinite(range.to)) return;
        this._syncing = true;
        target._syncing = true;
        target.chart.timeScale().setVisibleRange(range);
        target._syncing = false;
        this._syncing = false;
      };
      this.chart.timeScale().subscribeVisibleTimeRangeChange(syncRange);
    }

    _syncIndicatorRanges() {
      const mainScale = this.chart && this.chart.timeScale ? this.chart.timeScale() : null;
      if (!mainScale) return;
      const logicalRange =
        typeof mainScale.getVisibleLogicalRange === 'function' ? mainScale.getVisibleLogicalRange() : null;
      const timeRange = typeof mainScale.getVisibleRange === 'function' ? mainScale.getVisibleRange() : null;
      const targets = [this.indicatorChart, this.indicatorSecondaryChart];
      targets.forEach((targetChart) => {
        if (!targetChart || !targetChart.timeScale) return;
        const targetScale = targetChart.timeScale();
        if (!targetScale) return;
        if (
          logicalRange &&
          Number.isFinite(logicalRange.from) &&
          Number.isFinite(logicalRange.to) &&
          typeof targetScale.setVisibleLogicalRange === 'function'
        ) {
          withAutoRangeLock(() => {
            targetScale.setVisibleLogicalRange(logicalRange);
          });
          return;
        }
        if (
          timeRange &&
          Number.isFinite(timeRange.from) &&
          Number.isFinite(timeRange.to) &&
          typeof targetScale.setVisibleRange === 'function'
        ) {
          withAutoRangeLock(() => {
            targetScale.setVisibleRange(timeRange);
          });
        }
      });
    }

    _syncPriceScaleWidths() {
      const charts = [this.chart, this.indicatorChart, this.indicatorSecondaryChart].filter(Boolean);
      if (!charts.length) return;
      let maxWidth = PRICE_SCALE_MIN_WIDTH;
      charts.forEach((chartInstance) => {
        const scale =
          chartInstance && chartInstance.priceScale && typeof chartInstance.priceScale === 'function'
            ? chartInstance.priceScale('right')
            : null;
        if (!scale || typeof scale.width !== 'function') return;
        const width = Number(scale.width());
        if (Number.isFinite(width) && width > maxWidth) {
          maxWidth = width;
        }
      });
      const targetWidth = Math.max(PRICE_SCALE_MIN_WIDTH, Math.ceil(maxWidth));
      charts.forEach((chartInstance) => {
        const scale =
          chartInstance && chartInstance.priceScale && typeof chartInstance.priceScale === 'function'
            ? chartInstance.priceScale('right')
            : null;
        if (!scale || typeof scale.applyOptions !== 'function') return;
        scale.applyOptions({ minimumWidth: targetWidth });
      });
    }

    _syncPaneAlignment() {
      this._syncPriceScaleWidths();
      this._syncIndicatorRanges();
    }

    setData(bars, options = {}) {
      const { intervalSpec = null, fitContent = true } = options || {};
      const timeScale = this.chart && this.chart.timeScale ? this.chart.timeScale() : null;
      const logicalRange =
        !fitContent && timeScale && timeScale.getVisibleLogicalRange ? timeScale.getVisibleLogicalRange() : null;
      this._resetLiveState();
      this.lastLivePrice = null;
      this.ohlcData = this._sanitizeBars(bars);
      if (intervalSpec) {
        this.setIntervalSpec(intervalSpec, { preserveData: true });
      }
      this._updateSessionBands();
      this._rebuildTickIndexMap();
      if (this.candleSeries) {
        this.candleSeries.setData(this.ohlcData);
      }
      if (this.lineSeries) {
        this.lineSeries.setData(this._mapLineSeries());
      }
      this.updateOverlay();
      this.updateIndicator();
      if (this.chart && fitContent) {
        withAutoRangeLock(() => {
          this.chart.timeScale().fitContent();
        });
      } else if (logicalRange && timeScale && timeScale.setVisibleLogicalRange) {
        if (Number.isFinite(logicalRange.from) && Number.isFinite(logicalRange.to)) {
          withAutoRangeLock(() => {
            timeScale.setVisibleLogicalRange(logicalRange);
          });
        }
      }
      if (this.ohlcData.length) {
        const last = this.ohlcData[this.ohlcData.length - 1];
        if (last && typeof last.close === 'number') {
          this.updatePriceLine(last.close);
        }
      }
      if (this.lockPriceScale) {
        this._lockPriceScale({ allowAutoFit: Boolean(fitContent) });
      }
      this._syncPaneAlignment();
      this.renderOverlay();
    }

    mergeData(bars) {
      const cleaned = this._sanitizeBars(bars);
      if (!cleaned.length) return false;
      if (!Array.isArray(this.ohlcData) || !this.ohlcData.length) {
        this.setData(cleaned);
        return true;
      }
      const prevLen = this.ohlcData.length;
      const prevLastIndex = prevLen - 1;
      const timeScale = this.chart && this.chart.timeScale ? this.chart.timeScale() : null;
      const logicalRange = timeScale && timeScale.getVisibleLogicalRange ? timeScale.getVisibleLogicalRange() : null;
      const merged = [];
      let i = 0;
      let j = 0;
      const existing = this.ohlcData;
      while (i < existing.length && j < cleaned.length) {
        const left = existing[i];
        const right = cleaned[j];
        if (!left || !right) {
          if (left) {
            merged.push(left);
            i += 1;
          } else if (right) {
            merged.push(right);
            j += 1;
          } else {
            i += 1;
            j += 1;
          }
          continue;
        }
        if (left.time < right.time - 1e-6) {
          merged.push(left);
          i += 1;
        } else if (right.time < left.time - 1e-6) {
          merged.push(right);
          j += 1;
        } else {
          merged.push(right);
          i += 1;
          j += 1;
        }
      }
      while (i < existing.length) {
        merged.push(existing[i]);
        i += 1;
      }
      while (j < cleaned.length) {
        merged.push(cleaned[j]);
        j += 1;
      }
      this.ohlcData = merged;
      if (this.historyMaxBars > 0 && this.ohlcData.length > this.historyMaxBars) {
        this.ohlcData = detailUserPanned
          ? this.ohlcData.slice(0, this.historyMaxBars)
          : this.ohlcData.slice(-this.historyMaxBars);
      }
      this._updateSessionBands();
      const newLen = this.ohlcData.length;
      const delta = newLen - prevLen;
      if (this.candleSeries) {
        this.candleSeries.setData(this.ohlcData);
      }
      if (this.lineSeries) {
        this.lineSeries.setData(this._mapLineSeries());
      }
      this._updateVolumeSeries();
      if (this.intervalSpec && this.intervalSpec.unit === 'tick') {
        this._rebuildTickIndexMap();
      }
      this.updateOverlay();
      this.updateIndicator();
      if (this.ohlcData.length) {
        const last = this.ohlcData[this.ohlcData.length - 1];
        if (last && typeof last.close === 'number') {
          this.updatePriceLine(last.close);
        }
      }
      this._syncLiveStateFromData(this.intervalSpec);
      if (logicalRange && timeScale && timeScale.setVisibleLogicalRange && Number.isFinite(delta) && delta > 0) {
        if (!Number.isFinite(logicalRange.from) || !Number.isFinite(logicalRange.to)) {
          this._syncPaneAlignment();
          return true;
        }
        const shouldStickToRight = Number.isFinite(prevLastIndex) && logicalRange.to >= prevLastIndex - 1e-6;
        if (shouldStickToRight && !detailUserPanned) {
          withAutoRangeLock(() => {
            timeScale.setVisibleLogicalRange({
              from: logicalRange.from + delta,
              to: logicalRange.to + delta,
            });
          });
        }
      }
      this._syncPaneAlignment();
      return true;
    }

    setIntervalSpec(intervalSpec, { preserveData = false } = {}) {
      this.intervalSpec = intervalSpec;
      this._resetLiveState();
      if (intervalSpec && preserveData) {
        this._syncLiveStateFromData(intervalSpec);
      }
      // Use line series for all intervals to match TradingView-style line + volume layout.
      this.setSeriesMode('line');
      const isTick = Boolean(intervalSpec && intervalSpec.unit === 'tick');
      const isSecond = Boolean(intervalSpec && intervalSpec.unit === 'second');
      this.historyMaxBars = isTick || isSecond ? HIGH_FREQ_HISTORY_MAX_BARS : STANDARD_HISTORY_MAX_BARS;
      this.liveMaxBars = isTick || isSecond ? this.historyMaxBars : 800;
      this.axisMode = isTick ? 'tick' : 'time';
      const showSeconds = Boolean(intervalSpec && (isTick || isSecond));
      const isIntraday = Boolean(intervalSpec && intervalSpec.unit !== 'day');
      const includeDate = isIntraday;
      const fullDate = isIntraday;
      this.setAxisOptions({ showSeconds, timeVisible: isIntraday, includeDate, fullDate });
      if (isTick) {
        this._rebuildTickIndexMap();
      }
      if (this.chart) {
        this._applyLocalization(this.chart);
      }
      if (this.indicatorChart) {
        this._applyLocalization(this.indicatorChart);
      }
      if (this.indicatorSecondaryChart) {
        this._applyLocalization(this.indicatorSecondaryChart);
      }
      this._updateSessionBands();
    }

    setAxisOptions({ showSeconds = false, timeVisible = true, includeDate = null, fullDate = null } = {}) {
      this.axisShowSeconds = Boolean(showSeconds);
      if (includeDate === null) {
        this.axisIncludeDate = !timeVisible;
      } else {
        this.axisIncludeDate = Boolean(includeDate);
      }
      if (fullDate === null) {
        this.axisFullDate = false;
      } else {
        this.axisFullDate = Boolean(fullDate);
      }
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
      if (this.indicatorSecondaryChart) {
        this._applyLocalization(this.indicatorSecondaryChart);
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
      if (this.indicatorSecondaryChart) {
        this._applyLocalization(this.indicatorSecondaryChart);
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

    setAnalysisOverlay(payload) {
      if (!payload || typeof payload !== 'object' || !Array.isArray(payload.points)) {
        this.analysisOverlay = null;
        this.renderOverlay();
        return;
      }
      const points = payload.points
        .map((point) => {
          if (!point || typeof point !== 'object') return null;
          const time = normalizeEpochSeconds(point.time);
          const price = Number.isFinite(point.price) ? point.price : Number.parseFloat(point.price);
          if (!Number.isFinite(time) || !Number.isFinite(price)) return null;
          return {
            time,
            price,
            kind: point.kind || '',
            label: point.label ? String(point.label) : '',
          };
        })
        .filter(Boolean);
      this.analysisOverlay = points.length
        ? {
            points,
            direction: payload.direction || 'neutral',
          }
        : null;
      this.renderOverlay();
    }

    clearAnalysisOverlay() {
      this.analysisOverlay = null;
      this.renderOverlay();
    }

    setAnalysisOverlayVisible(visible) {
      this.analysisOverlayVisible = Boolean(visible);
      this.renderOverlay();
    }

    updatePriceLine(price) {
      const series = this.priceSeriesMode === 'line' ? this.lineSeries : this.candleSeries;
      if (!series || typeof price !== 'number' || Number.isNaN(price)) return;
      const color = this.lastLivePrice !== null && price < this.lastLivePrice ? '#ef4444' : '#16a34a';
      if (!this.priceLine) {
        this.priceLine = series.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle: 2,
          axisLabelVisible: true,
          title: this.langPrefix === 'zh' ? '最新' : 'Last',
        });
      } else {
        this.priceLine.applyOptions({ price, color });
      }
      this.lastLivePrice = price;
    }

    updateLivePrice(price, ts, size = 0) {
      if (typeof price !== 'number' || Number.isNaN(price)) return;
      const normalizedTs = normalizeEpochSeconds(ts);
      const timestamp = Number.isFinite(normalizedTs) ? normalizedTs : Date.now() / 1000;
      this.applyTradeUpdate({ price, size, ts: timestamp });
    }

    applyTradeUpdate(trade) {
      if (!trade || typeof trade.price !== 'number' || Number.isNaN(trade.price)) return;
      if (!this.candleSeries && !this.lineSeries) return;
      const normalizedTs = normalizeEpochSeconds(trade.ts);
      const timestamp = Number.isFinite(normalizedTs) ? normalizedTs : Date.now() / 1000;
      const price = trade.price;
      const size = Number.isFinite(trade.size) ? trade.size : 0;
      this.updatePriceLine(price);
      const interval = this.intervalSpec;
      if (!interval) return;
      if (interval.unit === 'tick') {
        this._applyTickTrade({ price, size, ts: timestamp }, interval.value);
        return;
      }
      const intervalSeconds = interval.seconds || 1;
      this._applyTimeTrade({ time: timestamp, price, size }, intervalSeconds);
    }

    updateCurrentBar(price, ts, intervalSec) {
      if (!this.candleSeries && !this.lineSeries) return;
      if (!Array.isArray(this.ohlcData) || !this.ohlcData.length) return;
      if (typeof price !== 'number' || Number.isNaN(price)) return;
      const interval = Math.max(1, Number.parseInt(intervalSec, 10) || 0);
      if (!interval) return;
      const normalizedTs = normalizeEpochSeconds(ts);
      const timestamp = Number.isFinite(normalizedTs) ? normalizedTs : Date.now() / 1000;
      const bucket = Math.floor(timestamp / interval) * interval;
      const lastIndex = this.ohlcData.length - 1;
      const last = this.ohlcData[lastIndex];
      if (!last || typeof last.time !== 'number') return;
      if (bucket < last.time - 1e-6) return;
      if (bucket === last.time) {
        const updated = { ...last };
        const nextOpen = typeof updated.open === 'number' ? updated.open : price;
        const nextHigh = typeof updated.high === 'number' ? Math.max(updated.high, price) : price;
        const nextLow = typeof updated.low === 'number' ? Math.min(updated.low, price) : price;
        updated.open = nextOpen;
        updated.high = nextHigh;
        updated.low = nextLow;
        updated.close = price;
        if (this._upsertTimeBar(updated)) {
          this.liveBar = { ...updated };
          this.liveBucket = bucket;
          this.lastLiveTime = bucket;
        }
        return;
      }
      const bar = { time: bucket, open: price, high: price, low: price, close: price };
      if (this._upsertTimeBar(bar)) {
        this.liveBar = { ...bar };
        this.liveBucket = bucket;
        this.lastLiveTime = bucket;
      }
    }

    _pushLiveBar(bar) {
      this.ohlcData.push(bar);
      if (this.ohlcData.length > this.liveMaxBars) {
        this.ohlcData = detailUserPanned
          ? this.ohlcData.slice(0, this.liveMaxBars)
          : this.ohlcData.slice(-this.liveMaxBars);
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
            this._formatAxisTime(timestamp, {
              context: 'tooltip',
              includeDate: true,
              fullDate: true,
            }),
        },
        timeScale: {
          tickMarkFormatter: (timestamp) => this._formatAxisTime(timestamp, { context: 'axis' }),
        },
      });
    }

    _getVisibleRangeSeconds() {
      const timeScale = this.chart && this.chart.timeScale ? this.chart.timeScale() : null;
      if (!timeScale || typeof timeScale.getVisibleRange !== 'function') return null;
      const range = timeScale.getVisibleRange();
      if (!range) return null;
      const from = normalizeEpochSeconds(range.from);
      const to = normalizeEpochSeconds(range.to);
      if (!Number.isFinite(from) || !Number.isFinite(to)) return null;
      return Math.abs(to - from);
    }

    _resolveAxisLabelOptions(options) {
      const resolved = { ...options };
      const intervalSpec = this.intervalSpec;
      if (intervalSpec && intervalSpec.unit === 'day') {
        resolved.dateOnly = true;
        resolved.includeDate = true;
        resolved.fullDate = false;
        resolved.showSeconds = false;
        return resolved;
      }
      const span = this._getVisibleRangeSeconds();
      if (!Number.isFinite(span)) return resolved;
      const daySeconds = 86400;
      if (span >= daySeconds * 2) {
        resolved.dateOnly = true;
        resolved.includeDate = true;
        resolved.fullDate = false;
        resolved.showSeconds = false;
      } else {
        resolved.includeDate = false;
        resolved.fullDate = false;
      }
      return resolved;
    }

    _resolveTickLabelStep() {
      const timeScale = this.chart && this.chart.timeScale ? this.chart.timeScale() : null;
      const width = this.container ? this.container.clientWidth || 0 : 0;
      if (!timeScale || width <= 0 || typeof timeScale.getVisibleLogicalRange !== 'function') return 1;
      const logicalRange = timeScale.getVisibleLogicalRange();
      if (!logicalRange || !Number.isFinite(logicalRange.from) || !Number.isFinite(logicalRange.to)) return 1;
      const span = Math.max(1, logicalRange.to - logicalRange.from);
      const spacing = width / span;
      if (spacing < 3) return 100;
      if (spacing < 4) return 50;
      if (spacing < 6) return 20;
      if (spacing < 8) return 10;
      if (spacing < 12) return 5;
      if (spacing < 20) return 2;
      return 1;
    }

    _formatTickAxisLabel(timestamp, options) {
      const tickIndex = this._resolveTickIndex(timestamp);
      if (tickIndex === null) return '';
      const step = this._resolveTickLabelStep();
      if (step > 1 && tickIndex % step !== 0) return '';
      const timeScale = this.chart && this.chart.timeScale ? this.chart.timeScale() : null;
      const width = this.container ? this.container.clientWidth || 0 : 0;
      let spacing = null;
      if (timeScale && width > 0 && typeof timeScale.getVisibleLogicalRange === 'function') {
        const logicalRange = timeScale.getVisibleLogicalRange();
        if (logicalRange && Number.isFinite(logicalRange.from) && Number.isFinite(logicalRange.to)) {
          const span = Math.max(1, logicalRange.to - logicalRange.from);
          spacing = width / span;
        }
      }
      const tickTime = this._resolveTickTime(timestamp);
      const showSeconds = spacing !== null ? spacing >= 10 : true;
      const label = formatAxisTime(tickTime, {
        ...options,
        includeDate: false,
        fullDate: false,
        dateOnly: false,
        showSeconds,
      });
      if (spacing !== null && spacing < 6) {
        return `T${tickIndex}`;
      }
      return label || `T${tickIndex}`;
    }

    _formatAxisTime(timestamp, overrides = {}) {
      const normalized = normalizeEpochSeconds(timestamp);
      if (!Number.isFinite(normalized)) return '';
      const { context, ...restOverrides } = overrides || {};
      let options = {
        timezoneMode: this.timezoneMode,
        showSeconds: this.axisShowSeconds,
        includeDate: this.axisIncludeDate,
        fullDate: this.axisFullDate,
        ...restOverrides,
      };
      if (this.axisMode !== 'tick') {
        if (context === 'axis') {
          options = this._resolveAxisLabelOptions(options);
        }
        return formatAxisTime(normalized, options);
      }
      if (context === 'axis') {
        return this._formatTickAxisLabel(normalized, options);
      }
      const tickIndex = this._resolveTickIndex(normalized);
      const tickTime = this._resolveTickTime(normalized);
      const timeLabel = formatAxisTime(tickTime, {
        ...options,
        showSeconds: true,
      });
      if (tickIndex === null) {
        return timeLabel;
      }
      return timeLabel ? `${timeLabel} · T${tickIndex}` : `T${tickIndex}`;
    }

    _resetLiveState() {
      this.liveBar = null;
      this.liveBucket = null;
      this.liveTickCount = 0;
      this.liveTickTarget = 0;
      this.lastLiveTime = null;
    }

    _tickKey(time) {
      const normalized = normalizeEpochSeconds(time);
      if (!Number.isFinite(normalized)) return '';
      return normalized.toFixed(6);
    }

    _rebuildTickIndexMap() {
      this.tickIndexMap.clear();
      this.tickTimeMap.clear();
      this.tickIndexCounter = 0;
      if (!Array.isArray(this.ohlcData) || !this.ohlcData.length) return;
      this.ohlcData.forEach((bar, idx) => {
        const key = this._tickKey(bar.time);
        if (!key) return;
        this.tickIndexMap.set(key, idx);
        this.tickTimeMap.set(idx, bar.time);
        this.tickIndexCounter = idx + 1;
      });
    }

    _assignTickIndex(bar) {
      if (!bar) return null;
      const key = this._tickKey(bar.time);
      if (!key) return null;
      if (!this.tickIndexMap.has(key)) {
        this.tickIndexMap.set(key, this.tickIndexCounter);
        this.tickTimeMap.set(this.tickIndexCounter, bar.time);
        this.tickIndexCounter += 1;
      }
      const idx = this.tickIndexMap.get(key);
      if (idx !== undefined) {
        this.tickTimeMap.set(idx, bar.time);
        return idx;
      }
      return null;
    }

    _resolveTickIndex(time) {
      const key = this._tickKey(time);
      if (key && this.tickIndexMap.has(key)) {
        return this.tickIndexMap.get(key);
      }
      if (Number.isFinite(time) && time >= 0 && time < this.ohlcData.length) {
        return Math.round(time);
      }
      if (!Array.isArray(this.ohlcData) || !this.ohlcData.length) return null;
      if (Number.isFinite(time)) {
        const bucket = Math.floor(time);
        for (let i = this.ohlcData.length - 1; i >= 0; i -= 1) {
          const bar = this.ohlcData[i];
          if (!bar || typeof bar.time !== 'number') continue;
          if (Math.floor(bar.time) === bucket) {
            return i;
          }
          if (bar.time < time - 1) {
            break;
          }
        }
      }
      for (let i = this.ohlcData.length - 1; i >= 0; i -= 1) {
        const bar = this.ohlcData[i];
        if (!bar || typeof bar.time !== 'number') continue;
        if (bar.time <= time + 1e-6) {
          return i;
        }
      }
      return null;
    }

    _resolveTickTime(time) {
      const idx = this._resolveTickIndex(time);
      if (idx !== null && this.tickTimeMap.has(idx)) {
        return this.tickTimeMap.get(idx);
      }
      if (idx !== null && this.ohlcData[idx] && Number.isFinite(this.ohlcData[idx].time)) {
        return this.ohlcData[idx].time;
      }
      return time;
    }

    _resetTickAxis() {
      this.tickIndex = 0;
      this.tickEpochBase = Math.floor(Date.now() / 1000);
    }

    _applyTickTimeIndex() {
      if (!this.ohlcData.length) return;
      const base = Math.floor(Date.now() / 1000) - this.ohlcData.length;
      this.tickEpochBase = base;
      this.tickIndex = this.ohlcData.length;
      this.ohlcData = this.ohlcData.map((bar, idx) => ({
        ...bar,
        time: base + idx,
      }));
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
        this.tickIndex = this.ohlcData.length;
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

    _applyTickTrade({ price, size, ts }, ticksPerBar) {
      const target = Math.max(1, Number.parseInt(ticksPerBar, 10) || 1);
      const timeSource = Number.isFinite(ts) ? ts : Date.now() / 1000;
      if (!this.liveBar || this.liveTickCount >= target) {
        const time = this._ensureMonotonicTime(timeSource);
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

    _nextTickTime() {
      if (!Number.isFinite(this.tickEpochBase)) {
        this.tickEpochBase = Math.floor(Date.now() / 1000);
      }
      const time = this.tickEpochBase + this.tickIndex;
      this.tickIndex += 1;
      return time;
    }

    _applyTimeTrade({ time, price, size }, intervalSeconds) {
      if (!Number.isFinite(time) || !Number.isFinite(price)) return false;
      const bucket = Math.floor(time / intervalSeconds) * intervalSeconds;
      if (!Number.isFinite(bucket)) return false;
      const last = Array.isArray(this.ohlcData) && this.ohlcData.length
        ? this.ohlcData[this.ohlcData.length - 1]
        : null;
      if (last && Number.isFinite(last.time) && bucket < last.time - 1e-6) {
        return false;
      }
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
        const upserted = this._upsertTimeBar(bar);
        if (upserted) {
          this.liveBar = { ...bar };
          this.liveBucket = bucket;
          this.lastLiveTime = bucket;
        }
        return upserted;
      }
      this.liveBar.high = Math.max(this.liveBar.high, price);
      this.liveBar.low = Math.min(this.liveBar.low, price);
      this.liveBar.close = price;
      this.liveBar.volume = (this.liveBar.volume || 0) + size;
      this.liveBar.trade_count = (this.liveBar.trade_count || 0) + 1;
      const upserted = this._upsertTimeBar(this.liveBar);
      if (upserted) {
        this.lastLiveTime = bucket;
      }
      return upserted;
    }

    _appendOrUpdate(bar, isNew) {
      if (!this.candleSeries && !this.lineSeries) return;
      if (
        !bar ||
        !Number.isFinite(bar.time) ||
        !Number.isFinite(bar.open) ||
        !Number.isFinite(bar.high) ||
        !Number.isFinite(bar.low) ||
        !Number.isFinite(bar.close)
      ) {
        return;
      }
      if (!Array.isArray(this.ohlcData)) {
        this.ohlcData = [];
      }
      const lastIndex = this.ohlcData.length - 1;
      const last = this.ohlcData[lastIndex];
      if (isNew || !last) {
        this.ohlcData.push({ ...bar });
        this._trimLiveBars();
        if (this.intervalSpec && this.intervalSpec.unit === 'tick') {
          this._assignTickIndex(bar);
        }
        if (this.candleSeries) {
          this.candleSeries.update({ ...bar });
        }
        if (this.lineSeries) {
          this.lineSeries.update({ time: bar.time, value: bar.close });
        }
        this._updateVolumeSeriesEntry(bar);
        return;
      }
      this.ohlcData[lastIndex] = { ...bar };
      if (this.intervalSpec && this.intervalSpec.unit === 'tick') {
        this._assignTickIndex(bar);
      }
      if (this.candleSeries) {
        this.candleSeries.update({ ...bar });
      }
      if (this.lineSeries) {
        this.lineSeries.update({ time: bar.time, value: bar.close });
      }
      this._updateVolumeSeriesEntry(bar);
    }

    _upsertTimeBar(bar) {
      if (!this.candleSeries && !this.lineSeries) return false;
      if (
        !bar ||
        !Number.isFinite(bar.time) ||
        !Number.isFinite(bar.open) ||
        !Number.isFinite(bar.high) ||
        !Number.isFinite(bar.low) ||
        !Number.isFinite(bar.close)
      ) {
        return false;
      }
      if (!Array.isArray(this.ohlcData)) {
        this.ohlcData = [];
      }
      const lastIndex = this.ohlcData.length - 1;
      const last = this.ohlcData[lastIndex];
      if (!last) {
        this.ohlcData.push({ ...bar });
        this._trimLiveBars();
        if (this.candleSeries) {
          this.candleSeries.update({ ...bar });
        }
        if (this.lineSeries) {
          this.lineSeries.update({ time: bar.time, value: bar.close });
        }
        this._updateVolumeSeriesEntry(bar);
        return true;
      }
      if (!Number.isFinite(last.time)) {
        this.ohlcData[lastIndex] = { ...bar };
        if (this.candleSeries) {
          this.candleSeries.update({ ...bar });
        }
        if (this.lineSeries) {
          this.lineSeries.update({ time: bar.time, value: bar.close });
        }
        this._updateVolumeSeriesEntry(bar);
        return true;
      }
      if (bar.time < last.time - 1e-6) {
        return false;
      }
      if (Math.abs(bar.time - last.time) <= 1e-6) {
        this.ohlcData[lastIndex] = { ...bar };
      } else {
        this.ohlcData.push({ ...bar });
        this._trimLiveBars();
      }
      if (this.candleSeries) {
        this.candleSeries.update({ ...bar });
      }
      if (this.lineSeries) {
        this.lineSeries.update({ time: bar.time, value: bar.close });
      }
      this._updateVolumeSeriesEntry(bar);
      return true;
    }

    _trimLiveBars() {
      if (this.ohlcData.length <= this.liveMaxBars) return;
      this.ohlcData = detailUserPanned
        ? this.ohlcData.slice(0, this.liveMaxBars)
        : this.ohlcData.slice(-this.liveMaxBars);
      if (this.candleSeries) {
        this.candleSeries.setData(this.ohlcData);
      }
      if (this.lineSeries) {
        this.lineSeries.setData(this._mapLineSeries());
      }
      this._updateVolumeSeries();
      if (this.intervalSpec && this.intervalSpec.unit === 'tick') {
        this._rebuildTickIndexMap();
      }
    }

    prependData(bars) {
      if (!Array.isArray(bars) || !bars.length) return false;
      const cleaned = this._sanitizeBars(bars);
      if (!cleaned.length) return false;
      const prevLen = this.ohlcData ? this.ohlcData.length : 0;
      const earliest = this.ohlcData && this.ohlcData.length ? this.ohlcData[0].time : null;
      const filtered =
        Number.isFinite(earliest) ? cleaned.filter((bar) => bar.time < earliest - 1e-6) : cleaned;
      if (!filtered.length) return false;
      const timeScale = this.chart ? this.chart.timeScale() : null;
      const logicalRange = timeScale && timeScale.getVisibleLogicalRange ? timeScale.getVisibleLogicalRange() : null;
      this.ohlcData = filtered.concat(this.ohlcData || []);
      if (this.historyMaxBars > 0 && this.ohlcData.length > this.historyMaxBars) {
        this.ohlcData = detailUserPanned
          ? this.ohlcData.slice(0, this.historyMaxBars)
          : this.ohlcData.slice(-this.historyMaxBars);
      }
      this._updateSessionBands();
      const newLen = this.ohlcData.length;
      const delta = newLen - prevLen;
      if (this.candleSeries) {
        this.candleSeries.setData(this.ohlcData);
      }
      if (this.lineSeries) {
        this.lineSeries.setData(this._mapLineSeries());
      }
      this._updateVolumeSeries();
      if (this.intervalSpec && this.intervalSpec.unit === 'tick') {
        this._rebuildTickIndexMap();
      }
      if (this.overlayMode && this.overlayMode !== 'none') {
        this.updateOverlay();
      }
      if (this.indicatorMode && this.indicatorMode !== 'none') {
        this.updateIndicator();
      }
      this._syncLiveStateFromData(this.intervalSpec);
      if (
        logicalRange &&
        timeScale &&
        timeScale.setVisibleLogicalRange &&
        Number.isFinite(delta) &&
        delta > 0
      ) {
        withAutoRangeLock(() => {
          timeScale.setVisibleLogicalRange({
            from: logicalRange.from + delta,
            to: logicalRange.to + delta,
          });
        });
      }
      this._syncPaneAlignment();
      return true;
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
      if (this.indicatorSecondaryChart && this.indicatorSecondaryContainer) {
        const width = this.indicatorSecondaryContainer.clientWidth || 0;
        const height = this.indicatorSecondaryContainer.clientHeight || 0;
        if (width > 0 && height > 0) {
          this.indicatorSecondaryChart.applyOptions({ width, height });
        }
      }
      this._syncPaneAlignment();
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

      const modeAliases = {
        sma: 'sma20',
        ema: 'ema20',
        bbands: 'bbands20',
      };
      const mode = modeAliases[this.overlayMode] || this.overlayMode;
      const lineStyle = (color, width = 2) => this.chart.addLineSeries({ color, lineWidth: width });

      if (mode === 'sma20' || mode === 'sma50') {
        const period = mode === 'sma50' ? 50 : 20;
        const values = indicatorLib.SMA.calculate({ period, values: closes });
        const color = period === 50 ? '#2563eb' : '#0ea5e9';
        const series = lineStyle(color, 2);
        series.setData(this._mapSeries(values, period));
        this.overlaySeries.push(series);
      } else if (mode === 'ema20' || mode === 'ema50') {
        const period = mode === 'ema50' ? 50 : 20;
        const values = indicatorLib.EMA.calculate({ period, values: closes });
        const color = period === 50 ? '#e11d48' : '#f59e0b';
        const series = lineStyle(color, 2);
        series.setData(this._mapSeries(values, period));
        this.overlaySeries.push(series);
      } else if (mode === 'vwap') {
        const vwapData = [];
        let cumulativeVolume = 0;
        let cumulativePv = 0;
        this.ohlcData.forEach((bar) => {
          if (!bar || !Number.isFinite(bar.time)) return;
          const close = Number.isFinite(bar.close) ? bar.close : Number.parseFloat(bar.close);
          const high = Number.isFinite(bar.high) ? bar.high : close;
          const low = Number.isFinite(bar.low) ? bar.low : close;
          const volume = Number.isFinite(bar.volume) ? bar.volume : Number.parseFloat(bar.volume);
          if (!Number.isFinite(close) || !Number.isFinite(volume) || volume <= 0) return;
          const typical = (high + low + close) / 3;
          cumulativeVolume += volume;
          cumulativePv += typical * volume;
          if (cumulativeVolume <= 0) return;
          vwapData.push({ time: bar.time, value: cumulativePv / cumulativeVolume });
        });
        const series = lineStyle('#0f766e', 2);
        series.setData(vwapData);
        this.overlaySeries.push(series);
      } else if (mode === 'bbands20') {
        const values = indicatorLib.BollingerBands.calculate({ period: 20, values: closes, stdDev: 2 });
        const upper = lineStyle('rgba(59, 130, 246, 0.86)', 1);
        const middle = lineStyle('rgba(14, 165, 233, 0.92)', 1);
        const lower = lineStyle('rgba(59, 130, 246, 0.86)', 1);
        upper.setData(this._mapBand(values, 20, 'upper'));
        middle.setData(this._mapBand(values, 20, 'middle'));
        lower.setData(this._mapBand(values, 20, 'lower'));
        this.overlaySeries.push(upper, middle, lower);
      }
    }

    updateIndicator() {
      this._ensureVolumeSeries();
      this._updateVolumeSeries();
      const clearPane = (chart, series, container) => {
        if (chart && Array.isArray(series)) {
          series.forEach((item) => {
            try {
              chart.removeSeries(item);
            } catch (_error) {
              // ignore stale series handles
            }
          });
        }
        if (Array.isArray(series)) series.length = 0;
        if (container) container.hidden = true;
      };
      clearPane(this.indicatorChart, this.indicatorSeries, this.indicatorContainer);
      clearPane(this.indicatorSecondaryChart, this.indicatorSecondarySeries, this.indicatorSecondaryContainer);
      if (!this.ohlcData.length || this.indicatorMode === 'none') return;

      const indicatorLib = getIndicatorLib();
      if (!indicatorLib || !indicatorLib.RSI || !indicatorLib.MACD) return;
      const closes = this._getCloses();
      const highs = this.ohlcData.map((bar) => {
        const high = Number(bar && bar.high);
        const close = Number(bar && bar.close);
        return Number.isFinite(high) ? high : close;
      });
      const lows = this.ohlcData.map((bar) => {
        const low = Number(bar && bar.low);
        const close = Number(bar && bar.close);
        return Number.isFinite(low) ? low : close;
      });
      if (!closes.length) return;

      const modeAliases = { macd_rsi: 'rsi_macd' };
      const mode = modeAliases[this.indicatorMode] || this.indicatorMode;
      const primaryMode = mode === 'rsi_macd' ? 'rsi' : mode;
      const secondaryMode = mode === 'rsi_macd' ? 'macd' : 'none';

      const renderMode = (chart, seriesStore, modeKey, container) => {
        if (!chart || !Array.isArray(seriesStore) || !container || modeKey === 'none') return false;
        let rendered = false;
        if (modeKey === 'rsi') {
          const values = indicatorLib.RSI.calculate({ period: 14, values: closes });
          const line = chart.addLineSeries({ color: '#6366f1', lineWidth: 2 });
          line.setData(this._mapSeriesAligned(values, 14));
          try {
            line.createPriceLine({ price: 70, color: 'rgba(220, 38, 38, 0.45)', lineWidth: 1, lineStyle: 2 });
            line.createPriceLine({ price: 30, color: 'rgba(22, 163, 74, 0.45)', lineWidth: 1, lineStyle: 2 });
          } catch (_error) {
            // optional
          }
          seriesStore.push(line);
          rendered = true;
        } else if (modeKey === 'macd') {
          const values = indicatorLib.MACD.calculate({
            values: closes,
            fastPeriod: 12,
            slowPeriod: 26,
            signalPeriod: 9,
            SimpleMAOscillator: false,
            SimpleMASignal: false,
          });
          const startIndex = this.ohlcData.length - values.length;
          if (startIndex < 0) return false;
          const histogram = chart.addHistogramSeries({ color: '#94a3b8' });
          const macdLine = chart.addLineSeries({ color: '#0ea5e9', lineWidth: 2 });
          const signalLine = chart.addLineSeries({ color: '#f97316', lineWidth: 2 });
          const histogramData = [];
          const macdData = [];
          const signalData = [];
          for (let barIndex = 0; barIndex < this.ohlcData.length; barIndex += 1) {
            const bar = this.ohlcData[barIndex];
            if (!bar || !Number.isFinite(bar.time)) continue;
            const valueIndex = barIndex - startIndex;
            if (valueIndex < 0 || valueIndex >= values.length) {
              histogramData.push({ time: bar.time });
              macdData.push({ time: bar.time });
              signalData.push({ time: bar.time });
              continue;
            }
            const item = values[valueIndex];
            const histValue =
              item && Number.isFinite(item.histogram) ? item.histogram : Number.parseFloat(item && item.histogram);
            const macdValue = item && Number.isFinite(item.MACD) ? item.MACD : Number.parseFloat(item && item.MACD);
            const signalValue =
              item && Number.isFinite(item.signal) ? item.signal : Number.parseFloat(item && item.signal);
            if (Number.isFinite(histValue)) {
              histogramData.push({
                time: bar.time,
                value: histValue,
                color: histValue >= 0 ? 'rgba(34, 197, 94, 0.65)' : 'rgba(239, 68, 68, 0.65)',
              });
            } else {
              histogramData.push({ time: bar.time });
            }
            if (Number.isFinite(macdValue)) {
              macdData.push({ time: bar.time, value: macdValue });
            } else {
              macdData.push({ time: bar.time });
            }
            if (Number.isFinite(signalValue)) {
              signalData.push({ time: bar.time, value: signalValue });
            } else {
              signalData.push({ time: bar.time });
            }
          }
          histogram.setData(histogramData);
          macdLine.setData(macdData);
          signalLine.setData(signalData);
          seriesStore.push(histogram, macdLine, signalLine);
          rendered = true;
        } else if (modeKey === 'stoch' && indicatorLib.Stochastic && highs.length && lows.length) {
          const values = indicatorLib.Stochastic.calculate({
            high: highs,
            low: lows,
            close: closes,
            period: 14,
            signalPeriod: 3,
          });
          const startIndex = this.ohlcData.length - values.length;
          if (startIndex < 0) return false;
          const kLine = chart.addLineSeries({ color: '#2563eb', lineWidth: 2 });
          const dLine = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5 });
          const kData = [];
          const dData = [];
          for (let barIndex = 0; barIndex < this.ohlcData.length; barIndex += 1) {
            const bar = this.ohlcData[barIndex];
            if (!bar || !Number.isFinite(bar.time)) continue;
            const valueIndex = barIndex - startIndex;
            if (valueIndex < 0 || valueIndex >= values.length) {
              kData.push({ time: bar.time });
              dData.push({ time: bar.time });
              continue;
            }
            const item = values[valueIndex];
            const kVal = Number.parseFloat(item && item.k);
            const dVal = Number.parseFloat(item && item.d);
            if (Number.isFinite(kVal)) {
              kData.push({ time: bar.time, value: kVal });
            } else {
              kData.push({ time: bar.time });
            }
            if (Number.isFinite(dVal)) {
              dData.push({ time: bar.time, value: dVal });
            } else {
              dData.push({ time: bar.time });
            }
          }
          kLine.setData(kData);
          dLine.setData(dData);
          try {
            kLine.createPriceLine({ price: 80, color: 'rgba(220, 38, 38, 0.45)', lineWidth: 1, lineStyle: 2 });
            kLine.createPriceLine({ price: 20, color: 'rgba(22, 163, 74, 0.45)', lineWidth: 1, lineStyle: 2 });
          } catch (_error) {
            // optional
          }
          seriesStore.push(kLine, dLine);
          rendered = true;
        } else if (modeKey === 'atr' && indicatorLib.ATR && highs.length && lows.length) {
          const values = indicatorLib.ATR.calculate({ high: highs, low: lows, close: closes, period: 14 });
          const line = chart.addLineSeries({ color: '#7c3aed', lineWidth: 2 });
          line.setData(this._mapSeriesAligned(values, 14));
          seriesStore.push(line);
          rendered = true;
        }
        container.hidden = !rendered;
        return rendered;
      };

      const primaryRendered = renderMode(
        this.indicatorChart,
        this.indicatorSeries,
        primaryMode,
        this.indicatorContainer
      );
      const secondaryRendered = renderMode(
        this.indicatorSecondaryChart,
        this.indicatorSecondarySeries,
        secondaryMode,
        this.indicatorSecondaryContainer
      );
      if (!primaryRendered && !secondaryRendered) return;
      this.resize();
      this._syncPaneAlignment();
    }

    _getCloses() {
      return this.ohlcData.map((bar) => bar.close).filter((val) => Number.isFinite(val));
    }

    setSeriesMode(mode) {
      const nextMode = mode === 'line' ? 'line' : 'candlestick';
      this.priceSeriesMode = nextMode;
      if (this.candleSeries) {
        this.candleSeries.applyOptions({ visible: nextMode === 'candlestick' });
      }
      if (this.lineSeries) {
        this.lineSeries.applyOptions({ visible: nextMode === 'line' });
      }
      if (this.priceLine) {
        try {
          if (this.candleSeries && this.candleSeries.removePriceLine) {
            this.candleSeries.removePriceLine(this.priceLine);
          }
        } catch (error) {
          // noop
        }
        try {
          if (this.lineSeries && this.lineSeries.removePriceLine) {
            this.lineSeries.removePriceLine(this.priceLine);
          }
        } catch (error) {
          // noop
        }
      }
      this.priceLine = null;
      if (this.lastLivePrice !== null) {
        this.updatePriceLine(this.lastLivePrice);
      }
    }

    resetZoom() {
      if (this.chart) {
        withAutoRangeLock(() => {
          this.chart.timeScale().fitContent();
        });
      }
      if (this.lockPriceScale) {
        this._lockPriceScale({ allowAutoFit: true });
      }
      if (this.indicatorChart) {
        withAutoRangeLock(() => {
          this.indicatorChart.timeScale().fitContent();
        });
      }
      if (this.indicatorSecondaryChart) {
        withAutoRangeLock(() => {
          this.indicatorSecondaryChart.timeScale().fitContent();
        });
      }
      this._syncPaneAlignment();
      detailUserPanned = false;
    }

    _lockPriceScale({ allowAutoFit = false } = {}) {
      if (!this.chart || !this.chart.priceScale) return;
      if (allowAutoFit) {
        this._clearManualPriceRange();
      }
      if (!this.lockPriceScale) return;
      if (this.manualPriceRange && !allowAutoFit) return;
      const range = this._computeVisiblePriceRange();
      if (range) {
        this._setManualPriceRange(range);
      }
    }

    _getActivePriceSeries() {
      if (this.priceSeriesMode === 'line') {
        return this.lineSeries || this.candleSeries;
      }
      return this.candleSeries || this.lineSeries;
    }

    _applyManualAutoscaleProvider() {
      if (!this.chart) return;
      if (!this._manualAutoscaleProvider) {
        this._manualAutoscaleProvider = () => {
          if (!this.manualPriceRange) return null;
          const { min, max } = this.manualPriceRange;
          if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return null;
          return { priceRange: { minValue: min, maxValue: max } };
        };
      }
      if (this.candleSeries) {
        this.candleSeries.applyOptions({ autoscaleInfoProvider: this._manualAutoscaleProvider });
      }
      if (this.lineSeries) {
        this.lineSeries.applyOptions({ autoscaleInfoProvider: this._manualAutoscaleProvider });
      }
      if (this.chart && this.chart.priceScale) {
        const priceScale = this.chart.priceScale('right');
        if (priceScale && typeof priceScale.applyOptions === 'function') {
          priceScale.applyOptions({ autoScale: true });
        }
      }
    }

    _setManualPriceRange(range) {
      if (!range || !Number.isFinite(range.min) || !Number.isFinite(range.max)) return;
      if (range.min === range.max) return;
      this.manualPriceRange = { min: range.min, max: range.max };
      this._applyManualAutoscaleProvider();
      this._priceScaleLocked = true;
    }

    _clearManualPriceRange() {
      this.manualPriceRange = null;
      this._applyManualAutoscaleProvider();
      this._priceScaleLocked = false;
    }

    _computeVisiblePriceRange() {
      if (!this.chart || !Array.isArray(this.ohlcData) || !this.ohlcData.length) return null;
      const timeScale = this.chart.timeScale && this.chart.timeScale();
      const visible = timeScale && typeof timeScale.getVisibleRange === 'function' ? timeScale.getVisibleRange() : null;
      const from = visible && Number.isFinite(visible.from) ? visible.from : null;
      const to = visible && Number.isFinite(visible.to) ? visible.to : null;
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      this.ohlcData.forEach((bar) => {
        if (!bar || !Number.isFinite(bar.time)) return;
        if (from !== null && bar.time < from) return;
        if (to !== null && bar.time > to) return;
        const low = Number.isFinite(bar.low) ? bar.low : Number.isFinite(bar.close) ? bar.close : null;
        const high = Number.isFinite(bar.high) ? bar.high : Number.isFinite(bar.close) ? bar.close : null;
        if (!Number.isFinite(low) || !Number.isFinite(high)) return;
        if (low < min) min = low;
        if (high > max) max = high;
      });
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        return null;
      }
      if (min === max) {
        const base = min;
        const delta = Math.max(1e-6, Math.abs(base) * 0.001);
        return { min: base - delta, max: base + delta };
      }
      return { min, max };
    }

    _mapLineSeries() {
      return this.ohlcData
        .filter((bar) => Number.isFinite(bar.time) && Number.isFinite(bar.close))
        .map((bar) => ({ time: bar.time, value: bar.close }));
    }

    _mapSeries(values, period) {
      const offset = Math.max(0, period - 1);
      const points = [];
      values.forEach((value, index) => {
        const bar = this.ohlcData[index + offset];
        const numericValue = Number.isFinite(value) ? value : Number.parseFloat(value);
        if (!bar || !Number.isFinite(bar.time) || !Number.isFinite(numericValue)) return;
        points.push({ time: bar.time, value: numericValue });
      });
      return points;
    }

    _mapSeriesAligned(values, period) {
      const offset = Math.max(0, period - 1);
      const points = [];
      for (let barIndex = 0; barIndex < this.ohlcData.length; barIndex += 1) {
        const bar = this.ohlcData[barIndex];
        if (!bar || !Number.isFinite(bar.time)) continue;
        const valueIndex = barIndex - offset;
        if (valueIndex < 0 || valueIndex >= values.length) {
          points.push({ time: bar.time });
          continue;
        }
        const raw = values[valueIndex];
        const numeric = Number.isFinite(raw) ? raw : Number.parseFloat(raw);
        if (!Number.isFinite(numeric)) {
          points.push({ time: bar.time });
          continue;
        }
        points.push({ time: bar.time, value: numeric });
      }
      return points;
    }

    _mapBand(values, period, key) {
      const offset = Math.max(0, period - 1);
      const points = [];
      values.forEach((value, index) => {
        const bar = this.ohlcData[index + offset];
        const bandValue = value ? value[key] : null;
        const numericValue = Number.isFinite(bandValue) ? bandValue : Number.parseFloat(bandValue);
        if (!bar || !Number.isFinite(bar.time) || !Number.isFinite(numericValue)) return;
        points.push({ time: bar.time, value: numericValue });
      });
      return points;
    }

    _mapBandAligned(values, period, key) {
      const offset = Math.max(0, period - 1);
      const points = [];
      for (let barIndex = 0; barIndex < this.ohlcData.length; barIndex += 1) {
        const bar = this.ohlcData[barIndex];
        if (!bar || !Number.isFinite(bar.time)) continue;
        const valueIndex = barIndex - offset;
        if (valueIndex < 0 || valueIndex >= values.length) {
          points.push({ time: bar.time });
          continue;
        }
        const item = values[valueIndex];
        const bandRaw = item && typeof item === 'object' ? item[key] : null;
        const numeric = Number.isFinite(bandRaw) ? bandRaw : Number.parseFloat(bandRaw);
        if (!Number.isFinite(numeric)) {
          points.push({ time: bar.time });
          continue;
        }
        points.push({ time: bar.time, value: numeric });
      }
      return points;
    }

    _updateVolumeSeries() {
      this._ensureVolumeSeries();
      if (!this.volumeSeries) return;
      const data = [];
      this.ohlcData.forEach((bar) => {
        const volumeRaw = Number.isFinite(bar.volume) ? bar.volume : Number.parseFloat(bar.volume);
        if (!Number.isFinite(bar.time)) return;
        const volume = Number.isFinite(volumeRaw) ? volumeRaw : 0;
        const isUp = Number.isFinite(bar.close) && Number.isFinite(bar.open) ? bar.close >= bar.open : true;
        data.push({
          time: bar.time,
          value: volume,
          color: isUp ? 'rgba(16, 185, 129, 0.55)' : 'rgba(239, 68, 68, 0.55)',
        });
      });
      this.volumeSeries.setData(data);
    }

    _updateVolumeSeriesEntry(bar) {
      if (!bar) return;
      this._ensureVolumeSeries();
      if (!this.volumeSeries) return;
      const volumeRaw = Number.isFinite(bar.volume) ? bar.volume : Number.parseFloat(bar.volume);
      if (!Number.isFinite(bar.time)) return;
      const volume = Number.isFinite(volumeRaw) ? volumeRaw : 0;
      const isUp = Number.isFinite(bar.close) && Number.isFinite(bar.open) ? bar.close >= bar.open : true;
      this.volumeSeries.update({
        time: bar.time,
        value: volume,
        color: isUp ? 'rgba(16, 185, 129, 0.55)' : 'rgba(239, 68, 68, 0.55)',
      });
    }

    _ensureVolumeSeries() {
      if (this.volumeSeries || !this.chart) return;
      this.volumeSeries = this.chart.addHistogramSeries({
        priceScaleId: this.volumeScaleId,
        priceFormat: { type: 'volume' },
        color: 'rgba(148, 163, 184, 0.6)',
      });
      this._wrapSeriesDiagnostics(this.volumeSeries, 'volume');
      if (this.chart && this.chart.priceScale) {
        const volumeScale = this.chart.priceScale(this.volumeScaleId);
        if (volumeScale && volumeScale.applyOptions) {
          volumeScale.applyOptions({
            scaleMargins: { top: 0.78, bottom: 0 },
            borderVisible: false,
            visible: false,
          });
        }
        this._applyPriceScaleMargins();
      }
    }

    _sanitizeBars(bars) {
      if (!Array.isArray(bars)) return [];
      const cleaned = [];
      bars.forEach((bar) => {
        if (!bar) return;
        const time = normalizeEpochSeconds(bar.time);
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
      cleaned.sort((a, b) => a.time - b.time);
      if (cleaned.length <= 1) {
        return cleaned;
      }
      const deduped = [];
      for (let i = 0; i < cleaned.length; i += 1) {
        const item = cleaned[i];
        const prev = deduped[deduped.length - 1];
        if (prev && Number.isFinite(item.time) && item.time === prev.time) {
          deduped[deduped.length - 1] = item;
          continue;
        }
        deduped.push(item);
      }
      return deduped;
    }

    _initTooltip() {
      if (!this.container || !this.chart) return;
      this.tooltipEl = document.createElement('div');
      this.tooltipEl.className = 'market-chart-tooltip';
      this.tooltipEl.hidden = true;
      this.container.appendChild(this.tooltipEl);
      this.chart.subscribeCrosshairMove((param) => this._handleCrosshairMove(param));
    }

    _findBarByTime(time) {
      if (!Array.isArray(this.ohlcData) || !this.ohlcData.length) return null;
      for (let i = this.ohlcData.length - 1; i >= 0; i -= 1) {
        const bar = this.ohlcData[i];
        if (bar && bar.time === time) return bar;
      }
      return null;
    }

    _handleCrosshairMove(param) {
      if (!this.tooltipEl) return;
      if (!param || !param.time || !param.point) {
        this.tooltipEl.hidden = true;
        return;
      }
      const seriesData = param.seriesData;
      let ohlc = seriesData.get(this.candleSeries);
      if (!ohlc || typeof ohlc.open !== 'number') {
        const linePoint = seriesData.get(this.lineSeries);
        if (linePoint && typeof linePoint.value === 'number') {
          ohlc = { open: linePoint.value, high: linePoint.value, low: linePoint.value, close: linePoint.value };
        }
      }
      if (!ohlc || typeof ohlc.open !== 'number') {
        this.tooltipEl.hidden = true;
        return;
      }
      const tickIndex = this.axisMode === 'tick' ? this._resolveTickIndex(param.time) : null;
      const volumeBar = this._findBarByTime(param.time);
      const volume = volumeBar && Number.isFinite(volumeBar.volume) ? volumeBar.volume : null;
      const timeLabel = this._formatAxisTime(param.time, {
        context: 'tooltip',
        includeDate: true,
        fullDate: true,
      });
      const parts = [
        `${TEXT.chartTooltipTime || 'Time'}: ${timeLabel}`,
        `${TEXT.chartTooltipOpen || 'O'}: ${formatPrice4(ohlc.open)}`,
        `${TEXT.chartTooltipHigh || 'H'}: ${formatPrice4(ohlc.high)}`,
        `${TEXT.chartTooltipLow || 'L'}: ${formatPrice4(ohlc.low)}`,
        `${TEXT.chartTooltipClose || 'C'}: ${formatPrice4(ohlc.close)}`,
      ];
      if (tickIndex !== null) {
        parts.unshift(`${TEXT.chartTooltipTick || 'T'}: ${tickIndex}`);
      }
      if (volume !== null) {
        parts.push(`${TEXT.chartTooltipVolume || 'Vol'}: ${formatCompactNumber(volume)}`);
      }
      this.tooltipEl.textContent = parts.join(' · ');
      const margin = 8;
      const x = Math.min(
        Math.max(param.point.x + margin, margin),
        (this.container.clientWidth || 0) - margin
      );
      const y = Math.max(param.point.y - 10, margin);
      this.tooltipEl.style.transform = `translate(${x}px, ${y}px)`;
      this.tooltipEl.hidden = false;
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

    _schedulePulseFrame() {
      if (!this.pulseEnabled || !this.overlayCanvas || !this.chart) return;
      if (this._pulseFrame) return;
      this._pulseFrame = window.requestAnimationFrame((ts) => {
        this._pulseFrame = null;
        if (!this.pulseEnabled || !this.chart || !this.overlayCanvas) return;
        if (document && document.hidden) return;
        const width = this.container ? this.container.clientWidth || 0 : 0;
        const height = this.container ? this.container.clientHeight || 0 : 0;
        if (width <= 0 || height <= 0) return;
        const minFrameMs = 1000 / CHART_PULSE_MAX_FPS;
        if (ts - this._pulseLastRender < minFrameMs) {
          this._schedulePulseFrame();
          return;
        }
        this._pulseLastRender = ts;
        this.renderOverlay();
      });
    }

    _bindResize() {
      if (typeof ResizeObserver === 'undefined') return;
      if (this._resizeObserver) return;
      this._resizeObserver = new ResizeObserver(() => this.resize());
      if (this.container) this._resizeObserver.observe(this.container);
      if (this.indicatorContainer) this._resizeObserver.observe(this.indicatorContainer);
      if (this.indicatorSecondaryContainer) this._resizeObserver.observe(this.indicatorSecondaryContainer);
    }

    _bindTimeSync() {
      if (!this.chart) return;
      const timeScale = this.chart.timeScale();
      timeScale.subscribeVisibleTimeRangeChange(() => this.renderOverlay());
      const priceScale = this.chart.priceScale && this.chart.priceScale('right');
      if (priceScale && typeof priceScale.subscribeVisibleLogicalRangeChange === 'function') {
        priceScale.subscribeVisibleLogicalRangeChange(() => this.renderOverlay());
      }
      const syncTargets = [
        {
          chart: this.indicatorChart,
        },
        {
          chart: this.indicatorSecondaryChart,
        },
      ];
      const syncToTargets = (range, methodName) => {
        if (this._syncingIndicator || !range) return;
        if (!Number.isFinite(range.from) || !Number.isFinite(range.to)) return;
        this._syncingIndicator = true;
        syncTargets.forEach((target) => {
          if (!target.chart || !target.chart.timeScale) return;
          const targetScale = target.chart.timeScale();
          if (!targetScale || typeof targetScale[methodName] !== 'function') return;
          try {
            targetScale[methodName](range);
          } catch (_error) {
            // ignore single target sync failures
          }
        });
        this._syncingIndicator = false;
      };
      const syncToMain = (range, methodName) => {
        if (this._syncingIndicator || !range) return;
        if (!Number.isFinite(range.from) || !Number.isFinite(range.to)) return;
        if (!this.chart || !this.chart.timeScale) return;
        const mainScale = this.chart.timeScale();
        if (!mainScale || typeof mainScale[methodName] !== 'function') return;
        this._syncingIndicator = true;
        try {
          mainScale[methodName](range);
        } catch (_error) {
          // ignore reverse sync errors
        }
        this._syncingIndicator = false;
      };
      const bindReverseSync = (targetChart) => {
        if (!targetChart || !targetChart.timeScale) return;
        const targetScale = targetChart.timeScale();
        if (!targetScale) return;
        if (typeof targetScale.subscribeVisibleTimeRangeChange === 'function') {
          targetScale.subscribeVisibleTimeRangeChange((range) => syncToMain(range, 'setVisibleRange'));
        }
        if (typeof targetScale.subscribeVisibleLogicalRangeChange === 'function') {
          targetScale.subscribeVisibleLogicalRangeChange((range) => syncToMain(range, 'setVisibleLogicalRange'));
        }
      };
      timeScale.subscribeVisibleTimeRangeChange((range) => syncToTargets(range, 'setVisibleRange'));
      if (typeof timeScale.subscribeVisibleLogicalRangeChange === 'function') {
        timeScale.subscribeVisibleLogicalRangeChange((range) => syncToTargets(range, 'setVisibleLogicalRange'));
      }
      syncTargets.forEach((target) => bindReverseSync(target.chart));
    }

    _bindPriceScaleWheel() {
      if (!this.container || !this.chart || this._priceScaleWheelBound) return;
      this._priceScaleWheelBound = true;
      this.container.addEventListener(
        'wheel',
        (event) => this._handlePriceScaleWheel(event),
        { passive: false, capture: true }
      );
    }

    _handlePriceScaleWheel(event) {
      if (!event || !this.chart || !this.container) return;
      if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
      if (Math.abs(event.deltaY) < Math.abs(event.deltaX || 0)) return;
      if (!this._isOverPriceScale(event)) return;
      event.preventDefault();
      if (typeof event.stopImmediatePropagation === 'function') {
        event.stopImmediatePropagation();
      }
      if (typeof event.stopPropagation === 'function') {
        event.stopPropagation();
      }
      const rect = this.container.getBoundingClientRect();
      const height = rect.height || this.container.clientHeight || 0;
      if (!height) return;
      const y = Math.max(0, Math.min(height, event.clientY - rect.top));
      const series = this._getActivePriceSeries();
      if (!series || typeof series.coordinateToPrice !== 'function') return;
      const anchorPriceRaw = series.coordinateToPrice(y);
      const dataRange = this._computeVisiblePriceRange();
      if (!dataRange || !Number.isFinite(dataRange.min) || !Number.isFinite(dataRange.max)) return;
      let baseRange = this.manualPriceRange || dataRange;
      if (
        baseRange &&
        (baseRange.max < dataRange.min || baseRange.min > dataRange.max)
      ) {
        baseRange = dataRange;
      }
      if (!baseRange || !Number.isFinite(baseRange.min) || !Number.isFinite(baseRange.max)) return;
      let anchorPrice = Number.isFinite(anchorPriceRaw)
        ? anchorPriceRaw
        : (baseRange.min + baseRange.max) / 2;
      anchorPrice = Math.min(dataRange.max, Math.max(dataRange.min, anchorPrice));
      const zoomFactor = Math.exp(event.deltaY * PRICE_SCALE_WHEEL_SPEED);
      let nextMin = anchorPrice - (anchorPrice - baseRange.min) * zoomFactor;
      let nextMax = anchorPrice + (baseRange.max - anchorPrice) * zoomFactor;
      if (!Number.isFinite(nextMin) || !Number.isFinite(nextMax)) return;
      if (nextMin === nextMax) {
        const delta = Math.max(1e-9, Math.abs(anchorPrice) * 1e-6);
        nextMin = anchorPrice - delta;
        nextMax = anchorPrice + delta;
      }
      const min = Math.min(nextMin, nextMax);
      const max = Math.max(nextMin, nextMax);
      this._setManualPriceRange({ min, max });
      this.renderOverlay();
    }

    _bindPanDrag() {
      if (!this.container || !this.chart || this._panBound) return;
      this._panBound = true;
      this.container.addEventListener('pointerdown', (event) => this._handlePanStart(event));
      this.container.addEventListener(
        'wheel',
        () => {
          detailLastRangeInteractionAt = Date.now();
          if (typeof this._onRangeInteraction === 'function' && this.chart && this.chart.timeScale) {
            const timeScale = this.chart.timeScale();
            const range =
              timeScale && typeof timeScale.getVisibleLogicalRange === 'function'
                ? timeScale.getVisibleLogicalRange()
                : null;
            this._onRangeInteraction(range);
          }
        },
        { passive: true }
      );
      this.container.addEventListener(
        'touchstart',
        () => {
          detailLastRangeInteractionAt = Date.now();
        },
        { passive: true }
      );
      this.container.addEventListener(
        'touchmove',
        () => {
          detailLastRangeInteractionAt = Date.now();
          if (typeof this._onRangeInteraction === 'function' && this.chart && this.chart.timeScale) {
            const timeScale = this.chart.timeScale();
            const range =
              timeScale && typeof timeScale.getVisibleLogicalRange === 'function'
                ? timeScale.getVisibleLogicalRange()
                : null;
            this._onRangeInteraction(range);
          }
        },
        { passive: true }
      );
      window.addEventListener('pointermove', (event) => this._handlePanMove(event));
      window.addEventListener('pointerup', (event) => this._handlePanEnd(event));
      window.addEventListener('pointercancel', (event) => this._handlePanEnd(event));
    }

    _handlePanStart(event) {
      if (!event || !this.chart || !this.container) return;
      if (event.button !== 0) return;
      if (this.drawMode && this.drawMode !== 'none') return;
      if (this._isOverPriceScale(event)) return;
      detailLastRangeInteractionAt = Date.now();
      const timeScale = this.chart.timeScale();
      const logicalRange =
        timeScale && typeof timeScale.getVisibleLogicalRange === 'function'
          ? timeScale.getVisibleLogicalRange()
          : null;
      if (!logicalRange || !Number.isFinite(logicalRange.from) || !Number.isFinite(logicalRange.to)) return;
      const priceRange = this.manualPriceRange || this._computeVisiblePriceRange();
      if (!priceRange || !Number.isFinite(priceRange.min) || !Number.isFinite(priceRange.max)) return;
      this._panActive = true;
      this._panPointerId = event.pointerId;
      this._panStart = { x: event.clientX, y: event.clientY };
      this._panStartLogicalRange = { from: logicalRange.from, to: logicalRange.to };
      this._panStartPriceRange = { min: priceRange.min, max: priceRange.max };
      if (typeof this.container.setPointerCapture === 'function') {
        this.container.setPointerCapture(event.pointerId);
      }
      event.preventDefault();
      if (typeof event.stopPropagation === 'function') {
        event.stopPropagation();
      }
    }

    _handlePanMove(event) {
      if (!this._panActive || !event) return;
      if (this._panPointerId !== null && event.pointerId !== this._panPointerId) return;
      if (!this._panStart || !this._panStartLogicalRange || !this._panStartPriceRange) return;
      if (!this.chart || !this.container) return;
      detailLastRangeInteractionAt = Date.now();
      const width = this.container.clientWidth || 1;
      const height = this.container.clientHeight || 1;
      const dx = event.clientX - this._panStart.x;
      const dy = event.clientY - this._panStart.y;
      const timeSpan = this._panStartLogicalRange.to - this._panStartLogicalRange.from;
      if (Number.isFinite(timeSpan) && timeSpan > 0) {
        const shift = (dx / width) * timeSpan;
        const nextRange = {
          from: this._panStartLogicalRange.from - shift,
          to: this._panStartLogicalRange.to - shift,
        };
        const timeScale = this.chart.timeScale();
        if (timeScale && typeof timeScale.setVisibleLogicalRange === 'function') {
          withAutoRangeLock(() => {
            timeScale.setVisibleLogicalRange(nextRange);
          });
        }
        if (typeof this._onRangeInteraction === 'function') {
          this._onRangeInteraction(nextRange);
        }
      }
      const priceSpan = this._panStartPriceRange.max - this._panStartPriceRange.min;
      if (Number.isFinite(priceSpan) && priceSpan > 0) {
        const priceShift = (dy / height) * priceSpan;
        const nextMin = this._panStartPriceRange.min + priceShift;
        const nextMax = this._panStartPriceRange.max + priceShift;
        this._setManualPriceRange({ min: nextMin, max: nextMax });
      }
      detailUserPanned = true;
      this.renderOverlay();
      event.preventDefault();
      if (typeof event.stopPropagation === 'function') {
        event.stopPropagation();
      }
    }

    _handlePanEnd(event) {
      if (!this._panActive) return;
      if (event && this._panPointerId !== null && event.pointerId !== this._panPointerId) return;
      if (this.container && typeof this.container.releasePointerCapture === 'function' && this._panPointerId !== null) {
        try {
          this.container.releasePointerCapture(this._panPointerId);
        } catch (error) {
          // noop
        }
      }
      this._panActive = false;
      this._panPointerId = null;
      this._panStart = null;
      this._panStartLogicalRange = null;
      this._panStartPriceRange = null;
    }

    _isOverPriceScale(event) {
      if (!this.chart || !this.container || !event) return false;
      const rect = this.container.getBoundingClientRect();
      if (!rect || rect.width <= 0) return false;
      const axisWidth =
        this.chart.priceScale && this.chart.priceScale('right') && this.chart.priceScale('right').width
          ? this.chart.priceScale('right').width()
          : 0;
      const effectiveWidth = axisWidth && axisWidth > 0 ? axisWidth : PRICE_SCALE_HIT_WIDTH;
      const x = event.clientX - rect.left;
      return x >= rect.width - effectiveWidth;
    }

    _applyPriceScaleMargins() {
      if (!this.chart || !this.chart.priceScale) return;
      const priceScale = this.chart.priceScale('right');
      if (!priceScale || typeof priceScale.applyOptions !== 'function') return;
      priceScale.applyOptions({
        scaleMargins: {
          top: PRICE_SCALE_TOP_BASE,
          bottom: PRICE_SCALE_BOTTOM_BASE,
        },
        minimumWidth: PRICE_SCALE_MIN_WIDTH,
      });
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

    _updateSessionBands() {
      if (this.axisMode === 'tick' || !Array.isArray(this.ohlcData) || !this.ohlcData.length) {
        this.sessionBands = [];
        return;
      }
      if (this.intervalSpec && this.intervalSpec.unit === 'day') {
        this.sessionBands = [];
        return;
      }
      const bands = [];
      let current = null;
      this.ohlcData.forEach((bar, idx) => {
        const time = normalizeEpochSeconds(bar && bar.time);
        if (!Number.isFinite(time)) return;
        const dateKey = getEtDateKey(time);
        if (!dateKey) return;
        const minutes = getEtMinutes(time);
        const session = classifyEtSession(minutes);
        if (session === 'regular') {
          if (current) {
            bands.push(current);
            current = null;
          }
          return;
        }
        if (!current || current.type !== session || current.dateKey !== dateKey) {
          if (current) {
            bands.push(current);
          }
          current = {
            type: session,
            dateKey,
            start: time,
            end: time,
            startIndex: idx,
            endIndex: idx,
          };
          return;
        }
        current.end = Math.max(current.end, time);
        current.endIndex = idx;
      });
      if (current) {
        bands.push(current);
      }
      this.sessionBands = bands;
    }

    _renderSessionBands(ctx) {
      if (!this.chart || !this.container || !this.sessionBands || !this.sessionBands.length) return;
      const timeScale = this.chart.timeScale();
      const visibleRange =
        timeScale && typeof timeScale.getVisibleRange === 'function' ? timeScale.getVisibleRange() : null;
      const rangeFrom = visibleRange ? normalizeEpochSeconds(visibleRange.from) : null;
      const rangeTo = visibleRange ? normalizeEpochSeconds(visibleRange.to) : null;
      const width = this.container.clientWidth || 0;
      const height = this.container.clientHeight || 0;
      if (height <= 0) return;
      this.sessionBands.forEach((band) => {
        if (Number.isFinite(rangeFrom) && Number.isFinite(band.end) && band.end < rangeFrom) return;
        if (Number.isFinite(rangeTo) && Number.isFinite(band.start) && band.start > rangeTo) return;
        let startTime = Number.isFinite(band.start) ? band.start : null;
        let endTime = Number.isFinite(band.end) ? band.end : null;
        if (
          Number.isFinite(band.endIndex) &&
          Array.isArray(this.ohlcData) &&
          band.endIndex + 1 < this.ohlcData.length
        ) {
          const nextTime = normalizeEpochSeconds(this.ohlcData[band.endIndex + 1] && this.ohlcData[band.endIndex + 1].time);
          if (Number.isFinite(nextTime)) {
            endTime = nextTime;
          }
        }
        let x1 = null;
        let x2 = null;
        if (startTime !== null && endTime !== null) {
          x1 = timeScale.timeToCoordinate(startTime);
          x2 = timeScale.timeToCoordinate(endTime);
        }
        if (
          (x1 === null || x2 === null) &&
          Number.isFinite(band.startIndex) &&
          Number.isFinite(band.endIndex) &&
          typeof timeScale.logicalToCoordinate === 'function'
        ) {
          x1 = timeScale.logicalToCoordinate(band.startIndex);
          x2 = timeScale.logicalToCoordinate(band.endIndex + 1);
        }
        if (x1 === null || x2 === null) return;
        let left = Math.min(x1, x2);
        let right = Math.max(x1, x2);
        const bandWidth = right - left;
        if (!Number.isFinite(bandWidth) || bandWidth <= 0) return;
        if (right < 0 || left > width) return;
        left = Math.max(0, left);
        right = Math.min(width, right);
        const renderWidth = right - left;
        if (renderWidth <= 0) return;
        ctx.fillStyle = SESSION_COLORS[band.type] || 'rgba(148, 163, 184, 0.05)';
        ctx.fillRect(left, 0, renderWidth, height);
      });
    }

    _renderLatestPulse(ctx) {
      if (!this.chart || !this.container || !this.pulseEnabled) return;
      const anchor = this._resolvePulseAnchor();
      if (!anchor || !Number.isFinite(anchor.time) || !Number.isFinite(anchor.close)) return;
      const series = this.priceSeriesMode === 'line' && this.lineSeries ? this.lineSeries : this.candleSeries;
      if (!series) return;
      const markerPinned = this._updatePulseMarker(anchor);
      const timeScale = this.chart.timeScale();
      const x = timeScale.timeToCoordinate(anchor.time);
      const y = series.priceToCoordinate(anchor.close);
      if (x === null || y === null) return;
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const base = Number.isFinite(this._pulseBase) ? this._pulseBase : now;
      const elapsed = (now - base) % CHART_PULSE_PERIOD_MS;
      const phase = Math.min(1, Math.max(0, elapsed / CHART_PULSE_PERIOD_MS));
      const radius =
        CHART_PULSE_MIN_RADIUS + (CHART_PULSE_MAX_RADIUS - CHART_PULSE_MIN_RADIUS) * phase;
      const alpha = CHART_PULSE_ALPHA * (1 - phase);
      ctx.save();
      ctx.globalAlpha = Math.max(0, alpha);
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      if (!markerPinned) {
        ctx.save();
        ctx.fillStyle = '#3b82f6';
        ctx.beginPath();
        ctx.arc(x, y, CHART_PULSE_DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    _updatePulseMarker(anchor) {
      if (!CHART_PULSE_MARKER_ENABLED) {
        if (this._pulseMarkerSeries && typeof this._pulseMarkerSeries.setMarkers === 'function') {
          try {
            this._pulseMarkerSeries.setMarkers([]);
          } catch (error) {
            // noop
          }
        }
        this._pulseMarkerSeries = null;
        this._pulseMarkerTime = null;
        return false;
      }
      const series = this.priceSeriesMode === 'line' && this.lineSeries ? this.lineSeries : this.candleSeries;
      if (!series || typeof series.setMarkers !== 'function') return false;
      if (this._pulseMarkerSeries && this._pulseMarkerSeries !== series) {
        try {
          this._pulseMarkerSeries.setMarkers([]);
        } catch (error) {
          // noop
        }
        this._pulseMarkerSeries = null;
        this._pulseMarkerTime = null;
      }
      this._pulseMarkerSeries = series;
      if (!anchor || !Number.isFinite(anchor.time)) {
        if (this._pulseMarkerTime !== null) {
          series.setMarkers([]);
          this._pulseMarkerTime = null;
        }
        return true;
      }
      if (this._pulseMarkerTime === anchor.time) return true;
      this._pulseMarkerTime = anchor.time;
      series.setMarkers([
        {
          time: anchor.time,
          position: 'inBar',
          color: '#3b82f6',
          shape: 'circle',
          size: CHART_PULSE_MARKER_SIZE,
        },
      ]);
      return true;
    }

    _resolvePulseAnchor() {
      if (!Array.isArray(this.ohlcData) || !this.ohlcData.length) return null;
      const last = this.ohlcData[this.ohlcData.length - 1];
      if (!detailUserPanned) return last;
      const timeScale = this.chart && this.chart.timeScale ? this.chart.timeScale() : null;
      const visible = timeScale && typeof timeScale.getVisibleRange === 'function' ? timeScale.getVisibleRange() : null;
      if (!visible || !Number.isFinite(visible.from) || !Number.isFinite(visible.to)) return last;
      const from = visible.from;
      const to = visible.to;
      for (let i = this.ohlcData.length - 1; i >= 0; i -= 1) {
        const bar = this.ohlcData[i];
        if (!bar || !Number.isFinite(bar.time)) continue;
        if (bar.time < from) break;
        if (bar.time <= to) return bar;
      }
      return last;
    }

    _renderAnalysisOverlay(ctx) {
      if (!this.analysisOverlayVisible) return;
      if (!this.analysisOverlay || !Array.isArray(this.analysisOverlay.points) || !this.analysisOverlay.points.length) return;
      if (!this.chart || !this.candleSeries) return;
      const timeScale = this.chart.timeScale();
      if (!timeScale) return;
      const direction = (this.analysisOverlay.direction || 'neutral').toString().toLowerCase();
      const lineColor =
        direction === 'up'
          ? 'rgba(22, 163, 74, 0.92)'
          : direction === 'down'
          ? 'rgba(220, 38, 38, 0.92)'
          : 'rgba(30, 64, 175, 0.9)';
      const points = [];
      this.analysisOverlay.points.forEach((point) => {
        const x = timeScale.timeToCoordinate(point.time);
        const y = this.candleSeries.priceToCoordinate(point.price);
        if (x === null || y === null) return;
        points.push({ x, y, kind: point.kind || '', label: point.label || '' });
      });
      if (points.length < 2) return;

      ctx.save();
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 2.1;
      ctx.shadowColor = lineColor;
      ctx.shadowBlur = 6;
      if (direction === 'neutral') {
        ctx.setLineDash([4, 4]);
      }
      ctx.beginPath();
      points.forEach((point, idx) => {
        if (idx === 0) {
          ctx.moveTo(point.x, point.y);
        } else {
          ctx.lineTo(point.x, point.y);
        }
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;

      points.forEach((point) => {
        const isHigh = point.kind === 'H';
        const fillColor = isHigh ? '#dc2626' : point.kind === 'L' ? '#16a34a' : '#2563eb';
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4.1, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (point.label) {
          ctx.font = '11px "Space Grotesk", "Noto Sans SC", sans-serif';
          const label = String(point.label);
          const textWidth = ctx.measureText(label).width;
          const paddingX = 4;
          const paddingY = 2;
          const textX = point.x + 7;
          const textY = point.y - 8;
          ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
          ctx.fillRect(textX - paddingX, textY - 10 - paddingY, textWidth + paddingX * 2, 12 + paddingY * 2);
          ctx.fillStyle = '#f8fafc';
          ctx.fillText(label, textX, textY);
        }
      });
      ctx.restore();
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
      this._renderSessionBands(ctx);
      this._renderLatestPulse(ctx);
      this._renderAnalysisOverlay(ctx);
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
      this._schedulePulseFrame();
    }
  }

  if (typeaheadHint) {
    typeaheadHint.textContent = TEXT.typeaheadHint;
  }

  function setTradeSymbol(symbol) {
    if (tradeSymbolEl) {
      tradeSymbolEl.textContent = symbol || '--';
    }
    renderRecentOrder(null);
    updateTradeEstimates();
  }

  function showTradeToast(message, tone = 'success') {
    if (!tradeToastEl || !message) return;
    tradeToastEl.textContent = message;
    tradeToastEl.classList.toggle('is-error', tone === 'error');
    tradeToastEl.classList.add('is-visible');
    if (tradeToastTimer) {
      window.clearTimeout(tradeToastTimer);
    }
    tradeToastTimer = window.setTimeout(() => {
      tradeToastEl.classList.remove('is-visible');
    }, 2400);
  }

  function setTradeStatus(message, tone = 'neutral', options = {}) {
    if (!tradeStatusEl) return;
    tradeStatusEl.textContent = message || '';
    tradeStatusEl.classList.remove('is-success', 'is-error');
    if (tone === 'success') {
      tradeStatusEl.classList.add('is-success');
    } else if (tone === 'error') {
      tradeStatusEl.classList.add('is-error');
    }
    if (tradeRetryBtn) {
      const retryable = tone === 'error' && options.retryable;
      tradeRetryBtn.classList.toggle('is-visible', retryable);
    }
  }

  function updateTradeUnitUI() {
    if (tradeUnitButtons.length) {
      tradeUnitButtons.forEach((btn) => {
        const unit = btn.dataset.unit;
        btn.classList.toggle('is-active', unit === tradeUnit);
      });
    }
    if (tradeQtyInput) {
      tradeQtyInput.classList.toggle('is-hidden', tradeUnit !== 'shares');
    }
    if (tradeNotionalInput) {
      tradeNotionalInput.classList.toggle('is-hidden', tradeUnit !== 'notional');
    }
    if (tradeInputAddon) {
      tradeInputAddon.textContent =
        tradeUnit === 'shares'
          ? TEXT.tradeUnitAddonShares || 'Shares'
          : TEXT.tradeUnitAddonNotional || 'USD';
    }
    if (tradeInputHint) {
      tradeInputHint.textContent =
        tradeUnit === 'shares' ? TEXT.tradeMinQty(TRADE_MIN_QTY) : TEXT.tradeMinNotional(TRADE_MIN_NOTIONAL);
    }
    updateTradeEstimates();
  }

  function setTradeUnit(unit) {
    const normalized = (unit || '').toLowerCase();
    if (!normalized || normalized === tradeUnit) return;
    tradeUnit = normalized === 'notional' ? 'notional' : 'shares';
    updateTradeUnitUI();
  }

  function updateTradeOrderMoreUI() {
    if (tradeOrderMorePanel) {
      tradeOrderMorePanel.classList.toggle('is-open', tradeOrderMoreOpen);
    }
    if (tradeOrderMoreToggle) {
      tradeOrderMoreToggle.classList.toggle('is-active', tradeOrderMoreOpen);
    }
  }

  function updateTradeEstimates() {
    if (!tradeEstimateNotional) return;
    const price = getDetailPriceValue();
    let notional = null;
    if (tradeUnit === 'shares') {
      const qtyVal = parseNumberFromInput(tradeQtyInput);
      if (Number.isFinite(price) && Number.isFinite(qtyVal) && qtyVal > 0) {
        notional = price * qtyVal;
      }
    } else {
      const notionalVal = parseNumberFromInput(tradeNotionalInput);
      if (Number.isFinite(notionalVal) && notionalVal > 0) {
        notional = notionalVal;
      }
    }
    tradeEstimateNotional.textContent = Number.isFinite(notional) ? formatCurrency(notional) : '—';
    if (tradeBuyingPowerEl) {
      tradeBuyingPowerEl.textContent = Number.isFinite(tradeBuyingPower) ? formatCurrency(tradeBuyingPower) : TEXT.tradeBuyingPowerUnknown;
    }
    if (tradeRemainingCashEl) {
      if (Number.isFinite(tradeBuyingPower) && Number.isFinite(notional)) {
        tradeRemainingCashEl.textContent = formatCurrency(tradeBuyingPower - notional);
      } else {
        tradeRemainingCashEl.textContent = '—';
      }
    }
  }

  function setDetailPriceValue(price) {
    if (detailPriceEl) {
      detailPriceEl.textContent = Number.isFinite(price) ? formatPrice4(price) : '--';
    }
    updateTradeEstimates();
  }

  function updateAccountExecution() {
    if (!accountExecutionEl) return;
    if (!tradeExecutionEnabled) {
      accountExecutionEl.textContent = TEXT.accountExecutionOff;
      return;
    }
    accountExecutionEl.textContent = TEXT.accountExecutionOn;
  }

  function setAccountStatus(message) {
    if (!accountStatusEl) return;
    accountStatusEl.textContent = message || '';
  }

  function applyAccountSnapshot(payload) {
    if (!payload || typeof payload !== 'object') return;
    const account = payload.account || {};
    const mode = payload.mode || tradeMode;
    if (accountModeEl) {
      accountModeEl.textContent = mode === 'live' ? TEXT.tradeModeLive : TEXT.tradeModePaper;
      accountModeEl.classList.toggle('is-live', mode === 'live');
    }
    if (accountEquityEl) {
      accountEquityEl.textContent = Number.isFinite(account.equity) ? formatCurrency(account.equity) : '—';
    }
    if (accountCashEl) {
      accountCashEl.textContent = Number.isFinite(account.cash) ? formatCurrency(account.cash) : '—';
    }
    if (accountBuyingPowerEl) {
      accountBuyingPowerEl.textContent = Number.isFinite(account.buying_power) ? formatCurrency(account.buying_power) : '—';
    }
    if (accountPortfolioEl) {
      accountPortfolioEl.textContent = Number.isFinite(account.portfolio_value)
        ? formatCurrency(account.portfolio_value)
        : '—';
    }
    if (Number.isFinite(account.buying_power)) {
      tradeBuyingPower = account.buying_power;
      updateTradeEstimates();
    }
    if (accountUpdatedEl) {
      const updated = payload.updated_at ? formatDisplayTime(payload.updated_at) : '';
      accountUpdatedEl.textContent = updated ? `${TEXT.accountUpdatedPrefix} ${updated}` : '';
    }
    updateAccountExecution();
    if (account.status) {
      const statusLabel = langPrefix === 'zh' ? '状态' : 'Status';
      setAccountStatus(`${statusLabel}: ${account.status}`);
    } else {
      setAccountStatus('');
    }
  }

  function resetAccountSnapshot(message) {
    if (accountEquityEl) accountEquityEl.textContent = '—';
    if (accountCashEl) accountCashEl.textContent = '—';
    if (accountBuyingPowerEl) accountBuyingPowerEl.textContent = '—';
    if (accountPortfolioEl) accountPortfolioEl.textContent = '—';
    if (accountUpdatedEl) accountUpdatedEl.textContent = '';
    tradeBuyingPower = null;
    updateTradeEstimates();
    setAccountStatus(message || '');
  }

  async function refreshAccountSnapshot(options = {}) {
    if (!accountEndpoint || !accountCard) return;
    const mode = tradeMode === 'live' ? 'live' : 'paper';
    setAccountStatus(TEXT.accountLoading);
    if (accountRefreshBtn) {
      accountRefreshBtn.disabled = true;
    }
    try {
      const params = new URLSearchParams({ mode });
      const response = await fetch(`${accountEndpoint}?${params.toString()}`, {
        headers: { Accept: 'application/json' },
        credentials: 'same-origin',
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        const err = payload && payload.error ? payload.error : '';
        if (err === 'missing_credentials') {
          resetAccountSnapshot(TEXT.accountMissing);
        } else {
          resetAccountSnapshot(TEXT.accountError);
        }
        return;
      }
      applyAccountSnapshot(payload);
    } catch (err) {
      resetAccountSnapshot(TEXT.accountError);
    } finally {
      if (accountRefreshBtn) {
        accountRefreshBtn.disabled = false;
      }
    }
  }

  function renderRecentOrder(payload) {
    if (!tradeRecentBody) return;
    if (!payload) {
      tradeRecentBody.textContent = TEXT.tradeRecentEmpty;
      return;
    }
    const sideLabel =
      payload.side === 'sell'
        ? langPrefix === 'zh'
          ? '卖出'
          : 'Sell'
        : langPrefix === 'zh'
          ? '买入'
          : 'Buy';
    const qtyText = payload.qty ? `${payload.qty}` : '';
    const notionalText = payload.notional ? formatCurrency(payload.notional) : '';
    const sizeText = qtyText || notionalText || '--';
    const status = payload.status ? `${payload.status}` : '';
    const orderId = payload.order_id ? `#${payload.order_id}` : '';
    const orderLine = [sideLabel, payload.symbol, sizeText, status, orderId].filter(Boolean).join(' · ');
    const positionLine =
      langPrefix === 'zh' ? '持仓变化：—' : 'Position change: —';
    tradeRecentBody.innerHTML = `<div>${orderLine}</div><div class="detail-trade-recent-sub">${positionLine}</div>`;
  }


  function applyTradeModeState(payload) {
    if (!payload || typeof payload !== 'object') return;
    if (typeof payload.mode === 'string') {
      tradeMode = payload.mode.toLowerCase();
    }
    if (typeof payload.trading_enabled === 'boolean') {
      tradeEnabled = payload.trading_enabled;
    }
    if (typeof payload.execution_enabled === 'boolean') {
      tradeExecutionEnabled = payload.execution_enabled;
    }
    tradeModeError = '';
    updateTradeModeUI();
    updateAccountExecution();
  }

  function updateTradeModeUI() {
    if (tradeModeError && tradeModeStatus) {
      tradeModeStatus.textContent = tradeModeError;
      tradeModeStatus.classList.remove('is-live');
      if (tradeModePill) {
        tradeModePill.textContent = tradeModeError;
        tradeModePill.classList.remove('is-live');
      }
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
    const isLive = tradeMode === 'live';
    if (!tradeExecutionEnabled) {
      if (tradeModeStatus) {
        tradeModeStatus.textContent = TEXT.tradeModeExecutionOff;
        tradeModeStatus.classList.remove('is-live');
      }
      if (tradeModePill) {
        tradeModePill.textContent = TEXT.tradeModeExecutionOff;
        tradeModePill.classList.remove('is-live');
      }
      return;
    }
    if (tradeModeStatus) {
      tradeModeStatus.textContent = isLive ? TEXT.tradeModeLive : TEXT.tradeModePaper;
      tradeModeStatus.classList.toggle('is-live', isLive);
    }
    if (tradeModePill) {
      tradeModePill.textContent = isLive ? TEXT.tradeModeLive : TEXT.tradeModePaper;
      tradeModePill.classList.toggle('is-live', isLive);
    }
    if (accountModeEl) {
      accountModeEl.textContent = isLive ? TEXT.tradeModeLive : TEXT.tradeModePaper;
      accountModeEl.classList.toggle('is-live', isLive);
    }
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
        refreshAccountSnapshot();
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
      refreshAccountSnapshot();
    } catch (err) {
      tradeModeError = err && err.message ? err.message : TEXT.tradeModeFailed;
    } finally {
      tradeModeBusy = false;
      updateTradeModeUI();
    }
  }

  function resolveTradeInput(symbol) {
    const normalizedSymbol = (symbol || '').trim();
    if (!normalizedSymbol || normalizedSymbol === '--') {
      return { error: TEXT.tradeMissingSymbol };
    }
    let qtyVal = null;
    let notionalVal = null;
    if (tradeUnit === 'shares') {
      qtyVal = parseNumberFromInput(tradeQtyInput);
      if (!Number.isFinite(qtyVal) || qtyVal <= 0) {
        return { error: TEXT.tradeMissingQty };
      }
      if (qtyVal < TRADE_MIN_QTY) {
        return { error: TEXT.tradeMinQty(TRADE_MIN_QTY) };
      }
    } else {
      notionalVal = parseNumberFromInput(tradeNotionalInput);
      if (!Number.isFinite(notionalVal) || notionalVal <= 0) {
        return { error: TEXT.tradeMissingNotional };
      }
      if (notionalVal < TRADE_MIN_NOTIONAL) {
        return { error: TEXT.tradeMinNotional(TRADE_MIN_NOTIONAL) };
      }
    }
    const price = getDetailPriceValue();
    const notionalCheck =
      tradeUnit === 'shares' && Number.isFinite(price) && Number.isFinite(qtyVal)
        ? price * qtyVal
        : Number.isFinite(notionalVal)
          ? notionalVal
          : null;
    if (Number.isFinite(tradeBuyingPower) && Number.isFinite(notionalCheck) && notionalCheck > tradeBuyingPower) {
      return { error: TEXT.tradeInsufficientBuyingPower };
    }
    return { symbol: normalizedSymbol, qty: qtyVal, notional: notionalVal };
  }

  async function submitTrade(side) {
    const symbol = (detailSymbol || (detailSymbolEl && detailSymbolEl.textContent) || '').trim();
    const resolved = resolveTradeInput(symbol);
    if (resolved.error) {
      setTradeStatus(resolved.error, 'error');
      return;
    }
    const confirmText = tradeMode === 'live' ? TEXT.tradeSellConfirmLive : TEXT.tradeSellConfirm;
    if (side === 'sell' && confirmText && !window.confirm(confirmText)) {
      return;
    }
    setTradeStatus(TEXT.tradeSubmitting);
    tradeLastAttempt = { side, symbol: resolved.symbol, qty: resolved.qty, notional: resolved.notional };
    try {
      const payloadBody = { symbol: resolved.symbol, side };
      if (tradeUnit === 'shares' && Number.isFinite(resolved.qty)) {
        payloadBody.qty = resolved.qty;
      } else if (tradeUnit === 'notional' && Number.isFinite(resolved.notional)) {
        payloadBody.notional = resolved.notional;
      }
      const response = await fetch(orderEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfToken(),
        },
        body: JSON.stringify(payloadBody),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        setTradeStatus(buildOrderErrorMessage(payload && payload.message ? payload.message : ''), 'error', { retryable: true });
        return;
      }
      setTradeStatus(payload.message || TEXT.tradeSuccess, 'success');
      showTradeToast(payload.message || TEXT.tradeSuccess, 'success');
      renderRecentOrder({
        symbol: resolved.symbol,
        side,
        qty: resolved.qty,
        notional: resolved.notional,
        status: payload.status,
        order_id: payload.order_id,
      });
    } catch (err) {
      setTradeStatus(TEXT.tradeFailed, 'error', { retryable: true });
    }
  }

  if (tradeBuyBtn) {
    tradeBuyBtn.addEventListener('click', () => submitTrade('buy'));
  }
  if (tradeSellBtn) {
    tradeSellBtn.addEventListener('click', () => submitTrade('sell'));
  }
  if (tradeRetryBtn) {
    tradeRetryBtn.addEventListener('click', () => {
      if (!tradeLastAttempt) return;
      if (Number.isFinite(tradeLastAttempt.notional)) {
        setTradeUnit('notional');
        if (tradeNotionalInput) {
          tradeNotionalInput.value = tradeLastAttempt.notional;
        }
      } else {
        setTradeUnit('shares');
        if (tradeQtyInput) {
          tradeQtyInput.value = tradeLastAttempt.qty || '';
        }
      }
      updateTradeEstimates();
      submitTrade(tradeLastAttempt.side);
    });
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
  if (tradeUnitButtons.length) {
    tradeUnitButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const unit = btn.dataset.unit;
        if (!unit) return;
        setTradeUnit(unit);
      });
    });
  }
  if (tradeQtyInput) {
    tradeQtyInput.addEventListener('input', updateTradeEstimates);
  }
  if (tradeNotionalInput) {
    tradeNotionalInput.addEventListener('input', updateTradeEstimates);
  }
  if (tradeOrderTypeButtons.length) {
    tradeOrderTypeButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const type = btn.dataset.type || 'market';
        tradeOrderType = type;
        tradeOrderTypeButtons.forEach((item) => {
          item.classList.toggle('is-active', item.dataset.type === type);
        });
      });
    });
  }
  if (tradeOrderMoreToggle) {
    tradeOrderMoreToggle.addEventListener('click', () => {
      tradeOrderMoreOpen = !tradeOrderMoreOpen;
      updateTradeOrderMoreUI();
    });
  }
  if (accountRefreshBtn) {
    accountRefreshBtn.addEventListener('click', () => {
      refreshAccountSnapshot({ manual: true });
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
      timeframeButtons.forEach((b) => {
        const isActive = b === btn;
        b.classList.toggle('is-active', isActive);
        b.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });
      currentTimeframe = btn.getAttribute('data-timeframe') || '1mo';
      rankSort = 'default';
      rankItemsBase = [];
      rankItems = [];
      updateSortIndicator();
      updateStatusContext();
      if (currentListType === 'all') {
        const query = (searchInput && searchInput.value.trim()) || allStocksQuery || '';
        loadAllStocks({ query, page: 1, letter: allStocksLetter });
      } else {
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

  if (detailWatchToggle) {
    detailWatchToggle.addEventListener('click', () => {
      if (!detailSymbol) {
        setStatus(TEXT.statusNeedSymbol);
        return;
      }
      const isWatched = watchPool.includes(detailSymbol);
      const action = isWatched ? 'remove' : 'add';
      const isAllMode = currentListType === 'all';
      loadData(detailSymbol, {
        watchAction: action,
        listType: isAllMode ? 'gainers' : currentListType,
        keepListType: true,
        skipListRender: isAllMode,
        openDetail: !isAllMode,
      });
    });
  }

  if (statusRefresh) {
    statusRefresh.addEventListener('click', () => {
      triggerRefresh();
    });
  }

  if (statusControlsToggle) {
    statusControlsToggle.addEventListener('click', () => {
      setControlsExpanded(!controlsExpanded);
    });
  }

  if (controlsPopoverBackdrop) {
    controlsPopoverBackdrop.addEventListener('click', () => {
      setControlsExpanded(false);
    });
  }

  if (controlsPopoverClose) {
    controlsPopoverClose.addEventListener('click', () => {
      setControlsExpanded(false);
      if (statusControlsToggle && typeof statusControlsToggle.focus === 'function') {
        statusControlsToggle.focus({ preventScroll: true });
      }
    });
  }

  if (quickPanelCollapse) {
    quickPanelCollapse.addEventListener('click', () => {
      setQuickRailCollapsed(!quickRailCollapsed);
    });
  }

  document.addEventListener('click', (event) => {
    if (!quickRail || !isDesktopQuickDrawer()) return;
    if (quickRailCollapsed) return;
    const target = event.target;
    if (!(target instanceof Node)) return;
    if (quickRail.contains(target)) return;
    setQuickRailCollapsed(true);
  });

  document.addEventListener('keydown', (event) => {
    if (controlsExpanded && event.key === 'Tab') {
      trapControlsPopoverFocus(event);
      return;
    }
    if (event.key !== 'Escape') return;
    if (controlsExpanded) {
      setControlsExpanded(false);
      if (statusControlsToggle && typeof statusControlsToggle.focus === 'function') {
        statusControlsToggle.focus({ preventScroll: true });
      }
      return;
    }
    if (!quickRail || !isDesktopQuickDrawer()) return;
    if (quickRailCollapsed) return;
    setQuickRailCollapsed(true);
    if (quickPanelCollapse && typeof quickPanelCollapse.focus === 'function') {
      quickPanelCollapse.focus({ preventScroll: true });
    }
  });

  if (timezoneToggles.length) {
    timezoneToggles.forEach((toggle) => {
      toggle.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-tz]');
        if (!button) return;
        const nextMode = button.dataset.tz;
        setTimezoneMode(nextMode);
      });
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

  function clampTrackValue(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 0;
    return Math.min(Math.max(Math.round(numeric), 0), 100);
  }

  function setQuickRailCollapsed(collapsed, { persist = true } = {}) {
    quickRailCollapsed = Boolean(collapsed);
    const forceExpanded = Boolean(window.matchMedia && window.matchMedia('(max-width: 1023px)').matches);
    const effectiveCollapsed = forceExpanded ? false : quickRailCollapsed;
    if (quickRail) {
      quickRail.classList.toggle('is-quick-collapsed', effectiveCollapsed);
      quickRail.setAttribute('aria-expanded', effectiveCollapsed ? 'false' : 'true');
    }
    if (statusSection) {
      statusSection.classList.toggle('is-quick-collapsed', effectiveCollapsed);
    }
    if (quickRailContent) {
      const drawerOpen = isDesktopQuickDrawer() && !effectiveCollapsed;
      quickRailContent.classList.toggle('is-drawer-open', drawerOpen);
    }
    if (quickPanelCollapse) {
      quickPanelCollapse.setAttribute('aria-expanded', effectiveCollapsed ? 'false' : 'true');
      quickPanelCollapse.innerHTML = effectiveCollapsed ? '&#187;' : '&#171;';
      quickPanelCollapse.classList.toggle('is-collapsed', effectiveCollapsed);
    }
    if (persist) {
      savePreference(PREF_QUICK_RAIL_COLLAPSED_KEY, quickRailCollapsed ? '1' : '0');
    }
  }

  function isDesktopQuickDrawer() {
    return !(window.matchMedia && window.matchMedia('(max-width: 1023px)').matches);
  }

  function renderStatusTrack() {
    if (!statusTrackShell || !statusTrackFill || !statusTrackText) return;
    const state = currentStatusState || deriveStatusState();
    let progressValue = 0;
    let trackText = (statusText && statusText.textContent && statusText.textContent.trim()) || getStatusMessage(state);
    const hasSnapshot = snapshotState && typeof snapshotState === 'object';
    const showSnapshotProgress = hasSnapshot && snapshotState.showProgress;
    const showSnapshotWarning = hasSnapshot && snapshotState.warning;

    if (showSnapshotProgress) {
      progressValue = clampTrackValue(snapshotState.progressValue);
      trackText = snapshotState.text || trackText;
    } else if (state === 'ready') {
      progressValue = 100;
    } else if (state === 'partial_ready') {
      progressValue = 100;
      trackText = (hasSnapshot && snapshotState.text) || trackText;
    } else if (state === 'snapshot_building') {
      progressValue = Math.max(clampTrackValue(hasSnapshot ? snapshotState.progressValue : 0), 24);
      trackText = (hasSnapshot && snapshotState.text) || trackText;
    } else if (state === 'refreshing') {
      progressValue = 42;
    } else if (state === 'stale') {
      progressValue = 12;
    }

    statusTrackFill.style.width = `${progressValue}%`;
    statusTrackText.textContent = trackText || getStatusMessage(state);
    statusTrackShell.dataset.state = state || 'ready';
    statusTrackShell.classList.toggle('is-loading', state === 'refreshing' || state === 'snapshot_building' || showSnapshotProgress);
    statusTrackShell.classList.toggle('is-warning', state === 'partial_ready' || showSnapshotWarning);
    statusTrackShell.classList.toggle('is-ready', state === 'ready' && !showSnapshotWarning);
    statusTrackShell.classList.toggle('is-stale', state === 'stale');
    const progressBar = statusTrackFill.closest('.mi-hub-v2-track-bar, .mi-hub-status-track-bar');
    if (progressBar) {
      progressBar.setAttribute('aria-valuenow', String(progressValue));
    }
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
    renderStatusTrack();
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
    const formatted = value ? formatDisplayTime(value) : '';
    if (statusUpdated) {
      statusUpdated.textContent = formatted || '';
      if (statusUpdated.tagName === 'TIME') {
        statusUpdated.setAttribute('datetime', value || '');
      }
    }
    updateDenseStrip();
  }

  function updateControlsSummary() {
    if (autoRefreshSummary) {
      let autoMode = TEXT.autoSummaryOff;
      if (autoRefreshMs) {
        autoMode = `${Math.round(autoRefreshMs / 1000)}s`;
        if (autoRefreshPaused || autoRefreshSuspended) {
          autoMode = `${autoMode} · ${TEXT.paused}`;
        }
      }
      autoRefreshSummary.textContent = `${TEXT.autoSummaryLabel}: ${autoMode}`;
    }
    if (timezoneSummary) {
      const timezoneLabel = timezoneMode === 'local' ? TEXT.timezoneLocalLabel : 'UTC';
      timezoneSummary.textContent = `${TEXT.timezoneSummaryLabel}: ${timezoneLabel}`;
    }
    if (statusControlsToggle) {
      statusControlsToggle.textContent = controlsExpanded ? TEXT.controlsCollapse : TEXT.controlsExpand;
    }
  }

  function getControlsPopoverFocusableElements() {
    if (!statusControlsAdvanced) return [];
    const selector = [
      'button:not([disabled])',
      '[href]',
      'input:not([disabled]):not([type="hidden"])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])'
    ].join(',');
    const nodes = Array.prototype.slice.call(statusControlsAdvanced.querySelectorAll(selector));
    return nodes.filter((node) => {
      if (!(node instanceof HTMLElement)) return false;
      if (node.hidden || node.getAttribute('aria-hidden') === 'true') return false;
      const style = window.getComputedStyle(node);
      if (style.display === 'none' || style.visibility === 'hidden') return false;
      return node.getClientRects().length > 0;
    });
  }

  function focusFirstControlsPopoverControl() {
    if (!controlsExpanded || !statusControlsAdvanced) return;
    const preferred = statusControlsAdvanced.querySelector('[data-role="auto-refresh-toggle"] button:not([disabled])');
    if (preferred instanceof HTMLElement) {
      preferred.focus({ preventScroll: true });
      return;
    }
    const focusables = getControlsPopoverFocusableElements();
    if (focusables.length && focusables[0] instanceof HTMLElement) {
      focusables[0].focus({ preventScroll: true });
    }
  }

  function trapControlsPopoverFocus(event) {
    if (!controlsExpanded || event.key !== 'Tab') return;
    const focusables = getControlsPopoverFocusableElements();
    if (!focusables.length) {
      event.preventDefault();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    const activeInside = active instanceof HTMLElement && statusControlsAdvanced && statusControlsAdvanced.contains(active);
    if (!activeInside) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus({ preventScroll: true });
      return;
    }
    if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus({ preventScroll: true });
      return;
    }
    if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus({ preventScroll: true });
    }
  }

  function positionControlsPopover() {
    if (!statusControlsAdvanced || !statusControlsToggle) return;
    statusControlsAdvanced.style.left = '';
    statusControlsAdvanced.style.top = '';
    statusControlsAdvanced.style.right = '';
    statusControlsAdvanced.style.bottom = '';
    statusControlsAdvanced.style.transform = '';
    statusControlsAdvanced.style.maxHeight = '';

    const isCompactViewport = Boolean(window.matchMedia && window.matchMedia('(max-width: 1023px)').matches);
    if (isCompactViewport) return;

    const gap = 10;
    const margin = 12;
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1280;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 720;
    const toggleRect = statusControlsToggle.getBoundingClientRect();
    const panelRect = statusControlsAdvanced.getBoundingClientRect();

    let left = toggleRect.left;
    let top = toggleRect.bottom + gap;

    if (left + panelRect.width + margin > viewportWidth) {
      left = toggleRect.right - panelRect.width;
    }
    if (left < margin) {
      left = margin;
    }

    if (top + panelRect.height + margin > viewportHeight) {
      top = toggleRect.top - panelRect.height - gap;
    }
    if (top < margin) {
      top = margin;
    }

    statusControlsAdvanced.style.left = `${Math.round(left)}px`;
    statusControlsAdvanced.style.top = `${Math.round(top)}px`;
    statusControlsAdvanced.style.maxHeight = `${Math.max(220, viewportHeight - margin * 2)}px`;
  }

  function setControlsExpanded(expanded) {
    controlsExpanded = Boolean(expanded);
    if (statusControlsAdvanced) {
      statusControlsAdvanced.hidden = !controlsExpanded;
      if (!controlsExpanded) {
        statusControlsAdvanced.style.left = '';
        statusControlsAdvanced.style.top = '';
        statusControlsAdvanced.style.right = '';
        statusControlsAdvanced.style.bottom = '';
        statusControlsAdvanced.style.transform = '';
        statusControlsAdvanced.style.maxHeight = '';
      }
    }
    if (controlsPopoverBackdrop) {
      controlsPopoverBackdrop.hidden = !controlsExpanded;
    }
    if (statusControlsToggle) {
      statusControlsToggle.setAttribute('aria-expanded', controlsExpanded ? 'true' : 'false');
      statusControlsToggle.classList.toggle('is-active', controlsExpanded);
    }
    if (statusSection) {
      statusSection.classList.toggle('is-controls-expanded', controlsExpanded);
    }
    if (controlsExpanded) {
      window.requestAnimationFrame(() => {
        positionControlsPopover();
        focusFirstControlsPopoverControl();
      });
    }
    updateControlsSummary();
  }

  function updateDenseStrip() {
    if (denseList) {
      denseList.textContent = getListLabelText(denseListType || currentListType);
    }
    if (denseTimeframe) {
      denseTimeframe.textContent = (TEXT.timeframes && TEXT.timeframes[denseTimeframeKey]) || denseTimeframeKey || '--';
    }
    if (denseCount) {
      denseCount.textContent = Array.isArray(denseItems) ? String(denseItems.length) : '0';
    }
    const listMeta = getListMeta(denseListType || currentListType);
    const sortable = (Array.isArray(denseItems) ? denseItems : [])
      .map((item) => ({ item, metric: resolveMetricValue(item, denseListType || currentListType) }))
      .filter((entry) => typeof entry.metric === 'number' && Number.isFinite(entry.metric))
      .sort((left, right) => right.metric - left.metric);
    const leader = sortable[0] || null;
    const laggard = sortable[sortable.length - 1] || null;
    if (denseLeaderSymbol) {
      denseLeaderSymbol.textContent = leader && leader.item && leader.item.symbol ? leader.item.symbol : '—';
    }
    if (denseLeaderMetric) {
      denseLeaderMetric.textContent = leader ? formatMetricValue(leader.metric, listMeta) : '—';
      denseLeaderMetric.classList.remove('is-up', 'is-down');
      if (leader) applyChangeState(denseLeaderMetric, leader.metric, listMeta && listMeta.invert);
    }
    if (denseLaggardSymbol) {
      denseLaggardSymbol.textContent = laggard && laggard.item && laggard.item.symbol ? laggard.item.symbol : '—';
    }
    if (denseLaggardMetric) {
      denseLaggardMetric.textContent = laggard ? formatMetricValue(laggard.metric, listMeta) : '—';
      denseLaggardMetric.classList.remove('is-up', 'is-down');
      if (laggard) applyChangeState(denseLaggardMetric, laggard.metric, listMeta && listMeta.invert);
    }
    if (denseAuto) {
      let autoText = TEXT.autoSummaryOff || 'Off';
      if (autoRefreshMs) {
        autoText = `${Math.round(autoRefreshMs / 1000)}s`;
        if (autoRefreshPaused || autoRefreshSuspended) {
          autoText = `${autoText}/${TEXT.paused}`;
        }
      }
      denseAuto.textContent = autoText;
    }
    if (denseUpdated) {
      denseUpdated.textContent = lastStatusGeneratedAt ? formatDisplayTime(lastStatusGeneratedAt) : '—';
    }
    if (denseSource) {
      const labels = TEXT.sourceLabels || {};
      const label = labels[denseSourceKey] || labels.unknown || '—';
      denseSource.textContent = label || '—';
    }
  }

  function updateTimezoneToggle() {
    if (!timezoneToggles.length) return;
    timezoneToggles.forEach((toggle) => {
      toggle.querySelectorAll('button[data-tz]').forEach((btn) => {
        const isActive = btn.dataset.tz === timezoneMode;
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });
    });
    updateControlsSummary();
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
    updateControlsSummary();
    updateDenseStrip();
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
      const intervalSpec = resolveIntervalSpec(detailChartCache.interval || detailInterval);
      updateDetailTimes(detailChartCache.payload, {
        intervalSpec,
        bars: detailChartCache.bars,
      });
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
    denseListType = currentListType;
    denseTimeframeKey = currentTimeframe;
    updateDenseStrip();
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
    const maxAttempts = 50;
    const retryDelayMs = 150;
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
      if (attempts < maxAttempts) {
        chartInitTimer = setTimeout(tryInit, retryDelayMs);
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

  function setRankGroupsExpanded(expanded) {
    rankGroupsExpanded = Boolean(expanded);
    if (rankGroupsAdvanced) {
      rankGroupsAdvanced.hidden = !rankGroupsExpanded;
    }
    if (rankGroupsToggle) {
      rankGroupsToggle.setAttribute('aria-expanded', rankGroupsExpanded ? 'true' : 'false');
      rankGroupsToggle.textContent = rankGroupsExpanded ? TEXT.rankListsCollapse : TEXT.rankListsExpand;
      rankGroupsToggle.classList.toggle('is-active', rankGroupsExpanded);
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
    if (rankingsSection) {
      rankingsSection.hidden = false;
    }
    if (allStocksSection) {
      allStocksSection.hidden = true;
    }
    if (allStocksFilter) {
      allStocksFilter.hidden = type !== 'all';
    }
    if (type === 'all') {
      denseItems = [];
      setAllStocksLetter(allStocksLetter);
    }
    denseListType = type;
    updateDenseStrip();
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
        const leftValue = coerceNumber(left && (left.change_pct_period ?? left.change_pct));
        const rightValue = coerceNumber(right && (right.change_pct_period ?? right.change_pct));
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
    if (!isListVisible()) return;
    if (currentListType === 'all') {
      if (!allStocksIsLoadingMore && allStocksHasMore) {
        loadAllStocks({ append: true });
      }
      return;
    }
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
      { root: rankingScrollRoot || null, rootMargin: '200px' },
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
    const labels = TEXT.sourceLabels || {};
    const normalizedKey = sourceKey && labels[sourceKey] ? sourceKey : 'unknown';
    const label = labels[normalizedKey] || labels.unknown || '';
    denseSourceKey = normalizedKey;
    if (sourceText) {
      sourceText.textContent = `${TEXT.sourcePrefix || ''}${label}`;
    }
    updateDenseStrip();
  }

  function buildSnapshotPrefix(timeframeKey) {
    const key = typeof timeframeKey === 'string' ? timeframeKey : '';
    const label = key && key !== '1d' ? (TEXT.timeframes && TEXT.timeframes[key]) || key : '';
    return label ? `${TEXT.snapshotPrefix}(${label})` : TEXT.snapshotPrefix;
  }

  function formatSnapshotDuration(seconds) {
    const numeric = Number(seconds);
    if (!Number.isFinite(numeric) || numeric < 0) return '';
    const rounded = Math.round(numeric);
    const minutes = Math.floor(rounded / 60);
    const remainSeconds = rounded % 60;
    if (langPrefix === 'zh') {
      if (minutes <= 0) return `${rounded}秒`;
      return remainSeconds > 0 ? `${minutes}分${remainSeconds}秒` : `${minutes}分`;
    }
    if (minutes <= 0) return `${rounded}s`;
    return remainSeconds > 0 ? `${minutes}m ${remainSeconds}s` : `${minutes}m`;
  }

  function snapshotStatusLabel(status) {
    const normalized = String(status || '').toLowerCase();
    if (langPrefix === 'zh') {
      if (normalized === 'complete') return '完成';
      if (normalized === 'running') return '进行中';
      if (normalized === 'error') return '失败';
      if (normalized === 'stale') return '过期';
      return normalized || '未知';
    }
    if (normalized === 'complete') return 'complete';
    if (normalized === 'running') return 'running';
    if (normalized === 'error') return 'failed';
    if (normalized === 'stale') return 'stale';
    return normalized || 'unknown';
  }

  function collectUnresolvedTimeframes(timeframesMeta) {
    if (!timeframesMeta || typeof timeframesMeta !== 'object') return [];
    return Object.entries(timeframesMeta)
      .map(([key, payload]) => {
        if (!payload || typeof payload !== 'object') return null;
        const status = String(payload.status || '').toLowerCase();
        if (!status || status === 'complete') return null;
        const label = (TEXT.timeframes && TEXT.timeframes[key]) || key;
        const error = payload.error ? String(payload.error) : '';
        return { key, label, status, error };
      })
      .filter(Boolean);
  }

  function classifySnapshotError(rawMessage) {
    const message = String(rawMessage || '').trim();
    const normalized = message.toLowerCase();
    const fallback = {
      title: langPrefix === 'zh' ? '刷新任务异常' : 'Refresh job failure',
      advice:
        langPrefix === 'zh'
          ? '点击“重试”重新触发；若持续失败，请检查网络与数据源状态。'
          : 'Click retry to trigger a new refresh. If it keeps failing, check network and data source health.',
    };
    if (!message) return fallback;
    if (normalized.includes('missing_credentials')) {
      return {
        title: langPrefix === 'zh' ? '缺少 Alpaca 凭证' : 'Missing Alpaca credentials',
        advice:
          langPrefix === 'zh'
            ? '请在账户中心补全 API Key / Secret 后再重试。'
            : 'Configure API key/secret in account settings, then retry.',
      };
    }
    if (normalized.includes('429') || normalized.includes('rate') || normalized.includes('too many')) {
      return {
        title: langPrefix === 'zh' ? '触发接口限流' : 'API rate limit hit',
        advice:
          langPrefix === 'zh'
            ? '建议把自动刷新间隔调到 30s 或 60s，再次尝试。'
            : 'Use 30s/60s auto-refresh to reduce pressure before retrying.',
      };
    }
    if (normalized.includes('timeout') || normalized.includes('timed out')) {
      return {
        title: langPrefix === 'zh' ? '请求超时' : 'Request timeout',
        advice:
          langPrefix === 'zh'
            ? '网络或数据源响应较慢，请稍后重试。'
            : 'Network or upstream response is slow; retry shortly.',
      };
    }
    if (
      normalized.includes('connection') ||
      normalized.includes('network') ||
      normalized.includes('dns') ||
      normalized.includes('unreachable')
    ) {
      return {
        title: langPrefix === 'zh' ? '网络连接问题' : 'Network connectivity issue',
        advice:
          langPrefix === 'zh'
            ? '请检查当前网络或代理设置后重试。'
            : 'Check network/proxy connectivity and retry.',
      };
    }
    return fallback;
  }

  function buildSnapshotWarningDetail({
    reason,
    prefix,
    progress = null,
    errorMessage = '',
    unresolved = [],
    generatedAt = '',
  }) {
    const lines = [];
    if (reason === 'error') {
      const diagnosis = classifySnapshotError(errorMessage);
      lines.push(
        langPrefix === 'zh' ? `问题类型：${diagnosis.title}` : `Issue type: ${diagnosis.title}`
      );
      if (errorMessage) {
        lines.push(
          langPrefix === 'zh'
            ? `错误信息：${errorMessage}`
            : `Error message: ${errorMessage}`
        );
      }
      const apiCalls = Number(progress && progress.api_calls);
      if (Number.isFinite(apiCalls) && apiCalls >= 0) {
        lines.push(
          langPrefix === 'zh'
            ? `接口调用：${apiCalls} 次`
            : `API calls: ${apiCalls}`
        );
      }
      const elapsedText = formatSnapshotDuration(progress && progress.elapsed_seconds);
      if (elapsedText) {
        lines.push(
          langPrefix === 'zh'
            ? `已运行时长：${elapsedText}`
            : `Elapsed runtime: ${elapsedText}`
        );
      }
      lines.push(diagnosis.advice);
      return lines.join('\n');
    }

    if (reason === 'timeframes') {
      if (generatedAt) {
        lines.push(
          langPrefix === 'zh'
            ? `${prefix}主快照时间：${formatDisplayTime(generatedAt)}`
            : `${prefix} main snapshot time: ${formatDisplayTime(generatedAt)}`
        );
      }
      if (unresolved.length) {
        const issueList = unresolved.slice(0, 4).map((issue) => {
          const status = snapshotStatusLabel(issue.status);
          if (issue.error) {
            return langPrefix === 'zh'
              ? `${issue.label}（${status}）：${issue.error}`
              : `${issue.label} (${status}): ${issue.error}`;
          }
          return langPrefix === 'zh'
            ? `${issue.label}（${status}）`
            : `${issue.label} (${status})`;
        });
        lines.push(
          langPrefix === 'zh'
            ? `未完成时间窗：${issueList.join('；')}`
            : `Incomplete windows: ${issueList.join('; ')}`
        );
        if (unresolved.length > 4) {
          const extra = unresolved.length - 4;
          lines.push(
            langPrefix === 'zh'
              ? `其余 ${extra} 个时间窗仍在处理中。`
              : `${extra} more windows are still processing.`
          );
        }
      }
      lines.push(
        langPrefix === 'zh'
          ? '建议：可继续查看当前榜单；系统会自动补齐，也可手动点击“重试”。'
          : 'You can keep using current rankings; missing windows will backfill automatically, or click Retry now.'
      );
      return lines.join('\n');
    }

    if (reason === 'stale') {
      lines.push(
        langPrefix === 'zh'
          ? '尚未读取到有效快照生成时间，展示结果可能偏旧。'
          : 'No valid snapshot timestamp found; displayed results may be stale.'
      );
      lines.push(
        langPrefix === 'zh'
          ? '建议：点击“重试”刷新，或等待下一次自动刷新。'
          : 'Click Retry or wait for the next auto refresh.'
      );
      return lines.join('\n');
    }

    return TEXT.partialDetail;
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
      result.warning = true;
      result.warningText =
        langPrefix === 'zh' ? '暂无有效快照，显示可能过期' : 'No valid snapshot; results may be stale';
      result.warningDetail = buildSnapshotWarningDetail({ reason: 'stale', prefix });
      return result;
    }

    const progress = meta.progress;
    const timeframesMeta = meta.timeframes && typeof meta.timeframes === 'object' ? meta.timeframes : null;
    const unresolvedTimeframes = collectUnresolvedTimeframes(timeframesMeta);
    const latestSummary = meta.latest && typeof meta.latest === 'object' ? meta.latest : null;

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
      if (progressValue >= 100) {
        progressValue = 99;
      }
      const elapsedText = formatSnapshotDuration(progress.elapsed_seconds);
      const finalizingSuffix =
        progressValue >= 95
          ? (langPrefix === 'zh' ? ' · 收尾中' : ' · finalizing')
          : '';
      const elapsedSuffix = elapsedText
        ? (langPrefix === 'zh' ? ` · 已运行 ${elapsedText}` : ` · running ${elapsedText}`)
        : '';
      result.text = `${prefix}：${TEXT.snapshotRunning}${progressText}${elapsedSuffix}${finalizingSuffix}`;
      result.progressValue = progressValue;
      result.showProgress = true;
      result.state = 'snapshot_building';
      return result;
    }

    if (meta.error) {
      const errorMsg = meta.error && meta.error.error ? String(meta.error.error) : '';
      result.text = `${prefix}：${TEXT.snapshotError}`;
      result.state = 'partial_ready';
      result.warning = true;
      result.warningText =
        langPrefix === 'zh' ? '快照刷新失败，需要人工重试' : 'Snapshot refresh failed; retry required';
      result.warningDetail = buildSnapshotWarningDetail({
        reason: 'error',
        prefix,
        progress: meta.error,
        errorMessage: errorMsg,
      });
      result.progressValue = 0;
      return result;
    }

    let latest = null;
    if (timeframeKey && timeframeKey !== '1d' && timeframesMeta && timeframesMeta[timeframeKey]) {
      latest = timeframesMeta[timeframeKey];
    }
    if (!latest && latestSummary) {
      latest = latestSummary;
    }
    const generatedAt = latest && latest.generated_at ? String(latest.generated_at) : '';

    if (generatedAt) {
      result.text = `${prefix}：${TEXT.snapshotUpdated} ${formatDisplayTime(generatedAt)}`;
      result.progressValue = 100;
      result.showProgress = false;
      if (unresolvedTimeframes.length) {
        result.state = 'partial_ready';
        result.warning = true;
        result.warningText =
          langPrefix === 'zh'
            ? `部分数据尚未就绪（${unresolvedTimeframes.length}个时间窗）`
            : `Partial data pending (${unresolvedTimeframes.length} windows)`;
        result.warningDetail = buildSnapshotWarningDetail({
          reason: 'timeframes',
          prefix,
          unresolved: unresolvedTimeframes,
          generatedAt,
        });
      } else {
        result.state = 'ready';
      }
      return result;
    }

    if (unresolvedTimeframes.length) {
      result.state = 'partial_ready';
      result.warning = true;
      result.warningText =
        langPrefix === 'zh' ? '时间窗快照尚未完成' : 'Timeframe snapshots are not complete';
      result.warningDetail = buildSnapshotWarningDetail({
        reason: 'timeframes',
        prefix,
        unresolved: unresolvedTimeframes,
      });
      return result;
    }

    result.state = 'stale';
    result.warning = true;
    result.warningText =
      langPrefix === 'zh' ? '数据可能过期，请刷新' : 'Data may be stale, please refresh';
    result.warningDetail = buildSnapshotWarningDetail({ reason: 'stale', prefix });
    return result;
  }

  function setSnapshotStatus(meta, timeframeKey) {
    lastSnapshotMeta = meta;
    lastSnapshotKey = timeframeKey;
    const result = formatSnapshotStatus(meta, timeframeKey);
    if (snapshotText) {
      snapshotText.textContent = result.text;
      snapshotText.classList.toggle('is-error', result.state === 'partial_ready');
    }
    if (snapshotProgressFill) {
      const progressBar = snapshotProgressFill.closest('.market-progress-bar');
      const clamped = Math.min(Math.max(Number(result.progressValue) || 0, 0), 100);
      snapshotProgressFill.style.width = `${clamped}%`;
      if (progressBar) {
        progressBar.setAttribute('aria-valuenow', String(clamped));
      }
    }
    if (snapshotProgress) {
      snapshotProgress.hidden = true;
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
      if (snapshotDetails) {
        snapshotDetails.hidden = !result.warning;
      }
      snapshotWarning.hidden = !result.warning;
    }
    if (statusExpanded) {
      statusExpanded.hidden = !result.warning;
      statusExpanded.classList.toggle('is-active', result.warning);
    }
    snapshotState = result;
    refreshStatusState();
    renderStatusTrack();
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
        priceEl.textContent = formatPrice4(price);
      }
      if (dayEl && Number.isFinite(changePct)) {
        dayEl.textContent = formatChange(changePct);
        applyChangeState(dayEl, changePct, card.dataset.invert === '1', true);
      }
    });
    const tableRows = document.querySelectorAll(`tr[data-symbol="${symbol}"]`);
    tableRows.forEach((row) => {
      const cells = row.querySelectorAll('td');
      if (cells.length < 3) return;
      const priceCell = cells[1];
      const changeCell = cells[2];
      if (priceCell && Number.isFinite(price)) {
        priceCell.textContent = formatPrice4(price);
      }
      if (!changeCell || !Number.isFinite(changePct)) return;
      const isRankingRow = Boolean(row.closest('[data-role="ranking-list"]'));
      if (!isRankingRow) return;
      if (currentListType === 'all') {
        if (currentTimeframe !== '1d') return;
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
      if (Number.isFinite(price)) {
        setDetailPriceValue(price);
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
      stopLiveQuotePolling();
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
    const waitMs = isHighFreqInterval(detailInterval) ? Math.min(4000, LIVE_WAIT_MS) : LIVE_WAIT_MS;
    if (shouldPollChart(detailInterval)) {
      startChartPolling();
    }
    if (chartVisible && detailSymbol) {
      startTickFallbackPolling();
    }
    liveWaitTimer = setTimeout(() => {
      if (!chartVisible || !detailSymbol) return;
      if (lastLiveUpdateAt || lastFallbackUpdateAt) return;
      setDetailStatus(TEXT.detailLiveWaiting);
      startLiveQuotePolling(detailSymbol);
      if (shouldPollChart(detailInterval)) {
        startChartPolling();
      }
      startTickFallbackPolling();
    }, waitMs);
  }

  function stopLiveQuotePolling() {
    if (liveQuoteTimer) {
      clearInterval(liveQuoteTimer);
      liveQuoteTimer = null;
    }
  }

  function isHighFreqInterval(intervalKey) {
    const spec = resolveIntervalSpec(intervalKey);
    return Boolean(spec && (spec.unit === 'tick' || spec.unit === 'second'));
  }

  function shouldPollChart(intervalKey) {
    const spec = resolveIntervalSpec(intervalKey);
    if (!spec) return false;
    return true;
  }

  function resolveChartPollInterval(intervalKey) {
    const normalized = normalizeIntervalKey(intervalKey) || intervalKey;
    const spec = resolveIntervalSpec(normalized);
    if (!spec) return CHART_POLL_MS;
    if (spec.unit === 'tick') return CHART_POLL_MS_TICK;
    if (spec.unit === 'second') return CHART_POLL_MS_SECOND;
    if (spec.unit === 'minute') return CHART_POLL_MS_MINUTE;
    if (spec.unit === 'hour') return CHART_POLL_MS_HOUR;
    if (spec.unit === 'day') return CHART_POLL_MS_DAY;
    return CHART_POLL_MS;
  }

  function stopChartPolling() {
    if (chartPollTimer) {
      clearInterval(chartPollTimer);
      chartPollTimer = null;
    }
    chartPollInFlight = false;
    chartPollIntervalMs = CHART_POLL_MS;
  }

  function startChartPolling() {
    const targetMs = resolveChartPollInterval(detailInterval);
    if (chartPollTimer && chartPollIntervalMs === targetMs) return;
    stopChartPolling();
    chartPollIntervalMs = targetMs;
    chartPollTimer = setInterval(() => {
      refreshChartBarsOnly();
    }, chartPollIntervalMs);
    refreshChartBarsOnly();
  }

  function stopTickFallbackPolling() {
    if (chartTickPollTimer) {
      clearInterval(chartTickPollTimer);
      chartTickPollTimer = null;
    }
    chartTickPollInFlight = false;
  }

  function startTickFallbackPolling() {
    if (chartTickPollTimer) return;
    chartTickPollTimer = setInterval(() => {
      refreshTickFallback();
    }, CHART_TICK_FALLBACK_MS);
    refreshTickFallback();
  }

  function seedTickFallbackCursor(bars) {
    if (!Array.isArray(bars) || !bars.length) {
      chartTickCursor = null;
      return;
    }
    const lastBar = bars[bars.length - 1];
    const lastTime = lastBar ? normalizeEpochSeconds(lastBar.time) : null;
    if (!Number.isFinite(lastTime)) {
      chartTickCursor = null;
      return;
    }
    const now = Date.now() / 1000;
    const minCursor = now - CHART_TICK_FALLBACK_WINDOW_SEC;
    const candidate = Math.max(lastTime, minCursor);
    chartTickCursor = Number.isFinite(chartTickCursor) ? Math.max(chartTickCursor, candidate) : candidate;
  }

  async function refreshTickFallback() {
    if (chartTickPollInFlight) return;
    if (!chartVisible || !detailSymbol || !detailManager) return;
    if (isChartRealtimeActive()) return;
    const chartContext = getChartContextSnapshot();
    const symbol = detailSymbol;
    const rangeKey = detailRange;
    chartTickPollInFlight = true;
    try {
      const endTs = Date.now() / 1000;
      let startTs = endTs - CHART_TICK_FALLBACK_WINDOW_SEC;
      if (Number.isFinite(chartTickCursor)) {
        startTs = Math.max(startTs, chartTickCursor + 1e-6);
      }
      if (!Number.isFinite(startTs) || startTs >= endTs) return;
      const payload = await fetchChartBars(symbol, rangeKey, '1t', { startTs, endTs });
      if (!isChartContextCurrent(chartContext)) return;
      const bars = payload && Array.isArray(payload.bars) ? payload.bars : [];
      if (!bars.length) return;
      let lastPrice = null;
      let lastTs = null;
      let applied = 0;
      const cursor = Number.isFinite(chartTickCursor) ? chartTickCursor : null;
      bars.forEach((bar) => {
        if (!bar) return;
        const ts = normalizeEpochSeconds(bar.time);
        if (!Number.isFinite(ts)) return;
        if (cursor !== null && ts <= cursor + 1e-6) return;
        const price = typeof bar.close === 'number' ? bar.close : Number.parseFloat(bar.close);
        if (!Number.isFinite(price)) return;
        const size = typeof bar.volume === 'number' ? bar.volume : Number.parseFloat(bar.volume);
        detailManager.applyTradeUpdate({ price, size: Number.isFinite(size) ? size : 0, ts });
        chartTickCursor = ts;
        lastPrice = price;
        lastTs = ts;
        applied += 1;
      });
      if (!applied || lastPrice === null || lastTs === null) return;
      lastFallbackUpdateAt = Date.now();
      setDetailPriceValue(lastPrice);
      if (detailChangeEl && detailManager && detailManager.ohlcData.length > 1) {
        const last = detailManager.ohlcData[detailManager.ohlcData.length - 1];
        const prev = detailManager.ohlcData[detailManager.ohlcData.length - 2];
        const changePct = prev && prev.close ? ((last.close / prev.close) - 1) * 100 : null;
        detailChangeEl.textContent = typeof changePct === 'number' ? formatChange(changePct) : '--';
        detailChangeEl.classList.remove('is-up', 'is-down');
        applyChangeState(detailChangeEl, changePct, false);
      }
      if (detailSubtitle && detailManager && detailManager.ohlcData.length) {
        const lastBar = detailManager.ohlcData[detailManager.ohlcData.length - 1];
        const lastBarTs = lastBar ? normalizeEpochSeconds(lastBar.time) : null;
        if (Number.isFinite(lastBarTs)) {
          const intervalSpec = detailManager.intervalSpec;
          const showSeconds = Boolean(intervalSpec && (intervalSpec.unit === 'tick' || intervalSpec.unit === 'second'));
          const timeLabel = formatAxisTime(lastBarTs, {
            timezoneMode,
            showSeconds,
            includeDate: true,
            fullDate: true,
          });
          detailSubtitle.textContent = langPrefix === 'zh' ? `数据时间：${timeLabel}` : `Data time: ${timeLabel}`;
        }
      }
      const latencyTs =
        payload && Number.isFinite(payload.latest_trade_ts) ? payload.latest_trade_ts : lastTs;
      if (Number.isFinite(latencyTs)) {
        const intervalSpec = detailManager ? detailManager.intervalSpec : null;
        updateDetailLatencyFromTs(latencyTs, { intervalSpec });
      }
      if (detailStatusOverride === TEXT.detailLiveWaiting) {
        setDetailStatus('');
      }
    } catch (error) {
      // ignore fallback errors
    } finally {
      chartTickPollInFlight = false;
    }
  }

  async function refreshChartBarsOnly() {
    if (chartPollInFlight) return;
    if (!chartVisible || !detailSymbol) return;
    if (!shouldPollChart(detailInterval)) return;
    if (chartSocketReady && isChartRealtimeActive()) return;
    const chartContext = getChartContextSnapshot();
    const symbol = detailSymbol;
    const rangeKey = detailRange;
    chartPollInFlight = true;
    try {
      const intervalKey = normalizeIntervalKey(detailInterval) || resolveDefaultInterval(rangeKey);
      if (!intervalKey) return;
      const payload = await fetchChartBars(symbol, rangeKey, intervalKey);
      if (!isChartContextCurrent(chartContext)) return;
      if (!payload || !Array.isArray(payload.bars) || !payload.bars.length) return;
      const intervalSpec = resolveIntervalSpec(intervalKey);
      if (detailManager) {
        const merged = detailManager.mergeData(payload.bars);
        if (!merged) {
          detailManager.setData(payload.bars, { intervalSpec, fitContent: false });
        }
        chartLazyCursor = detailManager.ohlcData[0] ? detailManager.ohlcData[0].time : chartLazyCursor;
        chartLazyExhausted = false;
        chartLazyEmptyStreak = 0;
        scheduleWaveAutoAnalyze();
      }
      updateDetailTimes(payload, {
        updateSubtitle: true,
        intervalSpec,
        bars: detailManager && detailManager.ohlcData ? detailManager.ohlcData : payload.bars,
      });
      updateDetailLatencyFromTs(payload.latest_trade_ts || payload.server_ts);
      if (payload.window_limited) {
        setDetailStatus(TEXT.detailWindowLimited);
      } else if (payload.downgrade_message) {
        setDetailStatus(TEXT.detailDowngraded(payload.downgrade_message));
      } else {
        setDetailStatus('');
      }
    } catch (error) {
      // keep silent; status is handled by live stream
    } finally {
      chartPollInFlight = false;
    }
  }

  async function loadOlderChartBars() {
    if (chartLazyLoading || chartLazyExhausted) return;
    if (!detailSymbol || !detailManager) return;
    const chartContext = getChartContextSnapshot();
    const symbol = detailSymbol;
    const rangeKey = detailRange;
    const intervalKey = normalizeIntervalKey(detailInterval) || resolveDefaultInterval(rangeKey);
    const intervalSpec = resolveIntervalSpec(intervalKey);
    if (!intervalSpec) return;
    const earliest =
      Number.isFinite(chartLazyCursor) ? chartLazyCursor : detailManager.ohlcData[0] && detailManager.ohlcData[0].time;
    if (!Number.isFinite(earliest)) return;
    chartLazyLoading = true;
    setDetailLazy(true);
    try {
      const step = intervalSpec.unit === 'tick' ? 0.000001 : Math.max(1, intervalSpec.seconds || 1);
      const rangeSpan = resolveRangeWindowSeconds(rangeKey);
      const jumpSeconds = Math.max(CHART_LAZY_MIN_JUMP_SECONDS, Math.ceil(rangeSpan));
      let probeCursor = earliest;
      let inserted = false;
      let latestTradeTs = null;
      for (let attempt = 0; attempt < CHART_LAZY_EMPTY_RETRY_ATTEMPTS; attempt += 1) {
        const endTs = probeCursor - step;
        const payload = await fetchChartBars(symbol, rangeKey, intervalKey, { endTs });
        if (!isChartContextCurrent(chartContext)) return;
        const bars = payload && Array.isArray(payload.bars) ? payload.bars : [];
        if (bars.length && detailManager.prependData(bars)) {
          inserted = true;
          latestTradeTs = payload.latest_trade_ts || null;
          break;
        }
        probeCursor = Math.max(0, endTs - jumpSeconds);
      }
      if (inserted) {
        chartLazyCursor = detailManager.ohlcData[0] ? detailManager.ohlcData[0].time : chartLazyCursor;
        chartLazyEmptyStreak = 0;
        chartLazyExhausted = false;
        if (latestTradeTs) {
          updateDetailLatencyFromTs(latestTradeTs, { intervalSpec });
        }
        scheduleWaveAutoAnalyze();
        return;
      }
      chartLazyCursor = Math.max(0, probeCursor);
      chartLazyEmptyStreak += 1;
      if (chartLazyEmptyStreak >= CHART_LAZY_MAX_EMPTY_STREAK) {
        chartLazyExhausted = true;
      }
    } catch (error) {
      if (error && error.status === 404) {
        chartLazyExhausted = true;
      }
    } finally {
      chartLazyLoading = false;
      setDetailLazy(false);
    }
  }

  function bindChartLazyLoad() {
    if (!detailManager || !detailManager.chart) return;
    if (detailManager._lazyLoadBound) return;
    const timeScale = detailManager.chart.timeScale();
    if (!timeScale || typeof timeScale.subscribeVisibleTimeRangeChange !== 'function') return;
    detailManager._lazyLoadBound = true;
    const handleMaybeLoad = (logicalRangeHint = null, options = {}) => {
      if (!chartVisible || !detailSymbol || !detailManager) return;
      const fromInteraction = Boolean(options && options.fromInteraction);
      const recentlyInteracted = Date.now() - detailLastRangeInteractionAt < 900;
      if (
        detailAutoRangeLock &&
        !fromInteraction &&
        !recentlyInteracted &&
        !(detailManager && detailManager._panActive)
      ) {
        return;
      }
      if (chartLazyLoading || chartLazyExhausted) return;
      let logicalRange = logicalRangeHint;
      if (
        !logicalRange ||
        !Number.isFinite(logicalRange.from) ||
        !Number.isFinite(logicalRange.to)
      ) {
        logicalRange =
          typeof timeScale.getVisibleLogicalRange === 'function' ? timeScale.getVisibleLogicalRange() : null;
      }
      if (!logicalRange || !Number.isFinite(logicalRange.from) || !Number.isFinite(logicalRange.to)) return;
      const total = detailManager.ohlcData ? detailManager.ohlcData.length : 0;
      if (!total) return;
      const lastIndex = total - 1;
      const span = Math.max(1, logicalRange.to - logicalRange.from);
      const nearRight = logicalRange.to >= lastIndex - 1;
      detailUserPanned = !nearRight;
      const maxFrom = lastIndex;
      if (logicalRange.from > maxFrom + 1e-6) {
        withAutoRangeLock(() => {
          timeScale.setVisibleLogicalRange({ from: maxFrom, to: maxFrom + span });
        });
        return;
      }
      const futureGap = logicalRange.to - lastIndex;
      const loadGap = span * 0.4;
      const leftEdgeThreshold = Math.max(2, span * 0.1);
      if (futureGap >= loadGap || logicalRange.from <= leftEdgeThreshold) {
        loadOlderChartBars();
      }
    };
    detailManager._onRangeInteraction = (range) => {
      handleMaybeLoad(range, { fromInteraction: true });
    };
    detailManager._triggerLazyLoadCheck = () => {
      handleMaybeLoad(null, { fromInteraction: false });
    };
    timeScale.subscribeVisibleTimeRangeChange(() => handleMaybeLoad(null, { fromInteraction: false }));
    if (typeof timeScale.subscribeVisibleLogicalRangeChange === 'function') {
      timeScale.subscribeVisibleLogicalRangeChange(() => handleMaybeLoad(null, { fromInteraction: false }));
    }
    window.requestAnimationFrame(() => {
      if (detailManager && typeof detailManager._triggerLazyLoadCheck === 'function') {
        detailManager._triggerLazyLoadCheck();
      }
    });
  }

  async function fetchLiveQuote(symbol) {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return;
    const endpointBase = apiUrl || '/api/market/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({ quote: '1', symbol: normalized });
    if (langPrefix) {
      params.set('lang', langPrefix);
    }
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
    if (langPrefix) {
      params.set('lang', langPrefix);
    }
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
    if (langPrefix) {
      params.set('lang', langPrefix);
    }
    fetch(`${endpoint}?${params.toString()}`, {
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      credentials: 'same-origin',
    }).catch(() => {});
  }

  function computeDataStatus() {
    const spec = resolveIntervalSpec(detailInterval);
    const ageSeconds = Number.isFinite(lastTradeAgeSeconds) ? lastTradeAgeSeconds : null;
    const intervalSec = detailBarIntervalSec || (spec && spec.seconds) || 60;
    const isHighFreq = Boolean(spec && (spec.unit === 'tick' || spec.unit === 'second'));
    const liveThreshold = isHighFreq ? 15 : Math.max(60, intervalSec * 2);
    const delayedThreshold = isHighFreq ? 60 : Math.max(180, intervalSec * 4);
    let state = 'disconnected';
    if (!detailSymbol) {
      state = 'disconnected';
    } else if (ageSeconds === null) {
      state = chartSocketReady ? 'delayed' : 'disconnected';
    } else if (ageSeconds <= liveThreshold && (isChartRealtimeActive() || chartSocketReady)) {
      state = 'live';
    } else if (ageSeconds <= delayedThreshold) {
      state = 'delayed';
    } else {
      state = chartSocketReady ? 'stale' : 'disconnected';
    }
    return {
      state,
      ageSeconds,
      liveThreshold,
      delayedThreshold,
    };
  }

  function applyStatusChip(chipEl, textEl, metaEl, status) {
    if (!chipEl || !status) return;
    const state = status.state || 'disconnected';
    chipEl.classList.remove('is-live', 'is-delayed', 'is-stale', 'is-disconnected');
    chipEl.classList.add(`is-${state}`);
    const labels = TEXT.detailStatusLabels || {};
    if (textEl) {
      textEl.textContent = labels[state] || labels.disconnected || '—';
    }
    if (metaEl) {
      const ageText = Number.isFinite(status.ageSeconds) ? formatAge(status.ageSeconds) : '';
      metaEl.textContent = ageText ? TEXT.detailStatusMeta(ageText) : '';
    }
  }

  function renderDetailStatusCard() {
    if (!detailStatus) return;
    const status = computeDataStatus();
    const messages = TEXT.detailStatusMessages || {};
    const fallbackMessage = messages[status.state] || '';
    const hasOverride = Boolean(detailStatusOverride);
    const message = detailStatusOverride || fallbackMessage;
    const showCard = hasOverride || status.state !== 'live';
    detailStatus.hidden = !showCard;
    detailStatus.classList.toggle('is-error', Boolean(detailStatusIsError));
    if (detailStatusCardText) {
      detailStatusCardText.textContent = message || '';
    }
    if (detailStatusCardMeta) {
      const ageText = Number.isFinite(status.ageSeconds) ? formatAge(status.ageSeconds) : '';
      const metaParts = [];
      if (ageText) {
        metaParts.push(TEXT.detailStatusMeta(ageText));
      }
      if (status.state === 'disconnected') {
        if (chartVisible && chartSocketRetryAttempts > 0 && TEXT.detailStatusReconnectAttempt) {
          const attemptText = TEXT.detailStatusReconnectAttempt(chartSocketRetryAttempts);
          if (attemptText) metaParts.push(attemptText);
        }
        if (chartSocketLastClose && TEXT.detailStatusCloseCode) {
          const codeVal = Number.isFinite(chartSocketLastClose.code) ? chartSocketLastClose.code : null;
          const rawReason = chartSocketLastClose.reason ? String(chartSocketLastClose.reason).trim() : '';
          const reasonVal = rawReason.length > 80 ? `${rawReason.slice(0, 80)}…` : rawReason;
          const closeText = TEXT.detailStatusCloseCode(codeVal, reasonVal);
          if (closeText) metaParts.push(closeText);
        }
      }
      detailStatusCardMeta.textContent = metaParts.join(' · ');
    }
    const showActions =
      status.state === 'disconnected' ||
      detailStatusOverride === TEXT.detailLiveWaiting ||
      detailStatusOverride === TEXT.detailError;
    if (detailStatusCardCta) {
      detailStatusCardCta.hidden = !showActions;
      detailStatusCardCta.textContent = TEXT.detailStatusMonitor || detailStatusCardCta.textContent;
    }
    if (detailStatusReconnect) {
      detailStatusReconnect.hidden = !showActions;
      detailStatusReconnect.textContent = TEXT.detailStatusReconnect || detailStatusReconnect.textContent;
    }
  }

  function updateDataStatusUI() {
    const status = computeDataStatus();
    applyStatusChip(headerStatusChip, headerStatusText, null, status);
    applyStatusChip(chartStatusChip, chartStatusText, chartStatusMeta, status);
    if (headerStatusCta) {
      headerStatusCta.hidden = status.state !== 'disconnected';
    }
    renderDetailStatusCard();
  }

  function setDetailStatus(message, isError) {
    detailStatusOverride = message || '';
    detailStatusIsError = Boolean(isError);
    updateDataStatusUI();
  }

  function setAiLoading(isLoading) {
    if (aiSkeleton) {
      aiSkeleton.hidden = !isLoading;
    }
    if (aiSummaryBody) {
      aiSummaryBody.hidden = isLoading;
    }
    if (aiSummary) {
      aiSummary.classList.toggle('is-loading', Boolean(isLoading));
    }
  }

  function setAiUpdated(value) {
    if (!aiUpdated) return;
    const display = value ? formatDisplayTime(value) : '';
    aiUpdated.textContent = display || '—';
  }

  function updateAiRefreshButton() {
    if (!aiRefreshBtn) return;
    const now = Date.now();
    if (!aiRefreshCooldownUntil || aiRefreshCooldownUntil <= now) {
      aiRefreshBtn.disabled = false;
      aiRefreshBtn.textContent = TEXT.aiRefresh || aiRefreshBtn.textContent;
      return;
    }
    const remaining = Math.max(1, Math.ceil((aiRefreshCooldownUntil - now) / 1000));
    aiRefreshBtn.disabled = true;
    aiRefreshBtn.textContent = TEXT.aiRefreshCooldown ? TEXT.aiRefreshCooldown(remaining) : `Refresh in ${remaining}s`;
  }

  function startAiRefreshCooldown() {
    aiRefreshCooldownUntil = Date.now() + AI_REFRESH_COOLDOWN_MS;
    updateAiRefreshButton();
    if (aiRefreshCooldownTimer) {
      clearInterval(aiRefreshCooldownTimer);
    }
    aiRefreshCooldownTimer = setInterval(() => {
      if (!aiRefreshCooldownUntil || aiRefreshCooldownUntil <= Date.now()) {
        clearInterval(aiRefreshCooldownTimer);
        aiRefreshCooldownTimer = null;
        aiRefreshCooldownUntil = 0;
        updateAiRefreshButton();
        return;
      }
      updateAiRefreshButton();
    }, 500);
  }

  function updateContextTabIndicator() {
    if (!contextTabs.length) return;
    const activeTab = contextTabs.find((btn) => btn.classList.contains('is-active')) || contextTabs[0];
    if (!activeTab) return;
    const tabWrap = activeTab.parentElement;
    if (!tabWrap) return;
    const wrapRect = tabWrap.getBoundingClientRect();
    const tabRect = activeTab.getBoundingClientRect();
    const left = tabRect.left - wrapRect.left + tabWrap.scrollLeft;
    const width = tabRect.width;
    tabWrap.style.setProperty('--tab-left', `${left}px`);
    tabWrap.style.setProperty('--tab-width', `${width}px`);
  }

  function setContextTab(tabKey, { persist = true, scrollIntoView = false } = {}) {
    if (!contextTabs.length || !contextPanels.length) return;
    const resolvedTab =
      contextTabs.find((btn) => btn.dataset.tab === tabKey)?.dataset.tab || contextTabs[0].dataset.tab;
    activeContextTab = resolvedTab;
    if (persist) {
      savePreference(PREF_CONTEXT_TAB_KEY, resolvedTab);
    }
    contextTabs.forEach((btn) => {
      const isActive = btn.dataset.tab === resolvedTab;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    contextPanels.forEach((panel) => {
      const isActive = panel.dataset.tab === resolvedTab;
      panel.classList.toggle('is-active', isActive);
      panel.hidden = !isActive;
    });
    requestAnimationFrame(updateContextTabIndicator);
    if (scrollIntoView && contextCard && contextCard.scrollIntoView) {
      contextCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function setWorkbenchPanel(panelKey) {
    if (!workbenchPanels.length) return;
    const resolved =
      workbenchPanels.find((panel) => panel.dataset.panel === panelKey)?.dataset.panel ||
      workbenchPanels[0].dataset.panel;
    activeWorkbenchPanel = resolved;
    workbenchPanels.forEach((panel) => {
      const isActive = panel.dataset.panel === resolved;
      panel.classList.toggle('is-active', isActive);
      panel.hidden = !isActive;
    });
    if (paneTabs.length) {
      paneTabs.forEach((tab) => {
        const tabView = tab.dataset.view || 'detail';
        const tabPanel = tab.dataset.panel || '';
        const isActive = tabView === currentView && (tabView !== 'detail' || tabPanel === resolved);
        tab.classList.toggle('is-active', isActive);
        tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
        tab.setAttribute('tabindex', isActive ? '0' : '-1');
      });
    }
    updateNewsLoadMoreState();
    if (resolved === 'news') {
      maybeLoadMoreNews();
    }
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
      marketPage.classList.toggle('has-detail', Boolean(detailSymbol));
    }
    if (currentView === 'detail') {
      requestAnimationFrame(() => {
        updateContextTabIndicator();
        setTimeout(updateContextTabIndicator, 0);
      });
    }

    if (paneTabs.length) {
      paneTabs.forEach((tab) => {
        const tabView = tab.dataset.view || 'detail';
        const tabPanel = tab.dataset.panel || '';
        const matches = tabView === currentView && (tabView !== 'detail' || tabPanel === activeWorkbenchPanel);
        tab.classList.toggle('is-active', matches);
        tab.setAttribute('aria-selected', matches ? 'true' : 'false');
        tab.setAttribute('tabindex', matches ? '0' : '-1');
      });
    }
    updateNewsLoadMoreState();

    if (currentView === 'detail' && workbenchPanels.length) {
      setWorkbenchPanel(activeWorkbenchPanel || 'overview');
    }

    if (chartVisible) {
      resizeDetailChart();
      connectChartSocket();
      if (detailSymbol) {
        setChartSocketSymbol(detailSymbol);
      }
      setWaveOverlayEnabled(waveOverlayEnabled);
      if (!waveMetaLoaded) {
        refreshWaveMeta();
      }
      const expectedChartKey =
        detailSymbol && detailRange && detailInterval ? `${detailSymbol}|${detailRange}|${detailInterval}` : '';
      if (expectedChartKey && detailLastChartKey === expectedChartKey) {
        scheduleWaveAutoAnalyze();
      }
    }
    if (isListVisible()) {
      scheduleAutoRefresh();
    }
    if (!chartVisible) {
      if (waveAnalyzeTimer) {
        window.clearTimeout(waveAnalyzeTimer);
        waveAnalyzeTimer = null;
      }
      waveAnalyzeQueuedTask = null;
      clearLiveWait();
      stopLiveQuotePolling();
      stopChartPolling();
      stopTickFallbackPolling();
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
      detailCompanyName = displayName || '';
    }
    if (detailMetaEl) {
      const profile = payload && payload.profile ? payload.profile : {};
      const exchange = profile && profile.exchange ? profile.exchange : '';
      const sector = profile && profile.sector ? profile.sector : '';
      const industry = profile && profile.industry ? profile.industry : '';
      const metaParts = [exchange, sector, industry].filter(Boolean);
      detailMetaEl.textContent = metaParts.join(' · ');
    }
    const hasBars = Array.isArray(bars) && bars.length > 0;
    setDetailPriceValue(hasBars ? bars[bars.length - 1].close : null);
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
  }

  function updateDetailTimes(payload, options = {}) {
    const { updateSubtitle = true, intervalSpec = null, bars = null } = options;
    const raw = payload && payload.generated_at ? payload.generated_at : '';
    const display = raw ? formatDisplayTime(raw) : '';
    if (detailUpdated) {
      detailUpdated.textContent = display ? `${TEXT.updatedLabel} ${display}` : '';
    }
    if (detailUpdatedCompact) {
      detailUpdatedCompact.textContent = display || '—';
      if (detailUpdatedCompact.tagName === 'TIME') {
        detailUpdatedCompact.setAttribute('datetime', raw || '');
      }
    }
    if (detailSubtitle && updateSubtitle) {
      let dataTs = null;
      if (payload) {
        dataTs = normalizeEpochSeconds(payload.latest_trade_ts);
      }
      const resolveLatestFromBars = (list) => {
        if (!Array.isArray(list) || !list.length) return null;
        let latest = null;
        list.forEach((bar) => {
          const ts = normalizeEpochSeconds(bar && bar.time);
          if (!Number.isFinite(ts)) return;
          if (!Number.isFinite(latest) || ts > latest) {
            latest = ts;
          }
        });
        return latest;
      };
      if (!Number.isFinite(dataTs)) {
        const managed = detailManager && Array.isArray(detailManager.ohlcData) ? detailManager.ohlcData : null;
        dataTs = resolveLatestFromBars(managed);
      }
      if (!Number.isFinite(dataTs)) {
        dataTs = resolveLatestFromBars(bars);
      }
      if (Number.isFinite(dataTs)) {
        const showSeconds = Boolean(intervalSpec && (intervalSpec.unit === 'tick' || intervalSpec.unit === 'second'));
        const dataDisplay = formatAxisTime(dataTs, {
          timezoneMode,
          showSeconds,
          includeDate: true,
          fullDate: true,
        });
        detailSubtitle.textContent = langPrefix === 'zh' ? `数据时间：${dataDisplay}` : `Data time: ${dataDisplay}`;
      } else if (display) {
        detailSubtitle.textContent = langPrefix === 'zh' ? `更新时间：${display}` : `Updated: ${display}`;
      } else {
        detailSubtitle.textContent = '';
      }
    }
  }

  function setDetailLatencyText(text) {
    if (!detailLatency) return;
    detailLatency.textContent = text || '';
  }

  function setDetailLazy(isLoading) {
    if (!detailLazy) return;
    if (isLoading) {
      detailLazy.textContent = TEXT.detailLazyLoading || TEXT.loadingMore || 'Loading…';
      detailLazy.hidden = false;
      return;
    }
    detailLazy.hidden = true;
  }

  function updateDetailLatencyFromTs(ts, options = {}) {
    const numeric = normalizeEpochSeconds(ts);
    if (!Number.isFinite(numeric)) {
      if (detailLatency) {
        setDetailLatencyText('');
      }
      lastTradeAgeSeconds = null;
      lastTradeStaleLabel = '';
      updateDataStatusUI();
      return;
    }
    const nowSec = Date.now() / 1000;
    const ageSeconds = Math.max(0, nowSec - numeric);
    const latencyText = formatLatencyValue(ageSeconds * 1000);
    if (detailLatency) {
      setDetailLatencyText(latencyText ? `${TEXT.latencyLabel} ${latencyText}` : '');
    }
    lastTradeAgeSeconds = ageSeconds;
    const intervalSpec = options.intervalSpec || (detailManager ? detailManager.intervalSpec : null);
    if (intervalSpec && (intervalSpec.unit === 'tick' || intervalSpec.unit === 'second')) {
      const threshold = intervalSpec.unit === 'tick' ? 15 : 60;
      if (ageSeconds > threshold) {
        lastTradeStaleLabel = TEXT.detailStale(formatAge(ageSeconds));
      } else {
        lastTradeStaleLabel = '';
      }
    } else {
      lastTradeStaleLabel = '';
    }
    updateDataStatusUI();
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
        indicatorSecondaryContainer: detailIndicatorSecondaryEl,
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
        detailManager.setAnalysisOverlayVisible(waveOverlayEnabled);
        bindChartLazyLoad();
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

  function normalizeMissingReasons(value) {
    if (!value || typeof value !== 'object') return {};
    const normalized = {};
    Object.keys(value).forEach((key) => {
      const field = (key || '').toString().trim();
      const code = (value[key] || '').toString().trim();
      if (!field || !code) return;
      normalized[field] = code;
    });
    return normalized;
  }

  function getMissingReasonText(reasonCode) {
    const code = (reasonCode || '').toString().trim();
    const reasonMap = TEXT.missingReasons || {};
    const fallback = reasonMap.source_not_provided || { short: '--', full: '' };
    return reasonMap[code] || fallback;
  }

  function clearMissingReasonState(el) {
    if (!el) return;
    el.classList.remove('is-missing-reason');
    if (el.hasAttribute('title')) {
      el.removeAttribute('title');
    }
  }

  function renderMissingReasonValue(el, reasonCode) {
    if (!el) return;
    const reason = getMissingReasonText(reasonCode);
    el.textContent = reason.short || '--';
    el.classList.add('is-missing-reason');
    if (reason.full) {
      el.setAttribute('title', reason.full);
    } else if (el.hasAttribute('title')) {
      el.removeAttribute('title');
    }
  }

  function setTextOrReason(el, value, reasonCode) {
    if (!el) return;
    const text = value === null || value === undefined ? '' : String(value).trim();
    if (text) {
      clearMissingReasonState(el);
      el.textContent = text;
      return;
    }
    renderMissingReasonValue(el, reasonCode || 'source_not_provided');
  }

  function renderProfilePanel(profile, missingReasonMap = null) {
    if (!profileSummary) return;
    const meta = profile && typeof profile === 'object' ? profile : {};
    const missingReasons = normalizeMissingReasons(missingReasonMap);
    const rawSummary = meta.summary || meta.name || '';
    const summary =
      rawSummary && rawSummary.length > 180 ? `${rawSummary.slice(0, 180)}…` : rawSummary;
    profileSummary.textContent = summary || TEXT.profileEmpty;
    if (profileSector) {
      setTextOrReason(profileSector, meta.sector, missingReasons.sector);
    }
    if (profileIndustry) {
      setTextOrReason(profileIndustry, meta.industry, missingReasons.industry);
    }
    if (profileCeo) {
      setTextOrReason(profileCeo, meta.ceo || meta.chief_executive, missingReasons.ceo);
    }
    if (profileHq) {
      const hq =
        meta.headquarters ||
        meta.hq ||
        meta.location ||
        [meta.city, meta.state, meta.country].filter(Boolean).join(', ');
      setTextOrReason(profileHq, hq, missingReasons.hq);
    }
    if (profileLogo) {
      const label = meta.shortName || meta.name || detailSymbol || '';
      const initials = label
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map((part) => part[0])
        .join('')
        .toUpperCase();
      profileLogo.textContent = initials || (detailSymbol || '—');
      profileLogo.classList.remove('has-logo');
      profileLogo.style.backgroundImage = '';
      const symbolForLogo = (meta.symbol || detailSymbol || '').toString().trim();
      const logoUrl = getSymbolLogoUrl(symbolForLogo);
      if (logoUrl) {
        const logoKey = `${symbolForLogo}|${logoUrl}`;
        profileLogo.dataset.logoKey = logoKey;
        const img = new Image();
        img.referrerPolicy = 'no-referrer';
        img.onload = () => {
          if (profileLogo.dataset.logoKey !== logoKey) return;
          profileLogo.style.backgroundImage = `url("${logoUrl}")`;
          profileLogo.classList.add('has-logo');
        };
        img.onerror = () => {
          if (profileLogo.dataset.logoKey !== logoKey) return;
          profileLogo.classList.remove('has-logo');
          profileLogo.style.backgroundImage = '';
        };
        img.src = logoUrl;
      }
    }
    const hasData =
      Boolean(summary && summary !== TEXT.profileEmpty) ||
      Boolean(meta.sector || meta.industry || meta.ceo || meta.headquarters || meta.hq || meta.location) ||
      Boolean(missingReasons.sector || missingReasons.industry || missingReasons.ceo || missingReasons.hq);
    const fundamentalsMissing = !(meta.sector || meta.industry);
    if (profileNote) {
      profileNote.textContent = TEXT.profileFundamentalNote || '';
      profileNote.hidden = !fundamentalsMissing || !hasData;
    }
    if (profileOverview) {
      profileOverview.hidden = !hasData;
    }
    if (profileSkeleton) {
      profileSkeleton.hidden = hasData;
    }
  }

  function formatPriceValue(value) {
    return formatPrice4(value);
  }

  function setStatValue(el, value, formatter) {
    if (!el) return false;
    clearMissingReasonState(el);
    if (value === null || value === undefined) {
      el.textContent = '—';
      return false;
    }
    if (typeof value === 'number' && !Number.isFinite(value)) {
      el.textContent = '—';
      return false;
    }
    const text = formatter ? formatter(value) : value;
    if (text === null || text === undefined || text === '' || text === '--') {
      el.textContent = '—';
      return false;
    }
    el.textContent = text;
    return true;
  }

  function setStatValueWithReason(el, value, formatter, reasonCode) {
    if (!el) return false;
    if (value === null || value === undefined) {
      renderMissingReasonValue(el, reasonCode || 'source_not_provided');
      return false;
    }
    if (typeof value === 'number' && !Number.isFinite(value)) {
      renderMissingReasonValue(el, reasonCode || 'source_not_provided');
      return false;
    }
    const text = formatter ? formatter(value) : value;
    if (text === null || text === undefined || text === '' || text === '--') {
      renderMissingReasonValue(el, reasonCode || 'source_not_provided');
      return false;
    }
    clearMissingReasonState(el);
    el.textContent = text;
    return true;
  }

  function computeAtr(bars, period) {
    if (!Array.isArray(bars) || bars.length < 2) return null;
    const trs = [];
    for (let i = 1; i < bars.length; i += 1) {
      const prev = bars[i - 1];
      const bar = bars[i];
      if (!bar || !prev) continue;
      const high = Number.parseFloat(bar.high);
      const low = Number.parseFloat(bar.low);
      const prevClose = Number.parseFloat(prev.close);
      if (!Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(prevClose)) continue;
      const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
      if (Number.isFinite(tr)) {
        trs.push(tr);
      }
    }
    if (!trs.length) return null;
    const slice = trs.slice(-Math.max(1, period));
    const sum = slice.reduce((acc, value) => acc + value, 0);
    return slice.length ? sum / slice.length : null;
  }

  function computeAvgVolume(bars, period) {
    if (!Array.isArray(bars) || !bars.length) return null;
    const volumes = bars
      .map((bar) => (bar ? Number.parseFloat(bar.volume) : NaN))
      .filter((value) => Number.isFinite(value));
    if (!volumes.length) return null;
    const slice = volumes.slice(-Math.max(1, period));
    const sum = slice.reduce((acc, value) => acc + value, 0);
    return slice.length ? sum / slice.length : null;
  }

  function computeVwap(bars) {
    if (!Array.isArray(bars) || !bars.length) return null;
    let pvSum = 0;
    let volSum = 0;
    bars.forEach((bar) => {
      if (!bar) return;
      const volume = Number.parseFloat(bar.volume);
      const close = Number.parseFloat(bar.close);
      if (!Number.isFinite(volume) || !Number.isFinite(close) || volume <= 0) return;
      pvSum += close * volume;
      volSum += volume;
    });
    if (!volSum) return null;
    return pvSum / volSum;
  }

  function renderKeyStats(payload, bars) {
    if (!keyStatsCard) return;
    if (keyStatsSkeleton) {
      keyStatsSkeleton.hidden = true;
    }
    if (keyStatsError) {
      keyStatsError.hidden = true;
    }
    const safeBars = Array.isArray(bars) ? bars.filter((bar) => bar && Number.isFinite(bar.close)) : [];
    const profile = payload && typeof payload === 'object' && payload.profile && typeof payload.profile === 'object'
      ? payload.profile
      : {};
    const profileMissingReasons =
      payload && typeof payload === 'object' ? normalizeMissingReasons(payload.profile_missing_reasons) : {};
    const serverStats = payload && typeof payload === 'object' ? payload.key_stats : null;
    const serverHigh52w = serverStats ? Number.parseFloat(serverStats.high_52w) : NaN;
    const serverLow52w = serverStats ? Number.parseFloat(serverStats.low_52w) : NaN;
    const hasServer52w = Number.isFinite(serverHigh52w) && Number.isFinite(serverLow52w);
    let computedHigh52w = null;
    let computedLow52w = null;
    let hasValue = false;
    if (safeBars.length) {
      const first = safeBars[0];
      const last = safeBars[safeBars.length - 1];
      const prev = safeBars.length > 1 ? safeBars[safeBars.length - 2] : null;
      const open = Number.isFinite(first.open) ? first.open : first.close;
      const high = Math.max(...safeBars.map((bar) => Number(bar.high)).filter((v) => Number.isFinite(v)));
      const low = Math.min(...safeBars.map((bar) => Number(bar.low)).filter((v) => Number.isFinite(v)));
      const prevClose = prev && Number.isFinite(prev.close) ? prev.close : null;
      const volume = safeBars.reduce((acc, bar) => acc + (Number.isFinite(bar.volume) ? Number(bar.volume) : 0), 0);
      const vwap = computeVwap(safeBars);
      const atr = computeAtr(safeBars, 14);
      const avgVol = computeAvgVolume(safeBars, 20);
      const timeSpan = safeBars.length > 1 ? normalizeEpochSeconds(last.time) - normalizeEpochSeconds(first.time) : 0;
      const has52w = Number.isFinite(timeSpan) && timeSpan >= 200 * 86400;
      if (!hasServer52w && has52w) {
        computedHigh52w = Number.isFinite(high) ? high : null;
        computedLow52w = Number.isFinite(low) ? low : null;
      }

      hasValue = setStatValue(statOpen, open, formatPriceValue) || hasValue;
      hasValue = setStatValue(statHigh, Number.isFinite(high) ? high : null, formatPriceValue) || hasValue;
      hasValue = setStatValue(statLow, Number.isFinite(low) ? low : null, formatPriceValue) || hasValue;
      hasValue = setStatValue(statPrevClose, prevClose, formatPriceValue) || hasValue;
      hasValue = setStatValue(statVolume, volume, formatCompactNumber) || hasValue;
      hasValue = setStatValue(statVwap, vwap, formatPriceValue) || hasValue;
      hasValue = setStatValue(statAtr, atr, formatPriceValue) || hasValue;
      hasValue = setStatValue(statAvgVol, avgVol, formatCompactNumber) || hasValue;

      setStatValue(quickOpenEl, open, formatPriceValue);
      setStatValue(quickHighEl, Number.isFinite(high) ? high : null, formatPriceValue);
      setStatValue(quickLowEl, Number.isFinite(low) ? low : null, formatPriceValue);
      setStatValue(quickPrevEl, prevClose, formatPriceValue);
      setStatValue(quickVolumeEl, volume, formatCompactNumber);
    } else {
      setStatValue(statOpen, null);
      setStatValue(statHigh, null);
      setStatValue(statLow, null);
      setStatValue(statPrevClose, null);
      setStatValue(statVolume, null);
      setStatValue(statVwap, null);
      setStatValue(statAtr, null);
      setStatValue(statAvgVol, null);
      setStatValue(quickOpenEl, null);
      setStatValue(quickHighEl, null);
      setStatValue(quickLowEl, null);
      setStatValue(quickPrevEl, null);
      setStatValue(quickVolumeEl, null);
    }

    const finalHigh52w = hasServer52w ? serverHigh52w : computedHigh52w;
    const finalLow52w = hasServer52w ? serverLow52w : computedLow52w;
    if (finalHigh52w !== null && finalLow52w !== null) {
      hasValue = setStatValue(stat52w, `${formatPriceValue(finalHigh52w)} / ${formatPriceValue(finalLow52w)}`) || hasValue;
      setStatValue(quick52wEl, `${formatPriceValue(finalHigh52w)} / ${formatPriceValue(finalLow52w)}`);
    } else {
      setStatValue(stat52w, null);
      setStatValue(quick52wEl, null);
    }

    const marketCap = profile && profile.market_cap ? profile.market_cap : null;
    const sector = profile && profile.sector ? profile.sector : null;
    const industry = profile && profile.industry ? profile.industry : null;
    hasValue =
      setStatValueWithReason(statMarketCap, marketCap, formatCompactCurrency, profileMissingReasons.market_cap) ||
      hasValue;
    hasValue = setStatValueWithReason(statSector, sector, null, profileMissingReasons.sector) || hasValue;
    hasValue = setStatValueWithReason(statIndustry, industry, null, profileMissingReasons.industry) || hasValue;
    if (keyStatsNote) {
      const fundamentalsMissing = !(marketCap || sector || industry);
      keyStatsNote.textContent = TEXT.keyStatsFundamentalNote || '';
      keyStatsNote.hidden = !fundamentalsMissing;
    }

    if (keyStatsEmpty) {
      keyStatsEmpty.textContent = TEXT.keyStatsEmpty || '';
      keyStatsEmpty.hidden = hasValue;
    }
    keyStatsCard.classList.toggle('is-empty', !hasValue);
  }

  const POSITIVE_SENTIMENT = [
    'beat',
    'beats',
    'surge',
    'surged',
    'upgrade',
    'upgraded',
    'record',
    'growth',
    'profit',
    'profits',
    'rally',
    'raises',
    'raised',
    'strong',
    'outperform',
    'buyback',
    'bullish',
    'expands',
    'expansion',
    'accelerate',
    'accelerates',
    '上调',
    '增长',
    '盈利',
    '创新高',
    '强劲',
    '利好',
    '回购',
    '看多',
  ];

  const NEGATIVE_SENTIMENT = [
    'miss',
    'missed',
    'downgrade',
    'downgraded',
    'falls',
    'fall',
    'drop',
    'drops',
    'lawsuit',
    'cuts',
    'cut',
    'weak',
    'warns',
    'warning',
    'bearish',
    'decline',
    'plunge',
    'slump',
    '下调',
    '下跌',
    '亏损',
    '利空',
    '警告',
    '看空',
  ];

  function inferSentiment(text) {
    const lowered = (text || '').toString().toLowerCase();
    if (!lowered) return 'neutral';
    let score = 0;
    POSITIVE_SENTIMENT.forEach((word) => {
      if (lowered.includes(word)) score += 1;
    });
    NEGATIVE_SENTIMENT.forEach((word) => {
      if (lowered.includes(word)) score -= 1;
    });
    if (score > 0) return 'bullish';
    if (score < 0) return 'bearish';
    return 'neutral';
  }

  function isKnownSymbol(symbol) {
    if (!symbol) return false;
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return false;
    if (normalized === detailSymbol) return true;
    if (suggestionPool.includes(normalized)) return true;
    if (watchPool.includes(normalized)) return true;
    if (recentPool.includes(normalized)) return true;
    return false;
  }

  function extractTickers(text) {
    const matches = (text || '').toString().match(/\$?[A-Z]{2,5}\b/g) || [];
    const unique = [];
    const seen = new Set();
    matches.forEach((raw) => {
      const symbol = normalizeSymbol(raw.replace('$', ''));
      if (!symbol || seen.has(symbol)) return;
      if (raw.startsWith('$') || isKnownSymbol(symbol)) {
        seen.add(symbol);
        unique.push(symbol);
      }
    });
    return unique;
  }

  function collectNewsRelatedSymbols(entry, text) {
    const unique = [];
    const seen = new Set();
    const pushSymbol = (raw) => {
      const symbol = normalizeSymbol(raw);
      if (!symbol || seen.has(symbol)) return;
      seen.add(symbol);
      unique.push(symbol);
    };
    if (entry && Array.isArray(entry.related_symbols)) {
      entry.related_symbols.forEach((raw) => pushSymbol(raw));
    }
    extractTickers(text).forEach((symbol) => pushSymbol(symbol));
    return unique;
  }

  function appendTextWithEntities(el, text) {
    if (!el) return;
    el.textContent = '';
    const tokens = (text || '').toString().split(/(\$?[A-Z]{2,5}\b)/g);
    tokens.forEach((token) => {
      const normalized = normalizeSymbol(token.replace('$', ''));
      if (token && (token.startsWith('$') || isKnownSymbol(normalized))) {
        const badge = document.createElement('span');
        badge.className = 'entity-badge entity-link';
        badge.dataset.symbol = normalized;
        badge.textContent = normalized;
        el.appendChild(badge);
      } else {
        el.appendChild(document.createTextNode(token));
      }
    });
  }

  function bindEntityLinks(container) {
    if (!container) return;
    container.querySelectorAll('.entity-link').forEach((node) => {
      node.addEventListener('click', () => {
        const symbol = node.dataset.symbol;
        if (!symbol) return;
        openDetailPanel(symbol);
      });
    });
  }

  function buildNewsItemKey(entry) {
    if (!entry || typeof entry !== 'object') return '';
    const url = (entry.url || '').toString().trim().toLowerCase();
    if (url) return `url:${url}`;
    const title = (entry.title || '').toString().trim().toLowerCase();
    const summary = (entry.summary || '').toString().trim().toLowerCase();
    const time = normalizeEpochSeconds(entry.time);
    if (title && Number.isFinite(time)) return `title:${title}|${Math.floor(time)}`;
    if (title) return `title:${title}`;
    if (summary) return `summary:${summary.slice(0, 120)}`;
    return '';
  }

  function resolveNewsSentimentKey(entry, index) {
    const stableKey = buildNewsItemKey(entry);
    if (stableKey) return stableKey;
    const title = (entry && entry.title ? entry.title : '').toString().trim().toLowerCase();
    const time = normalizeEpochSeconds(entry && entry.time);
    if (title) {
      return Number.isFinite(time) ? `fallback:${title}|${Math.floor(time)}` : `fallback:${title}|${index || 0}`;
    }
    return `fallback:index|${index || 0}`;
  }

  function mergeNewsItems(existingItems, incomingItems) {
    const merged = [];
    const seen = new Set();
    const pushUnique = (entry) => {
      if (!entry || typeof entry !== 'object') return;
      const key = buildNewsItemKey(entry);
      if (!key) return;
      if (seen.has(key)) return;
      seen.add(key);
      merged.push(entry);
    };
    (Array.isArray(existingItems) ? existingItems : []).forEach(pushUnique);
    (Array.isArray(incomingItems) ? incomingItems : []).forEach(pushUnique);
    merged.sort((left, right) => {
      const leftTs = normalizeEpochSeconds(left && left.time);
      const rightTs = normalizeEpochSeconds(right && right.time);
      const safeLeft = Number.isFinite(leftTs) ? leftTs : 0;
      const safeRight = Number.isFinite(rightTs) ? rightTs : 0;
      return safeRight - safeLeft;
    });
    return merged;
  }

  function normalizeNewsMeta(meta, fallbackCount) {
    const source = meta && typeof meta === 'object' ? meta : {};
    const offsetRaw = Number(source.offset);
    const limitRaw = Number(source.limit);
    const countRaw = Number(source.count);
    const nextOffsetRaw = Number(source.next_offset);
    const offset = Number.isFinite(offsetRaw) ? Math.max(0, offsetRaw) : 0;
    const limit = Number.isFinite(limitRaw) ? Math.max(1, limitRaw) : newsPageSize || NEWS_PAGE_SIZE;
    const count = Number.isFinite(countRaw) ? Math.max(0, countRaw) : Math.max(0, fallbackCount || 0);
    const defaultNext = offset + count;
    const nextOffset = Number.isFinite(nextOffsetRaw) ? Math.max(0, nextOffsetRaw) : defaultNext;
    const hasMore = Boolean(source.has_more) || nextOffset > defaultNext;
    return {
      offset,
      limit,
      count,
      nextOffset,
      hasMore,
    };
  }

  function resetNewsPagingState(symbol) {
    const normalizedSymbol = normalizeSymbol(symbol || '');
    newsListSymbol = normalizedSymbol;
    if (newsSentimentSymbol !== normalizedSymbol) {
      newsSentimentSymbol = normalizedSymbol;
      newsSentimentToken += 1;
      newsSentimentInFlight = false;
      newsSentimentCache = new Map();
      newsSentimentQueue = [];
      newsSentimentQueuedKeys = new Set();
    }
    newsPageSize = NEWS_PAGE_SIZE;
    newsNextOffset = 0;
    newsHasMore = false;
    newsLoadingMore = false;
    newsLoadMoreError = '';
    newsScrollArmed = false;
    if (newsScrollRoot) {
      newsScrollRoot.scrollTop = 0;
    }
  }

  function updateNewsLoadMoreState() {
    if (!newsLoadMore) return;
    const shouldShowHint = currentView === 'detail' && activeWorkbenchPanel === 'news';
    const blockedByAi =
      shouldShowHint &&
      newsHasMore &&
      (newsSentimentInFlight || (Array.isArray(newsSentimentQueue) && newsSentimentQueue.length > 0));
    if (!shouldShowHint) {
      newsLoadMore.hidden = true;
      newsLoadMore.classList.remove('is-error');
      newsLoadMore.textContent = '';
      return;
    }
    if (newsLoadingMore) {
      newsLoadMore.hidden = false;
      newsLoadMore.classList.remove('is-error');
      newsLoadMore.textContent = TEXT.newsLoadingMore || TEXT.loadingMore || '';
      return;
    }
    if (newsLoadMoreError) {
      newsLoadMore.hidden = false;
      newsLoadMore.classList.add('is-error');
      newsLoadMore.textContent = newsLoadMoreError;
      return;
    }
    if (blockedByAi) {
      newsLoadMore.hidden = false;
      newsLoadMore.classList.remove('is-error');
      newsLoadMore.textContent = TEXT.newsWaitAiBeforeLoadMore || TEXT.newsLoadingMore || TEXT.loadingMore || '';
      return;
    }
    newsLoadMore.hidden = true;
    newsLoadMore.classList.remove('is-error');
    newsLoadMore.textContent = '';
  }

  function applyNewsMeta(meta, fallbackCount) {
    const normalized = normalizeNewsMeta(meta, fallbackCount);
    newsPageSize = normalized.limit || NEWS_PAGE_SIZE;
    newsNextOffset = normalized.nextOffset;
    newsHasMore = normalized.hasMore;
    if (normalized.hasMore && normalized.count > 0) {
      newsNextOffset = normalized.offset + normalized.count;
    }
  }

  function applyNewsPayload(payload, { append = false } = {}) {
    const symbol = normalizeSymbol((payload && payload.symbol) || detailSymbol || '');
    if (!append) {
      resetNewsPagingState(symbol);
      const baseItems = Array.isArray(payload && payload.news) ? payload.news : [];
      renderNewsPanel(baseItems, { append: false });
      applyNewsMeta(payload && payload.news_meta, baseItems.length);
      newsLoadingMore = false;
      newsLoadMoreError = '';
      updateNewsLoadMoreState();
      return;
    }
    if (symbol && normalizeSymbol(detailSymbol) && symbol !== normalizeSymbol(detailSymbol)) {
      return;
    }
    const incoming = Array.isArray(payload && payload.news) ? payload.news : [];
    renderNewsPanel(incoming, { append: true });
    applyNewsMeta(payload && payload.news_meta, incoming.length);
    newsLoadingMore = false;
    newsLoadMoreError = '';
    updateNewsLoadMoreState();
  }

  function maybeLoadMoreNews() {
    if (!detailSymbol || newsLoadingMore || !newsHasMore) return;
    if (currentView !== 'detail' || activeWorkbenchPanel !== 'news') return;
    if (!newsScrollArmed) return;
    if (newsSentimentInFlight || (Array.isArray(newsSentimentQueue) && newsSentimentQueue.length > 0)) {
      updateNewsLoadMoreState();
      return;
    }
    loadMoreNews();
  }

  function setupNewsObserver() {
    if (!newsSentinel || !window.IntersectionObserver) return;
    if (newsObserver) {
      newsObserver.disconnect();
      newsObserver = null;
    }
    newsObserver = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        maybeLoadMoreNews();
      },
      { root: newsScrollRoot || null, rootMargin: NEWS_LOAD_MORE_OBSERVER_MARGIN },
    );
    newsObserver.observe(newsSentinel);
  }

  async function loadMoreNews() {
    if (!detailSymbol || newsLoadingMore || !newsHasMore) return;
    const symbol = detailSymbol;
    const requestId = (newsRequestSeq += 1);
    newsLoadingMore = true;
    newsLoadMoreError = '';
    updateNewsLoadMoreState();
    const endpointBase = apiUrl || '/api/market/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({
      detail: '1',
      symbol,
      range: detailRange || '1d',
      include_ai: '0',
      include_bars: '0',
      news_only: '1',
      news_offset: String(newsNextOffset || 0),
      news_limit: String(newsPageSize || NEWS_PAGE_SIZE),
    });
    if (langPrefix) {
      params.set('lang', langPrefix);
    }
    try {
      const response = await fetch(`${endpoint}?${params.toString()}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      const parsed = await parseApiResponse(response);
      const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
      if (requestId !== newsRequestSeq || symbol !== detailSymbol) return;
      if (!response.ok) {
        throw new Error(payload.error || payload.message || TEXT.statusError || TEXT.genericError);
      }
      applyNewsPayload(payload, { append: true });
    } catch (_error) {
      if (requestId !== newsRequestSeq || symbol !== detailSymbol) return;
      newsLoadingMore = false;
      newsLoadMoreError = TEXT.newsLoadMoreFailed || TEXT.statusError || TEXT.genericError;
      updateNewsLoadMoreState();
    }
  }

  function isStrongNewsItem(entry) {
    if (!entry || typeof entry !== 'object') return false;
    const title = (entry.title || '').toString();
    const summary = (entry.summary || '').toString();
    const text = `${title} ${summary}`.toLowerCase();
    if (!text) return false;
    const symbol = (detailSymbol || '').toLowerCase();
    if (symbol) {
      const symRegex = new RegExp(`\\b\\$?${symbol}\\b`, 'i');
      if (symRegex.test(text)) return true;
    }
    const name = (detailCompanyName || '').toLowerCase();
    if (name && name.length > 3 && text.includes(name)) return true;
    return false;
  }

  function setNewsFilter(mode) {
    newsFilterMode = mode === 'strong' ? 'strong' : 'all';
    if (newsFilterButtons.length) {
      newsFilterButtons.forEach((btn) => {
        const isActive = btn.dataset.filter === newsFilterMode;
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });
    }
    renderNewsPanel(latestNewsItems);
    maybeLoadMoreNews();
  }

  function renderNewsPanel(items, options = {}) {
    if (!newsList) return;
    const append = Boolean(options && options.append);
    const incomingItems = Array.isArray(items) ? items : [];
    latestNewsItems = append ? mergeNewsItems(latestNewsItems, incomingItems) : incomingItems.slice();
    const filteredItems = newsFilterMode === 'strong' ? latestNewsItems.filter(isStrongNewsItem) : latestNewsItems;
    newsList.innerHTML = '';
    if (!filteredItems.length) {
      const item = document.createElement('li');
      item.textContent = newsFilterMode === 'strong' ? TEXT.newsStrongEmpty || TEXT.newsEmpty : TEXT.newsEmpty;
      newsList.appendChild(item);
      updateNewsLoadMoreState();
      return;
    }
    filteredItems.forEach((entry, index) => {
      const newsKey = resolveNewsSentimentKey(entry, index);
      const title = entry.title || '';
      const url = entry.url || '#';
      const source = entry.source || '';
      const time =
        entry.time !== undefined && entry.time !== null && entry.time !== ''
          ? formatDisplayTime(entry.time)
          : '';
      const snippet = entry.summary || '';
      const item = document.createElement('li');
      item.className = 'news-item';
      item.dataset.newsKey = newsKey;
      const sentimentCell = document.createElement('div');
      sentimentCell.className = 'news-cell news-cell--sentiment';
      const sentimentTag = document.createElement('span');
      sentimentTag.className = 'news-sentiment news-sentiment--pending';
      sentimentTag.dataset.role = 'news-sentiment';
      sentimentTag.dataset.newsKey = newsKey;
      sentimentTag.textContent = TEXT.sentimentLoading || 'AI';
      sentimentCell.appendChild(sentimentTag);
      item.appendChild(sentimentCell);
      const cachedSentiment = newsSentimentCache.get(newsKey);
      if (cachedSentiment) {
        setNewsSentimentTag(sentimentTag, cachedSentiment);
      }

      const titleCell = document.createElement('div');
      titleCell.className = 'news-cell news-cell--title';
      if (title) {
        const link = document.createElement('a');
        link.href = url || '#';
        link.target = '_blank';
        link.rel = 'noopener';
        link.className = 'news-title';
        link.textContent = title;
        titleCell.appendChild(link);
      } else {
        const text = document.createElement('span');
        text.className = 'news-title';
        text.textContent = '—';
        titleCell.appendChild(text);
      }
      if (snippet) {
        const snippetEl = document.createElement('span');
        snippetEl.className = 'news-snippet';
        snippetEl.textContent = snippet.length > 120 ? `${snippet.slice(0, 120)}…` : snippet;
        titleCell.appendChild(snippetEl);
      }
      item.appendChild(titleCell);

      const sourceCell = document.createElement('div');
      sourceCell.className = 'news-cell news-cell--source';
      const sourceWrap = document.createElement('span');
      sourceWrap.className = 'news-source';
      let host = '';
      try {
        host = url ? new URL(url).hostname.replace(/^www\./, '') : '';
      } catch (err) {
        host = '';
      }
      if (host) {
        const icon = document.createElement('img');
        icon.className = 'news-favicon';
        icon.alt = '';
        icon.src = `https://www.google.com/s2/favicons?domain=${host}&sz=32`;
        sourceWrap.appendChild(icon);
      }
      const sourceLabel = document.createElement('strong');
      sourceLabel.textContent = source || host || 'News';
      sourceWrap.appendChild(sourceLabel);
      sourceCell.appendChild(sourceWrap);
      item.appendChild(sourceCell);

      const timeCell = document.createElement('div');
      timeCell.className = 'news-cell news-cell--time';
      const timeEl = document.createElement('span');
      timeEl.className = 'news-time';
      timeEl.textContent = time || '—';
      timeCell.appendChild(timeEl);
      item.appendChild(timeCell);

      const tickers = collectNewsRelatedSymbols(entry, `${title} ${snippet}`);
      const tagsCell = document.createElement('div');
      tagsCell.className = 'news-cell news-cell--tickers';
      if (tickers.length) {
        const tags = document.createElement('div');
        tags.className = 'news-tags';
        tickers.forEach((symbol) => {
          const badge = document.createElement('span');
          badge.className = 'news-badge entity-link';
          badge.dataset.symbol = symbol;
          badge.textContent = symbol;
          tags.appendChild(badge);
        });
        tagsCell.appendChild(tags);
      } else {
        const empty = document.createElement('span');
        empty.className = 'news-empty';
        empty.textContent = '—';
        tagsCell.appendChild(empty);
      }
      item.appendChild(tagsCell);

      newsList.appendChild(item);
    });
    bindEntityLinks(newsList);
    enqueueNewsSentiment(filteredItems);
    updateNewsLoadMoreState();
  }

  function normalizeSentimentLabel(label) {
    const raw = (label || '').toString().toLowerCase();
    if (['bullish', 'positive', 'pos', 'up', '利好', '看多'].some((key) => raw.includes(key))) {
      return 'bullish';
    }
    if (['bearish', 'negative', 'neg', 'down', '利空', '看空'].some((key) => raw.includes(key))) {
      return 'bearish';
    }
    return 'neutral';
  }

  function formatSentimentText(sentiment) {
    if (sentiment === 'bullish') return langPrefix === 'zh' ? '利好' : 'Bullish';
    if (sentiment === 'bearish') return langPrefix === 'zh' ? '利空' : 'Bearish';
    return langPrefix === 'zh' ? '中性' : 'Neutral';
  }

  function setNewsSentimentTag(tag, sentiment) {
    if (!tag) return;
    tag.classList.remove('news-sentiment--pending', 'news-sentiment--bullish', 'news-sentiment--bearish', 'news-sentiment--neutral');
    tag.classList.add(`news-sentiment--${sentiment}`);
    tag.textContent = formatSentimentText(sentiment);
  }

  function setNewsSentimentByKey(key, sentiment) {
    if (!newsList || !key) return;
    const tags = Array.prototype.slice.call(newsList.querySelectorAll('[data-role="news-sentiment"]'));
    tags.forEach((tag) => {
      if ((tag.dataset.newsKey || '') !== key) return;
      setNewsSentimentTag(tag, sentiment);
    });
  }

  function applyNewsSentiments(items, labels) {
    if (!Array.isArray(items) || !items.length) return;
    items.forEach((entry, idx) => {
      const key = entry && typeof entry === 'object' ? entry.key : '';
      if (!key) return;
      const rawLabel = Array.isArray(labels) ? labels[idx] : null;
      const sentiment = normalizeSentimentLabel(rawLabel);
      newsSentimentCache.set(key, sentiment);
      setNewsSentimentByKey(key, sentiment);
    });
  }

  function buildNewsSentimentFallback(items) {
    if (!Array.isArray(items)) return [];
    return items.map((entry) =>
      inferSentiment(
        `${(entry && entry.payload && entry.payload.title) || ''} ${(
          entry &&
          entry.payload &&
          entry.payload.summary
        ) || ''}`
      )
    );
  }

  function requestNewsSentiment(items) {
    if (!Array.isArray(items) || !items.length) {
      newsSentimentInFlight = false;
      updateNewsLoadMoreState();
      return;
    }
    const token = ++newsSentimentToken;
    newsSentimentInFlight = true;
    updateNewsLoadMoreState();
    const fallback = buildNewsSentimentFallback(items);
    const done = (labels) => {
      if (token !== newsSentimentToken) return;
      newsSentimentInFlight = false;
      applyNewsSentiments(items, labels);
      drainNewsSentimentQueue();
      updateNewsLoadMoreState();
    };
    if (!newsSentimentApiUrl) {
      done(fallback);
      return;
    }
    const payload = {
      symbol: detailSymbol || newsListSymbol || '',
      news: items.map((entry) => {
        const item = entry && entry.payload ? entry.payload : {};
        return {
          title: item.title || '',
          summary: item.summary || '',
          source: item.source || '',
          url: item.url || '',
          time: item.time || '',
        };
      }),
    };
    fetch(newsSentimentApiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Requested-With': 'XMLHttpRequest',
        'X-CSRFToken': getCsrfToken(),
      },
      credentials: 'same-origin',
      body: JSON.stringify(payload),
    })
      .then(async (resp) => {
        const parsed = await parseApiResponse(resp);
        const responsePayload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
        if (!resp.ok) {
          throw new Error(responsePayload.message || responsePayload.error || TEXT.statusError || TEXT.genericError);
        }
        return responsePayload;
      })
      .then((data) => {
        const labels = data && Array.isArray(data.labels) ? data.labels : [];
        if (labels.length === items.length) {
          done(labels);
          return;
        }
        done(fallback);
      })
      .catch(() => {
        done(fallback);
      });
  }

  function enqueueNewsSentiment(items) {
    if (!Array.isArray(items) || !items.length) return;
    items.forEach((item, index) => {
      const key = resolveNewsSentimentKey(item, index);
      if (!key) return;
      if (newsSentimentCache.has(key) || newsSentimentQueuedKeys.has(key)) return;
      newsSentimentQueuedKeys.add(key);
      newsSentimentQueue.push({
        key,
        payload: {
          title: item.title || '',
          summary: item.summary || '',
          source: item.source || '',
          url: item.url || '',
          time: item.time || '',
        },
      });
    });
    updateNewsLoadMoreState();
    drainNewsSentimentQueue();
  }

  function drainNewsSentimentQueue() {
    if (newsSentimentInFlight) return;
    if (!Array.isArray(newsSentimentQueue) || !newsSentimentQueue.length) return;
    const batch = newsSentimentQueue.splice(0, NEWS_SENTIMENT_BATCH_SIZE);
    batch.forEach((entry) => {
      if (entry && entry.key) {
        newsSentimentQueuedKeys.delete(entry.key);
      }
    });
    updateNewsLoadMoreState();
    requestNewsSentiment(batch);
  }

  function renderAiPanel(summary, meta, struct) {
    if (!aiSummary) return;
    const text = (summary || '').toString().trim();
    const fallback = TEXT.aiPlaceholder;
    const content = text || fallback;
    const metaObj = meta && typeof meta === 'object' ? meta : {};
    const statusKey = metaObj.status || '';
    let statusMessage = (metaObj.message || '').toString().trim();
    if (!statusMessage && TEXT.aiStatus && statusKey && TEXT.aiStatus[statusKey]) {
      statusMessage = TEXT.aiStatus[statusKey];
    }
    if (aiSummaryStatus) {
      aiSummaryStatus.textContent = statusMessage;
      aiSummaryStatus.hidden = !statusMessage;
    }
    if (statusKey === 'pending') {
      setAiLoading(true);
      return;
    }
    setAiLoading(false);
    const structObj = struct && typeof struct === 'object' ? struct : null;
    const structEvent = structObj && typeof structObj.event === 'string' ? structObj.event.trim() : '';
    const structImpact = structObj && typeof structObj.impact === 'string' ? structObj.impact.trim() : '';
    const structImplication =
      structObj && typeof structObj.implication === 'string' ? structObj.implication.trim() : '';
    const hasStruct = Boolean(structEvent || structImpact || structImplication);
    let eventText = structEvent;
    let impactText = structImpact;
    let implicationText = structImplication;
    if (!hasStruct) {
      const parts = content
        .split(/[\n。.!?]/)
        .map((item) => item.trim())
        .filter(Boolean);
      eventText = parts[0] || content;
      const hasDetailed = statusKey === 'llm' && parts.length >= 2;
      impactText = hasDetailed ? parts[1] || eventText : '—';
      implicationText = hasDetailed ? parts[2] || parts[1] || eventText : '—';
    }
    if (aiSummaryEvent) {
      appendTextWithEntities(aiSummaryEvent, eventText);
    }
    if (aiSummaryImpact) {
      appendTextWithEntities(aiSummaryImpact, impactText);
    }
    if (aiSummaryImplication) {
      appendTextWithEntities(aiSummaryImplication, implicationText);
    }
    bindEntityLinks(aiSummary);
  }

  function markAiPending(payload) {
    if (!payload || typeof payload !== 'object') return payload;
    payload.ai_summary = '';
    payload.ai_summary_struct = null;
    payload.ai_summary_meta = { status: 'pending', message: '', source: 'bailian' };
    return payload;
  }

  function updateInsightPanels(payload) {
    if (!payload || typeof payload !== 'object') return;
    renderProfilePanel(payload.profile, payload.profile_missing_reasons || {});
    applyNewsPayload(payload, { append: false });
    renderAiPanel(payload.ai_summary, payload.ai_summary_meta, payload.ai_summary_struct);
  }

  function applyDetailPayload(symbol, rangeKey, infoPayload, chartPayload, renderChart) {
    const payload = infoPayload || {};
    const chartData = chartPayload || {};
    const bars = Array.isArray(chartData.bars) ? chartData.bars : [];
    updateDetailHero(symbol, payload, bars);
    updateInsightPanels(payload || {});
    if (payload && payload.generated_at) {
      setAiUpdated(payload.generated_at);
    }
    renderKeyStats(payload || {}, bars);
    if (payload && Number.isFinite(payload.buying_power)) {
      tradeBuyingPower = payload.buying_power;
      updateTradeEstimates();
    }
    const intervalKey =
      (chartData.interval && chartData.interval.key) ||
      (chartData.interval_key ? chartData.interval_key : detailInterval);
    const intervalSpec = resolveIntervalSpec(intervalKey);
    detailBarIntervalSec =
      intervalSpec && intervalSpec.unit !== 'tick'
        ? resolveBarIntervalSeconds(intervalKey) || inferBarIntervalSeconds(bars)
        : null;
    updateDetailTimeScale(intervalKey);
    updateDetailTimes(chartData, { intervalSpec, bars });
    setDetailLazy(false);
    if (!renderChart) return;
    if (!ensureDetailChart() || !detailManager) {
      pendingChartRender = { symbol, rangeKey, payload, bars, chartPayload: chartData };
      scheduleChartInit();
      return;
    }
    const chartKey = `${symbol}|${rangeKey}|${intervalKey}`;
    const isNewChartKey = chartKey !== detailLastChartKey;
    if (isNewChartKey) {
      detailUserPanned = false;
    }
    detailManager.setData(bars, { intervalSpec, fitContent: isNewChartKey });
    seedTickFallbackCursor(detailManager.ohlcData);
    detailLastChartKey = chartKey;
    updateDetailTimes(chartData, { intervalSpec, bars: detailManager.ohlcData });
    chartLazyLoading = false;
    chartLazyExhausted = false;
    chartLazyEmptyStreak = 0;
    chartLazyCursor = detailManager.ohlcData[0] ? detailManager.ohlcData[0].time : null;
    if (detailManager && typeof detailManager._triggerLazyLoadCheck === 'function') {
      window.requestAnimationFrame(() => {
        if (detailManager && typeof detailManager._triggerLazyLoadCheck === 'function') {
          detailManager._triggerLazyLoadCheck();
        }
      });
    }
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
    if (chartData) {
      updateDetailLatencyFromTs(chartData.latest_trade_ts || chartData.server_ts, { intervalSpec });
    }
    if (detailTitle) {
      const intervalLabel = getIntervalLabel(intervalKey);
      detailTitle.textContent = intervalLabel ? `${symbol} · ${intervalLabel}` : symbol;
    }
    let statusMessage = '';
    if (chartData.downgrade_message) {
      statusMessage = TEXT.detailDowngraded(chartData.downgrade_message);
    } else if (chartData.window_limited) {
      statusMessage = TEXT.detailWindowLimited;
    }
    if (lastTradeStaleLabel) {
      statusMessage = statusMessage ? `${statusMessage} · ${lastTradeStaleLabel}` : lastTradeStaleLabel;
    }
    setDetailStatus(statusMessage);
    if (chartVisible && detailSymbol === symbol) {
      scheduleLiveWait();
      scheduleWaveAutoAnalyze();
    }
  }

  async function fetchChartBars(symbol, rangeKey, intervalKey, options = {}) {
    const endpointBase = chartApiUrl || '/api/market/chart/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({
      symbol,
      range: rangeKey,
      interval: intervalKey,
    });
    if (options && Number.isFinite(options.endTs)) {
      params.set('end', String(options.endTs));
    }
    if (options && Number.isFinite(options.startTs)) {
      params.set('start', String(options.startTs));
    }
    const response = await fetch(`${endpoint}?${params.toString()}`, {
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      credentials: 'same-origin',
    });
    const payload = await response.json();
    if (!response.ok) {
      const err = new Error(payload.error || TEXT.detailError);
      err.status = response.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  async function requestAiSummary(symbol, rangeKey, requestId) {
    if (!symbol) return;
    setAiLoading(true);
    const endpointBase = apiUrl || '/api/market/';
    const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
    const params = new URLSearchParams({
      detail: '1',
      symbol,
      range: rangeKey,
      ai_only: '1',
      include_ai: '1',
      include_bars: '0',
    });
    if (langPrefix) {
      params.set('lang', langPrefix);
    }
    try {
      const response = await fetch(`${endpoint}?${params.toString()}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || TEXT.detailError);
      }
      if (requestId !== activeDetailRequest) return;
      setAiLoading(false);
      if (payload && payload.generated_at) {
        setAiUpdated(payload.generated_at);
      } else {
        setAiUpdated(Date.now() / 1000);
      }
      if (hasCachedDetailInfo(symbol, rangeKey) && detailInfoCache.payload) {
        detailInfoCache.payload.ai_summary = payload.ai_summary;
        detailInfoCache.payload.ai_summary_meta = payload.ai_summary_meta;
        detailInfoCache.payload.ai_summary_struct = payload.ai_summary_struct;
      }
      renderAiPanel(payload.ai_summary, payload.ai_summary_meta, payload.ai_summary_struct);
    } catch (error) {
      if (requestId !== activeDetailRequest) return;
      setAiLoading(false);
      const fallbackStatus = TEXT.aiStatus && TEXT.aiStatus.error ? TEXT.aiStatus.error : '';
      renderAiPanel('', { status: 'error', message: fallbackStatus }, null);
    }
  }

  async function loadDetailData(symbol, rangeKey, options = {}) {
    if (!symbol) return;
    const requestId = (detailRequestSeq += 1);
    activeDetailRequest = requestId;
    const isStaleRequest = () => requestId !== activeDetailRequest;
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
    setAiLoading(true);
    if (keyStatsSkeleton) {
      keyStatsSkeleton.hidden = false;
    }
    if (keyStatsError) {
      keyStatsError.hidden = true;
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
    if (detailUpdatedCompact) {
      detailUpdatedCompact.textContent = '—';
      if (detailUpdatedCompact.tagName === 'TIME') {
        detailUpdatedCompact.setAttribute('datetime', '');
      }
    }

    let infoPayload = null;
    let chartPayload = null;

    if (allowCache && hasCachedDetailInfo(symbol, rangeKey)) {
      infoPayload = markAiPending(detailInfoCache.payload || null);
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
            include_ai: '0',
            include_bars: '0',
            news_offset: '0',
            news_limit: String(NEWS_PAGE_SIZE),
          });
          if (langPrefix) {
            params.set('lang', langPrefix);
          }
          const response = await fetch(`${endpoint}?${params.toString()}`, {
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
            credentials: 'same-origin',
          });
          const payload = await response.json();
          if (response.status === 429 || payload.rate_limited) {
            if (isStaleRequest()) {
              throw new Error('stale');
            }
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
            const payloadIntervalKey =
              normalizeIntervalKey(
                (payload && payload.interval && payload.interval.key) || payload.downgrade_to || intervalKey
              ) || intervalKey;
            if (payload && payload.downgrade_to) {
              const normalized = normalizeIntervalKey(payload.downgrade_to);
              if (normalized && normalized !== detailInterval) {
                setDetailInterval(normalized, { persist: true, skipReload: true });
              }
            }
            storeDetailChartCache(symbol, rangeKey, payloadIntervalKey, payload.bars || [], payload);
            return payload;
          });

    requestAiSummary(symbol, rangeKey, requestId);

    try {
      const [infoResult, chartResult] = await Promise.allSettled([infoPromise, chartPromise]);
      if (isStaleRequest()) {
        return;
      }
      if (infoResult.status === 'fulfilled') {
        infoPayload = infoResult.value;
      } else if (infoResult.reason && infoResult.reason.message !== 'rate_limited') {
        setDetailStatus(infoResult.reason.message || TEXT.detailError, true);
        updateInsightPanels(markAiPending({}));
        if (keyStatsSkeleton) {
          keyStatsSkeleton.hidden = true;
        }
        if (keyStatsError) {
          keyStatsError.textContent = TEXT.keyStatsError || '';
          keyStatsError.hidden = false;
        }
      }
      if (chartResult.status === 'fulfilled') {
        chartPayload = chartResult.value;
      } else if (chartResult.reason) {
        setDetailStatus(chartResult.reason.message || TEXT.detailError, true);
      }
      if (!chartPayload || !Array.isArray(chartPayload.bars) || !chartPayload.bars.length) {
        setDetailStatus(TEXT.detailEmpty);
        updateInsightPanels(infoPayload || {});
        if (keyStatsSkeleton) {
          keyStatsSkeleton.hidden = true;
        }
        return;
      }
      applyDetailPayload(symbol, rangeKey, infoPayload || {}, chartPayload, renderChart);
      if (chartVisible) {
        setChartSocketSymbol(symbol);
      }
    } catch (error) {
      if (error && (error.message === 'rate_limited' || error.message === 'stale')) return;
      setDetailStatus(TEXT.detailError, true);
      updateInsightPanels({});
    } finally {
      if (!isStaleRequest() && renderChart && detailChartEl) {
        detailChartEl.classList.remove('is-loading');
      }
    }
  }

  function setDetailRange(rangeKey) {
    const normalizedRange = (rangeKey || '').toString().trim().toLowerCase();
    if (normalizedRange !== detailRange) {
      bumpChartContextVersion();
    }
    detailRange = normalizedRange;
    if (detailSymbol) {
      resetWaveAnalysisState({ keepStatus: true });
    }
    detailRangeButtons.forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.range === normalizedRange);
    });
    const guard = guardHighFreqRange(normalizedRange, detailInterval);
    if (guard.downgraded && guard.intervalKey && guard.intervalKey !== detailInterval) {
      setDetailInterval(guard.intervalKey, { persist: true, skipReload: true });
      const label = getIntervalLabel(guard.intervalKey);
      setDetailStatus(TEXT.detailRangeGuard(label));
    }
    updateDetailTimeScale(detailInterval);
    chartLazyLoading = false;
    chartLazyExhausted = false;
    chartLazyEmptyStreak = 0;
    chartLazyCursor = null;
    chartTickCursor = null;
    lastFallbackUpdateAt = 0;
    chartTradeBuffer = [];
    if (chartTradeFlushHandle) {
      cancelAnimationFrame(chartTradeFlushHandle);
      chartTradeFlushHandle = null;
    }
    updateDataStatusUI();
  }

  function updateDetailTimeScale(intervalKey) {
    if (!detailManager || !detailManager.chart) return;
    const spec = resolveIntervalSpec(intervalKey);
    const isIntraday = Boolean(spec && spec.unit !== 'day');
    const showSeconds = Boolean(spec && (spec.unit === 'second' || spec.unit === 'tick'));
    const includeDate = spec && spec.unit !== 'day' ? true : null;
    const fullDate = Boolean(spec && spec.unit !== 'day');
    detailManager.setAxisOptions({ timeVisible: isIntraday, showSeconds, includeDate, fullDate });
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

  function loadIntervalGroupPrefs() {
    const raw = loadPreference(PREF_INTERVAL_GROUPS_KEY, '{}');
    try {
      const parsed = JSON.parse(raw);
      intervalGroupState = parsed && typeof parsed === 'object' ? parsed : {};
    } catch (error) {
      intervalGroupState = {};
    }
  }

  function saveIntervalGroupPrefs() {
    savePreference(PREF_INTERVAL_GROUPS_KEY, JSON.stringify(intervalGroupState || {}));
  }

  function applyIntervalGroupState() {
    if (!intervalGroupToggles.length) return;
    intervalGroupToggles.forEach((btn) => {
      const group = btn.dataset.group;
      if (!group) return;
      const groupEl = btn.closest('.detail-interval-group');
      if (!groupEl) return;
      const collapsed = Boolean(intervalGroupState && intervalGroupState[group]);
      groupEl.classList.toggle('is-collapsed', collapsed);
      btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    });
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
    const previousInterval = normalizeIntervalKey(detailInterval) || '';
    let normalized = normalizeIntervalKey(nextKey);
    if (!normalized) {
      setDetailStatus(TEXT.intervalInvalid, true);
      return;
    }
    const guard = guardHighFreqRange(detailRange, normalized);
    if (guard.downgraded && guard.intervalKey && guard.intervalKey !== normalized) {
      normalized = guard.intervalKey;
      const label = getIntervalLabel(normalized);
      setDetailStatus(TEXT.detailRangeGuard(label));
    }
    if (normalized !== previousInterval) {
      bumpChartContextVersion();
    }
    detailInterval = normalized;
    if (detailSymbol) {
      resetWaveAnalysisState({ keepStatus: true });
    }
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
    chartLazyLoading = false;
    chartLazyExhausted = false;
    chartLazyEmptyStreak = 0;
    chartLazyCursor = null;
    if (detailManager) {
      const spec = resolveIntervalSpec(normalized);
      if (spec) {
        // Prevent mixed granularity: clear old bars before the new-interval payload arrives.
        detailManager.setIntervalSpec(spec, { preserveData: false });
        detailManager.setData([], { intervalSpec: spec, fitContent: true });
        detailLastChartKey = '';
      }
    }
    chartTradeBuffer = [];
    if (chartTradeFlushHandle) {
      cancelAnimationFrame(chartTradeFlushHandle);
      chartTradeFlushHandle = null;
    }
    lastLiveUpdateAt = 0;
    lastChartSocketUpdateAt = 0;
    lastFallbackUpdateAt = 0;
    chartTickCursor = null;
    clearLiveWait();
    if (chartVisible && detailSymbol) {
      scheduleLiveWait();
    }
    if (!skipReload && detailSymbol) {
      loadDetailData(detailSymbol, detailRange, { renderChart: true, allowCache: false });
    }
    updateDataStatusUI();
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
    const options = Array.prototype.slice.call(intervalMenu.querySelectorAll('[data-role="detail-interval-select"]'));
    return options.filter((btn) => btn.offsetParent !== null);
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

  function updateDetailWatchToggle() {
    if (!detailWatchToggle) return;
    const normalized = normalizeSymbol(detailSymbol);
    const isWatched = normalized && watchPool.includes(normalized);
    detailWatchToggle.classList.toggle('is-active', Boolean(isWatched));
    detailWatchToggle.setAttribute('aria-pressed', isWatched ? 'true' : 'false');
    const label = isWatched ? TEXT.typeaheadRemove : TEXT.typeaheadAdd;
    if (label) {
      detailWatchToggle.setAttribute('aria-label', label);
    }
    detailWatchToggle.textContent = isWatched ? '★' : '☆';
  }

  function openDetailPanel(symbol) {
    bumpChartContextVersion();
    detailSymbol = symbol;
    lastLiveUpdateAt = 0;
    lastChartSocketUpdateAt = 0;
    lastFallbackUpdateAt = 0;
    chartTickCursor = null;
    detailStatusOverride = '';
    detailStatusIsError = false;
    resetWaveAnalysisState();
    const openChartDirectly = Boolean(waveToolLaunchPending);
    waveToolLaunchPending = false;
    setView(openChartDirectly ? 'chart' : 'detail');
    setWorkbenchPanel('overview');
    highlightSelectedRows(symbol);
    if (detailSymbolEl) {
      detailSymbolEl.textContent = symbol;
    }
    setTradeSymbol(symbol);
    if (detailNameEl) {
      detailNameEl.textContent = symbol;
    }
    setDetailPriceValue(null);
    if (detailChangeEl) {
      detailChangeEl.textContent = '--';
      detailChangeEl.classList.remove('is-up', 'is-down');
    }
    if (detailMetaEl) {
      detailMetaEl.textContent = '';
    }
    if (detailLatency) {
      detailLatency.textContent = '';
    }
    updateDetailWatchToggle();
    setDetailRange(detailRange);
    resetNewsPagingState(symbol);
    renderNewsPanel([], { append: false });
    loadDetailData(symbol, detailRange, { renderChart: openChartDirectly, allowCache: true });
  }

  function openChartView() {
    if (!detailSymbol) {
      setStatus(TEXT.statusNeedSymbol);
      setView('list');
      return;
    }
    waveToolLaunchPending = false;
    setView('chart');
    setDetailRange(detailRange);
    if (!normalizeIntervalKey(detailInterval)) {
      setDetailInterval(resolveDefaultInterval(detailRange), { persist: true, skipReload: true });
    }
    if (!isChartContainerReady()) {
      scheduleChartInit();
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
    if (!chartVisible || !detailSymbol || !chartSocketReady) return false;
    if (!lastChartSocketUpdateAt) return false;
    return Date.now() - lastChartSocketUpdateAt < LIVE_CHART_STALE_MS;
  }

  function resetChartSocketBackoff() {
    chartSocketRetryAttempts = 0;
    chartSocketRetryDelay = CHART_SOCKET_RETRY_BASE_MS;
  }

  function scheduleChartSocketReconnect() {
    if (chartSocketRetryTimer) return;
    const delay = chartSocketRetryDelay || CHART_SOCKET_RETRY_BASE_MS;
    const jitter = Math.random() * 0.3 * delay;
    chartSocketRetryTimer = setTimeout(() => {
      chartSocketRetryTimer = null;
      connectChartSocket();
    }, delay + jitter);
    chartSocketRetryAttempts += 1;
    chartSocketRetryDelay = Math.min(CHART_SOCKET_RETRY_MAX_MS, delay * 2);
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
      chartSocketLastClose = null;
      resetChartSocketBackoff();
      updateDataStatusUI();
      if (chartSocketSymbol) {
        chartSocket.send(JSON.stringify({ action: 'subscribe', symbol: chartSocketSymbol }));
        scheduleLiveWait();
      }
      if (chartHeartbeatTimer) {
        clearInterval(chartHeartbeatTimer);
      }
      chartHeartbeatTimer = setInterval(() => {
        if (!chartSocket || chartSocket.readyState !== WebSocket.OPEN) return;
        chartSocket.send(JSON.stringify({ action: 'ping' }));
      }, 15000);
    };
    chartSocket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload && (payload.type === 'pong' || payload.action === 'pong')) {
          chartHeartbeatLastPong = Date.now();
          if (payload.server_ts) {
            updateDetailLatencyFromTs(payload.server_ts);
          }
          return;
        }
        applyChartTradeUpdate(payload);
      } catch (error) {
        return;
      }
    };
    chartSocket.onclose = (event) => {
      chartSocketReady = false;
      chartSocketLastClose = {
        code: event && Number.isFinite(event.code) ? event.code : null,
        reason: event && event.reason ? String(event.reason) : '',
        at: Date.now(),
      };
      updateDataStatusUI();
      if (chartHeartbeatTimer) {
        clearInterval(chartHeartbeatTimer);
        chartHeartbeatTimer = null;
      }
      if (chartVisible && detailSymbol) {
        scheduleChartSocketReconnect();
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
    chartSocketLastClose = null;
    updateDataStatusUI();
    resetChartSocketBackoff();
    chartTradeBuffer = [];
    if (chartHeartbeatTimer) {
      clearInterval(chartHeartbeatTimer);
      chartHeartbeatTimer = null;
    }
    stopChartPolling();
    if (chartTradeFlushHandle) {
      cancelAnimationFrame(chartTradeFlushHandle);
      chartTradeFlushHandle = null;
    }
  }

  function setChartSocketSymbol(symbol) {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return;
    if (chartSocketSymbol && chartSocketSymbol !== normalized && chartSocketReady && chartSocket) {
      chartSocket.send(JSON.stringify({ action: 'unsubscribe', symbol: chartSocketSymbol }));
    }
    chartSocketSymbol = normalized;
    chartTradeBuffer = [];
    if (chartTradeFlushHandle) {
      cancelAnimationFrame(chartTradeFlushHandle);
      chartTradeFlushHandle = null;
    }
    lastLiveUpdateAt = 0;
    lastChartSocketUpdateAt = 0;
    lastFallbackUpdateAt = 0;
    chartTickCursor = null;
    updateDataStatusUI();
    if (!chartSocketReady) {
      connectChartSocket();
      scheduleLiveWait();
      return;
    }
    chartSocket.send(JSON.stringify({ action: 'subscribe', symbol: chartSocketSymbol }));
    scheduleLiveWait();
  }

  function applyChartTradeUpdate(update) {
    enqueueChartTrade(update);
  }

  function applyChartTradeBatch(batch) {
    if (!Array.isArray(batch) || !batch.length) return;
    let lastUpdate = null;
    for (let i = 0; i < batch.length; i += 1) {
      const update = batch[i];
      if (!update || typeof update !== 'object') continue;
      const price = typeof update.price === 'number' ? update.price : Number.parseFloat(update.price);
      if (!Number.isFinite(price)) continue;
      const size = typeof update.size === 'number' ? update.size : Number.parseFloat(update.size);
      const ts = typeof update.ts === 'number' ? update.ts : Number.parseFloat(update.ts);
      if (detailManager) {
        detailManager.applyTradeUpdate({ price, size, ts });
      }
      lastUpdate = { update, price, ts };
    }
    if (!lastUpdate) return;
    setDetailPriceValue(lastUpdate.price);
    if (detailChangeEl && detailManager && detailManager.ohlcData.length > 1) {
      const last = detailManager.ohlcData[detailManager.ohlcData.length - 1];
      const prev = detailManager.ohlcData[detailManager.ohlcData.length - 2];
      const changePct = prev && prev.close ? ((last.close / prev.close) - 1) * 100 : null;
      detailChangeEl.textContent = typeof changePct === 'number' ? formatChange(changePct) : '--';
      detailChangeEl.classList.remove('is-up', 'is-down');
      applyChangeState(detailChangeEl, changePct, false);
    }
    if (detailSubtitle && detailManager && detailManager.ohlcData.length) {
      const lastBar = detailManager.ohlcData[detailManager.ohlcData.length - 1];
      const lastTs = lastBar ? normalizeEpochSeconds(lastBar.time) : null;
      if (Number.isFinite(lastTs)) {
        const intervalSpec = detailManager.intervalSpec;
        const showSeconds = Boolean(intervalSpec && (intervalSpec.unit === 'tick' || intervalSpec.unit === 'second'));
        const timeLabel = formatAxisTime(lastTs, {
          timezoneMode,
          showSeconds,
          includeDate: true,
          fullDate: true,
        });
        detailSubtitle.textContent = langPrefix === 'zh' ? `数据时间：${timeLabel}` : `Data time: ${timeLabel}`;
      }
    }
    const latencyTs = Number.isFinite(lastUpdate.update.server_ts)
      ? lastUpdate.update.server_ts
      : lastUpdate.ts;
    if (Number.isFinite(latencyTs)) {
      const intervalSpec = detailManager ? detailManager.intervalSpec : null;
      updateDetailLatencyFromTs(latencyTs, { intervalSpec });
    }
    if (detailStatus && !lastTradeStaleLabel) {
      const marker = langPrefix === 'zh' ? '行情已延迟' : 'Stale data';
      if ((detailStatus.textContent || '').includes(marker)) {
        setDetailStatus(TEXT.detailLive);
      }
    }
    stopLiveQuotePolling();
    stopChartPolling();
    const liveNow = Date.now();
    lastLiveUpdateAt = liveNow;
    lastChartSocketUpdateAt = liveNow;
    lastFallbackUpdateAt = 0;
    if (Number.isFinite(lastUpdate.ts)) {
      chartTickCursor = lastUpdate.ts;
    }
    if (detailStatusOverride === TEXT.detailLiveWaiting) {
      setDetailStatus('');
    }
    scheduleWaveAutoAnalyze();
  }

  function enqueueChartTrade(update) {
    if (!update || typeof update !== 'object') return;
    const symbol = normalizeSymbol(update.symbol || '');
    if (!symbol || symbol !== detailSymbol) return;
    const isTickInterval = Boolean(detailManager && detailManager.intervalSpec && detailManager.intervalSpec.unit === 'tick');
    if (isTickInterval) {
      const batch = [];
      if (Array.isArray(update.trades)) {
        update.trades.forEach((trade) => {
          if (!trade || typeof trade !== 'object') return;
          batch.push({
            symbol,
            price: trade.price,
            size: trade.size,
            ts: trade.ts,
            server_ts: update.server_ts,
          });
        });
      } else {
        batch.push(update);
      }
      applyChartTradeBatch(batch);
      return;
    }
    if (Array.isArray(update.trades)) {
      update.trades.forEach((trade) => {
        if (!trade || typeof trade !== 'object') return;
        chartTradeBuffer.push({
          symbol,
          price: trade.price,
          size: trade.size,
          ts: trade.ts,
          server_ts: update.server_ts,
        });
      });
    } else {
      chartTradeBuffer.push(update);
    }
    if (chartTradeBuffer.length > MAX_CHART_TRADE_BUFFER) {
      chartTradeBuffer = chartTradeBuffer.slice(-MAX_CHART_TRADE_BUFFER);
    }
    if (chartTradeFlushHandle) return;
    chartTradeFlushHandle = requestAnimationFrame(flushChartTradeBuffer);
  }

  function flushChartTradeBuffer() {
    chartTradeFlushHandle = null;
    if (!chartTradeBuffer.length) return;
    const batch = chartTradeBuffer;
    chartTradeBuffer = [];
    applyChartTradeBatch(batch);
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

  function getSymbolLogoUrl(symbol) {
    const safe = (symbol || '').toString().trim().toUpperCase();
    if (!safe) return '';
    return `https://storage.googleapis.com/iex/api/logos/${encodeURIComponent(safe)}.png`;
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
      metricKey: 'change_pct_period',
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
    if (listType === 'all') return coerceNumber(item.change_pct_period ?? item.change_pct);
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

  function buildChartContextKey(symbol = detailSymbol, rangeKey = detailRange, intervalKey = detailInterval) {
    const normalizedSymbol = normalizeSymbol(symbol);
    const normalizedRange = (rangeKey || '').toString().trim().toLowerCase();
    const normalizedInterval = normalizeIntervalKey(intervalKey) || '';
    return `${normalizedSymbol}|${normalizedRange}|${normalizedInterval}`;
  }

  function getChartContextSnapshot(symbol = detailSymbol, rangeKey = detailRange, intervalKey = detailInterval) {
    return {
      version: chartDataContextVersion,
      key: buildChartContextKey(symbol, rangeKey, intervalKey),
    };
  }

  function isChartContextCurrent(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return false;
    return snapshot.version === chartDataContextVersion && snapshot.key === buildChartContextKey();
  }

  function bumpChartContextVersion() {
    chartDataContextVersion += 1;
    return chartDataContextVersion;
  }

  function resolveRangeWindowSeconds(rangeKey) {
    const key = (rangeKey || '').toString().trim().toLowerCase();
    if (key.endsWith('mo') && /^\d+mo$/.test(key)) {
      const months = Math.max(1, parseInt(key.slice(0, -2), 10) || 1);
      return months * 30 * 86400;
    }
    if (key.endsWith('d') && /^\d+d$/.test(key)) {
      const days = Math.max(1, parseInt(key.slice(0, -1), 10) || 1);
      return days * 86400;
    }
    return 86400;
  }

  function guardHighFreqRange(rangeKey, intervalKey) {
    const spec = resolveIntervalSpec(intervalKey);
    if (!spec) return { intervalKey, downgraded: false };
    const isHigh = spec.unit === 'tick' || spec.unit === 'second';
    const key = (rangeKey || '').toString().trim().toLowerCase();
    const isLongRange = key === '1mo' || key === '6mo';
    if (isHigh && isLongRange) {
      const fallback = resolveDefaultInterval(key);
      return { intervalKey: fallback, downgraded: true };
    }
    return { intervalKey, downgraded: false };
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
    const lastTime = normalizeEpochSeconds(last.time);
    const prevTime = normalizeEpochSeconds(prev.time);
    if (!Number.isFinite(lastTime) || !Number.isFinite(prevTime)) return null;
    const delta = lastTime - prevTime;
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
    if (!payload) return;
    suggestionPool = normalizeList(payload.suggestions || []);
    recentPool = normalizeList(payload.recent_queries || []);
    watchPool = normalizeList(payload.watchlist || []);
    updateDetailWatchToggle();
    syncWatchButtons();
    if (!hasTypeaheadUi) return;
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

  function getResponseContentType(response) {
    if (!response || !response.headers || typeof response.headers.get !== 'function') {
      return '';
    }
    return String(response.headers.get('content-type') || '').toLowerCase();
  }

  async function parseApiResponse(response) {
    if (!getResponseContentType(response).includes('application/json')) {
      return { payload: null };
    }
    try {
      const payload = await response.json();
      return { payload: payload && typeof payload === 'object' ? payload : {} };
    } catch (_error) {
      return { payload: null };
    }
  }

  function resolveUnsupportedTimeframeMessage(payload) {
    let timeframeLabel = TEXT.timeframes[currentTimeframe] || currentTimeframe;
    const timeframe = payload && typeof payload.timeframe === 'object' ? payload.timeframe : null;
    if (timeframe) {
      const localizedLabel = langPrefix === 'zh' ? timeframe.label : timeframe.label_en;
      timeframeLabel = localizedLabel || timeframe.key || timeframeLabel;
    }
    if (typeof TEXT.timeframeNotSupported === 'function') {
      return TEXT.timeframeNotSupported(timeframeLabel);
    }
    return (payload && (payload.message || payload.error)) || TEXT.genericError;
  }

  function hasRenderedRankingRows() {
    if (!listContainer) return false;
    return Boolean(listContainer.querySelector('tr[data-symbol]'));
  }

  function buildSnapshotMetaFromPayload(payload) {
    if (!payload || typeof payload !== 'object') return null;
    if (payload.snapshot_refresh && typeof payload.snapshot_refresh === 'object') {
      return payload.snapshot_refresh;
    }
    const state = payload.snapshot_state;
    if (!state || typeof state !== 'object') return null;
    const building = Boolean(state.building);
    const progress = Number(state.building_progress);
    const generatedAt = state.active_generated_at;
    const meta = {
      status: building ? 'running' : 'idle',
      latest_summary: generatedAt ? { generated_at: generatedAt, generated_ts: generatedAt } : {},
    };
    if (building) {
      meta.progress = {
        status: 'running',
        chunks_completed: Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : 0,
        total_chunks: 100,
      };
    }
    return meta;
  }

  async function loadData(query = '', options = {}) {
    const rawQuery = (query || '').toString().trim();
    const normalizedQuery = normalizeSymbol(rawQuery);
    const activeListType = normalizeListType(options.listType || currentListType);
    const requestedTimeframeKey = currentTimeframe;
    const skipListRender = Boolean(options.skipListRender);
    const keepListType = Boolean(options.keepListType);
    const openDetail = options.openDetail !== false;
    const isAppend = Boolean(options.append);
    const pageSizeCandidate = Number.parseInt(options.pageSize ?? options.limit, 10);
    const pageSize = Number.isFinite(pageSizeCandidate) ? pageSizeCandidate : RANK_PAGE_SIZE;
    const offset = Number.isFinite(options.offset) ? Number(options.offset) : 0;
    if (activeListType === 'all') {
      if (rankingRequestController && !rankingRequestController.signal.aborted) {
        rankingRequestController.abort();
      }
      rankingRequestController = null;
      rankingRequestKey = '';
      await loadAllStocks({ query: rawQuery, page: 1 });
      return;
    }
    const requestKey = JSON.stringify({
      timeframe: currentTimeframe,
      list: activeListType,
      query: normalizedQuery,
      offset: isAppend ? offset : 0,
      limit: pageSize,
      mode: options.watchAction || options.recentAction || normalizedQuery ? 'post' : 'get',
    });
    if (
      requestKey === rankingRequestKey &&
      rankingRequestController &&
      !rankingRequestController.signal.aborted
    ) {
      if (!isAppend) {
        setStatus(TEXT.refreshSkipped, { forceState: 'refreshing', forceMessage: true });
      }
      return;
    }
    if (!isAppend && rankingRequestController && !rankingRequestController.signal.aborted) {
      rankingRequestController.abort();
    }
    const requestController = new AbortController();
    rankingRequestController = requestController;
    rankingRequestKey = requestKey;
    const requestNonce = ++rankingRequestNonce;
    const isCurrentRequest = () =>
      rankingRequestController === requestController && rankingRequestNonce === requestNonce;
    const preserveRowsWhileLoading = !isAppend && !skipListRender && hasRenderedRankingRows();
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
      if (!skipListRender && !preserveRowsWhileLoading) {
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
          signal: requestController.signal,
        });
      } else {
        const params = new URLSearchParams({ timeframe: currentTimeframe, list: activeListType });
        if (langPrefix) {
          params.set('lang', langPrefix);
        }
        if (!normalizedQuery) {
          params.set('limit', String(pageSize));
        }
        if (isAppend || offset) {
          params.set('offset', String(isAppend ? offset : 0));
        }
        response = await fetch(`${endpoint}?${params.toString()}`, {
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
          credentials: 'same-origin',
          signal: requestController.signal,
        });
      }
      if (!isCurrentRequest() || requestController.signal.aborted) {
        return;
      }
      const parsed = await parseApiResponse(response);
      const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
      if (response.status === 429 || payload.rate_limited) {
        const retryAfterHeader = Number.parseInt(response.headers.get('retry-after') || '', 10);
        const retryAfterSeconds = payload.retry_after_seconds || (Number.isFinite(retryAfterHeader) ? retryAfterHeader : null);
        const snapshotKey =
          (payload.ranking_timeframe && payload.ranking_timeframe.key) ||
          (payload.timeframe && payload.timeframe.key) ||
          currentTimeframe;
        const snapshotMeta = buildSnapshotMetaFromPayload(payload);
        if (snapshotMeta) {
          setSnapshotStatus(snapshotMeta, snapshotKey);
        }
        if (preserveRowsWhileLoading && hasRenderedRankingRows()) {
          setStatus(TEXT.keepPreviousOnRate, { forceState: 'stale', forceMessage: true });
          hideChipSkeleton(recentChips);
          hideChipSkeleton(watchlistChips);
        } else {
          setRetryingState(retryAfterSeconds);
        }
        return;
      }
      if (!response.ok) {
        if (payload.error_code === 'timeframe_not_supported') {
          const unsupportedMessage = resolveUnsupportedTimeframeMessage(payload);
          if (!isAppend) {
            renderEmpty(listContainer, unsupportedMessage);
            setStatus(unsupportedMessage, { forceState: 'stale', forceMessage: true });
            setSource('unknown');
            hideChipSkeleton(recentChips);
            hideChipSkeleton(watchlistChips);
          } else {
            setRankingLoadingMore(false);
          }
          return;
        }
        throw new Error(payload.error || payload.message || TEXT.genericError);
      }
      const responseListType = normalizeListType(payload.list_type || activeListType);
      const responseTimeframeKey =
        (payload.timeframe && payload.timeframe.key) ||
        (payload.ranking_timeframe && payload.ranking_timeframe.key) ||
        requestedTimeframeKey;
      if (
        responseListType !== activeListType ||
        (responseTimeframeKey && responseTimeframeKey !== requestedTimeframeKey)
      ) {
        return;
      }
      if (!keepListType && payload.list_type && responseListType !== currentListType) {
        setActiveListType(responseListType);
      }
      let items = Array.isArray(payload.items) ? payload.items : [];
      if (!items.length) {
        if (responseListType === 'losers') {
          items = payload.losers || [];
        } else if (responseListType === 'most_active') {
          items = payload.most_actives || [];
        } else if (responseListType === 'top_turnover') {
          items = payload.top_turnover || [];
        } else {
          items = payload.gainers || [];
        }
      }
      const payloadDataState = typeof payload.data_state === 'string' ? payload.data_state : '';
      const hasIncomingItems = Array.isArray(items) && items.length > 0;
      const keepExistingRowsForState =
        !isAppend &&
        preserveRowsWhileLoading &&
        hasRenderedRankingRows() &&
        !skipListRender &&
        !hasIncomingItems &&
        ['building', 'stale', 'limited'].includes(payloadDataState);
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
      const denseTimeframeFromPayload =
        (payload.ranking_timeframe && payload.ranking_timeframe.key) ||
        (payload.timeframe && payload.timeframe.key) ||
        currentTimeframe;
      denseItems = Array.isArray(items) ? items.slice() : [];
      denseListType = responseListType;
      denseTimeframeKey = denseTimeframeFromPayload;
      if (payload.data_source) {
        const sourceLabels = TEXT.sourceLabels || {};
        denseSourceKey = sourceLabels[payload.data_source] ? payload.data_source : 'unknown';
      }
      updateDenseStrip();
      if (!skipListRender) {
        if (keepExistingRowsForState) {
          setStatus(TEXT.keepPreviousOnRate, { forceState: 'stale', forceMessage: true });
        } else if (isAppend && rankSort === 'default') {
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
        syncGlobalQuickStorage();
        const tfKey = payload.timeframe && payload.timeframe.key;
        const tfLabel = payload.timeframe && (langPrefix === 'zh' ? payload.timeframe.label : payload.timeframe.label_en);
        const statusNotes = [];
        const rankingTimeframe = payload.ranking_timeframe;
        const snapshotStatePayload =
          payload.snapshot_state && typeof payload.snapshot_state === 'object' ? payload.snapshot_state : null;
        const dataState = typeof payload.data_state === 'string' ? payload.data_state : '';
        if (snapshotStatePayload && snapshotStatePayload.served_from === 'building_fallback') {
          statusNotes.push(TEXT.keepPreviousOnRate);
        } else if (dataState === 'building') {
          statusNotes.push(langPrefix === 'zh' ? '构建中' : 'Building');
        } else if (dataState === 'ready') {
          statusNotes.push(langPrefix === 'zh' ? '就绪' : 'Ready');
        } else if (dataState === 'stale') {
          statusNotes.push(langPrefix === 'zh' ? '沿用旧版' : 'Using previous');
        } else if (dataState === 'limited') {
          statusNotes.push(langPrefix === 'zh' ? '等待首轮构建' : 'Waiting first build');
        }
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
        const snapshotMeta = buildSnapshotMetaFromPayload(payload);
        if (snapshotMeta) {
          setSnapshotStatus(snapshotMeta, snapshotKey);
        }
        setSource(payload.data_source);
        if (normalizedQuery && openDetail) {
          openDetailPanel(normalizedQuery);
        }
      }
    } catch (error) {
      if (requestController.signal.aborted || (error && error.name === 'AbortError')) {
        return;
      }
      if (isAppend) {
        setRankingLoadingMore(false);
        return;
      }
      if (preserveRowsWhileLoading && hasRenderedRankingRows()) {
        setStatus(TEXT.keepPreviousOnError, { forceState: 'stale', forceMessage: true });
        hideChipSkeleton(recentChips);
        hideChipSkeleton(watchlistChips);
        return;
      }
      renderError(listContainer, error && error.message);
      setStatus(TEXT.statusError, { forceState: 'stale', forceMessage: true });
      setSource('unknown');
      hideChipSkeleton(recentChips);
      hideChipSkeleton(watchlistChips);
    } finally {
      const requestIsCurrent = isCurrentRequest();
      if (requestIsCurrent) {
        rankingRequestController = null;
        rankingRequestKey = '';
      }
      if (isAppend) {
        if (requestIsCurrent || !requestController.signal.aborted) {
          rankIsLoadingMore = false;
          setRankingLoadingMore(false);
        }
      } else {
        if (requestIsCurrent || !requestController.signal.aborted) {
          isListLoading = false;
          scheduleAutoRefresh();
          refreshStatusState();
        }
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
    const missingReasons = normalizeMissingReasons(item.missing_reasons);
    const row = document.createElement('tr');
    row.dataset.symbol = normalizeSymbol(symbol);
    row.setAttribute('aria-selected', 'false');
    const symbolCell = document.createElement('td');
    symbolCell.className = 'col-symbol';
    const metaWrap = document.createElement('div');
    metaWrap.className = 'rank-meta';
    const logoWrap = document.createElement('span');
    logoWrap.className = 'rank-logo';
    logoWrap.dataset.fallback = symbol.slice(0, 1);
    logoWrap.setAttribute('aria-hidden', 'true');
    const logoUrl = getSymbolLogoUrl(symbol);
    if (logoUrl) {
      const logoImg = document.createElement('img');
      logoImg.alt = langPrefix === 'zh' ? `${symbol} 标志` : `${symbol} logo`;
      logoImg.loading = 'lazy';
      logoImg.decoding = 'async';
      logoImg.referrerPolicy = 'no-referrer';
      logoImg.src = logoUrl;
      logoImg.onerror = () => {
        logoWrap.classList.add('is-fallback');
        logoImg.remove();
      };
      logoWrap.appendChild(logoImg);
    } else {
      logoWrap.classList.add('is-fallback');
    }
    const metaText = document.createElement('div');
    metaText.className = 'rank-meta-text';
    const symbolLine = document.createElement('div');
    symbolLine.className = 'rank-symbol';
    const symbolText = document.createElement('span');
    symbolText.className = 'rank-symbol-text';
    symbolText.textContent = symbol;
    symbolLine.appendChild(symbolText);
    if (item.exchange) {
      const exchangeBadge = document.createElement('span');
      exchangeBadge.className = 'rank-exchange-badge';
      exchangeBadge.textContent = item.exchange;
      symbolLine.appendChild(exchangeBadge);
    }
    const nameText = document.createElement('span');
    nameText.className = 'rank-name';
    if (item.name) {
      nameText.textContent = item.name;
    } else {
      nameText.textContent = '—';
      nameText.classList.add('is-empty');
    }
    metaText.appendChild(symbolLine);
    metaText.appendChild(nameText);
    metaWrap.appendChild(logoWrap);
    metaWrap.appendChild(metaText);
    symbolCell.appendChild(metaWrap);

    const priceCell = document.createElement('td');
    priceCell.className = 'col-price';
    const priceValue =
      typeof item.price === 'number'
        ? item.price
        : typeof item.last === 'number'
          ? item.last
          : Number.parseFloat(item.price);
    priceCell.textContent = Number.isFinite(priceValue) ? formatPrice4(priceValue) : '--';

    const changeCell = document.createElement('td');
    changeCell.className = 'col-metric';
    const changeValue =
      coerceNumber(item.change_pct_period) ??
      coerceNumber(item.change_pct_day) ??
      coerceNumber(item.change_pct);
    changeCell.textContent = formatChange(changeValue);
    applyChangeState(changeCell, changeValue, false);

    const volumeCell = document.createElement('td');
    volumeCell.className = 'col-volume';
    const volumeValue = coerceNumber(item.volume);
    if (volumeValue === null) {
      renderMissingReasonValue(volumeCell, missingReasons.volume);
    } else {
      clearMissingReasonState(volumeCell);
      volumeCell.textContent = formatCompactNumber(volumeValue);
    }
    volumeCell.classList.add('is-neutral');

    const turnoverCell = document.createElement('td');
    turnoverCell.className = 'col-turnover';
    const turnoverValue = coerceNumber(item.dollar_volume);
    if (turnoverValue === null) {
      renderMissingReasonValue(turnoverCell, missingReasons.dollar_volume);
    } else {
      clearMissingReasonState(turnoverCell);
      turnoverCell.textContent = formatCompactCurrency(turnoverValue);
    }
    turnoverCell.classList.add('is-neutral');

    const rangeCell = document.createElement('td');
    rangeCell.className = 'col-range';
    const rangeValue = coerceNumber(item.range_pct);
    if (rangeValue === null) {
      renderMissingReasonValue(rangeCell, missingReasons.range_pct);
    } else {
      clearMissingReasonState(rangeCell);
      rangeCell.textContent = formatChange(rangeValue);
    }
    rangeCell.classList.add('is-neutral');

    const prevCloseCell = document.createElement('td');
    prevCloseCell.className = 'col-prev-close';
    const prevCloseValue = coerceNumber(item.prev_close ?? item.prevClose);
    if (prevCloseValue === null) {
      renderMissingReasonValue(prevCloseCell, missingReasons.prev_close);
    } else {
      clearMissingReasonState(prevCloseCell);
      prevCloseCell.textContent = formatPrice4(prevCloseValue);
    }
    prevCloseCell.classList.add('is-neutral');

    const openCell = document.createElement('td');
    openCell.className = 'col-open';
    const openValue = coerceNumber(item.open);
    if (openValue === null) {
      renderMissingReasonValue(openCell, missingReasons.open);
    } else {
      clearMissingReasonState(openCell);
      openCell.textContent = formatPrice4(openValue);
    }
    openCell.classList.add('is-neutral');

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
    row.appendChild(priceCell);
    row.appendChild(changeCell);
    row.appendChild(volumeCell);
    row.appendChild(turnoverCell);
    row.appendChild(rangeCell);
    row.appendChild(prevCloseCell);
    row.appendChild(openCell);
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
      rankingChangeLabel.textContent = langPrefix === 'zh' ? '涨跌幅' : 'Chg%';
    } else if (rankingChangeHeader) {
      rankingChangeHeader.textContent = langPrefix === 'zh' ? '涨跌幅' : 'Chg%';
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
        priceEl.textContent = typeof item.price === 'number' ? formatPrice4(item.price) : '--';
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

  function updateAllStocksCount(total, page, totalPages, loadedCount) {
    if (!allStocksCount) return;
    const safeTotal = Number.isFinite(total) ? total : 0;
    const safePage = Number.isFinite(page) ? page : 1;
    const safeTotalPages = Number.isFinite(totalPages) ? totalPages : 1;
    const safeLoaded = Number.isFinite(loadedCount) ? loadedCount : 0;
    if (langPrefix === 'zh') {
      allStocksCount.textContent = `共 ${safeTotal} 只 · 已加载 ${safeLoaded} · 第 ${safePage}/${safeTotalPages} 页`;
    } else {
      allStocksCount.textContent = `${safeTotal} symbols · Loaded ${safeLoaded} · Page ${safePage}/${safeTotalPages}`;
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
    const append = Boolean(options.append);
    if (append && (allStocksIsLoadingMore || !allStocksHasMore)) {
      return;
    }

    if (!append) {
      const nextLetter = options.letter ? options.letter.toString().toUpperCase() : allStocksLetter;
      const nextQuery = typeof options.query === 'string' ? options.query.trim() : allStocksQuery;
      const nextSize =
        typeof options.size === 'number' && Number.isFinite(options.size) ? options.size : ALL_MODE_PAGE_SIZE;
      allStocksLetter = nextLetter || 'ALL';
      allStocksQuery = nextQuery;
      allStocksSize = Math.min(200, Math.max(20, nextSize || ALL_MODE_PAGE_SIZE));
      allStocksPage = 0;
      allStocksHasMore = false;
      allStocksIsLoadingMore = false;
      allStocksTotal = 0;
      rankItemsBase = [];
      rankItems = [];
      resetRankPaging(allStocksSize);
      setAllStocksLetter(allStocksLetter);
      if (listContainer) {
        setListLoading(listContainer);
      }
      updateStatusContext();
      setStatus(TEXT.allStocksLoading, { forceState: 'refreshing' });
      isListLoading = true;
    } else {
      allStocksIsLoadingMore = true;
      rankIsLoadingMore = true;
      setRankingLoadingMore(true);
    }

    const requestPage = append
      ? allStocksPage + 1
      : Math.max(1, Number.isFinite(options.page) ? Number(options.page) : 1);

    try {
      const endpointBase = assetsUrl || '/api/market/assets/';
      const endpoint = endpointBase.endsWith('/') ? endpointBase : `${endpointBase}/`;
      const params = new URLSearchParams({
        page: String(requestPage),
        size: String(allStocksSize || ALL_MODE_PAGE_SIZE),
      });
      params.set('timeframe', currentTimeframe || '1d');
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
      const parsed = await parseApiResponse(response);
      const payload = parsed.payload && typeof parsed.payload === 'object' ? parsed.payload : {};
      if (!response.ok) {
        if (payload.error_code === 'timeframe_not_supported') {
          const unsupportedMessage = resolveUnsupportedTimeframeMessage(payload);
          if (!append && listContainer) {
            renderEmpty(listContainer, unsupportedMessage);
          }
          setStatus(unsupportedMessage, { forceState: 'stale', forceMessage: true });
          setSource('unknown');
          return;
        }
        throw new Error(payload.error || payload.message || TEXT.genericError);
      }

      const rawItems = Array.isArray(payload.items) ? payload.items : [];
      const incomingItems = rawItems
        .map((item) => {
          const symbol = normalizeSymbol(item && item.symbol);
          if (!symbol) return null;
          const price = coerceNumber(item.last ?? item.price);
          const changePctPeriod = coerceNumber(item.change_pct_period ?? item.change_pct);
          const changePctDay = coerceNumber(item.change_pct_day ?? item.change_pct);
          return {
            ...item,
            symbol,
            price,
            last: price,
            change_pct: changePctPeriod,
            change_pct_period: changePctPeriod,
            change_pct_day: changePctDay === null ? changePctPeriod : changePctDay,
            volume: coerceNumber(item.volume),
            dollar_volume: coerceNumber(item.dollar_volume),
            range_pct: coerceNumber(item.range_pct),
            prev_close: coerceNumber(item.prev_close),
            open: coerceNumber(item.open),
            missing_reasons:
              item && typeof item.missing_reasons === 'object' ? item.missing_reasons : {},
          };
        })
        .filter(Boolean);

      const rankUpdate = updateRankItems(incomingItems, append);
      const total = Number(payload.total) || rankItemsBase.length;
      const page = Number(payload.page) || requestPage;
      const totalPages = Number(payload.total_pages) || 1;
      allStocksTotal = total;
      allStocksPage = page;
      allStocksHasMore = page < totalPages;
      rankHasMore = allStocksHasMore;
      rankNextOffset = allStocksHasMore ? page + 1 : null;
      rankPageSize = allStocksSize;
      if (rankSentinel) {
        rankSentinel.hidden = !allStocksHasMore;
      }

      denseListType = 'all';
      denseTimeframeKey = currentTimeframe;
      denseItems = rankUpdate.merged.slice();
      updateDenseStrip();
      updateAllStocksCount(total, page, totalPages, rankUpdate.merged.length);

      lastRankingTimeframe = {
        key: currentTimeframe,
        label: TEXT.timeframes[currentTimeframe] || currentTimeframe,
        label_en: TEXT.timeframes[currentTimeframe] || currentTimeframe,
      };
      lastRankingListType = 'all';

      if (listContainer) {
        if (append && rankSort === 'default') {
          appendList(listContainer, rankUpdate.appended, lastRankingTimeframe, 'all');
        } else {
          renderList(listContainer, rankUpdate.merged, lastRankingTimeframe, 'all');
          if (!rankUpdate.merged.length) {
            renderEmpty(listContainer, TEXT.allStocksEmpty || TEXT.emptyList);
          }
        }
      }

      if (rankUpdate.merged.length) {
        const liveSymbols = rankUpdate.merged
          .map((entry) => entry && entry.symbol)
          .filter(Boolean)
          .slice(0, 40);
        if (detailSymbol) {
          liveSymbols.unshift(detailSymbol);
        }
        requestLiveSymbols(liveSymbols);
      }

      setStatus(TEXT.updated);
      setStatusUpdated(TEXT.justNow);
    } catch (error) {
      if (!append && listContainer) {
        renderError(listContainer, error && error.message ? error.message : TEXT.genericError);
      }
      setStatus(TEXT.statusError, { forceState: 'stale', forceMessage: true });
    } finally {
      allStocksIsLoadingMore = false;
      rankIsLoadingMore = false;
      setRankingLoadingMore(false);
      isListLoading = false;
      scheduleAutoRefresh();
      refreshStatusState();
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
      const isAllMode = currentListType === 'all';
      loadData(symbol, {
        watchAction: action,
        listType: isAllMode ? 'gainers' : currentListType,
        keepListType: true,
        skipListRender: true,
        openDetail: !isAllMode,
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

  if (rankGroupsToggle) {
    rankGroupsToggle.addEventListener('click', () => {
      setRankGroupsExpanded(!rankGroupsExpanded);
    });
  }

  if (rankingSortButton) {
    rankingSortButton.addEventListener('click', () => {
      if (!rankItemsBase.length) return;
      rankSort = toggleSortState(rankSort);
      updateSortIndicator();
      rankItems = applyRankSort(rankItemsBase);
      if (listContainer) {
        const timeframe =
          lastRankingTimeframe || {
            key: currentTimeframe,
            label: TEXT.timeframes[currentTimeframe] || currentTimeframe,
            label_en: TEXT.timeframes[currentTimeframe] || currentTimeframe,
          };
        renderList(listContainer, rankItems, timeframe, currentListType);
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

  if (instrumentBackBtn) {
    instrumentBackBtn.addEventListener('click', () => {
      setView('list');
    });
  }

  if (paneTabs.length) {
    paneTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const targetView = tab.dataset.view || 'detail';
        const targetPanel = tab.dataset.panel || '';
        if (targetView === 'chart') {
          openChartView();
          return;
        }
        setView('detail');
        if (targetPanel) {
          setWorkbenchPanel(targetPanel);
        }
      });
    });
  }

  if (viewChartButton) {
    viewChartButton.addEventListener('click', () => {
      openChartView();
    });
  }

  if (contextTabs.length) {
    const savedTab = loadPreference(PREF_CONTEXT_TAB_KEY, '');
    setContextTab(savedTab, { persist: false });
    contextTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        if (!target) return;
        setContextTab(target, { persist: true });
      });
    });
  }

  if (keyStatsChartBtn) {
    keyStatsChartBtn.addEventListener('click', () => {
      openChartView();
    });
  }

  if (keyStatsNewsBtn) {
    keyStatsNewsBtn.addEventListener('click', () => {
      setContextTab('news', { persist: true, scrollIntoView: true });
    });
  }

  if (aiRefreshBtn) {
    aiRefreshBtn.addEventListener('click', () => {
      if (!detailSymbol) return;
      startAiRefreshCooldown();
      const requestId = activeDetailRequest || (detailRequestSeq += 1);
      activeDetailRequest = requestId;
      requestAiSummary(detailSymbol, detailRange, requestId);
    });
    updateAiRefreshButton();
  }

  if (newsFilterButtons.length) {
    newsFilterButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        setNewsFilter(btn.dataset.filter || 'all');
      });
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

  if (newsFilterButtons.length) {
    setNewsFilter(newsFilterMode);
  }

  buildIntervalLabelMap();
  loadIntervalPrefs();
  loadIntervalGroupPrefs();
  applyIntervalGroupState();
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

  if (intervalGroupToggles.length) {
    intervalGroupToggles.forEach((btn) => {
      btn.addEventListener('click', () => {
        const group = btn.dataset.group;
        if (!group) return;
        const groupEl = btn.closest('.detail-interval-group');
        if (!groupEl) return;
        const nextCollapsed = !groupEl.classList.contains('is-collapsed');
        groupEl.classList.toggle('is-collapsed', nextCollapsed);
        btn.setAttribute('aria-expanded', nextCollapsed ? 'false' : 'true');
        intervalGroupState = intervalGroupState || {};
        intervalGroupState[group] = nextCollapsed;
        saveIntervalGroupPrefs();
      });
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

  if (detailResetZoom) {
    detailResetZoom.addEventListener('click', () => {
      if (!ensureDetailChart() || !detailManager) return;
      detailManager.resetZoom();
    });
  }

  if (chartRefreshBtn) {
    chartRefreshBtn.addEventListener('click', () => {
      refreshChartBarsOnly();
    });
  }

  if (detailStatusReconnect) {
    detailStatusReconnect.addEventListener('click', () => {
      if (!detailSymbol) return;
      disconnectChartSocket();
      connectChartSocket();
      setChartSocketSymbol(detailSymbol);
      setDetailStatus(TEXT.detailLiveWaiting);
      refreshChartBarsOnly();
    });
  }

  if (waveOverlayToggle) {
    waveOverlayToggle.addEventListener('change', () => {
      setWaveOverlayEnabled(Boolean(waveOverlayToggle.checked));
    });
  }

  if (waveSeriesModeSelect) {
    waveSeriesModeSelect.addEventListener('change', () => {
      getWaveSeriesSettings();
      waveLastFingerprint = '';
      if (chartVisible) {
        scheduleWaveAutoAnalyze();
      }
    });
  }

  if (waveSmoothingWindowSelect) {
    waveSmoothingWindowSelect.addEventListener('change', () => {
      getWaveSeriesSettings();
      waveLastFingerprint = '';
      if (chartVisible) {
        scheduleWaveAutoAnalyze();
      }
    });
  }

  if (waveAnalyzeRun) {
    waveAnalyzeRun.addEventListener('click', () => {
      triggerWaveManualAnalyze();
    });
  }

  if (waveSampleSave) {
    waveSampleSave.addEventListener('click', () => {
      saveWaveSample();
    });
  }

  if (waveTrainRun) {
    waveTrainRun.addEventListener('click', () => {
      trainWaveModel();
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
    setQuickRailCollapsed(quickRailCollapsed, { persist: false });
    if (controlsExpanded) {
      positionControlsPopover();
    }
    resizeDetailChart();
    updateContextTabIndicator();
  });
  window.addEventListener(
    'scroll',
    () => {
      if (controlsExpanded) {
        positionControlsPopover();
      }
    },
    true
  );
  if (newsScrollRoot) {
    newsScrollRoot.addEventListener(
      'scroll',
      () => {
        if (currentView !== 'detail' || activeWorkbenchPanel !== 'news') return;
        const nearBottom = newsScrollRoot.scrollTop + newsScrollRoot.clientHeight >= newsScrollRoot.scrollHeight - 56;
        if (!nearBottom) return;
        newsScrollArmed = true;
        maybeLoadMoreNews();
      },
      { passive: true }
    );
  }
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
  setRankGroupsExpanded(false);
  setControlsExpanded(false);
  setQuickRailCollapsed(quickRailCollapsed, { persist: false });
  updateTimezoneToggle();
  updateAutoRefreshToggle();
  renderStatusTrack();
  updateTradeModeUI();
  updateAccountExecution();
  updateTradeUnitUI();
  updateTradeOrderMoreUI();
  updateTradeEstimates();
  renderRecentOrder(null);
  getWaveSeriesSettings();
  setWaveOverlayEnabled(waveOverlayEnabled);
  resetWavePanel();
  setWaveStatusText(TEXT.waveStatusWaiting);
  updateWaveButtons();
  refreshWaveMeta();
  setView(currentView || 'list');

  setupRankObserver();
  setupNewsObserver();
  const quickLaunchSymbol = consumeGlobalQuickLaunchSymbol();
  if (quickLaunchSymbol) {
    if (searchInput) {
      searchInput.value = quickLaunchSymbol;
    }
    loadData(quickLaunchSymbol, { openDetail: true });
  } else {
    loadData();
  }
  fetchTradeMode();
  refreshAccountSnapshot();
  connectMarketSocket();
})();
