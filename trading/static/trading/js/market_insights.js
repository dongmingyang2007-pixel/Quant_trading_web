(() => {
  const apiMeta = document.querySelector('meta[name="market-api"]');
  const apiUrl = apiMeta ? apiMeta.getAttribute('content') : window.MARKET_API_URL || '/market/api/';
  const langMeta = document.querySelector('meta[name="market-lang"]');
  const docLang = document.documentElement.getAttribute('lang');
  const langPrefix = ((langMeta && langMeta.getAttribute('content')) || docLang || navigator.language || 'zh')
    .toLowerCase()
    .slice(0, 2);

  const listGainers = document.querySelector('[data-role="gainers"]');
  const listLosers = document.querySelector('[data-role="losers"]');
  const statusText = document.querySelector('[data-role="status-text"]');
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

  const initialBtn = document.querySelector('.market-timeframe.is-active');
  let currentTimeframe = (initialBtn && initialBtn.getAttribute('data-timeframe')) || '1mo';
  let suggestionPool = [];
  let recentPool = [];
  let watchPool = [];
  let hideTypeaheadTimer = null;
  let typeaheadOptions = [];
  let typeaheadActiveIndex = -1;

  const TEXT = langPrefix === 'zh'
    ? {
        timeframes: { '1d': '近1日', '5d': '近5日', '1mo': '近1月', '6mo': '近6月' },
        loading: '正在加载',
        dataSuffix: '数据…',
        updated: '数据已更新',
        justNow: '刚刚',
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
      }
    : {
        timeframes: { '1d': '1D', '5d': '5D', '1mo': '1M', '6mo': '6M' },
        loading: 'Loading',
        dataSuffix: 'data…',
        updated: 'Data refreshed',
        justNow: 'just now',
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
      };

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

    const points = series.map((value) =>
      typeof value === 'number' ? Math.min(Math.max(value, 0), 1) : 0
    );

    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    if (invert) {
      gradient.addColorStop(0, 'rgba(244, 63, 94, 0.3)');
      gradient.addColorStop(1, 'rgba(244, 63, 94, 0)');
    } else {
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
    }

    ctx.beginPath();
    ctx.moveTo(0, height - points[0] * height);
    points.forEach((point, index) => {
      ctx.lineTo((index / (points.length - 1)) * width, height - point * height);
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
    const requestPayload = {
      timeframe: currentTimeframe,
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

    setStatus(`${TEXT.loading} ${TEXT.timeframes[currentTimeframe] || currentTimeframe} ${TEXT.dataSuffix}`);
    setListLoading(listGainers);
    setListLoading(listLosers);
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
        const params = new URLSearchParams({ timeframe: currentTimeframe });
        if (options.limit) {
          params.set('limit', options.limit);
        }
        response = await fetch(`${endpoint}?${params.toString()}`, {
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
          credentials: 'same-origin',
        });
      }
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || TEXT.genericError);
      }
      renderList(listGainers, payload.gainers || [], payload.timeframe);
      renderList(listLosers, payload.losers || [], payload.timeframe, true);
      if ((!payload.gainers || !payload.gainers.length) && (!payload.losers || !payload.losers.length)) {
        renderEmpty(listGainers, TEXT.emptySymbol);
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
    } catch (error) {
      renderError(listGainers, error && error.message);
      setStatus(TEXT.statusError);
      hideChipSkeleton(recentChips);
      hideChipSkeleton(watchlistChips);
    }
  }

  function renderList(container, items, timeframe, invert) {
    if (!container) return;
    clearListState(container);
    if (!items.length) {
      renderEmpty(container, TEXT.emptyList);
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

      if (symbolLink) {
        symbolLink.textContent = item.symbol || '';
        symbolLink.href = item.url || `https://www.tradingview.com/symbols/${item.symbol || ''}/`;
      }
      if (priceEl) {
        priceEl.textContent = typeof item.price === 'number' ? item.price.toFixed(2) : '--';
      }
      if (primaryLabelEl) {
        const fallback = TEXT.timeframes[currentTimeframe] || (langPrefix === 'zh' ? '近1月' : '1M');
        const tfLabel = timeframe ? (langPrefix === 'zh' ? timeframe.label : timeframe.label_en) : '';
        const itemLabel = langPrefix === 'zh' ? item.period_label : item.period_label_en;
        primaryLabelEl.textContent = itemLabel || tfLabel || fallback;
      }
      if (primaryEl) {
        primaryEl.textContent = formatChange(item.change_pct_period);
        applyChangeState(primaryEl, item.change_pct_period, invert);
      }
      if (dayEl) {
        dayEl.textContent = formatChange(item.change_pct_day);
        applyChangeState(dayEl, item.change_pct_day, invert, true);
      }
      if (windowLabel) {
        const fallback = TEXT.timeframes[currentTimeframe] || (langPrefix === 'zh' ? '近1月' : '1M');
        const tfWindow = timeframe ? (langPrefix === 'zh' ? timeframe.label : timeframe.label_en) : '';
        const itemWindow = langPrefix === 'zh' ? item.period_label : item.period_label_en;
        windowLabel.textContent = itemWindow || tfWindow || fallback;
      }
      if (updatedEl) {
        const timestamps = Array.isArray(item.timestamps) ? item.timestamps : [];
        const stamp = timestamps.length ? timestamps[timestamps.length - 1] : '';
        updatedEl.textContent = stamp ? `${TEXT.updatedLabel} ${stamp}` : '';
      }
      if (canvas) {
        drawSparkline(canvas, item.series || [], invert);
      }
      container.appendChild(fragment);
    });
  }

  attachChipHandler(recentChips, { allowRemove: true, onRemove: (symbol) => requestRecentAction('delete', symbol) });
  attachChipHandler(watchlistChips, { allowRemove: true, watch: true });

  loadData();
})();
