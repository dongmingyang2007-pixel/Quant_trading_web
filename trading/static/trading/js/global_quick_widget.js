(() => {
  const root = document.getElementById('global-quick-widget');
  if (!root) return;

  const panel = root.querySelector('.gqw-panel');
  const toggles = Array.prototype.slice.call(root.querySelectorAll('[data-role="gqw-toggle"]'));
  const dragHandle = root.querySelector('[data-role="gqw-drag-handle"]');
  const tabs = Array.prototype.slice.call(root.querySelectorAll('[data-role="gqw-tab"]'));
  const watchPanel = root.querySelector('[data-role="gqw-watch-panel"]');
  const recentPanel = root.querySelector('[data-role="gqw-recent-panel"]');
  const watchCount = root.querySelector('[data-role="gqw-watch-count"]');
  const recentCount = root.querySelector('[data-role="gqw-recent-count"]');
  const form = root.querySelector('[data-role="gqw-form"]');
  const input = root.querySelector('[data-role="gqw-input"]');

  const marketUrl = root.dataset.marketUrl || '/market/insights/';
  const lang = (root.dataset.lang || '').toLowerCase();
  const isZh = lang.startsWith('zh');

  const KEY_COLLAPSED = 'market.globalQuick.collapsed';
  const KEY_POSITION = 'market.globalQuick.position';
  const KEY_ACTIVE_TAB = 'market.globalQuick.activeTab';
  const KEY_WATCH = 'market.globalQuick.watchlist';
  const KEY_RECENT = 'market.globalQuick.recent';
  const KEY_LAUNCH = 'market.globalQuick.launchSymbol';

  let activeTab = 'watchlist';
  let dragState = null;
  let justDragged = false;
  const supportsPointer = typeof window.PointerEvent === 'function';

  function readStorage(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (error) {
      return null;
    }
  }

  function writeStorage(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (error) {
      return;
    }
  }

  function parseList(raw) {
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      const seen = new Set();
      const result = [];
      parsed.forEach((item) => {
        const symbol = normalizeSymbol(item);
        if (!symbol || seen.has(symbol)) return;
        seen.add(symbol);
        result.push(symbol);
      });
      return result.slice(0, 80);
    } catch (error) {
      return [];
    }
  }

  function normalizeSymbol(value) {
    const symbol = String(value || '').trim().toUpperCase();
    if (!symbol) return '';
    if (!/^[A-Z0-9.\-_]{1,16}$/.test(symbol)) return '';
    return symbol;
  }

  function clampPosition(x, y) {
    const width = root.offsetWidth || 64;
    const height = root.offsetHeight || 56;
    const maxX = Math.max(window.innerWidth - width - 8, 8);
    const maxY = Math.max(window.innerHeight - height - 8, 8);
    return {
      x: Math.min(Math.max(Math.round(x), 8), maxX),
      y: Math.min(Math.max(Math.round(y), 8), maxY),
    };
  }

  function applyPosition(x, y, { persist = true } = {}) {
    const pos = clampPosition(x, y);
    root.style.left = `${pos.x}px`;
    root.style.top = `${pos.y}px`;
    root.style.right = 'auto';
    root.style.bottom = 'auto';
    if (persist) {
      writeStorage(KEY_POSITION, JSON.stringify(pos));
    }
  }

  function restorePosition() {
    const raw = readStorage(KEY_POSITION);
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (parsed && Number.isFinite(parsed.x) && Number.isFinite(parsed.y)) {
          applyPosition(parsed.x, parsed.y, { persist: false });
          return;
        }
      } catch (error) {
        // ignore malformed stored value
      }
    }
    const defaultX = window.innerWidth - (root.offsetWidth || 64) - 24;
    const defaultY = window.innerHeight - (root.offsetHeight || 56) - 24;
    applyPosition(defaultX, defaultY, { persist: false });
  }

  function setCollapsed(collapsed, { persist = true } = {}) {
    const next = Boolean(collapsed);
    root.classList.toggle('is-collapsed', next);
    if (panel) {
      panel.hidden = next;
    }
    toggles.forEach((button) => {
      button.setAttribute('aria-expanded', next ? 'false' : 'true');
    });
    if (!next && input) {
      window.requestAnimationFrame(() => {
        if (typeof input.focus === 'function') {
          input.focus({ preventScroll: true });
        }
      });
    }
    if (persist) {
      writeStorage(KEY_COLLAPSED, next ? '1' : '0');
    }
    restorePosition();
  }

  function renderChips(container, symbols, emptyText) {
    if (!container) return;
    container.innerHTML = '';
    if (!symbols.length) {
      const empty = document.createElement('div');
      empty.className = 'gqw-empty';
      empty.textContent = emptyText;
      container.appendChild(empty);
      return;
    }
    symbols.forEach((symbol) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'gqw-chip';
      chip.dataset.symbol = symbol;
      chip.textContent = symbol;
      container.appendChild(chip);
    });
  }

  function renderData() {
    const watch = parseList(readStorage(KEY_WATCH));
    const recent = parseList(readStorage(KEY_RECENT));

    if (watchCount) watchCount.textContent = String(watch.length);
    if (recentCount) recentCount.textContent = String(recent.length);

    renderChips(
      watchPanel,
      watch,
      isZh ? '还没有自选股。' : 'No watchlist symbols yet.'
    );
    renderChips(
      recentPanel,
      recent,
      isZh ? '还没有最近记录。' : 'No recent symbols yet.'
    );
  }

  function setActiveTab(nextTab, { persist = true } = {}) {
    activeTab = nextTab === 'recent' ? 'recent' : 'watchlist';
    tabs.forEach((button) => {
      const isActive = button.dataset.target === activeTab;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    if (watchPanel) watchPanel.hidden = activeTab !== 'watchlist';
    if (recentPanel) recentPanel.hidden = activeTab !== 'recent';
    if (persist) {
      writeStorage(KEY_ACTIVE_TAB, activeTab);
    }
  }

  function openSymbol(symbol) {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) return;
    writeStorage(KEY_LAUNCH, normalized);
    window.location.href = marketUrl;
  }

  function beginDrag(clientX, clientY) {
    if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;
    dragState = {
      startX: clientX,
      startY: clientY,
      originLeft: root.offsetLeft,
      originTop: root.offsetTop,
      moved: false,
    };
    root.classList.add('is-dragging');
    if (supportsPointer) {
      window.addEventListener('pointermove', onPointerDragMove, { passive: false });
      window.addEventListener('pointerup', onAnyDragEnd, { passive: true });
      window.addEventListener('pointercancel', onAnyDragEnd, { passive: true });
    } else {
      window.addEventListener('mousemove', onMouseDragMove, { passive: false });
      window.addEventListener('mouseup', onAnyDragEnd, { passive: true });
      window.addEventListener('touchmove', onTouchDragMove, { passive: false });
      window.addEventListener('touchend', onAnyDragEnd, { passive: true });
      window.addEventListener('touchcancel', onAnyDragEnd, { passive: true });
    }
  }

  function onDragStart(event) {
    if (event.button !== undefined && event.button !== 0) return;
    if (!(event.target instanceof Element)) return;
    beginDrag(event.clientX, event.clientY);
  }

  function applyDragMove(clientX, clientY) {
    if (!dragState) return;
    const deltaX = clientX - dragState.startX;
    const deltaY = clientY - dragState.startY;
    if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {
      dragState.moved = true;
    }
    applyPosition(dragState.originLeft + deltaX, dragState.originTop + deltaY, { persist: false });
  }

  function onPointerDragMove(event) {
    event.preventDefault();
    applyDragMove(event.clientX, event.clientY);
  }

  function onMouseDragMove(event) {
    event.preventDefault();
    applyDragMove(event.clientX, event.clientY);
  }

  function onTouchDragMove(event) {
    if (!event.touches || !event.touches.length) return;
    event.preventDefault();
    const touch = event.touches[0];
    applyDragMove(touch.clientX, touch.clientY);
  }

  function onAnyDragEnd() {
    if (!dragState) return;
    if (dragState.moved) {
      justDragged = true;
      window.setTimeout(() => {
        justDragged = false;
      }, 120);
      writeStorage(KEY_POSITION, JSON.stringify({
        x: root.offsetLeft,
        y: root.offsetTop,
      }));
    }
    dragState = null;
    root.classList.remove('is-dragging');
    window.removeEventListener('pointermove', onPointerDragMove);
    window.removeEventListener('pointerup', onAnyDragEnd);
    window.removeEventListener('pointercancel', onAnyDragEnd);
    window.removeEventListener('mousemove', onMouseDragMove);
    window.removeEventListener('mouseup', onAnyDragEnd);
    window.removeEventListener('touchmove', onTouchDragMove);
    window.removeEventListener('touchend', onAnyDragEnd);
    window.removeEventListener('touchcancel', onAnyDragEnd);
  }

  toggles.forEach((button) => {
    button.addEventListener('click', (event) => {
      if (justDragged) {
        event.preventDefault();
        return;
      }
      setCollapsed(!root.classList.contains('is-collapsed'));
    });
  });

  tabs.forEach((button) => {
    button.addEventListener('click', () => {
      setActiveTab(button.dataset.target || 'watchlist');
    });
  });

  [watchPanel, recentPanel].forEach((container) => {
    if (!container) return;
    container.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const chip = target.closest('.gqw-chip');
      if (!chip) return;
      const symbol = chip.dataset.symbol || '';
      openSymbol(symbol);
    });
  });

  if (form) {
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (!input) return;
      openSymbol(input.value);
    });
  }

  if (dragHandle) {
    if (supportsPointer) {
      dragHandle.addEventListener('pointerdown', onDragStart);
    } else {
      dragHandle.addEventListener('mousedown', onDragStart);
      dragHandle.addEventListener('touchstart', (event) => {
        if (!event.touches || !event.touches.length) return;
        beginDrag(event.touches[0].clientX, event.touches[0].clientY);
      }, { passive: true });
    }
  }
  const launch = root.querySelector('.gqw-launch');
  if (launch) {
    if (supportsPointer) {
      launch.addEventListener('pointerdown', onDragStart);
    } else {
      launch.addEventListener('mousedown', onDragStart);
      launch.addEventListener('touchstart', (event) => {
        if (!event.touches || !event.touches.length) return;
        beginDrag(event.touches[0].clientX, event.touches[0].clientY);
      }, { passive: true });
    }
  }

  window.addEventListener('resize', () => {
    restorePosition();
  });
  window.addEventListener('blur', () => {
    onAnyDragEnd();
  });

  window.addEventListener('storage', (event) => {
    if (!event.key) return;
    if (event.key === KEY_WATCH || event.key === KEY_RECENT) {
      renderData();
    }
  });

  window.addEventListener('market:global-quick-updated', () => {
    renderData();
  });

  const savedCollapsed = (readStorage(KEY_COLLAPSED) || '1').toLowerCase();
  setCollapsed(savedCollapsed === '1' || savedCollapsed === 'true', { persist: false });
  setActiveTab(readStorage(KEY_ACTIVE_TAB) || 'watchlist', { persist: false });
  renderData();
  restorePosition();
})();
