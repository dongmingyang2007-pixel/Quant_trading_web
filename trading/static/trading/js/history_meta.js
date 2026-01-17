(() => {
  const accordions = document.querySelectorAll('.history-accordion');
  if (!accordions.length) return;

  const langIsZh = (document.documentElement.lang || '').toLowerCase().startsWith('zh');
  const TEXT = langIsZh
    ? {
        saving: '正在保存...',
        saved: '已保存',
        failed: '保存失败',
      }
    : {
        saving: 'Saving...',
        saved: 'Saved',
        failed: 'Save failed',
      };

  const getCsrfToken = () => {
    const input = document.querySelector('input[name="csrfmiddlewaretoken"]');
    if (input) return input.value;
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta) return meta.getAttribute('content') || '';
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? decodeURIComponent(match[1]) : '';
  };

  const parseTags = (raw) => {
    if (!raw) return [];
    return raw
      .replace(/，/g, ',')
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean);
  };

  const normalizeText = (value) => (value || '').toString().toLowerCase();

  const buildMetaUrl = (template, recordId) => {
    if (!template || !recordId) return '';
    return template.replace('RECORD_ID', encodeURIComponent(recordId));
  };

  const ensureTagOptions = (selectEl, tags) => {
    if (!selectEl || !tags.length) return;
    const existing = new Set(
      Array.from(selectEl.options).map((option) => normalizeText(option.value || option.textContent))
    );
    tags.forEach((tag) => {
      const normalized = normalizeText(tag);
      if (!normalized || existing.has(normalized)) return;
      const option = document.createElement('option');
      option.value = tag;
      option.textContent = tag;
      selectEl.appendChild(option);
      existing.add(normalized);
    });
  };

  accordions.forEach((accordion) => {
    const panel = accordion.parentElement || document;
    const controls = panel.querySelector('[data-role="history-controls"]');
    const searchInput = controls && controls.querySelector('[data-role="history-search"]');
    const tagFilter = controls && controls.querySelector('[data-role="history-tag-filter"]');
    const sortSelect = controls && controls.querySelector('[data-role="history-sort"]');
    const starredFilter = controls && controls.querySelector('[data-role="history-star-filter"]');
    const emptyId = accordion.dataset.emptyId;
    const emptyEl = emptyId ? document.getElementById(emptyId) : null;
    const metaUrlTemplate = accordion.dataset.metaUrlTemplate || '';
    const metaSavedMessage = accordion.dataset.metaSaved || TEXT.saved;
    const metaFailedMessage = accordion.dataset.metaFailed || TEXT.failed;

    const items = Array.from(accordion.querySelectorAll('.history-item'));

    const getItemData = (item) => {
      const tags = parseTags(item.dataset.tags || '');
      return {
        title: item.dataset.title || '',
        ticker: item.dataset.ticker || '',
        engine: item.dataset.engine || '',
        notes: item.dataset.notes || '',
        tags,
        starred: item.dataset.starred === 'true',
        timestamp: Date.parse(item.dataset.timestamp || '') || 0,
      };
    };

    const syncSummary = (item, titleText) => {
      const titleEl = item.querySelector('[data-role="history-title-text"]');
      const tickerEl = item.querySelector('[data-role="history-ticker"]');
      const ticker = item.dataset.ticker || '';
      const finalTitle = titleText || ticker;
      if (titleEl) titleEl.textContent = finalTitle;
      if (tickerEl) {
        if (titleText) {
          tickerEl.textContent = ticker;
          tickerEl.classList.remove('d-none');
        } else {
          tickerEl.classList.add('d-none');
        }
      }
    };

    const applyFilters = () => {
      const query = normalizeText(searchInput && searchInput.value);
      const tagValue = normalizeText(tagFilter && tagFilter.value);
      const starredOnly = Boolean(starredFilter && starredFilter.checked);
      const sortMode = (sortSelect && sortSelect.value) || 'newest';

      const sorted = items.slice().sort((a, b) => {
        const aData = getItemData(a);
        const bData = getItemData(b);
        if (sortMode === 'oldest') return aData.timestamp - bData.timestamp;
        if (sortMode === 'starred') {
          if (aData.starred !== bData.starred) return aData.starred ? -1 : 1;
          return bData.timestamp - aData.timestamp;
        }
        return bData.timestamp - aData.timestamp;
      });

      sorted.forEach((item) => accordion.appendChild(item));

      let visibleCount = 0;
      items.forEach((item) => {
        const data = getItemData(item);
        const searchText = normalizeText(
          [data.title, data.ticker, data.engine, data.notes, data.tags.join(' ')].join(' ')
        );
        const matchesQuery = !query || searchText.includes(query);
        const matchesTag = !tagValue || data.tags.map(normalizeText).includes(tagValue);
        const matchesStarred = !starredOnly || data.starred;
        const shouldShow = matchesQuery && matchesTag && matchesStarred;
        item.classList.toggle('d-none', !shouldShow);
        if (shouldShow) visibleCount += 1;
      });
      if (emptyEl) {
        emptyEl.classList.toggle('d-none', visibleCount > 0);
      }
    };

    if (searchInput) {
      searchInput.addEventListener('input', applyFilters);
    }
    if (tagFilter) {
      tagFilter.addEventListener('change', applyFilters);
    }
    if (sortSelect) {
      sortSelect.addEventListener('change', applyFilters);
    }
    if (starredFilter) {
      starredFilter.addEventListener('change', applyFilters);
    }

    accordion.addEventListener('click', async (event) => {
      const saveBtn = event.target.closest('[data-role="history-meta-save"]');
      const starBtn = event.target.closest('[data-role="history-star"]');

      if (!saveBtn && !starBtn) return;
      const item = event.target.closest('.history-item');
      if (!item) return;
      const recordId = item.dataset.recordId;
      const url = buildMetaUrl(metaUrlTemplate, recordId);
      if (!url) return;

      const statusEl = item.querySelector('[data-role="history-meta-status"]');
      if (statusEl) {
        statusEl.textContent = TEXT.saving;
      }

      const payload = { id: recordId };
      if (saveBtn) {
        const titleInput = item.querySelector('[data-role="history-title"]');
        const tagsInput = item.querySelector('[data-role="history-tags"]');
        const notesInput = item.querySelector('[data-role="history-notes"]');
        payload.title = titleInput ? titleInput.value.trim() : '';
        payload.tags = parseTags(tagsInput ? tagsInput.value : '');
        payload.notes = notesInput ? notesInput.value.trim() : '';
      }
      if (starBtn) {
        payload.starred = item.dataset.starred !== 'true';
      }

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken(),
            'X-Requested-With': 'XMLHttpRequest',
          },
          credentials: 'same-origin',
          body: JSON.stringify(payload),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.error || metaFailedMessage);
        }
        if (payload.title !== undefined) {
          item.dataset.title = payload.title || '';
          syncSummary(item, payload.title);
        }
        if (payload.tags !== undefined) {
          item.dataset.tags = payload.tags.join(', ');
          ensureTagOptions(tagFilter, payload.tags);
        }
        if (payload.notes !== undefined) {
          item.dataset.notes = payload.notes || '';
        }
        if (payload.starred !== undefined) {
          const starred = Boolean(payload.starred);
          item.dataset.starred = starred ? 'true' : 'false';
          if (starBtn) {
            starBtn.classList.toggle('is-active', starred);
            starBtn.setAttribute('aria-pressed', starred ? 'true' : 'false');
          }
        }
        if (statusEl) {
          statusEl.textContent = metaSavedMessage;
          window.setTimeout(() => {
            statusEl.textContent = '';
          }, 1800);
        }
        applyFilters();
      } catch (_error) {
        if (statusEl) {
          statusEl.textContent = metaFailedMessage;
        }
      }
    });

    applyFilters();
  });
})();
