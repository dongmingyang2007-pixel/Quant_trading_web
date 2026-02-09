(() => {
  const root = typeof globalThis !== 'undefined' ? globalThis : window;

  const safeString = (value) => (value === null || value === undefined ? '' : String(value)).trim();

  const buildRankingKey = (item) => {
    if (!item || typeof item !== 'object') return '';
    const symbol = safeString(item.symbol).toUpperCase();
    if (!symbol) return '';
    const exchange = safeString(item.exchange).toUpperCase();
    return exchange ? `${symbol}|${exchange}` : symbol;
  };

  const dedupeRankItems = (items, getKey = buildRankingKey) => {
    const result = [];
    const duplicates = [];
    const seen = new Set();
    if (!Array.isArray(items)) {
      return { items: result, duplicates };
    }
    items.forEach((item) => {
      const key = getKey(item);
      if (!key) return;
      if (seen.has(key)) {
        duplicates.push(item);
        return;
      }
      seen.add(key);
      result.push(item);
    });
    return { items: result, duplicates };
  };

  const mergeRankItems = (existing, incoming, getKey = buildRankingKey) => {
    const merged = [];
    const appended = [];
    const seen = new Set();
    if (Array.isArray(existing)) {
      existing.forEach((item) => {
        const key = getKey(item);
        if (!key || seen.has(key)) return;
        seen.add(key);
        merged.push(item);
      });
    }
    if (Array.isArray(incoming)) {
      incoming.forEach((item) => {
        const key = getKey(item);
        if (!key || seen.has(key)) return;
        seen.add(key);
        merged.push(item);
        appended.push(item);
      });
    }
    return { merged, appended };
  };

  const toggleSortState = (state) => {
    if (state === 'default') return 'desc';
    if (state === 'desc') return 'asc';
    return 'default';
  };

  const parseTimestamp = (value) => {
    if (value == null) return null;
    if (typeof value === 'number' && Number.isFinite(value)) {
      let ts = value;
      if (ts > 1e15) ts = ts / 1e9;
      else if (ts > 1e12) ts = ts / 1e3;
      return new Date(ts * 1000);
    }
    const raw = safeString(value);
    if (!raw) return null;
    if (/^\d+(\.\d+)?$/.test(raw)) {
      const num = Number.parseFloat(raw);
      if (!Number.isFinite(num)) return null;
      let ts = num;
      if (ts > 1e15) ts = ts / 1e9;
      else if (ts > 1e12) ts = ts / 1e3;
      return new Date(ts * 1000);
    }
    let normalized = raw;
    if (normalized.endsWith('UTC')) {
      normalized = normalized.replace(' UTC', 'Z');
    }
    if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}/.test(normalized) && !normalized.includes('T')) {
      normalized = normalized.replace(' ', 'T');
    }
    if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(normalized)) {
      normalized = `${normalized}Z`;
    }
    const parsed = new Date(normalized);
    if (Number.isNaN(parsed.getTime())) return null;
    return parsed;
  };

  const formatTimestamp = (value, tzMode, locale) => {
    const date = parseTimestamp(value);
    if (!date) return safeString(value);
    const options = {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    };
    if (tzMode === 'utc') {
      options.timeZone = 'UTC';
    }
    const formatter = new Intl.DateTimeFormat(locale || undefined, options);
    const formatted = formatter.format(date);
    return tzMode === 'utc' ? `${formatted} UTC` : formatted;
  };

  const loadPreference = (key, fallback, storage = null) => {
    const store = storage || root.localStorage;
    if (!store) return fallback;
    try {
      const raw = store.getItem(key);
      return raw === null || raw === '' ? fallback : raw;
    } catch (error) {
      return fallback;
    }
  };

  const savePreference = (key, value, storage = null) => {
    const store = storage || root.localStorage;
    if (!store) return;
    try {
      store.setItem(key, String(value));
    } catch (error) {
      return;
    }
  };

  root.MarketInsightsState = {
    buildRankingKey,
    dedupeRankItems,
    mergeRankItems,
    toggleSortState,
    parseTimestamp,
    formatTimestamp,
    loadPreference,
    savePreference,
  };
})();
