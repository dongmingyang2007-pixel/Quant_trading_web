(() => {
  const gridEl = document.getElementById('screener-grid');
  if (!gridEl || typeof Tabulator === 'undefined') return;

  const searchInput = document.getElementById('screener-search-input');
  const volumeSelect = document.getElementById('screener-volume-select');
  const chartTitle = document.querySelector('.chart-title');
  const chartKicker = document.querySelector('.chart-kicker');
  const symbolInput = document.getElementById('terminal-symbol-input');

  const getCSRFToken = () => {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta && meta.getAttribute('content')) return meta.getAttribute('content');
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? match[1] : '';
  };

  const updateMainSymbol = (symbol) => {
    const normalized = (symbol || '').toString().trim().toUpperCase();
    if (!normalized) return;
    if (chartTitle) {
      chartTitle.textContent = normalized;
    }
    if (chartKicker) {
      chartKicker.textContent = 'Market Pulse';
    }
    if (symbolInput) {
      symbolInput.value = normalized;
    }
    document.dispatchEvent(new CustomEvent('dashboard:symbol', { detail: { symbol: normalized } }));
  };

  const notifyWatchlistRefresh = () => {
    document.dispatchEvent(new Event('watchlist:refresh'));
    if (window.watchlistRealtime && typeof window.watchlistRealtime.refresh === 'function') {
      window.watchlistRealtime.refresh();
    }
  };

  const addToWatchlist = async (symbol) => {
    const normalized = (symbol || '').toString().trim().toUpperCase();
    if (!normalized) return;
    const response = await fetch('/api/market/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCSRFToken(),
        'X-Requested-With': 'XMLHttpRequest',
      },
      credentials: 'same-origin',
      body: JSON.stringify({ watch: 'add', query: normalized }),
    });
    if (!response.ok) {
      return;
    }
    notifyWatchlistRefresh();
  };

  const priceFormatter = (cell) => {
    const value = cell.getValue();
    if (!Number.isFinite(value)) return '--';
    return value.toFixed(2);
  };

  const changeFormatter = (cell) => {
    const value = cell.getValue();
    if (!Number.isFinite(value)) return '--';
    const sign = value > 0 ? '+' : '';
    const className = value > 0 ? 'is-up' : value < 0 ? 'is-down' : 'is-flat';
    return `<span class="${className}">${sign}${value.toFixed(2)}%</span>`;
  };

  const volumeFormatter = (cell) => {
    const value = cell.getValue();
    if (!Number.isFinite(value)) return '--';
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
    return value.toString();
  };

  const actionFormatter = () => {
    return '<button class="screener-action-btn" type="button">+</button>';
  };

  const table = new Tabulator(gridEl, {
    ajaxURL: '/api/market/screener/',
    ajaxConfig: 'GET',
    layout: 'fitColumns',
    height: '100%',
    pagination: true,
    paginationMode: 'remote',
    paginationSize: 25,
    paginationSizeSelector: [25, 50, 100],
    filterMode: 'remote',
    sortMode: 'remote',
    movableColumns: true,
    placeholder: '暂无筛选数据',
    ajaxParams: () => ({
      search: searchInput ? searchInput.value.trim() : '',
      min_volume: volumeSelect ? volumeSelect.value : '',
    }),
    columns: [
      {
        title: '',
        field: 'actions',
        width: 52,
        headerSort: false,
        hozAlign: 'center',
        formatter: actionFormatter,
        cellClick: async (e, cell) => {
          e.stopPropagation();
          const data = cell.getRow().getData();
          updateMainSymbol(data.ticker);
          await addToWatchlist(data.ticker);
        },
      },
      {
        title: '代码',
        field: 'ticker',
        headerFilter: 'input',
        sorter: 'string',
        width: 100,
      },
      {
        title: '名称',
        field: 'name',
        headerFilter: 'input',
        sorter: 'string',
        widthGrow: 2,
      },
      {
        title: '价格',
        field: 'price',
        sorter: 'number',
        hozAlign: 'right',
        formatter: priceFormatter,
        width: 110,
      },
      {
        title: '涨跌%',
        field: 'change_pct',
        sorter: 'number',
        hozAlign: 'right',
        formatter: changeFormatter,
        width: 110,
      },
      {
        title: '成交量',
        field: 'volume',
        sorter: 'number',
        hozAlign: 'right',
        formatter: volumeFormatter,
        headerFilter: 'input',
        width: 120,
      },
      {
        title: '行业',
        field: 'industry',
        headerFilter: 'input',
        sorter: 'string',
        widthGrow: 1,
      },
    ],
  });

  const refreshTable = () => {
    if (!table) return;
    table.setPage(1);
  };

  if (searchInput) {
    let timer = null;
    searchInput.addEventListener('input', () => {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(refreshTable, 250);
    });
  }

  if (volumeSelect) {
    volumeSelect.addEventListener('change', () => {
      refreshTable();
    });
  }

  table.on('rowClick', (_event, row) => {
    const data = row.getData();
    updateMainSymbol(data.ticker);
  });

  const tabs = document.querySelectorAll('.bottom-tab[data-pane-target]');
  const panes = document.querySelectorAll('.terminal-bottom-pane[data-pane]');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.getAttribute('data-pane-target');
      if (!target) return;
      tabs.forEach((btn) => btn.classList.toggle('is-active', btn === tab));
      panes.forEach((pane) => pane.classList.toggle('is-active', pane.getAttribute('data-pane') === target));
    });
  });
})();
