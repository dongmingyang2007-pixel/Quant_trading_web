(function () {
  'use strict';

  function render(root, payload) {
    if (!root || !payload || !payload.backtest) return;
    root.querySelectorAll('[data-backtest-summary="latest-ticker"]').forEach(function (node) {
      node.textContent = (payload.backtest.latest_run || {}).ticker || '-';
    });
  }

  window.BacktestWorkspaceBacktest = {
    render: render,
  };
})();
