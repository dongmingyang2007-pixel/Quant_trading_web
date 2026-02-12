(function () {
  'use strict';

  function ensureContainer() {
    let node = document.getElementById('backtest-toast-root');
    if (!node) {
      node = document.createElement('div');
      node.id = 'backtest-toast-root';
      node.className = 'backtest-toast-root';
      document.body.appendChild(node);
    }
    return node;
  }

  function notify(message, level, timeoutMs) {
    if (!message) return;
    const container = ensureContainer();
    const toast = document.createElement('div');
    toast.className = 'backtest-toast backtest-toast-' + String(level || 'info');
    toast.textContent = String(message);
    container.appendChild(toast);

    window.setTimeout(function () {
      toast.classList.add('is-leaving');
      window.setTimeout(function () {
        toast.remove();
      }, 220);
    }, Math.max(1200, Number(timeoutMs || 2800)));
  }

  window.BacktestUiToast = {
    info: function (message, timeoutMs) {
      notify(message, 'info', timeoutMs);
    },
    success: function (message, timeoutMs) {
      notify(message, 'success', timeoutMs);
    },
    warn: function (message, timeoutMs) {
      notify(message, 'warn', timeoutMs);
    },
    error: function (message, timeoutMs) {
      notify(message, 'error', timeoutMs);
    },
  };
})();
