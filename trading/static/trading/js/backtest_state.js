(function () {
  'use strict';

  function createStore(initialState) {
    const state = Object.assign(
      {
        workspace: 'trade',
        trade: null,
        backtest: null,
        review: null,
        tradingModeInfo: null,
      },
      initialState || {}
    );
    const listeners = new Set();

    function notify() {
      listeners.forEach(function (listener) {
        try {
          listener(state);
        } catch (_error) {
          // ignore listener errors
        }
      });
    }

    return {
      getState: function () {
        return state;
      },
      setState: function (patch) {
        if (!patch || typeof patch !== 'object') return;
        Object.assign(state, patch);
        notify();
      },
      subscribe: function (listener) {
        if (typeof listener !== 'function') return function () {};
        listeners.add(listener);
        return function () {
          listeners.delete(listener);
        };
      },
    };
  }

  window.BacktestState = {
    createStore: createStore,
  };
})();
