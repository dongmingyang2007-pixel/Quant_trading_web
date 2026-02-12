(function () {
  'use strict';

  const FIRST_PARTY_HINT = '/static/trading/js/';

  function isFirstPartySource(filename) {
    if (!filename) return false;
    const source = String(filename);
    return source.includes(FIRST_PARTY_HINT) || source.startsWith(window.location.origin);
  }

  function shouldIgnore(message, filename) {
    const text = String(message || '').toLowerCase();
    if (text.includes('document.body.scrollheight') && !isFirstPartySource(filename)) {
      return true;
    }
    return false;
  }

  function emit(level, message) {
    if (!window.BacktestUiToast) return;
    if (level === 'warn') {
      window.BacktestUiToast.warn(message);
    } else {
      window.BacktestUiToast.error(message);
    }
  }

  window.addEventListener('error', function (event) {
    const message = event && event.message ? String(event.message) : 'Unknown runtime error';
    const filename = event && event.filename ? String(event.filename) : '';

    if (shouldIgnore(message, filename)) {
      event.preventDefault();
      return;
    }

    if (!isFirstPartySource(filename)) {
      return;
    }

    emit('error', '前端异常：' + message);
  });

  window.addEventListener('unhandledrejection', function (event) {
    const reason = event && event.reason;
    const text = reason && reason.message ? String(reason.message) : String(reason || 'Unhandled promise rejection');
    emit('warn', '请求异常：' + text);
  });
})();
