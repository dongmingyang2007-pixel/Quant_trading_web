(function () {
  'use strict';

  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"], meta[name="csrfmiddlewaretoken"]');
    if (meta && meta.content) {
      return meta.content;
    }
    const input = document.querySelector('input[name="csrfmiddlewaretoken"]');
    if (input && input.value) {
      return input.value;
    }
    return getCookie('csrftoken');
  }

  function getCookie(name) {
    const cookies = document.cookie ? document.cookie.split(';') : [];
    for (let i = 0; i < cookies.length; i += 1) {
      const cookie = cookies[i].trim();
      if (cookie.startsWith(name + '=')) {
        return decodeURIComponent(cookie.slice(name.length + 1));
      }
    }
    return '';
  }

  function buildChecksText(checks) {
    if (!checks || typeof checks !== 'object') {
      return '';
    }
    const parts = [];
    Object.keys(checks).forEach(function (key) {
      const item = checks[key];
      const ok = item && item.ok;
      const detail =
        item && (item.detail || item.error_code || (typeof item.count === 'number' ? String(item.count) : ''));
      parts.push(key + ':' + (ok ? 'ok' : detail || 'fail'));
    });
    return parts.join(' | ');
  }

  function parseJsonSafely(raw) {
    if (!raw) {
      return {};
    }
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        return parsed;
      }
    } catch (_err) {
      return { raw: raw };
    }
    return {};
  }

  function buildErrorMessage(payload, provider, statusCode) {
    if (payload && payload.message) return String(payload.message);
    if (payload && payload.error) return String(payload.error);
    if (payload && payload.detail) return String(payload.detail);
    if (payload && payload.error_code) return String(payload.error_code);
    if (statusCode === 401) return '登录已失效，请重新登录后再试。';
    if (statusCode === 403) return 'CSRF 校验失败，请刷新页面后重试。';
    if (statusCode === 405) return '接口只支持 POST 诊断请求。';
    if (statusCode >= 500) return '服务异常，请稍后重试。';
    if (statusCode >= 400) return '请求失败，请检查设置后重试。';
    return provider === 'massive' ? 'Massive 诊断失败' : 'Alpaca 诊断失败';
  }

  document.addEventListener('DOMContentLoaded', function () {
    const button = document.querySelector('[data-role="provider-diagnose"]');
    if (!button) return;

    const result = document.querySelector('[data-role="provider-diagnose-result"]');
    const providerSelect = document.querySelector('#id_market_data_provider');

    button.addEventListener('click', function () {
      const endpoint = button.dataset.diagnoseEndpoint;
      if (!endpoint) {
        return;
      }
      const provider = providerSelect && providerSelect.value ? providerSelect.value : 'alpaca';
      const payload = {
        provider: provider,
        check_news: true,
        check_ws: true,
      };
      button.disabled = true;
      if (result) {
        result.textContent = provider === 'massive' ? '正在检测 Massive ...' : '正在检测 Alpaca ...';
      }

      fetch(endpoint, {
        method: 'POST',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
          'X-CSRFToken': getCsrfToken(),
          'X-Requested-With': 'XMLHttpRequest',
        },
        body: JSON.stringify(payload),
      })
        .then(function (resp) {
          return resp.text().then(function (raw) {
            return {
              statusCode: resp.status,
              ok: resp.ok,
              data: parseJsonSafely(raw),
            };
          });
        })
        .then(function (parsed) {
          if (!result) return;
          const data = parsed.data || {};
          const providerLabel = String(data.provider || provider).toUpperCase();
          const checksText = buildChecksText(data.checks);
          const message = parsed.ok
            ? String(data.message || '连接可用')
            : buildErrorMessage(data, provider, parsed.statusCode);
          result.textContent = providerLabel + ' · ' + message + (checksText ? ' · ' + checksText : '');
          const success = parsed.ok && !!data.ok;
          result.classList.remove('text-muted');
          result.classList.toggle('text-danger', !success);
          result.classList.toggle('text-success', success);
        })
        .catch(function (error) {
          if (result) {
            result.textContent = error && error.message ? error.message : '诊断失败';
            result.classList.remove('text-muted');
            result.classList.add('text-danger');
            result.classList.remove('text-success');
          }
        })
        .finally(function () {
          button.disabled = false;
        });
    });
  });
})();
