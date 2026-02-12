(function () {
  'use strict';

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

  function classifyHttpError(statusCode, payload) {
    const code = String((payload && (payload.error_code || payload.error)) || '').toLowerCase();
    if (statusCode === 401 || statusCode === 403) {
      return {
        code: code || 'auth_or_csrf_failed',
        message: '登录状态或 CSRF 已失效，请刷新后重试。',
      };
    }
    if (statusCode === 405) {
      return {
        code: code || 'method_not_allowed',
        message: '方法不允许，请检查接口调用方式。',
      };
    }
    if (statusCode >= 500) {
      return {
        code: code || 'server_error',
        message: '服务暂时异常，请稍后重试。',
      };
    }
    return {
      code: code || 'request_failed',
      message: String((payload && (payload.message || payload.detail || payload.error)) || '请求失败'),
    };
  }

  function request(url, options) {
    const requestOptions = Object.assign(
      {
        method: 'GET',
        credentials: 'same-origin',
        headers: {
          Accept: 'application/json',
        },
      },
      options || {}
    );

    return fetch(url, requestOptions).then(function (response) {
      return response
        .json()
        .catch(function () {
          return {};
        })
        .then(function (payload) {
          if (!response.ok) {
            const info = classifyHttpError(response.status, payload);
            const error = new Error(info.message);
            error.httpStatus = response.status;
            error.errorCode = info.code;
            error.payload = payload;
            throw error;
          }
          return payload;
        });
    });
  }

  function createClient(config) {
    const workbenchEndpoint = String((config && config.workbenchEndpoint) || '');
    const modeEndpoint = String((config && config.modeEndpoint) || '');
    const orderEndpoint = String((config && config.orderEndpoint) || '');
    const csrfToken = String((config && config.csrfToken) || getCookie('csrftoken'));

    return {
      getWorkbench: function (workspace) {
        const endpoint = new URL(workbenchEndpoint, window.location.origin);
        endpoint.searchParams.set('workspace', workspace || 'trade');
        return request(endpoint.toString(), { method: 'GET' });
      },
      getTradingModeInfo: function () {
        return request(modeEndpoint, { method: 'GET' });
      },
      setTradingMode: function (body) {
        return request(modeEndpoint, {
          method: 'POST',
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken,
          },
          body: JSON.stringify(body || {}),
        });
      },
      submitManualOrder: function (body) {
        return request(orderEndpoint, {
          method: 'POST',
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken,
          },
          body: JSON.stringify(body || {}),
        });
      },
    };
  }

  window.BacktestApi = {
    createClient: createClient,
  };
})();
