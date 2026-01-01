(function () {
    const form = document.getElementById('analysis-form');
    if (!form) return;

    const langIsZh = (form.dataset.lang || document.documentElement.lang || '').toLowerCase().startsWith('zh');
    const defaultsNode = document.getElementById('advanced-strategy-defaults');
    let advancedDefaults = {};
    if (defaultsNode) {
        try {
            advancedDefaults = JSON.parse(defaultsNode.textContent || '{}');
        } catch (error) {
            console.warn('Failed to parse advanced defaults', error);
        }
    }

    let activeDock = window.taskDock || null;
    const pendingDockOps = [];
    const withTaskDock = (fn) => {
        if (activeDock) {
            try {
                fn(activeDock);
            } catch (error) {
                console.error('Task dock update error:', error);
            }
            return;
        }
        pendingDockOps.push(fn);
    };
    const setDock = (dock) => {
        if (!dock) return;
        activeDock = dock;
        while (pendingDockOps.length) {
            const op = pendingDockOps.shift();
            try {
                op(dock);
            } catch (error) {
                console.error('Task dock pending op error:', error);
            }
        }
    };
    if (activeDock) {
        setDock(activeDock);
    }
    window.addEventListener('taskdock:ready', (event) => {
        setDock(event.detail || window.taskDock || null);
    });

    const updateDockTask = (id, payload) =>
        withTaskDock((dock) => {
            if (typeof dock.updateTask === 'function') {
                dock.updateTask(id, payload);
            }
        });
    const removeDockTask = (id) =>
        withTaskDock((dock) => {
            if (typeof dock.removeTask === 'function') {
                dock.removeTask(id);
            }
        });

    const taskEndpoint = form.dataset.taskEndpoint || (typeof TASK_ENDPOINT !== 'undefined' ? TASK_ENDPOINT : '');
    const statusTemplate =
        form.dataset.taskStatusTemplate || (typeof TASK_STATUS_TEMPLATE !== 'undefined' ? TASK_STATUS_TEMPLATE : '');
    const historyBase =
        form.dataset.historyBase ||
        (typeof HISTORY_LOAD_BASE !== 'undefined' && HISTORY_LOAD_BASE) ||
        (typeof BACKTEST_PATH !== 'undefined' ? BACKTEST_PATH : '');
    const csrfInput = form.querySelector('input[name=csrfmiddlewaretoken]');
    const csrfToken = csrfInput ? csrfInput.value : '';
    const TASK_POLL_INTERVAL_MS = 2500;
    let pageUnloading = false;
    window.addEventListener('beforeunload', () => {
        pageUnloading = true;
    });
    window.addEventListener('pagehide', () => {
        pageUnloading = true;
    });

    const resetAdvancedDefaults = () => {
        if (!advancedDefaults || typeof advancedDefaults !== 'object') return;
        Object.entries(advancedDefaults).forEach(([key, value]) => {
            const field = form.querySelector(`[name="${key}"]`);
            if (!field) return;
            if (field.type === 'checkbox') {
                field.checked = Boolean(value);
                field.dispatchEvent(new Event('change', { bubbles: true }));
                return;
            }
            const option = field.tagName === 'SELECT' ? field.querySelector(`option[value="${value}"]`) : null;
            if (option || field.tagName !== 'SELECT') {
                field.value = value ?? '';
                field.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    };
    const resetButton = form.querySelector('[data-role="advanced-reset"]');
    if (resetButton) {
        resetButton.addEventListener('click', () => resetAdvancedDefaults());
    }

    const asyncText = langIsZh
        ? {
              queue: '任务已排队，等待执行…',
              running: '策略引擎运行中…',
              retry: '任务重试中…',
              success: '任务完成，可随时查看报告。',
              failure: '任务执行失败，请重试。',
              submitting: '正在提交回测…',
              loadingBanner: '加载中：有任务执行中…',
          }
        : {
              queue: 'Task queued and awaiting execution…',
              running: 'Strategy engines are running…',
              retry: 'Task is retrying…',
              success: 'Task complete. You can view the report anytime.',
              failure: 'Task failed. Please retry.',
              submitting: 'Submitting backtest…',
              loadingBanner: 'Loading: a task is still running…',
          };

    const buildStatusUrl = (taskId) =>
        statusTemplate ? statusTemplate.replace('TASK_ID_PLACEHOLDER', encodeURIComponent(taskId)) : '';

    const serializeFormParams = () => {
        const data = new FormData(form);
        const params = {};
        data.forEach((value, key) => {
            if (key === 'csrfmiddlewaretoken') return;
            params[key] = value;
        });
        ['start_date', 'end_date'].forEach((k) => {
            if (params[k]) params[k] = String(params[k]).replace(/\//g, '-');
        });
        const todayIso = new Date().toISOString().slice(0, 10);
        if (!params.end_date) params.end_date = todayIso;
        if (!params.start_date) params.start_date = todayIso;
        return params;
    };

    const pollTask = (taskId, startedAt) => {
        let handledByDock = false;
        withTaskDock((dock) => {
            if (typeof dock.pollTask === 'function') {
                dock.pollTask(taskId, startedAt);
                handledByDock = true;
            }
        });
        if (handledByDock) {
            return;
        }
        const statusUrl = buildStatusUrl(taskId);
        if (!statusUrl) return;
        fetch(statusUrl, { headers: { 'X-Requested-With': 'XMLHttpRequest' } })
            .then((res) => res.json().catch(() => ({})))
            .then((data) => {
                const state = data.state || '';
                const meta = data.meta || {};
                const progress = typeof meta.progress === 'number' && !Number.isNaN(meta.progress) ? meta.progress : null;
                if (state === 'SUCCESS') {
                    const historyId = data.result && data.result.history_id;
                    updateDockTask(taskId, {
                        status: 'SUCCESS',
                        message: asyncText.success,
                        link: historyId ? `${historyBase}?history_id=${encodeURIComponent(historyId)}` : null,
                        progress: 100,
                        taskId,
                    });
                    return;
                }
                if (state === 'FAILURE' || state === 'REVOKED') {
                    updateDockTask(taskId, {
                        status: 'FAILURE',
                        message: data.error || asyncText.failure,
                        progress,
                        taskId,
                    });
                    return;
                }
                const runningMsg =
                    state === 'STARTED' || state === 'PROGRESS'
                        ? asyncText.running
                        : state === 'RETRY'
                        ? asyncText.retry
                        : asyncText.queue;
                updateDockTask(taskId, { status: state || 'PENDING', message: runningMsg, progress, taskId });
                // 继续轮询直到拿到终态，避免长任务或页面挂起后状态停留在排队/运行中
                window.setTimeout(() => pollTask(taskId, startedAt), TASK_POLL_INTERVAL_MS);
            })
            .catch(() => {});
    };

    const renderPageBanner = (tasksSnapshot) => {
        const bannerId = 'backtest-page-loading';
        const container = document.getElementById('analysis-form-card') || form;
        if (!container) return;
        const hasRunning = (tasksSnapshot || []).some((t) =>
            ['PENDING', 'STARTED', 'RETRY', 'SUBMIT', ''].includes((t.status || '').toUpperCase())
        );
        let banner = document.getElementById(bannerId);
        if (!hasRunning) {
            if (banner && banner.parentNode) banner.parentNode.removeChild(banner);
            return;
        }
        if (!banner) {
            banner = document.createElement('div');
            banner.id = bannerId;
            banner.className = 'alert alert-info d-flex align-items-center gap-2 mb-3';
            container.prepend(banner);
        }
        banner.textContent = asyncText.loadingBanner;
    };

    try {
        const storeRaw = localStorage.getItem('backtestTaskStore');
        if (storeRaw) {
            const parsed = JSON.parse(storeRaw);
            renderPageBanner(parsed.tasks || []);
        }
    } catch (_error) {
        // ignore
    }
    window.addEventListener('taskdock:update', (event) => {
        const tasksSnapshot = event.detail && event.detail.tasks ? event.detail.tasks : [];
        renderPageBanner(tasksSnapshot);
    });

    const submitAsync = (params, placeholderId) => {
        return fetch(taskEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify(params),
            keepalive: true,
        })
            .then((response) => {
                return response
                    .json()
                    .catch(() => ({}))
                    .then((data) => {
                        if (!response.ok) {
                            const firstError = (() => {
                                if (!data || typeof data !== 'object') return null;
                                if (data.error) return data.error;
                                if (data.detail) return data.detail;
                                const firstVal = Object.values(data)[0];
                                if (Array.isArray(firstVal) && firstVal.length) return firstVal[0];
                                return null;
                            })();
                            const msg =
                                firstError ||
                                (response.status === 400
                                    ? langIsZh
                                        ? '提交数据格式有误，请检查表单。'
                                        : 'Validation failed. Please check the form.'
                                    : `HTTP ${response.status}`);
                            throw new Error(msg);
                        }
                        return data;
                    });
            })
            .then((data) => {
                const ticker = params.ticker || 'Task';
                const created = new Date().toLocaleTimeString();
                if (placeholderId) {
                    removeDockTask(placeholderId);
                }
                const immediateHistory = data.history_id || (data.result && data.result.history_id);
                if (immediateHistory) {
                    const link = `${historyBase}?history_id=${encodeURIComponent(immediateHistory)}`;
                    updateDockTask(`history-${immediateHistory}`, {
                        status: 'SUCCESS',
                        message: asyncText.success,
                        ticker,
                        created,
                        link,
                        taskId: immediateHistory,
                    });
                    return;
                }
                if (data.task_id) {
                    const taskId = data.task_id;
                    const shortId = String(taskId).slice(-6).toUpperCase();
                    updateDockTask(taskId, {
                        status: 'PENDING',
                        message: `${asyncText.queue} · #${shortId}`,
                        ticker,
                        created,
                        taskId,
                    });
                    window.setTimeout(() => pollTask(taskId, Date.now()), TASK_POLL_INTERVAL_MS);
                    return;
                }
                if (data.result) {
                    updateDockTask(`sync-${Date.now()}`, {
                        status: 'SUCCESS',
                        message: asyncText.success,
                        ticker,
                        created,
                        link: null,
                        taskId: null,
                    });
                    return;
                }
                throw new Error('Task id missing from response.');
            });
    };

    const backtestShell = document.querySelector('.backtest-shell');
    const loadingPanel = document.getElementById('backtest-loading-panel');
    const setInPlaceLoading = (flag) => {
        if (!backtestShell || !loadingPanel) return;
        if (flag) {
            loadingPanel.hidden = false;
            backtestShell.classList.add('is-loading');
        } else {
            loadingPanel.hidden = true;
            backtestShell.classList.remove('is-loading');
        }
    };

    form.addEventListener('submit', (event) => {
        if (!taskEndpoint) {
            alert(langIsZh ? '未配置异步回测端点，将使用传统提交。' : 'Async endpoint missing, falling back to sync submit.');
            return;
        }
        event.preventDefault();
        const params = serializeFormParams();
        const ticker = params.ticker || 'Task';
        const created = new Date().toLocaleTimeString();
        const placeholderId = `pending-${Date.now()}`;
        updateDockTask(placeholderId, {
            status: 'SUBMIT',
            message: asyncText.submitting,
            ticker,
            created,
        });
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.dataset.originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = langIsZh ? '正在提交…' : 'Submitting…';
        }
        setInPlaceLoading(true);
        submitAsync(params, placeholderId)
            .catch((error) => {
                if (pageUnloading) {
                    console.warn('Backtest submission interrupted by navigation:', error);
                    return;
                }
                updateDockTask(placeholderId, { status: 'FAILURE', message: error.message || asyncText.failure });
                alert(error.message || (langIsZh ? '回测提交失败，请稍后重试。' : 'Backtest submit failed. Please try again.'));
            })
            .finally(() => {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    if (submitBtn.dataset.originalText) {
                        submitBtn.innerHTML = submitBtn.dataset.originalText;
                    }
                }
                setInPlaceLoading(false);
            });
    });
})();
