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
    const preflightEndpoint = form.dataset.preflightEndpoint || '';
    const historyBase =
        form.dataset.historyBase ||
        (typeof HISTORY_LOAD_BASE !== 'undefined' && HISTORY_LOAD_BASE) ||
        (typeof BACKTEST_PATH !== 'undefined' ? BACKTEST_PATH : '');
    const csrfInput = form.querySelector('input[name=csrfmiddlewaretoken]');
    const csrfToken = csrfInput ? csrfInput.value : '';
    const TASK_POLL_INTERVAL_MS = 2500;
    const PENDING_SUBMIT_KEY = 'backtestPendingSubmissions';
    let pageUnloading = false;
    window.addEventListener('beforeunload', () => {
        pageUnloading = true;
    });
    window.addEventListener('pagehide', () => {
        pageUnloading = true;
    });

    const loadPendingSubmissions = () => {
        try {
            const raw = localStorage.getItem(PENDING_SUBMIT_KEY);
            const parsed = raw ? JSON.parse(raw) : [];
            return Array.isArray(parsed) ? parsed : [];
        } catch (_error) {
            return [];
        }
    };

    const savePendingSubmissions = (entries) => {
        try {
            if (!entries || !entries.length) {
                localStorage.removeItem(PENDING_SUBMIT_KEY);
                return;
            }
            localStorage.setItem(PENDING_SUBMIT_KEY, JSON.stringify(entries));
        } catch (_error) {
            // ignore storage failures
        }
    };

    const upsertPendingSubmission = (entry) => {
        if (!entry || !entry.id) return;
        const entries = loadPendingSubmissions().filter((item) => item && item.id !== entry.id);
        entries.push(entry);
        savePendingSubmissions(entries);
    };

    const removePendingSubmission = (id) => {
        if (!id) return;
        const entries = loadPendingSubmissions().filter((item) => item && item.id !== id);
        savePendingSubmissions(entries);
    };

    const createClientRequestId = () => {
        if (window.crypto && typeof window.crypto.randomUUID === 'function') {
            return window.crypto.randomUUID();
        }
        return `client-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
    };

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
              cancelled: '任务已取消。',
              submitting: '正在提交回测…',
              loadingBanner: '加载中：有任务执行中…',
          }
        : {
              queue: 'Task queued and awaiting execution…',
              running: 'Strategy engines are running…',
              retry: 'Task is retrying…',
              success: 'Task complete. You can view the report anytime.',
              failure: 'Task failed. Please retry.',
              cancelled: 'Task cancelled.',
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

    const runPreflight = async (params) => {
        if (!preflightEndpoint) return null;
        const response = await fetch(preflightEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify(params),
        });
        if (!response.ok) {
            throw new Error(`Preflight HTTP ${response.status}`);
        }
        const payload = await response.json().catch(() => null);
        return payload && typeof payload === 'object' ? payload : null;
    };

    const buildPreflightWarnings = (payload) => {
        const warnings = [];
        const rows = Number(payload && payload.rows ? payload.rows : 0);
        const minRequired = Number(payload && payload.min_required ? payload.min_required : 0);
        const requestedStart = payload && payload.requested_start ? String(payload.requested_start) : '';
        const requestedEnd = payload && payload.requested_end ? String(payload.requested_end) : '';
        const effectiveStart = payload && payload.effective_start ? String(payload.effective_start) : '';
        const effectiveEnd = payload && payload.effective_end ? String(payload.effective_end) : '';
        if (!rows) {
            warnings.push(langIsZh ? '没有可用数据行，可能无法回测。' : 'No available rows found for this window.');
        } else if (minRequired && rows < minRequired) {
            warnings.push(
                langIsZh
                    ? `数据量不足：${rows} 行 < 最小要求 ${minRequired} 行。`
                    : `Insufficient rows: ${rows} < minimum ${minRequired}.`
            );
        }
        if (requestedStart && effectiveStart && requestedStart !== effectiveStart) {
            warnings.push(
                langIsZh
                    ? `起始日期已调整：${requestedStart} → ${effectiveStart}。`
                    : `Start date adjusted: ${requestedStart} → ${effectiveStart}.`
            );
        }
        if (requestedEnd && effectiveEnd && requestedEnd !== effectiveEnd) {
            warnings.push(
                langIsZh
                    ? `结束日期已调整：${requestedEnd} → ${effectiveEnd}。`
                    : `End date adjusted: ${requestedEnd} → ${effectiveEnd}.`
            );
        }
        const quality = payload && payload.data_quality && typeof payload.data_quality === 'object' ? payload.data_quality : {};
        const missingRatio = Number(quality.missing_ratio || 0);
        if (missingRatio > 0.05) {
            warnings.push(
                langIsZh
                    ? `缺失比例偏高：${(missingRatio * 100).toFixed(1)}%。`
                    : `High missing ratio: ${(missingRatio * 100).toFixed(1)}%.`
            );
        }
        const zeroVolumeDays = Number(quality.zero_volume_days || 0);
        if (zeroVolumeDays > 0) {
            warnings.push(
                langIsZh
                    ? `存在 ${zeroVolumeDays} 天零成交量。`
                    : `${zeroVolumeDays} zero-volume days detected.`
            );
        }
        const stalePriceDays = Number(quality.stale_price_days || 0);
        if (stalePriceDays > 0) {
            warnings.push(
                langIsZh
                    ? `存在 ${stalePriceDays} 天价格无变动。`
                    : `${stalePriceDays} stale-price days detected.`
            );
        }
        const notes = Array.isArray(payload && payload.notes) ? payload.notes : [];
        if (notes.length) {
            const trimmed = notes.slice(0, 3).map((note) => String(note));
            warnings.push(...trimmed);
        }
        return warnings;
    };

    const confirmPreflight = async (params) => {
        if (!preflightEndpoint) return true;
        try {
            const payload = await runPreflight(params);
            if (!payload) return true;
            const warnings = buildPreflightWarnings(payload);
            if (!warnings.length) return true;
            const heading = langIsZh ? '预检查提示' : 'Preflight alerts';
            const prompt = langIsZh ? '是否继续提交回测？' : 'Proceed with the backtest?';
            const message = `${heading}\n${warnings.map((item) => `- ${item}`).join('\n')}\n\n${prompt}`;
            return window.confirm(message);
        } catch (error) {
            console.warn('Preflight failed:', error);
            return true;
        }
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
                if (state === 'FAILURE') {
                    updateDockTask(taskId, {
                        status: 'FAILURE',
                        message: data.error || asyncText.failure,
                        progress,
                        taskId,
                    });
                    return;
                }
                if (state === 'REVOKED') {
                    updateDockTask(taskId, {
                        status: 'REVOKED',
                        message: asyncText.cancelled,
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
        const submissionId = params && params.client_request_id ? params.client_request_id : null;
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
                if (submissionId) {
                    removePendingSubmission(submissionId);
                }
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

    form.addEventListener('submit', async (event) => {
        if (!taskEndpoint) {
            alert(langIsZh ? '未配置异步回测端点，将使用传统提交。' : 'Async endpoint missing, falling back to sync submit.');
            return;
        }
        event.preventDefault();
        const params = serializeFormParams();
        const proceed = await confirmPreflight(params);
        if (!proceed) {
            return;
        }
        const submissionId = createClientRequestId();
        params.client_request_id = submissionId;
        upsertPendingSubmission({
            id: submissionId,
            endpoint: taskEndpoint,
            payload: params,
            createdAt: Date.now(),
            attempts: 0,
        });
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
