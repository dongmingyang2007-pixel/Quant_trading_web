(function () {
'use strict';

function parseJsonSafely(raw, fallback) {
    if (!raw) return fallback;
    try {
        return JSON.parse(raw);
    } catch (_error) {
        return fallback;
    }
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

const bootstrapNode = document.getElementById('backtest-ai-bootstrap');
const bootstrap = parseJsonSafely(bootstrapNode ? bootstrapNode.textContent : '{}', {});
const reportContext = parseJsonSafely(bootstrap.report_context_json || '{}', {});
const AI_FETCH_TIMEOUT_MS = Number.parseInt(String(bootstrap.ai_fetch_timeout_ms || '120000'), 10) || 120000;

if (!reportContext.chat_history) {
    reportContext.chat_history = [];
}
let chatHistory = reportContext.chat_history.slice();
let lastAiAnswer = (reportContext.last_ai_answer || '').trim();
const fromHistory = !!reportContext.from_history;
const docLang = (document.documentElement.getAttribute('lang') || navigator.language || '').toLowerCase();
const langIsZh = docLang.startsWith('zh');
const TASK_ENDPOINT = String(bootstrap.task_endpoint || '');
const TASK_STATUS_TEMPLATE = String(bootstrap.task_status_template || '');
const HISTORY_LOAD_BASE = String(bootstrap.history_load_base || '');
const BACKTEST_PATH = String(bootstrap.backtest_path || '/backtest/');
const AI_ENDPOINT = String(bootstrap.ai_endpoint || '');
const AI_STREAM_ENDPOINT = String(bootstrap.ai_stream_endpoint || '');
const csrfInput =
    document.querySelector('#analysis-form input[name=csrfmiddlewaretoken]') ||
    document.querySelector('input[name=csrfmiddlewaretoken]');
const csrfToken = (csrfInput && csrfInput.value) ? csrfInput.value : getCookie('csrftoken');
const TASK_POLL_INTERVAL_MS = 2500;
const TASK_MAX_WAIT_MS = 10 * 60 * 1000; // 10 分钟
const TASK_REFRESH_INTERVAL = 2500;
const aiText = langIsZh
    ? {
          hideSignals: '隐藏历史明细',
          showSignals: '显示历史明细',
          webOn: '🌐 联网：开',
          webOff: '🌐 联网：关',
          calloutRealtime: '实时资讯引用',
          calloutQuick: '快速解读',
          calloutNotice: '注意',
          thinkingPlaceholder: 'AI 正在分析',
          webResultDefault: '实时资讯',
          webResultHeading: '联网参考',
          assistantRole: 'AI 投研',
          userRole: '你的提问',
          badgeOnline: '联网分析',
          badgeOffline: '回测解读',
          badgeUser: '自定义提问',
          reasoningDefault: '推演',
          reasoningWeb: '联网检索',
          reasoningReview: '复核推演',
          stepStatusWarn: '需关注',
          stepStatusOk: '推演',
          statusReadyHistory: 'AI 助手已准备就绪，可继续提问。',
          statusReadyFresh: 'AI 助手已准备就绪，可直接输入问题。',
          confirmClear: '确定要清空本页的 AI 对话记录吗？',
          statusHistoryCleared: 'AI 历史已清空，可重新发起分析或提问。',
          promptEnableWeb: '需要联网搜索前，请先点击上方的“🌐 联网”按钮以启用实时检索。',
          statusWebOff: '联网已关闭，请先启用再重试。',
          statusFetching: 'AI 正在联网检索，请稍候…',
          statusAnalyzing: 'AI 正在分析，请稍候…',
          statusComplete: 'AI 分析完成，可继续提问。',
          statusTimeout: 'AI 分析超过 2 分钟已自动取消，请稍后重试。',
          statusUnavailable: 'AI 服务暂时不可用，请稍后重试。',
          apology: '抱歉，暂时无法获取新的投资建议。',
          statusSynthesizing: 'AI 正在生成整体解读…',
          statusSummaryMissing: 'AI 总结暂未生成，可稍后手动提问。',
      }
    : {
          hideSignals: 'Hide signal details',
          showSignals: 'Show signal details',
          webOn: '🌐 Web: On',
          webOff: '🌐 Web: Off',
          calloutRealtime: 'Live market cite',
          calloutQuick: 'Quick brief',
          calloutNotice: 'Notice',
          thinkingPlaceholder: 'AI is analyzing',
          webResultDefault: 'Market update',
          webResultHeading: 'Web references',
          assistantRole: 'AI Research',
          userRole: 'Your question',
          badgeOnline: 'Live search',
          badgeOffline: 'Backtest brief',
          badgeUser: 'Custom prompt',
          reasoningDefault: 'Reasoning',
          reasoningWeb: 'Web lookup',
          reasoningReview: 'Secondary review',
          stepStatusWarn: 'Needs attention',
          stepStatusOk: 'Insight',
          statusReadyHistory: 'AI assistant is ready—feel free to ask follow-ups.',
          statusReadyFresh: 'AI assistant is ready—ask away.',
          confirmClear: 'Clear the AI conversation on this page?',
          statusHistoryCleared: 'Conversation cleared. Start a fresh analysis anytime.',
          promptEnableWeb: 'Toggle “🌐 Web” above before requesting live search.',
          statusWebOff: 'Live search is off. Turn it on and try again.',
          statusFetching: 'AI is querying live sources…',
          statusAnalyzing: 'AI is analyzing—please hold…',
          statusComplete: 'Analysis complete—feel free to ask more.',
          statusTimeout: 'Analysis took over two minutes and was cancelled. Please retry.',
          statusUnavailable: 'AI service is unavailable right now. Please try again.',
          apology: 'Sorry, we can’t provide fresh guidance right now.',
          statusSynthesizing: 'AI is compiling the overall brief…',
          statusSummaryMissing: 'Summary is not available yet. Please ask manually later.',
      };

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analysis-form');
    const navItems = Array.from(document.querySelectorAll('.backtest-tab'));
    const panels = Array.from(document.querySelectorAll('.backtest-panel'));
    const copyButtons = Array.from(document.querySelectorAll('.meta-copy[data-copy-value]'));
    const copyLabel = langIsZh ? '复制' : 'Copy';
    const copiedLabel = langIsZh ? '已复制' : 'Copied';

    const copyText = async (value) => {
        if (!value) return false;
        if (!document.body) return false;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            try {
                await navigator.clipboard.writeText(value);
                return true;
            } catch (_error) {
                // fallback
            }
        }
        const textarea = document.createElement('textarea');
        textarea.value = value;
        textarea.setAttribute('readonly', 'readonly');
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        let ok = false;
        try {
            ok = document.execCommand('copy');
        } catch (_error) {
            ok = false;
        }
        textarea.remove();
        return ok;
    };

    copyButtons.forEach((button) => {
        const raw = button.dataset.copyValue || '';
        if (!raw) {
            button.disabled = true;
            return;
        }
        const original = button.textContent || copyLabel;
        button.addEventListener('click', async () => {
            const ok = await copyText(raw);
            if (!ok) return;
            button.textContent = copiedLabel;
            window.setTimeout(() => {
                button.textContent = original;
            }, 1200);
        });
    });

    function activatePanel(name) {
        let targetPanel = null;
        panels.forEach((panel) => {
            const hit = panel.dataset.panel === name;
            panel.classList.toggle('active', hit);
            if (hit) targetPanel = panel;
        });
        navItems.forEach((tab, index) => {
            const hit = tab.dataset.panel === name;
            tab.classList.toggle('active', hit);
            if (!targetPanel && index === 0) {
                tab.classList.add('active');
            }
        });
        if (!targetPanel && panels.length) {
            targetPanel = panels[0];
            targetPanel.classList.add('active');
        }
        return targetPanel;
    }

    window.backtestActivatePanel = activatePanel;

    let defaultPanel = fromHistory ? 'history' : 'overview';
    if (!panels.some((panel) => panel.dataset.panel === defaultPanel)) {
        defaultPanel = panels[0]?.dataset.panel;
    }
    const initialPanel = defaultPanel ? activatePanel(defaultPanel) : null;
    if (initialPanel) {
        initialPanel.scrollIntoView({ behavior: 'auto', block: 'start' });
    }
    navItems.forEach((tab) => {
        tab.addEventListener('click', (event) => {
            event.preventDefault();
            const target = tab.dataset.panel;
            if (!target) return;
            const panelEl = activatePanel(target);
            if (panelEl) {
                panelEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                document.querySelector('.backtest-panels')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    const anchorPanelMap = {
        '#advisor-pane': 'advisor',
        '#analysis-form': 'config',
        '#model-weights-title': 'expert',
        '#risk-dashboard-title': 'expert',
        '#knowledge-graph-title': 'expert',
        '#factor-effectiveness-title': 'expert',
        '#macro-dashboard-title': 'expert',
        '#ai-panel': 'ai',
        '#opportunity-radar': 'overview',
        '#scenario-board': 'overview',
        '#quick-glance': 'overview'
    };

    document.querySelectorAll('.briefing-back-link, .briefing-link').forEach((link) => {


        link.addEventListener('click', (event) => {
            const target = link.getAttribute('href');
            if (!target || !target.startsWith('#')) return;
            event.preventDefault();
            let panel = anchorPanelMap[target];
            if (!panel) {
                const ownerPanel = document.querySelector(target)?.closest('.backtest-panel');
                panel = ownerPanel?.dataset.panel;
            }
            let panelEl = null;
            if (panel) {
                panelEl = activatePanel(panel);
            }
            const node = document.querySelector(target);
            if (node) {
                if (node.tagName === 'DETAILS' && !node.open) {
                    node.open = true;
                }
                window.requestAnimationFrame(() => {
                    node.scrollIntoView({ behavior: 'smooth', block: 'start' });
                });
            } else if (panelEl) {
                panelEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            window.location.hash = target;
        });
    });

    document.querySelectorAll('[data-panel-target]').forEach((link) => {
        link.addEventListener('click', (event) => {
            const panelName = link.dataset.panelTarget;
            if (!panelName) return;
            const href = link.getAttribute('href') || '';
            const isHashLink = href.startsWith('#');
            if (isHashLink) {
                event.preventDefault();
            }
            const panelEl = activatePanel(panelName);
            const targetNode = isHashLink ? document.querySelector(href) : null;
            if (targetNode) {
                if (targetNode.tagName === 'DETAILS' && !targetNode.open) {
                    targetNode.open = true;
                }
                window.requestAnimationFrame(() => {
                    targetNode.scrollIntoView({ behavior: 'smooth', block: 'start' });
                });
            } else if (panelEl) {
                panelEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            if (isHashLink) {
                window.history.replaceState(null, '', href);
            }
        });
    });

    document.querySelectorAll('.backtest-sidebar-btn[data-target-panel]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.targetPanel;
            if (!target) return;
            const panelEl = activatePanel(target);
            if (panelEl) {
                panelEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else if (target === 'config') {
                document.getElementById('analysis-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                document.querySelector(`[data-panel=\"${target}\"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    if (fromHistory) {
        document.querySelector('[data-panel="history"]')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    const toggleButton = document.getElementById('toggle-signals');
    const signalsTable = document.getElementById('signals-table');
    if (toggleButton && signalsTable) {
        let visible = true;
        const updateSignalToggle = () => {
            toggleButton.textContent = visible ? aiText.hideSignals : aiText.showSignals;
        };
        updateSignalToggle();
        toggleButton.addEventListener('click', () => {
            visible = !visible;
            signalsTable.classList.toggle('d-none', !visible);
            updateSignalToggle();
        });
    }

    document.querySelectorAll('.js-open-expert').forEach((btn) => {
        btn.addEventListener('click', () => {
            const expertPanel = activatePanel('expert');
            (expertPanel || document.querySelector('.backtest-panels'))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            const expertTab = document.querySelector('.backtest-tab[data-panel="expert"]');
            if (expertTab) {
                expertTab.focus();
                expertTab.classList.add('focus-ring');
                window.setTimeout(() => expertTab.classList.remove('focus-ring'), 600);
            }
        });
    });

    const HISTORY_WINDOW = 8;
    const aiStatus = document.getElementById('ai-status');
    const aiMeta = document.getElementById('ai-meta');
    const aiMessages = document.getElementById('ai-messages');
    const aiThinking = document.getElementById('ai-thinking');
    const aiForm = document.getElementById('ai-chat-form');
    const aiInput = document.getElementById('ai-input');
    const aiEndpoint = AI_ENDPOINT;
    const aiStreamEndpoint = AI_STREAM_ENDPOINT;
    const aiStreamSupported = typeof ReadableStream !== 'undefined' && typeof TextDecoder !== 'undefined';

    let activeAiController = null;

    function beginAiFetch(endpoint, payload) {
        if (!endpoint) {
            return { controller: null, fetchPromise: Promise.reject(new Error('ai_endpoint_missing')) };
        }
        if (!csrfToken) {
            return { controller: null, fetchPromise: Promise.reject(new Error('csrf_missing')) };
        }
        if (activeAiController) {
            try {
                activeAiController.abort();
            } catch (error) {
                // ignore abort errors
            }
        }
        const controller = new AbortController();
        activeAiController = controller;
        const timer = window.setTimeout(() => controller.abort(), AI_FETCH_TIMEOUT_MS);
        const fetchPromise = fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify(payload),
            signal: controller.signal,
        }).finally(() => {
            window.clearTimeout(timer);
            if (activeAiController === controller) {
                activeAiController = null;
            }
        });
        return { controller, fetchPromise };
    }

    function readAiResponse(response) {
        if (!response.ok) {
            return response
                .json()
                .catch(() => ({}))
                .then((data) => {
                    const message = (data && data.error) || `AI 接口返回 ${response.status}`;
                    throw new Error(message);
                });
        }
        return response.json();
    }

    function legacyAiRequest(payload) {
        const { fetchPromise } = beginAiFetch(aiEndpoint, payload);
        return fetchPromise.then(readAiResponse);
    }

    function sendAiStreamRequest(payload, options = {}) {
        if (!aiStreamSupported || !aiStreamEndpoint) {
            return legacyAiRequest(payload);
        }
        const { fetchPromise, controller } = beginAiFetch(aiStreamEndpoint, payload);
        const softTimeoutMs = AI_FETCH_TIMEOUT_MS * 0.5;
        const hardTimeoutMs = AI_FETCH_TIMEOUT_MS;
        let softTimer = null;
        let hardTimer = null;
        return fetchPromise.then((response) => {
            if (!response.ok) {
                return readAiResponse(response);
            }
            if (!response.body || !response.body.getReader) {
                return response.json();
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let finalPayload = null;
            return new Promise((resolve, reject) => {
                const clearTimers = () => {
                    if (softTimer) window.clearTimeout(softTimer);
                    if (hardTimer) window.clearTimeout(hardTimer);
                    softTimer = null;
                    hardTimer = null;
                };
                const handleEvent = (eventType, payload) => {
                    if (eventType === 'progress') {
                        if (options.onProgress && payload) {
                            options.onProgress(payload);
                        }
                    } else if (eventType === 'delta') {
                        if (options.onProgress && payload) {
                            payload.stage = 'delta';
                            options.onProgress(payload);
                        }
                    } else if (eventType === 'message') {
                        finalPayload = payload || finalPayload;
                    } else if (eventType === 'error') {
                        const message = (payload && payload.error) || aiText.statusUnavailable;
                        clearTimers();
                        reject(new Error(message));
                    } else if (eventType === 'end') {
                        clearTimers();
                        resolve(finalPayload || payload || {});
                    }
                };

                const parseBuffer = () => {
                    let boundary = buffer.indexOf('\n\n');
                    while (boundary >= 0) {
                        const rawEvent = buffer.slice(0, boundary).trim();
                        buffer = buffer.slice(boundary + 2);
                        if (rawEvent) {
                            const lines = rawEvent.split(/\r?\n/);
                            let eventType = 'message';
                            const dataLines = [];
                            lines.forEach((line) => {
                                if (line.startsWith('event:')) {
                                    eventType = line.slice(6).trim();
                                } else if (line.startsWith('data:')) {
                                    dataLines.push(line.slice(5).trim());
                                }
                            });
                            let payload = null;
                            if (dataLines.length) {
                                const joined = dataLines.join('\n');
                                try {
                                    payload = joined ? JSON.parse(joined) : null;
                                } catch (error) {
                                    console.warn('无法解析 AI 流事件', joined, error);
                                }
                            }
                            handleEvent(eventType, payload);
                        }
                        boundary = buffer.indexOf('\n\n');
                    }
                };

                const pump = () => {
                    reader
                        .read()
                        .then(({ done, value }) => {
                            if (done) {
                                handleEvent('end', finalPayload);
                                return;
                            }
                            buffer += decoder.decode(value, { stream: true });
                            parseBuffer();
                            pump();
                        })
                        .catch((error) => {
                            clearTimers();
                            reject(error);
                        });
                };
                softTimer = window.setTimeout(() => {
                    handleEvent('progress', { message: aiText.statusAnalyzing });
                }, softTimeoutMs);
                hardTimer = window.setTimeout(() => {
                    try {
                        controller.abort();
                    } catch (error) {
                        // ignore
                    }
                }, hardTimeoutMs);
                pump();
            });
        });
    }

    function requestAi(payload, options = {}) {
        return sendAiStreamRequest(payload, options);
    }

    // 🌐 Web Search toggle (globe button)
    const aiToggle = document.getElementById('ai-toggle-web');
    const aiClearBtn = document.getElementById('ai-clear-history');
    const aiAbortBtn = document.getElementById('ai-abort');
    const aiContinueBtn = document.getElementById('ai-continue');
    let webEnabled;
    if (reportContext && typeof reportContext.enable_web === 'boolean') {
        webEnabled = !!reportContext.enable_web;
    } else {
        webEnabled = false;
        if (reportContext) {
            reportContext.enable_web = false;
        }
    }
    const MODEL_STORAGE_KEY = 'quant-ai-preferred-model';
    const HARDCODED_DEFAULT_MODEL = 'deepseek-r1:8b';
    const modelInput = document.getElementById('ai-model-input');
    const modelOptions = document.getElementById('ai-model-input');
    let availableModels = Array.isArray(reportContext.ai_model_choices)
        ? reportContext.ai_model_choices.filter((item, index, arr) => item && arr.indexOf(item) === index)
        : [];
    let preferredModel = (reportContext.ai_model || '').trim() || HARDCODED_DEFAULT_MODEL;
    try {
        const stored = window.sessionStorage.getItem(MODEL_STORAGE_KEY);
        if (stored && stored.trim()) {
            preferredModel = stored.trim();
        }
    } catch (error) {
        // ignore storage errors (private mode etc.)
    }
    if (preferredModel && !availableModels.includes(preferredModel)) {
        availableModels.unshift(preferredModel);
    }

    function buildAutoPrompt(ctx) {
        if (!ctx) return langIsZh ? '请基于最新回测输出要点。' : 'Please summarize the latest backtest.';
        const safeText = (value) => {
            if (value === undefined || value === null) return '';
            return toPlainText(String(value)).replace(/\s+/g, ' ').trim();
        };
        const params = (ctx && typeof ctx.params === 'object' && ctx.params) || {};
        const stats = (ctx && typeof ctx.stats === 'object' && ctx.stats) || {};
        const ticker = safeText(
            ctx.ticker ||
                (ctx.snapshot && ctx.snapshot.ticker) ||
                params.ticker ||
                params.symbol ||
                ''
        );
        const benchmark = safeText(
            ctx.benchmark ||
                stats.benchmark ||
                params.benchmark_ticker ||
                ''
        );
        let windowLabel = '';
        if (ctx.start_date && ctx.end_date) {
            windowLabel = `${safeText(ctx.start_date)} → ${safeText(ctx.end_date)}`;
        } else if (params.start_date && params.end_date) {
            windowLabel = `${safeText(params.start_date)} → ${safeText(params.end_date)}`;
        }
        const metricLines = (Array.isArray(ctx.metrics) ? ctx.metrics : [])
            .slice(0, 4)
            .map((item) => {
                if (!item || !item.label) return '';
                const label = safeText(item.label);
                const value = safeText(item.value !== undefined ? item.value : '');
                return label && value ? `${label}: ${value}` : label || value;
            })
            .filter(Boolean);
        if (!metricLines.length) {
            ['total_return', 'cagr', 'sharpe', 'max_drawdown'].forEach((key) => {
                if (stats[key] !== undefined && stats[key] !== null) {
                    const labelMap = {
                        total_return: langIsZh ? '累计收益' : 'Total return',
                        cagr: 'CAGR',
                        sharpe: 'Sharpe',
                        max_drawdown: langIsZh ? '最大回撤' : 'Max drawdown',
                    };
                    metricLines.push(`${labelMap[key] || key}: ${safeText(stats[key])}`);
                }
            });
        }
        const quickPoints = (Array.isArray(ctx.quick_summary) ? ctx.quick_summary : [])
            .map(safeText)
            .filter(Boolean)
            .slice(0, 3);
        const riskPoints = (Array.isArray(ctx.risk_alerts) ? ctx.risk_alerts : [])
            .map(safeText)
            .filter(Boolean)
            .slice(0, 2);
        const sections = [];
        if (langIsZh) {
            sections.push(
                `请扮演资深投顾，基于最新回测结果，向用户主动汇报 ${ticker || '该标的'} 的核心表现。`
            );
            if (windowLabel) {
                sections.push(`回测区间：${windowLabel}${benchmark ? `，对比 ${benchmark}` : ''}。`);
            }
            if (metricLines.length) {
                sections.push(`关键指标：${metricLines.join('；')}。`);
            }
            if (quickPoints.length) {
                sections.push(`亮点：${quickPoints.join('；')}`);
            }
            if (riskPoints.length) {
                sections.push(`风险提示：${riskPoints.join('；')}`);
            }
            sections.push('请用条理清晰的 3-4 段文字输出要点，并给出下一步建议。');
        } else {
            sections.push(
                `You are a senior investment analyst. Proactively brief the user on the newest backtest for ${ticker || 'the instrument'}.`
            );
            if (windowLabel) {
                sections.push(
                    `Window: ${windowLabel}${benchmark ? `, benchmark ${benchmark}` : ''}.`
                );
            }
            if (metricLines.length) {
                sections.push(`Key metrics: ${metricLines.join('; ')}.`);
            }
            if (quickPoints.length) {
                sections.push(`Highlights: ${quickPoints.join('; ')}`);
            }
            if (riskPoints.length) {
                sections.push(`Risks: ${riskPoints.join('; ')}`);
            }
            sections.push(
                'Respond in 3-4 concise paragraphs including next-step guidance tailored to the user profile.'
            );
        }
        return sections.join(' ');
    }
    if (availableModels.length === 0 && preferredModel) {
        availableModels.push(preferredModel);
    }
    const defaultModel = preferredModel || (availableModels[0] || HARDCODED_DEFAULT_MODEL);
    reportContext.ai_model = defaultModel;
    function ensureModelOption(value) {
        if (!modelOptions || !value) return;
        const existing = Array.from(modelOptions.options || []).some((option) => option.value === value);
        if (!existing) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            modelOptions.appendChild(option);
        }
    }
    if (modelOptions) {
        modelOptions.innerHTML = '';
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = langIsZh ? '请选择模型' : 'Select model';
        placeholder.disabled = true;
        placeholder.selected = !defaultModel;
        modelOptions.appendChild(placeholder);
        availableModels.forEach((model) => {
            ensureModelOption(model);
        });
    }
    if (modelInput) {
        if (defaultModel) {
            modelInput.value = defaultModel;
        }
        const handleModelInput = () => {
            const value = modelInput.value.trim();
            reportContext.ai_model = value;
            try {
                if (value) {
                    window.sessionStorage.setItem(MODEL_STORAGE_KEY, value);
                } else {
                    window.sessionStorage.removeItem(MODEL_STORAGE_KEY);
                }
            } catch (error) {
                // ignore storage errors
            }
        };
        modelInput.addEventListener('change', handleModelInput);
    }

    function setWebToggleUi() {
        if (!aiToggle) return;
        aiToggle.setAttribute('aria-pressed', webEnabled ? 'true' : 'false');
        aiToggle.textContent = webEnabled ? aiText.webOn : aiText.webOff;
        aiToggle.classList.toggle('solid', webEnabled);
        aiToggle.classList.toggle('ghost', !webEnabled);
    }

    if (aiToggle) {
        setWebToggleUi();
        aiToggle.addEventListener('click', () => {
            webEnabled = !webEnabled;
            if (reportContext) reportContext.enable_web = webEnabled; // 记录到上下文
            setWebToggleUi();
        });
    }

    if (aiAbortBtn) {
        aiAbortBtn.addEventListener('click', () => {
            if (activeAiController) {
                try {
                    activeAiController.abort();
                    setAiStatus('warn', aiText.statusTimeout);
                } catch (error) {
                    // ignore
                }
            }
        });
    }

    if (aiContinueBtn) {
        aiContinueBtn.addEventListener('click', () => {
            if (!lastAiAnswer) return;
            if (!aiInput) return;
            aiInput.value = langIsZh ? '继续' : 'continue';
            aiForm?.dispatchEvent(new Event('submit'));
        });
    }

    function setAiStatus(kind, message) {
        if (!aiStatus) return;
        aiStatus.textContent = message || '';
        const variant = kind ? ` ai-status-${kind}` : '';
        aiStatus.className = `ai-status-chip${variant}`;
    }

    function inferProvider(modelName) {
        const lowered = (modelName || '').toLowerCase();
        if (lowered.startsWith('bailian:') || lowered.startsWith('dashscope:') || lowered.startsWith('aliyun:')) {
            return 'bailian';
        }
        if (lowered.includes('gemini')) {
            return 'gemini';
        }
        return 'ollama';
    }

    function setAiMeta(data = {}) {
        if (!aiMeta) return;
        const model = (Array.isArray(data.models) && data.models[0]) || null;
        const modelName = (model && model.name) || data.selected_model || reportContext.ai_model || '';
        const provider = (model && model.provider) || reportContext.ai_provider || inferProvider(modelName);
        const web = data.web_used ? (langIsZh ? '联网' : 'Online') : (langIsZh ? '离线' : 'Offline');
        const profile = data.profile || {};
        const costHint = [];
        if (profile.total_sec !== undefined) {
            costHint.push(`耗时 ${profile.total_sec}s`);
        }
        if (profile.tokens) {
            const t = profile.tokens;
            const parts = [];
            if (t.prompt !== undefined) parts.push(`in ${t.prompt}`);
            if (t.output !== undefined) parts.push(`out ${t.output}`);
            if (parts.length) {
                costHint.push(`tokens ${parts.join(' / ')}`);
            } else if (t.total !== undefined) {
                costHint.push(`tokens ${t.total}`);
            }
        }
        const metaParts = [
            modelName ? `${langIsZh ? '模型' : 'Model'}: ${escapeHtml(modelName)}` : '',
            provider ? `${langIsZh ? '供应商' : 'Provider'}: ${escapeHtml(provider)}` : '',
            `${langIsZh ? '模式' : 'Mode'}: ${web}`,
            costHint.join(' · '),
        ].filter(Boolean);
        aiMeta.innerHTML = metaParts.length
            ? `<span class="ai-meta-pill">${metaParts.join(' · ')}</span>`
            : '';
    }

    function syncHistory() {
        if (reportContext) {
            reportContext.chat_history = chatHistory.slice();
        }
    }

    function escapeHtml(text) {
        if (!text) return '';
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function toPlainText(text) {
        if (!text) return '';
        let output = String(text);
        output = output.replace(/```[\s\S]*?```/g, '');
        output = output.replace(/###\s*/g, '');
        output = output.replace(/^- /gm, '• ');
        output = output.replace(/\r/g, '');
        return output.replace(/\n{3,}/g, '\n\n').trim();
    }

    function formatAiAnswer(text) {
        if (!text) return '';
        const renderInline = (input) => {
            let safe = escapeHtml(input || '');
            safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            safe = safe.replace(/\*(.+?)\*/g, '<em>$1</em>');
            safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
            safe = safe.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, label, url) => {
                const cleanUrl = url.replace(/"/g, '&quot;');
                return `<a href="${cleanUrl}" target="_blank" rel="noopener">${label}</a>`;
            });
            return safe;
        };

        const lines = text.split(/\n+/);
        let html = '';
        let listType = null;
        let tableRows = [];

        const closeList = () => {
            if (listType === 'ul') {
                html += '</ul>';
            } else if (listType === 'ol') {
                html += '</ol>';
            }
            listType = null;
        };

        const flushTable = () => {
            if (!tableRows.length) return;
            const rows = [...tableRows];
            tableRows = [];
            const parseRow = (row) =>
                row
                    .split('|')
                    .slice(1, -1)
                    .map((cell) => renderInline(cell.trim()));
            const hasDivider = rows.length > 1 && /^[-:\\s|]+$/.test(rows[1]);
            let tableHtml = '<table class="ai-table">';
            if (hasDivider) {
                const headerCells = parseRow(rows.shift());
                rows.shift();
                tableHtml += '<thead><tr>';
                headerCells.forEach((cell) => {
                    tableHtml += `<th>${cell || '&nbsp;'}</th>`;
                });
                tableHtml += '</tr></thead>';
            }
            tableHtml += '<tbody>';
            rows.forEach((row) => {
                const cells = parseRow(row);
                if (!cells.length) return;
                tableHtml += '<tr>';
                cells.forEach((cell) => {
                    tableHtml += `<td>${cell || '&nbsp;'}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';
            html += tableHtml;
        };

        const calloutRules = [
            { keyword: aiText.calloutRealtime, icon: '🌐', tone: 'info' },
            { keyword: aiText.calloutQuick, icon: '🧭', tone: 'highlight' },
            { keyword: aiText.calloutNotice, icon: '⚠️', tone: 'warn' },
        ];

        const tryCallout = (raw) => {
            const match = calloutRules.find((rule) => raw.startsWith(rule.keyword));
            if (!match) return false;
            closeList();
            flushTable();
            html += `<div class="ai-callout ai-callout-${match.tone}"><div class="ai-callout-title">${match.icon} ${renderInline(raw)}</div></div>`;
            return true;
        };

        lines.forEach((line) => {
            const trimmed = line.trim();
            if (!trimmed) return;
            if (tryCallout(trimmed)) return;
            if (/^\|.+\|$/.test(trimmed)) {
                closeList();
                tableRows.push(trimmed);
                return;
            } else if (tableRows.length) {
                flushTable();
            }
            if (trimmed.startsWith('### ')) {
                closeList();
                flushTable();
                html += `<h6 class="ai-section-title">${renderInline(trimmed.slice(4))}</h6>`;
            } else if (trimmed.startsWith('- ')) {
                if (listType !== 'ul') {
                    closeList();
                    html += '<ul class="ai-section-list">';
                    listType = 'ul';
                }
                html += `<li>${renderInline(trimmed.slice(2))}</li>`;
            } else if (/^\\d+\\./.test(trimmed)) {
                if (listType !== 'ol') {
                    closeList();
                    html += '<ol class="ai-section-ol">';
                    listType = 'ol';
                }
                html += `<li>${renderInline(trimmed.replace(/^\\d+\\.\\s*/, ''))}</li>`;
            } else if (trimmed.startsWith('> ')) {
                closeList();
                flushTable();
                html += `<div class="ai-callout ai-callout-neutral">${renderInline(trimmed.slice(1).trim())}</div>`;
            } else {
                closeList();
                flushTable();
                html += `<p>${renderInline(trimmed)}</p>`;
            }
        });

        closeList();
        flushTable();
        return html;
    }

    function animateTypewriter(node, sourceText, finalHtml, options = {}) {
        if (!node) return;
        const plain = toPlainText(sourceText);
        if (!plain) {
            if (finalHtml) {
                node.innerHTML = finalHtml;
            } else {
                node.textContent = '';
            }
            if (node.classList) node.classList.remove('typing');
            if (typeof options.onComplete === 'function') {
                options.onComplete();
            }
            return;
        }
        const speed = options.speed || (plain.length > 800 ? 4 : plain.length > 400 ? 8 : 14);
        const step = Math.max(1, options.step || (plain.length > 400 ? 2 : 1));
        let index = 0;
        const total = plain.length;
        if (node.classList) node.classList.add('typing');

        const writeChunk = () => {
            index += step;
            node.textContent = plain.slice(0, index);
            if (index < total) {
                node.__typingTimer = window.setTimeout(writeChunk, speed);
            } else {
                if (finalHtml) {
                    node.innerHTML = finalHtml;
                } else {
                    node.textContent = plain;
                }
                if (node.classList) node.classList.remove('typing');
                if (typeof options.onComplete === 'function') {
                    options.onComplete();
                }
                node.__typingTimer = null;
                node.__typingDelay = null;
            }
        };

        const start = () => {
            if (node.__typingTimer) {
                window.clearTimeout(node.__typingTimer);
            }
            if (node.__typingDelay) {
                window.clearTimeout(node.__typingDelay);
                node.__typingDelay = null;
            }
            writeChunk();
        };

        if (options.delay) {
            node.__typingDelay = window.setTimeout(start, options.delay);
        } else {
            start();
        }
    }

    let thinkingPlaceholderTimer = null;
    function startThinkingAnimation() {
        if (!aiThinking) return;
        stopThinkingAnimation();
        aiThinking.classList.remove('d-none');
        aiThinking.innerHTML = `<div class="ai-thinking-placeholder"><span class="ai-thinking-text">${aiText.thinkingPlaceholder}</span><span class="ai-thinking-dots">...</span></div>`;
        const dotNode = aiThinking.querySelector('.ai-thinking-dots');
        let dots = 1;
        thinkingPlaceholderTimer = window.setInterval(() => {
            dots = (dots + 1) % 4;
            if (dotNode) {
                dotNode.textContent = '.'.repeat(dots);
            }
        }, 420);
    }

    function stopThinkingAnimation() {
        if (thinkingPlaceholderTimer) {
            window.clearInterval(thinkingPlaceholderTimer);
            thinkingPlaceholderTimer = null;
        }
    }

    function renderWebResults(list) {
        if (!Array.isArray(list) || list.length === 0) return '';
        const items = list
            .map((item) => {
                const title = escapeHtml(item.title || item.heading || aiText.webResultDefault);
                const url = item.url || item.href || '#';
                const source = escapeHtml(item.source || item.publisher || item.host || '');
                const retrieved = item.retrieved_at ? escapeHtml(item.retrieved_at) : '';
                const snippet = escapeHtml(item.snippet || '');
                const metaParts = [];
                if (source) metaParts.push(source);
                if (retrieved) metaParts.push(retrieved);
                return `<li>
                    <span class="ai-web-icon">🔗</span>
                    <div class="ai-web-block">
                        <a href="${url}" target="_blank" rel="noopener">${title}</a>
                        ${metaParts.length ? `<span class="ai-web-source">${metaParts.join(' · ')}</span>` : ''}
                        ${snippet ? `<p class="ai-web-snippet">${snippet}</p>` : ''}
                    </div>
                </li>`;
            })
            .join('');
        return `
            <div class="ai-web-results">
                <div class="ai-web-title">${aiText.webResultHeading}</div>
                <ul>${items}</ul>
            </div>`;
    }

    let streamingBuffer = '';
    let streamingBubble = null;

    function appendMessage(role, content, options = {}) {
        if (!aiMessages || content === undefined || content === null) return null;
        const wrapper = document.createElement('article');
        wrapper.className = `ai-message ai-message-${role}`;
        const card = document.createElement('div');
        card.className = 'ai-message-card';
        const head = document.createElement('header');
        head.className = 'ai-message-head';
        const label = document.createElement('span');
        label.className = 'ai-message-role';
        label.textContent = role === 'assistant' ? aiText.assistantRole : aiText.userRole;
        const badge = document.createElement('span');
        badge.className = 'ai-message-tag';
        if (role === 'assistant') {
            const online = options.webUsed !== undefined ? !!options.webUsed : webEnabled;
            badge.textContent = online ? aiText.badgeOnline : aiText.badgeOffline;
            badge.classList.add(online ? 'tag-online' : 'tag-offline');
        } else {
            badge.textContent = aiText.badgeUser;
            badge.classList.add('tag-user');
        }
        head.appendChild(label);
        head.appendChild(badge);

        const body = document.createElement('div');
        body.className = 'ai-message-bubble';
        const shouldAnimate = role === 'assistant' && options.animate !== false;
        if (role === 'assistant') {
            if (shouldAnimate) {
                animateTypewriter(body, content, formatAiAnswer(content));
            } else {
                body.innerHTML = formatAiAnswer(content);
            }
        } else {
            body.textContent = content;
        }

        card.appendChild(head);
        card.appendChild(body);
        if (options.webResults && options.webResults.length && role === 'assistant') {
            const webBlock = document.createElement('div');
            webBlock.innerHTML = renderWebResults(options.webResults);
            card.appendChild(webBlock);
        }
        wrapper.appendChild(card);
        aiMessages.appendChild(wrapper);
        aiMessages.scrollTop = aiMessages.scrollHeight;
        if (role === 'assistant' && options.trackStreaming) {
            streamingBubble = body;
            streamingBuffer = content || '';
        }
        return { wrapper, body };
    }

    function renderHistory(animate = true) {
        if (!aiMessages) return;
        stopThinkingAnimation();
        aiMessages.innerHTML = '';
        chatHistory.forEach((entry) => {
            appendMessage(
                entry.role === 'user' ? 'user' : 'assistant',
                entry.content || '',
                {
                    animate: animate && entry.role !== 'user',
                    webUsed: entry.web_used,
                    webResults: entry.web_results,
                }
            );
        });
    }

    function renderThinking(thoughts, animate = true) {
        if (!aiThinking) return;
        stopThinkingAnimation();
        aiThinking.innerHTML = '';
        const normalized = [];
        (thoughts || []).forEach((item) => {
            if (!item) return;
            if (typeof item === 'string') {
                normalized.push({ title: aiText.reasoningDefault, status: 'ok', body: item });
                return;
            }
            if (Array.isArray(item.thoughts) && item.thoughts.length) {
                normalized.push({
                    title: item.model || aiText.reasoningDefault,
                    status: (item.status || 'ok').toLowerCase().includes('error') ? 'warn' : 'ok',
                    body: item.thoughts.join('\n'),
                });
                return;
            }
            if (typeof item === 'object') {
                const body = item.detail || item.answer || item.content || '';
                if (!body) return;
                normalized.push({
                    title: item.title || item.model || aiText.reasoningDefault,
                    status: (item.status || 'ok').toLowerCase().includes('error') ? 'warn' : 'ok',
                    body,
                });
            }
        });
        if (!normalized.length) {
            aiThinking.classList.add('d-none');
            return;
        }
        aiThinking.classList.remove('d-none');
        const timeline = document.createElement('div');
        timeline.className = 'ai-thinking-timeline';
        const resolveMeta = (entry) => {
            const model = (entry.title || '').toLowerCase();
            if (model.includes('web')) {
                return { icon: '🔍', label: aiText.reasoningWeb };
            }
            if (model.includes('secondary') || model.includes('qwen')) {
                return { icon: '🧠', label: entry.title || aiText.reasoningReview };
            }
            return { icon: '🤖', label: entry.title || aiText.reasoningDefault };
        };
        normalized.forEach((entry, idx) => {
            const step = document.createElement('div');
            const warn = entry.status === 'warn';
            step.className = `ai-thinking-step${warn ? ' ai-thinking-warn' : ''}`;
            const meta = resolveMeta(entry);
            step.innerHTML = `
                <span class="ai-step-index">${meta.icon || String(idx + 1).padStart(2, '0')}</span>
                <div class="ai-step-body">
                    <div class="ai-step-head">
                        <span class="ai-step-model">${meta.label}</span>
                <span class="ai-step-status">${warn ? aiText.stepStatusWarn : aiText.stepStatusOk}</span>
                    </div>
                    <div class="ai-step-text"></div>
                </div>`;
            const textNode = step.querySelector('.ai-step-text');
            if (animate && textNode) {
                animateTypewriter(textNode, entry.body, formatAiAnswer(entry.body), {
                    delay: idx * 320,
                    speed: 12,
                });
            } else if (textNode) {
                textNode.innerHTML = formatAiAnswer(entry.body);
            }
            timeline.appendChild(step);
        });
        const accordion = document.createElement('div');
        accordion.className = 'ai-thinking-accordion';
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'ai-thinking-toggle';
        toggle.setAttribute('aria-expanded', 'false');
        const toggleLabel = (collapsed) =>
            langIsZh
                ? `${collapsed ? '展开' : '收起'} AI 推理（${normalized.length} 步）`
                : `${collapsed ? 'Expand' : 'Collapse'} AI reasoning (${normalized.length} steps)`;
        toggle.textContent = toggleLabel(true);
        const body = document.createElement('div');
        body.className = 'ai-thinking-body collapsed';
        body.appendChild(timeline);
        toggle.addEventListener('click', () => {
            const collapsed = body.classList.toggle('collapsed');
            toggle.setAttribute('aria-expanded', (!collapsed).toString());
            toggle.textContent = toggleLabel(collapsed);
        });
        accordion.appendChild(toggle);
        accordion.appendChild(body);
        aiThinking.appendChild(accordion);
    }

    if (chatHistory.length) {
        renderHistory(false);
        setAiStatus('info', aiText.statusReadyHistory);
    } else {
        setAiStatus('info', aiText.statusReadyFresh);
        if (lastAiAnswer) {
            appendMessage('assistant', lastAiAnswer, { animate: false, webUsed: reportContext && reportContext.enable_web });
        }
    }

    if (Array.isArray(reportContext.ai_thinking) && reportContext.ai_thinking.length) {
        renderThinking(reportContext.ai_thinking, false);
    }
    function handleAiProgress(payload) {
        if (!payload) return;
        if (payload.error) {
            setAiStatus('warn', payload.error);
            return;
        }
        if (payload.stage === 'delta' && payload.text) {
            // incremental append
            if (!streamingBubble) {
                appendMessage('assistant', '', { animate: false, trackStreaming: true, webUsed: webEnabled });
            }
            streamingBuffer += String(payload.text);
            if (streamingBubble) {
                streamingBubble.textContent = streamingBuffer;
                aiMessages.scrollTop = aiMessages.scrollHeight;
            }
            return;
        }
        if (payload.message) {
            setAiStatus('info', payload.message);
        }
    }

    function applyAiResponse(data) {
        stopThinkingAnimation();
        // finalize streaming bubble if exists
        if (streamingBubble) {
            streamingBubble.innerHTML = formatAiAnswer(data.answer || streamingBuffer || '');
            streamingBubble = null;
            streamingBuffer = '';
        }
        let historySupplied = false;
        if (Array.isArray(data.history) && data.history.length) {
            chatHistory = data.history;
            syncHistory();
            renderHistory(false);
            historySupplied = true;
        }
        const answer = data.answer || data.response || data.message;
        const webUsed = !!data.web_used;
        if (answer) {
            lastAiAnswer = answer;
            if (!historySupplied) {
                chatHistory.push({ role: 'assistant', content: answer, web_used: webUsed, web_results: data.web_results });
                syncHistory();
                if (!streamingBubble) {
                    appendMessage('assistant', answer, { webUsed, webResults: data.web_results });
                }
            }
        }
        if (Array.isArray(data.thinking)) {
            renderThinking(data.thinking);
            if (reportContext) {
                reportContext.ai_thinking = data.thinking;
            }
        } else if (reportContext) {
            reportContext.ai_thinking = [];
        }
        const statusText = data.summary || aiText.statusComplete;
        const note = data.web_note ? ` ${data.web_note}` : '';
        setAiStatus('success', statusText + note);
        const modelUsed =
            (Array.isArray(data.models) && data.models.length && data.models[0].name) ||
            data.selected_model ||
            reportContext.ai_model ||
            '';
        if (modelUsed) {
            reportContext.ai_model = modelUsed;
            if (modelInput) {
                modelInput.value = modelUsed;
            }
            ensureModelOption(modelUsed);
            try {
                window.sessionStorage.setItem(MODEL_STORAGE_KEY, modelUsed);
            } catch (error) {
                // ignore storage errors
            }
        }
        setAiMeta(data);
    }

    document.querySelectorAll('.ai-preset').forEach((btn) => {
        btn.addEventListener('click', () => {
            if (!aiInput) return;
            aiInput.value = btn.dataset.question || btn.textContent || '';
            aiInput.focus();
        });
    });

    if (aiClearBtn) {
        aiClearBtn.addEventListener('click', () => {
            if (!window.confirm(aiText.confirmClear)) return;
            chatHistory = [];
            lastAiAnswer = '';
            syncHistory();
            if (reportContext) {
                reportContext.ai_thinking = [];
            }
            if (aiMessages) aiMessages.innerHTML = '';
            if (aiThinking) {
                aiThinking.classList.add('d-none');
                aiThinking.innerHTML = '';
            }
            setAiStatus('info', aiText.statusHistoryCleared);
        });
    }

    if (aiForm && aiInput) {
        aiForm.addEventListener('submit', (event) => {
            event.preventDefault();
            const message = aiInput.value.trim();
            if (!message) return;
            if (!csrfToken) {
                setAiStatus(
                    'warn',
                    langIsZh ? '登录或 CSRF 失效，请刷新页面后重试。' : 'Login/CSRF token expired. Refresh the page and retry.'
                );
                return;
            }

            appendMessage('user', message);
            chatHistory.push({ role: 'user', content: message });
            syncHistory();
            streamingBuffer = '';
            streamingBubble = null;
            aiInput.value = '';
            const keywordRegex = /联网|上网|搜索|查网|找新闻|实时|news|资讯|internet|web|look up|fetch/gi;
            const shouldUseWeb = webEnabled || keywordRegex.test(message);
            if (!webEnabled && keywordRegex.test(message)) {
                appendMessage('assistant', aiText.promptEnableWeb);
                setAiStatus('warn', aiText.statusWebOff);
                stopThinkingAnimation();
                if (aiThinking) {
                    aiThinking.classList.add('d-none');
                    aiThinking.innerHTML = '';
                }
                return;
            }
            setAiStatus('info', shouldUseWeb ? aiText.statusFetching : aiText.statusAnalyzing);
            startThinkingAnimation();
            if (modelInput) {
                reportContext.ai_model = modelInput.value.trim();
            }
            const payloadHistory = chatHistory.slice(-HISTORY_WINDOW);

            requestAi(
                {
                    context: reportContext,
                    message,
                    history: payloadHistory,
                    show_thoughts: true,
                    enable_web: webEnabled,
                    model: reportContext.ai_model || '',
                },
                { onProgress: handleAiProgress }
            )
                .then((data) => {
                    applyAiResponse(data);
                })
                .catch((error) => {
                    console.error(error);
                    if (error && error.message === 'csrf_missing') {
                        setAiStatus(
                            'warn',
                            langIsZh
                                ? '登录或 CSRF 失效，请刷新页面后重试。'
                                : 'Login/CSRF token expired. Refresh the page and retry.'
                        );
                    } else if (error.name === 'AbortError') {
                        setAiStatus('warn', aiText.statusTimeout);
                    } else {
                        setAiStatus('warn', aiText.statusUnavailable);
                    }
                    stopThinkingAnimation();
                    appendMessage('assistant', aiText.apology, { animate: false });
                    if (aiThinking) {
                        aiThinking.classList.add('d-none');
                        aiThinking.innerHTML = '';
                    }
                });
        });
    }

    let autoAiRequested = false;
    function triggerInitialAiAnalysis() {
        if (autoAiRequested || !reportContext.include_ai || !aiEndpoint) return;
        if (!csrfToken) {
            setAiStatus(
                'warn',
                langIsZh ? '登录或 CSRF 失效，请刷新页面后重试。' : 'Login/CSRF token expired. Refresh the page and retry.'
            );
            return;
        }
        autoAiRequested = true;
        setAiStatus('info', aiText.statusSynthesizing);
        startThinkingAnimation();
        if (modelInput) {
            reportContext.ai_model = modelInput.value.trim();
        }
        const payloadHistory = chatHistory.slice(-HISTORY_WINDOW);
        const autoPrompt = buildAutoPrompt(reportContext);
        reportContext.ai_auto_prompt = autoPrompt;
        requestAi(
            {
                context: reportContext,
                message: autoPrompt,
                history: payloadHistory,
                show_thoughts: reportContext.show_ai_thoughts !== false,
                enable_web: webEnabled,
                model: reportContext.ai_model || '',
            },
            { onProgress: handleAiProgress }
        )
            .then((data) => {
                applyAiResponse(data);
            })
            .catch((error) => {
                console.error(error);
                if (error && error.message === 'csrf_missing') {
                    setAiStatus(
                        'warn',
                        langIsZh
                            ? '登录或 CSRF 失效，请刷新页面后重试。'
                            : 'Login/CSRF token expired. Refresh the page and retry.'
                    );
                } else if (error.name === 'AbortError') {
                    setAiStatus('warn', aiText.statusTimeout);
                } else {
                    setAiStatus('warn', aiText.statusSummaryMissing);
                }
                stopThinkingAnimation();
                if (aiThinking) {
                    aiThinking.classList.add('d-none');
                    aiThinking.innerHTML = '';
                }
            });
    }

    if (!chatHistory.length && reportContext.include_ai) {
        window.setTimeout(triggerInitialAiAnalysis, 500);
    }
});
})();
