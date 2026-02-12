(() => {
    const userMenus = Array.from(document.querySelectorAll("[data-user-menu]"));
    const apiMenus = Array.from(document.querySelectorAll("[data-api-menu]"));
    const observerMenus = Array.from(document.querySelectorAll("[data-observer-menu]"));
    const floatingMenus = [];

    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

    const resetPanelStyles = (panel) => {
        if (!panel) return;
        panel.style.position = "";
        panel.style.top = "";
        panel.style.left = "";
        panel.style.right = "";
        panel.style.maxHeight = "";
        panel.style.width = "";
    };

    const isMenuOpen = (menu) => {
        if (!menu) return false;
        return menu.classList.contains("is-open");
    };

    const rememberFloatingMenu = (menu, toggle, panel) => {
        if (!menu || !toggle || !panel) return;
        if (floatingMenus.some((entry) => entry.menu === menu)) return;
        floatingMenus.push({ menu, toggle, panel });
    };

    const forgetFloatingMenu = (menu) => {
        if (!menu) return;
        const index = floatingMenus.findIndex((entry) => entry.menu === menu);
        if (index >= 0) {
            floatingMenus.splice(index, 1);
        }
    };

    const positionFloatingPanel = (toggle, panel) => {
        if (!toggle || !panel) return;
        const viewportPadding = 10;
        const spacing = 8;
        panel.style.position = "fixed";
        panel.style.right = "auto";
        panel.style.left = "0";
        panel.style.top = "0";
        panel.style.maxHeight = `${Math.max(220, Math.floor(window.innerHeight - viewportPadding * 2))}px`;
        panel.style.width = "";

        const toggleRect = toggle.getBoundingClientRect();
        const panelRect = panel.getBoundingClientRect();
        const panelWidth = panelRect.width;
        const panelHeight = panelRect.height;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        const preferredLeft = toggleRect.right - panelWidth;
        const left = clamp(preferredLeft, viewportPadding, Math.max(viewportPadding, viewportWidth - panelWidth - viewportPadding));

        let top = toggleRect.bottom + spacing;
        const maxBottom = viewportHeight - viewportPadding;
        if (top + panelHeight > maxBottom) {
            const candidateTop = toggleRect.top - panelHeight - spacing;
            if (candidateTop >= viewportPadding) {
                top = candidateTop;
            } else {
                top = viewportPadding;
            }
        }

        panel.style.left = `${Math.round(left)}px`;
        panel.style.top = `${Math.round(top)}px`;
    };

    const refreshOpenFloatingPanels = () => {
        floatingMenus.forEach(({ menu, toggle, panel }) => {
            if (!isMenuOpen(menu) || panel.hidden) {
                return;
            }
            positionFloatingPanel(toggle, panel);
        });
    };

    const closeUserMenus = (exception) => {
        userMenus.forEach((menu) => {
            if (exception && menu === exception) return;
            const panel = menu.querySelector("[data-menu-panel]");
            const toggle = menu.querySelector("[data-menu-toggle]");
            if (!panel) return;
            panel.hidden = true;
            menu.classList.remove("is-open");
            if (toggle) toggle.setAttribute("aria-expanded", "false");
            resetPanelStyles(panel);
            forgetFloatingMenu(menu);
        });
    };

    const closeApiMenus = (exception) => {
        apiMenus.forEach((menu) => {
            if (exception && menu === exception) return;
            const panel = menu.querySelector("[data-api-panel]");
            const toggle = menu.querySelector("[data-api-toggle]");
            if (!panel) return;
            panel.hidden = true;
            menu.classList.remove("is-open");
            if (toggle) toggle.setAttribute("aria-expanded", "false");
            resetPanelStyles(panel);
            forgetFloatingMenu(menu);
        });
    };

    const closeObserverMenus = (exception) => {
        observerMenus.forEach((menu) => {
            if (exception && menu === exception) return;
            const panel = menu.querySelector("[data-observer-panel]");
            const toggle = menu.querySelector("[data-observer-toggle]");
            if (!panel) return;
            panel.hidden = true;
            menu.classList.remove("is-open");
            if (toggle) toggle.setAttribute("aria-expanded", "false");
            resetPanelStyles(panel);
            forgetFloatingMenu(menu);
        });
    };

    const closeAllMenus = ({ keepUser = null, keepApi = null, keepObserver = null } = {}) => {
        closeUserMenus(keepUser);
        closeApiMenus(keepApi);
        closeObserverMenus(keepObserver);
    };

    userMenus.forEach((menu) => {
        const toggle = menu.querySelector("[data-menu-toggle]");
        const panel = menu.querySelector("[data-menu-panel]");
        if (!toggle || !panel) return;
        panel.hidden = true;
        toggle.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const willOpen = panel.hidden;
            closeAllMenus({ keepUser: willOpen ? menu : null });
            if (willOpen) {
                panel.hidden = false;
                menu.classList.add("is-open");
                toggle.setAttribute("aria-expanded", "true");
                positionFloatingPanel(toggle, panel);
                rememberFloatingMenu(menu, toggle, panel);
            } else {
                panel.hidden = true;
                menu.classList.remove("is-open");
                toggle.setAttribute("aria-expanded", "false");
                resetPanelStyles(panel);
                forgetFloatingMenu(menu);
            }
        });
        panel.addEventListener("click", (event) => event.stopPropagation());
    });

    const I18N = {
        zh: {
            updating: "更新中...",
            updatedAt: "更新",
            levelNormal: "正常",
            levelBusy: "较忙",
            levelHigh: "高频",
            levelError: "异常",
            observeLoading: "正在读取榜单状态...",
            observeReady: "状态正常",
            observeBuilding: (count) => `构建中 ${count} 项`,
            observeNoData: "暂无可用状态",
            observeError: "读取失败",
            observeUpdated: "生效",
            observeStale: "陈旧",
            observeProgress: "进度",
            observeUnknown: "未知",
            observeStable: "已生效",
            observePending: "等待首轮构建",
            observeUnsupported: "当前不支持",
            observeNextRefresh: "下一轮",
            observeCycle: "刷新周期",
            observeRefreshIdle: "等待刷新",
            observeStalled: "构建停滞",
        },
        en: {
            updating: "Updating...",
            updatedAt: "Updated",
            levelNormal: "Normal",
            levelBusy: "Busy",
            levelHigh: "High",
            levelError: "Error",
            observeLoading: "Loading leaderboard status...",
            observeReady: "Healthy",
            observeBuilding: (count) => `${count} building`,
            observeNoData: "No status available",
            observeError: "Failed to load",
            observeUpdated: "Active",
            observeStale: "Stale",
            observeProgress: "Progress",
            observeUnknown: "Unknown",
            observeStable: "Stable",
            observePending: "Pending first build",
            observeUnsupported: "Not supported",
            observeNextRefresh: "Next refresh",
            observeCycle: "Cycle",
            observeRefreshIdle: "Idle",
            observeStalled: "Stalled",
        },
    };

    const OBSERVE_LIST_LABELS = {
        zh: {
            gainers: "涨幅榜",
            losers: "跌幅榜",
            most_active: "活跃榜",
            top_turnover: "成交额",
        },
        en: {
            gainers: "Gainers",
            losers: "Losers",
            most_active: "Most Active",
            top_turnover: "Turnover",
        },
    };

    const OBSERVE_TIMEFRAME_LABELS = {
        zh: { "1d": "实时榜", "5d": "近5日", "1mo": "近1月", "6mo": "近6月" },
        en: { "1d": "Realtime", "5d": "5D", "1mo": "1M", "6mo": "6M" },
    };
    const OBSERVE_TIMEFRAME_ORDER = ["1d", "5d", "1mo", "6mo"];
    const OBSERVE_LIST_ORDER = ["gainers", "losers", "most_active", "top_turnover"];

    const formatClock = (lang) => {
        const now = new Date();
        try {
            return now.toLocaleTimeString(lang === "zh" ? "zh-CN" : "en-US", {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
            });
        } catch (_error) {
            return `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(
                now.getSeconds()
            ).padStart(2, "0")}`;
        }
    };

    const parseTimestamp = (value) => {
        if (typeof value === "number" && Number.isFinite(value)) {
            const millis = value > 10_000_000_000 ? value : value * 1000;
            return new Date(millis);
        }
        if (typeof value === "string" && value.trim()) {
            const parsed = Date.parse(value);
            if (!Number.isNaN(parsed)) return new Date(parsed);
        }
        return null;
    };

    const formatTimestamp = (value, lang) => {
        const date = parseTimestamp(value);
        if (!date) return "--";
        try {
            return date.toLocaleTimeString(lang === "zh" ? "zh-CN" : "en-US", {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
            });
        } catch (_error) {
            return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}:${String(
                date.getSeconds()
            ).padStart(2, "0")}`;
        }
    };

    apiMenus.forEach((menu) => {
        const toggle = menu.querySelector("[data-api-toggle]");
        const panel = menu.querySelector("[data-api-panel]");
        const usageUrl = String(menu.getAttribute("data-url") || "").trim();
        const lang = String(menu.getAttribute("data-lang") || "zh").trim().toLowerCase().startsWith("zh") ? "zh" : "en";
        const text = I18N[lang];
        const usageValue = menu.querySelector("[data-role='nav-api-usage-value']");
        const totalEl = menu.querySelector("[data-role='nav-api-total']");
        const massiveEl = menu.querySelector("[data-role='nav-api-massive']");
        const alpacaEl = menu.querySelector("[data-role='nav-api-alpaca']");
        const levelEl = menu.querySelector("[data-role='nav-api-level']");
        const updatedEl = menu.querySelector("[data-role='nav-api-updated']");
        let usageTimer = null;
        let inFlight = false;

        if (panel) {
            panel.hidden = true;
            panel.addEventListener("click", (event) => event.stopPropagation());
        }

        if (toggle && panel) {
            toggle.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                const willOpen = panel.hidden;
                closeAllMenus({ keepApi: willOpen ? menu : null });
                if (willOpen) {
                    panel.hidden = false;
                    menu.classList.add("is-open");
                    toggle.setAttribute("aria-expanded", "true");
                    positionFloatingPanel(toggle, panel);
                    rememberFloatingMenu(menu, toggle, panel);
                } else {
                    panel.hidden = true;
                    menu.classList.remove("is-open");
                    toggle.setAttribute("aria-expanded", "false");
                    resetPanelStyles(panel);
                    forgetFloatingMenu(menu);
                }
            });
        }

        if (updatedEl) {
            updatedEl.textContent = text.updating;
        }

        const renderUsage = (payload) => {
            const total = Number(payload && payload.user_total);
            const byProvider = (payload && payload.user_by_provider) || {};
            const massive = Number(byProvider.massive || 0);
            const alpaca = Number(byProvider.alpaca || 0);
            const safeTotal = Number.isFinite(total) ? total : 0;
            const safeMassive = Number.isFinite(massive) ? massive : 0;
            const safeAlpaca = Number.isFinite(alpaca) ? alpaca : 0;

            if (usageValue) usageValue.textContent = String(safeTotal);
            if (totalEl) totalEl.textContent = String(safeTotal);
            if (massiveEl) massiveEl.textContent = String(safeMassive);
            if (alpacaEl) alpacaEl.textContent = String(safeAlpaca);

            let level = text.levelNormal;
            if (safeTotal > 180) level = text.levelHigh;
            else if (safeTotal > 60) level = text.levelBusy;
            if (levelEl) levelEl.textContent = level;

            if (updatedEl) {
                updatedEl.textContent = `${text.updatedAt} ${formatClock(lang)}`;
            }
            menu.classList.remove("is-error");
        };

        const renderError = () => {
            if (usageValue) usageValue.textContent = "--";
            if (totalEl) totalEl.textContent = "--";
            if (massiveEl) massiveEl.textContent = "--";
            if (alpacaEl) alpacaEl.textContent = "--";
            if (levelEl) levelEl.textContent = text.levelError;
            if (updatedEl) updatedEl.textContent = text.updating;
            menu.classList.add("is-error");
        };

        const updateUsage = async () => {
            if (!usageUrl || inFlight) return;
            inFlight = true;
            try {
                const response = await fetch(usageUrl, {
                    method: "GET",
                    credentials: "same-origin",
                    headers: { Accept: "application/json" },
                });
                if (!response.ok) throw new Error(`http_${response.status}`);
                const payload = await response.json();
                renderUsage(payload);
            } catch (_error) {
                renderError();
            } finally {
                inFlight = false;
            }
        };

        updateUsage();
        usageTimer = window.setInterval(updateUsage, 5000);
        window.addEventListener("beforeunload", () => {
            if (usageTimer) {
                window.clearInterval(usageTimer);
                usageTimer = null;
            }
        });
    });

    observerMenus.forEach((menu) => {
        const toggle = menu.querySelector("[data-observer-toggle]");
        const panel = menu.querySelector("[data-observer-panel]");
        const statusUrl = String(menu.getAttribute("data-url") || "").trim();
        const lang = String(menu.getAttribute("data-lang") || "zh").trim().toLowerCase().startsWith("zh") ? "zh" : "en";
        const text = I18N[lang];
        const listLabels = OBSERVE_LIST_LABELS[lang] || OBSERVE_LIST_LABELS.zh;
        const timeframeLabels = OBSERVE_TIMEFRAME_LABELS[lang] || OBSERVE_TIMEFRAME_LABELS.zh;
        const countEl = menu.querySelector("[data-role='nav-observer-count']");
        const summaryEl = menu.querySelector("[data-role='nav-observer-summary']");
        const listEl = menu.querySelector("[data-role='nav-observer-list']");
        let statusTimer = null;
        let inFlight = false;

        if (panel) {
            panel.hidden = true;
            panel.addEventListener("click", (event) => event.stopPropagation());
        }

        if (toggle && panel) {
            toggle.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                const willOpen = panel.hidden;
                closeAllMenus({ keepObserver: willOpen ? menu : null });
                if (willOpen) {
                    panel.hidden = false;
                    menu.classList.add("is-open");
                    toggle.setAttribute("aria-expanded", "true");
                    positionFloatingPanel(toggle, panel);
                    rememberFloatingMenu(menu, toggle, panel);
                } else {
                    panel.hidden = true;
                    menu.classList.remove("is-open");
                    toggle.setAttribute("aria-expanded", "false");
                    resetPanelStyles(panel);
                    forgetFloatingMenu(menu);
                }
            });
        }

        const renderObserverError = () => {
            if (countEl) countEl.textContent = "!";
            if (summaryEl) summaryEl.textContent = text.observeError;
            if (listEl) listEl.innerHTML = "";
            menu.classList.add("is-error");
        };

        const buildObserverGroups = (payload) => {
            const groupsFromApi = Array.isArray(payload && payload.groups) ? payload.groups : [];
            if (groupsFromApi.length) {
                const normalized = groupsFromApi
                    .map((group) => {
                        const timeframe = group && typeof group.timeframe === "object" ? group.timeframe : {};
                        const timeframeKey = String(timeframe.key || "");
                        if (!timeframeKey) return null;
                        const rows = Array.isArray(group.items) ? group.items : [];
                        const byList = new Map();
                        rows.forEach((row) => {
                            const listType = String(row && row.list_type ? row.list_type : "");
                            if (!listType) return;
                            byList.set(listType, row);
                        });
                        const items = OBSERVE_LIST_ORDER.map((listType) => {
                            const row = byList.get(listType);
                            if (row && typeof row === "object") {
                                return {
                                    ...row,
                                    list_type: listType,
                                    supported: row.supported !== false,
                                };
                            }
                            return {
                                list_type: listType,
                                active_generated_at: null,
                                building: false,
                                progress: 0,
                                build_progress: 0,
                                cycle_progress: 0,
                                build_state: "idle",
                                provider: payload && payload.provider ? payload.provider : "unknown",
                                stale_seconds: null,
                                supported: false,
                            };
                        });
                        return {
                            timeframe: {
                                key: timeframeKey,
                                label: timeframe.label,
                                label_en: timeframe.label_en,
                            },
                            items,
                        };
                    })
                    .filter(Boolean);
                normalized.sort((a, b) => {
                    const ai = OBSERVE_TIMEFRAME_ORDER.indexOf(a.timeframe.key);
                    const bi = OBSERVE_TIMEFRAME_ORDER.indexOf(b.timeframe.key);
                    const av = ai === -1 ? Number.MAX_SAFE_INTEGER : ai;
                    const bv = bi === -1 ? Number.MAX_SAFE_INTEGER : bi;
                    return av - bv;
                });
                return normalized;
            }

            const flatItems = Array.isArray(payload && payload.items) ? payload.items : [];
            if (!flatItems.length) return [];
            const grouped = new Map();
            flatItems.forEach((row) => {
                if (!row || typeof row !== "object") return;
                const timeframe = row.timeframe && typeof row.timeframe === "object" ? row.timeframe : {};
                const timeframeKey = String(timeframe.key || "");
                const listType = String(row.list_type || "");
                if (!timeframeKey || !listType) return;
                if (!grouped.has(timeframeKey)) {
                    grouped.set(timeframeKey, {
                        timeframe: {
                            key: timeframeKey,
                            label: timeframe.label,
                            label_en: timeframe.label_en,
                        },
                        byList: new Map(),
                    });
                }
                grouped.get(timeframeKey).byList.set(listType, row);
            });
            const groups = [];
            grouped.forEach((group) => {
                const items = OBSERVE_LIST_ORDER.map((listType) => {
                    const row = group.byList.get(listType);
                    if (row && typeof row === "object") {
                        return {
                            ...row,
                            list_type: listType,
                            supported: row.supported !== false,
                        };
                    }
                    return {
                        list_type: listType,
                        active_generated_at: null,
                        building: false,
                        progress: 0,
                        build_progress: 0,
                        cycle_progress: 0,
                        build_state: "idle",
                        provider: payload && payload.provider ? payload.provider : "unknown",
                        stale_seconds: null,
                        supported: false,
                    };
                });
                groups.push({
                    timeframe: group.timeframe,
                    items,
                });
            });
            groups.sort((a, b) => {
                const ai = OBSERVE_TIMEFRAME_ORDER.indexOf(a.timeframe.key);
                const bi = OBSERVE_TIMEFRAME_ORDER.indexOf(b.timeframe.key);
                const av = ai === -1 ? Number.MAX_SAFE_INTEGER : ai;
                const bv = bi === -1 ? Number.MAX_SAFE_INTEGER : bi;
                return av - bv;
            });
            return groups;
        };

        const renderObserverRows = (payload) => {
            const groups = buildObserverGroups(payload);
            const allItems = groups.flatMap((group) => (Array.isArray(group.items) ? group.items : []));
            const buildingCount = allItems.filter((item) => {
                if (!item || item.supported === false) return false;
                const state = String(item.build_state || "").toLowerCase();
                return state === "running";
            }).length;
            const hasAnyActive = allItems.some((item) => {
                if (!item || item.supported === false) return false;
                return Boolean(parseTimestamp(item.active_generated_at));
            });
            if (countEl) {
                if (buildingCount > 0) {
                    countEl.textContent = String(buildingCount);
                } else {
                    countEl.textContent = hasAnyActive ? "OK" : "--";
                }
            }
            if (summaryEl) {
                if (!groups.length) {
                    summaryEl.textContent = text.observeNoData;
                } else if (buildingCount > 0) {
                    summaryEl.textContent = text.observeBuilding(buildingCount);
                } else if (!hasAnyActive) {
                    summaryEl.textContent = text.observePending;
                } else {
                    summaryEl.textContent = text.observeReady;
                }
            }
            if (listEl) {
                listEl.innerHTML = "";
                groups.forEach((group) => {
                    const timeframe = group.timeframe && typeof group.timeframe === "object" ? group.timeframe : {};
                    const tfKey = String(timeframe.key || "");
                    const groupSection = document.createElement("section");
                    groupSection.className = "nav-observer-group";
                    const groupTitle = document.createElement("h4");
                    groupTitle.className = "nav-observer-group-title";
                    groupTitle.textContent =
                        (lang === "zh" ? timeframe.label : timeframe.label_en) ||
                        timeframeLabels[tfKey] ||
                        tfKey ||
                        text.observeUnknown;
                    const groupHeader = document.createElement("div");
                    groupHeader.className = "nav-observer-group-header";
                    groupHeader.appendChild(groupTitle);
                    const groupMeta = document.createElement("div");
                    groupMeta.className = "nav-observer-group-meta";
                    const groupProgress = document.createElement("div");
                    groupProgress.className = "nav-observer-group-progress";
                    const groupProgressTrack = document.createElement("div");
                    groupProgressTrack.className = "nav-observer-progress-track";
                    const groupProgressFill = document.createElement("span");
                    groupProgressFill.className = "nav-observer-progress-fill";
                    groupProgressTrack.appendChild(groupProgressFill);
                    groupProgress.appendChild(groupProgressTrack);
                    const groupProgressValue = document.createElement("span");
                    groupProgressValue.className = "nav-observer-progress-value";
                    const groupItems = Array.isArray(group.items) ? group.items : [];
                    const runningItems = groupItems.filter((item) => {
                        if (!item || item.supported === false) return false;
                        return String(item.build_state || "").toLowerCase() === "running";
                    });
                    const activeItems = groupItems.filter((item) => {
                        if (!item || item.supported === false) return false;
                        return Boolean(parseTimestamp(item.active_generated_at));
                    });
                    const stalledItems = groupItems.filter((item) => {
                        if (!item || item.supported === false) return false;
                        return String(item.build_state || "").toLowerCase() === "stalled";
                    });
                    const errorItems = groupItems.filter((item) => {
                        if (!item || item.supported === false) return false;
                        const state = String(item.build_state || "").toLowerCase();
                        return state === "error";
                    });
                    let groupBuildProgress = 0;
                    if (runningItems.length) {
                        const progressValues = runningItems
                            .map((item) => Math.max(0, Math.min(100, Number(item && item.build_progress) || 0)));
                        groupBuildProgress = Math.round(
                            progressValues.reduce((total, value) => total + value, 0) / progressValues.length
                        );
                        groupMeta.textContent = `${text.observeProgress} ${groupBuildProgress}%`;
                    } else if (stalledItems.length) {
                        groupMeta.textContent = text.observeStalled;
                    } else if (errorItems.length) {
                        groupMeta.textContent = text.observeError;
                    } else if (!activeItems.length) {
                        groupMeta.textContent = text.observePending;
                    } else {
                        groupMeta.textContent = text.observeStable;
                    }
                    groupProgressValue.textContent = `${groupBuildProgress}%`;
                    groupProgressFill.style.width = `${groupBuildProgress}%`;
                    groupProgress.classList.toggle("is-hidden", !runningItems.length);
                    groupHeader.appendChild(groupMeta);
                    if (runningItems.length) {
                        groupHeader.appendChild(groupProgress);
                        groupHeader.appendChild(groupProgressValue);
                    }
                    groupSection.appendChild(groupHeader);
                    const groupList = document.createElement("div");
                    groupList.className = "nav-observer-group-list";
                    groupItems.forEach((item) => {
                        const row = document.createElement("div");
                        row.className = "nav-observer-item";
                        const listKey = String(item && item.list_type ? item.list_type : "");
                        const supported = item && item.supported !== false;
                        const buildState = String(item && item.build_state ? item.build_state : "").toLowerCase();
                        const isBuilding = supported && buildState === "running";
                        const isBuildError = supported && buildState === "error";
                        const isBuildStalled = supported && buildState === "stalled";
                        const isSchemaInvalid = supported && item && item.active_schema_valid === false;
                        const stale = Number(item && item.stale_seconds);
                        const staleText =
                            Number.isFinite(stale) && stale > 0
                                ? ` · ${text.observeStale} ${Math.max(0, Math.round(stale))}s`
                                : "";
                        row.classList.toggle("is-building", isBuilding);
                        row.classList.toggle(
                            "is-ready",
                            supported &&
                                !isBuilding &&
                                !isBuildError &&
                                !isBuildStalled &&
                                !isSchemaInvalid &&
                                parseTimestamp(item && item.active_generated_at)
                        );
                        row.classList.toggle(
                            "is-stale",
                            supported &&
                                !isBuilding &&
                                !isBuildError &&
                                !isBuildStalled &&
                                !isSchemaInvalid &&
                                !parseTimestamp(item && item.active_generated_at)
                        );
                        row.classList.toggle("is-error", isBuildError);
                        row.classList.toggle("is-stalled", isBuildStalled);
                        row.classList.toggle("is-unsupported", !supported);
                        const primary = document.createElement("div");
                        primary.className = "nav-observer-primary";
                        primary.textContent = listLabels[listKey] || listKey || text.observeUnknown;
                        const secondary = document.createElement("div");
                        secondary.className = "nav-observer-secondary";
                        const stamp = formatTimestamp(item && item.active_generated_at, lang);
                        const hasActiveStamp = Boolean(parseTimestamp(item && item.active_generated_at));
                        if (!supported) {
                            secondary.textContent = text.observeUnsupported;
                        } else if (isBuilding) {
                            secondary.textContent = `${text.observeUpdated} ${stamp} · ${text.observeProgress}`;
                        } else if (isBuildError) {
                            secondary.textContent = `${text.observeUpdated} ${stamp} · ${text.observeError}`;
                        } else if (isBuildStalled) {
                            secondary.textContent = `${text.observeUpdated} ${stamp} · ${text.observeStalled}`;
                        } else if (isSchemaInvalid) {
                            secondary.textContent = `${text.observeUpdated} ${stamp} · ${text.observeError}`;
                        } else if (!hasActiveStamp) {
                            secondary.textContent = text.observePending;
                        } else {
                            secondary.textContent = `${text.observeUpdated} ${stamp} · ${text.observeStable}${staleText}`;
                        }
                        row.appendChild(primary);
                        row.appendChild(secondary);
                        groupList.appendChild(row);
                    });
                    groupSection.appendChild(groupList);
                    listEl.appendChild(groupSection);
                });
            }
            menu.classList.remove("is-error");
        };

        const updateObserver = async () => {
            if (!statusUrl || inFlight) return;
            inFlight = true;
            try {
                const response = await fetch(statusUrl, {
                    method: "GET",
                    credentials: "same-origin",
                    headers: { Accept: "application/json" },
                });
                if (!response.ok) throw new Error(`http_${response.status}`);
                const payload = await response.json();
                renderObserverRows(payload);
            } catch (_error) {
                renderObserverError();
            } finally {
                inFlight = false;
            }
        };

        if (summaryEl) summaryEl.textContent = text.observeLoading;
        updateObserver();
        statusTimer = window.setInterval(updateObserver, 6000);
        window.addEventListener("beforeunload", () => {
            if (statusTimer) {
                window.clearInterval(statusTimer);
                statusTimer = null;
            }
        });
    });

    document.addEventListener("click", () => closeAllMenus());
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeAllMenus();
        }
    });
    window.addEventListener("resize", refreshOpenFloatingPanels, { passive: true });
    window.addEventListener("scroll", refreshOpenFloatingPanels, { passive: true });
})();
