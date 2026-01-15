(() => {
    const init = () => {
        const page = document.querySelector(".account-page");
        const nav = document.querySelector("[data-role='account-tab-nav']");
        const panels = Array.from(document.querySelectorAll("[data-tab-panel]"));
        if (!page || !nav || !panels.length) {
            return;
        }
        const buttons = Array.from(nav.querySelectorAll("[data-tab-target]"));
        if (!buttons.length) {
            return;
        }

        page.classList.add("has-account-tabs");

        const SETTINGS_QUERY_KEY = "tab";
        const readSettingsQuery = () => {
            const params = new URLSearchParams(window.location.search);
            return params.get(SETTINGS_QUERY_KEY);
        };
        const updateSettingsQuery = (value) => {
            const url = new URL(window.location.href);
            if (value) {
                url.searchParams.set(SETTINGS_QUERY_KEY, value);
            } else {
                url.searchParams.delete(SETTINGS_QUERY_KEY);
            }
            window.history.pushState({ [SETTINGS_QUERY_KEY]: value }, "", url);
        };

        const findPanel = (name) => panels.find((panel) => panel.dataset.tabPanel === name);

        const activate = (name, { updateHash = true } = {}) => {
            if (!name) {
                return;
            }
            const panel = findPanel(name);
            if (!panel) {
                return;
            }
            panels.forEach((item) => item.classList.toggle("is-active", item === panel));
            buttons.forEach((btn) => {
                const isActive = btn.dataset.tabTarget === name;
                btn.classList.toggle("is-active", isActive);
                btn.setAttribute("aria-selected", String(isActive));
            });
            page.dataset.activeTab = name;
            page.classList.toggle("is-compact-header", name !== "overview");
            page.classList.toggle("is-settings-tab", name === "settings");
            if (updateHash) {
                const targetId = panel.id || `account-tab-${name}`;
                if (window.history && window.history.replaceState) {
                    window.history.replaceState(null, "", `#${targetId}`);
                } else {
                    window.location.hash = targetId;
                }
            }
        };

        const resolveInitial = () => {
            const hash = window.location.hash.replace("#", "");
            if (hash) {
                const direct = panels.find((panel) => panel.id === hash);
                if (direct) {
                    return direct.dataset.tabPanel;
                }
                const normalized = hash.replace("account-tab-", "");
                if (buttons.find((btn) => btn.dataset.tabTarget === normalized)) {
                    return normalized;
                }
            }
            return nav.dataset.defaultTab || (buttons[0] && buttons[0].dataset.tabTarget);
        };

        const initial = resolveInitial();
        activate(initial, { updateHash: false });

        buttons.forEach((btn) => {
            btn.addEventListener("click", () => activate(btn.dataset.tabTarget));
        });

        let activateSettings = null;

        document.querySelectorAll("[data-tab-link]").forEach((link) => {
            link.addEventListener("click", (event) => {
                const target = link.dataset.tabLink;
                if (!target) {
                    return;
                }
                event.preventDefault();
                activate(target);
                const panel = findPanel(target);
                const openSettings = link.dataset.openSettings;
                const scrollTarget = link.dataset.scrollTarget;
                if (target === "settings") {
                    if (activateSettings && openSettings) {
                        activateSettings(openSettings);
                    }
                    const settingsCard = document.querySelector("[data-role='settings-card']");
                    const body = settingsCard && settingsCard.querySelector("[data-role='settings-body']");
                    const toggleButton = settingsCard && settingsCard.querySelector("[data-role='toggle-settings']");
                    if (body && body.classList.contains("d-none")) {
                        body.classList.remove("d-none");
                        if (toggleButton) {
                            toggleButton.textContent = toggleButton.dataset.labelOpen || toggleButton.textContent;
                        }
                    }
                    if (scrollTarget) {
                        const focusTarget = document.querySelector(`[data-role='${scrollTarget}']`);
                        if (focusTarget) {
                            focusTarget.scrollIntoView({ behavior: "smooth", block: "start" });
                            return;
                        }
                    }
                }
                if (panel) {
                    panel.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            });
        });

        const settingsNav = document.querySelector("[data-role='settings-nav']");
        if (settingsNav) {
            const settingsPanels = Array.from(document.querySelectorAll("[data-settings-panel]"));
            const settingsButtons = Array.from(settingsNav.querySelectorAll("[data-settings-target]"));
            if (settingsPanels.length && settingsButtons.length) {
                const findSettingsPanel = (name) =>
                    settingsPanels.find((panel) => panel.dataset.settingsPanel === name);
                const hasSettingsTarget = (name) =>
                    settingsButtons.some((btn) => btn.dataset.settingsTarget === name);

                activateSettings = (name, { updateUrl = false } = {}) => {
                    if (!name) {
                        return;
                    }
                    const panel = findSettingsPanel(name);
                    if (!panel) {
                        return;
                    }
                    settingsPanels.forEach((item) => item.classList.toggle("is-active", item === panel));
                    settingsButtons.forEach((btn) => {
                        const isActive = btn.dataset.settingsTarget === name;
                        btn.classList.toggle("is-active", isActive);
                        btn.classList.toggle("active", isActive);
                        btn.setAttribute("aria-selected", String(isActive));
                    });
                    if (updateUrl) {
                        updateSettingsQuery(name);
                    }
                };

                const queryTab = readSettingsQuery();
                let initialSettings = hasSettingsTarget(queryTab)
                    ? queryTab
                    : settingsNav.dataset.defaultSection || "profile";
                if (!hasSettingsTarget(initialSettings)) {
                    initialSettings = settingsButtons[0]?.dataset.settingsTarget || "profile";
                }
                activateSettings(initialSettings);

                settingsButtons.forEach((btn) => {
                    btn.addEventListener("click", () =>
                        activateSettings(btn.dataset.settingsTarget, { updateUrl: true })
                    );
                });
            }
        }

        document.querySelectorAll(".account-media").forEach((card) => {
            const manage = card.querySelector("[data-role='media-manage']");
            if (!manage) {
                return;
            }
            const syncManage = () => {
                card.classList.toggle("is-managing", manage.open);
            };
            syncManage();
            manage.addEventListener("toggle", syncManage);
        });

        const filterContainer = document.querySelector("[data-role='timeline-filters']");
        if (filterContainer) {
            const filterButtons = Array.from(filterContainer.querySelectorAll("[data-timeline-filter]"));
            const timelineItems = Array.from(document.querySelectorAll("[data-timeline-item]"));
            const applyFilter = (kind) => {
                filterButtons.forEach((btn) => {
                    const active = btn.dataset.timelineFilter === kind;
                    btn.classList.toggle("is-active", active);
                });
                timelineItems.forEach((item) => {
                    const itemKind = item.dataset.timelineItem;
                    const show = kind === "all" || itemKind === kind;
                    item.classList.toggle("d-none", !show);
                });
            };
            if (filterButtons.length) {
                filterButtons.forEach((btn) => {
                    btn.addEventListener("click", () => {
                        applyFilter(btn.dataset.timelineFilter);
                    });
                });
            }
        }

        const timelineModal = document.getElementById("timeline-media-modal");
        if (timelineModal) {
            const modalImage = timelineModal.querySelector("[data-role='timeline-modal-image']");
            const mediaTriggers = Array.from(document.querySelectorAll("[data-role='timeline-media-trigger']"));
            const updateModal = (source) => {
                if (!modalImage) {
                    return;
                }
                modalImage.src = source || "";
            };
            mediaTriggers.forEach((trigger) => {
                trigger.addEventListener("click", () => {
                    updateModal(trigger.dataset.image);
                });
            });
        }

        document.querySelectorAll("[data-role='api-secret-field']").forEach((field) => {
            const input = field.querySelector("input");
            const toggle = field.querySelector("[data-role='api-secret-toggle']");
            const copy = field.querySelector("[data-role='api-secret-copy']");
            const showIcon = field.querySelector("[data-role='api-secret-icon-show']");
            const hideIcon = field.querySelector("[data-role='api-secret-icon-hide']");
            if (!input) {
                return;
            }
            if (input.getAttribute("type") !== "password") {
                input.setAttribute("type", "password");
            }
            const syncToggle = () => {
                if (!toggle) {
                    return;
                }
                const isHidden = input.getAttribute("type") === "password";
                toggle.setAttribute("aria-pressed", String(!isHidden));
                if (showIcon && hideIcon) {
                    showIcon.classList.toggle("d-none", !isHidden);
                    hideIcon.classList.toggle("d-none", isHidden);
                }
            };
            syncToggle();
            if (toggle) {
                toggle.addEventListener("click", (event) => {
                    event.preventDefault();
                    const isHidden = input.getAttribute("type") === "password";
                    input.setAttribute("type", isHidden ? "text" : "password");
                    syncToggle();
                });
            }
            if (copy) {
                const copyLabel = copy.textContent;
                const feedback = copy.dataset.copyFeedback || "Copied!";
                copy.addEventListener("click", async (event) => {
                    event.preventDefault();
                    const value = input.value || "";
                    if (!value) {
                        return;
                    }
                    const originalType = input.getAttribute("type");
                    if (originalType === "password") {
                        input.setAttribute("type", "text");
                    }
                    let copied = false;
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        try {
                            await navigator.clipboard.writeText(value);
                            copied = true;
                        } catch (err) {
                            copied = false;
                        }
                    }
                    if (!copied) {
                        input.focus();
                        input.select();
                        try {
                            copied = document.execCommand("copy");
                        } catch (err) {
                            copied = false;
                        }
                        input.setSelectionRange(0, 0);
                    }
                    if (originalType === "password") {
                        input.setAttribute("type", "password");
                    }
                    syncToggle();
                    if (copied) {
                        copy.textContent = feedback;
                        copy.disabled = true;
                        window.setTimeout(() => {
                            copy.textContent = copyLabel;
                            copy.disabled = false;
                        }, 1500);
                    }
                });
            }
        });
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }
})();
