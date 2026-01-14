(() => {
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

            activateSettings = (name) => {
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
                    btn.setAttribute("aria-selected", String(isActive));
                });
            };

            const initialSettings =
                settingsNav.dataset.defaultSection || settingsButtons[0]?.dataset.settingsTarget;
            activateSettings(initialSettings);

            settingsButtons.forEach((btn) => {
                btn.addEventListener("click", () => activateSettings(btn.dataset.settingsTarget));
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
})();
