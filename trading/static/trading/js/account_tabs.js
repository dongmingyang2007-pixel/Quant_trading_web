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

    document.querySelectorAll("[data-tab-link]").forEach((link) => {
        link.addEventListener("click", (event) => {
            const target = link.dataset.tabLink;
            if (!target) {
                return;
            }
            event.preventDefault();
            activate(target);
            const panel = findPanel(target);
            if (panel) {
                panel.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        });
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
})();
