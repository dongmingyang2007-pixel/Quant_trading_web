(() => {
    const closeMenus = (exception) => {
        document.querySelectorAll("[data-user-menu]").forEach((menu) => {
            if (exception && menu === exception) return;
            const panel = menu.querySelector("[data-menu-panel]");
            const toggle = menu.querySelector("[data-menu-toggle]");
            if (panel && !panel.hidden) {
                panel.hidden = true;
                menu.classList.remove("is-open");
                if (toggle) toggle.setAttribute("aria-expanded", "false");
            }
        });
    };

    document.querySelectorAll("[data-user-menu]").forEach((menu) => {
        const toggle = menu.querySelector("[data-menu-toggle]");
        const panel = menu.querySelector("[data-menu-panel]");
        if (!toggle || !panel) return;
        panel.hidden = true;
        toggle.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const willOpen = panel.hidden;
            closeMenus(willOpen ? null : undefined);
            if (willOpen) {
                panel.hidden = false;
                menu.classList.add("is-open");
                toggle.setAttribute("aria-expanded", "true");
            } else {
                panel.hidden = true;
                menu.classList.remove("is-open");
                toggle.setAttribute("aria-expanded", "false");
            }
        });
        panel.addEventListener("click", (event) => event.stopPropagation());
    });

    document.addEventListener("click", () => closeMenus());
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeMenus();
        }
    });
})();
