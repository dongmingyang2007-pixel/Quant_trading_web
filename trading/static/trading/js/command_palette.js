(() => {
    const DATA_ID = "command-palette-data";
    const STORAGE_KEY = "qt_theme";

    const dataEl = document.getElementById(DATA_ID);
    const paletteEl = document.getElementById("command-palette");
    if (!dataEl || !paletteEl) {
        return;
    }

    const parseCommands = () => {
        try {
            return JSON.parse(dataEl.textContent || "[]");
        } catch (_error) {
            return [];
        }
    };

    const commands = parseCommands();
    const doc = document.documentElement;
    if (!doc.getAttribute("data-theme")) {
        doc.setAttribute("data-theme", "light");
    }
    const themeApi = window.qTheme || {
        getTheme: () => (doc.getAttribute("data-theme") === "dark" ? "dark" : "light"),
        toggleTheme: () => {
            const current = doc.getAttribute("data-theme") === "dark" ? "dark" : "light";
            const next = current === "dark" ? "light" : "dark";
            doc.setAttribute("data-theme", next);
            try {
                window.localStorage.setItem(STORAGE_KEY, next);
            } catch (_error) {
                // ignore
            }
        },
    };

    const inputEl = paletteEl.querySelector("[data-role='palette-input']");
    const listEl = paletteEl.querySelector("[data-role='palette-list']");
    const emptyEl = paletteEl.querySelector("[data-role='palette-empty']");
    const dismissEl = paletteEl.querySelector("[data-role='palette-dismiss']");

    const state = {
        filtered: commands.slice(),
        activeIndex: 0,
        visible: false,
    };

    const setBodyLock = (locked) => {
        document.body.classList.toggle("command-palette-open", locked);
    };

    const runCommand = (command) => {
        if (!command) return;
        if (command.type === "route" && command.href) {
            window.location.assign(command.href);
            closePalette();
            return;
        }
        if (command.type === "action" && command.action) {
            if (command.action === "toggleTheme") {
                themeApi.toggleTheme();
                renderList(inputEl.value || "");
            }
            closePalette();
        }
    };

    const highlightIndex = (index) => {
        state.activeIndex = index;
        const items = listEl.querySelectorAll("[data-command-index]");
        items.forEach((item) => {
            const match = Number(item.dataset.commandIndex) === state.activeIndex;
            item.classList.toggle("is-active", match);
            if (match) {
                item.scrollIntoView({ block: "nearest" });
            }
        });
    };

    const renderList = (query) => {
        const keyword = (query || "").trim().toLowerCase();
        const themedCommands = commands.slice();
        state.filtered = themedCommands.filter((cmd) => {
            if (!keyword) return true;
            return (
                (cmd.label && cmd.label.toLowerCase().includes(keyword)) ||
                (cmd.description && cmd.description.toLowerCase().includes(keyword)) ||
                (cmd.shortcut && cmd.shortcut.toLowerCase() === keyword)
            );
        });
        listEl.innerHTML = "";
        if (!state.filtered.length) {
            emptyEl.classList.remove("hidden");
            return;
        }
        emptyEl.classList.add("hidden");
        state.filtered.forEach((command, index) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "command-palette__item";
            button.dataset.commandIndex = String(index);
            button.dataset.commandId = command.id;
            button.innerHTML = `
                <div>
                    <div class="command-palette__item-title">${command.label || command.id}</div>
                    <div class="command-palette__item-desc">${command.description || ""}</div>
                </div>
                ${
                    command.shortcut
                        ? `<span class="command-palette__shortcut">${command.shortcut.toUpperCase()}</span>`
                        : ""
                }
            `;
            listEl.appendChild(button);
        });
        state.activeIndex = 0;
        highlightIndex(0);
    };

    const openPalette = () => {
        if (state.visible) return;
        state.visible = true;
        paletteEl.hidden = false;
        paletteEl.classList.add("is-visible");
        paletteEl.setAttribute("aria-hidden", "false");
        setBodyLock(true);
        renderList("");
        window.requestAnimationFrame(() => {
            inputEl.value = "";
            inputEl.focus();
        });
    };

    const closePalette = () => {
        if (!state.visible) return;
        state.visible = false;
        paletteEl.classList.remove("is-visible");
        paletteEl.hidden = true;
        paletteEl.setAttribute("aria-hidden", "true");
        setBodyLock(false);
        inputEl.blur();
    };

    const handleGlobalKey = (event) => {
        const isModifier = event.metaKey || event.ctrlKey;
        if (isModifier && event.key.toLowerCase() === "k") {
            event.preventDefault();
            if (state.visible) {
                closePalette();
            } else {
                openPalette();
            }
            return;
        }
        if (event.key === "Escape" && state.visible) {
            event.preventDefault();
            closePalette();
        }
    };

    const handleInputKey = (event) => {
        if (!state.visible) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            if (!state.filtered.length) return;
            const delta = event.key === "ArrowDown" ? 1 : -1;
            const nextIndex =
                (state.activeIndex + delta + state.filtered.length) % state.filtered.length;
            highlightIndex(nextIndex);
        } else if (event.key === "Enter") {
            event.preventDefault();
            runCommand(state.filtered[state.activeIndex]);
        }
    };

    const handleInput = () => {
        renderList(inputEl.value || "");
    };

    listEl.addEventListener("click", (event) => {
        const target = event.target.closest("[data-command-id]");
        if (!target) return;
        const index = Number(target.dataset.commandIndex);
        if (!Number.isNaN(index) && state.filtered[index]) {
            runCommand(state.filtered[index]);
        }
    });

    listEl.addEventListener("mousemove", (event) => {
        const target = event.target.closest("[data-command-index]");
        if (!target) return;
        highlightIndex(Number(target.dataset.commandIndex));
    });

    dismissEl?.addEventListener("click", () => closePalette());
    inputEl.addEventListener("keydown", handleInputKey);
    inputEl.addEventListener("input", handleInput);
    document.addEventListener("keydown", handleGlobalKey);
})();
