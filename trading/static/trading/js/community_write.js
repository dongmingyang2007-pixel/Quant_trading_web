(() => {
    const form = document.querySelector("[data-role='write-form']");
    if (!form) return;

    const editorContainer = form.querySelector("#editor") || form.querySelector("[data-role='editor']");
    const contentInput = form.querySelector("[data-role='content-input']");
    const actionInput = form.querySelector("[data-role='action-input']");
    const saveDraftBtn = form.querySelector("[data-role='save-draft']");
    const statusEl = form.querySelector("[data-role='draft-status']");
    const postIdInput = form.querySelector("[data-role='post-id']");
    const titleInput = form.querySelector("input[name='title']");
    const editBase = form.dataset.editBase || "";
    const uploadUrl = form.dataset.uploadUrl || editorContainer?.dataset.uploadUrl || "";
    const draftsEndpoint = form.dataset.draftsEndpoint || "";
    const deleteBase = form.dataset.deleteBase || "";
    const csrfToken = form.querySelector("input[name='csrfmiddlewaretoken']")?.value || "";
    const draftsDrawer = document.getElementById("draftsDrawer");
    const draftsList = draftsDrawer?.querySelector("[data-role='drafts-list']");
    const draftsEmpty = draftsDrawer?.querySelector("[data-role='drafts-empty']");
    const draftNewBtn = draftsDrawer?.querySelector("[data-role='draft-new']");
    const coverInput = form.querySelector("[data-role='cover-input']");
    const coverPreview = form.querySelector("[data-role='cover-preview']");
    const coverImage = form.querySelector("[data-role='cover-image']");
    const coverPlaceholder = form.querySelector("[data-role='cover-placeholder']");
    const coverPickers = form.querySelectorAll("[data-role='cover-picker'], [data-role='cover-change']");
    const coverRemoveBtn = form.querySelector("[data-role='cover-remove']");
    const wordCountEl = form.querySelector("[data-role='word-count']");

    const statusMessages = {
        saving: statusEl?.dataset.savingText || "Saving draft...",
        saved: statusEl?.dataset.savedText || "Draft saved.",
        failed: statusEl?.dataset.failedText || "Save failed. Please try again.",
    };

    const mathModal = document.getElementById("mathModal");
    const mathField = mathModal?.querySelector("#math-input");
    const mathInsertBtn = mathModal?.querySelector("[data-role='math-insert']");
    const mathCancelBtn = mathModal?.querySelector("[data-role='math-cancel']");
    const mathCloseBtn = mathModal?.querySelector("[data-role='math-close']");
    const backtestModal = document.getElementById("backtestSelectModal");
    const backtestListContainer = document.getElementById("backtest-list-container");
    let mathModalInstance = null;
    let backtestModalInstance = null;
    let lastFormulaRange = null;
    let backtestInsertRange = null;

    const AUTO_SAVE_DELAY = 1200;
    const AUTO_SAVE_INTERVAL = 30000;

    let quill = null;
    let pendingSave = false;
    let isSaving = false;
    let isPublishing = false;
    let lastSavedSnapshot = "";
    let autoSaveTimer = null;
    let intervalId = null;
    let coverObjectUrl = "";

    const getCookie = (name) => {
        if (!document.cookie) return "";
        const cookies = document.cookie.split(";").map((cookie) => cookie.trim());
        for (const cookie of cookies) {
            if (cookie.startsWith(`${name}=`)) {
                return decodeURIComponent(cookie.substring(name.length + 1));
            }
        }
        return "";
    };

    const getCsrfToken = () =>
        getCookie("csrftoken") || form.querySelector("input[name='csrfmiddlewaretoken']")?.value || "";

    const normalizeHtml = (html) => {
        const trimmed = (html || "").trim();
        return trimmed === "<p><br></p>" ? "" : trimmed;
    };

    const resolveFormAction = () => form.getAttribute("action") || window.location.pathname;
    const resolveDeleteUrl = (postId) => {
        const base = deleteBase || "/community/delete/0/";
        if (base.includes("/0/")) {
            return base.replace(/\/0\/?$/, `/${postId}/`);
        }
        return `${base}${postId}/`;
    };

    const updateDraftUrl = (postId, replace = true) => {
        const url = new URL(window.location.href);
        if (postId) {
            url.searchParams.set("id", postId);
        } else {
            url.searchParams.delete("id");
        }
        const newUrl = `${url.pathname}${url.search}`;
        if (replace) {
            window.history.replaceState(null, "", newUrl);
        } else {
            window.history.pushState(null, "", newUrl);
        }
    };

    const syncContent = () => {
        if (!quill || !contentInput) return;
        contentInput.value = normalizeHtml(quill.root.innerHTML);
    };

    const setStatus = (message) => {
        if (!statusEl) return;
        statusEl.textContent = message || "";
    };

    const countWords = (text) => {
        const normalized = (text || "").replace(/\s+/g, " ").trim();
        if (!normalized) return 0;
        const cjkMatches = normalized.match(/[\u4E00-\u9FFF]/g) || [];
        const wordMatches = normalized.match(/[A-Za-z0-9_'-]+/g) || [];
        if (cjkMatches.length || wordMatches.length) {
            return cjkMatches.length + wordMatches.length;
        }
        return normalized.split(" ").filter(Boolean).length;
    };

    const updateWordCount = () => {
        if (!wordCountEl) return;
        const text = quill ? quill.getText() : contentInput?.value || "";
        const count = countWords(text);
        const label = wordCountEl.dataset.label || "Words";
        wordCountEl.textContent = `${label} ${count}`;
    };

    const clearCoverPreview = () => {
        if (!coverPreview || !coverPlaceholder || !coverImage) return;
        if (coverObjectUrl) {
            URL.revokeObjectURL(coverObjectUrl);
            coverObjectUrl = "";
        }
        coverImage.removeAttribute("src");
        coverPreview.classList.add("d-none");
        coverPlaceholder.classList.remove("d-none");
        if (coverInput) {
            coverInput.value = "";
        }
    };

    const showCoverPreview = (file) => {
        if (!coverPreview || !coverPlaceholder || !coverImage || !file) return;
        if (coverObjectUrl) {
            URL.revokeObjectURL(coverObjectUrl);
        }
        coverObjectUrl = URL.createObjectURL(file);
        coverImage.src = coverObjectUrl;
        coverPreview.classList.remove("d-none");
        coverPlaceholder.classList.add("d-none");
    };

    const openCoverPicker = () => {
        if (coverInput) {
            coverInput.click();
        }
    };

    const generateSparklineSVG = (dataPoints, width = 200, height = 44) => {
        if (!Array.isArray(dataPoints) || dataPoints.length < 2) return "";
        const values = dataPoints.map((value) => Number(value)).filter((value) => Number.isFinite(value));
        if (values.length < 2) return "";
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal || 1;
        const padding = 2;
        const innerWidth = Math.max(width - padding * 2, 1);
        const innerHeight = Math.max(height - padding * 2, 1);
        const step = values.length > 1 ? innerWidth / (values.length - 1) : innerWidth;
        const points = values.map((value, index) => {
            const x = padding + index * step;
            const y = padding + (1 - (value - minVal) / range) * innerHeight;
            return `${x.toFixed(2)},${y.toFixed(2)}`;
        });
        const path = points.reduce((acc, point, index) => `${acc}${index === 0 ? "M" : " L"} ${point}`, "");
        return `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-hidden="true"><path d="${path}" stroke="#0d6efd" stroke-width="1.5" fill="none"/></svg>`;
    };

    const openBacktestModal = () => {
        if (!backtestModal) return;
        if (!csrfToken) {
            alert("Missing CSRF token.");
            return;
        }
        backtestInsertRange = quill ? quill.getSelection(true) : null;
        if (window.bootstrap && typeof window.bootstrap.Modal === "function") {
            if (!backtestModalInstance) {
                backtestModalInstance = new window.bootstrap.Modal(backtestModal);
            }
            backtestModalInstance.show();
        } else {
            backtestModal.removeAttribute("aria-hidden");
        }
    };

    const renderBacktestList = (items) => {
        if (!backtestListContainer) return;
        backtestListContainer.innerHTML = "";
        if (!items || items.length === 0) {
            const empty = document.createElement("div");
            empty.className = "text-muted";
            empty.textContent = "No backtests found.";
            backtestListContainer.appendChild(empty);
            return;
        }
        const list = document.createElement("div");
        list.className = "d-flex flex-column gap-3";
        items.forEach((item) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "backtest-list-item text-start";
            button.dataset.id = item.id || "";
            button.dataset.name = item.strategy_name || "";
            button.dataset.return = item.total_return || "";

            const header = document.createElement("div");
            header.className = "backtest-list-header";

            const name = document.createElement("div");
            name.className = "backtest-list-name";
            name.textContent = item.strategy_name || "Strategy";

            const returnEl = document.createElement("div");
            returnEl.className = "backtest-list-return";
            const returnValue = item.total_return || "--";
            const parsedReturn = Number.parseFloat(String(returnValue).replace("%", ""));
            if (Number.isFinite(parsedReturn)) {
                returnEl.classList.add(parsedReturn >= 0 ? "is-positive" : "is-negative");
            } else {
                returnEl.classList.add("is-neutral");
            }
            returnEl.textContent = returnValue;

            header.appendChild(name);
            header.appendChild(returnEl);

            const meta = document.createElement("div");
            meta.className = "backtest-list-meta";
            meta.textContent = item.created_at || "";

            button.appendChild(header);
            button.appendChild(meta);

            button.addEventListener("click", () => {
                if (!quill) return;
                const payload = {
                    id: item.id || "",
                    name: item.strategy_name || "Strategy",
                    return: item.total_return || "--",
                    sharpe: item.sharpe || "N/A",
                    max_drawdown: item.max_drawdown || "N/A",
                    win_rate: item.win_rate || "N/A",
                    date: item.created_at || "",
                    equity_curve: Array.isArray(item.equity_curve) ? item.equity_curve : null,
                };
                const range = backtestInsertRange || quill.getSelection(true) || { index: quill.getLength(), length: 0 };
                quill.insertEmbed(range.index, "backtestCard", payload, "user");
                quill.insertText(range.index + 1, "\n", "user");
                quill.setSelection(range.index + 2, 0, "silent");
                backtestInsertRange = null;
                if (backtestModalInstance) {
                    backtestModalInstance.hide();
                } else if (backtestModal) {
                    backtestModal.setAttribute("aria-hidden", "true");
                }
            });

            list.appendChild(button);
        });
        backtestListContainer.appendChild(list);
    };

    const loadBacktests = async () => {
        if (!backtestListContainer) return;
        backtestListContainer.innerHTML =
            '<div class="d-flex align-items-center justify-content-center gap-2 text-muted"><div class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></div><span>Loading backtests...</span></div>';
        try {
            const response = await fetch("/api/get_user_backtests/", {
                method: "GET",
                credentials: "same-origin",
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                    "X-CSRFToken": csrfToken,
                },
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload?.error || `加载失败（${response.status}）`);
            }
            renderBacktestList(payload.backtests || []);
        } catch (error) {
            backtestListContainer.innerHTML = "";
            const fallback = document.createElement("div");
            fallback.className = "text-danger";
            fallback.textContent = error.message || "加载回测失败，请稍后重试。";
            backtestListContainer.appendChild(fallback);
        }
    };

    const showMathModal = () => {
        if (!mathModal) return;
        if (quill) {
            lastFormulaRange = quill.getSelection(true);
        }
        if (window.bootstrap && typeof window.bootstrap.Modal === "function") {
            if (!mathModalInstance) {
                mathModalInstance = new window.bootstrap.Modal(mathModal);
            }
            mathModalInstance.show();
        } else {
            mathModal.removeAttribute("aria-hidden");
        }
        if (mathField && typeof mathField.focus === "function") {
            setTimeout(() => {
                mathField.focus();
                if (typeof mathField.executeCommand === "function") {
                    mathField.executeCommand("showVirtualKeyboard");
                } else if (window.mathVirtualKeyboard?.show) {
                    window.mathVirtualKeyboard.show();
                }
            }, 80);
        }
    };

    const hideMathKeyboard = () => {
        if (mathField && typeof mathField.executeCommand === "function") {
            try {
                mathField.executeCommand("hideVirtualKeyboard");
            } catch (error) {
                // Ignore MathLive command errors and fall back to global API.
            }
        }
        if (window.mathVirtualKeyboard?.hide) {
            window.mathVirtualKeyboard.hide();
        }
    };

    const hideMathModal = () => {
        hideMathKeyboard();
        if (mathModalInstance) {
            mathModalInstance.hide();
        } else if (mathModal) {
            mathModal.setAttribute("aria-hidden", "true");
        }
    };

    const getTopicValue = () => {
        const selected = form.querySelector("input[name='topic']:checked");
        if (selected) return selected.value || "";
        const select = form.querySelector("select[name='topic']");
        return select ? select.value || "" : "";
    };

    const getSnapshot = () => {
        const title = (titleInput?.value || "").trim();
        const content = quill ? normalizeHtml(quill.root.innerHTML) : normalizeHtml(contentInput?.value || "");
        const topic = getTopicValue();
        return JSON.stringify({ title, content, topic });
    };

    const buildPayload = (action) => ({
        action,
        status: action === "publish" ? "published" : "draft",
        title: (titleInput?.value || "").trim(),
        content: quill ? normalizeHtml(quill.root.innerHTML) : normalizeHtml(contentInput?.value || ""),
        topic: getTopicValue(),
        post_id: postIdInput?.value || "",
    });

    const markDirty = () => {
        pendingSave = true;
        if (autoSaveTimer) {
            window.clearTimeout(autoSaveTimer);
        }
        autoSaveTimer = window.setTimeout(() => {
            if (pendingSave) {
                saveDraft("auto");
            }
        }, AUTO_SAVE_DELAY);
    };

    const saveDraft = async (trigger = "auto") => {
        if (isSaving || isPublishing) return;
        const snapshot = getSnapshot();
        const parsed = JSON.parse(snapshot);
        const hasPayload = parsed.title || parsed.content;
        if (trigger !== "manual" && !pendingSave && snapshot === lastSavedSnapshot) return;
        if (!hasPayload && trigger !== "manual") return;

        isSaving = true;
        setStatus(statusMessages.saving);
        syncContent();
        const payload = buildPayload("save_draft");
        try {
            const response = await fetch(resolveFormAction(), {
                method: "POST",
                credentials: "same-origin",
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                    "X-CSRFToken": getCsrfToken(),
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });
            const responseData = await response.json().catch(() => ({}));
            if (!response.ok || !responseData || responseData.status !== "success") {
                const message = responseData?.message || `保存失败（${response.status}）`;
                throw new Error(message);
            }
            if (postIdInput && responseData.post_id) {
                postIdInput.value = responseData.post_id;
            }
            if (responseData.post_id) {
                updateDraftUrl(responseData.post_id, true);
            }
            pendingSave = false;
            lastSavedSnapshot = snapshot;
            setStatus(statusMessages.saved);
        } catch (error) {
            setStatus(statusMessages.failed);
            alert(error.message || statusMessages.failed);
        } finally {
            isSaving = false;
        }
    };

    const publishPost = async () => {
        if (isPublishing || isSaving) return;
        if (titleInput) {
            titleInput.value = titleInput.value.trim();
            titleInput.required = true;
            if (!titleInput.value) {
                titleInput.reportValidity();
                return;
            }
        }
        isPublishing = true;
        syncContent();
        const payload = buildPayload("publish");
        try {
            const response = await fetch(resolveFormAction(), {
                method: "POST",
                credentials: "same-origin",
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                    "X-CSRFToken": getCsrfToken(),
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });
            const responseData = await response.json().catch(() => ({}));
            if (!response.ok || !responseData || responseData.status !== "success") {
                const message = responseData?.message || `发布失败（${response.status}）`;
                throw new Error(message);
            }
            const redirectUrl = responseData.redirect_url || form.dataset.redirectUrl || "";
            if (redirectUrl) {
                window.location.href = redirectUrl;
            } else {
                window.location.href = resolveFormAction();
            }
        } catch (error) {
            setStatus(statusMessages.failed);
            alert(error.message || "发布失败，请稍后重试。");
            isPublishing = false;
        }
    };

    const insertImageAtCursor = (url) => {
        if (!quill || !url) return;
        const range = quill.getSelection(true) || { index: quill.getLength(), length: 0 };
        quill.insertEmbed(range.index, "image", url, "user");
        quill.setSelection(range.index + 1, 0, "silent");
    };

    const uploadImage = async (file) => {
        if (!uploadUrl) {
            console.warn("Missing upload URL for community editor images.");
            return null;
        }
        const formData = new FormData();
        formData.append("image", file);
        if (csrfToken) {
            formData.append("csrfmiddlewaretoken", csrfToken);
        }
        const response = await fetch(uploadUrl, {
            method: "POST",
            headers: {
                "X-Requested-With": "XMLHttpRequest",
            },
            body: formData,
        });
        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload) {
            throw new Error("upload_failed");
        }
        return payload.url || payload.image_url || "";
    };

    const renderDrafts = (drafts) => {
        if (!draftsList) return;
        draftsList.innerHTML = "";
        if (!drafts || drafts.length === 0) {
            draftsEmpty?.classList.remove("d-none");
            return;
        }
        draftsEmpty?.classList.add("d-none");
        const currentDraftId = postIdInput?.value ? String(postIdInput.value) : "";
        drafts.forEach((draft) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "draft-item";
            button.dataset.draftId = draft.id;
            if (currentDraftId && String(draft.id) === currentDraftId) {
                button.classList.add("is-active");
                button.setAttribute("aria-current", "true");
            }

            const meta = document.createElement("div");
            meta.className = "draft-item-meta";

            const title = document.createElement("div");
            title.className = "draft-item-title";
            title.textContent = draft.title?.trim() || "无标题草稿";

            const time = document.createElement("div");
            time.className = "draft-item-time";
            time.textContent = draft.updated_at || "";

            meta.appendChild(title);
            meta.appendChild(time);
            button.appendChild(meta);

            const deleteBtn = document.createElement("button");
            deleteBtn.type = "button";
            deleteBtn.className = "draft-item-delete";
            deleteBtn.setAttribute("aria-label", "删除草稿");
            deleteBtn.innerHTML =
                '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 6h2v8h-2V9zm4 0h2v8h-2V9zM7 9h2v8H7V9z" fill="currentColor"/></svg>';
            deleteBtn.addEventListener("click", async (event) => {
                event.stopPropagation();
                if (!confirm("确定要永久删除这个草稿吗？")) return;
                try {
                    const response = await fetch(resolveDeleteUrl(draft.id), {
                        method: "POST",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRFToken": getCsrfToken(),
                            "Content-Type": "application/json",
                        },
                        credentials: "same-origin",
                        body: JSON.stringify({}),
                    });
                    const payload = await response.json().catch(() => ({}));
                    if (!response.ok || payload?.status !== "success") {
                        throw new Error(payload?.message || `删除失败（${response.status}）`);
                    }
                    button.remove();
                    if (!draftsList?.querySelector(".draft-item")) {
                        draftsEmpty?.classList.remove("d-none");
                    }
                    if (postIdInput?.value && String(postIdInput.value) === String(draft.id)) {
                        resetEditorForNewDoc();
                    }
                } catch (error) {
                    alert(error.message || "删除失败，请稍后重试。");
                }
            });
            button.appendChild(deleteBtn);

            button.addEventListener("click", () => {
                window.location.href = `?id=${encodeURIComponent(draft.id)}`;
            });
            draftsList.appendChild(button);
        });
    };

    const loadDrafts = async () => {
        if (!draftsEndpoint) return;
        draftsList && (draftsList.innerHTML = "");
        draftsEmpty?.classList.add("d-none");
        try {
            const response = await fetch(draftsEndpoint, {
                headers: { "X-Requested-With": "XMLHttpRequest" },
                credentials: "same-origin",
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload?.message || `加载草稿失败（${response.status}）`);
            }
            renderDrafts(payload.drafts || []);
        } catch (error) {
            if (draftsEmpty) {
                draftsEmpty.textContent = error.message || "加载草稿失败。";
                draftsEmpty.classList.remove("d-none");
            }
        }
    };

    const resetEditorForNewDoc = () => {
        if (titleInput) titleInput.value = "";
        if (contentInput) contentInput.value = "";
        if (postIdInput) postIdInput.value = "";
        draftsList?.querySelectorAll(".draft-item.is-active").forEach((item) => {
            item.classList.remove("is-active");
            item.removeAttribute("aria-current");
        });
        if (quill) {
            quill.setContents([]);
            quill.focus();
        }
        pendingSave = false;
        lastSavedSnapshot = getSnapshot();
        setStatus(statusMessages.saved);
        updateWordCount();
        updateDraftUrl("", false);
    };

    const handleImageFile = async (file) => {
        try {
            setStatus(statusMessages.saving);
            const url = await uploadImage(file);
            if (url) {
                insertImageAtCursor(url);
            } else {
                throw new Error("missing_url");
            }
            setStatus(statusMessages.saved);
            markDirty();
        } catch (error) {
            setStatus(statusMessages.failed);
        }
    };

    const pickImageFile = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.addEventListener("change", () => {
            const file = input.files && input.files[0];
            if (file) {
                handleImageFile(file);
            }
        });
        input.click();
    };

    const extractImageFile = (dataTransfer) => {
        if (!dataTransfer?.files?.length) return null;
        for (const file of dataTransfer.files) {
            if (file && file.type && file.type.startsWith("image/")) {
                return file;
            }
        }
        return null;
    };

    const extractClipboardImage = (clipboardData) => {
        if (!clipboardData) return null;
        if (clipboardData.files && clipboardData.files.length) {
            return extractImageFile({ files: clipboardData.files });
        }
        const items = clipboardData.items || [];
        for (const item of items) {
            if (item.type && item.type.startsWith("image/")) {
                return item.getAsFile();
            }
        }
        return null;
    };

    const initEditor = () => {
        if (!window.Quill || !editorContainer) {
            updateWordCount();
            return;
        }
        if (window.hljs && typeof window.hljs.configure === "function") {
            window.hljs.configure({ languages: ["python"] });
        }
        const BlockEmbed = window.Quill.import("blots/embed");

        class BacktestCard extends BlockEmbed {
            static create(value) {
                const node = super.create();
                const payload = value || {};
                const recordId = String(payload.id || "").trim();
                const name = payload.name || "Backtest Strategy";
                const totalReturn = payload.return || "--";
                const sharpe = payload.sharpe || "N/A";
                const maxDrawdown = payload.max_drawdown || payload.maxDrawdown || "N/A";
                const winRate = payload.win_rate || payload.winRate || "N/A";
                const date = payload.date || payload.created_at || "";
                const equityCurve = Array.isArray(payload.equity_curve) ? payload.equity_curve : null;

                node.setAttribute("contenteditable", "false");
                node.dataset.id = recordId;
                node.dataset.name = name;
                node.dataset.return = totalReturn;
                node.dataset.sharpe = sharpe;
                node.dataset.maxDrawdown = maxDrawdown;
                node.dataset.winRate = winRate;
                node.dataset.date = date;
                node.classList.add("backtest-card-embed");

                const header = document.createElement("div");
                header.className = "backtest-card-header";

                const titleGroup = document.createElement("div");
                titleGroup.className = "backtest-card-title-group";

                const icon = document.createElement("span");
                icon.className = "backtest-card-icon";
                icon.setAttribute("aria-hidden", "true");
                icon.innerHTML =
                    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19h16v2H2V3h2v16zm4-2H6V9h2v8zm5 0h-2V5h2v12zm5 0h-2v-6h2v6z" fill="currentColor"/></svg>';

                const title = document.createElement("div");
                title.className = "backtest-card-title";
                title.textContent = name;

                titleGroup.appendChild(icon);
                titleGroup.appendChild(title);

                const meta = document.createElement("div");
                meta.className = "backtest-card-meta";

                const dateEl = document.createElement("div");
                dateEl.className = "backtest-card-date";
                dateEl.textContent = date || "—";

                const action = document.createElement("div");
                action.className = "backtest-card-action";
                action.textContent = "VIEW REPORT";

                meta.appendChild(dateEl);
                meta.appendChild(action);

                header.appendChild(titleGroup);
                header.appendChild(meta);

                const metrics = document.createElement("div");
                metrics.className = "backtest-card-metrics";

                const createMetric = (labelText, valueText, valueClass) => {
                    const metric = document.createElement("div");
                    metric.className = "backtest-card-metric";

                    const label = document.createElement("div");
                    label.className = "backtest-card-metric-label";
                    label.textContent = labelText;

                    const value = document.createElement("div");
                    value.className = "backtest-card-metric-value";
                    if (valueClass) {
                        value.classList.add(valueClass);
                    }
                    value.textContent = valueText;

                    metric.appendChild(label);
                    metric.appendChild(value);
                    return metric;
                };

                const returnValue = Number.parseFloat(String(totalReturn).replace("%", ""));
                const returnClass = Number.isFinite(returnValue)
                    ? returnValue >= 0
                        ? "is-positive"
                        : "is-negative"
                    : "is-neutral";

                metrics.appendChild(createMetric("Return", totalReturn, returnClass));
                metrics.appendChild(createMetric("Sharpe", sharpe));
                metrics.appendChild(createMetric("Max DD", maxDrawdown, "is-negative"));
                metrics.appendChild(createMetric("Win Rate", winRate));

                node.appendChild(header);
                const sparklineSvg = generateSparklineSVG(equityCurve);
                if (sparklineSvg) {
                    const sparklineWrap = document.createElement("div");
                    sparklineWrap.className = "backtest-card-sparkline";
                    sparklineWrap.innerHTML = sparklineSvg;
                    node.appendChild(sparklineWrap);
                }
                node.appendChild(metrics);
                return node;
            }

            static value(node) {
                return {
                    id: node.dataset.id || "",
                    name: node.dataset.name || "",
                    return: node.dataset.return || "",
                    sharpe: node.dataset.sharpe || "",
                    max_drawdown: node.dataset.maxDrawdown || "",
                    win_rate: node.dataset.winRate || "",
                    date: node.dataset.date || "",
                };
            }
        }

        BacktestCard.blotName = "backtestCard";
        BacktestCard.tagName = "div";
        BacktestCard.className = "backtest-smart-card";

        window.Quill.register(BacktestCard, true);

        const fontWhitelist = ["SimSun", "SimHei", "Microsoft-YaHei", "Arial", "Times-New-Roman"];
        const sizeWhitelist = ["12px", "14px", "16px", "18px", "20px", "24px", "30px", "36px"];
        const Font = window.Quill.import("formats/font");
        Font.whitelist = fontWhitelist;
        window.Quill.register(Font, true);
        const SizeStyle = window.Quill.import("attributors/style/size");
        SizeStyle.whitelist = sizeWhitelist;
        window.Quill.register(SizeStyle, true);

        const icons = window.Quill.import("ui/icons");
        icons.undo =
            '<svg viewBox="0 0 18 18"><path d="M7.5 4.5H3.75V2L0 5.75 3.75 9.5V7h3.75a3.75 3.75 0 1 1 0 7.5H6v1.5h1.5a5.25 5.25 0 0 0 0-10.5z"/></svg>';
        icons.redo =
            '<svg viewBox="0 0 18 18"><path d="M10.5 4.5h3.75V2L18 5.75 14.25 9.5V7H10.5a3.75 3.75 0 1 0 0 7.5H12v1.5h-1.5a5.25 5.25 0 0 1 0-10.5z"/></svg>';
        icons.backtest =
            '<svg viewBox="0 0 24 24"><path d="M4 19h16v2H2V3h2v16zm4-2H6V9h2v8zm5 0h-2V5h2v12zm5 0h-2v-6h2v6z" fill="currentColor"/></svg>';

        const resolveImageResizeModule = () => {
            if (!window.Quill?.import) return null;
            try {
                const existing = window.Quill.import("modules/imageResize");
                if (typeof existing === "function") {
                    return existing;
                }
            } catch (error) {
                // Ignore missing module and fall back to global export.
            }
            const candidate = window.ImageResize?.default || window.ImageResize;
            if (typeof candidate === "function") {
                window.Quill.register("modules/imageResize", candidate);
                return candidate;
            }
            return null;
        };

        const imageResizeModule = resolveImageResizeModule();
        const placeholder =
            "开始写作...（选中文字可弹出快捷菜单，连按回车可跳出代码块）";
        const bounds = editorContainer.closest(".write-container") || editorContainer;
        const toolbarOptions = [
            ["undo", "redo"],
            [{ font: fontWhitelist }, { size: sizeWhitelist }],
            ["bold", "italic", "underline", "strike", { color: [] }, { background: [] }],
            [{ header: 1 }, { header: 2 }],
            [{ list: "ordered" }, { list: "bullet" }, { indent: "-1" }, { indent: "+1" }],
            [{ align: [] }],
            ["link", "image", "video", "formula", "backtest"],
            ["clean"],
        ];
        const modules = {
            syntax: true,
            formula: true,
            history: {
                delay: 1000,
                maxStack: 200,
                userOnly: true,
            },
            toolbar: {
                container: toolbarOptions,
                handlers: {
                    undo() {
                        this.quill.history.undo();
                    },
                    redo() {
                        this.quill.history.redo();
                    },
                    image: () => pickImageFile(),
                    formula: () => showMathModal(),
                    backtest: () => {
                        if (!quill) return;
                        openBacktestModal();
                    },
                },
            },
        };
        if (imageResizeModule) {
            modules.imageResize = {
                modules: ["Resize", "DisplaySize", "Toolbar"],
            };
        }
        quill = new window.Quill(editorContainer, {
            theme: "snow",
            placeholder,
            bounds,
            modules,
        });

        const clearBlockFormats = (range) => {
            quill.formatLine(range.index, 1, "code-block", false);
            quill.formatLine(range.index, 1, "blockquote", false);
        };

        quill.keyboard.addBinding(
            { key: 13, collapsed: true, format: ["code-block"] },
            (range, context) => {
                if (!context.empty) return true;
                clearBlockFormats(range);
                return false;
            },
        );

        quill.keyboard.addBinding(
            { key: 13, collapsed: true, format: ["blockquote"] },
            (range, context) => {
                if (!context.empty) return true;
                clearBlockFormats(range);
                return false;
            },
        );

        const toolbarModule = quill.getModule("toolbar");
        if (toolbarModule && typeof toolbarModule.update === "function") {
            const updateToolbar = (range) => {
                try {
                    toolbarModule.update(range);
                } catch (error) {
                    toolbarModule.update();
                }
            };
            quill.on("selection-change", (range) => updateToolbar(range));
            quill.on("text-change", () => updateToolbar(quill.getSelection()));
        }

        if (toolbarModule?.container) {
            const tooltipMap = {
                "ql-bold": "加粗 (Ctrl+B)",
                "ql-italic": "斜体 (Ctrl+I)",
                "ql-header": "标题",
                "ql-list": "列表",
                "ql-image": "插入图片",
                "ql-formula": "插入公式",
                "ql-backtest": "插入回测卡片",
                "ql-code-block": "代码块",
            };
            const applyTooltip = (selector, title) => {
                toolbarModule.container.querySelectorAll(selector).forEach((el) => {
                    if (!el.getAttribute("title")) {
                        el.setAttribute("title", title);
                    }
                    if (window.bootstrap?.Tooltip) {
                        new window.bootstrap.Tooltip(el, { container: "body" });
                    }
                });
            };

            applyTooltip(".ql-bold", tooltipMap["ql-bold"]);
            applyTooltip(".ql-italic", tooltipMap["ql-italic"]);
            applyTooltip(".ql-header", tooltipMap["ql-header"]);
            applyTooltip(".ql-header .ql-picker-label", tooltipMap["ql-header"]);
            applyTooltip(".ql-list", tooltipMap["ql-list"]);
            applyTooltip(".ql-image", tooltipMap["ql-image"]);
            applyTooltip(".ql-formula", tooltipMap["ql-formula"]);
            applyTooltip(".ql-backtest", tooltipMap["ql-backtest"]);
            applyTooltip(".ql-code-block", tooltipMap["ql-code-block"]);
        }

        if (contentInput && contentInput.value) {
            const initialValue = contentInput.value.trim();
            if (initialValue) {
                if (initialValue.includes("<")) {
                    quill.clipboard.dangerouslyPasteHTML(initialValue);
                } else {
                    quill.setText(initialValue);
                }
            }
        }

        lastSavedSnapshot = getSnapshot();
        updateWordCount();

        quill.on("text-change", () => {
            markDirty();
            updateWordCount();
        });

        quill.root.addEventListener("drop", (event) => {
            const file = extractImageFile(event.dataTransfer);
            if (!file) return;
            event.preventDefault();
            handleImageFile(file);
        });

        quill.root.addEventListener("paste", (event) => {
            const file = extractClipboardImage(event.clipboardData);
            if (!file) return;
            event.preventDefault();
            handleImageFile(file);
        });
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initEditor);
    } else {
        initEditor();
    }

    if (mathModal) {
        mathModal.addEventListener("shown.bs.modal", () => {
            if (mathField && typeof mathField.focus === "function") {
                mathField.focus();
                if (typeof mathField.executeCommand === "function") {
                    mathField.executeCommand("showVirtualKeyboard");
                } else if (window.mathVirtualKeyboard?.show) {
                    window.mathVirtualKeyboard.show();
                }
            }
        });
        mathModal.addEventListener("hidden.bs.modal", () => {
            hideMathKeyboard();
            if (quill) {
                quill.focus();
            }
        });
    }

    if (backtestModal) {
        backtestModal.addEventListener("shown.bs.modal", () => {
            loadBacktests();
        });
    }

    if (mathInsertBtn) {
        mathInsertBtn.addEventListener("click", () => {
            hideMathKeyboard();
            if (!quill || !mathField) return;
            const latex = (mathField.value || "").trim();
            if (!latex) return;
            const range = lastFormulaRange || quill.getSelection(true);
            const index = range ? range.index : quill.getLength();
            quill.insertEmbed(index, "formula", latex, "user");
            quill.setSelection(index + 1, 0, "silent");
            mathField.value = "";
            hideMathModal();
        });
    }

    if (mathCancelBtn) {
        mathCancelBtn.addEventListener("click", () => {
            hideMathKeyboard();
        });
    }

    if (mathCloseBtn) {
        mathCloseBtn.addEventListener("click", () => {
            hideMathKeyboard();
        });
    }

    if (titleInput) {
        titleInput.addEventListener("input", markDirty);
    }

    if (coverPickers.length) {
        coverPickers.forEach((button) => {
            button.addEventListener("click", openCoverPicker);
        });
    }

    if (coverInput) {
        coverInput.addEventListener("change", () => {
            const file = coverInput.files && coverInput.files[0];
            if (!file) {
                clearCoverPreview();
                return;
            }
            if (!file.type || !file.type.startsWith("image/")) {
                alert("请选择图片文件。");
                coverInput.value = "";
                return;
            }
            showCoverPreview(file);
        });
    }

    if (coverRemoveBtn) {
        coverRemoveBtn.addEventListener("click", () => {
            clearCoverPreview();
        });
    }

    form.querySelectorAll("input[name='topic'], select[name='topic']").forEach((el) => {
        el.addEventListener("change", markDirty);
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        publishPost();
    });

    if (saveDraftBtn) {
        saveDraftBtn.addEventListener("click", () => {
            saveDraft("manual");
        });
    }

    if (draftsDrawer) {
        draftsDrawer.addEventListener("shown.bs.offcanvas", () => {
            loadDrafts();
        });
    }

    if (draftNewBtn) {
        draftNewBtn.addEventListener("click", () => {
            resetEditorForNewDoc();
            if (window.bootstrap?.Offcanvas) {
                const instance = window.bootstrap.Offcanvas.getInstance(draftsDrawer);
                instance?.hide();
            }
        });
    }

    intervalId = window.setInterval(() => {
        if (pendingSave) {
            saveDraft("interval");
        }
    }, AUTO_SAVE_INTERVAL);

    window.addEventListener("beforeunload", () => {
        if (autoSaveTimer) {
            window.clearTimeout(autoSaveTimer);
        }
        if (intervalId) {
            window.clearInterval(intervalId);
        }
        if (coverObjectUrl) {
            URL.revokeObjectURL(coverObjectUrl);
            coverObjectUrl = "";
        }
    });
})();
