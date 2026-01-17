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

    const statusMessages = {
        saving: statusEl?.dataset.savingText || "Saving draft...",
        saved: statusEl?.dataset.savedText || "Draft saved.",
        failed: statusEl?.dataset.failedText || "Save failed. Please try again.",
    };

    const mathModal = document.getElementById("mathModal");
    const mathField = mathModal?.querySelector("#math-input");
    const mathInsertBtn = mathModal?.querySelector("[data-role='math-insert']");
    let mathModalInstance = null;
    let lastFormulaRange = null;

    const AUTO_SAVE_DELAY = 1200;
    const AUTO_SAVE_INTERVAL = 30000;

    let quill = null;
    let pendingSave = false;
    let isSaving = false;
    let isPublishing = false;
    let lastSavedSnapshot = "";
    let autoSaveTimer = null;
    let intervalId = null;

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

    const hideMathModal = () => {
        if (mathModalInstance) {
            mathModalInstance.hide();
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
        if (!window.Quill || !editorContainer) return;
        const placeholder =
            "开始写作...（选中文字可弹出快捷菜单，连按回车可跳出代码块）";
        const bounds = editorContainer.closest(".write-container") || editorContainer;
        quill = new window.Quill(editorContainer, {
            theme: "snow",
            placeholder,
            bounds,
            modules: {
                formula: true,
                toolbar: {
                    container: [
                        [{ header: [1, 2, 3, false] }],
                        [{ font: [] }],
                        ["bold", "italic", "underline", "strike"],
                        [{ color: [] }, { background: [] }],
                        [{ script: "sub" }, { script: "super" }],
                        [{ list: "ordered" }, { list: "bullet" }],
                        [{ indent: "-1" }, { indent: "+1" }],
                        [{ align: [] }],
                        ["link", "image", "code-block", "formula"],
                        ["clean"],
                    ],
                    handlers: {
                        image: () => pickImageFile(),
                        formula: () => showMathModal(),
                    },
                },
            },
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
            applyTooltip(".ql-header .ql-picker-label", tooltipMap["ql-header"]);
            applyTooltip(".ql-list", tooltipMap["ql-list"]);
            applyTooltip(".ql-image", tooltipMap["ql-image"]);
            applyTooltip(".ql-formula", tooltipMap["ql-formula"]);
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

        quill.on("text-change", () => {
            markDirty();
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
            if (window.mathVirtualKeyboard?.hide) {
                window.mathVirtualKeyboard.hide();
            }
            if (quill) {
                quill.focus();
            }
        });
    }

    if (mathInsertBtn) {
        mathInsertBtn.addEventListener("click", () => {
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

    if (titleInput) {
        titleInput.addEventListener("input", markDirty);
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
    });
})();
