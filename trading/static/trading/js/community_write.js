(() => {
    const form = document.querySelector("[data-role='write-form']");
    if (!form) return;

    const editorContainer = form.querySelector("[data-role='editor']");
    const contentInput = form.querySelector("[data-role='content-input']");
    const actionInput = form.querySelector("[data-role='action-input']");
    const saveDraftBtn = form.querySelector("[data-role='save-draft']");
    const statusEl = form.querySelector("[data-role='draft-status']");
    const postIdInput = form.querySelector("[data-role='post-id']");
    const titleInput = form.querySelector("input[name='title']");
    const editBase = form.dataset.editBase || "";
    const uploadUrl = form.dataset.uploadUrl || editorContainer?.dataset.uploadUrl || "";
    const csrfToken = form.querySelector("input[name='csrfmiddlewaretoken']")?.value || "";

    const statusMessages = {
        saving: statusEl?.dataset.savingText || "Saving draft...",
        saved: statusEl?.dataset.savedText || "Draft saved.",
        failed: statusEl?.dataset.failedText || "Save failed. Please try again.",
    };

    const AUTO_SAVE_DELAY = 1200;
    const AUTO_SAVE_INTERVAL = 30000;

    let quill = null;
    let pendingSave = false;
    let isSaving = false;
    let isPublishing = false;
    let lastSavedSnapshot = "";
    let autoSaveTimer = null;
    let intervalId = null;

    const normalizeHtml = (html) => {
        const trimmed = (html || "").trim();
        return trimmed === "<p><br></p>" ? "" : trimmed;
    };

    const syncContent = () => {
        if (!quill || !contentInput) return;
        contentInput.value = normalizeHtml(quill.root.innerHTML);
    };

    const setStatus = (message) => {
        if (!statusEl) return;
        statusEl.textContent = message || "";
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
        const formData = new FormData(form);
        formData.set("action", "save_draft");
        try {
            const response = await fetch(form.action || window.location.pathname, {
                method: "POST",
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                },
                body: formData,
            });
            const payload = await response.json();
            if (!response.ok || !payload || payload.status !== "success") {
                throw new Error("save_failed");
            }
            if (postIdInput && payload.post_id) {
                postIdInput.value = payload.post_id;
            }
            if (editBase && payload.post_id && window.location.pathname === editBase) {
                const base = editBase.endsWith("/") ? editBase : `${editBase}/`;
                window.history.replaceState(null, "", `${base}${payload.post_id}/`);
            }
            pendingSave = false;
            lastSavedSnapshot = snapshot;
            setStatus(statusMessages.saved);
        } catch (error) {
            setStatus(statusMessages.failed);
        } finally {
            isSaving = false;
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
        const placeholder = editorContainer.dataset.placeholder || "Write something thoughtful...";
        const bounds = editorContainer.closest(".write-canvas") || editorContainer;
        quill = new window.Quill(editorContainer, {
            theme: "bubble",
            placeholder,
            bounds,
            modules: {
                toolbar: {
                    container: [
                        ["bold", "italic", "underline"],
                        [{ header: 1 }, { header: 2 }],
                        [{ list: "ordered" }, { list: "bullet" }],
                        ["blockquote", "code-block", "link", "image"],
                        ["clean"],
                    ],
                    handlers: {
                        image: () => pickImageFile(),
                    },
                },
            },
        });

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

    if (titleInput) {
        titleInput.addEventListener("input", markDirty);
    }

    form.querySelectorAll("input[name='topic'], select[name='topic']").forEach((el) => {
        el.addEventListener("change", markDirty);
    });

    form.addEventListener("submit", (event) => {
        if (isSaving) {
            event.preventDefault();
            return;
        }
        if (actionInput) {
            actionInput.value = "publish";
        }
        if (titleInput) {
            titleInput.value = titleInput.value.trim();
            titleInput.required = true;
            if (!titleInput.value) {
                event.preventDefault();
                titleInput.reportValidity();
                return;
            }
        }
        isPublishing = true;
        syncContent();
    });

    if (saveDraftBtn) {
        saveDraftBtn.addEventListener("click", () => {
            saveDraft("manual");
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
