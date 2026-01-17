(() => {
    const root = document.querySelector(".community-page[data-like-endpoint]");
    if (!root) {
        return;
    }

    const csrfToken = () => {
        const input = root.querySelector("input[name='csrfmiddlewaretoken']");
        return input ? input.value : "";
    };

    const composePanel = root.querySelector("[data-role='compose-panel']");
    const openComposeBtn = root.querySelector("[data-role='open-compose']");
    const closeComposeBtn = composePanel?.querySelector("[data-role='close-compose']");
    const toggleNewTopicBtn = composePanel?.querySelector("[data-role='toggle-new-topic']");
    const newTopicFields = composePanel?.querySelector("[data-role='new-topic-fields']");
    const selectedTopicBadge = composePanel?.querySelector("[data-role='selected-topic']");
    const topicInput = composePanel?.querySelector("[data-role='topic-input']");
    const createTopicBtn = root.querySelector("[data-role='create-topic']");
    const composeTriggers = [];
    if (openComposeBtn) composeTriggers.push(openComposeBtn);
    if (createTopicBtn) composeTriggers.push(createTopicBtn);
    const composerForm = composePanel?.querySelector("[data-role='composer']");
    const contentTextarea = composerForm?.querySelector("[data-role='content-input']");
    const croppedInput = composerForm?.querySelector("[data-role='cropped-data']");
    const imageInput = composerForm?.querySelector("[data-role='image-input']");
    const mediaPanel = composerForm?.querySelector("[data-role='media-panel']");
    const mediaPlaceholder = composerForm?.querySelector("[data-role='media-placeholder']");
    const mediaPreview = composerForm?.querySelector("[data-role='media-preview']");
    const previewImage = composerForm?.querySelector("[data-role='preview-image']");
    const pickImageBtn = composerForm?.querySelector("[data-role='pick-image']");
    const replaceImageBtn = composerForm?.querySelector("[data-role='replace-image']");
    const removeImageBtn = composerForm?.querySelector("[data-role='remove-image']");
    const mathModal = document.getElementById("mathModal");
    const mathField = mathModal?.querySelector("#math-input");
    const mathInsertBtn = mathModal?.querySelector("[data-role='math-insert']");
    let mathModalInstance = null;
    const langAttr = (document.documentElement.getAttribute("lang") || "zh").toLowerCase();
    const langIsZh = langAttr.indexOf("zh") === 0;

    const cropperModal = document.querySelector("[data-role='cropper-modal']");
    const cropperImage = cropperModal?.querySelector("[data-role='crop-image']");
    const cropperStage = cropperModal?.querySelector("[data-role='crop-stage']");
    const cropZoom = cropperModal?.querySelector("[data-role='crop-zoom']");
    const cropCancel = cropperModal?.querySelector("[data-role='crop-cancel']");
    const cropSkip = cropperModal?.querySelector("[data-role='crop-skip']");
    const cropConfirm = cropperModal?.querySelector("[data-role='crop-confirm']");

    const likeEndpoint = root.dataset.likeEndpoint;
    const commentEndpoint = root.dataset.commentEndpoint;
    const postAlert = root.querySelector("[data-role='post-alert']");
    const activeTopicId = root.dataset.activeTopicId;
    const activeTopicName = root.dataset.activeTopicName;
    const shareHistoryId = root.dataset.shareHistoryId;

    const highlightBlocks = (scope) => {
        if (!window.hljs || typeof window.hljs.highlightElement !== "function") return;
        const target = scope || document;
        target.querySelectorAll("pre code").forEach((block) => {
            window.hljs.highlightElement(block);
        });
    };

    const editorContainer = composerForm?.querySelector("[data-role='editor']");
    let quill = null;
    let objectUrl = null;
    let lastComposeTrigger = null;
    let lastDialogFocus = null;

    const cropState = {
        dataUrl: "",
        imgWidth: 0,
        imgHeight: 0,
        baseScale: 1,
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
        stageWidth: 0,
        stageHeight: 0,
        pointer: null,
        originalFile: null,
    };

    const clampOffsets = () => {
        if (!cropperStage) return;
        const scale = cropState.baseScale * cropState.zoom;
        const displayWidth = cropState.imgWidth * scale;
        const displayHeight = cropState.imgHeight * scale;
        const maxOffsetX = Math.max(0, (displayWidth - cropState.stageWidth) / 2);
        const maxOffsetY = Math.max(0, (displayHeight - cropState.stageHeight) / 2);
        cropState.offsetX = Math.min(maxOffsetX, Math.max(-maxOffsetX, cropState.offsetX));
        cropState.offsetY = Math.min(maxOffsetY, Math.max(-maxOffsetY, cropState.offsetY));
    };

    const updateCropperTransform = () => {
        if (!cropperImage) return;
        clampOffsets();
        const scale = cropState.baseScale * cropState.zoom;
        cropperImage.style.transform = `translate(-50%, -50%) translate(${cropState.offsetX}px, ${cropState.offsetY}px) scale(${scale})`;
    };

    const setComposeExpanded = (expanded) => {
        composeTriggers.forEach((btn) => {
            if (btn) {
                btn.setAttribute("aria-expanded", expanded ? "true" : "false");
            }
        });
        if (composePanel) {
            composePanel.setAttribute("aria-hidden", expanded ? "false" : "true");
        }
    };

    const openCompose = (opts = {}) => {
        if (!composePanel) return;
        composePanel.classList.remove("d-none");
        setComposeExpanded(true);
        if (quill) {
            setTimeout(() => quill.focus(), 50);
        } else if (contentTextarea) {
            setTimeout(() => contentTextarea.focus(), 50);
        }
        if (topicInput && !opts.newTopic) {
            topicInput.value = activeTopicId || "";
        }
        if (selectedTopicBadge) {
            selectedTopicBadge.textContent = `#${opts.newTopic ? "新话题" : activeTopicName || "话题"}`;
        }
        if (newTopicFields) {
            const willShow = Boolean(opts.newTopic);
            newTopicFields.classList.toggle("d-none", !willShow);
            newTopicFields.setAttribute("aria-hidden", willShow ? "false" : "true");
            if (!willShow) {
                newTopicFields.querySelectorAll("input,textarea").forEach((el) => {
                    el.value = "";
                });
            }
        }
        if (toggleNewTopicBtn) {
            toggleNewTopicBtn.setAttribute("aria-expanded", opts.newTopic ? "true" : "false");
        }
        if (topicInput && opts.newTopic) {
            topicInput.value = "";
        }
    };

    const closeCompose = () => {
        if (!composePanel) return;
        composePanel.classList.add("d-none");
        setComposeExpanded(false);
        if (newTopicFields) {
            newTopicFields.classList.add("d-none");
            newTopicFields.setAttribute("aria-hidden", "true");
            newTopicFields.querySelectorAll("input,textarea").forEach((el) => (el.value = ""));
        }
        if (toggleNewTopicBtn) {
            toggleNewTopicBtn.setAttribute("aria-expanded", "false");
        }
        if (topicInput) {
            topicInput.value = activeTopicId || "";
        }
        if (selectedTopicBadge) {
            selectedTopicBadge.textContent = `#${activeTopicName || "话题"}`;
        }
        if (lastComposeTrigger) {
            lastComposeTrigger.focus();
            lastComposeTrigger = null;
        }
    };

    const resetMedia = () => {
        if (imageInput) imageInput.value = "";
        if (croppedInput) croppedInput.value = "";
        if (previewImage) previewImage.src = "";
        mediaPreview?.classList.add("d-none");
        mediaPlaceholder?.classList.remove("d-none");
        if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
            objectUrl = null;
        }
    };

    const openCropper = (dataUrl, file) => {
        if (!cropperModal || !cropperImage || !cropperStage || !cropZoom) return;
        cropState.dataUrl = dataUrl;
        cropState.originalFile = file;
        lastDialogFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        cropperModal.classList.remove("d-none");
        cropperModal.setAttribute("aria-hidden", "false");
        cropperModal.focus({ preventScroll: true });
        cropZoom.value = "1";
        cropState.zoom = 1;
        cropState.offsetX = 0;
        cropState.offsetY = 0;
        cropperImage.onload = () => {
            window.requestAnimationFrame(() => {
                cropState.imgWidth = cropperImage.naturalWidth;
                cropState.imgHeight = cropperImage.naturalHeight;
                const rect = cropperStage.getBoundingClientRect();
                cropState.stageWidth = rect.width || cropperStage.offsetWidth || 640;
                cropState.stageHeight = rect.height || cropperStage.offsetHeight || 360;
                cropState.baseScale = Math.max(
                    cropState.stageWidth / cropState.imgWidth,
                    cropState.stageHeight / cropState.imgHeight
                );
                updateCropperTransform();
            });
        };
        cropperImage.src = dataUrl;
    };

    const closeCropper = () => {
        if (!cropperModal) return;
        cropperModal.classList.add("d-none");
        cropperModal.setAttribute("aria-hidden", "true");
        cropState.pointer = null;
        if (lastDialogFocus) {
            lastDialogFocus.focus();
            lastDialogFocus = null;
        }
    };

    const applyCroppedImage = () => {
        if (!cropperImage || !previewImage || !mediaPreview || !mediaPlaceholder || !croppedInput) return;
        const canvas = document.createElement("canvas");
        const targetWidth = 1280;
        const targetHeight = Math.round(targetWidth * (cropState.stageHeight / cropState.stageWidth));
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const scale = cropState.baseScale * cropState.zoom;
        const displayWidth = cropState.imgWidth * scale;
        const displayHeight = cropState.imgHeight * scale;
        const topLeftX = (cropState.stageWidth - displayWidth) / 2 + cropState.offsetX;
        const topLeftY = (cropState.stageHeight - displayHeight) / 2 + cropState.offsetY;
        const sx = Math.max(0, (-topLeftX / displayWidth) * cropState.imgWidth);
        const sy = Math.max(0, (-topLeftY / displayHeight) * cropState.imgHeight);
        const sWidth = Math.min(
            cropState.imgWidth,
            (cropState.stageWidth / displayWidth) * cropState.imgWidth
        );
        const sHeight = Math.min(
            cropState.imgHeight,
            (cropState.stageHeight / displayHeight) * cropState.imgHeight
        );
        ctx.drawImage(
            cropperImage,
            sx,
            sy,
            sWidth,
            sHeight,
            0,
            0,
            targetWidth,
            targetHeight
        );
        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        croppedInput.value = dataUrl;
        if (imageInput) {
            imageInput.value = "";
        }
        previewImage.src = dataUrl;
        mediaPreview.classList.remove("d-none");
        mediaPlaceholder.classList.add("d-none");
    };

    const useOriginalImage = () => {
        if (!cropState.originalFile || !previewImage || !mediaPreview || !mediaPlaceholder || !imageInput || !croppedInput) {
            return;
        }
        croppedInput.value = "";
        const url = URL.createObjectURL(cropState.originalFile);
        if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
        }
        objectUrl = url;
        previewImage.src = url;
        mediaPreview.classList.remove("d-none");
        mediaPlaceholder.classList.add("d-none");
    };

    const clearMathField = () => {
        if (!mathField) return;
        if (typeof mathField.setValue === "function") {
            mathField.setValue("");
        } else {
            mathField.value = "";
        }
    };

    const getMathLatex = () => {
        if (!mathField) return "";
        if (typeof mathField.getValue === "function") {
            return (mathField.getValue("latex") || "").trim();
        }
        return (mathField.value || "").trim();
    };

    const getMathModalInstance = () => {
        if (!mathModal || !window.bootstrap) return null;
        if (!mathModalInstance) {
            mathModalInstance = window.bootstrap.Modal.getOrCreateInstance(mathModal);
        }
        return mathModalInstance;
    };

    const showMathModal = () => {
        if (!mathModal) return;
        clearMathField();
        const modal = getMathModalInstance();
        if (modal) {
            modal.show();
        } else {
            mathModal.classList.add("show");
            mathModal.style.display = "block";
            mathModal.removeAttribute("aria-hidden");
        }
    };

    const hideMathModal = () => {
        if (!mathModal) return;
        const modal = getMathModalInstance();
        if (modal) {
            modal.hide();
        } else {
            mathModal.classList.remove("show");
            mathModal.style.display = "none";
            mathModal.setAttribute("aria-hidden", "true");
        }
    };

    const initEditor = () => {
        if (!editorContainer || !window.Quill) return;
        const placeholder = editorContainer.dataset.placeholder || "分享你的策略灵感或市场观察...";
        const toolbarConfig = {
            container: [
                ["bold", "italic", "underline"],
                [{ size: ["small", false, "large", "huge"] }],
                [{ color: [] }, { background: [] }],
                [{ list: "ordered" }, { list: "bullet" }],
                ["blockquote", "math"],
                ["clean"],
            ],
            handlers: {
                math: () => showMathModal(),
            },
        };
        quill = new window.Quill(editorContainer, {
            theme: "snow",
            placeholder,
            modules: {
                toolbar: toolbarConfig,
            },
        });
        const toolbar = quill.getModule("toolbar");
        const mathButton = toolbar?.container?.querySelector(".ql-math");
        if (mathButton) {
            mathButton.innerHTML = "Σ";
            mathButton.setAttribute("type", "button");
            mathButton.setAttribute("aria-label", langIsZh ? "插入公式" : "Insert formula");
        }
        if (contentTextarea && contentTextarea.value) {
            const initialValue = contentTextarea.value.trim();
            if (initialValue) {
                if (initialValue.includes("<")) {
                    quill.clipboard.dangerouslyPasteHTML(initialValue);
                } else {
                    quill.setText(initialValue);
                }
            }
        }
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initEditor);
    } else {
        initEditor();
    }

    if (composerForm) {
        composerForm.addEventListener("submit", () => {
            if (!quill || !contentTextarea) return;
            const html = quill.root.innerHTML;
            contentTextarea.value = html === "<p><br></p>" ? "" : html;
        });
    }

    if (mathModal) {
        mathModal.addEventListener("shown.bs.modal", () => {
            if (mathField && typeof mathField.focus === "function") {
                mathField.focus();
            }
        });
        mathModal.addEventListener("hidden.bs.modal", () => {
            if (quill) {
                quill.focus();
            }
        });
    }

    if (mathInsertBtn) {
        mathInsertBtn.addEventListener("click", () => {
            if (!quill) return;
            const latex = getMathLatex();
            if (!latex) return;
            const range = quill.getSelection(true);
            const index = range ? range.index : quill.getLength();
            if (window.katex && typeof quill.insertEmbed === "function") {
                quill.insertEmbed(index, "formula", latex, "user");
                quill.setSelection(index + 1, 0, "silent");
            } else {
                quill.insertText(index, latex, "user");
                quill.setSelection(index + latex.length, 0, "silent");
            }
            hideMathModal();
        });
    }

    if (openComposeBtn && composePanel) {
        openComposeBtn.addEventListener("click", (event) => {
            lastComposeTrigger = event.currentTarget;
            openCompose();
        });
    }

    if (closeComposeBtn) {
        closeComposeBtn.addEventListener("click", () => {
            closeCompose();
            resetMedia();
        });
    }

    if (toggleNewTopicBtn && newTopicFields) {
        toggleNewTopicBtn.addEventListener("click", () => {
            const willShow = newTopicFields.classList.contains("d-none");
            newTopicFields.classList.toggle("d-none", !willShow);
            newTopicFields.setAttribute("aria-hidden", willShow ? "false" : "true");
            toggleNewTopicBtn.setAttribute("aria-expanded", willShow ? "true" : "false");
            if (topicInput) {
                topicInput.value = willShow ? "" : activeTopicId || "";
            }
            if (selectedTopicBadge) {
                selectedTopicBadge.textContent = `#${willShow ? "新话题" : activeTopicName || "话题"}`;
            }
            if (!willShow) {
                newTopicFields.querySelectorAll("input,textarea").forEach((el) => (el.value = ""));
            } else {
                const input = newTopicFields.querySelector("input");
                if (input) {
                    setTimeout(() => input.focus(), 60);
                }
            }
        });
    }

    if (createTopicBtn) {
        createTopicBtn.addEventListener("click", (event) => {
            lastComposeTrigger = event.currentTarget;
            openCompose({ newTopic: true });
            const input = newTopicFields?.querySelector("input");
            if (input) {
                setTimeout(() => input.focus(), 60);
            }
        });
    }

    if (shareHistoryId) {
        openCompose();
        composePanel?.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    if (pickImageBtn) {
        pickImageBtn.addEventListener("click", () => imageInput?.click());
    }

    if (replaceImageBtn) {
        replaceImageBtn.addEventListener("click", () => imageInput?.click());
    }

    if (removeImageBtn) {
        removeImageBtn.addEventListener("click", () => {
            resetMedia();
        });
    }

    if (imageInput && cropperModal) {
        imageInput.addEventListener("change", (event) => {
            const file = event.target.files && event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
                const dataUrl = ev.target?.result;
                if (typeof dataUrl === "string") {
                    openCropper(dataUrl, file);
                }
            };
            reader.readAsDataURL(file);
        });
    }

    if (cropZoom) {
        cropZoom.addEventListener("input", (event) => {
            const target = event.target;
            cropState.zoom = parseFloat(target.value) || 1;
            updateCropperTransform();
        });
    }

    if (cropperStage) {
        const startPointer = (event) => {
            cropState.pointer = {
                id: event.pointerId,
                x: event.clientX,
                y: event.clientY,
                offsetX: cropState.offsetX,
                offsetY: cropState.offsetY,
            };
            cropperStage.setPointerCapture(event.pointerId);
        };
        const movePointer = (event) => {
            if (!cropState.pointer || cropState.pointer.id !== event.pointerId) return;
            const deltaX = event.clientX - cropState.pointer.x;
            const deltaY = event.clientY - cropState.pointer.y;
            cropState.offsetX = cropState.pointer.offsetX + deltaX;
            cropState.offsetY = cropState.pointer.offsetY + deltaY;
            updateCropperTransform();
        };
        const endPointer = (event) => {
            if (cropState.pointer && cropState.pointer.id === event.pointerId) {
                cropState.pointer = null;
                cropperStage.releasePointerCapture(event.pointerId);
            }
        };
        cropperStage.addEventListener("pointerdown", startPointer);
        cropperStage.addEventListener("pointermove", movePointer);
        cropperStage.addEventListener("pointerup", endPointer);
        cropperStage.addEventListener("pointercancel", endPointer);
    }

    if (cropCancel) {
        cropCancel.addEventListener("click", () => {
            closeCropper();
            resetMedia();
        });
    }

    if (cropSkip) {
        cropSkip.addEventListener("click", () => {
            closeCropper();
            useOriginalImage();
        });
    }

    if (cropConfirm) {
        cropConfirm.addEventListener("click", () => {
            applyCroppedImage();
            closeCropper();
        });
    }

    document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        if (cropperModal && !cropperModal.classList.contains("d-none")) {
            event.preventDefault();
            closeCropper();
        }
    });

    const escapeHtml = (value) =>
        value
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");

    const toggleLike = (button) => {
        if (!likeEndpoint) return;
        const post = button.closest("[data-post-id]");
        if (!post) return;
        const postId = post.dataset.postId;
        const formData = new URLSearchParams();
        formData.append("post_id", postId);
        fetch(likeEndpoint, {
            method: "POST",
            headers: {
                "X-CSRFToken": csrfToken(),
                "X-Requested-With": "XMLHttpRequest",
            },
            body: formData,
        })
            .then((res) => res.json())
            .then((data) => {
                if (data && typeof data.like_count !== "undefined") {
                    const countEl = button.querySelector("[data-role='like-count']");
                    if (countEl) {
                        countEl.textContent = data.like_count;
                    }
                    if (data.liked) {
                        button.classList.add("is-active");
                    } else {
                        button.classList.remove("is-active");
                    }
                }
            })
            .catch(() => {
                button.classList.toggle("shake");
                setTimeout(() => button.classList.remove("shake"), 500);
            });
    };

    const submitComment = (post) => {
        if (!commentEndpoint) return;
        const textarea = post.querySelector("[data-role='comment-input']");
        const countEl = post.querySelector("[data-role='comment-count']");
        const list = post.querySelector(".community-comment-list");
        if (!(textarea && list)) return;
        const content = textarea.value.trim();
        if (!content) {
            textarea.focus();
            return;
        }
        const formData = new URLSearchParams();
        formData.append("post_id", post.dataset.postId || "");
        formData.append("content", content);
        fetch(commentEndpoint, {
            method: "POST",
            headers: {
                "X-CSRFToken": csrfToken(),
                "X-Requested-With": "XMLHttpRequest",
            },
            body: formData,
        })
            .then((res) => res.json())
            .then((data) => {
                if (!data || !data.comment) return;
                const li = document.createElement("li");
                li.className = "community-comment-item";
                const avatar = escapeHtml(data.comment.avatar_url || "");
                const author = escapeHtml(data.comment.author || "匿名");
                const text = escapeHtml(data.comment.content || "");
                const time = escapeHtml(data.comment.created_at || "");
                li.innerHTML = `
                    <span class="community-post-avatar small">
                        ${
                            avatar
                                ? `<img src="${avatar}" alt="${author}">`
                                : author.charAt(0).toUpperCase()
                        }
                    </span>
                    <div class="community-comment-body">
                        <div class="community-comment-meta">
                            <span class="community-comment-author">${author}</span>
                            <span class="community-comment-time text-muted small">${time}</span>
                        </div>
                        <p>${text.replace(/\n/g, "<br>")}</p>
                    </div>
                `;
                list.prepend(li);
                textarea.value = "";
                if (countEl) {
                    const current = parseInt(countEl.textContent || "0", 10) || 0;
                    countEl.textContent = current + 1;
                }
            })
            .catch(() => {
                textarea.classList.add("is-invalid");
                setTimeout(() => textarea.classList.remove("is-invalid"), 600);
            });
    };

    const showPostAlert = (kind, message) => {
        if (!postAlert) return;
        postAlert.textContent = message;
        postAlert.className = `alert alert-${kind}`;
        postAlert.classList.remove("d-none");
        clearTimeout(showPostAlert._timer);
        showPostAlert._timer = setTimeout(() => postAlert.classList.add("d-none"), 4000);
    };

    const requestDelete = (form) => {
        const post = form.closest("[data-post-id]");
        if (!post) return;
        const confirmMessage =
            form.dataset.confirm || "Are you sure you want to delete this post? This action cannot be undone.";
        if (!window.confirm(confirmMessage)) {
            return;
        }
        const button = form.querySelector("[data-role='delete-button']");
        if (button) {
            button.disabled = true;
            button.classList.add("opacity-75");
        }
        const tokenInput = form.querySelector("input[name='csrfmiddlewaretoken']");
        const csrfValue = tokenInput ? tokenInput.value : csrfToken();
        const payload = new URLSearchParams(new FormData(form));
        fetch(form.getAttribute("action"), {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRFToken": csrfValue,
                "X-Requested-With": "XMLHttpRequest",
            },
            body: payload.toString(),
        })
            .then((res) => {
                if (!res.ok) {
                    throw new Error("delete_failed");
                }
                return res.json();
            })
            .then((data) => {
                if (data && data.deleted) {
                    showPostAlert("success", data.message || "帖子已删除。");
                    post.classList.add("community-post-removing");
                    setTimeout(() => {
                        post.remove();
                        const remaining = root.querySelectorAll("[data-post-id]");
                        if (!remaining.length) {
                            const emptyState = root.querySelector(".community-empty");
                            emptyState?.classList.remove("d-none");
                        }
                    }, 200);
                } else {
                    throw new Error("delete_failed");
                }
            })
            .catch(() => {
                if (button) {
                    button.disabled = false;
                    button.classList.remove("opacity-75");
                }
                showPostAlert("danger", "删除失败，请稍后再试。");
            });
    };

    root.addEventListener("click", (event) => {
        const likeButton = event.target.closest("[data-role='like-button']");
        if (likeButton && root.contains(likeButton)) {
            toggleLike(likeButton);
            return;
        }

        const toggleButton = event.target.closest("[data-role='toggle-comments']");
        if (toggleButton && root.contains(toggleButton)) {
            const post = toggleButton.closest("[data-post-id]");
            if (!post) return;
            const comments = post.querySelector("[data-role='comments']");
            if (!comments) return;
            comments.classList.toggle("d-none");
            return;
        }

        const submitButton = event.target.closest("[data-role='submit-comment']");
        if (submitButton && root.contains(submitButton)) {
            const post = submitButton.closest("[data-post-id]");
            if (!post) return;
            submitComment(post);
        }
    });

    root.addEventListener("keydown", (event) => {
        const textarea = event.target.closest("[data-role='comment-input']");
        if (!textarea || !root.contains(textarea)) return;
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            const post = textarea.closest("[data-post-id]");
            if (!post) return;
            submitComment(post);
        }
    });

    root.addEventListener("submit", (event) => {
        const form = event.target.closest("[data-role='delete-form']");
        if (form && root.contains(form)) {
            event.preventDefault();
            const confirmMessage =
                form.dataset.confirm || "Are you sure you want to delete this post? This action cannot be undone.";
            if (!window.confirm(confirmMessage)) {
                return;
            }
            requestDelete(form);
        }
    });

    document.body.addEventListener("htmx:afterSwap", (event) => {
        const target = event.detail?.target;
        if (!target) return;
        if (target.id === "community-post-list" || target.closest?.("#community-post-list")) {
            highlightBlocks(target);
        }
    });

    highlightBlocks(root);
})();
