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
    const composerForm = composePanel?.querySelector("[data-role='composer']");
    const contentTextarea = composerForm?.querySelector("textarea[name='content']");
    const croppedInput = composerForm?.querySelector("[data-role='cropped-data']");
    const imageInput = composerForm?.querySelector("[data-role='image-input']");
    const mediaPanel = composerForm?.querySelector("[data-role='media-panel']");
    const mediaPlaceholder = composerForm?.querySelector("[data-role='media-placeholder']");
    const mediaPreview = composerForm?.querySelector("[data-role='media-preview']");
    const previewImage = composerForm?.querySelector("[data-role='preview-image']");
    const pickImageBtn = composerForm?.querySelector("[data-role='pick-image']");
    const replaceImageBtn = composerForm?.querySelector("[data-role='replace-image']");
    const removeImageBtn = composerForm?.querySelector("[data-role='remove-image']");

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

    let objectUrl = null;

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

    const openCompose = (opts = {}) => {
        if (!composePanel) return;
        composePanel.classList.remove("d-none");
        if (contentTextarea) {
            setTimeout(() => contentTextarea.focus(), 50);
        }
        if (topicInput && !opts.newTopic) {
            topicInput.value = activeTopicId || "";
        }
        if (selectedTopicBadge) {
            selectedTopicBadge.textContent = `#${opts.newTopic ? "新话题" : activeTopicName || "话题"}`;
        }
        if (newTopicFields) {
            newTopicFields.classList.toggle("d-none", !opts.newTopic);
            if (!opts.newTopic) {
                newTopicFields.querySelectorAll("input,textarea").forEach((el) => {
                    el.value = "";
                });
            }
        }
        if (topicInput && opts.newTopic) {
            topicInput.value = "";
        }
    };

    const closeCompose = () => {
        if (!composePanel) return;
        composePanel.classList.add("d-none");
        if (newTopicFields) {
            newTopicFields.classList.add("d-none");
            newTopicFields.querySelectorAll("input,textarea").forEach((el) => (el.value = ""));
        }
        if (topicInput) {
            topicInput.value = activeTopicId || "";
        }
        if (selectedTopicBadge) {
            selectedTopicBadge.textContent = `#${activeTopicName || "话题"}`;
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
        cropperModal.classList.remove("d-none");
        cropperModal.setAttribute("aria-hidden", "false");
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

    if (openComposeBtn && composePanel) {
        openComposeBtn.addEventListener("click", () => openCompose());
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
        createTopicBtn.addEventListener("click", () => {
            openCompose({ newTopic: true });
            const input = newTopicFields?.querySelector("input");
            if (input) {
                setTimeout(() => input.focus(), 60);
            }
        });
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

    root.querySelectorAll("[data-role='like-button']").forEach((button) => {
        button.addEventListener("click", () => toggleLike(button));
    });

    root.querySelectorAll("[data-role='toggle-comments']").forEach((button) => {
        button.addEventListener("click", () => {
            const post = button.closest("[data-post-id]");
            if (!post) return;
            const comments = post.querySelector("[data-role='comments']");
            if (!comments) return;
            comments.classList.toggle("d-none");
        });
    });

    root.querySelectorAll("[data-role='submit-comment']").forEach((button) => {
        const post = button.closest("[data-post-id]");
        if (!post) return;
        button.addEventListener("click", () => submitComment(post));
        const textarea = post.querySelector("[data-role='comment-input']");
        if (textarea) {
            textarea.addEventListener("keydown", (event) => {
                if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                    event.preventDefault();
                    submitComment(post);
                }
            });
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
})();
