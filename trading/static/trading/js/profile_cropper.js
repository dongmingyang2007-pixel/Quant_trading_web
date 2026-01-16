(() => {
    const ready = (fn) => {
        if (document.readyState === "complete" || document.readyState === "interactive") {
            fn();
        } else {
            document.addEventListener("DOMContentLoaded", fn);
        }
    };

    ready(() => {
        const initFeatureUploader = () => {
            // --- 第一部分：Feature Uploader (展示照片) ---
            const featureUploader = document.querySelector("[data-role='feature-uploader']");
            if (featureUploader) {
                const uploader = featureUploader;
                const fileInput = uploader.querySelector("[data-role='feature-input']");
                const croppedInput = uploader.querySelector("[data-role='feature-cropped']");
                const placeholder = uploader.querySelector("[data-role='feature-placeholder']");
                const preview = uploader.querySelector("[data-role='feature-preview']");
                const previewImage = uploader.querySelector("[data-role='feature-preview-image']");
                const pickBtn = uploader.querySelector("[data-role='feature-pick']");
                const replaceBtn = uploader.querySelector("[data-role='feature-replace']");
                const removeBtn = uploader.querySelector("[data-role='feature-remove']");
                const statusLabel = uploader.querySelector("[data-role='feature-status']");
    
                const modal = document.querySelector("[data-role='feature-cropper']");
                const cropperImage = modal ? modal.querySelector("[data-role='crop-image']") : null;
                const cropperStage = modal ? modal.querySelector("[data-role='crop-stage']") : null;
                const cropperFrame = modal ? modal.querySelector("[data-role='crop-frame']") : null;
                const cropZoom = modal ? modal.querySelector("[data-role='crop-zoom']") : null;
                const cropCancel = modal ? modal.querySelector("[data-role='crop-cancel']") : null;
                const cropSkip = modal ? modal.querySelector("[data-role='crop-skip']") : null;
                const cropConfirm = modal ? modal.querySelector("[data-role='crop-confirm']") : null;
    
                const openPicker = () => {
                    if (fileInput) fileInput.click();
                };
    
                const fallbackToNative = () => {
                    if (fileInput) {
                        fileInput.classList.remove("d-none");
                        fileInput.removeAttribute("data-role");
                        fileInput.classList.add("form-control");
                    }
                    if (pickBtn) pickBtn.remove();
                    if (replaceBtn) replaceBtn.remove();
                    if (removeBtn) removeBtn.remove();
                    if (placeholder) placeholder.remove();
                    if (preview) preview.remove();
                    if (statusLabel) statusLabel.textContent = "";
                };
    
                const canUseCropper =
                    fileInput &&
                    croppedInput &&
                    placeholder &&
                    preview &&
                    previewImage &&
                    pickBtn &&
                    replaceBtn &&
                    removeBtn &&
                    statusLabel &&
                    modal &&
                    cropperImage &&
                    cropperStage &&
                    cropZoom &&
                    cropCancel &&
                    cropSkip &&
                    cropConfirm;
    
                if (!canUseCropper) {
                    fallbackToNative();
                }
    
                if (canUseCropper) {
                    const langAttr = (document.documentElement.getAttribute("lang") || "zh").toLowerCase();
                    const langIsZh = langAttr.indexOf("zh") === 0;
    
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
                    };
    
                    const releaseObjectUrl = () => {
                        if (objectUrl) {
                            URL.revokeObjectURL(objectUrl);
                            objectUrl = null;
                        }
                    };
    
                    const showStatus = (message, variant) => {
                        const tone = variant || "muted";
                        statusLabel.textContent = message || "";
                        statusLabel.classList.remove("text-muted", "text-success", "text-danger");
                        statusLabel.classList.add("text-" + tone);
                    };
    
                    const showPreview = (src) => {
                        previewImage.src = src;
                        preview.classList.remove("d-none");
                        placeholder.classList.add("d-none");
                        replaceBtn.classList.remove("d-none");
                        removeBtn.classList.remove("d-none");
                    };
    
                    const resetPreview = () => {
                        releaseObjectUrl();
                        preview.classList.add("d-none");
                        previewImage.src = "";
                        placeholder.classList.remove("d-none");
                        replaceBtn.classList.add("d-none");
                        removeBtn.classList.add("d-none");
                        fileInput.value = "";
                        croppedInput.value = "";
                        showStatus("", "muted");
                    };
    
                    const clampOffsets = () => {
                        const scale = cropState.baseScale * cropState.zoom;
                        const displayWidth = cropState.imgWidth * scale;
                        const displayHeight = cropState.imgHeight * scale;
                        const maxOffsetX = Math.max(0, (displayWidth - cropState.stageWidth) / 2);
                        const maxOffsetY = Math.max(0, (displayHeight - cropState.stageHeight) / 2);
                        cropState.offsetX = Math.min(maxOffsetX, Math.max(-maxOffsetX, cropState.offsetX));
                        cropState.offsetY = Math.min(maxOffsetY, Math.max(-maxOffsetY, cropState.offsetY));
                    };
    
                    const updateCropperTransform = () => {
                        clampOffsets();
                        const scale = cropState.baseScale * cropState.zoom;
                        cropperImage.style.transform =
                            "translate(-50%, -50%) translate(" + cropState.offsetX + "px, " + cropState.offsetY + "px) scale(" + scale + ")";
                    };
    
                    const openCropper = (dataUrl) => {
                        cropState.dataUrl = dataUrl;
                        cropState.offsetX = 0;
                        cropState.offsetY = 0;
                        cropState.zoom = 1;
                        cropZoom.value = "1";
                        cropperImage.onload = () => {
                            window.requestAnimationFrame(() => {
                            const stageSource = cropperFrame || cropperStage;
                            const rect = stageSource.getBoundingClientRect();
                            cropState.stageWidth = rect.width || stageSource.offsetWidth || 640;
                            cropState.stageHeight = rect.height || stageSource.offsetHeight || 360;
                                cropState.imgWidth = cropperImage.naturalWidth;
                                cropState.imgHeight = cropperImage.naturalHeight;
                                cropState.baseScale = Math.max(
                                    cropState.stageWidth / cropState.imgWidth,
                                    cropState.stageHeight / cropState.imgHeight
                                );
                                updateCropperTransform();
                            });
                        };
                        cropperImage.src = dataUrl;
                        document.body.classList.add("profile-cropper-open");
                        modal.classList.remove("d-none");
                        modal.setAttribute("aria-hidden", "false");
                    };
    
                    const closeCropper = () => {
                        modal.classList.add("d-none");
                        modal.setAttribute("aria-hidden", "true");
                        cropState.pointer = null;
                        document.body.classList.remove("profile-cropper-open");
                    };
    
                    const applyCrop = () => {
                        const canvas = document.createElement("canvas");
                        const targetWidth = 1280;
                        const ratio = cropState.stageHeight / cropState.stageWidth || 0.5625;
                        const targetHeight = Math.round(targetWidth * ratio);
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
                        const sWidth = Math.min(cropState.imgWidth, (cropState.stageWidth / displayWidth) * cropState.imgWidth);
                        const sHeight = Math.min(cropState.imgHeight, (cropState.stageHeight / displayHeight) * cropState.imgHeight);
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
                        fileInput.value = "";
                        showPreview(dataUrl);
                        showStatus(langIsZh ? "已选择（裁剪后）" : "Selected (cropped)", "success");
                        closeCropper();
                    };
    
                    const handleFile = (file) => {
                        if (!file) return;
                        releaseObjectUrl();
                        const reader = new FileReader();
                        let resolved = false;
                        reader.onload = (event) => {
                            const dataUrl = event && event.target ? event.target.result : null;
                            if (typeof dataUrl === "string") {
                                resolved = true;
                                openCropper(dataUrl);
                            }
                        };
                        reader.onloadend = () => {
                            if (!resolved) {
                                objectUrl = URL.createObjectURL(file);
                                openCropper(objectUrl);
                            }
                        };
                        reader.readAsDataURL(file);
                    };
    
                    cropZoom.addEventListener("input", () => {
                        cropState.zoom = parseFloat(cropZoom.value) || 1;
                        updateCropperTransform();
                    });
    
                    cropperStage.addEventListener("pointerdown", (event) => {
                        cropState.pointer = { x: event.clientX, y: event.clientY };
                        cropperStage.setPointerCapture(event.pointerId);
                    });
    
                    cropperStage.addEventListener("pointermove", (event) => {
                        if (!cropState.pointer) return;
                        const dx = event.clientX - cropState.pointer.x;
                        const dy = event.clientY - cropState.pointer.y;
                        cropState.pointer = { x: event.clientX, y: event.clientY };
                        cropState.offsetX += dx;
                        cropState.offsetY += dy;
                        updateCropperTransform();
                    });
    
                    const releasePointer = (event) => {
                        if (cropState.pointer) {
                            cropperStage.releasePointerCapture(event.pointerId);
                        }
                        cropState.pointer = null;
                    };
    
                    cropperStage.addEventListener("pointerup", releasePointer);
                    cropperStage.addEventListener("pointercancel", releasePointer);
    
                    cropConfirm.addEventListener("click", applyCrop);
                    cropSkip.addEventListener("click", () => {
                        showStatus(langIsZh ? "已选择原图" : "Using original", "success");
                        replaceBtn.classList.remove("d-none");
                        removeBtn.classList.remove("d-none");
                        closeCropper();
                    });
                    cropCancel.addEventListener("click", () => {
                        fileInput.value = "";
                        closeCropper();
                    });
    
                    window.addEventListener("keydown", (event) => {
                        if (event.key === "Escape" && !modal.classList.contains("d-none")) {
                            closeCropper();
                        }
                    });
    
                    pickBtn.addEventListener("click", openPicker);
                    replaceBtn.addEventListener("click", openPicker);
                    removeBtn.addEventListener("click", resetPreview);
    
                    fileInput.addEventListener("change", () => {
                        const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
                        if (!file) return;
                        handleFile(file);
                    });
    
                    const initialImage = uploader.getAttribute("data-initial-image");
                    if (initialImage) {
                        showPreview(initialImage);
                        showStatus(langIsZh ? "已上传" : "Photo uploaded", "muted");
                    } else {
                        resetPreview();
                    }
                }
            } // 这里原来多了一个 '}', 导致后面的 Avatar 代码无法执行。已修复。
    
        };
        const initAvatarUploader = () => {
            // --- 第二部分：Avatar Uploader (头像上传) ---
            const avatarUploader = document.querySelector("[data-role='avatar-uploader']") || document;
            if (avatarUploader) {
                const fileInput =
                    avatarUploader.querySelector("[data-role='avatar-input']") ||
                    avatarUploader.querySelector("#avatar-file-input") ||
                    avatarUploader.querySelector("input[type='file'][name='avatar']") ||
                    document.getElementById("avatar-file-input") ||
                    document.getElementById("id_avatar");
                const croppedInput =
                    avatarUploader.querySelector("[data-role='avatar-cropped']") ||
                    avatarUploader.querySelector("#avatar-cropped-input") ||
                    avatarUploader.querySelector("input[name='avatar_cropped_data']") ||
                    document.getElementById("avatar-cropped-input");
            const dropzone = avatarUploader.querySelector("[data-role='avatar-dropzone']");
            const pickBtn = avatarUploader.querySelector("[data-role='avatar-pick']");
            const cropper = avatarUploader.querySelector("[data-role='avatar-cropper']");
            const cropperStage = avatarUploader.querySelector("[data-role='avatar-crop-stage']");
            const cropperFrame = avatarUploader.querySelector("[data-role='avatar-crop-frame']");
            const cropperImage = avatarUploader.querySelector("[data-role='avatar-crop-image']");
                const cropZoom = avatarUploader.querySelector("[data-role='avatar-crop-zoom']");
                const statusLabel = avatarUploader.querySelector("[data-role='avatar-status']");
                const saveBtn = document.querySelector("[data-role='avatar-save']");
                const modalEl = document.getElementById("avatar-upload-modal");
                const modalInstance = modalEl && window.bootstrap
                    ? window.bootstrap.Modal.getOrCreateInstance(modalEl)
                    : null;
                let pendingSave = false;
                let pendingPick = false;
                let lastInputValue = "";
                let pickPoll = null;
                let modalPoll = null;
                let lastFileSignature = "";
                let globalModalPoll = null;
    
                if (fileInput && !fileInput.hasAttribute("data-role")) {
                    fileInput.setAttribute("data-role", "avatar-input");
                }
                if (croppedInput && !croppedInput.hasAttribute("data-role")) {
                    croppedInput.setAttribute("data-role", "avatar-cropped");
                }
    
                const hasPicker = Boolean(dropzone || pickBtn);
                if (!fileInput || !saveBtn || !hasPicker) {
                    if (fileInput) {
                        fileInput.classList.remove("d-none");
                        fileInput.classList.add("form-control");
                    }
                    return;
                }
    
                const langAttr = (document.documentElement.getAttribute("lang") || "zh").toLowerCase();
                const langIsZh = langAttr.indexOf("zh") === 0;
                let objectUrl = null;
                const supportsCropper = Boolean(croppedInput && cropper && cropperStage && cropperImage && cropZoom);
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
                };
    
                const showStatus = (message, variant) => {
                    if (!statusLabel) return;
                    const tone = variant || "muted";
                    statusLabel.textContent = message || "";
                    statusLabel.classList.remove("text-muted", "text-success", "text-danger");
                    statusLabel.classList.add("text-" + tone);
                };
    
                const getFileSignature = (file) => {
                    if (!file) return "";
                    return `${file.name}-${file.size}-${file.lastModified}`;
                };
    
                const resetUploader = () => {
                    if (cropper) {
                        cropper.classList.add("d-none");
                    }
                    if (dropzone) {
                        dropzone.classList.remove("d-none");
                    }
                    if (cropperImage) {
                        cropperImage.src = "";
                    }
                    cropState.dataUrl = "";
                    cropState.pointer = null;
                    if (cropZoom) {
                        cropZoom.value = "1";
                    }
                    fileInput.value = "";
                    if (croppedInput) {
                        croppedInput.value = "";
                    }
                    lastFileSignature = "";
                    if (objectUrl) {
                        URL.revokeObjectURL(objectUrl);
                        objectUrl = null;
                    }
                    showStatus("", "muted");
                };
    
                const clampOffsets = () => {
                    const scale = cropState.baseScale * cropState.zoom;
                    const displayWidth = cropState.imgWidth * scale;
                    const displayHeight = cropState.imgHeight * scale;
                    const maxOffsetX = Math.max(0, (displayWidth - cropState.stageWidth) / 2);
                    const maxOffsetY = Math.max(0, (displayHeight - cropState.stageHeight) / 2);
                    cropState.offsetX = Math.min(maxOffsetX, Math.max(-maxOffsetX, cropState.offsetX));
                    cropState.offsetY = Math.min(maxOffsetY, Math.max(-maxOffsetY, cropState.offsetY));
                };
    
                const updateCropperTransform = () => {
                    clampOffsets();
                    const scale = cropState.baseScale * cropState.zoom;
                    cropperImage.style.transform =
                        "translate(-50%, -50%) translate(" + cropState.offsetX + "px, " + cropState.offsetY + "px) scale(" + scale + ")";
                };
    
                const openCropper = (dataUrl) => {
                    if (!supportsCropper) {
                        showStatus(langIsZh ? "已选择头像，点击保存" : "Avatar selected. Click save.", "success");
                        return;
                    }
                    cropState.dataUrl = dataUrl;
                    cropState.offsetX = 0;
                    cropState.offsetY = 0;
                    cropState.zoom = 1;
                    cropZoom.value = "1";
                    cropperImage.onload = () => {
                        window.requestAnimationFrame(() => {
                        const stageSource = cropperFrame || cropperStage;
                        const rect = stageSource.getBoundingClientRect();
                        cropState.stageWidth = rect.width || stageSource.offsetWidth || 320;
                        cropState.stageHeight = rect.height || stageSource.offsetHeight || 320;
                            cropState.imgWidth = cropperImage.naturalWidth;
                            cropState.imgHeight = cropperImage.naturalHeight;
                            cropState.baseScale = Math.max(
                                cropState.stageWidth / cropState.imgWidth,
                                cropState.stageHeight / cropState.imgHeight
                            );
                            updateCropperTransform();
                        });
                    };
                    cropperImage.src = dataUrl;
                    if (dropzone) {
                        dropzone.classList.add("d-none");
                    }
                    cropper.classList.remove("d-none");
                };
    
                const applyCrop = () => {
                    if (!supportsCropper) {
                        return null;
                    }
                    if (!cropState.dataUrl) {
                        showStatus(langIsZh ? "请先选择图片" : "Select an image first", "danger");
                        return null;
                    }
                    const canvas = document.createElement("canvas");
                    const targetWidth = 640;
                    const targetHeight = 640;
                    canvas.width = targetWidth;
                    canvas.height = targetHeight;
                    const ctx = canvas.getContext("2d");
                    if (!ctx) return null;
                    const scale = cropState.baseScale * cropState.zoom;
                    const displayWidth = cropState.imgWidth * scale;
                    const displayHeight = cropState.imgHeight * scale;
                    const topLeftX = (cropState.stageWidth - displayWidth) / 2 + cropState.offsetX;
                    const topLeftY = (cropState.stageHeight - displayHeight) / 2 + cropState.offsetY;
                    const sx = Math.max(0, (-topLeftX / displayWidth) * cropState.imgWidth);
                    const sy = Math.max(0, (-topLeftY / displayHeight) * cropState.imgHeight);
                    const sWidth = Math.min(cropState.imgWidth, (cropState.stageWidth / displayWidth) * cropState.imgWidth);
                    const sHeight = Math.min(cropState.imgHeight, (cropState.stageHeight / displayHeight) * cropState.imgHeight);
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
                    fileInput.value = "";
                    showStatus(langIsZh ? "头像已就绪，可保存" : "Avatar ready to save", "success");
                    return dataUrl;
                };
    
                const handleFile = (file) => {
                    if (!file) return;
                    lastFileSignature = getFileSignature(file);
                    showStatus(langIsZh ? "正在加载..." : "Loading image...", "muted");
                    if (objectUrl) {
                        URL.revokeObjectURL(objectUrl);
                        objectUrl = null;
                    }
                    const reader = new FileReader();
                    let resolved = false;
                    reader.onload = (event) => {
                        const dataUrl = event && event.target ? event.target.result : null;
                        if (typeof dataUrl === "string") {
                            resolved = true;
                            openCropper(dataUrl);
                        }
                    };
                    reader.onloadend = () => {
                        if (!resolved) {
                            objectUrl = URL.createObjectURL(file);
                            openCropper(objectUrl);
                        }
                    };
                    reader.readAsDataURL(file);
                };
    
                const startPickPoll = () => {
                    if (pickPoll) {
                        window.clearInterval(pickPoll);
                        pickPoll = null;
                    }
                    const startAt = Date.now();
                    pickPoll = window.setInterval(() => {
                        const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
                        if (file) {
                            window.clearInterval(pickPoll);
                            pickPoll = null;
                            handleFile(file);
                            return;
                        }
                        if (Date.now() - startAt > 3500) {
                            window.clearInterval(pickPoll);
                            pickPoll = null;
                        }
                    }, 200);
                };
    
                const startModalWatch = () => {
                    if (modalPoll) {
                        window.clearInterval(modalPoll);
                        modalPoll = null;
                    }
                    modalPoll = window.setInterval(() => {
                        if (!supportsCropper || !cropper) {
                            return;
                        }
                        const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
                        if (!file) {
                            return;
                        }
                        const signature = getFileSignature(file);
                        const isSame = signature && signature === lastFileSignature;
                        const cropperVisible = !cropper.classList.contains("d-none");
                        if (!cropperVisible || !isSame) {
                            handleFile(file);
                        }
                    }, 300);
                };
    
                const stopModalWatch = () => {
                    if (modalPoll) {
                        window.clearInterval(modalPoll);
                        modalPoll = null;
                    }
                };
    
                const startGlobalModalWatch = () => {
                    if (globalModalPoll) {
                        window.clearInterval(globalModalPoll);
                        globalModalPoll = null;
                    }
                    globalModalPoll = window.setInterval(() => {
                        if (!modalEl || !modalEl.classList.contains("show")) {
                            return;
                        }
                        if (!supportsCropper || !cropper) {
                            return;
                        }
                        const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
                        if (!file) {
                            return;
                        }
                        const signature = getFileSignature(file);
                        const isSame = signature && signature === lastFileSignature;
                        const cropperVisible = !cropper.classList.contains("d-none");
                        if (!cropperVisible || !isSame) {
                            handleFile(file);
                        }
                    }, 400);
                };
    
                const pickFile = () => {
                    pendingPick = true;
                    lastInputValue = fileInput.value;
                    startPickPoll();
                    if (typeof fileInput.showPicker === "function") {
                        fileInput.showPicker();
                    } else {
                        fileInput.click();
                    }
                };
    
                if (pickBtn) {
                    pickBtn.addEventListener("click", pickFile);
                }
                if (dropzone) {
                    dropzone.addEventListener("click", () => {
                        pendingPick = true;
                        lastInputValue = fileInput.value;
                        startPickPoll();
                    });
                    if (dropzone.tagName.toLowerCase() !== "label") {
                        dropzone.addEventListener("click", pickFile);
                    }
                }
    
                if (dropzone) {
                    ["dragenter", "dragover"].forEach((eventName) => {
                        dropzone.addEventListener(eventName, (event) => {
                            event.preventDefault();
                            dropzone.classList.add("is-dragging");
                        });
                    });
                    ["dragleave", "dragend"].forEach((eventName) => {
                        dropzone.addEventListener(eventName, () => {
                            dropzone.classList.remove("is-dragging");
                        });
                    });
                    dropzone.addEventListener("drop", (event) => {
                        event.preventDefault();
                        dropzone.classList.remove("is-dragging");
                        const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
                        if (file) {
                            handleFile(file);
                        }
                    });
                }
    
                const handleFileChange = (input) => {
                    const file = input.files && input.files.length ? input.files[0] : null;
                    if (!file) return;
                    handleFile(file);
                };
    
                fileInput.addEventListener("change", () => {
                    handleFileChange(fileInput);
                });
    
                fileInput.addEventListener("input", () => {
                    handleFileChange(fileInput);
                });
    
                fileInput.addEventListener("click", () => {
                    pendingPick = true;
                    lastInputValue = fileInput.value;
                    startPickPoll();
                });
    
                document.addEventListener(
                    "change",
                    (event) => {
                        const target = event.target;
                        if (
                            target &&
                            target !== fileInput &&
                            target.matches("input[type='file'][name='avatar'], #id_avatar, [data-role='avatar-input']")
                        ) {
                            handleFileChange(target);
                        }
                    },
                    true
                );
    
                window.addEventListener("focus", () => {
                    if (!pendingPick) return;
                    pendingPick = false;
                    const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
                    if (file) {
                        handleFile(file);
                        return;
                    }
                    if (fileInput.value && fileInput.value !== lastInputValue) {
                        const fallbackFile = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
                        if (fallbackFile) {
                            handleFile(fallbackFile);
                        }
                    }
                });
    
                if (supportsCropper) {
                    cropZoom.addEventListener("input", () => {
                        cropState.zoom = parseFloat(cropZoom.value) || 1;
                        updateCropperTransform();
                    });
    
                    cropperStage.addEventListener("pointerdown", (event) => {
                        cropState.pointer = { x: event.clientX, y: event.clientY };
                        cropperStage.setPointerCapture(event.pointerId);
                    });
                    cropperStage.addEventListener("pointermove", (event) => {
                        if (!cropState.pointer) return;
                        const dx = event.clientX - cropState.pointer.x;
                        const dy = event.clientY - cropState.pointer.y;
                        cropState.pointer = { x: event.clientX, y: event.clientY };
                        cropState.offsetX += dx;
                        cropState.offsetY += dy;
                        updateCropperTransform();
                    });
                    const releasePointer = (event) => {
                        if (cropState.pointer) {
                            cropperStage.releasePointerCapture(event.pointerId);
                        }
                        cropState.pointer = null;
                    };
                    cropperStage.addEventListener("pointerup", releasePointer);
                    cropperStage.addEventListener("pointercancel", releasePointer);
                }
    
                saveBtn.addEventListener("click", (event) => {
                    event.preventDefault();
                    const form = document.getElementById("account-profile-form");
                    const hasFile = fileInput.files && fileInput.files.length;
                    if (!supportsCropper) {
                        if (hasFile && form) {
                            pendingSave = true;
                            if (typeof form.requestSubmit === "function") {
                                form.requestSubmit();
                            } else {
                                form.submit();
                            }
                            if (modalInstance) {
                                modalInstance.hide();
                            }
                        }
                        return;
                    }
    
                    const dataUrl = applyCrop();
                    if (!dataUrl) {
                        if (hasFile && form) {
                            pendingSave = true;
                            if (typeof form.requestSubmit === "function") {
                                form.requestSubmit();
                            } else {
                                form.submit();
                            }
                            if (modalInstance) {
                                modalInstance.hide();
                            }
                        }
                        return;
                    }
                    pendingSave = true;
                    if (form && typeof form.requestSubmit === "function") {
                        form.requestSubmit();
                    } else if (form) {
                        form.submit();
                    }
                    if (modalInstance) {
                        modalInstance.hide();
                    }
                });
    
                if (modalEl) {
                    modalEl.addEventListener("shown.bs.modal", startModalWatch);
                    modalEl.addEventListener("hidden.bs.modal", () => {
                        stopModalWatch();
                        if (!pendingSave) {
                            resetUploader();
                        }
                        pendingSave = false;
                    });
                    if (modalEl.classList.contains("show")) {
                        startModalWatch();
                    }
                }
                startGlobalModalWatch();
            }
        };
        initFeatureUploader();
        initAvatarUploader();
    });
})();
