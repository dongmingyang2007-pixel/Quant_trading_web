(() => {
    const ready = (fn) => {
        if (document.readyState === "complete" || document.readyState === "interactive") {
            fn();
        } else {
            document.addEventListener("DOMContentLoaded", fn);
        }
    };

    ready(() => {
        const uploader = document.querySelector("[data-role='feature-uploader']");
        if (!uploader) return;

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
            if (pickBtn) {
                pickBtn.remove();
            }
            if (replaceBtn) replaceBtn.remove();
            if (removeBtn) removeBtn.remove();
            if (placeholder) placeholder.remove();
            if (preview) preview.remove();
            if (statusLabel) statusLabel.textContent = "";
        };

        if (
            !fileInput ||
            !croppedInput ||
            !placeholder ||
            !preview ||
            !previewImage ||
            !pickBtn ||
            !replaceBtn ||
            !removeBtn ||
            !statusLabel ||
            !modal ||
            !cropperImage ||
            !cropperStage ||
            !cropZoom ||
            !cropCancel ||
            !cropSkip ||
            !cropConfirm
        ) {
            fallbackToNative();
            return;
        }

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
                    const rect = cropperStage.getBoundingClientRect();
                    cropState.stageWidth = rect.width || cropperStage.offsetWidth || 640;
                    cropState.stageHeight = rect.height || cropperStage.offsetHeight || 360;
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
    });
})();
