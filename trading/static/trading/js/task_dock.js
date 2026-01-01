(() => {
    const STORAGE_KEY = "backtestTaskStore";
    if (window.__globalTaskDockMounted) return;
    window.__globalTaskDockMounted = true;

    const langIsZh = (document.documentElement.lang || "").toLowerCase().startsWith("zh");
    const bodyDataset = document.body ? document.body.dataset || {} : {};
    const statusTemplate = bodyDataset.taskStatusTemplate || "";
    const historyBase = bodyDataset.taskHistoryBase || "";
    const cancelTemplate = bodyDataset.taskCancelTemplate || "";
    const pollInterval = 5000;
    const statusLabels = langIsZh
        ? {
              SUBMIT: "已提交",
              SUBMITTED: "已提交",
              PENDING: "已排队",
              STARTED: "运行中",
              PROGRESS: "运行中",
              RETRY: "重试中",
              SUCCESS: "已完成",
              FAILURE: "执行失败",
              REVOKED: "已取消",
          }
        : {
              SUBMIT: "Submitted",
              SUBMITTED: "Submitted",
              PENDING: "Queued",
              STARTED: "Running",
              PROGRESS: "Running",
              RETRY: "Retrying",
              SUCCESS: "Complete",
              FAILURE: "Failed",
              REVOKED: "Cancelled",
          };
    const stageLabels = langIsZh
        ? {
              bootstrap: "准备中",
              finalizing: "收尾中",
              benchmark: "评估中",
              rl_pipeline: "强化学习中",
          }
        : {
              bootstrap: "Bootstrapping",
              finalizing: "Finalizing",
              benchmark: "Benchmarking",
              rl_pipeline: "RL pipeline",
          };

    const labelForStatus = (status) => statusLabels[status] || (langIsZh ? "排队中" : "Queued");
    const labelForStage = (stage) => stageLabels[stage] || stage || "";
    const statusClassFor = (status) => {
        if (status === "SUCCESS") return "success";
        if (status === "FAILURE" || status === "REVOKED") return "failure";
        return "pending";
    };
    const shouldHideId = (value) => /^pending-|^history-|^sync-/i.test(String(value || ""));
    const shortenId = (value) => {
        const raw = String(value || "").trim();
        if (!raw) return "";
        if (raw.length <= 6) return raw.toUpperCase();
        return raw.slice(-6).toUpperCase();
    };
    const copyTaskId = (value, button) => {
        if (!value) return;
        const original = button ? button.textContent : "";
        const feedback = (success) => {
            if (!button) return;
            button.dataset.originalLabel = original || button.dataset.originalLabel || "";
            button.textContent = success ? (langIsZh ? "已复制" : "Copied") : langIsZh ? "复制失败" : "Copy failed";
            window.setTimeout(() => {
                button.textContent = button.dataset.originalLabel || (langIsZh ? "复制 ID" : "Copy ID");
            }, 1500);
        };
        const fallback = () => {
            try {
                const textarea = document.createElement("textarea");
                textarea.value = value;
                textarea.style.position = "fixed";
                textarea.style.opacity = "0";
                document.body.appendChild(textarea);
                textarea.focus();
                textarea.select();
                const ok = document.execCommand("copy");
                document.body.removeChild(textarea);
                feedback(ok);
            } catch (_error) {
                feedback(false);
            }
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(value).then(
                () => feedback(true),
                () => fallback()
            );
            return;
        }
        fallback();
    };

    const getCsrfToken = () => {
        const match = document.cookie.match(/csrftoken=([^;]+)/i);
        return match ? decodeURIComponent(match[1]) : "";
    };

    const loadStore = () => {
        try {
            const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
            if (!parsed || !Array.isArray(parsed.tasks)) return { tasks: [] };
            return parsed;
        } catch (_error) {
            return { tasks: [] };
        }
    };

    const saveStore = (tasks) => {
        try {
            localStorage.setItem(
                STORAGE_KEY,
                JSON.stringify({
                    statusTemplate,
                    historyBaseUrl: historyBase,
                    tasks,
                })
            );
        } catch (_error) {
            // ignore quota failures
        }
    };

    const buildStatusUrl = (taskId) =>
        statusTemplate && taskId
            ? statusTemplate.replace("TASK_ID_PLACEHOLDER", encodeURIComponent(taskId))
            : "";
    const buildCancelUrl = (taskId) =>
        cancelTemplate && taskId
            ? cancelTemplate.replace("TASK_ID_PLACEHOLDER", encodeURIComponent(taskId))
            : "";

const dock =
    document.getElementById("task-dock") ||
    (() => {
        const node = document.createElement("div");
        node.id = "task-dock";
            node.className = "task-dock minimized";
            node.innerHTML = `
                <div id="task-dock-header" class="task-dock-header">
                    <div class="task-header-left">
                        <span class="task-dot"></span>
                        <span class="task-title">${langIsZh ? "回测任务" : "Backtest Tasks"}</span>
                    </div>
                    <div class="task-header-right">
                        <span id="task-count" class="task-count">0</span>
                        <button id="task-toggle" class="task-toggle" aria-label="${langIsZh ? "展开/折叠任务" : "Toggle tasks"}">▾</button>
                    </div>
                </div>
                <div id="task-dock-body" class="task-dock-body" hidden></div>
            `;
            document.body.appendChild(node);
            return node;
        })();

const dockHeader = dock.querySelector("#task-dock-header");
const dockBody = dock.querySelector("#task-dock-body");
const dockToggle = dock.querySelector("#task-toggle");
const dockCount = dock.querySelector("#task-count");
const dockClear = dock.querySelector("#task-clear");
const toastContainer =
    document.getElementById("task-toast-container") ||
    (() => {
        const node = document.createElement("div");
        node.id = "task-toast-container";
        document.body.appendChild(node);
        return node;
    })();
let dockMinimized = true;
let dragState = null;

const tasks = new Map();

const showToast = (task) => {
    if (!toastContainer || !task) return;
    const label = task.ticker || (langIsZh ? "回测任务" : "Backtest task");
    const linkLabel = langIsZh ? "查看报告" : "View report";
    const message = task.message || (langIsZh ? "回测完成，可查看报告。" : "Backtest finished. View the report.");
    const toast = document.createElement("div");
    toast.className = "task-toast";
    toast.innerHTML = `
        <div class="task-toast-head">
            <span class="task-toast-label">${label}</span>
            <span class="task-toast-status success">${labelForStatus("SUCCESS")}</span>
            <button type="button" class="task-toast-close" aria-label="${langIsZh ? "关闭提醒" : "Dismiss"}">×</button>
        </div>
        <div class="task-toast-body">${message}</div>
        ${task.link ? `<a href="${task.link}" target="_blank" rel="noopener" class="task-toast-link">${linkLabel}</a>` : ""}
    `;
    const close = () => {
        toast.classList.add("is-leaving");
        window.setTimeout(() => toast.remove(), 220);
    };
    toast.querySelector(".task-toast-close")?.addEventListener("click", (event) => {
        event.stopPropagation();
        close();
    });
    toast.addEventListener("click", () => {
        if (task.link) {
            window.open(task.link, "_blank", "noopener");
        }
        close();
    });
    toastContainer.appendChild(toast);
    window.setTimeout(close, 4200);
};

    const renderTasks = () => {
        if (!dockBody) return;
        dockBody.innerHTML = "";
        if (tasks.size === 0) {
            const empty = document.createElement("div");
            empty.className = "task-pill";
            empty.innerHTML = `<div class="task-meta">${langIsZh ? "暂无任务" : "No tasks yet"}</div>`;
            dockBody.appendChild(empty);
        }
        tasks.forEach((task, mapId) => {
            const pill = document.createElement("div");
            pill.className = "task-pill";
            const label = task.ticker || "Task";
            const normalizedStatus = String(task.status || "PENDING").toUpperCase();
            const statusClass = statusClassFor(normalizedStatus);
            const statusText = labelForStatus(normalizedStatus);
            const rawId = task.taskId && task.taskId.trim() ? task.taskId : shouldHideId(mapId) ? "" : mapId;
            const shortId = rawId ? shortenId(rawId) : "";
            const progressValue =
                typeof task.progress === "number" && !Number.isNaN(task.progress)
                    ? Math.max(0, Math.min(100, task.progress))
                    : null;
            const stageLabel = labelForStage(task.stage || (task.meta && task.meta.stage));
            const progressNote = stageLabel
                ? `${stageLabel}${progressValue !== null ? ` · ${Math.round(progressValue)}%` : ""}`
                : progressValue !== null
                  ? `${Math.round(progressValue)}%`
                  : "";
            pill.innerHTML = `
                <h6>
                    ${label}
                    ${shortId ? `<span class="task-id" title="${rawId}">#${shortId}</span>` : ""}
                    <span class="status ${statusClass}">${statusText}</span>
                </h6>
                <div class="task-meta">
                    ${task.message ? `<span>${task.message}</span>` : ""}
                    ${task.created ? `<span>${task.created}</span>` : ""}
                    ${task.link ? `<a href="${task.link}" target="_blank" rel="noopener">${langIsZh ? "查看报告" : "View report"}</a>` : ""}
                </div>
                ${
                    progressValue !== null
                        ? `<div class="task-progress"><div class="task-progress-bar" style="width:${progressValue}%"></div></div>`
                        : ""
                }
                ${progressNote ? `<div class="task-progress-note">${progressNote}</div>` : ""}
            `;
            const actions = document.createElement("div");
            actions.className = "task-actions";
            if (rawId) {
                const copyBtn = document.createElement("button");
                copyBtn.type = "button";
                copyBtn.className = "task-copy";
                copyBtn.textContent = langIsZh ? "复制 ID" : "Copy ID";
                copyBtn.addEventListener("click", (event) => {
                    event.stopPropagation();
                    copyTaskId(rawId, copyBtn);
                });
                actions.appendChild(copyBtn);
            }
            if (rawId && cancelTemplate && !shouldHideId(rawId)) {
                const isTerminal = ["SUCCESS", "FAILURE", "REVOKED"].includes(normalizedStatus);
                if (!isTerminal) {
                    const cancelBtn = document.createElement("button");
                    cancelBtn.type = "button";
                    cancelBtn.className = "task-cancel";
                    cancelBtn.textContent = langIsZh ? "取消" : "Cancel";
                    cancelBtn.addEventListener("click", (event) => {
                        event.stopPropagation();
                        cancelBtn.disabled = true;
                        cancelBtn.textContent = langIsZh ? "取消中..." : "Cancelling...";
                        const url = buildCancelUrl(rawId);
                        if (!url) return;
                        fetch(url, {
                            method: "POST",
                            headers: {
                                "X-CSRFToken": getCsrfToken(),
                                "X-Requested-With": "XMLHttpRequest",
                            },
                        })
                            .then((res) => res.json().catch(() => ({})))
                            .then((data) => {
                                updateTask(mapId, {
                                    status: data.state || "REVOKED",
                                    message: langIsZh ? "已取消" : "Cancelled",
                                    taskId: rawId,
                                });
                            })
                            .catch(() => {
                                cancelBtn.disabled = false;
                                cancelBtn.textContent = langIsZh ? "取消" : "Cancel";
                            });
                    });
                    actions.appendChild(cancelBtn);
                }
            }
            const dismiss = document.createElement("button");
            dismiss.textContent = langIsZh ? "移除" : "Dismiss";
            dismiss.addEventListener("click", (event) => {
                event.stopPropagation();
                tasks.delete(mapId);
                const snapshot = flushSnapshot();
                saveStore(snapshot);
                broadcastSnapshot({ id: mapId, task: null, tasks: snapshot });
                renderTasks();
            });
            actions.appendChild(dismiss);
            pill.appendChild(actions);
            dockBody.appendChild(pill);
        });
        dockCount.textContent = tasks.size;
        dockBody.hidden = dockMinimized;
        dock.classList.toggle("minimized", dockMinimized);
    };

    const flushSnapshot = () => Array.from(tasks.entries()).map(([id, val]) => ({ id, ...val }));

    const broadcastSnapshot = (detail) => {
        window.dispatchEvent(
            new CustomEvent("taskdock:update", {
                detail,
            })
        );
    };

const updateTask = (taskId, data) => {
    if (!taskId) return;
    const existing = tasks.get(taskId) || {};
    const prevSize = tasks.size;
    const merged = { ...existing, ...data };
        if (data && data.taskId) {
            merged.taskId = data.taskId;
        } else if (existing.taskId) {
            merged.taskId = existing.taskId;
        }
        if (data && data.meta) {
            merged.meta = { ...(existing.meta || {}), ...(data.meta || {}) };
            if (typeof data.meta.progress === "number") {
                merged.progress = data.meta.progress;
            }
            if (data.meta.stage) {
                merged.stage = data.meta.stage;
            }
        }
    tasks.set(taskId, merged);
    if (merged.status === "SUCCESS" && existing.status !== "SUCCESS") {
        showToast(merged);
    }
    while (tasks.size > 5) {
        const oldestKey = tasks.keys().next().value;
        tasks.delete(oldestKey);
    }
        renderTasks();
        if (tasks.size > prevSize) {
            dockMinimized = false;
            renderTasks();
        }
        const snapshot = flushSnapshot();
        saveStore(snapshot);
        broadcastSnapshot({ id: taskId, task: tasks.get(taskId), tasks: snapshot });
    };

    const pollTask = (taskId) => {
        const url = buildStatusUrl(taskId);
        if (!url) return;
        fetch(url, { headers: { "X-Requested-With": "XMLHttpRequest" } })
            .then((res) => res.json().catch(() => ({})))
            .then((data) => {
                const state = data.state || "";
                const meta = data.meta || {};
                const progress =
                    typeof meta.progress === "number" && !Number.isNaN(meta.progress) ? meta.progress : null;
                const stage = meta.stage || meta.detail || "";
                if (state === "SUCCESS") {
                    const hid = data.result && data.result.history_id;
                    updateTask(taskId, {
                        status: "SUCCESS",
                        message: langIsZh ? "任务完成，可查看报告。" : "Task complete.",
                        link: hid ? `${historyBase}?history_id=${encodeURIComponent(hid)}` : null,
                        progress: 100,
                        taskId,
                    });
                    return;
                }
                if (state === "FAILURE" || state === "REVOKED") {
                    updateTask(taskId, {
                        status: "FAILURE",
                        message: data.error || (langIsZh ? "执行失败" : "Failed"),
                        progress,
                        taskId,
                    });
                    return;
                }
                updateTask(taskId, {
                    status: state || "PENDING",
                    message:
                        state === "STARTED" || state === "PROGRESS"
                            ? langIsZh
                                ? "运行中..."
                                : "Running..."
                            : langIsZh
                            ? "已排队..."
                            : "Queued...",
                    progress,
                    stage,
                    meta,
                    taskId,
                });
                window.setTimeout(() => pollTask(taskId), pollInterval);
            })
            .catch(() => {});
    };

    const syncFromStore = () => {
        const store = loadStore();
        tasks.clear();
        store.tasks.forEach((item) => {
            if (item && item.id) {
                tasks.set(item.id, item);
            }
        });
        renderTasks();
        broadcastSnapshot({ id: null, task: null, tasks: store.tasks });
        store.tasks.forEach((item) => {
            if (!item || !item.id) return;
            const status = String(item.status || "").toUpperCase();
            const isTerminal = ["SUCCESS", "FAILURE", "REVOKED"].includes(status);
            if (!isTerminal) {
                window.setTimeout(() => pollTask(item.id), 500);
            }
        });
    };

    const handleDrag = (event) => {
        if (!dragState) return;
        const dx = event.clientX - dragState.startX;
        const dy = event.clientY - dragState.startY;
        dock.style.left = `${dragState.startLeft + dx}px`;
        dock.style.top = `${dragState.startTop + dy}px`;
    };
    const stopDrag = () => {
        dragState = null;
        document.removeEventListener("mousemove", handleDrag);
        document.removeEventListener("mouseup", stopDrag);
    };

    if (dockToggle) {
        dockToggle.addEventListener("click", (event) => {
            event.stopPropagation();
            dockMinimized = !dockMinimized;
            renderTasks();
        });
    }
    if (dockClear) {
        dockClear.addEventListener("click", (event) => {
            event.stopPropagation();
            tasks.clear();
            renderTasks();
            const snapshot = [];
            saveStore(snapshot);
            broadcastSnapshot({ id: null, task: null, tasks: snapshot });
        });
    }
    if (dockHeader) {
        dockHeader.addEventListener("mousedown", (event) => {
            dragState = {
                startX: event.clientX,
                startY: event.clientY,
                startLeft: dock.offsetLeft,
                startTop: dock.offsetTop,
            };
            document.addEventListener("mousemove", handleDrag);
            document.addEventListener("mouseup", stopDrag);
        });
        dockHeader.addEventListener("click", () => {
            if (dragState) return;
            dockMinimized = !dockMinimized;
            renderTasks();
        });
    }

    window.addEventListener("storage", (event) => {
        if (event.key === STORAGE_KEY) {
            syncFromStore();
        }
    });

    renderTasks();
    syncFromStore();

    window.taskDock = {
        updateTask,
        pollTask,
        buildStatusUrl,
        renderTasks,
        removeTask: (taskId) => {
            if (!taskId || !tasks.has(taskId)) return;
            tasks.delete(taskId);
            const snapshot = flushSnapshot();
            saveStore(snapshot);
            renderTasks();
            broadcastSnapshot({ id: taskId, task: null, tasks: snapshot });
        },
        get statusTemplate() {
            return statusTemplate;
        },
        get historyBase() {
            return historyBase;
        },
    };

    window.dispatchEvent(new CustomEvent("taskdock:ready", { detail: window.taskDock }));
})();
