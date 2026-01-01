(() => {
  const form = document.getElementById("analysis-form");
  if (!form) return;
  const panel = form.querySelector('[data-role="preset-panel"]');
  if (!panel) return;

  const apiEndpoint = panel.dataset.apiEndpoint || "";
  const detailTemplate = panel.dataset.detailTemplate || "";
  const selectEl = panel.querySelector('[data-role="preset-select"]');
  const nameInput = panel.querySelector('[data-role="preset-name"]');
  const descInput = panel.querySelector('[data-role="preset-description"]');
  const statusEl = panel.querySelector('[data-role="preset-status"]');
  const defaultCheckbox = panel.querySelector('[data-role="preset-default"]');
  const includeDatesCheckbox = panel.querySelector('[data-role="preset-include-dates"]');
  const listEl = panel.querySelector('[data-role="preset-list"]');
  const managePanel = panel.querySelector('[data-role="preset-manage"]');

  const langIsZh = (form.dataset.lang || document.documentElement.lang || "").toLowerCase().startsWith("zh");
  const TEXT = langIsZh
    ? {
        selectPlaceholder: "请选择预设",
        emptyList: "暂无预设。",
        apply: "应用预设",
        save: "保存预设",
        manage: "管理预设",
        setDefault: "设为默认",
        defaulted: "默认",
        delete: "删除",
        applySuccess: (name) => `已应用预设：${name}`,
        saveSuccess: (name) => `已保存预设：${name}`,
        deleteSuccess: (name) => `已删除预设：${name}`,
        defaultSuccess: (name) => `已设为默认：${name}`,
        needName: "请填写预设名称。",
        needSelect: "请先选择一个预设。",
        loadFailed: "加载预设失败，请稍后重试。",
        saveFailed: "保存失败，请检查输入后重试。",
        applyFailed: "应用失败，请稍后重试。",
        updateFailed: "更新失败，请稍后重试。",
        deleteFailed: "删除失败，请稍后重试。",
      }
    : {
        selectPlaceholder: "Choose a preset",
        emptyList: "No presets yet.",
        apply: "Apply preset",
        save: "Save preset",
        manage: "Manage presets",
        setDefault: "Set default",
        defaulted: "Default",
        delete: "Delete",
        applySuccess: (name) => `Preset applied: ${name}`,
        saveSuccess: (name) => `Preset saved: ${name}`,
        deleteSuccess: (name) => `Preset deleted: ${name}`,
        defaultSuccess: (name) => `Default set: ${name}`,
        needName: "Please enter a preset name.",
        needSelect: "Select a preset first.",
        loadFailed: "Failed to load presets. Please try again later.",
        saveFailed: "Save failed. Check the inputs and retry.",
        applyFailed: "Apply failed. Please retry.",
        updateFailed: "Update failed. Please retry.",
        deleteFailed: "Delete failed. Please retry.",
      };

  let presets = [];

  const getCsrfToken = () => {
    const input = form.querySelector('input[name="csrfmiddlewaretoken"]');
    if (input && input.value) return input.value;
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? decodeURIComponent(match[1]) : "";
  };

  const showStatus = (message, variant = "info") => {
    if (!statusEl) return;
    statusEl.classList.remove("d-none", "alert-info", "alert-danger", "alert-success");
    statusEl.classList.add(`alert-${variant}`);
    statusEl.textContent = message;
  };

  const clearStatus = () => {
    if (!statusEl) return;
    statusEl.classList.add("d-none");
    statusEl.textContent = "";
  };

  const buildPayloadFromForm = () => {
    const payload = {};
    const data = new FormData(form);
    data.forEach((value, key) => {
      if (key === "csrfmiddlewaretoken") return;
      payload[key] = value;
    });
    form.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
      if (!checkbox.name) return;
      payload[checkbox.name] = checkbox.checked;
    });
    return payload;
  };

  const applyPayloadToForm = (payload) => {
    if (!payload || typeof payload !== "object") return;
    const includeTickerDates = Boolean(payload.include_ticker_dates);
    Object.entries(payload).forEach(([key, value]) => {
      if (key === "include_ticker_dates") return;
      if (!includeTickerDates && ["ticker", "start_date", "end_date"].includes(key)) {
        return;
      }
      const field = form.querySelector(`[name="${key}"]`);
      if (!field) return;
      if (field.type === "checkbox") {
        field.checked = Boolean(value);
      } else {
        field.value = value ?? "";
      }
      field.dispatchEvent(new Event("change", { bubbles: true }));
    });
  };

  const buildDetailUrl = (presetId) => {
    if (!detailTemplate) return "";
    const encoded = encodeURIComponent(presetId);
    if (detailTemplate.includes("PRESET_ID")) {
      return detailTemplate.replace("PRESET_ID", encoded);
    }
    const uuidPlaceholder = "00000000-0000-0000-0000-000000000000";
    if (detailTemplate.includes(uuidPlaceholder)) {
      return detailTemplate.replace(uuidPlaceholder, encoded);
    }
    return detailTemplate.replace(/\/?$/, "/") + `${encoded}/`;
  };

  const renderSelect = () => {
    if (!selectEl) return;
    selectEl.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = TEXT.selectPlaceholder;
    selectEl.appendChild(placeholder);
    presets.forEach((preset) => {
      const option = document.createElement("option");
      option.value = preset.preset_id;
      option.textContent = preset.name;
      if (preset.is_default) {
        option.textContent = `${preset.name} · ${TEXT.defaulted}`;
      }
      selectEl.appendChild(option);
    });
  };

  const renderList = () => {
    if (!listEl) return;
    listEl.innerHTML = "";
    if (!presets.length) {
      const empty = document.createElement("div");
      empty.className = "list-group-item text-muted";
      empty.textContent = TEXT.emptyList;
      listEl.appendChild(empty);
      return;
    }
    presets.forEach((preset) => {
      const item = document.createElement("div");
      item.className = "list-group-item d-flex align-items-start justify-content-between flex-wrap gap-3";
      item.dataset.presetId = preset.preset_id;

      const payload = preset.payload || {};
      const includeTickerDates = Boolean(payload.include_ticker_dates);
      const ticker = includeTickerDates && payload.ticker ? String(payload.ticker).toUpperCase() : "";
      const period =
        includeTickerDates && payload.start_date && payload.end_date ? `${payload.start_date} → ${payload.end_date}` : "";
      const metaLine = [ticker, period].filter(Boolean).join(" · ");

      item.innerHTML = `
        <div>
          <div class="fw-semibold">
            ${preset.name}
            ${preset.is_default ? `<span class="badge text-bg-primary ms-1">${TEXT.defaulted}</span>` : ""}
          </div>
          ${preset.description ? `<div class="text-muted small">${preset.description}</div>` : ""}
          ${metaLine ? `<div class="text-muted small">${metaLine}</div>` : ""}
        </div>
        <div class="btn-group btn-group-sm">
          <button type="button" class="btn btn-outline-primary" data-action="apply">${TEXT.apply}</button>
          <button type="button" class="btn btn-outline-secondary" data-action="default">${TEXT.setDefault}</button>
          <button type="button" class="btn btn-outline-danger" data-action="delete">${TEXT.delete}</button>
        </div>
      `;
      listEl.appendChild(item);
    });
  };

  const refreshPresets = async () => {
    if (!apiEndpoint) return;
    try {
      const response = await fetch(apiEndpoint, { headers: { "X-Requested-With": "XMLHttpRequest" } });
      if (!response.ok) throw new Error(TEXT.loadFailed);
      const payload = await response.json();
      presets = Array.isArray(payload.presets) ? payload.presets : [];
      renderSelect();
      renderList();
    } catch (error) {
      showStatus(TEXT.loadFailed, "danger");
    }
  };

  const findPresetById = (presetId) => presets.find((preset) => preset.preset_id === presetId);

  const applyPreset = (presetId) => {
    const preset = findPresetById(presetId);
    if (!preset) {
      showStatus(TEXT.needSelect, "danger");
      return;
    }
    applyPayloadToForm(preset.payload || {});
    showStatus(TEXT.applySuccess(preset.name), "success");
  };

  const savePreset = async () => {
    clearStatus();
    const name = (nameInput && nameInput.value.trim()) || "";
    if (!name) {
      showStatus(TEXT.needName, "danger");
      return;
    }
    const payload = buildPayloadFromForm();
    const includeTickerDates = includeDatesCheckbox ? includeDatesCheckbox.checked : false;
    if (!includeTickerDates) {
      ["ticker", "start_date", "end_date"].forEach((key) => delete payload[key]);
    }
    if (includeTickerDates) {
      payload.include_ticker_dates = true;
    } else {
      delete payload.include_ticker_dates;
    }
    const body = {
      name,
      description: (descInput && descInput.value.trim()) || "",
      payload,
      is_default: defaultCheckbox ? defaultCheckbox.checked : false,
    };
    try {
      const response = await fetch(apiEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCsrfToken(),
          "X-Requested-With": "XMLHttpRequest",
        },
        credentials: "same-origin",
        body: JSON.stringify(body),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || TEXT.saveFailed);
      }
      await refreshPresets();
      if (selectEl) {
        selectEl.value = data.preset_id || "";
      }
      showStatus(TEXT.saveSuccess(data.name || name), "success");
    } catch (error) {
      showStatus(TEXT.saveFailed, "danger");
    }
  };

  const updatePreset = async (presetId, payload) => {
    const url = buildDetailUrl(presetId);
    if (!url) return;
    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCsrfToken(),
        "X-Requested-With": "XMLHttpRequest",
      },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || TEXT.updateFailed);
    }
    return data;
  };

  const deletePreset = async (presetId) => {
    const url = buildDetailUrl(presetId);
    if (!url) return;
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        "X-CSRFToken": getCsrfToken(),
        "X-Requested-With": "XMLHttpRequest",
      },
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(TEXT.deleteFailed);
    }
  };

  if (selectEl) {
    selectEl.addEventListener("change", () => {
      if (!nameInput || !descInput) return;
      const preset = findPresetById(selectEl.value);
      if (!preset) return;
      nameInput.value = preset.name || "";
      descInput.value = preset.description || "";
      if (defaultCheckbox) {
        defaultCheckbox.checked = Boolean(preset.is_default);
      }
      if (includeDatesCheckbox) {
        includeDatesCheckbox.checked = Boolean(preset.payload && preset.payload.include_ticker_dates);
      }
    });
  }

  panel.querySelector('[data-role="preset-apply"]')?.addEventListener("click", () => {
    clearStatus();
    const presetId = selectEl ? selectEl.value : "";
    if (!presetId) {
      showStatus(TEXT.needSelect, "danger");
      return;
    }
    try {
      applyPreset(presetId);
    } catch (error) {
      showStatus(TEXT.applyFailed, "danger");
    }
  });

  panel.querySelector('[data-role="preset-save"]')?.addEventListener("click", () => {
    savePreset();
  });

  panel.querySelector('[data-role="preset-toggle"]')?.addEventListener("click", () => {
    if (!managePanel) return;
    managePanel.classList.toggle("d-none");
  });

  panel.querySelector('[data-role="preset-hide"]')?.addEventListener("click", () => {
    if (!managePanel) return;
    managePanel.classList.add("d-none");
  });

  if (listEl) {
    listEl.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-action]");
      if (!button) return;
      const item = button.closest("[data-preset-id]");
      if (!item) return;
      const presetId = item.dataset.presetId;
      const preset = findPresetById(presetId);
      if (!preset) return;
      const action = button.dataset.action;
      try {
        if (action === "apply") {
          applyPreset(presetId);
          return;
        }
        if (action === "default") {
          const updated = await updatePreset(presetId, { is_default: true });
          await refreshPresets();
          showStatus(TEXT.defaultSuccess(updated.name || preset.name), "success");
          return;
        }
        if (action === "delete") {
          await deletePreset(presetId);
          await refreshPresets();
          showStatus(TEXT.deleteSuccess(preset.name), "success");
        }
      } catch (error) {
        showStatus(TEXT.updateFailed, "danger");
      }
    });
  }

  refreshPresets();
})();
