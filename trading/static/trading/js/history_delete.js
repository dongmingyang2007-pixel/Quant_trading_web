document.addEventListener('DOMContentLoaded', () => {
  const langIsZh = (document.documentElement.lang || '').toLowerCase().startsWith('zh');
  const TEXT = {
    minSelect: langIsZh ? '请至少选择两条记录再进行对比。' : 'Select at least two runs before comparing.',
    maxSelect: langIsZh ? '最多选择五条记录进行对比。' : 'Select up to five runs for comparison.',
    hintEmpty: langIsZh ? '请选择 2-5 条历史记录以开始对比。' : 'Select 2-5 runs to start comparison.',
    hintNeedMore: langIsZh ? '再选择至少一条记录即可对比。' : 'Select at least one more run.',
    hintReady: langIsZh ? '已就绪，点击右侧按钮进入对比。' : 'Ready. Click the button to compare.',
    hintFull: langIsZh ? '已达上限，可删除勾选项后重选。' : 'Maximum reached. Remove one to adjust.',
  };

  const getCSRFToken = () => {
    const csrfInput = document.querySelector('input[name=csrfmiddlewaretoken]');
    if (csrfInput) return csrfInput.value;
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? decodeURIComponent(match[1]) : '';
  };

  const csrfToken = getCSRFToken();
  const compareState = new Map();
  const MAX_COMPARE = 5;

  const findBar = (accordionId) =>
    document.querySelector(`[data-role="history-compare"][data-accordion-id="${accordionId}"]`);

  const updateBar = (accordionId) => {
    const bar = findBar(accordionId);
    if (!bar) return;
    const selections = compareState.get(accordionId) || [];
    const countNode = bar.querySelector('[data-role="compare-count"]');
    const noteNode = bar.querySelector('[data-role="compare-note"]');
    const launchBtn = bar.querySelector('[data-role="compare-launch"]');
    if (countNode) countNode.textContent = `${selections.length}`;
    if (noteNode) {
      if (selections.length >= MAX_COMPARE) noteNode.textContent = TEXT.hintFull;
      else if (selections.length >= 2) noteNode.textContent = TEXT.hintReady;
      else if (selections.length === 1) noteNode.textContent = TEXT.hintNeedMore;
      else noteNode.textContent = TEXT.hintEmpty;
    }
    if (launchBtn) launchBtn.disabled = selections.length < 2;
  };

  const syncSelections = (accordionId) => {
    if (!accordionId) return;
    const container = document.getElementById(accordionId);
    const selections = [];
    if (container) {
      container.querySelectorAll('[data-role="history-compare-toggle"]').forEach((checkbox) => {
        if (checkbox.checked && checkbox.value && !selections.includes(checkbox.value)) {
          selections.push(checkbox.value);
        }
      });
    }
    compareState.set(accordionId, selections);
    updateBar(accordionId);
    return selections;
  };

  const enforceLimit = (accordionId, checkbox) => {
    const selections = compareState.get(accordionId) || [];
    if (selections.length > MAX_COMPARE && checkbox.checked) {
      checkbox.checked = false;
      syncSelections(accordionId);
      alert(TEXT.maxSelect);
      return false;
    }
    return true;
  };

  document.body.addEventListener('change', (event) => {
    const checkbox = event.target.closest('[data-role="history-compare-toggle"]');
    if (!checkbox) return;
    const container = checkbox.closest('.history-accordion');
    const accordionId = checkbox.dataset.compareId || container?.id;
    if (!accordionId) return;
    syncSelections(accordionId);
    enforceLimit(accordionId, checkbox);
  });

  document.body.addEventListener('click', async (event) => {
    const deleteBtn = event.target.closest('.history-delete');
    if (deleteBtn) {
      event.preventDefault();
      const recordId = deleteBtn.dataset.id;
      const container = deleteBtn.closest('.history-accordion');
      const accordionId = container?.id;
      const urlBase = container?.dataset.deleteUrlBase || null;
      const urlTemplate = container?.dataset.deleteUrlTemplate || null;
      const confirmMessage =
        container?.dataset.confirmDelete || '确定删除该历史回测记录吗？此操作不可撤销。';
      const deleteFailedMessage =
        container?.dataset.deleteFailed || '删除失败，请稍后重试。';
      const networkErrorMessage =
        container?.dataset.deleteNetwork || '删除时发生网络错误，请稍后再试。';

      if (!recordId) return;
      let targetUrl = null;
      if (urlTemplate && urlTemplate.includes('placeholder')) {
        targetUrl = urlTemplate.replace('placeholder', recordId);
      } else if (urlBase) {
        targetUrl = `${urlBase.replace(/\/+$/, '')}/${recordId}/`;
      }
      if (!targetUrl) return;
      if (!window.confirm(confirmMessage)) return;

      try {
        const response = await fetch(targetUrl, {
          method: 'POST',
          headers: {
            'X-CSRFToken': csrfToken,
            'Content-Type': 'application/json',
          },
          credentials: 'same-origin',
        });
        if (!response.ok) {
          throw new Error(deleteFailedMessage);
        }
        const item = deleteBtn.closest('.history-item');
        if (item) item.remove();
        if (container && !container.querySelector('.history-item')) {
          const emptyId = container.dataset.emptyId;
          const emptyEl = emptyId ? document.getElementById(emptyId) : null;
          if (emptyEl) emptyEl.classList.remove('d-none');
          container.classList.add('d-none');
        }
        if (accordionId) syncSelections(accordionId);
      } catch (error) {
        alert(`${networkErrorMessage} ${error.message || ''}`);
      }
      return;
    }

    const launchBtn = event.target.closest('[data-role="compare-launch"]');
    if (launchBtn) {
      const bar = launchBtn.closest('[data-role="history-compare"]');
      if (!bar) return;
      const accordionId = bar.dataset.accordionId;
      const selections = compareState.get(accordionId) || [];
      if (selections.length < 2) {
        alert(TEXT.minSelect);
        return;
      }
      const baseUrl = bar.dataset.compareUrl;
      if (!baseUrl) return;
      const url = new URL(baseUrl, window.location.origin);
      url.searchParams.set('records', selections.join(','));
      window.location.href = url.toString();
    }
  });

  document.querySelectorAll('[data-role="history-compare"]').forEach((bar) => {
    const accordionId = bar.dataset.accordionId;
    if (!accordionId) return;
    syncSelections(accordionId);
  });
});
