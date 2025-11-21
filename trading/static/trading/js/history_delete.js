document.addEventListener('DOMContentLoaded', () => {
  const langIsZh = (document.documentElement.lang || '').toLowerCase().startsWith('zh');
  const TEXT = {
    minSelect: langIsZh ? '请至少选择两条记录再进行对比。' : 'Select at least two runs before comparing.',
    maxSelect: langIsZh ? '最多选择三条记录进行对比。' : 'Select up to three runs for comparison.',
  };

  function getCSRFToken() {
    const csrfInput = document.querySelector('input[name=csrfmiddlewaretoken]');
    if (csrfInput) return csrfInput.value;
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? decodeURIComponent(match[1]) : '';
  }

  const csrfToken = getCSRFToken();
  const compareState = new Map();
  const MAX_COMPARE = 3;

  function findCompareBar(accordion) {
    if (!accordion || !accordion.id) return null;
    return document.querySelector(`[data-role="history-compare"][data-accordion-id="${accordion.id}"]`);
  }

  function updateCompareBar(bar) {
    if (!bar) return;
    const selection = compareState.get(bar) || [];
    const countNode = bar.querySelector('[data-role="compare-count"]');
    const launchBtn = bar.querySelector('[data-role="compare-launch"]');
    if (countNode) {
      countNode.textContent = `${selection.length}/${MAX_COMPARE}`;
    }
    if (launchBtn) {
      launchBtn.disabled = selection.length < 2;
    }
  }

  function toggleCompareSelection(checkbox) {
    const accordion = checkbox.closest('.history-accordion');
    const bar = findCompareBar(accordion);
    if (!bar) return;
    if (!compareState.has(bar)) {
      compareState.set(bar, []);
    }
    const selection = compareState.get(bar);
    const recordId = checkbox.value;
    if (!recordId) return;
    if (checkbox.checked && !selection.includes(recordId)) {
      if (selection.length >= MAX_COMPARE) {
        checkbox.checked = false;
        alert(TEXT.maxSelect);
        return;
      }
      selection.push(recordId);
    } else if (!checkbox.checked) {
      const idx = selection.indexOf(recordId);
      if (idx >= 0) selection.splice(idx, 1);
    }
    updateCompareBar(bar);
  }

  function purgeSelection(recordId) {
    compareState.forEach((selection, bar) => {
      const idx = selection.indexOf(recordId);
      if (idx >= 0) {
        selection.splice(idx, 1);
        updateCompareBar(bar);
      }
    });
  }

  document.body.addEventListener('click', async (event) => {
    const deleteBtn = event.target.closest('.history-delete');
    if (deleteBtn) {
      event.preventDefault();
      const id = deleteBtn.dataset.id;
      const container = deleteBtn.closest('.history-accordion');
      const urlBase = container ? container.dataset.deleteUrlBase : null;
      const urlTemplate = container ? container.dataset.deleteUrlTemplate : null;
      const confirmMessage = container?.dataset.confirmDelete || '确定删除该历史回测记录吗？此操作不可撤销。';
      const deleteFailedMessage = container?.dataset.deleteFailed || '删除失败，请稍后重试。';
      const networkErrorMessage = container?.dataset.deleteNetwork || '删除时发生网络错误，请稍后再试。';

      if (!id) return;
      let targetUrl = null;
      if (urlTemplate && urlTemplate.includes('placeholder')) {
        targetUrl = urlTemplate.replace('placeholder', id);
      } else if (urlBase) {
        const normalized = urlBase.replace(/\/+$/, '');
        targetUrl = `${normalized}/${id}/`;
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
          let errorDetail = deleteFailedMessage;
          try {
            const errorData = await response.json();
            if (errorData && errorData.detail) {
              errorDetail = errorData.detail;
            }
          } catch (err) {
            // ignore parsing errors
          }
          alert(errorDetail);
          return;
        }

        const item = deleteBtn.closest('.history-item');
        if (item) {
          purgeSelection(id);
          item.remove();
        }

        if (container && !container.querySelector('.history-item')) {
          const emptyId = container.dataset.emptyId;
          const emptyEl = emptyId ? document.getElementById(emptyId) : null;
          if (emptyEl) emptyEl.classList.remove('d-none');
          container.classList.add('d-none');
        }
      } catch (error) {
        alert(`${networkErrorMessage} ${error.message || ''}`);
      }
      return;
    }

    const launchBtn = event.target.closest('[data-role="compare-launch"]');
    if (launchBtn) {
      const bar = launchBtn.closest('[data-role="history-compare"]');
      if (!bar) return;
      const selection = compareState.get(bar) || [];
      if (selection.length < 2) {
        alert(TEXT.minSelect);
        return;
      }
      const baseUrl = bar.dataset.compareUrl;
      if (!baseUrl) return;
      const url = new URL(baseUrl, window.location.origin);
      url.searchParams.set('records', selection.join(','));
      window.location.href = url.toString();
    }
  });

  document.body.addEventListener('change', (event) => {
    const checkbox = event.target.closest('[data-role="history-compare-toggle"]');
    if (checkbox && checkbox.type === 'checkbox') {
      toggleCompareSelection(checkbox);
    }
  });
});
