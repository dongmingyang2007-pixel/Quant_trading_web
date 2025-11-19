document.addEventListener('DOMContentLoaded', function () {
    function getCSRFToken() {
        const csrfInput = document.querySelector('input[name=csrfmiddlewaretoken]');
        if (csrfInput) {
            return csrfInput.value;
        }
        const match = document.cookie.match(/csrftoken=([^;]+)/i);
        return match ? decodeURIComponent(match[1]) : '';
    }

    const csrfToken = getCSRFToken();

    document.body.addEventListener('click', async function (event) {
        const deleteBtn = event.target.closest('.history-delete');
        if (!deleteBtn) return;

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
        if (!confirm(confirmMessage)) return;

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
            if (item) item.remove();

            if (container && !container.querySelector('.history-item')) {
                const emptyId = container.dataset.emptyId;
                const emptyEl = emptyId ? document.getElementById(emptyId) : null;
                if (emptyEl) emptyEl.classList.remove('d-none');
                container.classList.add('d-none');
            }
        } catch (error) {
            alert(`${networkErrorMessage} ${error.message || ''}`);
        }
    });
});
