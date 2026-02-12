(function () {
  'use strict';

  function render(root, payload) {
    if (!root || !payload || !payload.review) return;

    const executionList = root.querySelector('[data-review-role="execution-list"]');
    const riskList = root.querySelector('[data-review-role="risk-list"]');
    const aiList = root.querySelector('[data-review-role="ai-list"]');

    function fillList(node, items, emptyText) {
      if (!node) return;
      node.innerHTML = '';
      if (!Array.isArray(items) || !items.length) {
        const li = document.createElement('li');
        li.textContent = emptyText;
        node.appendChild(li);
        return;
      }
      items.forEach(function (item) {
        const li = document.createElement('li');
        if (item && typeof item === 'object') {
          const label = item.label || item.title || '-';
          const value = item.value || item.body || '-';
          li.textContent = label + ': ' + value;
        } else {
          li.textContent = String(item);
        }
        node.appendChild(li);
      });
    }

    fillList(executionList, payload.review.execution_diagnostics || [], 'No execution diagnostics.');
    fillList(riskList, payload.review.risk_diagnostics || [], 'No risk diagnostics.');
    fillList(aiList, payload.review.ai_briefs || [], 'No AI briefs.');
  }

  window.BacktestWorkspaceReview = {
    render: render,
  };
})();
