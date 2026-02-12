(() => {
  const init = () => {
    const settingsCard = document.querySelector("[data-role='settings-card']");
    if (!settingsCard) {
      return;
    }

    const toggleButton = settingsCard.querySelector("[data-role='toggle-settings']");
    const body = settingsCard.querySelector("[data-role='settings-body']");
    const openClass = 'd-none';

    if (!body) {
      return;
    }

    const isInitiallyOpen = settingsCard.dataset.settingsOpen === '1';
    if (isInitiallyOpen) {
      body.classList.remove(openClass);
    }

    if (!toggleButton) {
      return;
    }

    const labelOpen = toggleButton.dataset.labelOpen || 'Collapse';
    const labelClosed = toggleButton.dataset.labelClosed || 'Edit';

    const syncLabel = () => {
      const isHidden = body.classList.contains(openClass);
      toggleButton.textContent = isHidden ? labelClosed : labelOpen;
    };

    syncLabel();
    toggleButton.addEventListener('click', () => {
      const willOpen = body.classList.contains(openClass);
      body.classList.toggle(openClass, !willOpen);
      syncLabel();
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
