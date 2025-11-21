(() => {
  const counters = document.querySelectorAll('.summary-value[data-countup]');
  if (!counters.length) return;

  const reduceMotion = Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  counters.forEach((node) => {
    const raw = (node.getAttribute('data-countup') || node.textContent || '').trim();
    if (!raw) return;
    const matcher = /^([^0-9+\-]*)([+\-]?[0-9][0-9,]*(?:\.[0-9]+)?)(.*)$/;
    const match = raw.match(matcher);
    if (!match) return;
    const prefix = match[1] || '';
    const numericToken = match[2];
    const suffix = match[3] || '';
    const target = parseFloat(numericToken.replace(/,/g, ''));
    if (!Number.isFinite(target)) return;
    const decimalPlaces = (numericToken.split('.')[1] || '').length;

    const formatValue = (value) => {
      const formattedNumber = Number(value).toLocaleString(undefined, {
        minimumFractionDigits: decimalPlaces,
        maximumFractionDigits: decimalPlaces,
      });
      return `${prefix}${formattedNumber}${suffix}`;
    };

    if (reduceMotion) {
      node.textContent = formatValue(target);
      return;
    }

    const clampedDuration = Math.min(2400, Math.max(900, Math.abs(target) * 8 + 900));
    const startValue = 0;
    const startTime = performance.now();

    const tick = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / clampedDuration);
      const eased = 1 - Math.pow(1 - progress, 3);
      const currentValue = startValue + (target - startValue) * eased;
      node.textContent = formatValue(currentValue);
      if (progress < 1) {
        requestAnimationFrame(tick);
      } else {
        node.textContent = formatValue(target);
      }
    };

    node.textContent = formatValue(startValue);
    requestAnimationFrame(tick);
  });
})();
