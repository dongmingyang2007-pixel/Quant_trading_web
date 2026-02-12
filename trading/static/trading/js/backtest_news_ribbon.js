document.addEventListener('DOMContentLoaded', () => {
    const ribbon = document.getElementById('news-ribbon');
    if (!ribbon) return;

    const viewport = ribbon.querySelector('.news-ribbon-viewport');
    const track = ribbon.querySelector('.news-ribbon-track');
    if (!viewport || !track) return;

    const sourceCards = Array.from(track.children).filter((node) => node.classList && node.classList.contains('news-card'));
    if (!sourceCards.length) return;

    const prefersReduce = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
    let reducedMotion = prefersReduce ? !!prefersReduce.matches : false;

    const copies = 3;
    const speedAttr = Number.parseFloat(String(ribbon.dataset.scrollSpeed || ''));
    const baseSpeed = Number.isFinite(speedAttr) && speedAttr > 0 ? speedAttr : (sourceCards.length >= 6 ? 110 : 92);
    let pxPerSecond = reducedMotion ? 0 : baseSpeed;

    let laneWidth = 0;
    let rafId = 0;
    let lastTs = 0;

    let hoverPause = false;
    let wheelPause = false;
    let dragPause = false;
    let hiddenPause = false;

    let wheelTimer = null;
    let resizeTimer = null;

    let pointerActive = false;
    let pointerId = null;
    let dragStartX = 0;
    let dragStartLeft = 0;
    let movedDuringDrag = false;

    const cloneBlueprint = sourceCards.map((card) => card.cloneNode(true));

    const setScrollInstant = (value) => {
        viewport.scrollLeft = value;
    };

    const getRelativeRatio = () => {
        if (!laneWidth) return 0;
        return (viewport.scrollLeft - laneWidth) / laneWidth;
    };

    const applyAccessibility = () => {
        const cards = Array.from(track.children).filter((node) => node.classList && node.classList.contains('news-card'));
        cards.forEach((card) => {
            const lane = Number.parseInt(card.dataset.lane || '0', 10);
            const interactive = lane === 1;
            card.setAttribute('aria-hidden', interactive ? 'false' : 'true');
            card.tabIndex = interactive ? 0 : -1;
        });
    };

    const rebuildTrack = (ratio = 0) => {
        const fragment = document.createDocumentFragment();
        for (let lane = 0; lane < copies; lane += 1) {
            cloneBlueprint.forEach((blueprint) => {
                const clone = blueprint.cloneNode(true);
                clone.dataset.lane = String(lane);
                fragment.appendChild(clone);
            });
        }
        track.innerHTML = '';
        track.appendChild(fragment);

        laneWidth = Math.max(1, track.scrollWidth / copies);
        const clampedRatio = Math.max(-0.45, Math.min(0.45, ratio));
        setScrollInstant(laneWidth + clampedRatio * laneWidth);
        applyAccessibility();
    };

    const normalizeScroll = () => {
        if (!laneWidth) return;
        const minBound = laneWidth * 0.5;
        const maxBound = laneWidth * 1.5;
        if (viewport.scrollLeft < minBound) {
            setScrollInstant(viewport.scrollLeft + laneWidth);
        } else if (viewport.scrollLeft > maxBound) {
            setScrollInstant(viewport.scrollLeft - laneWidth);
        }
    };

    const shouldAutoRun = () => !reducedMotion && !hoverPause && !wheelPause && !dragPause && !hiddenPause;

    const animate = (ts) => {
        if (!lastTs) lastTs = ts;
        const delta = ts - lastTs;
        lastTs = ts;
        if (shouldAutoRun() && laneWidth > 1) {
            setScrollInstant(viewport.scrollLeft + (delta / 1000) * pxPerSecond);
            normalizeScroll();
        }
        rafId = window.requestAnimationFrame(animate);
    };

    const scheduleWheelResume = (delay = 900) => {
        if (wheelTimer) window.clearTimeout(wheelTimer);
        wheelTimer = window.setTimeout(() => {
            wheelPause = false;
        }, delay);
    };

    const openCard = (card) => {
        if (!card) return;
        const url = card.getAttribute('data-url');
        if (url) window.open(url, '_blank', 'noopener');
    };

    rebuildTrack(0);
    rafId = window.requestAnimationFrame(animate);

    viewport.addEventListener('pointerdown', (event) => {
        pointerActive = true;
        pointerId = event.pointerId;
        dragStartX = event.clientX;
        dragStartLeft = viewport.scrollLeft;
        movedDuringDrag = false;
        dragPause = true;
        viewport.classList.add('is-dragging');
        viewport.setPointerCapture(pointerId);
    });

    viewport.addEventListener('pointermove', (event) => {
        if (!pointerActive) return;
        const delta = event.clientX - dragStartX;
        if (Math.abs(delta) > 6) movedDuringDrag = true;
        setScrollInstant(dragStartLeft - delta);
        normalizeScroll();
        if (event.cancelable) event.preventDefault();
    });

    const finishDrag = () => {
        if (!pointerActive) return;
        pointerActive = false;
        dragPause = false;
        viewport.classList.remove('is-dragging');
        if (pointerId !== null) {
            try {
                viewport.releasePointerCapture(pointerId);
            } catch (_error) {
                // ignore capture state errors
            }
        }
        pointerId = null;
    };

    viewport.addEventListener('pointerup', finishDrag);
    viewport.addEventListener('pointercancel', finishDrag);

    viewport.addEventListener(
        'wheel',
        (event) => {
            const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
            if (!delta) return;
            event.preventDefault();
            setScrollInstant(viewport.scrollLeft + delta);
            normalizeScroll();
            wheelPause = true;
            scheduleWheelResume(Math.abs(delta) > 60 ? 1600 : 900);
        },
        { passive: false },
    );

    viewport.addEventListener('mouseenter', () => {
        hoverPause = true;
    });

    viewport.addEventListener('mouseleave', () => {
        hoverPause = false;
    });

    track.addEventListener('mouseover', (event) => {
        const card = event.target.closest('.news-card');
        if (!card) return;
        if (event.relatedTarget && card.contains(event.relatedTarget)) return;
        card.classList.add('news-card-hover');
        hoverPause = true;
    });

    track.addEventListener('mouseout', (event) => {
        const card = event.target.closest('.news-card');
        if (!card) return;
        if (event.relatedTarget && card.contains(event.relatedTarget)) return;
        card.classList.remove('news-card-hover');
        hoverPause = false;
    });

    track.addEventListener('focusin', (event) => {
        const card = event.target.closest('.news-card');
        if (!card) return;
        card.classList.add('news-card-hover');
        hoverPause = true;
    });

    track.addEventListener('focusout', (event) => {
        const card = event.target.closest('.news-card');
        if (!card) return;
        card.classList.remove('news-card-hover');
        hoverPause = false;
    });

    track.addEventListener('click', (event) => {
        if (movedDuringDrag) return;
        const card = event.target.closest('.news-card[data-url]');
        openCard(card);
    });

    track.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const card = event.target.closest('.news-card[data-url]');
        if (!card) return;
        event.preventDefault();
        openCard(card);
    });

    window.addEventListener('resize', () => {
        if (resizeTimer) window.clearTimeout(resizeTimer);
        const ratio = getRelativeRatio();
        resizeTimer = window.setTimeout(() => {
            rebuildTrack(ratio);
        }, 180);
    });

    document.addEventListener('visibilitychange', () => {
        hiddenPause = !!document.hidden;
    });

    if (prefersReduce) {
        const handleMotion = (event) => {
            reducedMotion = !!event.matches;
            pxPerSecond = reducedMotion ? 0 : baseSpeed;
        };
        if (typeof prefersReduce.addEventListener === 'function') {
            prefersReduce.addEventListener('change', handleMotion);
        } else if (typeof prefersReduce.addListener === 'function') {
            prefersReduce.addListener(handleMotion);
        }
    }

    window.addEventListener('beforeunload', () => {
        if (rafId) window.cancelAnimationFrame(rafId);
        if (wheelTimer) window.clearTimeout(wheelTimer);
        if (resizeTimer) window.clearTimeout(resizeTimer);
    });
});
