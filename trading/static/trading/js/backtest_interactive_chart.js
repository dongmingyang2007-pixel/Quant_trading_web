window.addEventListener('load', () => {
    const chartDataNode = document.getElementById('interactive-chart-data');
    if (!chartDataNode) return;
    let chartPayload = null;
    try {
        chartPayload = JSON.parse(chartDataNode.textContent || '{}');
    } catch (_error) {
        chartPayload = null;
    }
    if (!chartPayload || !Array.isArray(chartPayload.candles) || !chartPayload.candles.length) {
        return;
    }
    if (typeof LightweightCharts === 'undefined') {
        window.setTimeout(() => {
            if (typeof LightweightCharts === 'undefined') {
                console.warn('LightweightCharts is unavailable.');
            }
        }, 0);
        return;
    }
    const root = document.getElementById('interactive-chart');
    if (!root) return;

    const chart = LightweightCharts.createChart(root, {
        layout: { background: { type: 'solid', color: '#f8fafc' }, textColor: '#0f172a' },
        rightPriceScale: { borderColor: '#e2e8f0' },
        timeScale: { borderColor: '#e2e8f0', timeVisible: true, secondsVisible: false },
        grid: {
            horzLines: { color: '#f1f5f9' },
            vertLines: { color: '#f1f5f9' },
        },
        localization: {
            locale: document.documentElement.lang || navigator.language,
            dateFormat: 'yyyy-MM-dd',
        },
    });

    const resizeChart = () => {
        const rect = root.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
            chart.resize(rect.width, rect.height);
            chart.timeScale().fitContent();
            return true;
        }
        return false;
    };
    const ensureVisible = () => {
        if (!resizeChart()) {
            window.setTimeout(resizeChart, 220);
        }
    };
    resizeChart();
    window.setTimeout(resizeChart, 250);
    window.addEventListener('resize', () => resizeChart());

    const chartsPanel = document.getElementById('charts-pane');
    if (chartsPanel) {
        if (chartsPanel.classList.contains('active')) {
            ensureVisible();
        }
        if (typeof MutationObserver !== 'undefined') {
            const observer = new MutationObserver(() => {
                if (chartsPanel.classList.contains('active')) {
                    ensureVisible();
                }
            });
            observer.observe(chartsPanel, { attributes: true, attributeFilter: ['class'] });
        }
    }

    const candleSeries = chart.addCandlestickSeries({
        upColor: '#16a34a',
        downColor: '#dc2626',
        borderVisible: false,
        wickUpColor: '#16a34a',
        wickDownColor: '#dc2626',
    });
    candleSeries.setData(chartPayload.candles);

    if (Array.isArray(chartPayload.sma_short) && chartPayload.sma_short.length) {
        const shortLine = chart.addLineSeries({
            color: '#0ea5e9',
            lineWidth: 2,
            title: `SMA(${chartPayload.short_window || 'S'})`,
        });
        shortLine.setData(chartPayload.sma_short);
    }
    if (Array.isArray(chartPayload.sma_long) && chartPayload.sma_long.length) {
        const longLine = chart.addLineSeries({
            color: '#f97316',
            lineWidth: 2,
            title: `SMA(${chartPayload.long_window || 'L'})`,
        });
        longLine.setData(chartPayload.sma_long);
    }

    const docLang = (document.documentElement.getAttribute('lang') || navigator.language || '').toLowerCase();
    const chartLangIsZh = docLang.startsWith('zh');
    const infoPlaceholder = chartLangIsZh
        ? '点击图中的买入或卖出图标查看这笔交易详情。'
        : 'Click any Buy (B) or Sell (S) marker to view trade rationale.';

    const legendBox = document.getElementById('chart-legend');
    const infoBox = document.getElementById('chart-trade-info');
    const tooltip = document.createElement('div');
    tooltip.className = 'chart-tooltip';
    tooltip.setAttribute('role', 'status');
    root.appendChild(tooltip);
    const hideTooltip = () => tooltip.classList.remove('is-visible');
    const showTooltip = (signalsAtPoint, coords) => {
        if (!Array.isArray(signalsAtPoint) || !signalsAtPoint.length || !coords) {
            hideTooltip();
            return;
        }
        tooltip.innerHTML = signalsAtPoint
            .map((sig) => {
                const descriptor = describeSignal(sig);
                const detailsList = descriptor.details.length
                    ? `<ul>${descriptor.details.map((item) => `<li>${item}</li>`).join('')}</ul>`
                    : '';
                return `<div class="tooltip-entry"><h4>${descriptor.heading}</h4><div>${descriptor.summary}</div>${detailsList}</div>`;
            })
            .join('');
        const bounds = root.getBoundingClientRect();
        const limitX = Math.min(Math.max(coords.x, 24), bounds.width - 24);
        const limitY = Math.min(Math.max(coords.y, 24), bounds.height - 24);
        tooltip.style.left = `${limitX}px`;
        tooltip.style.top = `${limitY}px`;
        tooltip.classList.add('is-visible');
    };

    const signals = Array.isArray(chartPayload.signals) ? chartPayload.signals : [];
    const signalsByTime = new Map();
    signals.forEach((signal) => {
        const key = signal.time;
        if (!signalsByTime.has(key)) signalsByTime.set(key, []);
        signalsByTime.get(key).push(signal);
    });
    if (signals.length) {
        // 聚合同一天多笔成交，避免多个图标挤在一起导致文字被遮挡
        const markerBuckets = new Map();
        signals.forEach((signal) => {
            const bucket = markerBuckets.get(signal.time) || { time: signal.time, buys: [], sells: [] };
            if (signal.type === 'buy') bucket.buys.push(signal);
            else bucket.sells.push(signal);
            markerBuckets.set(signal.time, bucket);
        });
        const markers = [];
        markerBuckets.forEach((bucket) => {
            // 仅保留卖出标记，买入箭头已取消显示，鼠标点击仍可查看对应交易详情
            if (bucket.sells.length) {
                markers.push({
                    time: bucket.time,
                    position: 'aboveBar',
                    shape: 'arrowDown',
                    color: '#dc2626',
                    size: bucket.sells.length > 1 ? 3 : 2,
                });
            }
        });
        markers.sort((a, b) => (a.time > b.time ? 1 : a.time < b.time ? -1 : 0));
        candleSeries.setMarkers(markers);
        displayTradeInfo([signals[signals.length - 1]]);
    } else {
        displayTradeInfo(null);
    }

    const highlightRow = (timeStr) => {
        document.querySelectorAll('[data-trade-time]').forEach((row) => {
            row.classList.toggle('is-hovered', !!timeStr && row.dataset.tradeTime === timeStr);
        });
    };

    const updateLegend = (timeStr, price) => {
        if (!legendBox) return;
        if (!timeStr || !price) {
            legendBox.textContent = '';
            return;
        }
        legendBox.innerHTML = `
            <div>${timeStr}</div>
            <div>O ${Number(price.open || 0).toFixed(2)} H ${Number(price.high || 0).toFixed(2)}</div>
            <div>L ${Number(price.low || 0).toFixed(2)} C ${Number(price.close || 0).toFixed(2)}</div>
        `;
    };

    function describeSignal(signal) {
        const ctx = signal.context || {};
        const action = signal.type === 'buy' ? (chartLangIsZh ? '买入' : 'Buy') : chartLangIsZh ? '卖出' : 'Sell';
        const summaryParts = [];
        if (signal.price !== undefined && signal.price !== null) {
            summaryParts.push(`${chartLangIsZh ? '价格' : 'Price'} ${Number(signal.price).toFixed(2)}`);
        }
        if (signal.daily_return !== undefined && signal.daily_return !== null) {
            summaryParts.push(`${chartLangIsZh ? '当日收益' : 'Daily'} ${(signal.daily_return * 100).toFixed(2)}%`);
        } else if (signal.cum_return !== undefined && signal.cum_return !== null) {
            const cumulativePct = (signal.cum_return - 1) * 100;
            summaryParts.push(`${chartLangIsZh ? '累计' : 'Cumulative'} ${cumulativePct.toFixed(2)}%`);
        }
        const details = [];
        if (ctx.rsi !== undefined) {
            let state = '';
            if (ctx.rsi <= 30) state = chartLangIsZh ? '（超卖）' : ' (oversold)';
            else if (ctx.rsi >= 70) state = chartLangIsZh ? '（超买）' : ' (overbought)';
            details.push(`RSI ${ctx.rsi}${state}`);
        }
        if (ctx.probability !== undefined) {
            details.push(`${chartLangIsZh ? '模型概率' : 'Model confidence'} ${(ctx.probability * 100).toFixed(1)}%`);
        }
        if (ctx.leverage !== undefined) {
            details.push(`${chartLangIsZh ? '杠杆' : 'Leverage'} ×${ctx.leverage}`);
        }
        if (ctx.strategy_return !== undefined && ctx.strategy_return !== null) {
            details.push(`${chartLangIsZh ? '策略日收益' : 'Strategy'} ${(ctx.strategy_return * 100).toFixed(2)}%`);
        }
        return {
            heading: `${action} · ${signal.time}`,
            summary: summaryParts.join(' • '),
            details,
        };
    }

    function displayTradeInfo(signalList) {
        if (!infoBox) return;
        if (!signalList || !signalList.length) {
            infoBox.textContent = infoPlaceholder;
            hideTooltip();
            return;
        }
        infoBox.innerHTML = signalList
            .map((sig) => {
                const descriptor = describeSignal(sig);
                const detailText = descriptor.details.length
                    ? descriptor.details.map((item) => `<span class="chart-trade-detail">${item}</span>`).join('')
                    : '';
                return `<div><span>${descriptor.heading}</span> ${detailText}</div>`;
            })
            .join('');
    }

    const normalizeTime = (input) => {
        if (!input) return null;
        if (typeof input === 'string') return input;
        if (typeof input === 'object') {
            const year = input.year ?? input.yearValue;
            const month = input.month ?? input.monthValue;
            const day = input.day ?? input.dayValue;
            if (year && month && day) {
                return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
            }
        }
        return null;
    };
    chart.subscribeCrosshairMove((param) => {
        if (!param || !param.time) {
            highlightRow(null);
            updateLegend(null, null);
            return;
        }
        const timeStr = normalizeTime(param.time);
        if (timeStr) highlightRow(timeStr);
        const price = param.seriesPrices?.get?.(candleSeries);
        updateLegend(timeStr, price);
    });

    chart.subscribeClick((param) => {
        if (!param || !param.time) {
            hideTooltip();
            return;
        }
        const timeStr = normalizeTime(param.time);
        if (!timeStr) {
            hideTooltip();
            return;
        }
        const matchedSignals = signalsByTime.get(timeStr);
        if (matchedSignals && matchedSignals.length) {
            displayTradeInfo(matchedSignals);
            highlightRow(timeStr);
            if (param.point) {
                showTooltip(matchedSignals, param.point);
            }
        } else {
            hideTooltip();
        }
    });

    chart.timeScale().fitContent();
});
