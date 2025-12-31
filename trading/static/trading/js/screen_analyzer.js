(() => {
  const apiMeta = document.querySelector('meta[name="screen-analyzer-api"]');
  if (!apiMeta) return;

  const apiUrl = apiMeta.content;
  const intervalMeta = document.querySelector('meta[name="screen-analyzer-interval"]');
  const sampleMeta = document.querySelector('meta[name="screen-analyzer-sample-api"]');
  const trainMeta = document.querySelector('meta[name="screen-analyzer-train-api"]');
  const ocrMeta = document.querySelector('meta[name="screen-analyzer-ocr"]');
  const defaultInterval = intervalMeta ? Number(intervalMeta.content) || 1200 : 1200;
  const defaultOcr = ocrMeta ? ocrMeta.content !== '0' : true;
  const sampleUrl = sampleMeta ? sampleMeta.content : '';
  const trainUrl = trainMeta ? trainMeta.content : '';
  const csrfToken = (() => {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta && meta.content) return meta.content;
    const match = document.cookie.match(/csrftoken=([^;]+)/i);
    return match ? match[1] : '';
  })();

  const patternCopy = (() => {
    const raw = document.getElementById('pattern-copy');
    if (!raw) return {};
    try {
      return JSON.parse(raw.textContent || '{}');
    } catch (err) {
      return {};
    }
  })();

  const waveCopy = (() => {
    const raw = document.getElementById('wave-copy');
    if (!raw) return {};
    try {
      return JSON.parse(raw.textContent || '{}');
    } catch (err) {
      return {};
    }
  })();

  const labels = (() => {
    const raw = document.getElementById('screen-analyzer-labels');
    if (!raw) return {};
    try {
      return JSON.parse(raw.textContent || '{}');
    } catch (err) {
      return {};
    }
  })();

  const sessionId = (() => {
    const key = 'screenAnalyzerSessionId';
    try {
      const stored = window.localStorage.getItem(key);
      if (stored) return stored;
    } catch (err) {
      // ignore
    }
    const fallback = `session-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    const generated = window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() : fallback;
    try {
      window.localStorage.setItem(key, generated);
    } catch (err) {
      // ignore
    }
    return generated;
  })();

  const includeWaves = true;
  const includeFusion = true;
  const includeTimings = true;

  const elements = {
    canvas: document.getElementById('screen-preview'),
    preview: document.querySelector('.screen-preview'),
    overlayImg: document.getElementById('screen-overlay-img'),
    overlayBox: document.querySelector('.screen-overlay'),
    status: document.querySelector('[data-role="status"]'),
    requestId: document.querySelector('[data-role="request-id"]'),
    timestamp: document.querySelector('[data-role="timestamp"]'),
    symbol: document.querySelector('[data-role="symbol"]'),
    timeframe: document.querySelector('[data-role="timeframe"]'),
    analysisMode: document.querySelector('[data-role="analysis-mode"]'),
    patternName: document.querySelector('[data-role="pattern-name"]'),
    patternConfidence: document.querySelector('[data-role="pattern-confidence"]'),
    patternBias: document.querySelector('[data-role="pattern-bias"]'),
    patternSuggestion: document.querySelector('[data-role="pattern-suggestion"]'),
    waveName: document.querySelector('[data-role="wave-name"]'),
    waveStage: document.querySelector('[data-role="wave-stage"]'),
    waveDirection: document.querySelector('[data-role="wave-direction"]'),
    waveConfidence: document.querySelector('[data-role="wave-confidence"]'),
    probSource: document.querySelector('[data-role="prob-source"]'),
    probUp: document.querySelector('[data-role="prob-up"]'),
    probDown: document.querySelector('[data-role="prob-down"]'),
    probNeutral: document.querySelector('[data-role="prob-neutral"]'),
    probUpLabel: document.querySelector('[data-role="prob-up-label"]'),
    probDownLabel: document.querySelector('[data-role="prob-down-label"]'),
    probNeutralLabel: document.querySelector('[data-role="prob-neutral-label"]'),
    diagnostics: document.querySelector('[data-role="diagnostics"]'),
    overlayLayers: document.querySelectorAll('[data-overlay-layer]'),
    interval: document.getElementById('screen-interval'),
    adaptive: document.getElementById('screen-adaptive'),
    quality: document.getElementById('screen-quality'),
    mode: document.getElementById('screen-mode'),
    ocr: document.getElementById('screen-ocr'),
    start: document.querySelector('[data-action="start"]'),
    stop: document.querySelector('[data-action="stop"]'),
    analyze: document.querySelector('[data-action="analyze"]'),
    toggle: document.querySelector('[data-action="toggle"]'),
    autoCrop: document.querySelector('[data-action="auto-crop"]'),
    clear: document.querySelector('[data-action="clear"]'),
    pickButtons: document.querySelectorAll('[data-pick]'),
    patternLabel: document.getElementById('pattern-label'),
    saveSample: document.querySelector('[data-action="save-sample"]'),
    trainModel: document.querySelector('[data-action="train-model"]'),
    trainingStatus: document.querySelector('[data-role="training-status"]'),
    trainingSummary: document.querySelector('[data-role="training-summary"]'),
  };

  if (!elements.canvas) return;
  if (elements.ocr) {
    elements.ocr.checked = defaultOcr;
    if (!defaultOcr) elements.ocr.disabled = true;
  }

  const ctx = elements.canvas.getContext('2d');
  const video = document.createElement('video');
  video.playsInline = true;

  const analysisCanvas = document.createElement('canvas');
  const analysisCtx = analysisCanvas.getContext('2d');
  const autoCanvas = document.createElement('canvas');
  const autoCtx = autoCanvas.getContext('2d');

  const state = {
    stream: null,
    capturing: false,
    analyzing: false,
    autoTimer: null,
    roi: null,
    selecting: false,
    selectionStart: null,
    inFlight: false,
    pickTarget: null,
    autoInterval: null,
    autoTimerInterval: null,
    failureCount: 0,
    calibration: {
      line: null,
      candle_up: null,
      candle_down: null,
    },
  };

  const setStatus = (stateKey, text) => {
    if (!elements.status) return;
    elements.status.dataset.state = stateKey;
    elements.status.textContent = text;
  };

  const setButtons = (enabled) => {
    if (elements.stop) elements.stop.disabled = !enabled;
    if (elements.analyze) elements.analyze.disabled = !enabled;
    if (elements.toggle) elements.toggle.disabled = !enabled;
    if (elements.autoCrop) elements.autoCrop.disabled = !enabled;
    if (elements.clear) elements.clear.disabled = !enabled;
    if (elements.pickButtons && elements.pickButtons.length) {
      elements.pickButtons.forEach((btn) => {
        btn.disabled = !enabled;
      });
    }
    if (elements.saveSample) elements.saveSample.disabled = !enabled;
  };

  const resizeCanvas = (width, height) => {
    if (!width || !height) return;
    elements.canvas.width = width;
    elements.canvas.height = height;
  };

  const drawFrame = () => {
    if (!state.capturing) return;
    if (video.readyState >= 2) {
      ctx.drawImage(video, 0, 0, elements.canvas.width, elements.canvas.height);
      drawRoi();
    }
    requestAnimationFrame(drawFrame);
  };

  const drawRoi = () => {
    if (!state.roi) return;
    const { x, y, w, h } = roiToPixels(state.roi);
    ctx.save();
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeStyle = '#38bdf8';
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  };

  const roiToPixels = (roi) => {
    const width = elements.canvas.width;
    const height = elements.canvas.height;
    return {
      x: Math.max(0, Math.round(roi.x * width)),
      y: Math.max(0, Math.round(roi.y * height)),
      w: Math.max(1, Math.round(roi.w * width)),
      h: Math.max(1, Math.round(roi.h * height)),
    };
  };

  const clampRoi = (roi) => {
    const x = Math.min(Math.max(roi.x, 0), 0.98);
    const y = Math.min(Math.max(roi.y, 0), 0.98);
    const w = Math.min(Math.max(roi.w, 0.02), 1 - x);
    const h = Math.min(Math.max(roi.h, 0.02), 1 - y);
    return { x, y, w, h };
  };

  const startCapture = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      setStatus('error', labels.msg_capture_unsupported || 'Capture unsupported');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 8 },
        audio: false,
      });
      state.stream = stream;
      stream.getTracks().forEach((track) => {
        track.addEventListener('ended', stopCapture);
        track.addEventListener('inactive', stopCapture);
      });
      video.srcObject = stream;
      await video.play();
      resizeCanvas(video.videoWidth || 960, video.videoHeight || 540);
      state.capturing = true;
      setButtons(true);
      setStatus('active', labels.status_active || 'Capturing');
      drawFrame();
      if (elements.start) elements.start.disabled = true;
    } catch (err) {
      setStatus('error', labels.msg_permission_denied || 'Permission denied');
    }
  };

  const stopCapture = () => {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
    }
    state.stream = null;
    state.capturing = false;
    state.pickTarget = null;
    if (elements.preview) elements.preview.classList.remove('is-picking');
    stopAutoAnalyze();
    setButtons(false);
    if (elements.start) elements.start.disabled = false;
    setStatus('idle', labels.status_stopped || 'Stopped');
  };

  const currentInterval = () => {
    if (elements.interval) {
      const value = Number(elements.interval.value);
      if (!Number.isNaN(value) && value > 200) return value;
    }
    return defaultInterval;
  };

  const adaptiveEnabled = () => {
    if (elements.adaptive) return elements.adaptive.checked;
    return false;
  };

  const currentQuality = () => {
    if (elements.quality) {
      const value = Number(elements.quality.value);
      if (!Number.isNaN(value)) return value;
    }
    return 0.7;
  };

  const currentMode = () => {
    if (elements.mode) return elements.mode.value || 'auto';
    return 'auto';
  };

  const currentOverlayLayers = () => {
    if (!elements.overlayLayers || elements.overlayLayers.length === 0) return [];
    const layers = [];
    elements.overlayLayers.forEach((input) => {
      if (input.checked && input.dataset.overlayLayer) {
        layers.push(input.dataset.overlayLayer);
      }
    });
    return layers;
  };

  const currentOcr = () => {
    if (elements.ocr) return elements.ocr.checked;
    return defaultOcr;
  };

  const rgbToHex = (rgb) => {
    const toHex = (value) => value.toString(16).padStart(2, '0');
    return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`;
  };

  const rgbToHsv = (rgb) => {
    const r = rgb.r / 255;
    const g = rgb.g / 255;
    const b = rgb.b / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    let h = 0;
    if (delta !== 0) {
      if (max === r) h = ((g - b) / delta) % 6;
      else if (max === g) h = (b - r) / delta + 2;
      else h = (r - g) / delta + 4;
      h = Math.round(h * 42.5);
      if (h < 0) h += 255;
    }
    const s = max === 0 ? 0 : delta / max;
    const v = max;
    return {
      h,
      s: Math.round(s * 255),
      v: Math.round(v * 255),
    };
  };

  const setCalibration = (key, rgb) => {
    const hsv = rgbToHsv(rgb);
    state.calibration[key] = { ...rgb, ...hsv };
    const swatch = document.querySelector(`[data-swatch="${key}"]`);
    const label = document.querySelector(`[data-label="${key}"]`);
    if (swatch) swatch.style.backgroundColor = rgbToHex(rgb);
    if (label) label.textContent = `${rgbToHex(rgb)} | h${hsv.h}`;
  };

  const buildCalibrationPayload = () => {
    const payload = {};
    Object.keys(state.calibration).forEach((key) => {
      if (state.calibration[key]) payload[key] = state.calibration[key];
    });
    return payload;
  };

  const captureRegion = (region, quality) => {
    const canvas = document.createElement('canvas');
    canvas.width = region.w;
    canvas.height = region.h;
    const localCtx = canvas.getContext('2d');
    if (!localCtx) return null;
    localCtx.drawImage(elements.canvas, region.x, region.y, region.w, region.h, 0, 0, region.w, region.h);
    return canvas.toDataURL('image/jpeg', quality);
  };

  const captureHeaderFrame = () => {
    if (!state.capturing || !currentOcr()) return null;
    const roi = state.roi ? roiToPixels(state.roi) : { x: 0, y: 0, w: elements.canvas.width, h: elements.canvas.height };
    const headerHeight = Math.round(roi.h * 0.22);
    const headerY = Math.max(0, roi.y - headerHeight);
    const headerRegion = {
      x: roi.x,
      y: headerY,
      w: roi.w,
      h: Math.max(1, headerHeight),
    };
    try {
      return captureRegion(headerRegion, Math.min(0.65, currentQuality()));
    } catch (err) {
      return null;
    }
  };

  const captureFrame = () => {
    if (!state.capturing) return null;
    if (!analysisCtx) return null;
    const roi = state.roi ? roiToPixels(state.roi) : { x: 0, y: 0, w: elements.canvas.width, h: elements.canvas.height };
    if (!roi.w || !roi.h) return null;
    analysisCanvas.width = roi.w;
    analysisCanvas.height = roi.h;
    analysisCtx.drawImage(elements.canvas, roi.x, roi.y, roi.w, roi.h, 0, 0, roi.w, roi.h);
    try {
      return analysisCanvas.toDataURL('image/jpeg', currentQuality());
    } catch (err) {
      return null;
    }
  };

  const ensureRoi = () => {
    if (!state.roi) autoCrop();
  };

  const fallbackRoi = () =>
    clampRoi({
      x: 0.06,
      y: 0.1,
      w: 0.78,
      h: 0.72,
    });

  const findPeakIndex = (values, start, end) => {
    let max = -1;
    let idx = start;
    for (let i = start; i < end; i += 1) {
      if (values[i] > max) {
        max = values[i];
        idx = i;
      }
    }
    return { idx, value: max };
  };

  const autoCrop = () => {
    if (!state.capturing || !autoCtx) {
      setStatus('error', labels.msg_missing_capture || 'Start capture first');
      return;
    }
    const width = elements.canvas.width;
    const height = elements.canvas.height;
    if (!width || !height) {
      setStatus('error', labels.msg_auto_crop_failed || 'Auto crop failed');
      return;
    }
    const targetWidth = 260;
    const scale = Math.min(1, targetWidth / width);
    const sampleW = Math.max(120, Math.round(width * scale));
    const sampleH = Math.max(90, Math.round(height * scale));
    autoCanvas.width = sampleW;
    autoCanvas.height = sampleH;
    autoCtx.drawImage(elements.canvas, 0, 0, sampleW, sampleH);
    const data = autoCtx.getImageData(0, 0, sampleW, sampleH).data;
    const total = sampleW * sampleH;
    const grays = new Float32Array(total);
    for (let i = 0, p = 0; i < total; i += 1, p += 4) {
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      grays[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    const rowEdges = new Float32Array(sampleH);
    const colEdges = new Float32Array(sampleW);
    for (let y = 1; y < sampleH; y += 1) {
      const rowOffset = y * sampleW;
      const prevOffset = rowOffset - sampleW;
      for (let x = 1; x < sampleW; x += 1) {
        const idx = rowOffset + x;
        const g = grays[idx];
        const gx = Math.abs(g - grays[idx - 1]);
        const gy = Math.abs(g - grays[prevOffset + x]);
        const edge = gx + gy;
        rowEdges[y] += edge;
        colEdges[x] += edge;
      }
    }
    const rowBandStart = Math.floor(sampleH * 0.2);
    const rowBandEnd = Math.floor(sampleH * 0.85);
    const colBandStart = Math.floor(sampleW * 0.12);
    const colBandEnd = Math.floor(sampleW * 0.9);
    const rowPeak = findPeakIndex(rowEdges, rowBandStart, rowBandEnd);
    const colPeak = findPeakIndex(colEdges, colBandStart, colBandEnd);
    if (rowPeak.value < 1 || colPeak.value < 1) {
      state.roi = fallbackRoi();
      setStatus('active', labels.msg_auto_crop_failed || 'Auto crop failed');
      return;
    }
    const rowThreshold = rowPeak.value * 0.35;
    const colThreshold = colPeak.value * 0.35;
    let top = rowPeak.idx;
    let bottom = rowPeak.idx;
    let left = colPeak.idx;
    let right = colPeak.idx;
    while (top > 0 && rowEdges[top] >= rowThreshold) top -= 1;
    while (bottom < sampleH - 1 && rowEdges[bottom] >= rowThreshold) bottom += 1;
    while (left > 0 && colEdges[left] >= colThreshold) left -= 1;
    while (right < sampleW - 1 && colEdges[right] >= colThreshold) right += 1;
    const minW = Math.floor(sampleW * 0.35);
    const minH = Math.floor(sampleH * 0.35);
    if (right - left < minW || bottom - top < minH) {
      state.roi = fallbackRoi();
      setStatus('active', labels.msg_auto_crop_failed || 'Auto crop failed');
      return;
    }
    const marginX = Math.max(2, Math.round(sampleW * 0.01));
    const marginY = Math.max(2, Math.round(sampleH * 0.015));
    left = Math.max(0, left + marginX);
    right = Math.min(sampleW - 1, right - marginX);
    top = Math.max(0, top + marginY);
    bottom = Math.min(sampleH - 1, bottom - marginY);
    const roi = clampRoi({
      x: left / sampleW,
      y: top / sampleH,
      w: (right - left) / sampleW,
      h: (bottom - top) / sampleH,
    });
    state.roi = roi;
    setStatus('active', labels.msg_auto_crop_success || 'Auto crop set');
  };

  const updateResult = (data) => {
    const key = data.pattern_key || 'unknown';
    const copy = patternCopy[key] || patternCopy.unknown || { name: key, bias: '-', suggestion: '-' };
    if (elements.patternName) elements.patternName.textContent = copy.name || key;
    if (elements.patternBias) {
      const biasPrefix = labels.bias_prefix || 'Bias: ';
      elements.patternBias.textContent = `${biasPrefix}${copy.bias || '-'}`;
    }
    if (elements.patternSuggestion) elements.patternSuggestion.textContent = copy.suggestion || '-';
    if (elements.patternConfidence) {
      const confidence = data.confidence != null ? Math.round(data.confidence * 100) : 0;
      const confidencePrefix = labels.confidence_prefix || 'Confidence: ';
      elements.patternConfidence.textContent = `${confidencePrefix}${confidence}%`;
    }
    if (elements.requestId) {
      const prefix = labels.request_id_prefix || 'ID ';
      elements.requestId.textContent = data.request_id ? `${prefix}${data.request_id}` : '';
    }
    if (elements.timestamp) elements.timestamp.textContent = new Date().toLocaleTimeString();
    if (elements.symbol) {
      const label = labels.label_symbol || 'Symbol';
      elements.symbol.textContent = `${label}: ${data.symbol || '-'}`;
    }
    if (elements.timeframe) {
      const label = labels.label_timeframe || 'TF';
      elements.timeframe.textContent = `${label}: ${data.timeframe || '-'}`;
    }
    if (elements.analysisMode) {
      const label = labels.label_mode || 'Mode';
      elements.analysisMode.textContent = `${label}: ${data.analysis_mode || '-'}`;
    }
    const wave = data.wave || null;
    const waveKey = wave && wave.wave_key ? wave.wave_key : 'unknown';
    const waveCopyItem = waveCopy[waveKey] || waveCopy.unknown || { name: waveKey };
    if (elements.waveName) elements.waveName.textContent = waveCopyItem.name || waveKey || '-';
    if (elements.waveStage) {
      const stageLabelMap = {
        impulse: labels.wave_stage_impulse || 'Impulse',
        correction: labels.wave_stage_correction || 'Correction',
        unknown: labels.wave_stage_unknown || 'Unknown',
      };
      const stage = wave && wave.stage ? wave.stage : 'unknown';
      elements.waveStage.textContent = stageLabelMap[stage] || stage;
    }
    if (elements.waveDirection) {
      const directionLabelMap = {
        up: labels.wave_direction_up || 'Up',
        down: labels.wave_direction_down || 'Down',
        neutral: labels.wave_direction_neutral || 'Neutral',
      };
      const direction = wave && wave.direction ? wave.direction : 'neutral';
      elements.waveDirection.textContent = directionLabelMap[direction] || direction;
    }
    if (elements.waveConfidence) {
      const waveConfidence = wave && wave.confidence != null ? Math.round(wave.confidence * 100) : null;
      elements.waveConfidence.textContent = waveConfidence != null ? `${waveConfidence}%` : '-';
    }
    if (elements.overlayImg && elements.overlayBox) {
      if (data.overlay_image) {
        elements.overlayImg.src = data.overlay_image;
        elements.overlayBox.classList.add('has-image');
      } else {
        elements.overlayImg.removeAttribute('src');
        elements.overlayBox.classList.remove('has-image');
      }
    }
    if (data.suggested_interval_ms && adaptiveEnabled()) {
      updateAutoInterval(data.suggested_interval_ms);
    }

    const prob = data.fused_probabilities || data.probabilities || {};
    if (elements.probSource) {
      elements.probSource.textContent = data.fused_probabilities
        ? labels.prob_source_fused || 'Fused probability'
        : labels.prob_source_pattern || 'Pattern probability';
    }
    updateProbability(elements.probUp, elements.probUpLabel, prob.up);
    updateProbability(elements.probDown, elements.probDownLabel, prob.down);
    updateProbability(elements.probNeutral, elements.probNeutralLabel, prob.neutral);

    if (elements.diagnostics) {
      const diagnostics = data.diagnostics || {};
      const labelCoverage = labels.diagnostic_coverage || 'Coverage';
      const labelRange = labels.diagnostic_range || 'Range';
      const labelExtrema = labels.diagnostic_extrema || 'Extrema';
      const labelInterval = labels.diagnostic_interval || 'Interval';
      const labelVolatility = labels.diagnostic_volatility || 'Volatility';
      const labelWavePivots = labels.diagnostic_wave_pivots || 'Wave pivots';
      const labelLatency = labels.diagnostic_latency || 'Latency';
      const ocrState =
        diagnostics.ocr_available === true ? labels.ocr_on || 'OCR on' : labels.ocr_off || 'OCR off';
      const modelState = diagnostics.model_used ? labels.model_on || 'Model on' : labels.model_off || 'Model off';
      const info = [
        `${labelCoverage}: ${diagnostics.coverage ?? '-'}`,
        `${labelRange}: ${diagnostics.median_range ?? '-'}`,
        `${labelExtrema}: ${diagnostics.maxima ?? '-'} / ${diagnostics.minima ?? '-'}`,
        `${labelInterval}: ${data.suggested_interval_ms ? `${data.suggested_interval_ms}ms` : '-'}`,
        `${labelVolatility}: ${diagnostics.analysis_volatility ?? '-'}`,
        `${labelWavePivots}: ${wave && wave.diagnostics ? wave.diagnostics.pivot_count ?? '-' : '-'}`,
        `${labelLatency}: ${data.timings_ms && data.timings_ms.total ? `${data.timings_ms.total}ms` : '-'}`,
        ocrState,
        modelState,
      ];
      elements.diagnostics.textContent = info.join(' | ');
    }
  };

  const updateProbability = (bar, label, value) => {
    if (!bar || !label) return;
    const pct = value != null ? Math.round(value * 100) : 0;
    bar.style.width = `${pct}%`;
    label.textContent = `${pct}%`;
  };

  const updateAutoInterval = (nextInterval) => {
    const interval = Math.round(Number(nextInterval));
    if (!interval || Number.isNaN(interval)) return;
    state.autoInterval = interval;
    if (!state.analyzing || !adaptiveEnabled()) return;
    const current = state.autoTimerInterval || currentInterval();
    if (Math.abs(interval - current) < 150) return;
    scheduleAutoAnalyze(interval);
  };

  const analyzeOnce = async () => {
    if (!state.capturing || state.inFlight) return;
    if (!state.stream || !state.stream.active) {
      stopCapture();
      setStatus('error', labels.msg_capture_unsupported || 'Capture ended');
      return;
    }
    ensureRoi();
    const image = captureFrame();
    if (!image) {
      setStatus('error', labels.msg_frame_missing || 'No frame');
      return;
    }
    const headerImage = captureHeaderFrame();
    const calibration = buildCalibrationPayload();
    const mode = currentMode();
    if (mode === 'candlestick' && !calibration.candle_up && !calibration.candle_down) {
      setStatus('error', labels.msg_need_candle_color || 'Pick candle color');
      return;
    }
    state.inFlight = true;
    setStatus('active', labels.status_analyzing || 'Analyzing');
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({
          image,
          header_image: headerImage,
          mode,
          calibration,
          ocr_enabled: currentOcr(),
          include_waves: includeWaves,
          include_fusion: includeFusion,
          include_timings: includeTimings,
          overlay_layers: currentOverlayLayers(),
          session_id: sessionId,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        const detail = data.error_detail ? ` (${data.error_detail})` : '';
        const message = `${data.next_action || data.error || labels.msg_analysis_failed || 'Analysis failed'}${detail}`;
        setStatus('error', message);
        state.failureCount += 1;
        if (state.failureCount >= 3) {
          stopAutoAnalyze();
        }
      } else {
        setStatus('active', labels.status_updated || 'Updated');
        updateResult(data);
        state.failureCount = 0;
      }
    } catch (err) {
      setStatus('error', labels.msg_network_error || 'Network error');
      state.failureCount += 1;
      if (state.failureCount >= 3) {
        stopAutoAnalyze();
      }
    } finally {
      state.inFlight = false;
    }
  };

  const scheduleAutoAnalyze = (interval) => {
    if (state.autoTimer) clearInterval(state.autoTimer);
    state.autoTimerInterval = interval;
    state.autoTimer = window.setInterval(analyzeOnce, interval);
  };

  const startAutoAnalyze = () => {
    if (state.analyzing) return;
    state.analyzing = true;
    if (elements.toggle) elements.toggle.textContent = labels.toggle_on || 'Stop auto';
    ensureRoi();
    const interval = adaptiveEnabled() && state.autoInterval ? state.autoInterval : currentInterval();
    scheduleAutoAnalyze(interval);
  };

  const stopAutoAnalyze = () => {
    state.analyzing = false;
    if (elements.toggle) elements.toggle.textContent = labels.toggle_off || 'Auto analyze';
    if (state.autoTimer) {
      clearInterval(state.autoTimer);
      state.autoTimer = null;
    }
    state.autoTimerInterval = null;
  };

  const toggleAuto = () => {
    if (state.analyzing) stopAutoAnalyze();
    else startAutoAnalyze();
  };

  const clearRoi = () => {
    state.roi = null;
  };

  const toCanvasPoint = (event) => {
    const rect = elements.canvas.getBoundingClientRect();
    const scaleX = elements.canvas.width / rect.width;
    const scaleY = elements.canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  const beginPick = (target) => {
    if (!state.capturing) {
      setStatus('error', labels.msg_missing_capture || 'Start capture first');
      return;
    }
    state.pickTarget = target;
    if (elements.preview) elements.preview.classList.add('is-picking');
    setStatus('pick', labels.msg_pick_prompt || 'Pick a color');
  };

  const handlePick = (event) => {
    if (!state.pickTarget || !state.capturing) return;
    const point = toCanvasPoint(event);
    const x = Math.max(0, Math.min(elements.canvas.width - 1, Math.round(point.x)));
    const y = Math.max(0, Math.min(elements.canvas.height - 1, Math.round(point.y)));
    const pixel = ctx.getImageData(x, y, 1, 1).data;
    const rgb = { r: pixel[0], g: pixel[1], b: pixel[2] };
    setCalibration(state.pickTarget, rgb);
    state.pickTarget = null;
    if (elements.preview) elements.preview.classList.remove('is-picking');
    setStatus('active', labels.status_active || 'Capturing');
  };

  const setTrainingStatus = (text) => {
    if (!elements.trainingStatus) return;
    elements.trainingStatus.textContent = text;
  };

  const updateTrainingSummary = (payload) => {
    if (!elements.trainingSummary) return;
    if (!payload) {
      elements.trainingSummary.textContent = '-';
      return;
    }
    const parts = [];
    if (payload.total_samples != null) parts.push(`samples=${payload.total_samples}`);
    if (payload.accuracy != null) parts.push(`acc=${Math.round(payload.accuracy * 100)}%`);
    if (payload.test_size != null) parts.push(`test=${payload.test_size}`);
    if (payload.override_threshold != null) {
      const thr = Number(payload.override_threshold);
      if (!Number.isNaN(thr)) parts.push(`thr=${thr.toFixed(2)}`);
    }
    if (payload.override_accuracy != null) {
      parts.push(`ovr_acc=${Math.round(payload.override_accuracy * 100)}%`);
    }
    if (payload.override_coverage != null) {
      parts.push(`cov=${Math.round(payload.override_coverage * 100)}%`);
    }
    elements.trainingSummary.textContent = parts.length ? parts.join(' | ') : '-';
  };

  const saveSample = async () => {
    if (!sampleUrl) return;
    if (!state.capturing) {
      setStatus('error', labels.msg_missing_capture || 'Start capture first');
      return;
    }
    const label = elements.patternLabel ? elements.patternLabel.value : '';
    if (!label) {
      setStatus('error', labels.msg_need_label || 'Select label');
      return;
    }
    const image = captureFrame();
    if (!image) {
      setStatus('error', labels.msg_frame_missing || 'No frame');
      return;
    }
    if (currentMode() === 'candlestick') {
      const calibration = buildCalibrationPayload();
      if (!calibration.candle_up && !calibration.candle_down) {
        setStatus('error', labels.msg_need_candle_color || 'Pick candle color');
        return;
      }
    }
    setTrainingStatus(labels.msg_sample_start || 'Saving...');
    try {
      const response = await fetch(sampleUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({
          image,
          label,
          mode: currentMode(),
          calibration: buildCalibrationPayload(),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setTrainingStatus(data.error || labels.msg_analysis_failed || 'Save failed');
        return;
      }
      setTrainingStatus(labels.msg_sample_saved || 'Sample saved');
      updateTrainingSummary({ total_samples: data.total_samples });
    } catch (err) {
      setTrainingStatus(labels.msg_network_error || 'Network error');
    }
  };

  const trainModel = async () => {
    if (!trainUrl) return;
    setTrainingStatus(labels.msg_train_start || 'Training...');
    try {
      const response = await fetch(trainUrl, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken },
      });
      const data = await response.json();
      if (!response.ok) {
        setTrainingStatus(data.error || labels.msg_train_fail || 'Train failed');
        return;
      }
      setTrainingStatus(labels.msg_train_success || 'Trained');
      updateTrainingSummary(data);
    } catch (err) {
      setTrainingStatus(labels.msg_network_error || 'Network error');
    }
  };

  elements.canvas.addEventListener('mousedown', (event) => {
    if (!state.capturing || state.pickTarget) return;
    state.selecting = true;
    state.selectionStart = toCanvasPoint(event);
  });

  elements.canvas.addEventListener('mousemove', (event) => {
    if (!state.selecting || !state.selectionStart) return;
    const current = toCanvasPoint(event);
    const x = Math.min(state.selectionStart.x, current.x);
    const y = Math.min(state.selectionStart.y, current.y);
    const w = Math.abs(state.selectionStart.x - current.x);
    const h = Math.abs(state.selectionStart.y - current.y);
    state.roi = clampRoi({
      x: x / elements.canvas.width,
      y: y / elements.canvas.height,
      w: w / elements.canvas.width,
      h: h / elements.canvas.height,
    });
  });

  elements.canvas.addEventListener('mouseup', () => {
    state.selecting = false;
    state.selectionStart = null;
  });

  elements.canvas.addEventListener('mouseleave', () => {
    state.selecting = false;
    state.selectionStart = null;
  });

  elements.canvas.addEventListener('click', (event) => {
    if (!state.pickTarget) return;
    handlePick(event);
  });

  if (elements.start) elements.start.addEventListener('click', startCapture);
  if (elements.stop) elements.stop.addEventListener('click', stopCapture);
  if (elements.analyze) elements.analyze.addEventListener('click', analyzeOnce);
  if (elements.toggle) elements.toggle.addEventListener('click', toggleAuto);
  if (elements.autoCrop) elements.autoCrop.addEventListener('click', autoCrop);
  if (elements.clear) elements.clear.addEventListener('click', clearRoi);
  if (elements.pickButtons && elements.pickButtons.length) {
    elements.pickButtons.forEach((btn) => {
      btn.addEventListener('click', () => beginPick(btn.dataset.pick));
    });
  }
  if (elements.saveSample) elements.saveSample.addEventListener('click', saveSample);
  if (elements.trainModel) elements.trainModel.addEventListener('click', trainModel);

  if (elements.interval) {
    elements.interval.value = `${defaultInterval}`;
    elements.interval.addEventListener('change', () => {
      state.autoInterval = null;
      if (state.analyzing) {
        stopAutoAnalyze();
        startAutoAnalyze();
      }
    });
  }
  if (elements.adaptive) {
    elements.adaptive.addEventListener('change', () => {
      state.autoInterval = null;
      if (state.analyzing) {
        stopAutoAnalyze();
        startAutoAnalyze();
      }
    });
  }
})();
