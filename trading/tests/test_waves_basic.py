import numpy as np

from trading.screen_waves import analyze_waves


def test_impulse_up_detected():
    series = np.array([0.2, 0.8, 0.3, 0.9, 0.4, 0.95, 0.6, 1.0])
    result = analyze_waves(series, symbol="TEST", timeframe="1m", analysis_mode="line")
    assert result.wave_key in ("impulse_up", "unknown")
    assert 0.0 <= result.confidence <= 1.0


def test_unknown_when_too_few_pivots():
    series = np.array([0.5, 0.51, 0.49, 0.5])
    result = analyze_waves(series, symbol="TEST", timeframe="1m", analysis_mode="line")
    assert result.wave_key == "unknown"
