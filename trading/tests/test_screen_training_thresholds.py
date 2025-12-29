from trading.screen_training import _select_override_threshold


def test_select_override_threshold_prefers_accuracy() -> None:
    probs = [
        [0.9, 0.1],
        [0.6, 0.4],
        [0.45, 0.55],
        [0.2, 0.8],
    ]
    labels = ["A", "B"]
    y_true = ["A", "A", "A", "B"]
    summary = _select_override_threshold(probs, y_true, labels=labels, thresholds=[0.5, 0.6, 0.7])
    assert summary is not None
    assert abs(summary["threshold"] - 0.6) < 1e-6
    assert 0.0 <= summary["coverage"] <= 1.0
    assert 0.0 <= summary["accuracy"] <= 1.0


def test_select_override_threshold_empty() -> None:
    assert _select_override_threshold([], [], labels=["A"]) is None
