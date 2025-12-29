from trading.screen_ocr import canonicalize_timeframe


def test_timeframe_alias_minute():
    assert canonicalize_timeframe("1min") == "1m"
    assert canonicalize_timeframe("15MIN") == "15m"
    assert canonicalize_timeframe("1 m") == "1m"


def test_timeframe_alias_week():
    assert canonicalize_timeframe("1wk") == "1w"
    assert canonicalize_timeframe("2WKS") == "2w"


def test_timeframe_seconds_and_empty():
    assert canonicalize_timeframe("60s") == "60s"
    assert canonicalize_timeframe("") is None
    assert canonicalize_timeframe(None) is None
