from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pandas.testing as pdt

from django.test import SimpleTestCase

from trading.strategies import StrategyInput, compute_indicators, build_feature_matrix


def _make_prices(rows: int = 90) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = np.linspace(100, 150, rows)
    close = base * 0.995
    open_prices = close * 1.002
    frame = pd.DataFrame(
        {
            "adj close": base,
            "close": close,
            "open": open_prices,
            "high": base * 1.01,
            "low": base * 0.99,
            "volume": np.linspace(1_000_000, 1_500_000, rows),
        },
        index=idx,
    )
    return frame


def _build_params(return_path: str, *, label_return_path: str | None = None) -> StrategyInput:
    today = date.today()
    return StrategyInput(
        ticker="TEST",
        benchmark_ticker="SPY",
        start_date=today - timedelta(days=365),
        end_date=today,
        short_window=5,
        long_window=20,
        rsi_period=5,
        include_plots=False,
        show_ai_thoughts=False,
        risk_profile="balanced",
        capital=100000.0,
        return_path=return_path,
        label_return_path=label_return_path,
    )


class ReturnPathTests(SimpleTestCase):
    def test_future_return_uses_close_to_close_path_by_default(self):
        prices = compute_indicators(_make_prices(), 5, 20, 5)
        dataset, _ = build_feature_matrix(prices, _build_params("close_to_close"))
        pdt.assert_series_equal(
            dataset["future_return"],
            dataset["forward_return_close"],
            check_names=False,
        )

    def test_future_return_switches_to_close_to_open_path(self):
        prices = compute_indicators(_make_prices(), 5, 20, 5)
        dataset, _ = build_feature_matrix(prices, _build_params("close_to_open"))
        pdt.assert_series_equal(
            dataset["future_return"],
            dataset["forward_return_open"],
            check_names=False,
        )

    def test_label_return_path_overrides_training_path(self):
        prices = compute_indicators(_make_prices(), 5, 20, 5)
        dataset, _ = build_feature_matrix(prices, _build_params("close_to_close", label_return_path="close_to_open"))
        pdt.assert_series_equal(
            dataset["future_return"],
            dataset["forward_return_open"],
            check_names=False,
        )

    def test_direction_label_uses_label_return_path(self):
        prices = compute_indicators(_make_prices(), 5, 20, 5)
        params = _build_params("close_to_close", label_return_path="close_to_open")
        dataset, _ = build_feature_matrix(prices, params)
        # open path has small positive drift; target应全部为1
        self.assertTrue((dataset["target"].dropna() == 1).all())

    def test_open_to_close_intraday_path(self):
        prices = compute_indicators(_make_prices(), 5, 20, 5)
        dataset, _ = build_feature_matrix(prices, _build_params("open_to_close"))
        pdt.assert_series_equal(
            dataset["future_return"],
            dataset["forward_return_open_to_close"],
            check_names=False,
        )
