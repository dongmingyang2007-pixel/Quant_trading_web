from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from trading.strategies import (
    StrategyInput,
    build_feature_matrix,
    compute_indicators,
    compute_triple_barrier_labels,
)


def _legacy_triple_barrier(
    price: pd.Series,
    up: float | pd.Series,
    down: float | pd.Series,
    max_holding: int,
) -> tuple[pd.Series, pd.Series]:
    idx = price.index
    arr = price.to_numpy(dtype=float)
    n = len(arr)
    binary = np.full(n, np.nan)
    multi = np.zeros(n)
    if isinstance(up, pd.Series):
        up_series = up.reindex(idx)
        fill_val = float(up_series.mean()) if not up_series.dropna().empty else 0.0
        up_arr = up_series.ffill().bfill().fillna(fill_val).to_numpy(dtype=float)
    else:
        up_arr = np.full(n, float(up))
    if isinstance(down, pd.Series):
        down_series = down.reindex(idx)
        fill_val = float(down_series.mean()) if not down_series.dropna().empty else 0.0
        down_arr = down_series.ffill().bfill().fillna(fill_val).to_numpy(dtype=float)
    else:
        down_arr = np.full(n, float(down))
    for i in range(n - 1):
        p0 = arr[i]
        horizon = min(n - i - 1, max_holding)
        if horizon <= 0:
            break
        future = arr[i + 1 : i + 1 + horizon]
        rets = future / p0 - 1
        up_thr = float(up_arr[i]) if i < len(up_arr) else float(up_arr[-1])
        dn_thr = float(down_arr[i]) if i < len(down_arr) else float(down_arr[-1])
        hit_up = np.where(rets >= up_thr)[0]
        hit_dn = np.where(rets <= -dn_thr)[0]
        t_up = hit_up[0] if hit_up.size else np.inf
        t_dn = hit_dn[0] if hit_dn.size else np.inf
        if t_up < t_dn:
            binary[i] = 1
            multi[i] = 1
        elif t_dn < t_up:
            binary[i] = 0
            multi[i] = -1
        else:
            terminal = 1 if rets[-1] > 0 else 0
            binary[i] = terminal
            multi[i] = 0
    return pd.Series(binary, index=idx), pd.Series(multi, index=idx)


class TripleBarrierTests(SimpleTestCase):
    def test_compute_triple_barrier_labels_multiclass(self):
        prices = pd.Series([100, 105, 110, 90, 95, 108], index=pd.date_range("2024-01-01", periods=6))
        binary, multiclass = compute_triple_barrier_labels(prices, up=0.04, down=0.04, max_holding=3)
        self.assertEqual(binary.iloc[0], 1)
        self.assertIn(multiclass.iloc[0], {1})
        self.assertIn(multiclass.iloc[2], {-1, 0})

    def test_triple_barrier_dynamic_thresholds_recorded(self):
        idx = pd.date_range("2023-01-01", periods=180, freq="B")
        base = pd.DataFrame(
            {
                "open": np.linspace(100, 130, len(idx)),
                "high": np.linspace(101, 131, len(idx)),
                "low": np.linspace(99, 129, len(idx)),
                "close": np.linspace(100, 130, len(idx)),
                "adj close": np.linspace(100, 130, len(idx)),
                "volume": np.linspace(1e6, 2e6, len(idx)),
            },
            index=idx,
        )
        prices = compute_indicators(base, short_window=10, long_window=30, rsi_period=14)
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=idx[0].date(),
            end_date=idx[-1].date(),
            short_window=10,
            long_window=30,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            label_style="triple_barrier",
            tb_dynamic=True,
            tb_vol_multiplier=2.0,
            tb_up=0.02,
            tb_down=0.02,
            tb_max_holding=5,
        )
        dataset, features = build_feature_matrix(prices, params)
        self.assertIn("tb_up_active", dataset.columns)
        self.assertIn("tb_down_active", dataset.columns)
        self.assertTrue(dataset["target"].notna().any())

    def test_triple_barrier_vectorized_matches_reference(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-02-01", periods=40, freq="B")
        prices = pd.Series(100 + rng.standard_normal(len(idx)).cumsum(), index=idx)
        up_series = pd.Series(np.abs(rng.normal(0.03, 0.01, len(idx))), index=idx)
        down_series = pd.Series(np.abs(rng.normal(0.03, 0.01, len(idx))), index=idx)
        binary_new, multi_new = compute_triple_barrier_labels(prices, up_series, down_series, max_holding=5)
        binary_ref, multi_ref = _legacy_triple_barrier(prices, up_series, down_series, max_holding=5)
        np.testing.assert_allclose(binary_new.values, binary_ref.values, equal_nan=True)
        np.testing.assert_allclose(multi_new.values, multi_ref.values, equal_nan=True)
