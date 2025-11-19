from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta

from django.test import SimpleTestCase

from trading.strategies import StrategyInput, _compute_oos_from_backtest, run_rl_policy_backtest, compute_indicators


class PFWSRLTests(SimpleTestCase):
    def test_compute_oos_from_backtest_for_rl_like_frame(self):
        idx = pd.date_range("2024-01-01", periods=120, freq="B")
        rng = np.random.default_rng(123)
        # 模拟 RL 策略收益
        strat_ret = pd.Series(rng.normal(0.0008, 0.01, len(idx)), index=idx)
        backtest = pd.DataFrame({"strategy_return": strat_ret})
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 6, 1),
            short_window=5,
            long_window=20,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            enforce_pfws_only=True,
            train_window=60,
            test_window=20,
            embargo_days=5,
        )
        report = _compute_oos_from_backtest(backtest, params)
        self.assertIsNotNone(report)
        self.assertEqual(report["train_window"], 60)
        self.assertEqual(report["test_window"], 20)
        self.assertEqual(report["embargo"], 5)
        self.assertGreaterEqual(report["folds"], 1)
        self.assertIn("sharpe", report["summary"])

    def test_run_rl_policy_outputs_oos_metrics_by_default(self):
        idx = pd.date_range("2024-01-01", periods=240, freq="B")
        base = 100 + np.cumsum(np.random.default_rng(7).normal(0.05, 0.9, len(idx)))
        prices = pd.DataFrame(
            {
                "adj close": base,
                "close": base * 0.999,
                "open": base * 1.001,
                "high": base * 1.01,
                "low": base * 0.99,
                "volume": np.linspace(1_000_000, 1_500_000, len(idx)),
            },
            index=idx,
        )
        prices = compute_indicators(prices, 5, 20, 14)
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 6, 1),
            short_window=3,
            long_window=10,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            enforce_pfws_only=False,  # 默认也应生成样本外指标
            train_window=80,
            test_window=20,
            embargo_days=5,
            return_path="open_to_close",
            label_return_path="open_to_close",
        )
        # 构造简化的 ML 结果以跳过 ML 训练长度要求
        ml_idx = prices.index[:220]
        ml_backtest = pd.DataFrame(
            {
                "asset_return": pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, len(ml_idx)), index=ml_idx),
                "probability": pd.Series(np.clip(np.random.default_rng(1).normal(0.5, 0.1, len(ml_idx)), 0, 1), index=ml_idx),
                "adj close": prices["adj close"].reindex(ml_idx).ffill(),
            }
        )
        ml_backtest["strategy_return"] = ml_backtest["asset_return"] * 0  # placeholder
        ml_metrics: list[dict[str, Any]] = []
        ml_stats: dict[str, Any] = {"feature_columns": [], "shap_img": None, "source_engine": "ml_mock"}

        backtest, _, stats = run_rl_policy_backtest(prices, params, ml_context=(ml_backtest, ml_metrics, ml_stats))
        self.assertIn("validation_penalized_sharpe", stats)
        self.assertIn("validation_oos_folds", stats)
