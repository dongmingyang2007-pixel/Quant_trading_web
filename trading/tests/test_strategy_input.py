from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from django.test import SimpleTestCase

from trading.views.dashboard import build_strategy_input


class StrategyInputOverrideTests(SimpleTestCase):
    def test_build_strategy_input_execution_fields(self):
        cleaned = {
            "ticker": "AAPL",
            "benchmark_ticker": "SPY",
            "start_date": date(2023, 1, 1),
            "end_date": date(2023, 6, 30),
            "capital": 250000,
            "return_path": "open_to_close",
            "max_adv_participation": 0.12,
            "execution_liquidity_buffer": 0.08,
            "execution_penalty_bps": 7.5,
            "limit_move_threshold": 0.095,
            "borrow_cost_bps": 1.2,
        }
        user = SimpleNamespace(id=1, is_authenticated=True)
        strategy_input, _ = build_strategy_input(cleaned, request_id="req-1", user=user)

        self.assertEqual(strategy_input.return_path, "open_to_close")
        self.assertAlmostEqual(strategy_input.max_adv_participation, 0.12)
        self.assertAlmostEqual(strategy_input.execution_liquidity_buffer, 0.08)
        self.assertAlmostEqual(strategy_input.execution_penalty_bps, 7.5)
        self.assertAlmostEqual(strategy_input.limit_move_threshold, 0.095)
        self.assertAlmostEqual(strategy_input.borrow_cost_bps, 1.2)

    def test_trading_focus_intraday_applies_short_term_profile(self):
        cleaned = {
            "ticker": "NVDA",
            "benchmark_ticker": "SPY",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 4, 30),
            "capital": 120000,
            "trading_focus": "intraday_retail",
        }
        user = SimpleNamespace(id=2, is_authenticated=True)
        strategy_input, config = build_strategy_input(cleaned, request_id="req-2", user=user)

        self.assertEqual(strategy_input.trading_focus, "intraday_retail")
        self.assertEqual(strategy_input.return_path, "open_to_close")
        self.assertEqual(strategy_input.min_holding_days, 0)
        self.assertEqual(strategy_input.execution_delay_days, 0)
        self.assertEqual(strategy_input.strategy_engine, "ml_momentum")
        self.assertEqual(config["short_window"], 8)
        self.assertEqual(config["long_window"], 34)

    def test_trading_focus_keeps_user_overrides(self):
        cleaned = {
            "ticker": "AAPL",
            "benchmark_ticker": "SPY",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 6, 30),
            "capital": 80000,
            "trading_focus": "scalp_experimental",
            "return_path": "close_to_open",
            "strategy_engine": "ml_momentum",
            "max_leverage": 2.2,
        }
        user = SimpleNamespace(id=3, is_authenticated=True)
        strategy_input, _ = build_strategy_input(cleaned, request_id="req-3", user=user)

        self.assertEqual(strategy_input.trading_focus, "scalp_experimental")
        self.assertEqual(strategy_input.return_path, "close_to_open")
        self.assertEqual(strategy_input.strategy_engine, "ml_momentum")
        self.assertAlmostEqual(strategy_input.max_leverage, 2.2)
