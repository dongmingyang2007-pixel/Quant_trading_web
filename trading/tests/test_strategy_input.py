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
