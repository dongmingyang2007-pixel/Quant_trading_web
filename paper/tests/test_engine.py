from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from paper.models import PaperTradingSession
from paper.engine import create_session, process_session, PaperTradingError
from trading.strategies import StrategyInput


class PaperTradingTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="alice", password="pw")

    def _params(self) -> StrategyInput:
        return StrategyInput(
            ticker="AAPL",
            benchmark_ticker="SPY",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 1),
            short_window=5,
            long_window=20,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            slippage_bps=0.0,
            transaction_cost_bps=0.0,
        )

    @patch("paper.engine.fetch_latest_quote")
    @patch("paper.engine.run_quant_pipeline")
    def test_process_session_rebalances_and_logs_trade(self, mock_pipeline, mock_quote):
        mock_pipeline.return_value = {"recent_rows": [{"position": 1}]}
        mock_quote.return_value = {"price": 100.0, "as_of": datetime(2024, 1, 1)}
        session = create_session(self.user, self._params(), initial_cash=Decimal("10000"))
        result = process_session(session, now=datetime(2024, 1, 1))

        refreshed = PaperTradingSession.objects.get(pk=session.pk)
        self.assertAlmostEqual(float(refreshed.last_equity), 10000.0, places=2)
        self.assertAlmostEqual(float(refreshed.current_cash), 0.0, places=2)
        self.assertAlmostEqual(refreshed.current_positions.get("AAPL"), 100.0, places=5)
        self.assertEqual(refreshed.status, "running")
        self.assertEqual(refreshed.trades.count(), 1)
        self.assertEqual(result["trades"][0]["side"], "buy")

    @patch("paper.engine.fetch_latest_quote")
    @patch("paper.engine.run_quant_pipeline")
    def test_process_session_prefers_price_cache(self, mock_pipeline, mock_quote):
        mock_pipeline.return_value = {"recent_rows": [{"position": 0.5}]}
        mock_quote.return_value = {"price": 999.0, "as_of": datetime(2024, 1, 1)}
        session = create_session(self.user, self._params(), initial_cash=Decimal("10000"))
        price_cache = {("AAPL", "5m"): {"price": 123.45, "as_of": datetime(2024, 1, 2)}}

        result = process_session(session, now=datetime(2024, 1, 2), price_cache=price_cache)

        refreshed = PaperTradingSession.objects.get(pk=session.pk)
        self.assertEqual(result["price"], 123.45)
        self.assertIn("AAPL", refreshed.current_positions)
        mock_quote.assert_not_called()

    @patch("paper.engine.fetch_latest_quote")
    @patch("paper.engine.run_quant_pipeline")
    def test_process_session_skips_when_quote_missing(self, mock_pipeline, mock_quote):
        mock_pipeline.return_value = {"recent_rows": [{"position": 1}]}
        mock_quote.return_value = {}
        session = create_session(self.user, self._params(), initial_cash=Decimal("10000"))
        result = process_session(session, now=datetime(2024, 1, 3), price_cache={})
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "quote_unavailable")

    @patch("paper.engine.fetch_latest_quote")
    @patch("paper.engine.run_quant_pipeline")
    def test_process_session_fallback_to_cached_signal(self, mock_pipeline, mock_quote):
        mock_pipeline.side_effect = PaperTradingError("boom")
        mock_quote.return_value = {"price": 50.0, "as_of": datetime(2024, 1, 4)}
        session = create_session(self.user, self._params(), initial_cash=Decimal("1000"))
        session.config["__last_signal"] = {"weight": 0.4, "at": datetime(2024, 1, 3).isoformat()}
        session.save(update_fields=["config"])
        result = process_session(session, now=datetime(2024, 1, 4), price_cache={})
        refreshed = PaperTradingSession.objects.get(pk=session.pk)
        self.assertFalse(result.get("skipped", False))
        self.assertAlmostEqual(float(refreshed.current_positions.get("AAPL", 0)), 8.0, places=4)
        self.assertEqual(result["price"], 50.0)

    @patch("paper.engine.fetch_latest_quote")
    @patch("paper.engine.run_quant_pipeline")
    def test_process_session_applies_slippage_and_commission(self, mock_pipeline, mock_quote):
        params = self._params()
        params.slippage_bps = 10.0
        params.transaction_cost_bps = 10.0
        mock_pipeline.return_value = {"recent_rows": [{"position": 0.8}]}
        mock_quote.return_value = {"price": 100.0, "as_of": datetime(2024, 1, 5)}
        session = create_session(self.user, params, initial_cash=Decimal("10000"))
        result = process_session(session, now=datetime(2024, 1, 5))
        refreshed = PaperTradingSession.objects.get(pk=session.pk)
        # target notional 8000, exec price with 10bps slippage => 100.1, qty ~79.92, commission ~8.01
        self.assertLess(float(refreshed.current_cash), 2000)  # spent cash including commission
        self.assertAlmostEqual(result["price"], 100.0)
        self.assertEqual(refreshed.trades.count(), 1)
