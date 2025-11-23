from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from paper.models import PaperTradingSession
from paper.engine import create_session, process_session
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
