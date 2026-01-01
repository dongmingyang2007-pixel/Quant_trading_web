from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from paper.models import PaperTradingSession, PaperTrade


class PaperSessionApiTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="paper_user", password="secret123")
        self.client.force_login(self.user)

    def _create_session(self, **kwargs) -> PaperTradingSession:
        defaults = {
            "user": self.user,
            "name": "AAPL Core",
            "ticker": "AAPL",
            "benchmark": "SPY",
            "status": "running",
            "current_cash": 90000,
            "initial_cash": 100000,
            "last_equity": 102000,
            "equity_curve": [
                {"ts": timezone.now().isoformat(), "equity": 100000},
                {"ts": timezone.now().isoformat(), "equity": 102000},
            ],
        }
        defaults.update(kwargs)
        return PaperTradingSession.objects.create(**defaults)

    def test_list_filters_and_preview(self):
        self._create_session()
        self._create_session(name="TSLA Plan", ticker="TSLA", status="paused")
        url = reverse("trading:api_v1_paper_sessions") + "?status=running&q=AAPL&limit=5"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        session = payload["sessions"][0]
        self.assertEqual(session["ticker"], "AAPL")
        self.assertIn("equity_preview", session)
        self.assertTrue(len(session["equity_preview"]) > 0)
        self.assertNotIn("config", session)

    def test_export_trades_csv(self):
        session = self._create_session()
        PaperTrade.objects.create(
            session=session,
            symbol="AAPL",
            side="buy",
            quantity=1,
            price=150,
            notional=150,
        )
        url = reverse("trading:api_v1_paper_session_trades", kwargs={"session_id": session.session_id})
        response = self.client.get(url + "?format=csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response["Content-Type"])
        self.assertIn("AAPL", response.content.decode("utf-8"))
