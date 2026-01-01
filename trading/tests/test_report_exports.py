from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from unittest import mock

from trading.reporting import ReportRenderingError
from trading.models import BacktestRecord as BacktestRecordModel


def _snapshot():
    return {
        "ticker": "AAPL",
        "benchmark_ticker": "SPY",
        "metrics": [
            {"label": "年化收益", "value": "18%", "description": "近 3 年复合"},
        ],
        "stats": {"cagr": "18%", "max_drawdown": "-12%"},
        "ai_summary": "维持多头，关注收益率曲线。",
    }


class ReportExportTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="reporter", password="test123")
        self.client.force_login(self.user)

    def _seed_session(self):
        session = self.client.session
        session["last_result"] = json.dumps(_snapshot(), ensure_ascii=False)
        session.save()

    def test_export_html(self):
        self._seed_session()
        response = self.client.get(reverse("trading:export_report") + "?format=html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response["Content-Type"])
        self.assertIn("AAPL", response.content.decode("utf-8"))

    def test_export_csv(self):
        self._seed_session()
        response = self.client.get(reverse("trading:export_report") + "?format=csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response["Content-Type"])
        self.assertIn("年化收益", response.content.decode("utf-8"))

    def test_export_pdf_fallbacks_to_html_when_renderer_missing(self):
        self._seed_session()
        with mock.patch("trading.reporting.render_report_pdf", side_effect=ReportRenderingError("missing")):
            response = self.client.get(reverse("trading:export_report") + "?format=pdf")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["X-Export-Fallback"], "html")
        self.assertIn("text/html", response["Content-Type"])

    def test_export_history_id_snapshot(self):
        record = BacktestRecordModel.objects.create(
            record_id="hist-export-1",
            user=self.user,
            timestamp=timezone.now(),
            ticker="AAPL",
            benchmark="SPY",
            engine="sma",
            start_date="2023-01-01",
            end_date="2023-06-30",
            metrics=[],
            stats={},
            params={},
            warnings=[],
            snapshot=_snapshot(),
        )
        url = reverse("trading:export_report") + f"?format=json&history_id={record.record_id}"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response["Content-Type"])
        self.assertIn("AAPL", response.content.decode("utf-8"))
