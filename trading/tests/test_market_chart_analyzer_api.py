from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from trading.screen_training import TrainingMetrics, load_samples


@override_settings(DEBUG=True)
class MarketChartAnalyzerApiTests(TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self._settings = override_settings(DATA_CACHE_DIR=Path(self._temp_dir.name))
        self._settings.enable()
        self.addCleanup(self._settings.disable)

        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="wave_user", password="secret123")
        self.client.force_login(self.user)

        self.analyze_url = reverse("trading:market_chart_analyze")
        self.sample_url = reverse("trading:market_chart_analyze_sample")
        self.train_url = reverse("trading:market_chart_analyze_train")
        self.meta_url = reverse("trading:market_chart_analyze_meta")
        self.screen_page_url = reverse("trading:screen_analyzer")
        self.legacy_api_names = (
            "trading:screen_analyzer_api",
            "trading:screen_analyzer_sample_api",
            "trading:screen_analyzer_train_api",
        )

    @staticmethod
    def _build_series(length: int = 96) -> list[float]:
        return [100.0 + 0.5 * idx + math.sin(idx / 5.0) * 2.0 for idx in range(length)]

    @classmethod
    def _build_bars(cls, length: int = 96) -> list[dict[str, float]]:
        closes = cls._build_series(length)
        bars = []
        for idx, close in enumerate(closes):
            high = close + 1.2
            low = close - 1.1
            open_price = close - 0.35
            bars.append(
                {
                    "time": float(1_700_000_000 + idx * 300),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": float(100_000 + idx * 200),
                }
            )
        return bars

    def test_market_chart_analyze_success(self):
        payload = {
            "symbol": "SPY",
            "range": "1mo",
            "interval": "1d",
            "series": self._build_series(120),
        }
        response = self.client.post(
            self.analyze_url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("pattern_key", body)
        self.assertIn("wave", body)
        self.assertIn("diagnostics", body)
        self.assertIn("request_id", body)

    def test_market_chart_analyze_series_mode_and_smoothing(self):
        payload = {
            "symbol": "SPY",
            "range": "1mo",
            "interval": "1h",
            "bars": self._build_bars(140),
            "series_mode": "ohlc4",
            "smoothing_window": 7,
        }
        response = self.client.post(
            self.analyze_url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("series_mode"), "ohlc4")
        self.assertEqual(body.get("smoothing_window"), 7)
        diagnostics = body.get("diagnostics", {})
        self.assertEqual(diagnostics.get("series_mode"), "ohlc4")
        self.assertEqual(diagnostics.get("smoothing_window"), 7)
        self.assertEqual(diagnostics.get("series_source"), "bars")
        self.assertTrue(isinstance(diagnostics.get("sample_index_map"), list))

    def test_market_chart_analyze_rejects_short_series(self):
        payload = {
            "symbol": "SPY",
            "range": "1mo",
            "interval": "1d",
            "series": self._build_series(8),
        }
        response = self.client.post(
            self.analyze_url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body.get("error_code"), "series_insufficient")

    def test_market_chart_sample_rejects_invalid_label(self):
        payload = {
            "symbol": "SPY",
            "range": "1mo",
            "interval": "1d",
            "series": self._build_series(64),
            "label": "invalid_label",
        }
        response = self.client.post(
            self.sample_url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body.get("error_code"), "invalid_label")

    def test_market_chart_sample_save_success(self):
        payload = {
            "symbol": "SPY",
            "range": "1mo",
            "interval": "1d",
            "series": self._build_series(64),
            "series_mode": "hlc3",
            "smoothing_window": 5,
            "label": "trend_up",
        }
        response = self.client.post(
            self.sample_url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("status"), "saved")
        self.assertEqual(body.get("label"), "trend_up")
        self.assertEqual(body.get("total_samples"), 1)
        samples = load_samples(namespace="market_chart_analyzer")
        self.assertEqual(len(samples), 1)
        sample_meta = samples[0].get("meta", {})
        self.assertEqual(sample_meta.get("series_mode"), "hlc3")
        self.assertEqual(sample_meta.get("smoothing_window"), 5)

    def test_market_chart_train_insufficient_samples(self):
        response = self.client.post(
            self.train_url,
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body.get("error_code"), "training_state_invalid")

    def test_market_chart_train_success_payload(self):
        metrics = TrainingMetrics(
            total_samples=24,
            classes={"trend_up": 12, "trend_down": 12},
            accuracy=0.81,
            test_size=6,
            override_threshold=0.63,
            override_accuracy=0.83,
            override_coverage=0.72,
            override_samples=5,
            override_source="validation",
        )
        with mock.patch("trading.views.market.train_screen_model", return_value=metrics):
            response = self.client.post(
                self.train_url,
                data=json.dumps({}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("status"), "trained")
        self.assertEqual(body.get("total_samples"), 24)
        self.assertIn("request_id", body)

    def test_market_chart_meta_reports_namespace(self):
        save_payload = {
            "symbol": "SPY",
            "range": "1mo",
            "interval": "1d",
            "series": self._build_series(64),
            "label": "trend_up",
        }
        save_response = self.client.post(
            self.sample_url,
            data=json.dumps(save_payload),
            content_type="application/json",
        )
        self.assertEqual(save_response.status_code, 200)

        response = self.client.get(self.meta_url)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("namespace"), "market_chart_analyzer")
        self.assertGreaterEqual(body.get("total_samples", 0), 1)

    def test_screen_analyzer_page_redirects_to_market_chart(self):
        response = self.client.get(self.screen_page_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/market/?view=chart&tool=wave", response.get("Location", ""))

    def test_legacy_screen_api_returns_migrated_error(self):
        for route_name in self.legacy_api_names:
            response = self.client.post(
                reverse(route_name),
                data=json.dumps({}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 410)
            body = response.json()
            self.assertEqual(body.get("error_code"), "screen_analyzer_migrated")
            self.assertIn("/market/?view=chart&tool=wave", body.get("redirect_to", ""))
