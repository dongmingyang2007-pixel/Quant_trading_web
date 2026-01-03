from __future__ import annotations

from datetime import date
from unittest import mock

from django.test import SimpleTestCase, override_settings

from trading.strategies import StrategyInput
from trading.strategies import pipeline as pipeline_module


class PipelineCacheTests(SimpleTestCase):
    def _params(self, *, user_id: str, request_id: str) -> StrategyInput:
        return StrategyInput(
            ticker="AAPL",
            benchmark_ticker="SPY",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 30),
            short_window=10,
            long_window=30,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            user_id=user_id,
            request_id=request_id,
        )

    def test_cache_signature_includes_user_id(self):
        params_a = self._params(user_id="user-a", request_id="req-a")
        params_b = self._params(user_id="user-b", request_id="req-b")
        sig_a = pipeline_module._params_cache_signature(params_a)
        sig_b = pipeline_module._params_cache_signature(params_b)
        self.assertNotEqual(sig_a, sig_b)

    @override_settings(STRATEGY_RESULT_CACHE_TTL=60, ENABLE_STRATEGY_RESULT_CACHE=True)
    @mock.patch("trading.strategies.pipeline.cache_get_object")
    @mock.patch("trading.strategies.pipeline._run_quant_pipeline_inner")
    def test_cache_hit_updates_request_and_user(self, mock_run, mock_cache_get):
        cached = {"params": {"user_id": "old-user", "request_id": "old-req"}, "stats": {}}
        mock_cache_get.return_value = cached

        params = self._params(user_id="new-user", request_id="new-req")
        result = pipeline_module.run_quant_pipeline(params)

        self.assertEqual(result["params"]["user_id"], "new-user")
        self.assertEqual(result["params"]["request_id"], "new-req")
        mock_run.assert_not_called()
