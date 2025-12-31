from __future__ import annotations

import json
from types import SimpleNamespace

from django.contrib.auth import get_user_model
from django.test import TestCase, SimpleTestCase, override_settings
from django.urls import reverse
from unittest import mock


def _form_payload() -> dict[str, str]:
    return {
        "ticker": "AAPL",
        "benchmark_ticker": "SPY",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "capital": "250000",
        "ml_mode": "light",
    }


class BacktestTaskApiTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="celery", password="secret123")
        self.client.force_login(self.user)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True, TASK_RETURN_SNAPSHOT=False)
    @mock.patch("trading.task_queue.execute_backtest")
    def test_enqueue_backtest_task_returns_history_for_sync(self, mock_execute):
        mock_execute.return_value = {"history_id": "history-1"}
        response = self.client.post(
            reverse("trading:enqueue_backtest_task"),
            data=json.dumps({"params": _form_payload()}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["task_id"].startswith("sync-"))
        self.assertEqual(payload["state"], "SUCCESS")
        self.assertIn("result", payload)
        self.assertEqual(payload["result"]["history_id"], "history-1")
        self.assertNotIn("result", payload["result"])
        mock_execute.assert_called_once()

    @override_settings(CELERY_TASK_ALWAYS_EAGER=False)
    @mock.patch("trading.task_queue.execute_backtest")
    def test_enqueue_backtest_task_returns_task_id_for_async(self, mock_execute):
        dummy_task = SimpleNamespace(id="celery-42", state="PENDING")
        with mock.patch("trading.task_queue.run_backtest_task") as mock_task:
            mock_task.delay.return_value = dummy_task
            response = self.client.post(
                reverse("trading:enqueue_backtest_task"),
                data=json.dumps({"params": _form_payload()}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_id"], "celery-42")
        self.assertEqual(payload["state"], "PENDING")
        self.assertNotIn("result", payload)
        mock_task.delay.assert_called_once()
        mock_execute.assert_not_called()

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    @mock.patch("trading.task_queue.execute_training_job")
    def test_enqueue_training_task_sync_result(self, mock_execute):
        mock_execute.return_value = {"history_id": None, "result": {"best": {"engine": "lightgbm"}}}
        form_payload = _form_payload()
        response = self.client.post(
            reverse("trading:enqueue_training_task"),
            data=json.dumps({"params": form_payload, "tickers": ["AAPL", "MSFT"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["task_id"].startswith("sync-"))
        self.assertEqual(data["state"], "SUCCESS")
        self.assertIn("result", data)
        self.assertIn("best", data["result"]["result"])
        mock_execute.assert_called_once()

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    @mock.patch("trading.task_queue.execute_rl_job")
    def test_enqueue_rl_task_sync_result(self, mock_execute):
        mock_execute.return_value = {"history_id": "hist-rl", "result": {"playbook": {"available": True}}}
        response = self.client.post(
            reverse("trading:enqueue_rl_task"),
            data=json.dumps({"params": _form_payload()}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["task_id"].startswith("sync-"))
        self.assertEqual(payload["state"], "SUCCESS")
        self.assertEqual(payload["result"]["history_id"], "hist-rl")
        mock_execute.assert_called_once()

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    @mock.patch("trading.task_queue.execute_backtest")
    def test_api_v1_backtest_endpoint(self, mock_execute):
        mock_execute.return_value = {"history_id": "api-history"}
        response = self.client.post(
            reverse("trading:api_v1_backtest_tasks"),
            data=json.dumps(_form_payload()),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["result"]["history_id"], "api-history")
        self.assertNotIn("result", payload["result"])

    def test_api_v1_task_status_endpoint(self):
        response = self.client.get(reverse("trading:api_v1_task_status", kwargs={"task_id": "sync-test"}))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["task_id"], "sync-test")
        self.assertEqual(data["state"], "SUCCESS")


class ExecuteBacktestReturnTests(SimpleTestCase):
    @override_settings(TASK_RETURN_SNAPSHOT=False)
    @mock.patch("trading.tasks._persist_history")
    @mock.patch("trading.tasks.run_quant_pipeline")
    @mock.patch("trading.tasks._deserialize_strategy_input")
    def test_execute_backtest_default_returns_history_only(self, mock_deserialize, mock_run, mock_persist):
        mock_deserialize.return_value = SimpleNamespace(user_id="user-1")
        mock_run.return_value = {"stats": {"alpha": 0.8}}
        mock_persist.return_value = "history-99"

        from trading import tasks as task_module

        result = task_module.execute_backtest({})

        self.assertEqual(result, {"history_id": "history-99"})

    @override_settings(TASK_RETURN_SNAPSHOT=True)
    @mock.patch("trading.tasks.sanitize_snapshot")
    @mock.patch("trading.tasks._persist_history")
    @mock.patch("trading.tasks.run_quant_pipeline")
    @mock.patch("trading.tasks._deserialize_strategy_input")
    def test_execute_backtest_can_return_snapshot(self, mock_deserialize, mock_run, mock_persist, mock_sanitize):
        mock_deserialize.return_value = SimpleNamespace(user_id="user-2")
        mock_run.return_value = {"stats": {"alpha": 0.9}}
        mock_persist.return_value = "history-100"
        mock_sanitize.return_value = {"alpha": 0.9}

        from trading import tasks as task_module

        result = task_module.execute_backtest({})

        self.assertEqual(result, {"history_id": "history-100", "result": {"alpha": 0.9}})


class ApiV1ScreenerTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="screener", password="pass123")
        self.client.force_login(self.user)

    @mock.patch("trading.screener.fetch_page")
    def test_screener_v1_endpoint(self, mock_fetch):
        mock_fetch.return_value = {
            "rows": [{"symbol": "AAPL"}],
            "offset": 0,
            "size": 20,
            "has_more": False,
            "total": 1,
        }
        response = self.client.get(reverse("trading:api_v1_screener"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["rows"][0]["symbol"], "AAPL")
