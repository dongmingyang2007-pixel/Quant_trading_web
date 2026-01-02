from __future__ import annotations

import json
from types import SimpleNamespace

from django.contrib.auth import get_user_model
from django.test import TestCase, SimpleTestCase, override_settings
from django.core.cache import cache
from django.urls import reverse
from unittest import mock

from trading.models import TaskExecution
from trading.task_queue import get_task_status, submit_backtest_task


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

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True, TASK_RETURN_SNAPSHOT=False, CELERY_BROKER_URL="memory://")
    @mock.patch("trading.task_queue.LOCAL_EXECUTOR")
    @mock.patch("trading.task_queue.execute_backtest")
    def test_enqueue_backtest_task_returns_local_task(self, mock_execute, mock_executor):
        mock_executor.submit.return_value = None
        mock_execute.return_value = {"history_id": "history-1"}
        response = self.client.post(
            reverse("trading:enqueue_backtest_task"),
            data=json.dumps({"params": _form_payload()}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["task_id"].startswith("local-"))
        self.assertEqual(payload["state"], "PENDING")
        self.assertNotIn("result", payload)
        mock_executor.submit.assert_called_once()

    @override_settings(CELERY_TASK_ALWAYS_EAGER=False, CELERY_BROKER_URL="redis://localhost:6379/0")
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

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True, CELERY_BROKER_URL="memory://")
    @mock.patch("trading.task_queue.LOCAL_EXECUTOR")
    @mock.patch("trading.task_queue.execute_backtest")
    def test_api_v1_backtest_endpoint(self, mock_execute, mock_executor):
        mock_executor.submit.return_value = None
        mock_execute.return_value = {"history_id": "api-history"}
        response = self.client.post(
            reverse("trading:api_v1_backtest_tasks"),
            data=json.dumps(_form_payload()),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertTrue(payload["task_id"].startswith("local-"))

    @override_settings(CELERY_TASK_ALWAYS_EAGER=False)
    @mock.patch("trading.api.views_v1.submit_backtest_task")
    def test_api_v1_backtest_dedupes_client_request_id(self, mock_submit):
        cache.clear()
        dummy_task = SimpleNamespace(id="celery-88", state="PENDING")
        mock_submit.return_value = dummy_task
        payload = _form_payload()
        payload["client_request_id"] = "client-abc"
        url = reverse("trading:api_v1_backtest_tasks")
        first = self.client.post(url, data=json.dumps(payload), content_type="application/json")
        second = self.client.post(url, data=json.dumps(payload), content_type="application/json")
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(second.json()["task_id"], "celery-88")
        mock_submit.assert_called_once()

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True, CELERY_BROKER_URL="memory://")
    @mock.patch("trading.api.views_v1.submit_backtest_task")
    def test_api_v1_backtest_accepts_advanced_fields(self, mock_submit):
        dummy_task = SimpleNamespace(id="local-adv", state="PENDING")
        mock_submit.return_value = dummy_task
        payload = _form_payload()
        payload.update(
            {
                "strategy_engine": "rl_policy",
                "risk_profile": "aggressive",
                "short_window": 20,
                "long_window": 80,
                "rsi_period": 14,
                "volatility_target": 0.18,
                "transaction_cost_bps": 9.5,
                "slippage_bps": 6.0,
                "min_holding_days": 2,
                "entry_threshold": 0.62,
                "exit_threshold": 0.38,
                "optimize_thresholds": False,
                "train_window": 360,
                "test_window": 30,
                "val_ratio": 0.25,
                "embargo_days": 5,
                "auto_apply_best_config": False,
                "enable_hyperopt": True,
                "hyperopt_trials": 12,
                "hyperopt_timeout": 120,
                "max_leverage": 2.2,
                "max_drawdown_stop": 0.2,
                "daily_exposure_limit": 1.1,
                "allow_short": False,
                "execution_delay_days": 3,
            }
        )
        response = self.client.post(
            reverse("trading:api_v1_backtest_tasks"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 202)
        submitted = mock_submit.call_args[0][0]
        self.assertEqual(submitted["strategy_engine"], "rl_policy")
        self.assertEqual(submitted["risk_profile"], "aggressive")
        self.assertEqual(submitted["short_window"], 20)
        self.assertEqual(submitted["long_window"], 80)
        self.assertEqual(submitted["rsi_period"], 14)
        self.assertEqual(submitted["execution_delay_days"], 3)
        self.assertFalse(submitted["allow_short"])

    def test_api_v1_task_status_endpoint(self):
        response = self.client.get(reverse("trading:api_v1_task_status", kwargs={"task_id": "sync-test"}))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["task_id"], "sync-test")
        self.assertEqual(data["state"], "SUCCESS")


class LocalTaskExecutionTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="local", password="secret123")

    def test_task_execution_crud(self):
        execution = TaskExecution.objects.create(
            task_id="local-crud",
            user=self.user,
            kind="backtest",
            state="PENDING",
            meta={"progress": 0},
        )
        self.assertEqual(TaskExecution.objects.count(), 1)
        TaskExecution.objects.filter(task_id="local-crud").update(state="PROGRESS", meta={"progress": 40})
        execution.refresh_from_db()
        self.assertEqual(execution.state, "PROGRESS")
        self.assertEqual(execution.meta["progress"], 40)
        execution.delete()
        self.assertEqual(TaskExecution.objects.count(), 0)

    def test_get_task_status_for_local_task(self):
        TaskExecution.objects.create(
            task_id="local-status",
            user=self.user,
            kind="backtest",
            state="PROGRESS",
            meta={"progress": 55, "stage": "running_backtest"},
        )
        payload = get_task_status("local-status")
        self.assertEqual(payload["task_id"], "local-status")
        self.assertEqual(payload["state"], "PROGRESS")
        self.assertEqual(payload["meta"]["progress"], 55)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=False, CELERY_BROKER_URL="memory://")
    @mock.patch("trading.task_queue.LOCAL_EXECUTOR")
    def test_submit_backtest_task_falls_back_to_local(self, mock_executor):
        mock_executor.submit.return_value = None
        job = submit_backtest_task({"user_id": self.user.id, "ticker": "AAPL"})
        self.assertTrue(job.id.startswith("local-"))
        execution = TaskExecution.objects.get(task_id=job.id)
        self.assertEqual(execution.state, "PENDING")
        self.assertEqual(execution.kind, "backtest")


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
