from __future__ import annotations

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from trading import tasks as trading_tasks


@override_settings(MARKET_RANKINGS_BACKGROUND_ONLY=True)
class MarketRankingsStaleRefreshTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="rankings_stale", password="secret123")
        self.client.force_login(self.user)
        self.market_url = reverse("trading:market_insights_data")
        self._trigger_patcher = mock.patch(
            "trading.views.market._maybe_trigger_snapshot_refresh_nonblocking",
            return_value=None,
        )
        self._trigger_patcher.start()

    def tearDown(self):
        self._trigger_patcher.stop()
        super().tearDown()

    def test_non_1d_background_only_does_not_run_sync_universe_window_refresh(self):
        with mock.patch(
            "trading.views.market.resolve_market_provider",
            return_value="massive",
        ), mock.patch(
            "trading.views.market.has_market_data_credentials",
            return_value=True,
        ), mock.patch(
            "trading.views.market._load_timeframe_snapshot_rows",
            return_value=None,
        ), mock.patch(
            "trading.views.market._resolve_universe_window_rankings",
        ) as resolve_window_rankings, mock.patch(
            "trading.views.market._snapshot_refresh_meta",
            return_value={"progress": {"status": "running", "chunks_completed": 2, "total_chunks": 10}},
        ), mock.patch(
            "trading.views.market._resolve_building_snapshot_payload",
            return_value={
                "status": "running",
                "source": "massive",
                "chunks_completed": 2,
                "total_chunks": 10,
                "progress": {"status": "running", "chunks_completed": 2, "total_chunks": 10},
            },
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value={},
        ):
            response = self.client.get(self.market_url, {"list": "most_active", "timeframe": "1mo"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("items"), [])
        self.assertEqual(payload.get("data_state"), "building")
        self.assertEqual(payload.get("snapshot_state", {}).get("served_from"), "none")
        resolve_window_rankings.assert_not_called()

    def test_non_1d_uses_active_snapshot_while_background_refresh_is_running(self):
        timeframe_snapshot_rows = [
            {
                "symbol": "AAA",
                "price": 10.0,
                "change_pct_period": 5.2,
                "change_pct_day": 0.8,
                "volume": 1200.0,
                "dollar_volume": 12000.0,
                "open": 9.7,
                "prev_close": 9.5,
                "range_pct": 3.1,
            },
            {
                "symbol": "BBB",
                "price": 8.0,
                "change_pct_period": -3.6,
                "change_pct_day": -1.1,
                "volume": 900.0,
                "dollar_volume": 7200.0,
                "open": 8.4,
                "prev_close": 8.6,
                "range_pct": 4.0,
            },
            {
                "symbol": "CCC",
                "price": 12.0,
                "change_pct_period": 1.2,
                "change_pct_day": 0.6,
                "volume": 800.0,
                "dollar_volume": 9600.0,
                "open": 11.7,
                "prev_close": 11.4,
                "range_pct": 2.8,
            },
        ]
        with mock.patch(
            "trading.views.market.resolve_market_provider",
            return_value="massive",
        ), mock.patch(
            "trading.views.market.has_market_data_credentials",
            return_value=True,
        ), mock.patch(
            "trading.views.market._load_timeframe_snapshot_rows",
            return_value=timeframe_snapshot_rows,
        ), mock.patch(
            "trading.views.market._snapshot_refresh_meta",
            return_value={"progress": {"status": "running", "chunks_completed": 6, "total_chunks": 10}},
        ), mock.patch(
            "trading.views.market._resolve_building_snapshot_payload",
            return_value={
                "status": "running",
                "source": "massive",
                "chunks_completed": 6,
                "total_chunks": 10,
                "progress": {"status": "running", "chunks_completed": 6, "total_chunks": 10},
            },
        ), mock.patch(
            "trading.views.market._resolve_active_snapshot_payload",
            return_value={"generated_at": 1_707_000_000, "generated_ts": 1_707_000_000},
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value={},
        ):
            response = self.client.get(self.market_url, {"list": "gainers", "timeframe": "1mo"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        items = payload.get("items", [])
        self.assertTrue(items)
        self.assertEqual(items[0].get("symbol"), "AAA")
        self.assertEqual(payload.get("ranking_timeframe", {}).get("key"), "1mo")
        self.assertEqual(payload.get("snapshot_state", {}).get("served_from"), "building_fallback")
        self.assertEqual(payload.get("data_state"), "stale")


@override_settings(
    MARKET_RANKINGS_CONTINUOUS_LOOP=True,
    CELERY_TASK_ALWAYS_EAGER=False,
    MARKET_RANKINGS_LOOP_DELAY_SECONDS=4,
)
class MarketRankingsLoopTaskTests(TestCase):
    def test_refresh_task_schedules_next_round_on_complete(self):
        with mock.patch(
            "trading.tasks._resolve_snapshot_user_id",
            return_value="6",
        ), mock.patch(
            "trading.views.market.refresh_snapshot_rankings",
            return_value={"status": "complete"},
        ), mock.patch.object(
            trading_tasks.refresh_market_snapshot_rankings,
            "apply_async",
        ) as apply_async:
            payload = trading_tasks.refresh_market_snapshot_rankings()

        self.assertEqual(payload.get("status"), "complete")
        apply_async.assert_called_once_with(countdown=4)

    @override_settings(MARKET_RANKINGS_REFRESH_SECONDS=900)
    def test_refresh_task_backs_off_on_missing_credentials(self):
        with mock.patch(
            "trading.tasks._resolve_snapshot_user_id",
            return_value=None,
        ), mock.patch(
            "trading.views.market.refresh_snapshot_rankings",
            return_value={"status": "missing_credentials"},
        ), mock.patch.object(
            trading_tasks.refresh_market_snapshot_rankings,
            "apply_async",
        ) as apply_async:
            trading_tasks.refresh_market_snapshot_rankings()

        apply_async.assert_called_once_with(countdown=300)

    def test_trigger_market_snapshot_refresh_prefers_celery(self):
        with mock.patch.object(
            trading_tasks.refresh_market_snapshot_rankings,
            "apply_async",
            return_value=None,
        ) as apply_async, mock.patch(
            "trading.views.market.read_state",
            return_value={},
        ), mock.patch(
            "trading.views.market._resolve_building_snapshot_payload",
            return_value=None,
        ), mock.patch("trading.tasks.threading.Thread.start") as thread_start, mock.patch(
            "trading.tasks._SNAPSHOT_TRIGGER_THREAD",
            None,
        ), mock.patch(
            "trading.tasks._SNAPSHOT_TRIGGER_LAST_TS",
            0.0,
        ):
            result = trading_tasks.trigger_market_snapshot_refresh(user_id="6")

        self.assertEqual(result.get("via"), "celery")
        self.assertEqual(result.get("status"), "queued")
        apply_async.assert_called_once_with(kwargs={"user_id": "6"})
        thread_start.assert_not_called()

    def test_trigger_market_snapshot_refresh_falls_back_to_thread(self):
        with mock.patch.object(
            trading_tasks.refresh_market_snapshot_rankings,
            "apply_async",
            side_effect=RuntimeError("broker_down"),
        ) as apply_async, mock.patch(
            "trading.views.market.read_state",
            return_value={},
        ), mock.patch(
            "trading.views.market._resolve_building_snapshot_payload",
            return_value=None,
        ), mock.patch("trading.tasks.threading.Thread.start") as thread_start, mock.patch(
            "trading.tasks._SNAPSHOT_TRIGGER_THREAD",
            None,
        ), mock.patch(
            "trading.tasks._SNAPSHOT_TRIGGER_LAST_TS",
            0.0,
        ):
            result = trading_tasks.trigger_market_snapshot_refresh(user_id="6")

        self.assertEqual(result.get("via"), "thread")
        self.assertEqual(result.get("status"), "queued")
        apply_async.assert_called_once_with(kwargs={"user_id": "6"})
        thread_start.assert_called_once()
