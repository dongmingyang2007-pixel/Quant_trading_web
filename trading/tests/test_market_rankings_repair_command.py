from __future__ import annotations

from unittest import mock

from django.core.management import call_command
from django.test import SimpleTestCase


class RepairMarketRankingsStateCommandTests(SimpleTestCase):
    @mock.patch("trading.management.commands.repair_market_rankings_state.trigger_market_snapshot_refresh")
    @mock.patch("trading.management.commands.repair_market_rankings_state.market_views._resolve_active_snapshot_payload")
    @mock.patch("trading.management.commands.repair_market_rankings_state.write_state")
    @mock.patch("trading.management.commands.repair_market_rankings_state.read_state")
    @mock.patch("trading.management.commands.repair_market_rankings_state.resolve_market_provider")
    def test_marks_stalled_running_state_and_triggers_refresh(
        self,
        resolve_provider,
        read_state,
        write_state,
        resolve_active,
        trigger_refresh,
    ):
        resolve_provider.return_value = "massive"
        trigger_refresh.return_value = {"status": "queued", "via": "thread"}
        resolve_active.return_value = None

        stale_running = {
            "status": "running",
            "source": "massive",
            "started_ts": 1.0,
            "updated_ts": 1.0,
            "chunks_completed": 3,
            "total_chunks": 10,
        }

        def _read_state(name, default=None):
            if name.endswith("_building.json") or name == "market_rankings_snapshot_building.json":
                return dict(stale_running)
            return default

        read_state.side_effect = _read_state

        with mock.patch("trading.management.commands.repair_market_rankings_state.time.time", return_value=10_000.0):
            call_command("repair_market_rankings_state", "--user-id", "6")

        self.assertTrue(write_state.called)
        calls = write_state.call_args_list
        stalled_payloads = [call.args[1] for call in calls if isinstance(call.args[1], dict)]
        self.assertTrue(any(payload.get("status") == "stalled" for payload in stalled_payloads))
        trigger_refresh.assert_called_once_with(user_id="6")
