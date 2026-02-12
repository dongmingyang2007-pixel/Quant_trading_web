from django.test import SimpleTestCase

from trading.runtime.risk import RiskLimits, RiskManager


class RuntimeRiskManagerTests(SimpleTestCase):
    def test_kill_switch_triggers_on_daily_loss(self):
        manager = RiskManager(
            RiskLimits(
                max_position_weight=0.2,
                max_leverage=1.0,
                max_daily_loss_pct=0.02,
                kill_switch_cooldown_seconds=300,
            )
        )
        blocked, reason = manager.check_kill_switch(100.0, now_ts=1_700_000_000.0)
        self.assertFalse(blocked)
        self.assertIsNone(reason)

        blocked, reason = manager.check_kill_switch(97.0, now_ts=1_700_000_010.0)
        self.assertTrue(blocked)
        self.assertEqual(reason, "max_daily_loss")

    def test_kill_switch_cooldown_blocks_until_expiry(self):
        manager = RiskManager(
            RiskLimits(
                max_position_weight=0.2,
                max_leverage=1.0,
                max_daily_loss_pct=0.01,
                kill_switch_cooldown_seconds=120,
            )
        )
        manager.check_kill_switch(100.0, now_ts=1_700_000_000.0)
        blocked, reason = manager.check_kill_switch(98.5, now_ts=1_700_000_001.0)
        self.assertTrue(blocked)
        self.assertEqual(reason, "max_daily_loss")

        blocked, reason = manager.check_kill_switch(99.8, now_ts=1_700_000_050.0)
        self.assertTrue(blocked)
        self.assertEqual(reason, "daily_loss_cooldown")
