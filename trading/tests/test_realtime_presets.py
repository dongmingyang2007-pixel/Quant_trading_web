from django.test import SimpleTestCase

from trading.realtime.presets import (
    build_retail_short_term_template,
    list_realtime_templates,
)


class RealtimePresetTests(SimpleTestCase):
    def test_retail_minute_template_contains_short_term_risk_guard(self):
        payload = build_retail_short_term_template("retail_minute")
        risk = payload["trading"]["risk"]
        self.assertAlmostEqual(risk["max_daily_loss_pct"], 0.025)
        self.assertEqual(risk["kill_switch_cooldown_seconds"], 1800)

    def test_retail_second_template_is_more_aggressive_but_tighter_risk(self):
        payload = build_retail_short_term_template("retail_second")
        self.assertEqual(payload["engine"]["bar_aggregate_seconds"], 1)
        self.assertEqual(payload["trading"]["min_trade_interval_seconds"], 5)
        self.assertAlmostEqual(payload["trading"]["risk"]["max_daily_loss_pct"], 0.015)

    def test_template_catalog_contains_two_short_term_profiles(self):
        items = list_realtime_templates()
        keys = {item["key"] for item in items}
        self.assertIn("retail_minute", keys)
        self.assertIn("retail_second", keys)
