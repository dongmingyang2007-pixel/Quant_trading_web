from django.test import SimpleTestCase

from trading.realtime.config import load_realtime_config_from_payload


class RealtimeTradingConfigTests(SimpleTestCase):
    def test_trading_config_parses_strategies(self):
        payload = {
            "trading": {
                "enabled": True,
                "mode": "paper",
                "min_trade_interval_seconds": 15,
                "strategies": [
                    {"name": "momentum", "enabled": True, "weight": 0.6, "params": {"lookback_bars": 4}},
                    {"name": "mean_reversion", "enabled": False, "weight": 0.4},
                ],
                "combiner": {"method": "weighted_avg"},
                "risk": {"max_position_weight": 0.2},
                "execution": {"enabled": False, "max_orders_per_minute": 50},
            }
        }
        config = load_realtime_config_from_payload(payload)
        self.assertTrue(config.trading.enabled)
        self.assertEqual(config.trading.mode, "paper")
        self.assertEqual(config.trading.min_trade_interval_seconds, 15)
        self.assertEqual(config.trading.strategies[0].name, "momentum")
        self.assertEqual(config.trading.strategies[0].weight, 0.6)
        self.assertEqual(config.trading.strategies[0].params.get("lookback_bars"), 4)
        self.assertEqual(config.trading.combiner.method, "weighted_avg")
        self.assertEqual(config.trading.risk.max_position_weight, 0.2)
        self.assertEqual(config.trading.execution.max_orders_per_minute, 50)
        self.assertFalse(config.trading.execution.dry_run)
