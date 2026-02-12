from __future__ import annotations

import time

from django.test import SimpleTestCase

from trading.runtime.execution import OrderResult
from trading.runtime.pipeline import LiveTradingPipeline
from trading.runtime.signals import CombinedSignal, SignalAction


class _DummyPosition:
    def __init__(self, symbol: str, qty: float):
        self.symbol = symbol
        self.qty = qty


class _DummyAccountState:
    def __init__(self, equity: float):
        self.equity = equity


class _DummyAccountManager:
    def get_account_state(self):
        return _DummyAccountState(1000.0)

    def list_positions(self):
        return [_DummyPosition("AAPL", 1.0)]


class _DummyCombiner:
    def combine_signals(self, raw_signals, timestamp=None):
        return raw_signals


class _DummyRiskManager:
    def apply_risk_controls(self, signal):
        return signal


class _BlockingRiskManager(_DummyRiskManager):
    def check_kill_switch(self, _equity):
        return True, "max_daily_loss"

    def snapshot(self):
        return {"reason": "max_daily_loss"}


class _DummyExecutionClient:
    def __init__(self):
        self.calls = 0

    def submit_order(self, order):
        self.calls += 1
        return OrderResult(order_id="oid-1", status="accepted")


class RuntimePipelineTests(SimpleTestCase):
    def test_execute_signals_throttles_without_name_error(self):
        execution = _DummyExecutionClient()
        pipeline = LiveTradingPipeline(
            strategies=[],
            combiner=_DummyCombiner(),
            risk_manager=_DummyRiskManager(),
            execution=execution,
            account=_DummyAccountManager(),
            execution_config={"max_orders_per_minute": 1},
        )
        pipeline._order_timestamps.append(time.time())
        signals = [
            CombinedSignal(
                symbol="AAPL",
                weight=0.25,
                action=SignalAction.BUY,
                confidence=0.9,
            )
        ]

        results = pipeline.execute_signals(signals)
        self.assertEqual(len(results), 1)
        self.assertEqual(execution.calls, 0)
        order = results[0].get("order")
        self.assertIsInstance(order, OrderResult)
        self.assertEqual(order.status, "throttled")

    def test_execute_signals_respects_risk_kill_switch(self):
        execution = _DummyExecutionClient()
        pipeline = LiveTradingPipeline(
            strategies=[],
            combiner=_DummyCombiner(),
            risk_manager=_BlockingRiskManager(),
            execution=execution,
            account=_DummyAccountManager(),
            execution_config={"max_orders_per_minute": 5},
        )
        signals = [
            CombinedSignal(
                symbol="MSFT",
                weight=0.3,
                action=SignalAction.BUY,
                confidence=0.8,
            )
        ]

        results = pipeline.execute_signals(signals)
        self.assertEqual(execution.calls, 0)
        self.assertEqual(len(results), 1)
        order = results[0].get("order")
        self.assertIsInstance(order, OrderResult)
        self.assertEqual(order.status, "risk_blocked:max_daily_loss")
