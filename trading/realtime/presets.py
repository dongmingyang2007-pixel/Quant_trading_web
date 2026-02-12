from __future__ import annotations

from typing import Any

from .config import normalize_realtime_payload


SUPPORTED_REALTIME_TEMPLATE_KEYS = {"retail_minute", "retail_second"}


def build_retail_short_term_template(template_key: str) -> dict[str, Any]:
    key = str(template_key or "").strip().lower()
    if key not in SUPPORTED_REALTIME_TEMPLATE_KEYS:
        key = "retail_minute"

    payload = normalize_realtime_payload({})
    engine = payload.setdefault("engine", {})
    signals = payload.setdefault("signals", {})
    trading = payload.setdefault("trading", {})
    risk = trading.setdefault("risk", {})
    execution = trading.setdefault("execution", {})

    # Base configuration shared by both short-term modes.
    engine.update(
        {
            "feed": "sip",
            "stream_enabled": True,
            "stream_trades": True,
            "stream_quotes": True,
            "universe_refresh_seconds": 300,
            "focus_refresh_seconds": 20,
            "bar_timeframe": "1Min",
            "bar_limit": 240,
            "bar_interval_seconds": 1,
            "bar_aggregate_seconds": 5,
            "stale_seconds": 2.5,
            "reconnect_seconds": 5,
        }
    )
    signals.update(
        {
            "lookback_bars": 4,
            "entry_threshold": 0.0012,
            "exit_threshold": 0.0006,
            "min_volume": 50_000,
            "max_spread_bps": 20.0,
        }
    )
    trading.update(
        {
            "enabled": True,
            "mode": "paper",
            "min_trade_interval_seconds": 20,
            "strategies": [
                {"name": "momentum", "enabled": True, "weight": 0.6, "params": {"lookback_bars": 4, "entry_threshold": 0.0012, "exit_threshold": 0.0006}},
                {"name": "mean_reversion", "enabled": True, "weight": 0.4, "params": {"rsi_period": 9, "upper": 68, "lower": 32}},
            ],
            "combiner": {"method": "weighted_avg", "weights": {"momentum": 0.6, "mean_reversion": 0.4}},
        }
    )
    risk.update(
        {
            "max_position_weight": 0.12,
            "max_leverage": 1.6,
            "min_confidence": 0.55,
            "max_daily_loss_pct": 0.025,
            "kill_switch_cooldown_seconds": 1800,
        }
    )
    execution.update(
        {
            "enabled": True,
            "dry_run": False,
            "order_type": "market",
            "time_in_force": "day",
            "max_orders_per_minute": 120,
        }
    )

    if key == "retail_second":
        engine.update(
            {
                "focus_refresh_seconds": 10,
                "bar_aggregate_seconds": 1,
                "stale_seconds": 1.5,
            }
        )
        signals.update(
            {
                "lookback_bars": 6,
                "entry_threshold": 0.0018,
                "exit_threshold": 0.0009,
                "min_volume": 80_000,
                "max_spread_bps": 12.0,
            }
        )
        trading.update(
            {
                "min_trade_interval_seconds": 5,
                "strategies": [
                    {"name": "momentum", "enabled": True, "weight": 0.7, "params": {"lookback_bars": 6, "entry_threshold": 0.0018, "exit_threshold": 0.0009}},
                    {"name": "mean_reversion", "enabled": True, "weight": 0.3, "params": {"rsi_period": 7, "upper": 70, "lower": 30}},
                ],
                "combiner": {"method": "weighted_avg", "weights": {"momentum": 0.7, "mean_reversion": 0.3}},
            }
        )
        risk.update(
            {
                "max_position_weight": 0.08,
                "max_leverage": 1.25,
                "min_confidence": 0.65,
                "max_daily_loss_pct": 0.015,
                "kill_switch_cooldown_seconds": 2700,
            }
        )
        execution.update({"max_orders_per_minute": 240})

    return payload


def list_realtime_templates() -> list[dict[str, Any]]:
    return [
        {
            "key": "retail_minute",
            "label_zh": "分钟级短线模板",
            "label_en": "Retail minute template",
            "description_zh": "默认短线交易模板，强调分钟级节奏与日内风控。",
            "description_en": "Default short-term profile focused on minute-level rhythm and intraday risk control.",
        },
        {
            "key": "retail_second",
            "label_zh": "秒级超短模板",
            "label_en": "Retail second template",
            "description_zh": "更激进的秒级实验模板，对延迟与滑点更敏感。",
            "description_en": "More aggressive second-level template with higher sensitivity to latency and slippage.",
        },
    ]
