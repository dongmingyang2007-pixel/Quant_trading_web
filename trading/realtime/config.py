from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .storage import resolve_state_dir
from ..file_utils import read_json


@dataclass(slots=True)
class UniverseConfig:
    max_symbols: int = 1200
    top_n: int = 1000
    min_price: float = 2.0
    min_dollar_volume: float = 5_000_000.0
    min_volume: int = 200_000
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "dollar_volume": 1.0,
            "change_pct": 0.15,
            "volume": 0.05,
        }
    )
    symbols: list[str] | None = None


@dataclass(slots=True)
class FocusConfig:
    size: int = 200
    max_churn_per_refresh: int = 20
    min_residence_seconds: int = 300


@dataclass(slots=True)
class EngineConfig:
    universe_refresh_seconds: int = 900
    focus_refresh_seconds: int = 60
    bar_timeframe: str = "1Min"
    bar_limit: int = 300
    feed: str = "sip"
    log_every_seconds: int = 30
    stream_enabled: bool = True
    stream_trades: bool = True
    stream_quotes: bool = True
    stream_url: str | None = None
    bar_interval_seconds: int = 1
    bar_aggregate_seconds: int = 5
    stale_seconds: float = 2.5
    reconnect_seconds: int = 5


@dataclass(slots=True)
class SignalConfig:
    lookback_bars: int = 3
    entry_threshold: float = 0.002
    exit_threshold: float = 0.001
    min_volume: float = 10_000.0
    max_spread_bps: float = 25.0


@dataclass(slots=True)
class RealtimeConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)


DEFAULT_CONFIG_NAME = "realtime_profile.json"


def _coerce_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _coerce_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _merge_universe(config: UniverseConfig, payload: Mapping[str, Any]) -> UniverseConfig:
    config.max_symbols = _coerce_int(payload.get("max_symbols"), config.max_symbols, minimum=100, maximum=5000)
    config.top_n = _coerce_int(payload.get("top_n"), config.top_n, minimum=50, maximum=config.max_symbols)
    config.min_price = _coerce_float(payload.get("min_price"), config.min_price, minimum=0.1)
    config.min_dollar_volume = _coerce_float(payload.get("min_dollar_volume"), config.min_dollar_volume, minimum=0)
    config.min_volume = _coerce_int(payload.get("min_volume"), config.min_volume, minimum=0)
    raw_symbols = payload.get("symbols")
    if isinstance(raw_symbols, list):
        config.symbols = [str(sym).strip().upper() for sym in raw_symbols if str(sym).strip()]
    weights = payload.get("score_weights")
    if isinstance(weights, dict):
        merged = dict(config.score_weights)
        for key, value in weights.items():
            try:
                merged[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        config.score_weights = merged
    return config


def _merge_focus(config: FocusConfig, payload: Mapping[str, Any]) -> FocusConfig:
    config.size = _coerce_int(payload.get("size"), config.size, minimum=20, maximum=500)
    config.max_churn_per_refresh = _coerce_int(
        payload.get("max_churn_per_refresh"),
        config.max_churn_per_refresh,
        minimum=1,
        maximum=config.size,
    )
    config.min_residence_seconds = _coerce_int(
        payload.get("min_residence_seconds"),
        config.min_residence_seconds,
        minimum=0,
        maximum=24 * 3600,
    )
    return config


def _merge_engine(config: EngineConfig, payload: Mapping[str, Any]) -> EngineConfig:
    config.universe_refresh_seconds = _coerce_int(
        payload.get("universe_refresh_seconds"),
        config.universe_refresh_seconds,
        minimum=60,
        maximum=24 * 3600,
    )
    config.focus_refresh_seconds = _coerce_int(
        payload.get("focus_refresh_seconds"),
        config.focus_refresh_seconds,
        minimum=10,
        maximum=6 * 3600,
    )
    config.bar_limit = _coerce_int(payload.get("bar_limit"), config.bar_limit, minimum=10, maximum=2000)
    feed = payload.get("feed")
    if isinstance(feed, str) and feed.strip():
        config.feed = feed.strip()
    timeframe = payload.get("bar_timeframe")
    if isinstance(timeframe, str) and timeframe.strip():
        config.bar_timeframe = timeframe.strip()
    config.log_every_seconds = _coerce_int(
        payload.get("log_every_seconds"),
        config.log_every_seconds,
        minimum=5,
        maximum=3600,
    )
    config.stream_enabled = _coerce_bool(payload.get("stream_enabled"), config.stream_enabled)
    config.stream_trades = _coerce_bool(payload.get("stream_trades"), config.stream_trades)
    config.stream_quotes = _coerce_bool(payload.get("stream_quotes"), config.stream_quotes)
    stream_url = payload.get("stream_url")
    if isinstance(stream_url, str) and stream_url.strip():
        config.stream_url = stream_url.strip()
    config.bar_interval_seconds = _coerce_int(
        payload.get("bar_interval_seconds"),
        config.bar_interval_seconds,
        minimum=1,
        maximum=60,
    )
    config.bar_aggregate_seconds = _coerce_int(
        payload.get("bar_aggregate_seconds"),
        config.bar_aggregate_seconds,
        minimum=config.bar_interval_seconds,
        maximum=300,
    )
    config.stale_seconds = _coerce_float(
        payload.get("stale_seconds"),
        config.stale_seconds,
        minimum=0.1,
        maximum=30.0,
    )
    config.reconnect_seconds = _coerce_int(
        payload.get("reconnect_seconds"),
        config.reconnect_seconds,
        minimum=1,
        maximum=120,
    )
    return config


def _merge_signals(config: SignalConfig, payload: Mapping[str, Any]) -> SignalConfig:
    config.lookback_bars = _coerce_int(payload.get("lookback_bars"), config.lookback_bars, minimum=1, maximum=20)
    config.entry_threshold = _coerce_float(
        payload.get("entry_threshold"),
        config.entry_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    config.exit_threshold = _coerce_float(
        payload.get("exit_threshold"),
        config.exit_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    config.min_volume = _coerce_float(payload.get("min_volume"), config.min_volume, minimum=0.0)
    config.max_spread_bps = _coerce_float(
        payload.get("max_spread_bps"),
        config.max_spread_bps,
        minimum=0.0,
        maximum=500.0,
    )
    return config


def load_realtime_config_from_payload(payload: Mapping[str, Any] | None) -> RealtimeConfig:
    config = RealtimeConfig()
    if not isinstance(payload, Mapping):
        return config
    universe_payload = payload.get("universe")
    if isinstance(universe_payload, dict):
        config.universe = _merge_universe(config.universe, universe_payload)
    focus_payload = payload.get("focus")
    if isinstance(focus_payload, dict):
        config.focus = _merge_focus(config.focus, focus_payload)
    engine_payload = payload.get("engine")
    if isinstance(engine_payload, dict):
        config.engine = _merge_engine(config.engine, engine_payload)
    signals_payload = payload.get("signals")
    if isinstance(signals_payload, dict):
        config.signals = _merge_signals(config.signals, signals_payload)
    return config


def normalize_realtime_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    config = load_realtime_config_from_payload(payload)
    return asdict(config)


def load_realtime_config(path: Path | None = None) -> RealtimeConfig:
    path = path or (resolve_state_dir() / DEFAULT_CONFIG_NAME)
    raw = read_json(path, default={})
    return load_realtime_config_from_payload(raw if isinstance(raw, dict) else {})
