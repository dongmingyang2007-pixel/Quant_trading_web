from __future__ import annotations

from typing import Any

from .config import normalize_realtime_payload

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator, model_validator

    PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]
    ValidationError = Exception  # type: ignore[assignment]
    ConfigDict = dict  # type: ignore[assignment]

    def Field(*args, **kwargs):
        return None

    def field_validator(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def model_validator(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    PYDANTIC_AVAILABLE = False


class RealtimePayloadError(ValueError):
    pass


if PYDANTIC_AVAILABLE:

    class UniversePayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        max_symbols: int = Field(1200, ge=100, le=5000)
        top_n: int = Field(1000, ge=50, le=5000)
        min_price: float = Field(2.0, ge=0.1)
        min_dollar_volume: float = Field(5_000_000.0, ge=0.0)
        min_volume: int = Field(200_000, ge=0)
        score_weights: dict[str, float] = Field(
            default_factory=lambda: {
                "dollar_volume": 1.0,
                "change_pct": 0.15,
                "volume": 0.05,
            }
        )
        symbols: list[str] | None = None

        @field_validator("symbols", mode="before")
        @classmethod
        def _normalize_symbols(cls, value):
            if value is None:
                return None
            if isinstance(value, str):
                value = [item.strip() for item in value.split(",")]
            if not isinstance(value, list):
                return None
            cleaned = []
            for item in value:
                text = str(item).strip().upper()
                if text and text not in cleaned:
                    cleaned.append(text)
            return cleaned or None

        @field_validator("score_weights", mode="before")
        @classmethod
        def _normalize_weights(cls, value):
            if not isinstance(value, dict):
                return {
                    "dollar_volume": 1.0,
                    "change_pct": 0.15,
                    "volume": 0.05,
                }
            output: dict[str, float] = {}
            for key, val in value.items():
                try:
                    output[str(key)] = float(val)
                except (TypeError, ValueError):
                    continue
            return output

        @model_validator(mode="after")
        def _validate_limits(self):
            if self.top_n > self.max_symbols:
                raise ValueError("universe.top_n must be <= universe.max_symbols")
            return self


    class FocusPayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        size: int = Field(200, ge=20, le=500)
        max_churn_per_refresh: int = Field(20, ge=1, le=500)
        min_residence_seconds: int = Field(300, ge=0, le=86_400)

        @model_validator(mode="after")
        def _validate_churn(self):
            if self.max_churn_per_refresh > self.size:
                raise ValueError("focus.max_churn_per_refresh must be <= focus.size")
            return self


    class EnginePayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        universe_refresh_seconds: int = Field(900, ge=60, le=86_400)
        focus_refresh_seconds: int = Field(60, ge=10, le=21_600)
        bar_timeframe: str = "1Min"
        bar_limit: int = Field(300, ge=10, le=2000)
        feed: str = "sip"
        log_every_seconds: int = Field(30, ge=5, le=3600)
        stream_enabled: bool = True
        stream_trades: bool = True
        stream_quotes: bool = True
        stream_url: str | None = None
        bar_interval_seconds: int = Field(1, ge=1, le=60)
        bar_aggregate_seconds: int = Field(5, ge=1, le=300)
        stale_seconds: float = Field(2.5, ge=0.1, le=30.0)
        reconnect_seconds: int = Field(5, ge=1, le=120)

        @model_validator(mode="after")
        def _validate_intervals(self):
            if self.bar_aggregate_seconds < self.bar_interval_seconds:
                raise ValueError("engine.bar_aggregate_seconds must be >= engine.bar_interval_seconds")
            return self


    class SignalPayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        lookback_bars: int = Field(3, ge=1, le=20)
        entry_threshold: float = Field(0.002, ge=0.0, le=1.0)
        exit_threshold: float = Field(0.001, ge=0.0, le=1.0)
        min_volume: float = Field(10_000.0, ge=0.0)
        max_spread_bps: float = Field(25.0, ge=0.0, le=500.0)


    class StrategySpecPayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        name: str = "momentum"
        enabled: bool = True
        weight: float = Field(1.0, ge=0.0, le=10.0)
        params: dict[str, Any] = Field(default_factory=dict)

        @field_validator("name", mode="before")
        @classmethod
        def _normalize_name(cls, value):
            text = str(value or "").strip()
            if not text:
                raise ValueError("strategy.name cannot be empty")
            return text


    class CombinerPayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        method: str = "weighted_avg"
        weights: dict[str, float] = Field(default_factory=dict)

        @field_validator("weights", mode="before")
        @classmethod
        def _normalize_weights(cls, value):
            if not isinstance(value, dict):
                return {}
            cleaned: dict[str, float] = {}
            for key, val in value.items():
                try:
                    cleaned[str(key)] = float(val)
                except (TypeError, ValueError):
                    continue
            return cleaned


    class RiskPayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        max_position_weight: float = Field(0.2, ge=0.0, le=1.0)
        max_leverage: float = Field(1.0, ge=0.1, le=10.0)
        min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


    class ExecutionPayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        enabled: bool = True
        dry_run: bool = False
        order_type: str = "market"
        time_in_force: str = "day"
        max_orders_per_minute: int = Field(60, ge=1, le=5000)


    class TradingPayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        enabled: bool = True
        mode: str = "paper"
        min_trade_interval_seconds: int = Field(30, ge=5, le=3600)
        strategies: list[StrategySpecPayload] = Field(default_factory=lambda: [StrategySpecPayload()])
        combiner: CombinerPayload = Field(default_factory=CombinerPayload)
        risk: RiskPayload = Field(default_factory=RiskPayload)
        execution: ExecutionPayload = Field(default_factory=ExecutionPayload)


    class RealtimePayload(BaseModel):
        model_config = ConfigDict(extra="ignore")
        universe: UniversePayload = Field(default_factory=UniversePayload)
        focus: FocusPayload = Field(default_factory=FocusPayload)
        engine: EnginePayload = Field(default_factory=EnginePayload)
        signals: SignalPayload = Field(default_factory=SignalPayload)
        trading: TradingPayload = Field(default_factory=TradingPayload)


def _format_pydantic_error(exc: ValidationError) -> str:
    parts = []
    try:
        for err in exc.errors():
            loc = ".".join(str(item) for item in err.get("loc", []))
            msg = err.get("msg", "")
            parts.append(f"{loc}: {msg}".strip(": "))
    except Exception:
        parts.append(str(exc))
    return "; ".join(parts) or "Invalid payload."


def validate_realtime_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RealtimePayloadError("Payload must be a JSON object.")
    if not PYDANTIC_AVAILABLE:
        return normalize_realtime_payload(payload)
    try:
        model = RealtimePayload.model_validate(payload)
    except ValidationError as exc:
        raise RealtimePayloadError(_format_pydantic_error(exc)) from exc
    return model.model_dump()
