from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from ..observability import record_metric, track_latency
from ..alpaca_data import fetch_stock_snapshots
from .alpaca import fetch_bars_frame
from .bars import BarsProcessor
from .config import RealtimeConfig
from .focus import update_focus
from .signals import SignalEngine
from .stream_registry import acquire_stream, release_stream
from .subscriptions import read_subscription_state
from .storage import append_ndjson, write_state
from .universe import build_universe


@dataclass(slots=True)
class EngineState:
    last_universe_ts: float = 0.0
    last_focus_ts: float = 0.0
    last_log_ts: float = 0.0


def _extract_latest_bars(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return rows
    if isinstance(frame.columns, pd.MultiIndex):
        fields = ["Open", "High", "Low", "Close", "Volume"]
        for symbol in frame.columns.get_level_values(1).unique():
            try:
                sub = frame.xs(symbol, level=1, axis=1)
                if sub.empty:
                    continue
                last = sub.tail(1)
                ts = last.index[-1]
                payload = {"timestamp": ts.isoformat(), "symbol": str(symbol)}
                for field in fields:
                    if field in last.columns:
                        payload[field.lower()] = float(last[field].iloc[-1])
                rows.append(payload)
            except Exception:
                continue
    else:
        last = frame.tail(1)
        if not last.empty:
            ts = last.index[-1]
            payload = {"timestamp": ts.isoformat(), "symbol": ""}
            for field in ("Open", "High", "Low", "Close", "Volume"):
                if field in last.columns:
                    payload[field.lower()] = float(last[field].iloc[-1])
            rows.append(payload)
    return rows


try:  # pragma: no cover - optional dependency
    from .alpaca import AlpacaStreamClient
except Exception:  # pragma: no cover
    AlpacaStreamClient = None  # type: ignore[assignment]


class RealtimeEngine:
    def __init__(self, config: RealtimeConfig, *, user_id: str | None = None) -> None:
        self.config = config
        self.user_id = user_id
        self.state = EngineState()
        self.stream_client: AlpacaStreamClient | None = None
        self.bars_processor: BarsProcessor | None = None
        self.signal_engine: SignalEngine | None = None
        self._last_stream_bar_ts: float = 0.0
        self._last_rest_fallback_ts: float = 0.0
        self._channel_layer = get_channel_layer()
        self._prev_close: dict[str, float] = {}
        self._prev_close_ts: float = 0.0
        self._last_broadcast_ts: dict[str, float] = {}
        self._focus_symbols: list[str] = []
        self._extra_symbols: list[str] = []
        self._extra_symbols_ts: float = 0.0
        self._stream_symbols: list[str] = []

    def run_once(self) -> None:
        now = time.time()
        self._maybe_refresh_universe(now)
        self._refresh_extra_symbols(now)
        self._maybe_refresh_focus(now, force=True)
        self._maybe_fetch_bars(now, force=True)

    def run(self) -> None:
        self._ensure_stream()
        while True:
            now = time.time()
            self._maybe_refresh_universe(now)
            self._refresh_extra_symbols(now)
            self._maybe_refresh_focus(now)
            self._maybe_fetch_bars(now)
            self._maybe_log(now)
            time.sleep(1.0)

    def _resolve_stream_symbols(self) -> list[str]:
        combined: list[str] = []
        seen: set[str] = set()
        for raw in self._extra_symbols + self._focus_symbols:
            symbol = str(raw or "").upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            combined.append(symbol)
        return combined

    def _apply_stream_symbols(self) -> None:
        if not self.config.engine.stream_enabled or not self.stream_client:
            return
        symbols = self._resolve_stream_symbols()
        if symbols == self._stream_symbols:
            return
        self._stream_symbols = symbols
        if symbols:
            self.stream_client.set_symbols(symbols)

    def _refresh_extra_symbols(self, now: float) -> None:
        symbols, updated_at = read_subscription_state()
        if updated_at <= self._extra_symbols_ts:
            return
        self._extra_symbols_ts = updated_at
        self._extra_symbols = symbols
        self._maybe_refresh_prev_close(self._resolve_stream_symbols(), now)
        self._apply_stream_symbols()

    def _maybe_refresh_universe(self, now: float) -> None:
        if now - self.state.last_universe_ts < self.config.engine.universe_refresh_seconds:
            return
        with track_latency("realtime.universe.refresh"):
            entries = build_universe(
                self.config.universe,
                user_id=self.user_id,
                feed=self.config.engine.feed,
            )
        self.state.last_universe_ts = now
        write_state(
            "universe_state.json",
            {
                "updated_at": now,
                "count": len(entries),
            },
        )

    def _maybe_refresh_focus(self, now: float, *, force: bool = False) -> None:
        if not force and (now - self.state.last_focus_ts < self.config.engine.focus_refresh_seconds):
            return
        universe_payload = []
        try:
            from .storage import read_state

            universe_state = read_state("universe_ranked.json", default={})
            universe_payload = [
                str(item.get("symbol"))
                for item in (universe_state.get("entries") or [])
                if isinstance(item, dict) and item.get("symbol")
            ]
        except Exception:
            universe_payload = []
        focus_entries = update_focus(universe_payload, self.config.focus)
        self.state.last_focus_ts = now
        self._focus_symbols = [entry.symbol for entry in focus_entries]
        self._maybe_refresh_prev_close(self._resolve_stream_symbols(), now)
        write_state(
            "focus_summary.json",
            {
                "updated_at": now,
                "count": len(focus_entries),
                "symbols": [entry.symbol for entry in focus_entries],
            },
        )
        if self.config.engine.stream_enabled:
            self._ensure_stream()
            self._apply_stream_symbols()

    def _maybe_fetch_bars(self, now: float, *, force: bool = False) -> None:
        if self.config.engine.stream_enabled and self.stream_client:
            if self.stream_client.is_connected() and (now - self._last_stream_bar_ts) < self.config.engine.focus_refresh_seconds:
                return
            if (now - self._last_rest_fallback_ts) < self.config.engine.focus_refresh_seconds:
                return
        if not force and (now - self.state.last_focus_ts > self.config.engine.focus_refresh_seconds * 2):
            return
        try:
            from .storage import read_state

            focus_state = read_state("focus_summary.json", default={})
            symbols = focus_state.get("symbols") or []
        except Exception:
            symbols = []
        if not symbols:
            return
        with track_latency("realtime.bars.fetch", symbols=len(symbols)):
            frame = fetch_bars_frame(
                symbols,
                timeframe=self.config.engine.bar_timeframe,
                limit=self.config.engine.bar_limit,
                feed=self.config.engine.feed,
                user_id=self.user_id,
            )
        rows = _extract_latest_bars(frame)
        if rows:
            for row in rows:
                row.setdefault("source", "rest")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            append_ndjson(f"bars_{stamp}.ndjson", rows)
            write_state(
                "bars_latest.json",
                {"updated_at": now, "bars": rows},
            )
            record_metric("realtime.bars.updated", rows=len(rows))
            if self.config.engine.stream_enabled:
                record_metric("realtime.bars.fallback", rows=len(rows))
                self._last_rest_fallback_ts = now

    def _ensure_stream(self) -> None:
        if not self.config.engine.stream_enabled or self.stream_client:
            return
        if AlpacaStreamClient is None:
            record_metric("realtime.stream.disabled", reason="missing_dependency")
            return
        if not acquire_stream("engine"):
            record_metric("realtime.stream.locked", owner="engine")
            return
        self.signal_engine = SignalEngine(self.config.signals)
        self.bars_processor = BarsProcessor(
            bar_interval_seconds=self.config.engine.bar_interval_seconds,
            bar_aggregate_seconds=self.config.engine.bar_aggregate_seconds,
            stale_seconds=self.config.engine.stale_seconds,
            on_bar_5s=self._handle_stream_bar,
        )
        client = AlpacaStreamClient(
            user_id=self.user_id,
            feed=self.config.engine.feed,
            stream_url=self.config.engine.stream_url,
            stream_trades=self.config.engine.stream_trades,
            stream_quotes=self.config.engine.stream_quotes,
            reconnect_seconds=self.config.engine.reconnect_seconds,
            on_trade=self.bars_processor.on_trade if self.bars_processor else None,
            on_quote=self.bars_processor.on_quote if self.bars_processor else None,
            on_status=self._handle_stream_status,
        )
        if client.start():
            self.stream_client = client
        else:
            release_stream("engine")
            record_metric("realtime.stream.disabled", reason="credentials_missing")

    def _handle_stream_status(self, status: str, detail: str) -> None:
        record_metric("realtime.stream.status", status=status, detail=detail)
        write_state(
            "stream_state.json",
            {
                "updated_at": time.time(),
                "status": status,
                "detail": detail,
            },
        )

    def _handle_stream_bar(self, bar: dict[str, Any]) -> None:
        self._last_stream_bar_ts = time.time()
        if self.signal_engine:
            self.signal_engine.on_bar(bar)
        self._broadcast_market_update(bar)

    def _maybe_refresh_prev_close(self, symbols: list[str], now: float) -> None:
        if not symbols:
            return
        if now - self._prev_close_ts < 300:
            return
        snapshots = fetch_stock_snapshots(symbols, feed=self.config.engine.feed, user_id=self.user_id)
        if not isinstance(snapshots, dict):
            return
        for symbol, snapshot in snapshots.items():
            if not isinstance(snapshot, dict):
                continue
            daily = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
            prev = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
            prev_close = prev.get("c") or prev.get("close") or 0.0
            if not prev_close:
                prev_close = daily.get("o") or daily.get("open") or daily.get("c") or daily.get("close") or 0.0
            try:
                prev_close_val = float(prev_close)
            except (TypeError, ValueError):
                continue
            if prev_close_val > 0:
                self._prev_close[str(symbol).upper()] = prev_close_val
        self._prev_close_ts = now

    def _broadcast_market_update(self, bar: dict[str, Any]) -> None:
        if not self._channel_layer or not isinstance(bar, dict):
            return
        symbol = str(bar.get("symbol") or "").upper()
        if not symbol:
            return
        price = bar.get("close") or bar.get("mid") or bar.get("bid") or bar.get("ask")
        try:
            price_val = float(price)
        except (TypeError, ValueError):
            return
        prev_close = self._prev_close.get(symbol)
        if not prev_close:
            prev_close = bar.get("open")
        try:
            prev_close_val = float(prev_close) if prev_close else None
        except (TypeError, ValueError):
            prev_close_val = None
        change_pct = ((price_val / prev_close_val) - 1.0) * 100.0 if prev_close_val else 0.0
        now = time.time()
        last_ts = self._last_broadcast_ts.get(symbol, 0.0)
        if now - last_ts < 0.1:
            return
        self._last_broadcast_ts[symbol] = now
        try:
            async_to_sync(self._channel_layer.group_send)(
                "market_global",
                {
                    "type": "market_update",
                    "symbol": symbol,
                    "price": price_val,
                    "change_pct": change_pct,
                    "server_ts": now,
                    "source": "engine",
                },
            )
        except Exception as exc:
            record_metric("realtime.ws.broadcast_error", error=str(exc), symbol=symbol)

    def _maybe_log(self, now: float) -> None:
        if now - self.state.last_log_ts < self.config.engine.log_every_seconds:
            return
        self.state.last_log_ts = now
        record_metric(
            "realtime.heartbeat",
            universe_ts=self.state.last_universe_ts,
            focus_ts=self.state.last_focus_ts,
        )
