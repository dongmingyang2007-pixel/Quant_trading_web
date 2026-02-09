from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
import threading
import time

from ..observability import record_metric
from .storage import append_ndjson, write_state


def _normalize_epoch_seconds(ts: float) -> float:
    if ts > 1e15:
        return ts / 1e9
    if ts > 1e12:
        return ts / 1e3
    return ts


def parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return _normalize_epoch_seconds(ts)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.replace(".", "", 1).isdigit():
                if "." in text:
                    return _normalize_epoch_seconds(float(text))
                return _normalize_epoch_seconds(float(int(text)))
        except (TypeError, ValueError):
            pass
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _bucket_start(ts: float, interval: int) -> float:
    return float(int(ts // interval) * interval)


@dataclass
class BarBucket:
    start_ts: float
    interval: int
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float = 0.0
    tick_count: int = 0
    trade_count: int = 0
    quote_count: int = 0
    last_price: float | None = None
    last_bid: float | None = None
    last_ask: float | None = None
    last_bid_size: float | None = None
    last_ask_size: float | None = None

    def update_price(
        self,
        price: float | None,
        *,
        size: float | None = None,
        is_trade: bool = False,
        bid: float | None = None,
        ask: float | None = None,
        bid_size: float | None = None,
        ask_size: float | None = None,
    ) -> None:
        if price is None:
            return
        if self.open is None:
            self.open = price
            self.high = price
            self.low = price
        else:
            self.high = max(self.high or price, price)
            self.low = min(self.low or price, price)
        self.close = price
        self.last_price = price
        self.tick_count += 1
        if is_trade:
            self.trade_count += 1
            if size:
                self.volume += float(size)
        if bid is not None or ask is not None:
            self.quote_count += 1
            if bid is not None:
                self.last_bid = bid
            if ask is not None:
                self.last_ask = ask
            if bid_size is not None:
                self.last_bid_size = bid_size
            if ask_size is not None:
                self.last_ask_size = ask_size

    def update_from_bar(self, bar: dict[str, Any]) -> None:
        open_val = bar.get("open")
        high_val = bar.get("high")
        low_val = bar.get("low")
        close_val = bar.get("close")
        if open_val is not None:
            if self.open is None:
                self.open = float(open_val)
            self.close = float(close_val) if close_val is not None else self.close
            self.high = max(self.high or float(high_val), float(high_val)) if high_val is not None else self.high
            self.low = min(self.low or float(low_val), float(low_val)) if low_val is not None else self.low
        if close_val is not None:
            self.last_price = float(close_val)
        self.volume += float(bar.get("volume") or 0)
        self.tick_count += int(bar.get("tick_count") or 0)
        self.trade_count += int(bar.get("trade_count") or 0)
        self.quote_count += int(bar.get("quote_count") or 0)
        bid = bar.get("bid")
        ask = bar.get("ask")
        if bid is not None:
            self.last_bid = float(bid)
        if ask is not None:
            self.last_ask = float(ask)
        bid_size = bar.get("bid_size")
        ask_size = bar.get("ask_size")
        if bid_size is not None:
            self.last_bid_size = float(bid_size)
        if ask_size is not None:
            self.last_ask_size = float(ask_size)

    def to_row(
        self,
        symbol: str,
        *,
        source: str,
        timeframe: str,
        stale: bool,
        last_tick_ts: float | None,
    ) -> dict[str, Any]:
        mid = None
        if self.last_bid is not None and self.last_ask is not None:
            mid = (self.last_bid + self.last_ask) / 2
        elif self.last_price is not None:
            mid = self.last_price
        payload = {
            "symbol": symbol,
            "timestamp": _to_iso(self.start_ts),
            "timeframe": timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": round(self.volume, 4),
            "tick_count": self.tick_count,
            "trade_count": self.trade_count,
            "quote_count": self.quote_count,
            "mid": mid,
            "bid": self.last_bid,
            "ask": self.last_ask,
            "bid_size": self.last_bid_size,
            "ask_size": self.last_ask_size,
            "stale": stale,
            "stale_seconds": round(max(0.0, time.time() - last_tick_ts), 3) if stale and last_tick_ts else None,
            "source": source,
        }
        return payload


class BarsProcessor:
    def __init__(
        self,
        *,
        bar_interval_seconds: int,
        bar_aggregate_seconds: int,
        stale_seconds: float,
        on_bar_5s: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.bar_interval_seconds = max(1, int(bar_interval_seconds))
        self.bar_aggregate_seconds = max(self.bar_interval_seconds, int(bar_aggregate_seconds))
        self.stale_seconds = max(0.1, float(stale_seconds))
        self._lock = threading.Lock()
        self._bars_1s: dict[str, BarBucket] = {}
        self._bars_5s: dict[str, BarBucket] = {}
        self._last_tick_ts: dict[str, float] = {}
        self._latest_bars: dict[str, dict[str, Any]] = {}
        self._on_bar_5s = on_bar_5s

    def on_trade(self, symbol: str, price: float | None, size: float | None, ts: float | None) -> None:
        if not symbol or price is None or ts is None:
            return
        self._handle_tick(
            symbol=symbol.upper(),
            price=float(price),
            size=size,
            ts=float(ts),
            is_trade=True,
        )

    def on_quote(
        self,
        symbol: str,
        bid: float | None,
        ask: float | None,
        bid_size: float | None,
        ask_size: float | None,
        ts: float | None,
    ) -> None:
        if not symbol or ts is None:
            return
        mid = None
        if bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2
        self._handle_tick(
            symbol=symbol.upper(),
            price=mid,
            size=None,
            ts=float(ts),
            is_trade=False,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
        )

    def _handle_tick(
        self,
        *,
        symbol: str,
        price: float | None,
        size: float | None,
        ts: float,
        is_trade: bool,
        bid: float | None = None,
        ask: float | None = None,
        bid_size: float | None = None,
        ask_size: float | None = None,
    ) -> None:
        bucket_ts = _bucket_start(ts, self.bar_interval_seconds)
        with self._lock:
            current = self._bars_1s.get(symbol)
            if current and current.start_ts != bucket_ts:
                self._emit_1s_bar(symbol, current)
                current = None
            if current is None:
                current = BarBucket(start_ts=bucket_ts, interval=self.bar_interval_seconds)
                self._bars_1s[symbol] = current
            current.update_price(
                price,
                size=size,
                is_trade=is_trade,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
            )
            self._last_tick_ts[symbol] = ts

    def _emit_1s_bar(self, symbol: str, bar: BarBucket) -> None:
        last_tick_ts = self._last_tick_ts.get(symbol)
        stale = bool(last_tick_ts and (time.time() - last_tick_ts) > self.stale_seconds)
        row = bar.to_row(
            symbol,
            source="stream",
            timeframe=f"{self.bar_interval_seconds}s",
            stale=stale,
            last_tick_ts=last_tick_ts,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        append_ndjson(f"bars_{self.bar_interval_seconds}s_{stamp}.ndjson", [row])
        self._update_5s(symbol, row, bar.start_ts)

    def _update_5s(self, symbol: str, bar: dict[str, Any], bar_ts: float) -> None:
        bucket_ts = _bucket_start(bar_ts, self.bar_aggregate_seconds)
        current = self._bars_5s.get(symbol)
        if current and current.start_ts != bucket_ts:
            self._emit_5s_bar(symbol, current)
            current = None
        if current is None:
            current = BarBucket(start_ts=bucket_ts, interval=self.bar_aggregate_seconds)
            self._bars_5s[symbol] = current
        current.update_from_bar(bar)

    def _emit_5s_bar(self, symbol: str, bar: BarBucket) -> None:
        last_tick_ts = self._last_tick_ts.get(symbol)
        now_ts = bar.start_ts + self.bar_aggregate_seconds
        stale = bool(last_tick_ts and (now_ts - last_tick_ts) > self.stale_seconds)
        row = bar.to_row(
            symbol,
            source="stream",
            timeframe=f"{self.bar_aggregate_seconds}s",
            stale=stale,
            last_tick_ts=last_tick_ts,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        append_ndjson(f"bars_{self.bar_aggregate_seconds}s_{stamp}.ndjson", [row])
        self._latest_bars[symbol] = row
        self._write_latest()
        if self._on_bar_5s:
            try:
                self._on_bar_5s(row)
            except Exception as exc:
                record_metric("realtime.signals.error", error=str(exc), symbol=symbol)

    def _write_latest(self) -> None:
        rows = list(self._latest_bars.values())
        rows.sort(key=lambda item: item.get("symbol") or "")
        write_state("bars_latest.json", {"updated_at": time.time(), "bars": rows[:200]})
