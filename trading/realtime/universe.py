from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import screener
from ..market_provider import fetch_stock_snapshots, resolve_market_provider
from ..observability import record_metric, track_latency
from .config import UniverseConfig
from .storage import read_state, write_state


@dataclass(slots=True)
class UniverseEntry:
    symbol: str
    price: float
    volume: float
    dollar_volume: float
    change_pct: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": round(self.price, 4),
            "volume": round(self.volume, 2),
            "dollar_volume": round(self.dollar_volume, 2),
            "change_pct": round(self.change_pct, 4),
            "score": round(self.score, 6),
        }


def _extract_snapshot_fields(snapshot: dict[str, Any]) -> tuple[float, float, float]:
    daily = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
    prev = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
    latest_trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
    price = daily.get("c") or latest_trade.get("p") or 0.0
    volume = daily.get("v") or 0.0
    prev_close = prev.get("c") or 0.0
    return float(price or 0.0), float(volume or 0.0), float(prev_close or 0.0)


def build_universe(
    config: UniverseConfig,
    *,
    user_id: str | None,
    feed: str | None,
) -> list[UniverseEntry]:
    provider = resolve_market_provider(user_id=user_id)
    symbols = config.symbols or _load_asset_symbols() or screener.CORE_TICKERS_US
    if config.max_symbols and len(symbols) > config.max_symbols:
        symbols = list(symbols)[: config.max_symbols]
    if not symbols:
        return []

    weights = config.score_weights
    with track_latency("realtime.universe.fetch", symbols=len(symbols)):
        snapshots = fetch_stock_snapshots(symbols, feed=feed, user_id=user_id, provider=provider)

    entries: list[UniverseEntry] = []
    for symbol, snapshot in (snapshots or {}).items():
        if not isinstance(snapshot, dict):
            continue
        price, volume, prev_close = _extract_snapshot_fields(snapshot)
        if price <= 0:
            continue
        dollar_volume = price * max(volume, 0.0)
        if price < config.min_price:
            continue
        if volume < config.min_volume:
            continue
        if dollar_volume < config.min_dollar_volume:
            continue
        change_pct = 0.0
        if prev_close > 0:
            change_pct = ((price / prev_close) - 1.0) * 100.0
        score = (
            weights.get("dollar_volume", 0.0) * dollar_volume
            + weights.get("change_pct", 0.0) * change_pct
            + weights.get("volume", 0.0) * volume
        )
        entries.append(
            UniverseEntry(
                symbol=str(symbol).upper(),
                price=price,
                volume=volume,
                dollar_volume=dollar_volume,
                change_pct=change_pct,
                score=score,
            )
        )

    entries.sort(key=lambda item: item.score, reverse=True)
    top_n = entries[: config.top_n]
    record_metric("realtime.universe.built", total=len(entries), top=len(top_n))
    write_state("universe_ranked.json", {"total": len(entries), "entries": [e.to_dict() for e in top_n]})
    return top_n


def _load_asset_symbols() -> list[str]:
    payload = read_state("assets_master.json", default={})
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, list):
        return []
    symbols: list[str] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        symbol = str(asset.get("symbol") or "").upper()
        if not symbol:
            continue
        status = str(asset.get("status") or "").lower()
        tradable = asset.get("tradable")
        if status and status not in {"active", "tradable"}:
            continue
        if tradable is False:
            continue
        symbols.append(symbol)
    return symbols
