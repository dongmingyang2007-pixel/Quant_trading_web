from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..realtime.alpaca.stream import AlpacaStreamClient


@dataclass(slots=True)
class MarketDataEvent:
    symbol: str
    price: float | None
    size: float | None
    timestamp: float | None
    source: str = "alpaca"


class AlpacaMarketDataClient:
    def __init__(
        self,
        *,
        user_id: str | None,
        feed: str = "sip",
        stream_trades: bool = True,
        stream_quotes: bool = False,
    ) -> None:
        self._callbacks: list[Callable[[MarketDataEvent], None]] = []
        self._client = AlpacaStreamClient(
            user_id=user_id,
            feed=feed,
            stream_url=None,
            stream_trades=stream_trades,
            stream_quotes=stream_quotes,
            reconnect_seconds=5,
            on_trade=self._on_trade,
            on_quote=self._on_quote,
            on_status=None,
        )

    def add_callback(self, cb: Callable[[MarketDataEvent], None]) -> None:
        self._callbacks.append(cb)

    def start(self) -> bool:
        return self._client.start()

    def stop(self) -> None:
        self._client.stop()

    def set_symbols(self, symbols: list[str]) -> None:
        self._client.set_symbols(symbols)

    def _dispatch(self, event: MarketDataEvent) -> None:
        for cb in self._callbacks:
            cb(event)

    def _on_trade(self, symbol: str, price: float | None, size: float | None, ts: float | None) -> None:
        self._dispatch(MarketDataEvent(symbol=symbol, price=price, size=size, timestamp=ts))

    def _on_quote(
        self,
        symbol: str,
        bid: float | None,
        ask: float | None,
        bid_size: float | None,
        ask_size: float | None,
        ts: float | None,
    ) -> None:
        mid = None
        if bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2
        self._dispatch(MarketDataEvent(symbol=symbol, price=mid, size=None, timestamp=ts))
