from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution import AlpacaExecutionClient


@dataclass(slots=True)
class AccountState:
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    status: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float
    market_value: float | None = None
    avg_entry_price: float | None = None
    raw: dict[str, Any] | None = None


class AccountManager:
    def __init__(self, execution: AlpacaExecutionClient) -> None:
        self.execution = execution

    def get_account_state(self) -> AccountState:
        payload = self.execution.get_account() or {}
        equity = float(payload.get("equity") or 0.0)
        cash = float(payload.get("cash") or 0.0)
        buying_power = float(payload.get("buying_power") or 0.0)
        return AccountState(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            status=str(payload.get("status")) if payload else None,
            raw=payload if payload else None,
        )

    def list_positions(self) -> list[Position]:
        positions = self.execution.list_positions()
        results: list[Position] = []
        for item in positions:
            try:
                results.append(
                    Position(
                        symbol=str(item.get("symbol") or ""),
                        qty=float(item.get("qty") or 0.0),
                        market_value=float(item.get("market_value") or 0.0),
                        avg_entry_price=float(item.get("avg_entry_price") or 0.0),
                        raw=item,
                    )
                )
            except Exception:
                continue
        return results
