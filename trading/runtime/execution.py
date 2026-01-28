from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from ..alpaca_data import resolve_alpaca_credentials
from ..network import get_requests_session, resolve_retry_config, retry_call_result


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    qty: float | None = None
    notional: float | None = None
    side: str = "buy"
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: float | None = None
    client_order_id: str | None = None


@dataclass(slots=True)
class OrderResult:
    order_id: str | None
    status: str | None
    raw: dict[str, Any] | None = None


class AlpacaExecutionClient:
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, *, user_id: str | None, mode: str = "paper", strict_mode: bool = False) -> None:
        self.user_id = user_id
        self.mode = mode if mode in {"paper", "live"} else "paper"
        self.strict_mode = strict_mode

    def _base_url(self) -> str:
        return self.LIVE_URL if self.mode == "live" else self.PAPER_URL

    def _headers(self) -> dict[str, str]:
        key_id, secret = resolve_alpaca_credentials(
            user_id=self.user_id,
            mode=self.mode,
            strict_mode=self.strict_mode,
        )
        if not key_id or not secret:
            return {}
        return {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        }

    def submit_order(self, order: OrderRequest, *, timeout: float | None = None) -> OrderResult:
        headers = self._headers()
        if not headers:
            return OrderResult(order_id=None, status="missing_credentials", raw=None)
        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.qty is not None:
            payload["qty"] = str(order.qty)
        if order.notional is not None:
            payload["notional"] = str(order.notional)
        if order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        if order.client_order_id:
            payload["client_order_id"] = order.client_order_id

        config = resolve_retry_config(timeout=timeout)
        session = get_requests_session(config.timeout)
        url = f"{self._base_url().rstrip('/')}/v2/orders"

        def _call():
            return session.post(url, json=payload, headers=headers, timeout=config.timeout)

        try:
            response = retry_call_result(
                _call,
                config=config,
                exceptions=(requests.RequestException,),
                should_retry=lambda resp: resp.status_code in {408, 429} or resp.status_code >= 500,
            )
        except Exception as exc:
            return OrderResult(order_id=None, status=f"error:{exc}", raw=None)

        if response is None:
            return OrderResult(order_id=None, status="error", raw=None)
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.status_code >= 400:
            return OrderResult(order_id=None, status="error", raw=data)
        return OrderResult(order_id=str(data.get("id")) if isinstance(data, dict) else None, status=data.get("status") if isinstance(data, dict) else None, raw=data)

    def get_account(self, *, timeout: float | None = None) -> dict[str, Any] | None:
        headers = self._headers()
        if not headers:
            return None
        config = resolve_retry_config(timeout=timeout)
        session = get_requests_session(config.timeout)
        url = f"{self._base_url().rstrip('/')}/v2/account"
        try:
            response = session.get(url, headers=headers, timeout=config.timeout)
            if response.status_code >= 400:
                return None
            return response.json()
        except Exception:
            return None

    def list_positions(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        headers = self._headers()
        if not headers:
            return []
        config = resolve_retry_config(timeout=timeout)
        session = get_requests_session(config.timeout)
        url = f"{self._base_url().rstrip('/')}/v2/positions"
        try:
            response = session.get(url, headers=headers, timeout=config.timeout)
            if response.status_code >= 400:
                return []
            payload = response.json()
        except Exception:
            return []
        return payload if isinstance(payload, list) else []
