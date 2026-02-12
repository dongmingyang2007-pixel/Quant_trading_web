from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.http import require_http_methods

from ..observability import ensure_request_id
from ..runtime.execution import AlpacaExecutionClient, OrderRequest


def _normalize_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    return mode if mode in {"paper", "live"} else "paper"


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _serialize_account(account: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": account.get("id"),
        "status": account.get("status"),
        "currency": account.get("currency"),
        "equity": _parse_float(account.get("equity")),
        "cash": _parse_float(account.get("cash")),
        "buying_power": _parse_float(account.get("buying_power")),
        "portfolio_value": _parse_float(account.get("portfolio_value")),
        "last_equity": _parse_float(account.get("last_equity")),
        "pattern_day_trader": account.get("pattern_day_trader"),
        "trading_blocked": account.get("trading_blocked"),
        "transfers_blocked": account.get("transfers_blocked"),
        "account_blocked": account.get("account_blocked"),
    }


def _serialize_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        output.append(
            {
                "symbol": pos.get("symbol"),
                "qty": _parse_float(pos.get("qty")),
                "market_value": _parse_float(pos.get("market_value")),
                "current_price": _parse_float(pos.get("current_price")),
                "unrealized_pl": _parse_float(pos.get("unrealized_pl")),
                "unrealized_plpc": _parse_float(pos.get("unrealized_plpc")),
                "side": pos.get("side"),
            }
        )
    output.sort(key=lambda item: abs(item.get("market_value") or 0.0), reverse=True)
    return output


@login_required
def paper_trading(request):
    return redirect("/backtest/?workspace=trade")


@login_required
@require_http_methods(["GET"])
def paper_alpaca_account(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    mode = _normalize_mode(request.GET.get("mode"))
    client = AlpacaExecutionClient(user_id=str(request.user.id), mode=mode, strict_mode=True)
    account = client.get_account()
    if not account:
        return JsonResponse(
            {"ok": False, "error": "missing_credentials", "request_id": request_id},
            status=400,
        )
    positions = client.list_positions()
    return JsonResponse(
        {
            "ok": True,
            "mode": mode,
            "account": _serialize_account(account),
            "positions": _serialize_positions(positions),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
        }
    )


@login_required
@require_http_methods(["POST"])
def paper_alpaca_rebalance(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"ok": False, "error": "invalid_payload", "request_id": request_id}, status=400)
    if not isinstance(body, dict):
        return JsonResponse({"ok": False, "error": "invalid_payload", "request_id": request_id}, status=400)
    mode = _normalize_mode(body.get("mode"))
    client = AlpacaExecutionClient(user_id=str(request.user.id), mode=mode, strict_mode=True)
    account = client.get_account()
    if not account:
        return JsonResponse(
            {"ok": False, "error": "missing_credentials", "request_id": request_id},
            status=400,
        )
    equity = _parse_float(account.get("equity")) or _parse_float(account.get("portfolio_value")) or 0.0
    if equity <= 0:
        return JsonResponse({"ok": False, "error": "equity_unavailable", "request_id": request_id}, status=400)
    positions = client.list_positions()
    position_map = {str(p.get("symbol") or "").upper(): p for p in positions if isinstance(p, dict)}
    targets_raw = body.get("targets") if isinstance(body.get("targets"), list) else []
    liquidate_unlisted = bool(body.get("liquidate_unlisted"))
    liquidate_all = bool(body.get("liquidate_all"))

    min_notional = float(body.get("min_notional") or 10.0)
    orders: list[OrderRequest] = []
    plans: list[dict[str, Any]] = []

    if liquidate_all:
        for symbol, pos in position_map.items():
            qty = _parse_float(pos.get("qty")) or 0.0
            if qty == 0:
                continue
            side = "sell" if (pos.get("side") or "long") == "long" else "buy"
            orders.append(OrderRequest(symbol=symbol, qty=abs(qty), side=side))
            plans.append({"symbol": symbol, "action": side, "qty": abs(qty), "reason": "liquidate_all"})
    else:
        target_symbols: set[str] = set()
        for item in targets_raw:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            weight_raw = item.get("target_weight")
            weight_val = _parse_float(weight_raw)
            if weight_val is None:
                continue
            if weight_val > 1.0:
                weight_val = weight_val / 100.0
            weight_val = max(0.0, min(1.0, weight_val))
            target_symbols.add(symbol)
            target_notional = equity * weight_val
            pos = position_map.get(symbol, {})
            current_notional = _parse_float(pos.get("market_value")) or 0.0
            diff = target_notional - current_notional
            if abs(diff) < min_notional:
                continue
            if diff > 0:
                orders.append(OrderRequest(symbol=symbol, notional=diff, side="buy"))
                plans.append({"symbol": symbol, "action": "buy", "notional": diff})
            else:
                current_price = _parse_float(pos.get("current_price"))
                qty = abs(diff) / current_price if current_price else None
                if not qty:
                    plans.append({"symbol": symbol, "action": "sell", "error": "missing_price"})
                    continue
                orders.append(OrderRequest(symbol=symbol, qty=qty, side="sell"))
                plans.append({"symbol": symbol, "action": "sell", "qty": qty})

        if liquidate_unlisted:
            for symbol, pos in position_map.items():
                if symbol in target_symbols:
                    continue
                qty = _parse_float(pos.get("qty")) or 0.0
                if qty == 0:
                    continue
                side = "sell" if (pos.get("side") or "long") == "long" else "buy"
                orders.append(OrderRequest(symbol=symbol, qty=abs(qty), side=side))
                plans.append({"symbol": symbol, "action": side, "qty": abs(qty), "reason": "unlisted"})

    results: list[dict[str, Any]] = []
    for order in orders:
        result = client.submit_order(order)
        results.append(
            {
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "notional": order.notional,
                "status": result.status,
                "order_id": result.order_id,
            }
        )

    return JsonResponse(
        {
            "ok": True,
            "mode": mode,
            "planned": plans,
            "orders": results,
            "request_id": request_id,
        }
    )
