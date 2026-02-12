from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


from ..alpaca_data import resolve_alpaca_trading_mode
from ..file_utils import update_json_file
from ..models import RealtimeProfile
from ..runtime.execution import AlpacaExecutionClient, OrderRequest
from .config import load_realtime_config_from_payload
from .storage import append_ndjson, state_path


def _record_manual_order(payload: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d")
    append_ndjson(f"trade_orders_{stamp}.ndjson", [payload])

    def updater(current: Any) -> Any:
        if not isinstance(current, dict):
            current = {}
        orders = current.get("orders")
        if not isinstance(orders, list):
            orders = []
        orders.append(payload)
        current["orders"] = orders[-20:]
        current["updated_at"] = now.timestamp()
        return current

    update_json_file(state_path("trade_orders_latest.json"), default={"orders": []}, update_fn=updater)


def _get_active_profile(user) -> RealtimeProfile | None:
    if not user or not getattr(user, "is_authenticated", False):
        return None
    return RealtimeProfile.objects.filter(user=user, is_active=True).first()


def _is_daily_loss_guard_triggered(execution: AlpacaExecutionClient, max_daily_loss_pct: float) -> tuple[bool, float]:
    if max_daily_loss_pct <= 0:
        return False, 0.0
    account = execution.get_account() or {}
    try:
        equity = float(account.get("equity") or 0.0)
        baseline = float(account.get("last_equity") or 0.0)
    except (TypeError, ValueError):
        return False, 0.0
    if equity <= 0 or baseline <= 0:
        return False, 0.0
    loss_pct = max(0.0, (baseline - equity) / baseline)
    if loss_pct >= max_daily_loss_pct:
        return True, loss_pct
    return False, loss_pct


def submit_manual_order(
    *,
    user,
    symbol: str,
    side: str,
    qty: float = 0.0,
    notional: float = 0.0,
) -> dict[str, Any]:
    symbol = (symbol or "").strip().upper()
    side = (side or "").strip().lower()
    qty_val = float(qty or 0.0)
    notional_val = float(notional or 0.0)
    if not symbol or side not in {"buy", "sell"} or (qty_val <= 0 and notional_val <= 0):
        return {"ok": False, "message": "请输入正确的标的、方向与数量。"}

    profile = _get_active_profile(user)
    if not profile:
        return {"ok": False, "message": "尚未激活实时引擎配置档案。"}

    payload = profile.payload if isinstance(profile.payload, dict) else {}
    config = load_realtime_config_from_payload(payload)
    execution_cfg = config.trading.execution
    if not execution_cfg.enabled:
        return {"ok": False, "message": "执行模块未开启，无法提交订单。"}

    mode = config.trading.mode
    if mode not in {"paper", "live"}:
        mode = resolve_alpaca_trading_mode(user=user)

    execution = AlpacaExecutionClient(user_id=str(user.id), mode=mode)
    blocked, loss_pct = _is_daily_loss_guard_triggered(execution, config.trading.risk.max_daily_loss_pct)
    if blocked:
        return {
            "ok": False,
            "message": (
                "已触发日内风控，暂停手动下单。"
                f" 当前回撤 {loss_pct * 100:.2f}% ≥ 限额 {config.trading.risk.max_daily_loss_pct * 100:.2f}% 。"
            ),
            "status": "risk_blocked:max_daily_loss",
        }
    order = OrderRequest(
        symbol=symbol,
        qty=qty_val if qty_val > 0 else None,
        notional=notional_val if qty_val <= 0 else None,
        side=side,
        order_type=execution_cfg.order_type,
        time_in_force=execution_cfg.time_in_force,
    )

    result = execution.submit_order(order)
    status = result.status or "submitted"
    order_id = result.order_id

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "action": side,
        "weight": None,
        "status": status,
        "order_id": order_id,
        "source": "manual",
    }
    error_status = not status or status == "missing_credentials" or status.startswith("error")
    error_detail = ""
    if error_status and isinstance(result.raw, dict):
        error_detail = (
            result.raw.get("message")
            or result.raw.get("error")
            or result.raw.get("code")
            or ""
        )
    if error_status:
        payload["error"] = error_detail or result.raw
        _record_manual_order(payload)
        message = "订单提交失败。"
        if error_detail:
            message = f"订单提交失败：{error_detail}"
        return {"ok": False, "message": message, "status": status, "order_id": order_id}

    _record_manual_order(payload)
    return {"ok": True, "message": f"订单已提交（{status}）。", "status": status, "order_id": order_id}
