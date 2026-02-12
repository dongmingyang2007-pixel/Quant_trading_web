from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone as dt_timezone
from dataclasses import asdict
from typing import Any

import pandas as pd
from django.utils import timezone
from django.core.cache import cache
from django.db.models import Q

from rest_framework import status
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.renderers import BaseRenderer, BrowsableAPIRenderer, JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import HttpResponse

from ..observability import ensure_request_id
from ..error_contract import drf_error
from ..strategies import QuantStrategyError
from ..strategies.core import fetch_price_data
from ..task_queue import (
    SyncResult,
    cancel_task,
    get_task_status,
    submit_backtest_task,
    submit_robustness_task,
    submit_rl_task,
    submit_training_task,
)
from ..history import update_history_meta
from ..views.api import _build_screener_snapshot
from ..views.dashboard import build_strategy_input
from ..models import RealtimeProfile, StrategyPreset
from ..market_provider import (
    fetch_news as provider_fetch_news,
    fetch_stock_bars_frame as provider_fetch_stock_bars_frame,
    fetch_stock_snapshots as provider_fetch_stock_snapshots,
    fetch_stock_trades as provider_fetch_stock_trades,
    has_market_data_credentials,
    provider_caps,
    resolve_market_provider,
    resolve_massive_plan,
)
from ..massive_data import resolve_massive_ws_url
from ..api_usage import get_provider_api_usage
from ..realtime.storage import read_ndjson_tail, read_state
from ..services.strategy_workbench import (
    SHORTTERM_TRADING_MODE_MIGRATION,
    SHORTTERM_WORKBENCH_MIGRATION,
    WORKSPACE_CHOICES,
    apply_strategy_trading_mode,
    build_strategy_workbench_payload,
    describe_strategy_trading_mode,
    trade_source_unavailable,
)
from ..preprocessing import sanitize_price_history
from paper.engine import create_session, serialize_session
from paper.models import PaperTradingSession, PaperTrade
from .serializers import (
    StrategyTaskSerializer,
    TrainingTaskSerializer,
    PaperSessionCreateSerializer,
    RealtimeProfileSerializer,
    StrategyPresetSerializer,
    HistoryMetaSerializer,
)
from .throttles import TaskBurstThrottle


def _clamp_pagination(request, *, default_limit: int = 20, max_limit: int = 100) -> tuple[int, int]:
    try:
        limit = int(request.GET.get("limit", default_limit))
    except (TypeError, ValueError):
        limit = default_limit
    try:
        offset = int(request.GET.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    limit = max(1, min(max_limit, limit))
    offset = max(0, offset)
    return limit, offset


def _boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


class CsvRenderer(BaseRenderer):
    media_type = "text/csv"
    format = "csv"
    charset = "utf-8"
    render_style = "binary"

    def render(self, data, media_type=None, renderer_context=None):
        if data is None:
            return b""
        if isinstance(data, (bytes, bytearray)):
            return data
        return str(data).encode(self.charset or "utf-8")


class BaseTaskAPIView(APIView):
    throttle_classes = [TaskBurstThrottle]

    def _build_context(self, request) -> dict[str, Any]:
        return {"language": getattr(request, "LANGUAGE_CODE", None)}

    def _prepare_response(self, job, request_id: str) -> Response:
        payload = {
            "task_id": getattr(job, "id", ""),
            "state": getattr(job, "state", "PENDING"),
            "request_id": request_id,
        }
        status_code = status.HTTP_202_ACCEPTED
        if isinstance(job, SyncResult):
            payload["result"] = job.result
            status_code = status.HTTP_200_OK
        return Response(payload, status=status_code)


class BacktestTaskView(BaseTaskAPIView):
    _CLIENT_CACHE_TTL = 60 * 60
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def post(self, request):
        request_id = ensure_request_id(request)
        serializer = StrategyTaskSerializer(data=request.data, context=self._build_context(request))
        serializer.is_valid(raise_exception=True)
        cleaned = serializer.validated_data["_cleaned"]
        client_request_id = (serializer.validated_data.get("client_request_id") or "").strip()
        if client_request_id:
            user_key = str(getattr(request.user, "id", None) or getattr(request.user, "pk", None) or "anonymous")
            cache_key = f"backtest:client:{user_key}:{client_request_id}"
            cached = cache.get(cache_key)
            if isinstance(cached, dict):
                payload = dict(cached)
                status_code = payload.pop("_status", status.HTTP_202_ACCEPTED)
                payload["request_id"] = request_id
                return Response(payload, status=status_code)
        strategy_input, _ = build_strategy_input(cleaned, request_id=request_id, user=request.user)
        try:
            job = submit_backtest_task(asdict(strategy_input))
        except QuantStrategyError:
            return drf_error(
                error_code="invalid_backtest_params",
                message="Invalid backtest parameters.",
                status_code=status.HTTP_400_BAD_REQUEST,
                request_id=request_id,
                user_id=request.user.id,
                endpoint="api_v1.backtests.tasks",
            )
        except Exception:
            logging.getLogger(__name__).exception("Backtest job submission failed")
            return drf_error(
                error_code="backtest_submit_failed",
                message="Backtest execution failed.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                request_id=request_id,
                user_id=request.user.id,
                endpoint="api_v1.backtests.tasks",
            )
        response_payload = {
            "task_id": getattr(job, "id", ""),
            "state": getattr(job, "state", "PENDING"),
            "request_id": request_id,
        }
        status_code = status.HTTP_202_ACCEPTED
        if isinstance(job, SyncResult):
            response_payload["result"] = job.result
            status_code = status.HTTP_200_OK
        if client_request_id:
            cached_payload = dict(response_payload)
            cached_payload.pop("request_id", None)
            cached_payload["_status"] = status_code
            user_key = str(getattr(request.user, "id", None) or getattr(request.user, "pk", None) or "anonymous")
            cache_key = f"backtest:client:{user_key}:{client_request_id}"
            cache.set(cache_key, cached_payload, timeout=self._CLIENT_CACHE_TTL)
        return Response(response_payload, status=status_code)


class PreflightView(BaseTaskAPIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def post(self, request):
        request_id = ensure_request_id(request)
        serializer = StrategyTaskSerializer(data=request.data, context=self._build_context(request))
        serializer.is_valid(raise_exception=True)
        cleaned = serializer.validated_data["_cleaned"]
        strategy_input, _ = build_strategy_input(cleaned, request_id=request_id, user=request.user)
        try:
            prices, fetch_warnings = fetch_price_data(
                strategy_input.ticker,
                strategy_input.start_date,
                strategy_input.end_date,
                user_id=strategy_input.user_id,
            )
        except QuantStrategyError:
            return drf_error(
                error_code="preflight_invalid_params",
                message="Invalid preflight parameters.",
                status_code=status.HTTP_400_BAD_REQUEST,
                request_id=request_id,
                user_id=request.user.id,
                endpoint="api_v1.backtests.preflight",
            )
        except Exception:
            logging.getLogger(__name__).exception("Preflight data fetch failed")
            return drf_error(
                error_code="preflight_failed",
                message="Preflight failed.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                request_id=request_id,
                user_id=request.user.id,
                endpoint="api_v1.backtests.preflight",
            )

        prices, quality_report = sanitize_price_history(prices)
        rows = int(prices.shape[0]) if prices is not None else 0
        effective_start = None
        effective_end = None
        if rows:
            start_idx = prices.index[0]
            end_idx = prices.index[-1]
            effective_start = str(start_idx.date()) if hasattr(start_idx, "date") else str(start_idx)
            effective_end = str(end_idx.date()) if hasattr(end_idx, "date") else str(end_idx)
        data_quality = quality_report.to_dict() if quality_report else {}
        notes = list(fetch_warnings or [])
        if quality_report and quality_report.notes:
            notes.extend(quality_report.notes)
        min_required = max(
            strategy_input.long_window + strategy_input.rsi_period,
            strategy_input.long_window * 3,
            strategy_input.train_window + strategy_input.test_window,
            200,
        )
        if rows and rows < min_required:
            notes.append(f"数据行数 {rows} 低于最小要求 {min_required}，回测将自动扩展窗口。")
        response_payload = {
            "request_id": request_id,
            "ticker": strategy_input.ticker,
            "requested_start": strategy_input.start_date.isoformat(),
            "requested_end": strategy_input.end_date.isoformat(),
            "effective_start": effective_start,
            "effective_end": effective_end,
            "rows": rows,
            "min_required": min_required,
            "data_quality": data_quality,
            "notes": notes,
            "source": getattr(prices, "attrs", {}).get("data_source"),
            "cache_path": getattr(prices, "attrs", {}).get("cache_path"),
        }
        return Response(response_payload, status=status.HTTP_200_OK)


class TrainingTaskView(BaseTaskAPIView):
    def post(self, request):
        request_id = ensure_request_id(request)
        serializer = TrainingTaskSerializer(data=request.data, context=self._build_context(request))
        serializer.is_valid(raise_exception=True)
        cleaned = serializer.validated_data["_cleaned"]
        tickers = serializer.validated_data.get("tickers") or [cleaned["ticker"]]
        engines = serializer.validated_data.get("engines")
        strategy_input, _ = build_strategy_input(cleaned, request_id=request_id, user=request.user)
        payload: dict[str, Any] = {
            "base_params": asdict(strategy_input),
            "tickers": [symbol.upper() for symbol in tickers if symbol],
        }
        if engines:
            payload["engines"] = engines
        job = submit_training_task(payload)
        return self._prepare_response(job, request_id)


class RLTaskView(BaseTaskAPIView):
    def post(self, request):
        request_id = ensure_request_id(request)
        serializer = StrategyTaskSerializer(data=request.data, context=self._build_context(request))
        serializer.is_valid(raise_exception=True)
        cleaned = serializer.validated_data["_cleaned"]
        strategy_input, _ = build_strategy_input(cleaned, request_id=request_id, user=request.user)
        job = submit_rl_task(asdict(strategy_input))
        return self._prepare_response(job, request_id)


class RobustnessTaskView(BaseTaskAPIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def post(self, request):
        request_id = ensure_request_id(request)
        serializer = StrategyTaskSerializer(data=request.data, context=self._build_context(request))
        serializer.is_valid(raise_exception=True)
        cleaned = serializer.validated_data["_cleaned"]
        strategy_input, _ = build_strategy_input(cleaned, request_id=request_id, user=request.user)
        payload: dict[str, Any] = asdict(strategy_input)
        robustness_config = request.data.get("robustness")
        if isinstance(robustness_config, dict):
            payload["robustness"] = robustness_config
        job = submit_robustness_task(payload)
        return self._prepare_response(job, request_id)


class TaskStatusView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def get(self, request, task_id: str):
        payload = get_task_status(task_id)
        payload["request_id"] = ensure_request_id(request)
        return Response(payload)


class TaskCancelView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def post(self, request, task_id: str):
        payload = cancel_task(task_id)
        payload["request_id"] = ensure_request_id(request)
        return Response(payload)


class HistoryMetaView(APIView):
    def patch(self, request, record_id: str):
        request_id = ensure_request_id(request)
        if not request.user.is_authenticated:
            return Response({"error": "auth_required", "request_id": request_id}, status=status.HTTP_401_UNAUTHORIZED)
        serializer = HistoryMetaSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        if not payload:
            return Response({"error": "empty_payload", "request_id": request_id}, status=status.HTTP_400_BAD_REQUEST)
        updated = update_history_meta(record_id, user_id=str(request.user.id), **payload)
        if not updated:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        response = dict(updated)
        response["request_id"] = request_id
        return Response(response)


class ScreenerSnapshotView(APIView):
    """DRF wrapper for screener snapshot API under /api/v1."""

    def get(self, request):
        request_id = ensure_request_id(request)
        params = request.query_params
        if hasattr(params, "items"):
            params = {k: v for k, v in params.items()}
        payload, status_code = _build_screener_snapshot(
            user=request.user,
            params=params,
            request_id=request_id,
        )
        payload["request_id"] = request_id
        return Response(payload, status=status_code)


class PaperSessionView(APIView):
    """List/create paper trading sessions."""

    def get(self, request):
        request_id = ensure_request_id(request)
        limit, offset = _clamp_pagination(request)
        qs = PaperTradingSession.objects.filter(user=request.user).order_by("-updated_at")
        status_filter = (request.GET.get("status") or "").strip().lower()
        if status_filter and status_filter != "all":
            status_values = [value.strip() for value in status_filter.split(",") if value.strip()]
            qs = qs.filter(status__in=status_values)
        query = (request.GET.get("q") or "").strip()
        if query:
            qs = qs.filter(Q(ticker__icontains=query) | Q(name__icontains=query))
        sort = (request.GET.get("sort") or "updated").strip().lower()
        if sort == "equity":
            qs = qs.order_by("-last_equity", "-updated_at")
        elif sort == "cash":
            qs = qs.order_by("-current_cash", "-updated_at")
        total = qs.count()
        sessions = qs[offset : offset + limit]
        payload = {
            "sessions": [serialize_session(session, include_config=False) for session in sessions],
            "limit": limit,
            "offset": offset,
            "next_offset": offset + len(sessions),
            "prev_offset": max(0, offset - limit),
            "total": total,
            "has_next": offset + len(sessions) < total,
            "has_prev": offset > 0,
            "request_id": request_id,
        }
        return Response(payload)

    def post(self, request):
        request_id = ensure_request_id(request)
        serializer = PaperSessionCreateSerializer(data=request.data, context={"language": getattr(request, "LANGUAGE_CODE", None)})
        serializer.is_valid(raise_exception=True)
        cleaned = serializer.validated_data["_cleaned"]
        initial_cash = serializer.validated_data.get("initial_cash")
        interval_seconds = serializer.validated_data.get("interval_seconds")
        strategy_input, _ = build_strategy_input(cleaned, request_id=request_id, user=request.user)
        session = create_session(
            request.user,
            strategy_input,
            name=serializer.validated_data.get("name"),
            initial_cash=initial_cash,
        )
        if interval_seconds:
            session.interval_seconds = max(30, int(interval_seconds))
            session.next_run_at = None
            session.save(update_fields=["interval_seconds", "next_run_at"])
        payload = serialize_session(session)
        payload["request_id"] = request_id
        return Response(payload, status=status.HTTP_201_CREATED)


class PaperSessionDetailView(APIView):
    """Retrieve or mutate a single paper trading session."""

    def get_object(self, request, session_id: str) -> PaperTradingSession:
        return PaperTradingSession.objects.filter(session_id=session_id, user=request.user).first()

    def get(self, request, session_id: str):
        request_id = ensure_request_id(request)
        session = self.get_object(request, session_id)
        if not session:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        payload = serialize_session(session, include_details=True, trades_limit=50)
        payload["request_id"] = request_id
        return Response(payload)

    def post(self, request, session_id: str):
        request_id = ensure_request_id(request)
        session = self.get_object(request, session_id)
        if not session:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        action = (request.data or {}).get("action")
        if action == "pause":
            session.status = "paused"
        elif action == "resume":
            session.status = "running"
            session.next_run_at = None
        elif action == "stop":
            session.status = "stopped"
            session.ended_at = getattr(session, "ended_at", None) or timezone.now()
            session.next_run_at = None
        else:
            return Response({"error": "unsupported_action", "request_id": request_id}, status=status.HTTP_400_BAD_REQUEST)
        session.save(update_fields=["status", "next_run_at", "ended_at", "updated_at"])
        payload = serialize_session(session, include_details=True)
        payload["request_id"] = request_id
        return Response(payload)

    def delete(self, request, session_id: str):
        request_id = ensure_request_id(request)
        session = self.get_object(request, session_id)
        if not session:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        session.delete()
        return Response({"deleted": True, "session_id": session_id, "request_id": request_id})


class PaperSessionTradesView(APIView):
    """Export trades for a paper session."""

    renderer_classes = [JSONRenderer, BrowsableAPIRenderer, CsvRenderer]

    def get(self, request, session_id: str):
        request_id = ensure_request_id(request)
        session = PaperTradingSession.objects.filter(session_id=str(session_id)).first()
        if not session:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        if not request.user.is_authenticated or session.user_id != request.user.id:
            return Response({"error": "forbidden", "request_id": request_id}, status=status.HTTP_403_FORBIDDEN)
        limit, offset = _clamp_pagination(request, default_limit=200, max_limit=1000)
        trades_qs = PaperTrade.objects.filter(session=session).order_by("-executed_at")
        trades = trades_qs[offset : offset + limit]
        fmt = (request.GET.get("format") or "csv").strip().lower()
        payload = [
            {
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": float(trade.quantity),
                "price": float(trade.price),
                "notional": float(trade.notional),
                "executed_at": trade.executed_at.isoformat(),
            }
            for trade in trades
        ]
        if fmt == "json":
            return Response({"trades": payload, "request_id": request_id})
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["symbol", "side", "quantity", "price", "notional", "executed_at"])
        for row in payload:
            writer.writerow(
                [
                    row["symbol"],
                    row["side"],
                    row["quantity"],
                    row["price"],
                    row["notional"],
                    row["executed_at"],
                ]
            )
        response = HttpResponse(buffer.getvalue(), content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="paper_trades_{session_id}.csv"'
        response["X-Request-Id"] = request_id
        return response


class StrategyPresetView(APIView):
    def get(self, request):
        request_id = ensure_request_id(request)
        if not request.user.is_authenticated:
            return Response({"error": "auth_required", "request_id": request_id}, status=status.HTTP_401_UNAUTHORIZED)
        presets = StrategyPreset.objects.filter(user=request.user).order_by("-updated_at")
        payload = {
            "presets": [
                {
                    "preset_id": preset.preset_id,
                    "name": preset.name,
                    "description": preset.description,
                    "payload": preset.payload,
                    "is_default": preset.is_default,
                    "created_at": preset.created_at,
                    "updated_at": preset.updated_at,
                }
                for preset in presets
            ],
            "request_id": request_id,
        }
        return Response(payload)

    def post(self, request):
        request_id = ensure_request_id(request)
        if not request.user.is_authenticated:
            return Response({"error": "auth_required", "request_id": request_id}, status=status.HTTP_401_UNAUTHORIZED)
        serializer = StrategyPresetSerializer(
            data=request.data, context={"language": getattr(request, "LANGUAGE_CODE", None)}
        )
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        is_default = bool(data.get("is_default"))
        if is_default:
            StrategyPreset.objects.filter(user=request.user).update(is_default=False)
        preset = StrategyPreset.objects.create(
            user=request.user,
            name=data["name"],
            description=data.get("description", ""),
            payload=data.get("payload") or {},
            is_default=is_default,
        )
        response = {
            "preset_id": preset.preset_id,
            "name": preset.name,
            "description": preset.description,
            "payload": preset.payload,
            "is_default": preset.is_default,
            "created_at": preset.created_at,
            "updated_at": preset.updated_at,
            "request_id": request_id,
        }
        return Response(response, status=status.HTTP_201_CREATED)


class StrategyPresetDetailView(APIView):
    def _get_object(self, request, preset_id: str) -> StrategyPreset | None:
        if not request.user.is_authenticated:
            return None
        return StrategyPreset.objects.filter(preset_id=preset_id, user=request.user).first()

    def patch(self, request, preset_id: str):
        request_id = ensure_request_id(request)
        preset = self._get_object(request, preset_id)
        if not preset:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        serializer = StrategyPresetSerializer(
            preset,
            data=request.data,
            partial=True,
            context={"language": getattr(request, "LANGUAGE_CODE", None)},
        )
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        update_fields = []
        if "name" in data:
            preset.name = data["name"]
            update_fields.append("name")
        if "description" in data:
            preset.description = data.get("description", "")
            update_fields.append("description")
        if "payload" in data:
            preset.payload = data.get("payload") or {}
            update_fields.append("payload")
        if "is_default" in data:
            preset.is_default = bool(data.get("is_default"))
            update_fields.append("is_default")
            if preset.is_default:
                StrategyPreset.objects.filter(user=request.user).exclude(preset_id=preset.preset_id).update(
                    is_default=False
                )
        if update_fields:
            update_fields.append("updated_at")
            preset.save(update_fields=update_fields)
        response = {
            "preset_id": preset.preset_id,
            "name": preset.name,
            "description": preset.description,
            "payload": preset.payload,
            "is_default": preset.is_default,
            "created_at": preset.created_at,
            "updated_at": preset.updated_at,
            "request_id": request_id,
        }
        return Response(response)

    def delete(self, request, preset_id: str):
        request_id = ensure_request_id(request)
        preset = self._get_object(request, preset_id)
        if not preset:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        preset.delete()
        return Response({"deleted": True, "preset_id": preset_id, "request_id": request_id})


SHORTTERM_MIGRATION_HINT = {
    "deprecated": True,
    "migration": SHORTTERM_WORKBENCH_MIGRATION,
}


class ShortTermWorkbenchView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def get(self, request):
        request_id = ensure_request_id(request)
        response = build_strategy_workbench_payload(request.user, request_id=request_id, workspace="trade")
        response["deprecated"] = True
        response["migration"] = SHORTTERM_WORKBENCH_MIGRATION
        return Response(response, status=status.HTTP_200_OK)


class ShortTermTradingModeView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def get(self, request):
        request_id = ensure_request_id(request)
        response = describe_strategy_trading_mode(request.user, request_id=request_id)
        response["deprecated"] = True
        response["migration"] = SHORTTERM_TRADING_MODE_MIGRATION
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request):
        request_id = ensure_request_id(request)
        status_code, response = apply_strategy_trading_mode(
            request.user,
            request.data if isinstance(request.data, dict) else {},
            request_id=request_id,
        )
        response["deprecated"] = True
        response["migration"] = SHORTTERM_TRADING_MODE_MIGRATION
        return Response(response, status=status_code)


class StrategyWorkbenchView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def get(self, request):
        request_id = ensure_request_id(request)
        workspace = str(request.GET.get("workspace") or "trade").strip().lower()
        if workspace not in WORKSPACE_CHOICES:
            return Response(
                {
                    "error_code": "invalid_workspace",
                    "message": "Workspace must be trade, backtest, or review.",
                    "request_id": request_id,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        response = build_strategy_workbench_payload(request.user, request_id=request_id, workspace=workspace)
        if workspace == "trade" and trade_source_unavailable(response.get("trade")):
            market_data_source = str(response.get("market_data_source") or response.get("source") or "alpaca")
            source_label = market_data_source.upper()
            message = f"{source_label} market data is unavailable."
            return Response(
                {
                    "error_code": "market_data_unavailable",
                    "legacy_error_code": "alpaca_unavailable" if market_data_source == "alpaca" else "",
                    "provider": market_data_source,
                    "message": message,
                    "request_id": request_id,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(response, status=status.HTTP_200_OK)


class StrategyTradingModeView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def get(self, request):
        request_id = ensure_request_id(request)
        response = describe_strategy_trading_mode(request.user, request_id=request_id)
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request):
        request_id = ensure_request_id(request)
        status_code, response = apply_strategy_trading_mode(
            request.user,
            request.data if isinstance(request.data, dict) else {},
            request_id=request_id,
        )
        return Response(response, status=status_code)


class DataSourceDiagnoseView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def post(self, request):
        request_id = ensure_request_id(request)
        body = request.data if isinstance(request.data, dict) else {}
        configured_provider = resolve_market_provider(user=request.user)
        requested_provider = str(body.get("provider") or configured_provider).strip().lower()
        if requested_provider not in {"alpaca", "massive"}:
            return Response(
                {
                    "error_code": "provider_not_supported",
                    "message": "Provider must be alpaca or massive.",
                    "request_id": request_id,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        check_news = _boolish(body.get("check_news"), default=True)
        check_ws = _boolish(body.get("check_ws"), default=True)
        plan = resolve_massive_plan(user=request.user) if requested_provider == "massive" else "alpaca"
        caps = provider_caps(requested_provider, plan)
        checks: dict[str, dict[str, Any]] = {}
        credentials_ready = has_market_data_credentials(user=request.user, provider=requested_provider)
        checks["credentials"] = {"ok": credentials_ready}

        if not credentials_ready:
            checks["bars"] = {"ok": False, "detail": "missing_credentials"}
            checks["snapshot"] = {"ok": False, "detail": "missing_credentials"}
            checks["trades"] = {"ok": False, "detail": "missing_credentials"}
        else:
            try:
                end_dt = datetime.now(dt_timezone.utc)
                start_dt = end_dt - timedelta(days=5)
                bars_frame = provider_fetch_stock_bars_frame(
                    ["SPY"],
                    start=start_dt,
                    end=end_dt,
                    timeframe="1Day",
                    provider=requested_provider,
                    user=request.user,
                    timeout=8,
                )
            except Exception as exc:  # pragma: no cover - network/runtime variability
                checks["bars"] = {"ok": False, "detail": str(exc)}
            else:
                bars_ok = isinstance(bars_frame, pd.DataFrame) and not bars_frame.empty
                checks["bars"] = {
                    "ok": bars_ok,
                    "rows": int(getattr(bars_frame, "shape", [0])[0]) if isinstance(bars_frame, pd.DataFrame) else 0,
                    "detail": "ok" if bars_ok else "empty",
                }

            try:
                snapshots = provider_fetch_stock_snapshots(
                    ["SPY"],
                    provider=requested_provider,
                    user=request.user,
                    timeout=8,
                )
            except Exception as exc:  # pragma: no cover - network/runtime variability
                checks["snapshot"] = {"ok": False, "detail": str(exc)}
            else:
                checks["snapshot"] = {
                    "ok": isinstance(snapshots, dict) and bool(snapshots.get("SPY")),
                    "detail": "ok" if isinstance(snapshots, dict) and bool(snapshots.get("SPY")) else "empty",
                }

            try:
                trades, _next_token, _downgrade_to, _downgrade_message = provider_fetch_stock_trades(
                    "SPY",
                    limit=10,
                    max_pages=1,
                    sort="desc",
                    provider=requested_provider,
                    user=request.user,
                    timeout=8,
                )
            except Exception as exc:  # pragma: no cover - network/runtime variability
                checks["trades"] = {"ok": False, "detail": str(exc)}
            else:
                checks["trades"] = {"ok": isinstance(trades, list), "count": len(trades or [])}

        if check_news:
            if not bool(caps.get("news")):
                checks["news"] = {
                    "ok": False,
                    "error_code": "massive_news_not_entitled",
                    "detail": "Current plan does not include Massive news.",
                }
            elif not credentials_ready:
                checks["news"] = {
                    "ok": False,
                    "detail": "missing_credentials",
                }
            else:
                try:
                    items = provider_fetch_news(
                        symbols=["SPY"],
                        limit=5,
                        max_pages=1,
                        provider=requested_provider,
                        user=request.user,
                        timeout=8,
                    )
                except Exception as exc:  # pragma: no cover - network/runtime variability
                    checks["news"] = {"ok": False, "detail": str(exc)}
                else:
                    checks["news"] = {"ok": isinstance(items, list) and bool(items), "count": len(items or [])}

        if check_ws:
            if requested_provider == "massive":
                ws_url = resolve_massive_ws_url(user=request.user)
                checks["websocket"] = {"ok": bool(ws_url), "url": ws_url}
            else:
                checks["websocket"] = {"ok": credentials_ready}

        check_values = [bool(item.get("ok")) for item in checks.values()]
        ok = bool(check_values) and all(check_values)
        failed_checks = []
        for name, item in checks.items():
            if bool(item.get("ok")):
                continue
            detail = str(item.get("detail") or item.get("error_code") or "failed")
            failed_checks.append({"name": name, "detail": detail})

        message = "All checks passed." if ok else "One or more checks failed."
        error_code = None
        if not ok:
            if not credentials_ready and requested_provider == "massive":
                error_code = "massive_credentials_missing"
                message = "Massive credentials are missing. Save Massive API Key in API settings first."
            elif not credentials_ready and requested_provider == "alpaca":
                error_code = "market_data_unavailable"
                message = "Alpaca credentials are missing. Save Alpaca API keys in API settings first."
            elif checks.get("news", {}).get("error_code") == "massive_news_not_entitled":
                error_code = "massive_news_not_entitled"
                message = "Current Massive plan does not include news entitlement."
            if error_code is None:
                error_code = "market_data_unavailable"
            if failed_checks:
                failed_summary = ", ".join(f"{item['name']}:{item['detail']}" for item in failed_checks[:4])
                if failed_summary:
                    message = f"{message} Failed checks: {failed_summary}."

        response_payload: dict[str, Any] = {
            "ok": ok,
            "provider": requested_provider,
            "plan": plan,
            "checks": checks,
            "failed_checks": failed_checks,
            "message": message,
            "request_id": request_id,
        }
        if error_code:
            response_payload["error_code"] = error_code
        return Response(response_payload, status=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE)


class ApiUsageView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [BasicAuthentication, SessionAuthentication]

    def get(self, request):
        request_id = ensure_request_id(request)
        usage = get_provider_api_usage(user_id=str(getattr(request.user, "id", "") or ""))
        return Response(
            {
                **usage,
                "request_id": request_id,
            },
            status=status.HTTP_200_OK,
        )


class RealtimeProfileView(APIView):
    def get(self, request):
        request_id = ensure_request_id(request)
        if not request.user.is_authenticated:
            return Response({"error": "auth_required", "request_id": request_id}, status=status.HTTP_401_UNAUTHORIZED)
        profiles = RealtimeProfile.objects.filter(user=request.user).order_by("-is_active", "-updated_at")
        active_profile = next((profile for profile in profiles if profile.is_active), None)
        payload = {
            "profiles": [
                {
                    "profile_id": profile.profile_id,
                    "name": profile.name,
                    "description": profile.description,
                    "payload": profile.payload,
                    "is_active": profile.is_active,
                    "created_at": profile.created_at,
                    "updated_at": profile.updated_at,
                }
                for profile in profiles
            ],
            "active_profile_id": getattr(active_profile, "profile_id", None),
            "request_id": request_id,
        }
        payload.update(SHORTTERM_MIGRATION_HINT)
        return Response(payload)

    def post(self, request):
        request_id = ensure_request_id(request)
        if not request.user.is_authenticated:
            return Response({"error": "auth_required", "request_id": request_id}, status=status.HTTP_401_UNAUTHORIZED)
        serializer = RealtimeProfileSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        is_active = bool(data.get("is_active"))
        if is_active:
            RealtimeProfile.objects.filter(user=request.user).update(is_active=False)
        profile = RealtimeProfile.objects.create(
            user=request.user,
            name=data["name"],
            description=data.get("description", ""),
            payload=data.get("payload") or {},
            is_active=is_active,
        )
        response = {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "description": profile.description,
            "payload": profile.payload,
            "is_active": profile.is_active,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "request_id": request_id,
        }
        response.update(SHORTTERM_MIGRATION_HINT)
        return Response(response, status=status.HTTP_201_CREATED)


class RealtimeProfileDetailView(APIView):
    def _get_object(self, request, profile_id: str) -> RealtimeProfile | None:
        if not request.user.is_authenticated:
            return None
        return RealtimeProfile.objects.filter(profile_id=profile_id, user=request.user).first()

    def patch(self, request, profile_id: str):
        request_id = ensure_request_id(request)
        profile = self._get_object(request, profile_id)
        if not profile:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        serializer = RealtimeProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        update_fields = []
        if "name" in data:
            profile.name = data["name"]
            update_fields.append("name")
        if "description" in data:
            profile.description = data.get("description", "")
            update_fields.append("description")
        if "payload" in data:
            profile.payload = data.get("payload") or {}
            update_fields.append("payload")
        if "is_active" in data:
            profile.is_active = bool(data.get("is_active"))
            update_fields.append("is_active")
            if profile.is_active:
                RealtimeProfile.objects.filter(user=request.user).exclude(profile_id=profile.profile_id).update(
                    is_active=False
                )
        if update_fields:
            update_fields.append("updated_at")
            profile.save(update_fields=update_fields)
        response = {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "description": profile.description,
            "payload": profile.payload,
            "is_active": profile.is_active,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "request_id": request_id,
        }
        response.update(SHORTTERM_MIGRATION_HINT)
        return Response(response)

    def delete(self, request, profile_id: str):
        request_id = ensure_request_id(request)
        profile = self._get_object(request, profile_id)
        if not profile:
            return Response({"error": "not_found", "request_id": request_id}, status=status.HTTP_404_NOT_FOUND)
        profile.delete()
        response = {"deleted": True, "profile_id": profile_id, "request_id": request_id}
        response.update(SHORTTERM_MIGRATION_HINT)
        return Response(response)


class RealtimeSignalsView(APIView):
    def get(self, request):
        request_id = ensure_request_id(request)
        if not request.user.is_authenticated:
            return Response({"error": "auth_required", "request_id": request_id}, status=status.HTTP_401_UNAUTHORIZED)
        try:
            limit = int(request.GET.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(200, limit))
        source = (request.GET.get("source") or "").strip().lower()
        date = (request.GET.get("date") or "").strip()
        signals: list[dict[str, Any]]
        if source == "ndjson" or date:
            if not date:
                date = datetime.now(dt_timezone.utc).strftime("%Y%m%d")
            signals = read_ndjson_tail(f"signals_{date}.ndjson", limit=limit)
        else:
            payload = read_state("signals_latest.json", default={})
            raw = payload.get("signals") if isinstance(payload, dict) else []
            if isinstance(raw, list):
                signals = raw[-limit:]
            else:
                signals = []
        response = {"signals": signals, "request_id": request_id}
        response.update(SHORTTERM_MIGRATION_HINT)
        return Response(response)
