from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any
from django.utils import timezone

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from ..forms import QuantStrategyForm
from ..observability import ensure_request_id
from ..strategies import QuantStrategyError
from ..task_queue import (
    SyncResult,
    get_task_status,
    submit_backtest_task,
    submit_rl_task,
    submit_training_task,
)
from ..views.api import _build_screener_snapshot
from ..views.dashboard import build_strategy_input
from paper.engine import create_session, serialize_session
from paper.models import PaperTradingSession
from .serializers import StrategyTaskSerializer, TrainingTaskSerializer, PaperSessionCreateSerializer


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
from .throttles import TaskBurstThrottle


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
    def post(self, request):
        request_id = ensure_request_id(request)
        serializer = StrategyTaskSerializer(data=request.data, context=self._build_context(request))
        serializer.is_valid(raise_exception=True)
        cleaned = serializer.validated_data["_cleaned"]
        strategy_input, _ = build_strategy_input(cleaned, request_id=request_id, user=request.user)
        try:
            job = submit_backtest_task(asdict(strategy_input))
        except QuantStrategyError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logging.getLogger(__name__).exception("Backtest job submission failed")
            return Response({"error": "Backtest execution failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return self._prepare_response(job, request_id)


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


class TaskStatusView(APIView):
    def get(self, request, task_id: str):
        payload = get_task_status(task_id)
        payload["request_id"] = ensure_request_id(request)
        return Response(payload)


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
        qs = (
            PaperTradingSession.objects.filter(user=request.user)
            .order_by("-updated_at")
            .defer("equity_curve", "config")
        )
        total = qs.count()
        sessions = qs[offset : offset + limit]
        payload = {
            "sessions": [serialize_session(session) for session in sessions],
            "limit": limit,
            "offset": offset,
            "next_offset": offset + len(sessions),
            "total": total,
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
