from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

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
from .serializers import StrategyTaskSerializer, TrainingTaskSerializer
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
