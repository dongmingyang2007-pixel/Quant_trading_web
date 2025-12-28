from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Dict

from django.template.loader import render_to_string
from django.utils import timezone

try:  # optional dependency
    from weasyprint import HTML  # type: ignore
except Exception:  # pragma: no cover - optional
    HTML = None  # type: ignore


class ReportRenderingError(Exception):
    """Raised when a requested export format cannot be rendered."""


@dataclass
class ReportContext:
    snapshot: dict[str, Any]
    metrics: list[dict[str, Any]]
    stats: dict[str, Any]
    ai_summary: str
    generated_at: str
    user_name: str
    ticker: str
    benchmark: str
    params: dict[str, Any]
    data_quality: dict[str, Any]
    data_risks: list[str]
    signal_snapshot: dict[str, Any]
    target_portfolio: dict[str, Any]
    trade_list: list[dict[str, Any]]
    warnings: list[str]
    quick_summary: list[str]
    risk_alerts: list[str]
    disclaimer: str


def load_snapshot(raw: Any) -> dict[str, Any]:
    """Deserialize a snapshot stored in session/history."""

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def build_report_context(snapshot: dict[str, Any], *, user) -> ReportContext:
    metrics = snapshot.get("metrics") or []
    stats = snapshot.get("stats") or {}
    ai_summary = snapshot.get("ai_summary") or snapshot.get("ai_commentary") or ""
    generated_at = timezone.now().strftime("%Y-%m-%d %H:%M UTC")
    ticker = snapshot.get("ticker") or ""
    benchmark = snapshot.get("benchmark_ticker") or snapshot.get("benchmark") or ""
    params = snapshot.get("params") or {}
    data_quality = snapshot.get("data_quality") or stats.get("data_quality") or {}
    data_risks = snapshot.get("data_risks") or stats.get("data_risks") or []
    signal_snapshot = snapshot.get("signal_snapshot") or {}
    target_portfolio = snapshot.get("target_portfolio") or {}
    trade_list = snapshot.get("trade_list") or []
    warnings = snapshot.get("warnings") or []
    quick_summary = snapshot.get("quick_summary") or []
    risk_alerts = snapshot.get("risk_alerts") or []
    disclaimer = snapshot.get("disclaimer") or ""
    return ReportContext(
        snapshot=snapshot,
        metrics=metrics if isinstance(metrics, list) else [],
        stats=stats if isinstance(stats, dict) else {},
        ai_summary=ai_summary,
        generated_at=generated_at,
        user_name=getattr(user, "get_username", lambda: "")(),
        ticker=ticker,
        benchmark=benchmark,
        params=params if isinstance(params, dict) else {},
        data_quality=data_quality if isinstance(data_quality, dict) else {},
        data_risks=list(data_risks) if isinstance(data_risks, list) else [],
        signal_snapshot=signal_snapshot if isinstance(signal_snapshot, dict) else {},
        target_portfolio=target_portfolio if isinstance(target_portfolio, dict) else {},
        trade_list=trade_list if isinstance(trade_list, list) else [],
        warnings=list(warnings) if isinstance(warnings, list) else [],
        quick_summary=list(quick_summary) if isinstance(quick_summary, list) else [],
        risk_alerts=list(risk_alerts) if isinstance(risk_alerts, list) else [],
        disclaimer=str(disclaimer or ""),
    )


def render_report_html(context: ReportContext) -> str:
    """Render the report template to HTML."""

    return render_to_string(
        "trading/report/base.html",
        {
            "report": context,
        },
    )


def render_report_csv(context: ReportContext) -> str:
    """Render report metrics and stats as CSV."""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Ticker", context.ticker])
    writer.writerow(["Benchmark", context.benchmark])
    writer.writerow(["Generated At", context.generated_at])
    writer.writerow([])
    writer.writerow(["Metrics", "Value", "Insight"])
    for metric in context.metrics:
        writer.writerow(
            [
                metric.get("label", metric.get("name", "")),
                metric.get("value", ""),
                metric.get("description") or metric.get("insight", ""),
            ]
        )
    writer.writerow([])
    if context.data_quality:
        writer.writerow(["Data Quality"])
        for key, value in context.data_quality.items():
            writer.writerow([key, value])
        writer.writerow([])
    if context.trade_list:
        writer.writerow(["Trade List"])
        writer.writerow(["symbol", "side", "quantity", "price", "notional"])
        for trade in context.trade_list:
            writer.writerow(
                [
                    trade.get("symbol"),
                    trade.get("side"),
                    trade.get("quantity"),
                    trade.get("price"),
                    trade.get("notional"),
                ]
            )
        writer.writerow([])
    writer.writerow(["Statistic", "Value"])
    for key, value in context.stats.items():
        writer.writerow([key, value])
    return buffer.getvalue()


def render_report_pdf(context: ReportContext) -> bytes:
    """Render the HTML report as PDF using WeasyPrint (best-effort)."""

    if HTML is None:
        raise ReportRenderingError("WeasyPrint is not available.")
    html = render_report_html(context)
    try:
        return HTML(string=html).write_pdf()
    except Exception as exc:  # pragma: no cover - optional backend errors
        raise ReportRenderingError(str(exc)) from exc
