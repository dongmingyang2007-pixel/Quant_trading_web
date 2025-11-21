from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from ..history import get_history_record
from .account import _lang_helpers


@dataclass
class ArenaSeries:
    record_id: str
    label: str
    ticker: str
    engine: str
    period: str
    stats: dict[str, Any]
    points: list[dict[str, Any]]
    color: str
    derived: dict[str, Any]


def _normalize_ids(request) -> list[str]:
    raw_candidates: list[str] = []
    for key in ("records", "history", "ids", "record"):
        values = request.GET.getlist(key)
        if values:
            raw_candidates.extend(values)
    single = request.GET.get("id")
    if single:
        raw_candidates.append(single)
    normalized: list[str] = []
    for raw in raw_candidates:
        if not raw:
            continue
        for piece in str(raw).split(","):
            trimmed = piece.strip()
            if trimmed and trimmed not in normalized:
                normalized.append(trimmed)
            if len(normalized) >= 3:
                break
        if len(normalized) >= 3:
            break
    return normalized


def _build_curve(rows: list[dict[str, Any]], *, start_date: str, end_date: str, total_return: float | None) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    sorted_rows = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: row.get("date") or "",
    )
    for row in sorted_rows:
        date_value = row.get("date")
        value = row.get("cum_strategy")
        if date_value and value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            points.append({"time": str(date_value), "value": numeric})
    if not points:
        end_value = (total_return or 0.0) + 1.0
        start_time = start_date or end_date or "1970-01-01"
        end_time = end_date or start_time
        return [
            {"time": start_time, "value": 1.0},
            {"time": end_time, "value": round(end_value, 4)},
        ]
    baseline = points[0]["value"] or 1.0
    if baseline == 0:
        baseline = 1.0
    normalized_points = []
    for point in points:
        normalized_points.append(
            {
                "time": point["time"],
                "value": round(point["value"] / baseline, 4),
            }
        )
    return normalized_points


def _to_percent(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric * 100.0


@login_required
def history_compare(request):
    language, lang_is_zh, _msg = _lang_helpers(request)
    selected_ids = _normalize_ids(request)
    if len(selected_ids) < 2:
        return render(
            request,
            "trading/backtest_compare.html",
            {
                "comparisons": [],
                "series_payload": [],
                "error_message": _msg("Select at least two records to compare.", "请至少选择两条历史记录后再对比。"),
                "lang_is_zh": lang_is_zh,
            },
        )

    palette = ["#2563eb", "#10b981", "#f97316", "#8b5cf6"]
    comparisons: list[ArenaSeries] = []
    for idx, record_id in enumerate(selected_ids[:3]):
        record = get_history_record(record_id, user_id=str(request.user.id))
        if not record:
            continue
        snapshot = record.get("snapshot") or {}
        stats = record.get("stats") or {}
        recent_rows = snapshot.get("recent_rows") or []
        curve = _build_curve(recent_rows, start_date=record.get("start_date", ""), end_date=record.get("end_date", ""), total_return=stats.get("total_return"))
        color = palette[idx % len(palette)]
        label = f"{record.get('ticker', 'N/A')} · {record.get('engine', 'Strategy')}"
        derived = {
            "total_return_pct": _to_percent(stats.get("total_return")),
            "max_drawdown_pct": _to_percent(stats.get("max_drawdown")),
            "volatility_pct": _to_percent(stats.get("volatility")),
        }
        comparisons.append(
            ArenaSeries(
                record_id=record_id,
                label=label,
                ticker=record.get("ticker", ""),
                engine=record.get("engine", ""),
                period=f"{record.get('start_date', '--')} → {record.get('end_date', '--')}",
                stats=stats,
                points=curve,
                color=color,
                derived=derived,
            )
        )

    if len(comparisons) < 2:
        return render(
            request,
            "trading/backtest_compare.html",
            {
                "comparisons": [],
                "series_payload": [],
                "error_message": _msg("Not enough valid runs to compare.", "有效的历史记录不足，暂无法对比。"),
                "lang_is_zh": lang_is_zh,
            },
        )

    chart_payload = [
        {
            "label": entry.label,
            "color": entry.color,
            "points": entry.points,
        }
        for entry in comparisons
    ]
    return render(
        request,
        "trading/backtest_compare.html",
        {
            "comparisons": comparisons,
            "series_payload": chart_payload,
            "error_message": "",
            "lang_is_zh": lang_is_zh,
        },
    )
