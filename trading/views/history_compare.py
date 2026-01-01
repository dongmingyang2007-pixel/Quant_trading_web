from __future__ import annotations

from dataclasses import dataclass
import math
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
    risk: dict[str, Any]


MAX_COMPARE = 5
PALETTE = ["#2563eb", "#10b981", "#f97316", "#8b5cf6", "#ef4444"]


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
            if len(normalized) >= MAX_COMPARE:
                break
        if len(normalized) >= MAX_COMPARE:
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


def _extract_return_series(rows: list[dict[str, Any]]) -> dict[str, float]:
    series: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_value = row.get("date")
        daily_return = row.get("daily_return")
        if daily_return is None:
            daily_return = row.get("strategy_return")
        if date_value is None or daily_return is None:
            continue
        try:
            series[str(date_value)] = float(daily_return)
        except (TypeError, ValueError):
            continue
    return series


def _safe_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((val - mean) ** 2 for val in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (percent / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_vals[int(k)]
    weight = k - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _compute_corr(series_a: dict[str, float], series_b: dict[str, float]) -> float | None:
    shared_dates = [date for date in series_a.keys() if date in series_b]
    if len(shared_dates) < 2:
        return None
    xs = [series_a[date] for date in shared_dates]
    ys = [series_b[date] for date in shared_dates]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (len(xs) - 1)
    var_x = sum((x - mean_x) ** 2 for x in xs) / (len(xs) - 1)
    var_y = sum((y - mean_y) ** 2 for y in ys) / (len(ys) - 1)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _corr_color(value: float | None) -> str:
    if value is None:
        return "rgba(148, 163, 184, 0.12)"
    intensity = min(1.0, abs(value))
    base_alpha = 0.18 + 0.42 * intensity
    if value >= 0:
        return f"rgba(16, 185, 129, {base_alpha:.2f})"
    return f"rgba(248, 113, 113, {base_alpha:.2f})"


def _build_risk_metrics(returns: list[float], stats: dict[str, Any]) -> dict[str, Any]:
    avg_return = sum(returns) / len(returns) if returns else None
    downside = [val for val in returns if val < 0]
    downside_std = _safe_stdev(downside) if downside else None
    win_rate = (sum(1 for val in returns if val > 0) / len(returns)) if returns else None
    var_95 = _percentile(returns, 5.0) if returns else None
    cvar_95 = None
    if returns and var_95 is not None:
        tail = [val for val in returns if val <= var_95]
        if tail:
            cvar_95 = sum(tail) / len(tail)
    return {
        "avg_return_pct": _to_percent(avg_return),
        "win_rate_pct": _to_percent(win_rate),
        "downside_vol_pct": _to_percent(downside_std),
        "var_95_pct": _to_percent(var_95),
        "cvar_95_pct": _to_percent(cvar_95),
        "max_drawdown_pct": _to_percent(stats.get("max_drawdown")),
        "volatility_pct": _to_percent(stats.get("volatility")),
        "sharpe": stats.get("sharpe"),
        "sample_size": len(returns),
    }


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

    comparisons: list[ArenaSeries] = []
    series_map: dict[str, dict[str, float]] = {}
    for idx, record_id in enumerate(selected_ids[:MAX_COMPARE]):
        record = get_history_record(record_id, user_id=str(request.user.id))
        if not record:
            continue
        snapshot = record.get("snapshot") or {}
        stats = record.get("stats") or {}
        return_rows = snapshot.get("return_series")
        if not isinstance(return_rows, list) or not return_rows:
            return_rows = snapshot.get("recent_rows") or []
        curve = _build_curve(return_rows, start_date=record.get("start_date", ""), end_date=record.get("end_date", ""), total_return=stats.get("total_return"))
        color = PALETTE[idx % len(PALETTE)]
        label = f"{record.get('ticker', 'N/A')} · {record.get('engine', 'Strategy')}"
        derived = {
            "total_return_pct": _to_percent(stats.get("total_return")),
            "max_drawdown_pct": _to_percent(stats.get("max_drawdown")),
            "volatility_pct": _to_percent(stats.get("volatility")),
        }
        return_series = _extract_return_series(return_rows)
        series_map[record_id] = return_series
        risk = _build_risk_metrics(list(return_series.values()), stats)
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
                risk=risk,
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
    correlation_rows = []
    for base in comparisons:
        row_cells = []
        base_series = series_map.get(base.record_id, {})
        for other in comparisons:
            value = _compute_corr(base_series, series_map.get(other.record_id, {}))
            row_cells.append(
                {
                    "value": value,
                    "color": _corr_color(value),
                }
            )
        correlation_rows.append({"label": base.label, "cells": row_cells})
    correlation_matrix = {
        "labels": [entry.label for entry in comparisons],
        "rows": correlation_rows,
    }
    risk_rows = [
        {
            "label": entry.label,
            "ticker": entry.ticker,
            "engine": entry.engine,
            "color": entry.color,
            "metrics": entry.risk,
        }
        for entry in comparisons
    ]
    return render(
        request,
        "trading/backtest_compare.html",
        {
            "comparisons": comparisons,
            "series_payload": chart_payload,
            "correlation_matrix": correlation_matrix,
            "risk_rows": risk_rows,
            "error_message": "",
            "lang_is_zh": lang_is_zh,
        },
    )
