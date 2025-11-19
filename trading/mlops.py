from __future__ import annotations

import json
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import numpy as np
try:  # optional parallel backend
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore
    def delayed(func):  # type: ignore
        return func

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .strategies import StrategyInput  # type: ignore

from . import strategies  # local import to reduce circular imports

try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ks_2samp = None  # type: ignore


def walk_forward_train(
    base_params: "StrategyInput",
    *,
    horizon_days: int = 365,
    step_days: int = 30,
    output_dir: Path | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """
    Run a simple walk-forward training loop (monthly by default), capturing OOS metrics.
    """

    reports: list[dict[str, Any]] = []
    end_date = base_params.end_date
    output_dir = output_dir or (strategies.DATA_CACHE_DIR / "training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build rolling windows
    windows: list[tuple[date, date]] = []
    cursor = end_date
    while cursor > base_params.start_date:
        start = max(base_params.start_date, cursor - timedelta(days=horizon_days))
        windows.append((start, cursor))
        cursor = cursor - timedelta(days=step_days)

    def _run_window(start: date, end: date) -> dict[str, Any]:
        from dataclasses import replace

        params = replace(base_params, start_date=start, end_date=end)
        result = strategies.run_quant_pipeline(params)
        stats = result.get("stats", {})
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "cagr": stats.get("cagr"),
            "sharpe": stats.get("sharpe"),
            "var_95": stats.get("var_95"),
            "cvar_95": stats.get("cvar_95"),
            "max_drawdown": stats.get("max_drawdown"),
            "recovery_days": stats.get("recovery_days"),
            "trade_days": stats.get("trading_days"),
        }

    if n_jobs != 1 and Parallel is not None:
        reports = Parallel(n_jobs=n_jobs)(delayed(_run_window)(start, end) for start, end in windows)
    else:
        for start, end in windows:
            reports.append(_run_window(start, end))

    def _psi(series: list[float]) -> float | None:
        if len(series) < 2:
            return None
        import numpy as np

        expected = np.array(series[:-1])
        actual = np.array(series[1:])
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if expected.size == 0 or actual.size == 0:
            return None
        bins = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), 6)
        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)
        exp_pct = exp_counts / max(exp_counts.sum(), 1)
        act_pct = act_counts / max(act_counts.sum(), 1)
        mask = (exp_pct > 0) & (act_pct > 0)
        if not mask.any():
            return None
        psi = float(((act_pct[mask] - exp_pct[mask]) * np.log(act_pct[mask] / exp_pct[mask])).sum())
        return psi

    drift_report: dict[str, Any] = {}
    if reports:
        sharpe_series = [entry.get("sharpe") for entry in reports if entry.get("sharpe") is not None]
        cagr_series = [entry.get("cagr") for entry in reports if entry.get("cagr") is not None]
        drift_report["psi_sharpe"] = _psi(sharpe_series)
        drift_report["psi_cagr"] = _psi(cagr_series)
        if ks_2samp is not None and len(sharpe_series) > 3:
            try:
                drift_report["ks_sharpe"] = ks_2samp(sharpe_series[: len(sharpe_series)//2], sharpe_series[len(sharpe_series)//2 :]).statistic  # type: ignore[arg-type]
            except Exception:
                drift_report["ks_sharpe"] = None
        if ks_2samp is not None and len(cagr_series) > 3:
            try:
                drift_report["ks_cagr"] = ks_2samp(cagr_series[: len(cagr_series)//2], cagr_series[len(cagr_series)//2 :]).statistic  # type: ignore[arg-type]
            except Exception:
                drift_report["ks_cagr"] = None

    def _aggregate_metric(key: str) -> dict[str, float]:
        vals = [float(entry.get(key)) for entry in reports if entry.get(key) is not None]
        if not vals:
            return {}
        arr = np.array(vals, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "iqr": float(q3 - q1),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    summary = {
        "sharpe": _aggregate_metric("sharpe"),
        "cagr": _aggregate_metric("cagr"),
        "max_drawdown": _aggregate_metric("max_drawdown"),
        "var_95": _aggregate_metric("var_95"),
        "cvar_95": _aggregate_metric("cvar_95"),
    }

    ts = datetime.now(datetime.UTC)
    payload = {"generated_at": ts.isoformat(), "reports": reports, "drift_report": drift_report, "summary": summary}
    out_file = output_dir / f"walk_forward_report_{ts.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
