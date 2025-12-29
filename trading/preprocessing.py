from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json

import numpy as np
import pandas as pd

from django.conf import settings

DATA_CACHE_DIR = settings.DATA_CACHE_DIR
FEATURE_STORE_DIR = DATA_CACHE_DIR / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DataQualityReport:
    """Simple container summarising what the sanitizer changed."""

    dropped_duplicates: int = 0
    filled_gaps: int = 0
    clipped_outliers: int = 0
    total_rows: int = 0
    start_date: str = ""
    end_date: str = ""
    missing_ratio: float = 0.0
    zero_volume_days: int = 0
    stale_price_days: int = 0
    notes: list[str] = field(default_factory=list)

    def extend_notes(self, *entries: str) -> None:
        for entry in entries:
            if entry:
                self.notes.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped_duplicates": self.dropped_duplicates,
            "filled_gaps": self.filled_gaps,
            "clipped_outliers": self.clipped_outliers,
            "total_rows": self.total_rows,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "missing_ratio": round(self.missing_ratio, 4),
            "zero_volume_days": self.zero_volume_days,
            "stale_price_days": self.stale_price_days,
            "notes": self.notes,
        }


def _winsorize_extreme_returns(returns: pd.Series, threshold: float) -> pd.Series:
    """Clip extreme returns based on a percentile threshold."""
    if returns.dropna().empty:
        return returns
    lo = returns.quantile(threshold)
    hi = returns.quantile(1 - threshold)
    return returns.clip(lower=lo, upper=hi)


def sanitize_price_history(prices: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean historical OHLCV data before feature engineering.

    Steps:
    - enforce chronological order & drop duplicate dates
    - forward/back fill missing OHLC values and volumes
    - smooth extreme jumps in adjusted close via winsorised returns
    - add rolling z-score of prices for downstream models
    """
    report = DataQualityReport()
    if prices.empty:
        return prices, report

    clean = prices.copy().sort_index()
    clean.attrs = dict(getattr(prices, "attrs", {}))
    report.total_rows = int(clean.shape[0])
    if not clean.empty and hasattr(clean.index[0], "date"):
        report.start_date = str(clean.index[0].date())
        report.end_date = str(clean.index[-1].date())
    # Normalize timezone to UTC-naive to avoid duplicated dates when upstream mixes TZs
    try:
        if hasattr(clean.index, "tz") and clean.index.tz is not None:
            clean.index = clean.index.tz_convert("UTC").tz_localize(None)
            report.extend_notes("已将含时区的时间索引统一到 UTC，避免跨时区重复/缺口。")
    except Exception:
        # Safe-guard for non-datetime indexes
        pass

    duplicated = int(clean.index.duplicated().sum())
    if duplicated:
        clean = clean[~clean.index.duplicated(keep="last")]
        report.dropped_duplicates = duplicated
        report.extend_notes(f"检测到 {duplicated} 个重复交易日并已去重。")

    ohlc_cols = ["open", "high", "low", "close", "adj close"]
    price_block = clean[ohlc_cols]
    missing_ratio = float(price_block.isna().mean().mean())
    report.missing_ratio = missing_ratio
    missing_before = int(price_block.isna().sum().sum())
    clean[ohlc_cols] = price_block.ffill().bfill()
    volume_before = int(clean["volume"].isna().sum())
    clean["volume"] = clean["volume"].ffill().bfill().fillna(0.0)
    report.zero_volume_days = int((clean["volume"] <= 0).sum())
    missing_after = int(clean[ohlc_cols].isna().sum().sum())
    filled = max(0, missing_before + volume_before - missing_after - int(clean["volume"].isna().sum()))
    if filled:
        report.filled_gaps = filled
        report.extend_notes(f"填补 {filled} 个缺失的价格或成交量数据点。")

    returns = clean["adj close"].pct_change().replace([np.inf, -np.inf], np.nan)
    clipped = _winsorize_extreme_returns(returns, threshold=0.01)
    outlier_mask = (~returns.isna()) & (returns.ne(clipped))
    outlier_count = int(outlier_mask.sum())
    if outlier_count:
        # Replace the offending prices by integrating the clipped returns
        ratio = (clean["close"] / clean["adj close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        spreads = {col: clean[col] - clean["close"] for col in ("open", "high", "low")}
        adj = clean["adj close"].iloc[0]
        cumulative = (1 + clipped.fillna(0)).cumprod() * adj
        clean["adj close"] = cumulative
        clean["close"] = clean["adj close"] * ratio
        for col, spread in spreads.items():
            clean[col] = clean["close"] + spread
        report.clipped_outliers = outlier_count
        report.extend_notes(f"平滑 {outlier_count} 个极端价格跳变，降低噪声影响。")

    rolling_mean = clean["adj close"].rolling(64, min_periods=20).mean()
    rolling_std = clean["adj close"].rolling(64, min_periods=20).std().replace(0, np.nan)
    clean["price_z"] = ((clean["adj close"] - rolling_mean) / rolling_std).fillna(0.0)
    clean["adv"] = (clean["volume"] * clean["adj close"]).rolling(20, min_periods=5).mean()

    # Flag stale prices (no change) to warn about halts/illiquidity without mutating data
    try:
        flat_diff = clean["adj close"].diff().abs()
        change_groups = (flat_diff >= 1e-9).astype(int).cumsum()
        flat_runs = (flat_diff < 1e-9).astype(int).groupby(change_groups).cumsum()
        max_flat = int(flat_runs.max() or 0)
        report.stale_price_days = int((flat_diff < 1e-9).sum())
        if max_flat >= 5:
            report.extend_notes(f"发现最长 {max_flat} 天价格无变动，可能存在停牌或极度缺乏流动性。")
    except Exception:
        pass

    # Liquidity health note: share how many days sit near the bottom of ADV distribution
    try:
        adv = clean["adv"].dropna()
        if not adv.empty:
            median_adv = float(adv.median())
            if median_adv > 0:
                low_adv_mask = adv < median_adv * 0.1
                low_adv_ratio = low_adv_mask.mean()
                if low_adv_ratio > 0.05:
                    report.extend_notes(f"约 {low_adv_ratio*100:.1f}% 的交易日成交额低于中位数的 10%，信号可靠性可能受限。")
    except Exception:
        pass

    return clean, report


def _hash_frame(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    subset = frame
    if columns:
        existing = [col for col in columns if col in frame.columns]
        subset = frame[existing]
    hashed = pd.util.hash_pandas_object(subset, index=True).values  # type: ignore[attr-defined]
    return hashlib.sha1(hashed.tobytes()).hexdigest()


class FeatureStore:
    """Lightweight on-disk cache for expensive feature matrices."""

    def __init__(self, root: Path | None = None):
        self.root = root or FEATURE_STORE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _data_path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def _meta_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def fingerprint(self, prices: pd.DataFrame, params: Any) -> str:
        payload = {
            "ticker": getattr(params, "ticker", "").upper(),
            "short": getattr(params, "short_window", None),
            "long": getattr(params, "long_window", None),
            "rsi": getattr(params, "rsi_period", None),
            "label": getattr(params, "label_style", ""),
            "return_path": getattr(params, "return_path", ""),
            "label_return_path": getattr(params, "label_return_path", None),
            "tb": (
                getattr(params, "tb_up", None),
                getattr(params, "tb_down", None),
                getattr(params, "tb_max_holding", None),
                getattr(params, "tb_dynamic", None),
                getattr(params, "tb_vol_window", None),
                getattr(params, "tb_vol_multiplier", None),
            ),
            "class_weight_mode": getattr(params, "class_weight_mode", None),
            "price_hash": _hash_frame(prices, ["adj close", "volume", "open", "high", "low"]),
            "rows": int(prices.shape[0]),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()

    def load(self, key: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]] | None:
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            dataset = None
            if data_path.suffix == ".parquet":
                try:
                    dataset = pd.read_parquet(data_path)
                except Exception:
                    pkl_path = data_path.with_suffix(".pkl")
                    if pkl_path.exists():
                        dataset = pd.read_pickle(pkl_path)
            if dataset is None:
                dataset = pd.read_pickle(data_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            features = list(meta.get("feature_columns", []))
            return dataset, features, meta
        except Exception:
            return None

    def save(
        self,
        key: str,
        dataset: pd.DataFrame,
        feature_columns: list[str],
        pipeline_signature: str | None = None,
    ) -> None:
        data_path = self._data_path(key).with_suffix(".parquet")
        meta_path = self._meta_path(key)
        tmp_data = data_path.with_suffix(".tmp")
        tmp_meta = meta_path.with_suffix(".tmp")
        try:
            dataset.to_parquet(tmp_data, index=True)
            used_path = data_path
            meta = {
                "feature_columns": feature_columns,
                "rows": int(dataset.shape[0]),
                "label": dataset.get("label_style") if "label_style" in dataset.columns else None,
                "return_path": dataset.get("return_path") if "return_path" in dataset.columns else None,
                "window_signature": dataset.get("window_signature") if "window_signature" in dataset.columns else None,
            }
            if pipeline_signature:
                meta["pipeline_signature"] = pipeline_signature
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            tmp_data.replace(used_path)
            tmp_meta.replace(meta_path)
        finally:
            if tmp_data.exists():
                tmp_data.unlink()
            if tmp_meta.exists():
                tmp_meta.unlink()
