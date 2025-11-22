from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable
from pathlib import Path
import json
import time
import hashlib

import pandas as pd

from django.conf import settings

from .cache_utils import build_cache_key, cache_memoize

try:  # optional dependency
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

DATA_CACHE_DIR: Path = getattr(settings, "DATA_CACHE_DIR", Path(settings.DATA_ROOT) / "data_cache")
DISK_CACHE_DIR = DATA_CACHE_DIR / "market_snapshots"
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISK_CACHE_TTL_SECONDS = int(getattr(settings, "MARKET_DISK_CACHE_TTL", 6 * 3600))


def fetch(
    symbols: Iterable[str],
    *,
    period: str | None = None,
    interval: str | None = None,
    start: date | None = None,
    end: date | None = None,
    fields: str = "Adj Close",
    cache: bool = True,
    ttl: int | None = None,
    timeout: int | None = None,
    cache_alias: str | None = None,
) -> pd.DataFrame:
    """Fetch market data with simple TTL caching and graceful fallback."""

    unique = [sym for sym in dict.fromkeys(symbols) if sym]
    cache_key = build_cache_key("market-data", sorted(unique), period or start, end, interval, fields)

    def _cache_paths(key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        base = digest[:24]
        return DISK_CACHE_DIR / f"{base}.parquet", DISK_CACHE_DIR / f"{base}.json"

    def _load_disk_cache() -> pd.DataFrame:
        path, meta_path = _cache_paths(cache_key)
        if not path.exists() or not meta_path.exists():
            return pd.DataFrame()
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ts = float(meta.get("timestamp", 0))
            if time.time() - ts > DISK_CACHE_TTL_SECONDS:
                try:
                    path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return pd.DataFrame()
            df = pd.read_parquet(path)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _persist_disk_cache(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        try:
            path, meta_path = _cache_paths(cache_key)
            df.to_parquet(path)
            meta = {"timestamp": time.time(), "symbols": unique, "fields": fields, "cache_key": cache_key}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass

    if not unique:
        return _load_disk_cache()
    if yf is None:
        fallback = _load_disk_cache()
        return fallback

    def _builder() -> pd.DataFrame:
        try:
            data = yf.download(
                tickers=" ".join(unique),
                period=period,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=timeout or getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", None),
            )
            if not isinstance(data, pd.DataFrame):
                return pd.DataFrame()
            return data
        except Exception:
            return pd.DataFrame()

    if not cache:
        frame = _builder()
        if frame.empty:
            fallback = _load_disk_cache()
            return fallback
        _persist_disk_cache(frame)
        return frame

    cache_ttl = ttl if ttl is not None else getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300)
    result: pd.DataFrame | None = None
    try:
        result = cache_memoize(cache_key, _builder, cache_ttl, cache_alias=cache_alias)  # type: ignore[arg-type]
    except Exception:
        result = _builder()

    if isinstance(result, pd.DataFrame) and not result.empty:
        _persist_disk_cache(result)
        return result
    fallback = _load_disk_cache()
    return fallback if isinstance(fallback, pd.DataFrame) else pd.DataFrame()
