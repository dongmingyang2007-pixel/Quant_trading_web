from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable
from pathlib import Path
import json
import time
from datetime import datetime
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


def fetch_latest_quote(symbol: str, *, interval: str = "1m") -> dict[str, object]:
    """Fetch the latest quote for a symbol. Returns {'price': float, 'as_of': datetime}."""
    if not symbol:
        return {}
    if yf is None:  # pragma: no cover - network dependency
        return {}
    try:
        data = yf.download(
            tickers=symbol,
            period="5d",
            interval=interval,
            progress=False,
            threads=False,
        )
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Prefer Close/Adj Close if available
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            if isinstance(data.columns, pd.MultiIndex):
                # When multi-index, first level is field
                if col in data.columns.get_level_values(0):
                    try:
                        series = data.xs(col, level=0, axis=1)
                        if isinstance(series, pd.DataFrame):
                            series = series.iloc[:, 0]
                        price = float(series.dropna().iloc[-1])
                        ts = pd.to_datetime(series.dropna().index[-1])
                        return {"price": price, "as_of": ts.to_pydatetime()}
                    except Exception:
                        pass
            else:
                series = data[col].dropna() if col in data.columns else data.iloc[:, 0].dropna()
                if not series.empty:
                    price = float(series.iloc[-1])
                    ts = pd.to_datetime(series.index[-1])
                    return {"price": price, "as_of": ts.to_pydatetime()}
    except Exception:
        pass
    # Fallback to end-of-day fetch
    try:
        frame = fetch([symbol], period="5d", interval="1d", cache=False)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            try:
                if isinstance(frame.columns, pd.MultiIndex):
                    level0 = frame.columns.get_level_values(0)
                    col = "Adj Close" if "Adj Close" in level0 else level0[0]
                    series = frame.xs(col, level=0, axis=1).iloc[:, 0]
                else:
                    series = frame.iloc[:, 0]
            except Exception:
                series = frame.iloc[:, 0] if not isinstance(frame, pd.MultiIndex) else pd.Series(dtype=float)
            series = series.dropna()
            if not series.empty:
                ts = pd.to_datetime(series.index[-1])
                return {"price": float(series.iloc[-1]), "as_of": ts.to_pydatetime()}
    except Exception:
        pass
    return {}
