from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Iterable

import pandas as pd
import requests
from django.conf import settings

from .cache_utils import build_cache_key, cache_memoize
from .network import get_requests_session, resolve_retry_config, retry_call_result
from trading.observability import record_metric
from .alpaca_data import (
    DEFAULT_DATA_URL,
    DEFAULT_FEED,
    fetch_stock_bars_frame,
    fetch_stock_snapshots,
    resolve_alpaca_data_credentials,
)

RATE_LIMIT_PER_WINDOW = int(os.environ.get("MARKET_FETCH_RATE_LIMIT", "10000") or 0)
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("MARKET_FETCH_RATE_WINDOW", "60") or 0)

_RATE_LOCK = threading.Lock()
_RATE_BUCKET: deque[float] = deque()
LOGGER = logging.getLogger(__name__)
_PARQUET_AVAILABLE: bool | None = None
_PARQUET_WARNED = False

DATA_CACHE_DIR: Path = getattr(settings, "DATA_CACHE_DIR", Path(settings.DATA_ROOT) / "data_cache")
DISK_CACHE_DIR = DATA_CACHE_DIR / "market_snapshots"
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISK_CACHE_TTL_SECONDS = int(getattr(settings, "MARKET_DISK_CACHE_TTL", 6 * 3600))
try:
    LEGACY_CACHE_FILENAME_MAX_BYTES = int(getattr(settings, "MARKET_LEGACY_CACHE_FILENAME_MAX_BYTES", 240))
except (TypeError, ValueError):
    LEGACY_CACHE_FILENAME_MAX_BYTES = 240

_ALPACA_INTERVAL_MAP = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "60m": "1Hour",
    "1h": "1Hour",
    "1d": "1Day",
}


def _has_parquet_engine() -> bool:
    global _PARQUET_AVAILABLE
    if _PARQUET_AVAILABLE is not None:
        return _PARQUET_AVAILABLE
    for module in ("pyarrow", "fastparquet"):
        try:
            __import__(module)
            _PARQUET_AVAILABLE = True
            return True
        except Exception:
            continue
    _PARQUET_AVAILABLE = False
    return False


def _warn_parquet_missing(context: str) -> None:
    global _PARQUET_WARNED
    if _PARQUET_WARNED:
        return
    LOGGER.warning(
        "Parquet engine missing; %s skipped. Install pyarrow>=14 or fastparquet.",
        context,
    )
    _PARQUET_WARNED = True


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        df.to_parquet(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _market_retry_config(timeout: int | None = None):
    return resolve_retry_config(
        timeout=timeout,
        retries=os.environ.get("MARKET_FETCH_MAX_RETRIES"),
        backoff=os.environ.get("MARKET_FETCH_RETRY_BACKOFF"),
        default_timeout=getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", None),
    )


def _empty_frame(value: object) -> bool:
    return not isinstance(value, pd.DataFrame) or value.empty


def _rate_limited() -> bool:
    """Simple in-memory rate limiter: returns True if allowed, False if over budget."""
    if RATE_LIMIT_PER_WINDOW <= 0 or RATE_LIMIT_WINDOW_SECONDS <= 0:
        return True
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LOCK:
        # drop old
        while _RATE_BUCKET and _RATE_BUCKET[0] < window_start:
            _RATE_BUCKET.popleft()
        if len(_RATE_BUCKET) >= RATE_LIMIT_PER_WINDOW:
            LOGGER.warning("market_data rate limit hit: %s in %ss", len(_RATE_BUCKET), RATE_LIMIT_WINDOW_SECONDS)
            record_metric("market_rate_limited", count=len(_RATE_BUCKET), window=RATE_LIMIT_WINDOW_SECONDS)
            return False
        _RATE_BUCKET.append(now)
        return True


def _resolve_alpaca_timeframe(interval: str | None) -> str:
    if not interval:
        return "1Day"
    return _ALPACA_INTERVAL_MAP.get(interval.lower().strip(), "1Day")


def _feed_candidates(preferred: str | None = None) -> list[str]:
    primary = (preferred or DEFAULT_FEED or "").strip().lower()
    candidates: list[str] = []
    if primary:
        candidates.append(primary)
    for fallback in ("sip", "iex"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _period_to_timedelta(period: str | None) -> timedelta | None:
    if not period:
        return None
    text = str(period).strip().lower()
    if not text:
        return None
    if text.endswith("mo"):
        try:
            value = int(text[:-2])
        except ValueError:
            return None
        return timedelta(days=max(1, value) * 30)
    if text.endswith("y"):
        try:
            value = int(text[:-1])
        except ValueError:
            return None
        return timedelta(days=max(1, value) * 365)
    if text.endswith("d"):
        try:
            value = int(text[:-1])
        except ValueError:
            return None
        return timedelta(days=max(1, value))
    return None


def _resolve_alpaca_range(
    *,
    period: str | None,
    start: date | datetime | None,
    end: date | datetime | None,
    timeframe: str,
) -> tuple[date | datetime | None, date | datetime | None]:
    start_val = start
    end_val = end
    if not start_val and period:
        delta = _period_to_timedelta(period)
        if delta is not None:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - delta
            start_val = start_dt
            end_val = end_dt
    if isinstance(end_val, date) and not isinstance(end_val, datetime):
        end_val = end_val + timedelta(days=1)
    return start_val, end_val


def fetch_most_actives(
    *,
    by: str = "volume",
    limit: int = 20,
    user_id: str | None = None,
    timeout: int | None = None,
    base_url: str | None = None,
) -> list[dict[str, object]]:
    key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
    if not key_id or not secret:
        return []
    if not _rate_limited():
        return []

    params: dict[str, object] = {"by": by}
    if limit:
        params["top"] = int(limit)
    url = f"{(base_url or DEFAULT_DATA_URL).rstrip('/')}/v1beta1/screener/stocks/most-actives"
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }

    retry_config = resolve_retry_config(timeout=timeout)
    session = get_requests_session(retry_config.timeout)

    def _call():
        return session.get(url, params=params, headers=headers, timeout=retry_config.timeout)

    def _should_retry(response: requests.Response) -> bool:
        return response.status_code in {408, 429} or response.status_code >= 500

    try:
        response = retry_call_result(
            _call,
            config=retry_config,
            exceptions=(requests.RequestException,),
            should_retry=_should_retry,
        )
    except Exception:
        return []
    if response is None or response.status_code >= 400:
        return []
    try:
        payload = response.json()
    except ValueError:
        return []

    rows: list[dict[str, object]] = []
    items = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("most_actives", "mostActives", "data", "results", "stocks"):
            if isinstance(payload.get(key), list):
                items = payload.get(key)
                break
        if items is None:
            for key in ("most_actives_by_volume", "most_actives_by_trade_count"):
                if isinstance(payload.get(key), list):
                    items = payload.get(key)
                    break
    if not isinstance(items, list):
        return []

    for row in items:
        if not isinstance(row, dict):
            continue
        symbol = (row.get("symbol") or row.get("ticker") or row.get("S") or "").strip().upper()
        if not symbol:
            continue
        volume_raw = row.get("volume") or row.get("v") or row.get("trade_count") or row.get("trades")
        price_raw = row.get("price") or row.get("last_price") or row.get("last") or row.get("close") or row.get("c")
        change_raw = (
            row.get("change_pct")
            or row.get("change_percent")
            or row.get("pct_change")
            or row.get("percent_change")
        )
        volume_val = None
        price_val = None
        change_pct = None
        try:
            if volume_raw is not None:
                volume_val = float(volume_raw)
        except Exception:
            volume_val = None
        try:
            if price_raw is not None:
                price_val = float(price_raw)
        except Exception:
            price_val = None
        try:
            if change_raw is not None:
                change_pct = float(change_raw)
        except Exception:
            change_pct = None
        rows.append(
            {
                "symbol": symbol,
                "price": price_val,
                "change_pct_day": change_pct,
                "change_pct_period": change_pct,
                "volume": volume_val,
            }
        )

    if not rows:
        return []

    missing_symbols = [item["symbol"] for item in rows if item.get("price") is None]
    if missing_symbols:
        snapshots = fetch_stock_snapshots(missing_symbols, feed=DEFAULT_FEED, user_id=user_id)
        for entry in rows:
            if entry.get("price") is not None:
                continue
            snapshot = snapshots.get(entry["symbol"]) if isinstance(snapshots, dict) else None
            if not isinstance(snapshot, dict):
                continue
            latest_trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
            latest_quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
            daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
            prev_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
            price = (
                latest_trade.get("p")
                or daily_bar.get("c")
                or prev_bar.get("c")
                or latest_quote.get("ap")
                or latest_quote.get("bp")
            )
            if price is not None:
                try:
                    entry["price"] = float(price)
                except Exception:
                    pass
            if entry.get("change_pct_day") is None and daily_bar.get("c") is not None and prev_bar.get("c"):
                try:
                    entry["change_pct_day"] = (float(daily_bar["c"]) / float(prev_bar["c"]) - 1.0) * 100.0
                    entry["change_pct_period"] = entry["change_pct_day"]
                except Exception:
                    pass
            if entry.get("volume") is None and daily_bar.get("v") is not None:
                try:
                    entry["volume"] = float(daily_bar["v"])
                except Exception:
                    pass

    rows.sort(key=lambda item: item.get("volume") or 0, reverse=True)
    return rows[:limit] if limit else rows


def _normalize_mover_rows(items: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        symbol = (row.get("symbol") or row.get("ticker") or row.get("S") or "").strip().upper()
        if not symbol:
            continue
        price_raw = row.get("price") or row.get("last_price") or row.get("last") or row.get("close") or row.get("c")
        change_raw = (
            row.get("change_pct")
            or row.get("change_percent")
            or row.get("pct_change")
            or row.get("percent_change")
            or row.get("change")
        )
        volume_raw = row.get("volume") or row.get("v") or row.get("trade_count") or row.get("trades")
        price_val = None
        change_pct = None
        volume_val = None
        try:
            if price_raw is not None:
                price_val = float(price_raw)
        except Exception:
            price_val = None
        try:
            if change_raw is not None:
                change_pct = float(change_raw)
        except Exception:
            change_pct = None
        try:
            if volume_raw is not None:
                volume_val = float(volume_raw)
        except Exception:
            volume_val = None
        rows.append(
            {
                "symbol": symbol,
                "price": price_val,
                "change_pct_day": change_pct,
                "change_pct_period": change_pct,
                "volume": volume_val,
            }
        )
    return rows


def fetch_market_movers(
    *,
    limit: int = 20,
    user_id: str | None = None,
    timeout: int | None = None,
    base_url: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
    if not key_id or not secret:
        return {}
    if not _rate_limited():
        return {}

    params: dict[str, object] = {}
    if limit:
        params["top"] = int(limit)
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    retry_config = resolve_retry_config(timeout=timeout)
    session = get_requests_session(retry_config.timeout)

    def _should_retry(response: requests.Response) -> bool:
        return response.status_code in {408, 429} or response.status_code >= 500

    candidates = [
        "v1beta1/screener/stocks/movers",
        "v1beta1/screener/stocks/market-movers",
        "v1beta1/screener/stocks/market_movers",
        "v1beta1/screener/stocks/top-movers",
    ]
    for path in candidates:
        url = f"{(base_url or DEFAULT_DATA_URL).rstrip('/')}/{path}"

        def _call():
            return session.get(url, params=params, headers=headers, timeout=retry_config.timeout)

        try:
            response = retry_call_result(
                _call,
                config=retry_config,
                exceptions=(requests.RequestException,),
                should_retry=_should_retry,
            )
        except Exception:
            continue
        if response is None or response.status_code >= 400:
            if response is not None and response.status_code in {404}:
                continue
            return {}
        try:
            payload = response.json()
        except ValueError:
            continue

        gainers: list[dict[str, object]] = []
        losers: list[dict[str, object]] = []
        if isinstance(payload, dict):
            for key in ("gainers", "top_gainers", "topGainers", "gainers_data"):
                if isinstance(payload.get(key), list):
                    gainers = _normalize_mover_rows(payload.get(key))
                    break
            for key in ("losers", "top_losers", "topLosers", "losers_data"):
                if isinstance(payload.get(key), list):
                    losers = _normalize_mover_rows(payload.get(key))
                    break
            if not gainers and not losers:
                for key in ("market_movers", "marketMovers", "data", "results", "stocks"):
                    if isinstance(payload.get(key), list):
                        rows = _normalize_mover_rows(payload.get(key))
                        gainers = [row for row in rows if (row.get("change_pct_day") or 0) >= 0]
                        losers = [row for row in rows if (row.get("change_pct_day") or 0) < 0]
                        break
        elif isinstance(payload, list):
            rows = _normalize_mover_rows(payload)
            gainers = [row for row in rows if (row.get("change_pct_day") or 0) >= 0]
            losers = [row for row in rows if (row.get("change_pct_day") or 0) < 0]

        gainers.sort(key=lambda item: item.get("change_pct_day") or 0, reverse=True)
        losers.sort(key=lambda item: item.get("change_pct_day") or 0)
        if gainers or losers:
            return {
                "gainers": gainers[:limit] if limit else gainers,
                "losers": losers[:limit] if limit else losers,
            }

    return {}


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
    user_id: str | None = None,
) -> pd.DataFrame:
    """Fetch market data with simple TTL caching and disk fallback."""

    unique = [sym for sym in dict.fromkeys(symbols) if sym]
    cache_key = build_cache_key("market-data", "alpaca", sorted(unique), period or start, end, interval, fields)
    def _cache_paths(key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        base = digest[:24]
        return DISK_CACHE_DIR / f"{base}.parquet", DISK_CACHE_DIR / f"{base}.json"

    def _candidate_cache_paths(key: str) -> list[tuple[Path, Path]]:
        hashed = _cache_paths(key)
        paths: list[tuple[Path, Path]] = [hashed]
        legacy_parquet_name = f"{key}.parquet"
        legacy_meta_name = f"{key}.json"
        if (
            len(legacy_parquet_name.encode("utf-8")) <= LEGACY_CACHE_FILENAME_MAX_BYTES
            and len(legacy_meta_name.encode("utf-8")) <= LEGACY_CACHE_FILENAME_MAX_BYTES
        ):
            legacy = (DISK_CACHE_DIR / legacy_parquet_name, DISK_CACHE_DIR / legacy_meta_name)
            if hashed != legacy:
                paths.append(legacy)
        return paths

    def _load_disk_cache() -> pd.DataFrame:
        for path, meta_path in _candidate_cache_paths(cache_key):
            try:
                if not path.exists() or not meta_path.exists():
                    continue
            except OSError:
                continue
            try:
                if not _has_parquet_engine():
                    _warn_parquet_missing("market cache read")
                    return pd.DataFrame()
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    continue
                if meta.get("source") != "alpaca":
                    continue
                ts = float(meta.get("timestamp", 0))
                if time.time() - ts > DISK_CACHE_TTL_SECONDS:
                    try:
                        path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue
                df = pd.read_parquet(path)
                if isinstance(df, pd.DataFrame):
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def _persist_disk_cache(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        try:
            if not _has_parquet_engine():
                _warn_parquet_missing("market cache write")
                return
            path, meta_path = _cache_paths(cache_key)
            meta = {
                "timestamp": time.time(),
                "symbols": unique,
                "fields": fields,
                "cache_key": cache_key,
                "source": "alpaca",
            }
            _write_parquet_atomic(df, path)
            _write_json_atomic(meta_path, meta)
        except Exception:
            pass

    if not unique:
        return _load_disk_cache()
    def _builder() -> pd.DataFrame:
        timeframe = _resolve_alpaca_timeframe(interval)
        alpaca_timeout = _market_retry_config(timeout).timeout
        alpaca_start, alpaca_end = _resolve_alpaca_range(
            period=period,
            start=start,
            end=end,
            timeframe=timeframe,
        )
        key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
        if key_id and secret:
            for feed in _feed_candidates(DEFAULT_FEED):
                try:
                    frame = fetch_stock_bars_frame(
                        unique,
                        start=alpaca_start,
                        end=alpaca_end,
                        timeframe=timeframe,
                        feed=feed,
                        adjustment="split",
                        user_id=user_id,
                        timeout=alpaca_timeout,
                    )
                except Exception:
                    continue
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    return frame
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


def _extract_price_from_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Internal helper: pick latest price/timestamp from a price frame."""
    try:
        col = "Adj Close" if "Adj Close" in frame.columns else "Close"
        if isinstance(frame.columns, pd.MultiIndex):
            level0 = frame.columns.get_level_values(0)
            col = "Adj Close" if "Adj Close" in level0 else level0[0]
            series = frame.xs(col, level=0, axis=1)
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
        else:
            series = frame[col] if col in frame.columns else frame.iloc[:, 0]
        series = series.dropna()
        if series.empty:
            return {}
        price = float(series.iloc[-1])
        ts = pd.to_datetime(series.index[-1]).to_pydatetime()
        return {"price": price, "as_of": ts}
    except Exception:
        return {}


def fetch_recent_window(
    symbols: list[str] | tuple[str, ...] | str,
    *,
    interval: str = "1d",
    limit: int = 120,
    user_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Batch-fetch a recent window of price data for one or more symbols."""
    if not symbols:
        return {}
    if isinstance(symbols, str):
        symbols = [symbols]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for sym in symbols:
        if sym and sym not in seen:
            seen.add(sym)
            unique_symbols.append(sym)
    key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
    if key_id and secret:
        timeframe = _resolve_alpaca_timeframe(interval)
        data = pd.DataFrame()
        for feed in _feed_candidates(DEFAULT_FEED):
            try:
                data = fetch_stock_bars_frame(
                    unique_symbols,
                    timeframe=timeframe,
                    limit=limit,
                    feed=feed,
                    adjustment="split",
                    user_id=user_id,
                )
            except Exception:
                data = pd.DataFrame()
            if isinstance(data, pd.DataFrame) and not data.empty:
                break
        if isinstance(data, pd.DataFrame) and not data.empty:
            result: dict[str, pd.DataFrame] = {}
            if isinstance(data.columns, pd.MultiIndex):
                for sym in unique_symbols:
                    try:
                        sub = data.xs(sym, level=1, axis=1).dropna().tail(limit)
                        if not sub.empty:
                            result[sym] = sub
                    except Exception:
                        continue
            else:
                try:
                    frame = data.dropna().tail(limit)
                    if not frame.empty:
                        result[unique_symbols[0]] = frame
                except Exception:
                    pass
            if result:
                return result
    return {}


def fetch_latest_quote(
    symbol: str,
    *,
    interval: str = "1m",
    user_id: str | None = None,
) -> dict[str, object]:
    """Fetch the latest quote for a symbol. Returns {'price': float, 'as_of': datetime}."""
    if not symbol:
        return {}
    key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
    if key_id and secret:
        snapshots = fetch_stock_snapshots([symbol], user_id=user_id)
        snap = snapshots.get(symbol.upper()) if isinstance(snapshots, dict) else None
        if isinstance(snap, dict):
            latest_trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
            latest_quote = snap.get("latestQuote") or snap.get("latest_quote") or {}
            daily_bar = snap.get("dailyBar") or snap.get("daily_bar") or {}
            prev_bar = snap.get("prevDailyBar") or snap.get("prev_daily_bar") or {}
            price = (
                latest_trade.get("p")
                or daily_bar.get("c")
                or prev_bar.get("c")
                or latest_quote.get("ap")
                or latest_quote.get("bp")
            )
            ts = latest_trade.get("t") or latest_quote.get("t") or daily_bar.get("t")
            if price is not None:
                as_of = pd.to_datetime(ts, utc=True, errors="coerce") if ts else None
                as_of_val = as_of.to_pydatetime() if as_of is not None and not pd.isna(as_of) else None
                return {"price": float(price), "as_of": as_of_val, "source": "alpaca"}
    return {}
