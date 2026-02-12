from __future__ import annotations

import csv
import gzip
import io
import os
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings

from .file_utils import atomic_write_text
from .massive_data import resolve_massive_s3_credentials

try:  # pragma: no cover - optional dependency
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore


FLATFILES_BUCKET = os.environ.get("MASSIVE_FLATFILES_BUCKET", "flatfiles")
FLATFILES_ENDPOINT_URL = os.environ.get("MASSIVE_FLATFILES_ENDPOINT_URL", "https://files.polygon.io")
FLATFILES_REGION = os.environ.get("MASSIVE_FLATFILES_REGION", "us-east-1")
FLATFILES_MINUTE_KEY_TEMPLATE = os.environ.get(
    "MASSIVE_FLATFILES_MINUTE_KEY_TEMPLATE",
    "us_stocks_sip/minute_aggs_v1/{year:04d}/{month:02d}/{date}.csv.gz",
)

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("ticker", "symbol", "sym", "T", "S"),
    "open": ("open", "o", "Open"),
    "high": ("high", "h", "High"),
    "low": ("low", "l", "Low"),
    "close": ("close", "c", "Close"),
    "volume": ("volume", "v", "Volume", "size"),
    "time": ("window_start", "timestamp", "t", "sip_timestamp"),
}


def _cache_root() -> Path:
    base = Path(getattr(settings, "DATA_CACHE_DIR", Path(getattr(settings, "BASE_DIR", Path.cwd())) / "data_cache"))
    root = base / "massive_flatfiles" / "symbol"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _today_start_utc() -> datetime:
    # Resolve via zoneinfo only when available in runtime.
    try:
        from zoneinfo import ZoneInfo

        now_et = datetime.now(ZoneInfo("America/New_York"))
        return now_et.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    except Exception:
        now_utc = datetime.now(timezone.utc)
        return now_utc.replace(hour=0, minute=0, second=0, microsecond=0)


def _iter_weekdays(start_date: date_cls, end_date: date_cls) -> list[date_cls]:
    days: list[date_cls] = []
    cursor = start_date
    while cursor <= end_date:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _symbol_day_cache_paths(symbol: str, day: date_cls) -> tuple[Path, Path]:
    sym = _normalize_symbol(symbol)
    symbol_dir = _cache_root() / sym
    symbol_dir.mkdir(parents=True, exist_ok=True)
    stamp = day.isoformat()
    return symbol_dir / f"{stamp}.csv", symbol_dir / f"{stamp}.empty"


def _pick_value(row: dict[str, Any], kind: str) -> Any:
    for key in _FIELD_ALIASES.get(kind, ()):
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_ts(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            val = float(text)
        except ValueError:
            ts = pd.to_datetime(text, utc=True, errors="coerce")
            if ts is None or pd.isna(ts):
                return None
            return float(ts.timestamp())
    if val > 10_000_000_000_000:
        return val / 1_000_000_000.0
    if val > 10_000_000_000:
        return val / 1000.0
    return val


def _to_float(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if pd.isna(value):
        return None
    return value


def _read_cached_day(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(cache_path)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    if "time" not in frame.columns:
        return pd.DataFrame()
    try:
        time_index = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    except Exception:
        return pd.DataFrame()
    frame = frame.drop(columns=["time"], errors="ignore")
    frame.index = time_index
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    return frame.sort_index()


def _write_cached_day(cache_path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        atomic_write_text(cache_path, "time,Open,High,Low,Close,Volume\n")
        return
    output = frame.copy()
    output = output.sort_index()
    output.insert(0, "time", output.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    text_buffer = io.StringIO()
    output.to_csv(text_buffer, index=False)
    atomic_write_text(cache_path, text_buffer.getvalue())


def _resolve_flatfiles_config() -> tuple[str, str, str, str]:
    bucket = str(getattr(settings, "MASSIVE_FLATFILES_BUCKET", FLATFILES_BUCKET) or FLATFILES_BUCKET).strip() or FLATFILES_BUCKET
    endpoint = str(
        getattr(settings, "MASSIVE_FLATFILES_ENDPOINT_URL", FLATFILES_ENDPOINT_URL) or FLATFILES_ENDPOINT_URL
    ).strip() or FLATFILES_ENDPOINT_URL
    region = str(getattr(settings, "MASSIVE_FLATFILES_REGION", FLATFILES_REGION) or FLATFILES_REGION).strip() or FLATFILES_REGION
    key_template = str(
        getattr(settings, "MASSIVE_FLATFILES_MINUTE_KEY_TEMPLATE", FLATFILES_MINUTE_KEY_TEMPLATE)
        or FLATFILES_MINUTE_KEY_TEMPLATE
    ).strip() or FLATFILES_MINUTE_KEY_TEMPLATE
    return bucket, endpoint, region, key_template


def _build_s3_client(*, user_id: str | None = None) -> tuple[Any | None, str | None]:
    if boto3 is None:
        return None, "boto3_missing"
    access_key_id, secret_access_key = resolve_massive_s3_credentials(user_id=user_id)
    if not access_key_id or not secret_access_key:
        return None, "credentials_missing"
    _bucket, endpoint, region, _key_template = _resolve_flatfiles_config()
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint,
            region_name=region,
        )
    except Exception:
        return None, "client_init_failed"
    return client, None


def _day_key(day: date_cls) -> str:
    _bucket, _endpoint, _region, key_template = _resolve_flatfiles_config()
    return key_template.format(
        date=day.isoformat(),
        year=day.year,
        month=day.month,
        day=day.day,
    )


def _download_day_rows(
    client: Any,
    *,
    symbol: str,
    day: date_cls,
) -> tuple[list[dict[str, float]], str | None]:
    bucket, _endpoint, _region, _key_template = _resolve_flatfiles_config()
    key = _day_key(day)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return [], "object_fetch_failed"
    body = obj.get("Body")
    if body is None:
        return [], "empty_body"

    rows: list[dict[str, float]] = []
    sym = _normalize_symbol(symbol)
    try:
        with gzip.GzipFile(fileobj=body) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", errors="ignore") as text_stream:
                reader = csv.DictReader(text_stream)
                for raw in reader:
                    row = dict(raw) if isinstance(raw, dict) else {}
                    ticker = _normalize_symbol(str(_pick_value(row, "symbol") or ""))
                    if ticker != sym:
                        continue
                    ts_val = _parse_ts(_pick_value(row, "time"))
                    if ts_val is None:
                        continue
                    open_val = _to_float(_pick_value(row, "open"))
                    high_val = _to_float(_pick_value(row, "high"))
                    low_val = _to_float(_pick_value(row, "low"))
                    close_val = _to_float(_pick_value(row, "close"))
                    if None in (open_val, high_val, low_val, close_val):
                        continue
                    volume_val = _to_float(_pick_value(row, "volume")) or 0.0
                    rows.append(
                        {
                            "time": float(ts_val),
                            "Open": float(open_val),
                            "High": float(high_val),
                            "Low": float(low_val),
                            "Close": float(close_val),
                            "Volume": float(volume_val),
                        }
                    )
    except Exception:
        return [], "decompress_failed"
    return rows, None


def _rows_to_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    time_index = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
    frame = frame.drop(columns=["time"], errors="ignore")
    frame.index = time_index
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    return frame.sort_index()


def _resample_frame(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    text = str(interval or "1m").strip().lower()
    if text in {"1m", "1min", "1minute"}:
        return frame
    multiplier = 1
    unit = "m"
    digits = ""
    suffix = ""
    for ch in text:
        if ch.isdigit() and not suffix:
            digits += ch
        elif ch.isalpha():
            suffix += ch
    if digits:
        try:
            multiplier = max(1, int(digits))
        except ValueError:
            multiplier = 1
    if suffix.startswith("h"):
        unit = "h"
    elif suffix.startswith("d"):
        unit = "d"
    else:
        unit = "m"
    if unit == "h":
        rule = f"{multiplier}h"
    elif unit == "d":
        rule = f"{multiplier}d"
    else:
        rule = f"{multiplier}min"
    aggregated = frame.resample(rule, label="right", closed="right").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    return aggregated


def fetch_historical_bars(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    user_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = _normalize_symbol(symbol)
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    today_start = _today_start_utc()
    history_end = min(end_utc, today_start)
    if not symbol or start_utc >= history_end:
        return pd.DataFrame(), {
            "historical_source": "none",
            "cache_hit": 0,
            "cache_miss": 0,
            "warmed_days": [],
            "history_coverage": {
                "start_ts": None,
                "end_ts": None,
                "complete": True,
            },
            "history_cursor_next": None,
            "error": None,
        }

    client, client_error = _build_s3_client(user_id=user_id)
    start_date = start_utc.date()
    # history_end itself is exclusive.
    end_date = (history_end - timedelta(seconds=1)).date()
    trade_days = _iter_weekdays(start_date, end_date)

    cache_hit = 0
    cache_miss = 0
    warmed_days: list[str] = []
    error: str | None = client_error
    frames: list[pd.DataFrame] = []

    for day in trade_days:
        cache_path, empty_path = _symbol_day_cache_paths(symbol, day)
        if cache_path.exists():
            cache_hit += 1
            frame_day = _read_cached_day(cache_path)
            if not frame_day.empty:
                frames.append(frame_day)
            continue
        if empty_path.exists():
            cache_hit += 1
            continue
        cache_miss += 1
        warmed_days.append(day.isoformat())
        if client is None:
            continue
        rows, day_error = _download_day_rows(client, symbol=symbol, day=day)
        if day_error and error is None:
            error = day_error
        if not rows:
            atomic_write_text(empty_path, "")
            continue
        frame_day = _rows_to_frame(rows)
        if frame_day.empty:
            atomic_write_text(empty_path, "")
            continue
        _write_cached_day(cache_path, frame_day)
        frames.append(frame_day)

    if frames:
        frame = pd.concat(frames, axis=0)
        frame = frame.sort_index()
        frame = frame[(frame.index >= start_utc) & (frame.index < history_end)]
        frame = _resample_frame(frame, interval)
    else:
        frame = pd.DataFrame()

    history_start_ts = None
    history_end_ts = None
    if not frame.empty:
        history_start_ts = float(pd.Timestamp(frame.index[0]).timestamp())
        history_end_ts = float(pd.Timestamp(frame.index[-1]).timestamp())

    coverage_complete = (error is None) or (cache_miss == 0)
    if cache_miss == 0:
        error = None
    history_cursor_next = None
    if trade_days:
        history_cursor_next = datetime.combine(trade_days[0], datetime.min.time(), tzinfo=timezone.utc).timestamp()

    frame.attrs["market_source"] = "massive"
    frame.attrs["historical_source"] = "massive_flatfiles"
    return frame, {
        "historical_source": "massive_flatfiles",
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "warmed_days": warmed_days,
        "history_coverage": {
            "start_ts": history_start_ts,
            "end_ts": history_end_ts,
            "complete": coverage_complete,
        },
        "history_cursor_next": history_cursor_next,
        "error": error,
    }
