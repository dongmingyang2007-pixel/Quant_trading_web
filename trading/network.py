from __future__ import annotations

from dataclasses import dataclass
import os
import random
import threading
import time
from typing import Any, Callable, TypeVar

import requests

_T = TypeVar("_T")


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


DEFAULT_TIMEOUT_SECONDS = _coerce_float(
    os.environ.get("NETWORK_TIMEOUT_SECONDS", os.environ.get("HTTP_CLIENT_TIMEOUT_SECONDS", "8")),
    8.0,
)
DEFAULT_MAX_RETRIES = _coerce_int(
    os.environ.get("NETWORK_MAX_RETRIES", os.environ.get("HTTP_CLIENT_MAX_RETRIES", "2")),
    2,
)
DEFAULT_BACKOFF_SECONDS = _coerce_float(
    os.environ.get("NETWORK_BACKOFF_SECONDS", os.environ.get("HTTP_CLIENT_BACKOFF_SECONDS", "0.6")),
    0.6,
)
DEFAULT_RETRY_JITTER = _coerce_float(os.environ.get("NETWORK_RETRY_JITTER", "0"), 0.0)
SESSION_SCOPE = os.environ.get("NETWORK_SESSION_SCOPE", "thread").lower()


@dataclass(frozen=True)
class RetryConfig:
    timeout: float
    retries: int
    backoff: float
    jitter: float


def resolve_retry_config(
    *,
    timeout: object | None = None,
    retries: object | None = None,
    backoff: object | None = None,
    jitter: object | None = None,
    default_timeout: float | None = None,
    default_retries: int | None = None,
    default_backoff: float | None = None,
    default_jitter: float | None = None,
) -> RetryConfig:
    resolved_timeout = _coerce_float(
        timeout,
        default_timeout if default_timeout is not None else DEFAULT_TIMEOUT_SECONDS,
    )
    resolved_retries = _coerce_int(
        retries,
        default_retries if default_retries is not None else DEFAULT_MAX_RETRIES,
    )
    resolved_backoff = _coerce_float(
        backoff,
        default_backoff if default_backoff is not None else DEFAULT_BACKOFF_SECONDS,
    )
    resolved_jitter = _coerce_float(
        jitter,
        default_jitter if default_jitter is not None else DEFAULT_RETRY_JITTER,
    )
    return RetryConfig(
        timeout=resolved_timeout,
        retries=max(0, resolved_retries),
        backoff=max(0.0, resolved_backoff),
        jitter=max(0.0, resolved_jitter),
    )


def _sleep_with_backoff(attempt: int, config: RetryConfig) -> None:
    delay = config.backoff * (attempt + 1)
    if config.jitter:
        delay += random.random() * config.jitter
    try:
        time.sleep(delay)
    except Exception:
        pass


def retry_call(
    func: Callable[[], _T],
    *,
    config: RetryConfig,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> _T:
    last_exc: Exception | None = None
    attempts = max(0, config.retries) + 1
    for attempt in range(attempts):
        try:
            return func()
        except exceptions as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                break
            _sleep_with_backoff(attempt, config)
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_call exhausted without exception")


def retry_call_result(
    func: Callable[[], _T],
    *,
    config: RetryConfig,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    should_retry: Callable[[Any], bool] | None = None,
) -> _T:
    last_exc: Exception | None = None
    last_result: _T | None = None
    attempts = max(0, config.retries) + 1
    for attempt in range(attempts):
        try:
            result = func()
            last_result = result
            if should_retry and should_retry(result):
                if attempt >= attempts - 1:
                    return result
                _sleep_with_backoff(attempt, config)
                continue
            return result
        except exceptions as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                break
            _sleep_with_backoff(attempt, config)
    if last_exc:
        raise last_exc
    return last_result  # type: ignore[return-value]


class _TimeoutSession(requests.Session):
    def __init__(self, timeout: float):
        super().__init__()
        self._timeout = timeout

    def request(self, method: str, url: str, **kwargs):  # type: ignore[override]
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._timeout
        return super().request(method, url, **kwargs)


_GLOBAL_SESSION_LOCK = threading.Lock()
_GLOBAL_SESSIONS: dict[float, requests.Session] = {}
_THREAD_LOCAL = threading.local()


def _thread_sessions() -> dict[float, requests.Session]:
    cache = getattr(_THREAD_LOCAL, "sessions", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.sessions = cache
    return cache


def get_requests_session(timeout: float) -> requests.Session:
    normalized = float(timeout)
    if SESSION_SCOPE == "global":
        with _GLOBAL_SESSION_LOCK:
            session = _GLOBAL_SESSIONS.get(normalized)
            if session is None:
                session = _TimeoutSession(normalized)
                _GLOBAL_SESSIONS[normalized] = session
            return session
    cache = _thread_sessions()
    session = cache.get(normalized)
    if session is None:
        session = _TimeoutSession(normalized)
        cache[normalized] = session
    return session

