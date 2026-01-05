from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import logging
import os
import time
from urllib.parse import urlsplit

import requests
from requests import Response, RequestException, Session

from .network import DEFAULT_BACKOFF_SECONDS, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_SECONDS

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_AGENT = os.environ.get("HTTP_CLIENT_USER_AGENT", "QuantTradingWeb/1.0 (+https://quant.local)")


class HttpClientError(RuntimeError):
    """Raised when the HTTP client exhausts retries."""


@dataclass(slots=True)
class HttpRequestConfig:
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    retries: int = DEFAULT_MAX_RETRIES
    backoff: float = DEFAULT_BACKOFF_SECONDS
    headers: Mapping[str, str] | None = None
    stream: bool = False
    raise_for_status: bool = True


class HttpClient:
    """Thin wrapper around requests with retry/backoff and sane defaults."""

    def __init__(self) -> None:
        self._session: Session = requests.Session()

    def request(
        self,
        method: str,
        url: str,
        *,
        timeout: float | None = None,
        retries: int | None = None,
        backoff: float | None = None,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
        raise_for_status: bool = True,
        emit_metrics: bool = True,
        metric_event: str = "http.client.request",
        request_id: str | None = None,
        metric_tags: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Response:
        config = HttpRequestConfig(
            timeout=timeout or DEFAULT_TIMEOUT_SECONDS,
            retries=DEFAULT_MAX_RETRIES if retries is None else max(0, retries),
            backoff=DEFAULT_BACKOFF_SECONDS if backoff is None else max(0.0, backoff),
            headers=headers,
            stream=stream,
            raise_for_status=raise_for_status,
        )
        merged_headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json, application/*+json;q=0.8, */*;q=0.5",
        }
        if headers:
            merged_headers.update(headers)

        last_exc: Exception | None = None
        started = time.perf_counter()
        url_parts = urlsplit(url)
        host = url_parts.netloc
        path = url_parts.path or "/"
        attempts = config.retries + 1
        for attempt in range(attempts):
            try:
                response = self._session.request(
                    method,
                    url,
                    timeout=config.timeout,
                    headers=merged_headers,
                    stream=config.stream,
                    **kwargs,
                )
                if config.raise_for_status:
                    response.raise_for_status()
                if emit_metrics:
                    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
                    self._record_metric(
                        metric_event,
                        method=method.upper(),
                        host=host,
                        path=path,
                        status_code=response.status_code,
                        duration_ms=duration_ms,
                        success=True,
                        attempts=attempt + 1,
                        request_id=request_id,
                        **(metric_tags or {}),
                    )
                return response
            except RequestException as exc:  # pragma: no cover - network dependent
                last_exc = exc
                LOGGER.warning(
                    "HTTP %s %s failed (attempt %s/%s): %s",
                    method.upper(),
                    url,
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt >= attempts - 1:
                    if emit_metrics:
                        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
                        status_code = None
                        try:
                            status_code = getattr(exc.response, "status_code", None)
                        except Exception:
                            status_code = None
                        self._record_metric(
                            metric_event,
                            method=method.upper(),
                            host=host,
                            path=path,
                            status_code=status_code,
                            duration_ms=duration_ms,
                            success=False,
                            attempts=attempt + 1,
                            error=str(exc),
                            request_id=request_id,
                            **(metric_tags or {}),
                        )
                    break
                time.sleep(config.backoff * (attempt + 1))

        raise HttpClientError(str(last_exc) if last_exc else "HTTP request failed")

    @staticmethod
    def _record_metric(event: str, **fields: Any) -> None:
        try:
            from .observability import record_metric
        except Exception:
            return
        record_metric(event, **fields)

    def get(self, url: str, **kwargs: Any) -> Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response:
        return self.request("POST", url, **kwargs)


http_client = HttpClient()

__all__ = ["http_client", "HttpClientError"]
