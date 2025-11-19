from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import logging
import os
import time

import requests
from requests import Response, RequestException, Session

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("HTTP_CLIENT_TIMEOUT_SECONDS", "8"))
DEFAULT_MAX_RETRIES = int(os.environ.get("HTTP_CLIENT_MAX_RETRIES", "2"))
DEFAULT_BACKOFF_SECONDS = float(os.environ.get("HTTP_CLIENT_BACKOFF_SECONDS", "0.6"))
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
                    break
                time.sleep(config.backoff * (attempt + 1))

        raise HttpClientError(str(last_exc) if last_exc else "HTTP request failed")

    def get(self, url: str, **kwargs: Any) -> Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response:
        return self.request("POST", url, **kwargs)


http_client = HttpClient()

__all__ = ["http_client", "HttpClientError"]
