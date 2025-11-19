from __future__ import annotations

import secrets

from django.conf import settings


class SecurityHeadersMiddleware:
    """Attach统一的安全响应头（CSP、Referrer-Policy 等）。"""

    def __init__(self, get_response):
        self.get_response = get_response
        self._csp = getattr(settings, "CONTENT_SECURITY_POLICY", "")
        self._referrer = getattr(settings, "SECURE_REFERRER_POLICY", "strict-origin-when-cross-origin")
        self._coop = getattr(settings, "SECURE_CROSS_ORIGIN_OPENER_POLICY", "same-origin")

    def __call__(self, request):
        nonce = secrets.token_urlsafe(16)
        setattr(request, "csp_nonce", nonce)
        response = self.get_response(request)
        response.setdefault("X-Content-Type-Options", "nosniff")
        response.setdefault("Referrer-Policy", self._referrer)
        response.setdefault("Cross-Origin-Opener-Policy", self._coop)
        response.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        if self._csp:
            policy = self._build_csp_with_nonce(self._csp, nonce)
            response.setdefault("Content-Security-Policy", policy)
        return response

    @staticmethod
    def _build_csp_with_nonce(policy: str, nonce: str) -> str:
        if not policy:
            return policy
        nonce_token = f"'nonce-{nonce}'"
        if "{nonce}" in policy:
            return policy.format(nonce=nonce)

        parts = [segment.strip() for segment in policy.split(";") if segment.strip()]
        updated = False
        for index, segment in enumerate(parts):
            if segment.startswith("script-src"):
                if nonce_token not in segment:
                    parts[index] = f"{segment} {nonce_token}"
                updated = True
                break
        if not updated:
            parts.append(f"script-src {nonce_token}")
        result = "; ".join(parts)
        if policy.strip().endswith(";"):
            result += ";"
        return result
