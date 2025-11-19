from pathlib import Path
import os
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration
except Exception:  # pragma: no cover - optional during local dev
    sentry_sdk = None  # type: ignore
    DjangoIntegration = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent  # app_bundle
PROJECT_ROOT = BASE_DIR.parent


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


_default_storage_root = PROJECT_ROOT / "storage_bundle"
if not _default_storage_root.exists():
    _default_storage_root = BASE_DIR / "storage_bundle"

_configured_storage = os.environ.get("DJANGO_STORAGE_DIR")
if _configured_storage:
    DATA_ROOT = Path(_configured_storage).expanduser().resolve()
else:
    DATA_ROOT = _default_storage_root.resolve()

DATA_ROOT = _ensure_dir(DATA_ROOT)

if load_dotenv:
    load_dotenv(os.fspath(PROJECT_ROOT / ".env"))

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-change-me")

# Ensure SSL cert bundle is available for TLS connections (e.g., Gmail SMTP)
try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

DEBUG = os.environ.get("DJANGO_DEBUG", "1") not in {"0", "false", "False"}

if sentry_sdk and os.environ.get("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        integrations=[DjangoIntegration()] if DjangoIntegration else [],
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.15")),
        profiles_sample_rate=float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0")),
        send_default_pii=os.environ.get("SENTRY_SEND_PII", "1") in {"1", "true", "True"},
    )

def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

ALLOWED_HOSTS: list[str] = _split_env_list(os.environ.get("DJANGO_ALLOWED_HOSTS")) or ["127.0.0.1", "localhost"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "django_bootstrap5",
    "rest_framework",
    "trading",
]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "user": os.environ.get("API_USER_THROTTLE", "240/hour"),
        "api_task": os.environ.get("API_TASK_THROTTLE", "60/hour"),
    },
}

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "trading.middleware.SecurityHeadersMiddleware",
]

ROOT_URLCONF = "quant_trading_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "trading.context_processors.csp_nonce",
                "trading.context_processors.user_flags",
            ],
            "builtins": [
                "trading.templatetags.i18n_extras",
            ],
        },
    },
]

WSGI_APPLICATION = "quant_trading_site.wsgi.application"
ASGI_APPLICATION = "quant_trading_site.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATA_ROOT / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LANGUAGE_CODE = "zh-hans"

LANGUAGES = [
    ("zh-hans", "简体中文"),
    ("en", "English"),
]

LOCALE_PATHS = [
    BASE_DIR / "locale",
]

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

STATIC_URL = "/static/"

STATIC_ROOT = _ensure_dir(DATA_ROOT / "staticfiles")

STATICFILES_DIRS = [
    BASE_DIR / "trading" / "static",
]

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}
WHITENOISE_MANIFEST_STRICT = False

AI_CHAT_MAX_HISTORY = int(os.environ.get("AI_CHAT_MAX_HISTORY", 20))
AI_CHAT_MAX_MESSAGE_CHARS = int(os.environ.get("AI_CHAT_MAX_MESSAGE_CHARS", 4000))
AI_CHAT_MAX_PAYLOAD_BYTES = int(os.environ.get("AI_CHAT_MAX_PAYLOAD_BYTES", 512 * 1024))
AI_CHAT_TIMEOUT_SECONDS = int(os.environ.get("AI_CHAT_TIMEOUT_SECONDS", 120))
AI_CHAT_MAX_WORKERS = int(os.environ.get("AI_CHAT_MAX_WORKERS", 4))
AI_CHAT_MAX_IN_FLIGHT = int(
    os.environ.get(
        "AI_CHAT_MAX_IN_FLIGHT",
        max(8, AI_CHAT_MAX_WORKERS * 3),
    )
)
AI_CHAT_GUARD_WAIT_SECONDS = float(os.environ.get("AI_CHAT_GUARD_WAIT_SECONDS", 0.75))
AI_CHAT_RATE_WINDOW_SECONDS = int(os.environ.get("AI_CHAT_RATE_WINDOW_SECONDS", 60))
AI_CHAT_RATE_MAX_CALLS = int(os.environ.get("AI_CHAT_RATE_MAX_CALLS", 30))
AI_CHAT_RATE_CACHE_ALIAS = os.environ.get("AI_CHAT_RATE_CACHE_ALIAS", "default")
AI_CHAT_FETCH_TIMEOUT_MS = int(
    os.environ.get(
        "AI_CHAT_FETCH_TIMEOUT_MS",
        AI_CHAT_TIMEOUT_SECONDS * 1000,
    )
)
MARKET_DATA_TIMEOUT_SECONDS = int(os.environ.get("MARKET_DATA_TIMEOUT_SECONDS", 25))
MARKET_DATA_MAX_WORKERS = int(os.environ.get("MARKET_DATA_MAX_WORKERS", 4))
MARKET_DATA_RATE_WINDOW_SECONDS = int(os.environ.get("MARKET_DATA_RATE_WINDOW_SECONDS", 90))
MARKET_DATA_RATE_MAX_CALLS = int(os.environ.get("MARKET_DATA_RATE_MAX_CALLS", 45))
MARKET_DATA_RATE_CACHE_ALIAS = os.environ.get("MARKET_DATA_RATE_CACHE_ALIAS", "default")

PROFILE_MAX_GALLERY_IMAGES = int(os.environ.get("PROFILE_MAX_GALLERY_IMAGES", 12))
PROFILE_GALLERY_UPLOAD_LIMIT = int(os.environ.get("PROFILE_GALLERY_UPLOAD_LIMIT", 5))

RATELIMIT_USE_CACHE = "default"

REDIS_URL = os.environ.get("REDIS_URL")

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL or "memory://")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_TASK_ALWAYS_EAGER = os.environ.get("CELERY_ALWAYS_EAGER", "1" if CELERY_BROKER_URL == "memory://" else "0") in {
    "1",
    "true",
    "True",
}
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_TIMEZONE = TIME_ZONE

MARKET_HISTORY_CACHE_TTL = int(os.environ.get("MARKET_HISTORY_CACHE_TTL", 300))
SCREENER_CACHE_TTL = int(os.environ.get("SCREENER_CACHE_TTL", 180))
MACRO_DATA_CACHE_TTL = int(os.environ.get("MACRO_DATA_CACHE_TTL", 600))

MEDIA_URL = "/media/"
MEDIA_ROOT = _ensure_dir(DATA_ROOT / "media")

STORAGE_ROOT = DATA_ROOT
DATA_CACHE_DIR = _ensure_dir(DATA_ROOT / "data_cache")
LEARNING_CONTENT_DIR = _ensure_dir(DATA_ROOT / "learning_content")

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = os.environ.get("DJANGO_REFERRER_POLICY", "strict-origin-when-cross-origin")
SECURE_CROSS_ORIGIN_OPENER_POLICY = os.environ.get("DJANGO_COOP", "same-origin")
SESSION_COOKIE_SAMESITE = os.environ.get("DJANGO_SESSION_SAMESITE", "Lax")
CSRF_COOKIE_SAMESITE = os.environ.get("DJANGO_CSRF_SAMESITE", "Lax")
SESSION_COOKIE_SECURE = os.environ.get("DJANGO_SESSION_COOKIE_SECURE", "1" if not DEBUG else "0") in {"1", "true", "True"}
CSRF_COOKIE_SECURE = os.environ.get("DJANGO_CSRF_COOKIE_SECURE", "1" if not DEBUG else "0") in {"1", "true", "True"}
SECURE_SSL_REDIRECT = os.environ.get("DJANGO_SSL_REDIRECT", "0" if DEBUG else "1") in {"1", "true", "True"}
SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_HSTS_SECONDS", "31536000" if not DEBUG else "0") or 0)
SECURE_HSTS_INCLUDE_SUBDOMAINS = bool(SECURE_HSTS_SECONDS)
SECURE_HSTS_PRELOAD = bool(SECURE_HSTS_SECONDS)

CONTENT_SECURITY_POLICY = os.environ.get(
    "DJANGO_CONTENT_SECURITY_POLICY",
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https:; "
    "connect-src 'self' https://query1.finance.yahoo.com https://ollama.com https://ollama.com/api https://ollama.com/api/web_search;",
)

# 允许在开发/生产均通过环境变量设置 CSRF 可信域
_csrf_from_env = _split_env_list(os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS"))
if _csrf_from_env:
    CSRF_TRUSTED_ORIGINS = _csrf_from_env

from django.urls import reverse_lazy

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Auth redirects
LOGIN_URL = reverse_lazy("login")
LOGIN_REDIRECT_URL = reverse_lazy("trading:backtest")
LOGOUT_REDIRECT_URL = reverse_lazy("login")

# Email (development default: print emails to console)
if os.environ.get("EMAIL_HOST"):
    EMAIL_BACKEND = os.environ.get(
        "DJANGO_EMAIL_BACKEND",
        "django.core.mail.backends.smtp.EmailBackend",
    )
    EMAIL_HOST = os.environ.get("EMAIL_HOST")
    EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587"))
    EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", "")
    EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")
    # Read booleans from env; default to disabled to avoid conflicts
    EMAIL_USE_TLS = os.environ.get("EMAIL_USE_TLS", "0") in {"1", "true", "True"}
    EMAIL_USE_SSL = os.environ.get("EMAIL_USE_SSL", "0") in {"1", "true", "True"}
    # Ensure mutual exclusivity (prefer SSL if both accidentally set)
    if EMAIL_USE_SSL and EMAIL_USE_TLS:
        EMAIL_USE_TLS = False
else:
    EMAIL_BACKEND = os.environ.get(
        "DJANGO_EMAIL_BACKEND",
        "django.core.mail.backends.console.EmailBackend",
    )
DEFAULT_FROM_EMAIL = os.environ.get("DJANGO_DEFAULT_FROM_EMAIL", "no-reply@quant.local")
