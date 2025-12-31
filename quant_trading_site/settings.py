from pathlib import Path
import os
import sys

from django.urls import reverse_lazy
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

_configured_storage = os.environ.get("DJANGO_STORAGE_DIR")
if _configured_storage:
    DATA_ROOT = Path(_configured_storage).expanduser().resolve()
else:
    DATA_ROOT = _default_storage_root.resolve()

DATA_ROOT = _ensure_dir(DATA_ROOT)

if load_dotenv:
    load_dotenv(os.fspath(PROJECT_ROOT / ".env"))

DEFAULT_SECRET_KEY = "django-insecure-change-me"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", DEFAULT_SECRET_KEY)

# Ensure SSL cert bundle is available for TLS connections (e.g., Gmail SMTP)
try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

DEBUG_DEFAULT = "1" if "test" in sys.argv else "0"
DEBUG = os.environ.get("DJANGO_DEBUG", DEBUG_DEFAULT) not in {"0", "false", "False"}

if not DEBUG and SECRET_KEY == DEFAULT_SECRET_KEY and os.environ.get("DJANGO_ALLOW_INSECURE_KEY") not in {"1", "true", "True"}:
    raise RuntimeError("DJANGO_SECRET_KEY must be set when DEBUG=0")

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
    "paper",
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
AI_CHAT_RATE_CACHE_ALIAS = os.environ.get("AI_CHAT_RATE_CACHE_ALIAS", "ratelimit")
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
MARKET_DATA_RATE_CACHE_ALIAS = os.environ.get("MARKET_DATA_RATE_CACHE_ALIAS", "ratelimit")

SCREEN_ANALYZER_INTERVAL_MS = int(os.environ.get("SCREEN_ANALYZER_INTERVAL_MS", 1200))
SCREEN_ANALYZER_MAX_BYTES = int(os.environ.get("SCREEN_ANALYZER_MAX_BYTES", 900_000))
SCREEN_ANALYZER_MAX_REQUEST_BYTES = int(os.environ.get("SCREEN_ANALYZER_MAX_REQUEST_BYTES", 1_500_000))
SCREEN_ANALYZER_MAX_WIDTH = int(os.environ.get("SCREEN_ANALYZER_MAX_WIDTH", 1280))
SCREEN_ANALYZER_MAX_HEIGHT = int(os.environ.get("SCREEN_ANALYZER_MAX_HEIGHT", 720))
SCREEN_ANALYZER_RATE_WINDOW_SECONDS = int(os.environ.get("SCREEN_ANALYZER_RATE_WINDOW_SECONDS", 20))
SCREEN_ANALYZER_RATE_MAX_CALLS = int(os.environ.get("SCREEN_ANALYZER_RATE_MAX_CALLS", 30))
SCREEN_ANALYZER_RATE_CACHE_ALIAS = os.environ.get("SCREEN_ANALYZER_RATE_CACHE_ALIAS", "ratelimit")
SCREEN_ANALYZER_OCR_ENABLED = os.environ.get("SCREEN_ANALYZER_OCR_ENABLED", "1") in {"1", "true", "True"}
SCREEN_ANALYZER_TRAIN_MIN_SAMPLES = int(os.environ.get("SCREEN_ANALYZER_TRAIN_MIN_SAMPLES", 18))
SCREEN_ANALYZER_MODEL_MIN_CONF = float(os.environ.get("SCREEN_ANALYZER_MODEL_MIN_CONF", "0.55"))
SCREEN_ANALYZER_INCLUDE_WAVES = os.environ.get("SCREEN_ANALYZER_INCLUDE_WAVES", "1") in {"1", "true", "True"}
SCREEN_ANALYZER_INCLUDE_FUSION = os.environ.get("SCREEN_ANALYZER_INCLUDE_FUSION", "1") in {"1", "true", "True"}
SCREEN_ANALYZER_INCLUDE_TIMINGS = os.environ.get("SCREEN_ANALYZER_INCLUDE_TIMINGS", "0") in {"1", "true", "True"}
SCREEN_ANALYZER_OVERLAY_LAYERS = os.environ.get(
    "SCREEN_ANALYZER_OVERLAY_LAYERS",
    "trendlines",
)
SCREEN_ANALYZER_QUALITY_MIN = float(os.environ.get("SCREEN_ANALYZER_QUALITY_MIN", "0.35"))
SCREEN_ANALYZER_QUALITY_NEUTRAL = float(os.environ.get("SCREEN_ANALYZER_QUALITY_NEUTRAL", "0.45"))
SCREEN_FUSION_ALIGN_BONUS = float(os.environ.get("SCREEN_FUSION_ALIGN_BONUS", "0.05"))
SCREEN_FUSION_CONFLICT_PENALTY = float(os.environ.get("SCREEN_FUSION_CONFLICT_PENALTY", "0.07"))
SCREEN_FUSION_PIVOT_SHIFT_THRESHOLD = float(os.environ.get("SCREEN_FUSION_PIVOT_SHIFT_THRESHOLD", "0.02"))
SCREEN_ANALYZER_MULTI_SCALE_ENABLED = os.environ.get("SCREEN_ANALYZER_MULTI_SCALE_ENABLED", "1") in {
    "1",
    "true",
    "True",
}
SCREEN_ANALYZER_MULTI_SCALE_FACTORS = os.environ.get("SCREEN_ANALYZER_MULTI_SCALE_FACTORS", "1.0,0.8,0.6")
SCREEN_ANALYZER_SMOOTH_ENABLED = os.environ.get("SCREEN_ANALYZER_SMOOTH_ENABLED", "1") in {"1", "true", "True"}
SCREEN_ANALYZER_SMOOTH_WINDOW = int(os.environ.get("SCREEN_ANALYZER_SMOOTH_WINDOW", "5"))
SCREEN_ANALYZER_ADAPTIVE_ENABLED = os.environ.get("SCREEN_ANALYZER_ADAPTIVE_ENABLED", "1") in {"1", "true", "True"}
SCREEN_ANALYZER_ADAPTIVE_MIN_INTERVAL_MS = int(os.environ.get("SCREEN_ANALYZER_ADAPTIVE_MIN_INTERVAL_MS", "700"))
SCREEN_ANALYZER_ADAPTIVE_MAX_INTERVAL_MS = int(os.environ.get("SCREEN_ANALYZER_ADAPTIVE_MAX_INTERVAL_MS", "2000"))
SCREEN_ANALYZER_ADAPTIVE_VOL_LOW = float(os.environ.get("SCREEN_ANALYZER_ADAPTIVE_VOL_LOW", "0.004"))
SCREEN_ANALYZER_ADAPTIVE_VOL_HIGH = float(os.environ.get("SCREEN_ANALYZER_ADAPTIVE_VOL_HIGH", "0.02"))
SCREEN_WAVE_ENABLED = os.environ.get("SCREEN_WAVE_ENABLED", "1") in {"1", "true", "True"}
SCREEN_WAVE_MAX_POINTS = int(os.environ.get("SCREEN_WAVE_MAX_POINTS", 360))
SCREEN_WAVE_SMOOTH_WINDOW = int(os.environ.get("SCREEN_WAVE_SMOOTH_WINDOW", 3))
SCREEN_WAVE_PIVOT_ORDER = int(os.environ.get("SCREEN_WAVE_PIVOT_ORDER", 5))
SCREEN_WAVE_ZIGZAG = float(os.environ.get("SCREEN_WAVE_ZIGZAG", 0.03))
SCREEN_WAVE_STATE_TTL = int(os.environ.get("SCREEN_WAVE_STATE_TTL", 20))
SCREEN_WAVE_STATE_MAX = int(os.environ.get("SCREEN_WAVE_STATE_MAX", 200))
SCREEN_WAVE_STABILITY_BONUS = float(os.environ.get("SCREEN_WAVE_STABILITY_BONUS", 0.05))
SCREEN_WAVE_STABILITY_PENALTY = float(os.environ.get("SCREEN_WAVE_STABILITY_PENALTY", 0.1))
SCREEN_WAVE_STABILITY_THRESHOLD = float(os.environ.get("SCREEN_WAVE_STABILITY_THRESHOLD", 0.02))
SCREEN_WAVE_FUSION_PATTERN_WEIGHT = float(os.environ.get("SCREEN_WAVE_FUSION_PATTERN_WEIGHT", 0.6))
SCREEN_WAVE_FUSION_WAVE_WEIGHT = float(os.environ.get("SCREEN_WAVE_FUSION_WAVE_WEIGHT", 0.4))
SCREEN_WAVE_FUSION_MIN_CONF = float(os.environ.get("SCREEN_WAVE_FUSION_MIN_CONF", 0.35))
SCREEN_WAVE_FUSION_CONFLICT_THRESHOLD = float(os.environ.get("SCREEN_WAVE_FUSION_CONFLICT_THRESHOLD", 0.55))
SCREEN_WAVE_FUSION_CONFLICT_PULLBACK = float(os.environ.get("SCREEN_WAVE_FUSION_CONFLICT_PULLBACK", 0.12))
SCREEN_SIGNAL_MOMENTUM_WINDOW = int(os.environ.get("SCREEN_SIGNAL_MOMENTUM_WINDOW", "32"))
SCREEN_SIGNAL_MOMENTUM_MIN_DELTA = float(os.environ.get("SCREEN_SIGNAL_MOMENTUM_MIN_DELTA", "0.02"))
SCREEN_SIGNAL_MOMENTUM_DELTA_SCALE = float(os.environ.get("SCREEN_SIGNAL_MOMENTUM_DELTA_SCALE", "0.08"))
SCREEN_SIGNAL_MOMENTUM_MIN_CONF = float(os.environ.get("SCREEN_SIGNAL_MOMENTUM_MIN_CONF", "0.35"))
SCREEN_SIGNAL_MOMENTUM_WEIGHT = float(os.environ.get("SCREEN_SIGNAL_MOMENTUM_WEIGHT", "0.15"))
SCREEN_SIGNAL_SMOOTH_ALPHA = float(os.environ.get("SCREEN_SIGNAL_SMOOTH_ALPHA", "0.45"))
SCREEN_SIGNAL_STATE_TTL = int(os.environ.get("SCREEN_SIGNAL_STATE_TTL", "30"))
SCREEN_SIGNAL_STATE_MAX = int(os.environ.get("SCREEN_SIGNAL_STATE_MAX", "200"))
SCREEN_SIGNAL_COOLDOWN_SECONDS = int(os.environ.get("SCREEN_SIGNAL_COOLDOWN_SECONDS", "6"))
SCREEN_SIGNAL_COOLDOWN_PULLBACK = float(os.environ.get("SCREEN_SIGNAL_COOLDOWN_PULLBACK", "0.1"))
SCREEN_SIGNAL_CALIBRATION_TEMP = float(os.environ.get("SCREEN_SIGNAL_CALIBRATION_TEMP", "1.15"))
SCREEN_SIGNAL_DIRECTION_THRESHOLD = float(os.environ.get("SCREEN_SIGNAL_DIRECTION_THRESHOLD", "0.55"))

PROFILE_MAX_GALLERY_IMAGES = int(os.environ.get("PROFILE_MAX_GALLERY_IMAGES", 12))
PROFILE_GALLERY_UPLOAD_LIMIT = int(os.environ.get("PROFILE_GALLERY_UPLOAD_LIMIT", 5))

RATELIMIT_USE_CACHE = "ratelimit"

REDIS_URL = os.environ.get("REDIS_URL")
TASK_RETURN_SNAPSHOT = os.environ.get("TASK_RETURN_SNAPSHOT", "0") in {"1", "true", "True"}

if REDIS_URL:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "KEY_PREFIX": "quantweb",
        },
        "ratelimit": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "KEY_PREFIX": "quantweb:ratelimit",
        },
    }
else:
    CACHES = {
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
        "ratelimit": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
    }

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL or "memory://")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
# Celery 官方内存 backend 需要 cache 前缀，否则会尝试导入名为 memory 的后端模块
if CELERY_RESULT_BACKEND == "memory://":
    CELERY_RESULT_BACKEND = "cache+memory://"
CELERY_TASK_ALWAYS_EAGER = os.environ.get("CELERY_ALWAYS_EAGER", "1" if CELERY_BROKER_URL == "memory://" else "0") in {
    "1",
    "true",
    "True",
}
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_TIMEZONE = TIME_ZONE
CELERY_BEAT_SCHEDULE_FILENAME = os.environ.get(
    "CELERY_BEAT_SCHEDULE_FILENAME",
    os.fspath(DATA_ROOT / "celerybeat-schedule"),
)
PAPER_TRADING_INTERVAL_SECONDS = int(os.environ.get("PAPER_TRADING_INTERVAL_SECONDS", "300") or 300)
ENABLE_PAPER_TRADING_BEAT = os.environ.get("ENABLE_PAPER_TRADING_BEAT", "1") in {"1", "true", "True"}
CELERY_BEAT_SCHEDULE = globals().get("CELERY_BEAT_SCHEDULE", {})
if ENABLE_PAPER_TRADING_BEAT:
    CELERY_BEAT_SCHEDULE.setdefault(
        "paper_trading_heartbeat",
        {
            "task": "trading.tasks.run_paper_trading_heartbeat",
            "schedule": PAPER_TRADING_INTERVAL_SECONDS,
        },
    )

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
    "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https:; "
    "connect-src 'self' https://query1.finance.yahoo.com https://ollama.com https://ollama.com/api https://ollama.com/api/web_search;",
)

# 允许在开发/生产均通过环境变量设置 CSRF 可信域
_csrf_from_env = _split_env_list(os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS"))
if _csrf_from_env:
    CSRF_TRUSTED_ORIGINS = _csrf_from_env

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
