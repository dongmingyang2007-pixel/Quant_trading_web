import os
from pathlib import Path
import sys


def _bootstrap_pycache_prefix() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    project_root = base_dir.parent
    configured_root = os.environ.get("DJANGO_STORAGE_DIR")
    if configured_root:
        storage_root = Path(configured_root).expanduser().resolve()
    else:
        storage_root = (project_root / "storage_bundle").resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    pycache_root = storage_root / "pycache"
    pycache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", os.fspath(pycache_root))
    if getattr(sys, "pycache_prefix", None) != os.fspath(pycache_root):
        sys.pycache_prefix = os.fspath(pycache_root)


_bootstrap_pycache_prefix()

from django.core.asgi import get_asgi_application  # noqa: E402
from channels.auth import AuthMiddlewareStack  # noqa: E402
from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402

from trading import routing  # noqa: E402


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quant_trading_site.settings")

django_app = get_asgi_application()

application = ProtocolTypeRouter(
    {
        "http": django_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(routing.websocket_urlpatterns),
        ),
    }
)
