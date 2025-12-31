#!/usr/bin/env python3
import os
from pathlib import Path
import sys


def _bootstrap_pycache_prefix() -> None:
    base_dir = Path(__file__).resolve().parent
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


def main() -> None:
    """Run administrative tasks."""
    _bootstrap_pycache_prefix()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quant_trading_site.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django is not installed. Install dependencies from requirements.txt "
            "and activate your virtual environment."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
