from __future__ import annotations

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quant_trading_site.settings")

app = Celery("quant_trading_site")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    # pragma: no cover - only used for verifying Celery wiring
    print(f"Celery debug task executed: {self.request!r}")
