from django.apps import AppConfig


class TradingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "trading"

    def ready(self) -> None:
        from . import signals  # noqa: F401
        from .realtime.autostart import start_realtime_engine

        start_realtime_engine()
