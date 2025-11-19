from django.contrib import admin

from .models import BacktestRecord


@admin.register(BacktestRecord)
class BacktestRecordAdmin(admin.ModelAdmin):
    list_display = (
        "record_id",
        "user",
        "ticker",
        "benchmark",
        "engine",
        "start_date",
        "end_date",
        "timestamp",
    )
    list_filter = ("engine", "ticker", "user")
    search_fields = ("record_id", "ticker", "benchmark", "user__username", "user__email")
    readonly_fields = ("record_id", "timestamp")

