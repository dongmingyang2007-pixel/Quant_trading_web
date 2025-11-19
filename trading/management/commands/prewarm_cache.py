from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from django.core.management.base import BaseCommand
from django.utils import timezone

from trading import screener
from trading.views import market as market_views
from trading import data_sources


class Command(BaseCommand):
    help = "Prime frequently used screener/market/macro caches to improve first-byte latency."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=int,
            default=20,
            help="Number of core US tickers to prewarm (default: 20).",
        )
        parser.add_argument(
            "--timeframe",
            default="1mo",
            choices=list(market_views.TIMEFRAMES.keys()),
            help="Market insights timeframe key to prewarm (default: 1mo).",
        )

    def handle(self, *args, **options):
        limit = max(5, int(options["symbols"]))
        timeframe_key = options["timeframe"]
        timeframe = market_views.TIMEFRAMES.get(timeframe_key, market_views.DEFAULT_TIMEFRAME)
        self.stdout.write(f"→ Prewarming screener (top {limit} symbols)…")
        screener.fetch_page(offset=0, size=limit, market="us")
        symbols = screener.CORE_TICKERS_US[:limit]
        self.stdout.write(f"→ Prewarming market history for timeframe {timeframe.key}…")
        market_views._fetch_history(symbols, timeframe)  # type: ignore[attr-defined]
        now = timezone.now().date()
        self.stdout.write("→ Prewarming macro indicators…")
        data_sources._fetch_macro_series(now)  # type: ignore[attr-defined]
        self.stdout.write(self.style.SUCCESS("Cache prewarm finished."))
