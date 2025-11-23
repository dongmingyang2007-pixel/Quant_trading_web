from __future__ import annotations

from django.core.management.base import BaseCommand

from paper.engine import run_pending_sessions, _build_price_cache
from paper.models import PaperTradingSession


class Command(BaseCommand):
    help = "Run one paper-trading heartbeat tick and print a summary (for debugging)."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=20, help="Max sessions to process.")

    def handle(self, *args, **options):
        limit = options.get("limit") or 20
        qs = (
            PaperTradingSession.objects.filter(status="running")
            .order_by("next_run_at")
            .select_related("user")[:limit]
        )
        running = qs.count()
        self.stdout.write(self.style.WARNING(f"Running sessions: {running}, limit: {limit}"))
        price_cache = _build_price_cache(list(qs))
        results = run_pending_sessions(limit=limit, price_cache=price_cache)
        if not results:
            self.stdout.write(self.style.WARNING("No sessions processed."))
            return
        for item in results:
            if item.get("skipped"):
                self.stdout.write(self.style.WARNING(f"Skipped: {item.get('reason', 'unknown')}"))
                continue
            sid = item.get("session_id")
            price = item.get("price")
            equity = item.get("equity")
            self.stdout.write(self.style.SUCCESS(f"[{sid}] price={price} equity={equity} trades={len(item.get('trades', []))}"))
