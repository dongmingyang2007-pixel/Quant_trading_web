from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ...legacy_import import import_all


class Command(BaseCommand):
    help = "Import legacy JSON caches (profiles/community/backtests) into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--base-dir",
            dest="base_dir",
            help="Directory containing user_profiles.json, community_posts.json, backtest_history.json (defaults to DATA_CACHE_DIR).",
        )
        parser.add_argument("--profiles", action="store_true", help="Import only user profiles.")
        parser.add_argument("--community", action="store_true", help="Import only community topics/posts.")
        parser.add_argument("--backtests", action="store_true", help="Import only backtest history.")

    def handle(self, *args, **options):
        base_dir = options.get("base_dir") or settings.DATA_CACHE_DIR
        base_path = Path(base_dir)
        only_flags = [options.get("profiles"), options.get("community"), options.get("backtests")]
        if any(only_flags):
            include_profiles = bool(options.get("profiles"))
            include_community = bool(options.get("community"))
            include_backtests = bool(options.get("backtests"))
        else:
            include_profiles = include_community = include_backtests = True
        stats = import_all(
            base_path,
            include_profiles=include_profiles,
            include_community=include_community,
            include_backtests=include_backtests,
        )
        for key, value in stats.items():
            self.stdout.write(f"{key}: {value}")
        self.stdout.write(self.style.SUCCESS("Legacy cache import completed."))
