from __future__ import annotations

import time

from django.core.management.base import BaseCommand

from ...realtime.alpaca import fetch_assets
from ...realtime.storage import write_state


class Command(BaseCommand):
    help = "Refresh Alpaca assets master list for the realtime engine."

    def add_arguments(self, parser):
        parser.add_argument("--user-id", type=str, default=None, help="User id used to resolve Alpaca credentials.")
        parser.add_argument("--status", type=str, default="active", help="Asset status filter (default: active).")
        parser.add_argument("--asset-class", type=str, default="us_equity", help="Asset class filter.")

    def handle(self, *args, **options):
        user_id = options.get("user_id")
        status = options.get("status") or "active"
        asset_class = options.get("asset_class") or "us_equity"

        assets = fetch_assets(user_id=user_id, status=status, asset_class=asset_class)
        payload = {
            "updated_at": time.time(),
            "count": len(assets),
            "assets": assets,
            "filters": {"status": status, "asset_class": asset_class},
        }
        write_state("assets_master.json", payload)
        self.stdout.write(self.style.SUCCESS(f"Saved {len(assets)} assets to assets_master.json."))
