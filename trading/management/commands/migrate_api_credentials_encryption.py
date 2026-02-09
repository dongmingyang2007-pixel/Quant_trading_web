from __future__ import annotations

from django.core.management.base import BaseCommand

from trading.models import UserProfile
from trading.observability import record_metric
from trading.profile import migrate_legacy_api_credentials


class Command(BaseCommand):
    help = "Migrate plaintext UserProfile.api_credentials into encrypted storage."

    def add_arguments(self, parser):
        parser.add_argument("--user-id", dest="user_id", default="", help="Migrate only the given user id.")
        parser.add_argument("--dry-run", action="store_true", help="Preview migration results without writing changes.")

    def handle(self, *args, **options):
        user_id = (options.get("user_id") or "").strip()
        dry_run = bool(options.get("dry_run"))

        queryset = UserProfile.objects.all().order_by("id")
        if user_id:
            queryset = queryset.filter(user_id=user_id)

        total = 0
        migrated = 0
        skipped = 0
        failed = 0

        for profile in queryset.iterator():
            total += 1
            try:
                success, reason = migrate_legacy_api_credentials(profile, dry_run=dry_run)
            except Exception as exc:  # pragma: no cover - defensive
                failed += 1
                reason = f"command_error:{exc.__class__.__name__}"
                record_metric(
                    "credential.migration.failure",
                    user_id=str(profile.user_id),
                    reason=reason,
                )
                self.stderr.write(self.style.ERROR(f"user_id={profile.user_id}: {reason}"))
                continue

            if success:
                migrated += 1
                verb = "would_migrate" if dry_run else "migrated"
                self.stdout.write(self.style.SUCCESS(f"user_id={profile.user_id}: {verb} ({reason})"))
            elif reason in {"empty"}:
                skipped += 1
            else:
                failed += 1
                self.stderr.write(self.style.ERROR(f"user_id={profile.user_id}: failed ({reason})"))

        summary = f"total={total} migrated={migrated} skipped={skipped} failed={failed} dry_run={dry_run}"
        if failed:
            self.stderr.write(self.style.ERROR(summary))
        else:
            self.stdout.write(self.style.SUCCESS(summary))
