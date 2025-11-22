from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "清理旧版市场快照缓存（长文件名/含冒号的文件）。默认仅预览，可加 --apply 真正删除。"

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply",
            action="store_true",
            help="实际删除文件（默认仅显示将删除的列表）。",
        )
        parser.add_argument(
            "--max-age-days",
            type=int,
            default=None,
            help="仅删除超过指定天数的旧缓存（默认不限年龄）。",
        )

    def handle(self, *args, **options):
        apply = bool(options.get("apply"))
        max_age_days = options.get("max_age_days")
        cutoff_ts = None
        if max_age_days and max_age_days > 0:
            cutoff_ts = time.time() - timedelta(days=max_age_days).total_seconds()

        base_dir: Path = getattr(settings, "DATA_CACHE_DIR", Path(settings.DATA_ROOT) / "data_cache") / "market_snapshots"
        if not base_dir.exists():
            self.stdout.write(self.style.WARNING(f"目录不存在：{base_dir}"))
            return

        candidates: list[Path] = []
        for path in base_dir.iterdir():
            if path.is_dir():
                continue
            name = path.name
            # 旧格式包含冒号或超长文件名
            is_legacy = ":" in name or len(name) > 64
            if not is_legacy:
                continue
            if cutoff_ts and path.stat().st_mtime > cutoff_ts:
                continue
            candidates.append(path)

        if not candidates:
            self.stdout.write(self.style.SUCCESS("没有发现需要清理的旧版快照。"))
            return

        self.stdout.write(f"将清理 {len(candidates)} 个旧版快照缓存：")
        for path in candidates:
            self.stdout.write(f"- {path.name}")

        if not apply:
            self.stdout.write(self.style.WARNING("预览模式：未删除任何文件。添加 --apply 执行删除。"))
            return

        removed = 0
        for path in candidates:
            try:
                path.unlink()
                removed += 1
            except Exception as exc:  # pragma: no cover - best effort
                self.stdout.write(self.style.ERROR(f"删除失败 {path.name}: {exc}"))
        self.stdout.write(self.style.SUCCESS(f"已删除 {removed} 个旧版快照文件。"))
