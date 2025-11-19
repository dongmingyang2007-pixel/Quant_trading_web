from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import TestCase

from trading import observability


class ObservabilityTests(TestCase):
    def test_record_metric_writes_valid_json(self):
        original_dir = observability.DATA_CACHE_DIR
        original_path = observability.METRICS_PATH
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metrics_path = tmp_path / "telemetry.ndjson"
            observability.DATA_CACHE_DIR = tmp_path
            observability.METRICS_PATH = metrics_path
            try:
                observability.record_metric("test.event", sample="one")
                observability.record_metric("test.event", sample="two")
            finally:
                observability.DATA_CACHE_DIR = original_dir
                observability.METRICS_PATH = original_path
            payloads = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(payloads), 2)
            self.assertTrue(all(entry["event"] == "test.event" for entry in payloads))
