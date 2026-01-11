from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from trading.realtime.lock import InstanceLock, fcntl, msvcrt


class RealtimeLockTests(SimpleTestCase):
    def test_instance_lock_blocks_second_acquire(self):
        if fcntl is None and msvcrt is None:
            self.skipTest("Platform does not support advisory locks")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "realtime.pid"
            lock_a = InstanceLock(path)
            lock_b = InstanceLock(path)
            self.assertTrue(lock_a.acquire())
            self.assertFalse(lock_b.acquire())
            lock_a.release()
            self.assertTrue(lock_b.acquire())
            lock_b.release()
