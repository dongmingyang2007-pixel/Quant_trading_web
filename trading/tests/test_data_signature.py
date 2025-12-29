from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from trading.validation import build_data_signature


class DataSignatureTests(SimpleTestCase):
    def test_signature_is_stable_for_same_frame(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        prices = pd.DataFrame(
            {
                "adj close": np.linspace(100, 104, len(idx)),
                "close": np.linspace(100, 104, len(idx)),
            },
            index=idx,
        )
        sig_a = build_data_signature(prices, columns=["adj close", "close"])
        sig_b = build_data_signature(prices.copy(), columns=["adj close", "close"])
        self.assertEqual(sig_a["hash"], sig_b["hash"])
        self.assertEqual(sig_a["rows"], len(prices))
        self.assertEqual(sig_a["start"], "2024-01-01")
        self.assertEqual(sig_a["end"], "2024-01-05")

    def test_signature_handles_empty_frame(self):
        empty = pd.DataFrame()
        sig = build_data_signature(empty)
        self.assertEqual(sig["rows"], 0)
        self.assertIsNone(sig["hash"])
