from __future__ import annotations

from django.test import SimpleTestCase
import pandas as pd

from trading import cache_utils


class CacheUtilsTests(SimpleTestCase):
    def setUp(self):
        # Ensure a fresh in-memory cache per test.
        cache_utils._CACHE = cache_utils.InMemoryCache()

    def test_json_safe_values_round_trip(self):
        payload = {"a": 1, "b": [True, None, "x"]}
        cache_utils.cache_set_object("safe", payload, 30)
        self.assertEqual(cache_utils.cache_get_object("safe"), payload)

    def test_non_serializable_values_are_not_cached(self):
        class Dummy:
            pass

        calls: list[int] = []

        def builder():
            calls.append(1)
            return Dummy()

        first = cache_utils.cache_memoize("unsafe", builder, 10)
        second = cache_utils.cache_memoize("unsafe", builder, 10)
        self.assertIsInstance(first, Dummy)
        self.assertIsInstance(second, Dummy)
        # Builder runs twice because result was not cacheable.
        self.assertEqual(len(calls), 2)
        self.assertIsNone(cache_utils.cache_get_object("unsafe"))

    def test_dataframe_round_trip(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cache_utils.cache_set_object("df", df, 30)
        restored = cache_utils.cache_get_object("df")
        self.assertTrue(isinstance(restored, pd.DataFrame))
        self.assertTrue(df.equals(restored))

    def test_series_round_trip(self):
        series = pd.Series([1.1, 2.2, 3.3], index=["x", "y", "z"])
        cache_utils.cache_set_object("series", series, 30)
        restored = cache_utils.cache_get_object("series")
        self.assertTrue(isinstance(restored, pd.Series))
        self.assertTrue(series.equals(restored))

    def test_cache_alias_scopes_keys(self):
        cache_utils.cache_set_object("shared", {"value": "alpha"}, 30, cache_alias="alpha")
        cache_utils.cache_set_object("shared", {"value": "beta"}, 30, cache_alias="beta")
        self.assertEqual(
            cache_utils.cache_get_object("shared", cache_alias="alpha"),
            {"value": "alpha"},
        )
        self.assertEqual(
            cache_utils.cache_get_object("shared", cache_alias="beta"),
            {"value": "beta"},
        )
