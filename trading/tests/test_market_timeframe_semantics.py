from __future__ import annotations

from unittest import mock

import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse


def _build_ohlcv_frame(symbol: str, rows: list[dict[str, float | int | None]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], [symbol]])
    values = []
    for row in rows:
        values.append(
            [
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume"),
            ]
        )
    return pd.DataFrame(values, index=index, columns=columns)


class MarketTimeframeSemanticsTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="market_tf", password="secret123")
        self.client.force_login(self.user)
        self.market_url = reverse("trading:market_insights_data")
        self.assets_url = reverse("trading:market_assets_data")

    def _meta_map(self):
        return {
            "AAA": {"symbol": "AAA", "name": "AAA Corp", "exchange": "NASDAQ"},
            "BBB": {"symbol": "BBB", "name": "BBB Corp", "exchange": "NYSE"},
            "SPY": {"symbol": "SPY", "name": "State Street SPDR S&P 500 ETF Trust", "exchange": "ARCA"},
        }

    def test_most_active_respects_timeframe_semantics(self):
        def _window_rows(timeframe, **_kwargs):
            if timeframe.key != "1mo":
                return [], "unknown"
            return (
                [
                    {"symbol": "BBB", "price": 22.0, "change_pct_period": 2.0, "change_pct_day": 0.2, "volume": 900.0, "dollar_volume": 19_800.0},
                    {"symbol": "AAA", "price": 11.0, "change_pct_period": 1.0, "change_pct_day": 0.1, "volume": 200.0, "dollar_volume": 2_200.0},
                ],
                "alpaca",
            )

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market.market_data.fetch_most_actives",
            return_value=[
                {"symbol": "AAA", "price": 10.0, "change_pct_period": 0.8, "change_pct_day": 0.8, "volume": 500.0, "dollar_volume": 5_000.0},
                {"symbol": "BBB", "price": 20.0, "change_pct_period": 0.6, "change_pct_day": 0.6, "volume": 300.0, "dollar_volume": 6_000.0},
            ],
        ), mock.patch(
            "trading.views.market._resolve_universe_window_rankings",
            side_effect=_window_rows,
        ), mock.patch(
            "trading.views.market._build_snapshot_rankings",
            return_value=[],
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value=self._meta_map(),
        ):
            daily = self.client.get(self.market_url, {"list": "most_active", "timeframe": "1d"})
            monthly = self.client.get(self.market_url, {"list": "most_active", "timeframe": "1mo"})

        self.assertEqual(daily.status_code, 200)
        self.assertEqual(monthly.status_code, 200)
        daily_items = daily.json().get("items", [])
        monthly_items = monthly.json().get("items", [])
        self.assertTrue(daily_items)
        self.assertTrue(monthly_items)
        self.assertEqual(daily_items[0].get("symbol"), "AAA")
        self.assertEqual(monthly_items[0].get("symbol"), "BBB")
        self.assertEqual(monthly.json().get("ranking_timeframe", {}).get("key"), "1mo")

    def test_top_turnover_respects_timeframe_semantics(self):
        snapshot_rows = [
            {"symbol": "AAA", "price": 10.0, "change_pct_period": 0.8, "change_pct_day": 0.8, "volume": 500.0, "dollar_volume": 10_000.0},
            {"symbol": "BBB", "price": 20.0, "change_pct_period": 0.6, "change_pct_day": 0.6, "volume": 300.0, "dollar_volume": 6_000.0},
        ]

        def _window_rows(timeframe, **_kwargs):
            if timeframe.key != "6mo":
                return [], "unknown"
            return (
                [
                    {"symbol": "BBB", "price": 22.0, "change_pct_period": 2.0, "change_pct_day": 0.2, "volume": 900.0, "dollar_volume": 40_000.0},
                    {"symbol": "AAA", "price": 11.0, "change_pct_period": 1.0, "change_pct_day": 0.1, "volume": 200.0, "dollar_volume": 7_000.0},
                ],
                "alpaca",
            )

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._build_snapshot_rankings",
            return_value=snapshot_rows,
        ), mock.patch(
            "trading.views.market._resolve_universe_window_rankings",
            side_effect=_window_rows,
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value=self._meta_map(),
        ):
            daily = self.client.get(self.market_url, {"list": "top_turnover", "timeframe": "1d"})
            half_year = self.client.get(self.market_url, {"list": "top_turnover", "timeframe": "6mo"})

        self.assertEqual(daily.status_code, 200)
        self.assertEqual(half_year.status_code, 200)
        daily_items = daily.json().get("items", [])
        half_year_items = half_year.json().get("items", [])
        self.assertTrue(daily_items)
        self.assertTrue(half_year_items)
        self.assertEqual(daily_items[0].get("symbol"), "AAA")
        self.assertEqual(half_year_items[0].get("symbol"), "BBB")
        self.assertEqual(half_year.json().get("ranking_timeframe", {}).get("key"), "6mo")

    def test_market_assets_timeframe_and_period_change_fields(self):
        frame = _build_ohlcv_frame(
            "AAA",
            [
                {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 100},
                {"open": 10.6, "high": 11.3, "low": 10.2, "close": 11.0, "volume": 120},
                {"open": 11.1, "high": 11.8, "low": 10.9, "close": 11.6, "volume": 130},
            ],
        )
        assets_payload = [{"symbol": "AAA", "name": "AAA Corp", "exchange": "NASDAQ", "status": "active", "class": "us_equity", "tradable": True}]

        with mock.patch(
            "trading.views.market._load_assets_master",
            return_value=assets_payload,
        ), mock.patch(
            "trading.views.market.fetch_stock_snapshots",
            return_value={},
        ), mock.patch(
            "trading.views.market.market_data.fetch",
            return_value=frame,
        ):
            response = self.client.get(self.assets_url, {"timeframe": "1mo", "page": 1, "size": 20})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("timeframe", {}).get("key"), "1mo")
        item = payload["items"][0]
        self.assertIsNotNone(item.get("change_pct_period"))
        self.assertEqual(item.get("change_pct"), item.get("change_pct_period"))

    def test_market_assets_missing_reasons_for_insufficient_window(self):
        frame = _build_ohlcv_frame(
            "AAA",
            [
                {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": None},
            ],
        )
        assets_payload = [{"symbol": "AAA", "name": "AAA Corp", "exchange": "NASDAQ", "status": "active", "class": "us_equity", "tradable": True}]

        with mock.patch(
            "trading.views.market._load_assets_master",
            return_value=assets_payload,
        ), mock.patch(
            "trading.views.market.fetch_stock_snapshots",
            return_value={},
        ), mock.patch(
            "trading.views.market.market_data.fetch",
            return_value=frame,
        ):
            response = self.client.get(self.assets_url, {"timeframe": "1mo", "page": 1, "size": 20})

        self.assertEqual(response.status_code, 200)
        item = response.json()["items"][0]
        reasons = item.get("missing_reasons", {})
        self.assertEqual(reasons.get("prev_close"), "insufficient_window")

    def test_detail_profile_missing_reasons_for_etf(self):
        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._fetch_company_profile",
            return_value={"symbol": "SPY", "name": "State Street SPDR S&P 500 ETF Trust"},
        ), mock.patch(
            "trading.views.market._fetch_yfinance_fundamentals",
            return_value={"quote_type": "ETF"},
        ), mock.patch(
            "trading.views.market._fetch_symbol_news_page",
            return_value=([], {"offset": 0, "limit": 10, "count": 0, "has_more": False, "next_offset": None}),
        ), mock.patch(
            "trading.views.market._fetch_52w_stats",
            return_value={"high_52w": None, "low_52w": None, "as_of": None},
        ):
            response = self.client.get(
                self.market_url,
                {"detail": "1", "symbol": "SPY", "include_bars": "0", "include_ai": "0"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        reasons = payload.get("profile_missing_reasons", {})
        self.assertEqual(reasons.get("sector"), "not_applicable_fund")
        self.assertEqual(reasons.get("industry"), "not_applicable_fund")

    def test_detail_news_only_supports_pagination(self):
        news_rows = [
            {
                "headline": f"News {idx}",
                "summary": f"Summary {idx}",
                "url": f"https://example.com/news/{idx}",
                "source": "Example",
                "created_at": 1_700_000_000 + idx,
            }
            for idx in range(12)
        ]

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market.cache_get_object",
            return_value=None,
        ), mock.patch(
            "trading.views.market.cache_set_object",
        ), mock.patch(
            "trading.views.market._infer_news_symbols",
            return_value=["NEWSA"],
        ), mock.patch(
            "trading.views.market.fetch_news",
            return_value=news_rows,
        ):
            first_page = self.client.get(
                self.market_url,
                {"detail": "1", "symbol": "NEWSA", "news_only": "1", "news_limit": "5", "news_offset": "0"},
            )
            second_page = self.client.get(
                self.market_url,
                {"detail": "1", "symbol": "NEWSA", "news_only": "1", "news_limit": "5", "news_offset": "5"},
            )

        self.assertEqual(first_page.status_code, 200)
        self.assertEqual(second_page.status_code, 200)
        first_payload = first_page.json()
        second_payload = second_page.json()
        self.assertEqual(len(first_payload.get("news", [])), 5)
        self.assertEqual(len(second_payload.get("news", [])), 5)
        self.assertEqual(first_payload.get("news", [])[0].get("title"), "News 11")
        self.assertEqual(second_payload.get("news", [])[0].get("title"), "News 6")
        self.assertTrue(first_payload.get("news_meta", {}).get("has_more"))
        self.assertEqual(first_payload.get("news_meta", {}).get("next_offset"), 5)
        self.assertEqual(second_payload.get("news_meta", {}).get("next_offset"), 10)

    def test_detail_response_includes_news_meta_and_applies_limit(self):
        news_rows = [
            {
                "headline": f"Ticker story {idx}",
                "summary": f"Summary {idx}",
                "url": f"https://example.com/ticker/{idx}",
                "source": "Example",
                "created_at": 1_700_001_000 + idx,
            }
            for idx in range(9)
        ]

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market.cache_get_object",
            return_value=None,
        ), mock.patch(
            "trading.views.market.cache_set_object",
        ), mock.patch(
            "trading.views.market._infer_news_symbols",
            return_value=["NEWSB"],
        ), mock.patch(
            "trading.views.market.fetch_news",
            return_value=news_rows,
        ), mock.patch(
            "trading.views.market._fetch_company_profile",
            return_value={"symbol": "NEWSB", "name": "News B Corp"},
        ), mock.patch(
            "trading.views.market._fetch_yfinance_fundamentals",
            return_value={},
        ), mock.patch(
            "trading.views.market._fetch_52w_stats",
            return_value={"high_52w": None, "low_52w": None, "as_of": None},
        ):
            response = self.client.get(
                self.market_url,
                {"detail": "1", "symbol": "NEWSB", "include_bars": "0", "include_ai": "0", "news_limit": "4"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload.get("news", [])), 4)
        self.assertEqual(payload.get("news_meta", {}).get("limit"), 4)
        self.assertEqual(payload.get("news_meta", {}).get("count"), 4)
        self.assertTrue(payload.get("news_meta", {}).get("has_more"))

    def test_detail_news_response_includes_related_symbols_and_time(self):
        news_rows = [
            {
                "headline": "NVIDIA launches new stack",
                "summary": "Product event update",
                "url": "https://example.com/nvda-news",
                "source": "ExampleWire",
                "created_at": "2026-02-08T12:30:00Z",
                "symbols": ["NVDA", "SMCI"],
            }
        ]

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market.cache_get_object",
            return_value=None,
        ), mock.patch(
            "trading.views.market.cache_set_object",
        ), mock.patch(
            "trading.views.market._infer_news_symbols",
            return_value=["NVDA"],
        ), mock.patch(
            "trading.views.market.fetch_news",
            return_value=news_rows,
        ):
            response = self.client.get(
                self.market_url,
                {"detail": "1", "symbol": "NVDA", "news_only": "1", "news_limit": "10"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        items = payload.get("news", [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].get("related_symbols"), ["NVDA", "SMCI"])
        self.assertIsNotNone(items[0].get("time"))

    def test_detail_news_only_paginates_beyond_legacy_cap(self):
        news_rows = [
            {
                "headline": f"Deep News {idx}",
                "summary": f"Summary {idx}",
                "url": f"https://example.com/deep/{idx}",
                "source": "ExampleDeep",
                "created_at": 1_700_020_000 + idx,
            }
            for idx in range(150)
        ]

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market.cache_get_object",
            return_value=None,
        ), mock.patch(
            "trading.views.market.cache_set_object",
        ), mock.patch(
            "trading.views.market._infer_news_symbols",
            return_value=["NEWSC"],
        ), mock.patch(
            "trading.views.market.fetch_news",
            return_value=news_rows,
        ):
            response = self.client.get(
                self.market_url,
                {"detail": "1", "symbol": "NEWSC", "news_only": "1", "news_limit": "10", "news_offset": "120"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        items = payload.get("news", [])
        meta = payload.get("news_meta", {})
        self.assertEqual(len(items), 10)
        self.assertEqual(items[0].get("title"), "Deep News 29")
        self.assertTrue(meta.get("has_more"))
        self.assertEqual(meta.get("next_offset"), 130)

    def test_most_active_window_failure_falls_back_without_500(self):
        snapshot_rows = [
            {
                "symbol": "AAA",
                "price": 10.0,
                "change_pct_period": 1.2,
                "change_pct_day": 0.2,
                "volume": 900_000.0,
                "dollar_volume": 9_000_000.0,
            },
            {
                "symbol": "BBB",
                "price": 20.0,
                "change_pct_period": 0.8,
                "change_pct_day": 0.1,
                "volume": 600_000.0,
                "dollar_volume": 12_000_000.0,
            },
        ]

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._resolve_universe_window_rankings",
            side_effect=RuntimeError("boom"),
        ), mock.patch(
            "trading.views.market._build_snapshot_rankings",
            return_value=snapshot_rows,
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value=self._meta_map(),
        ):
            response = self.client.get(self.market_url, {"list": "most_active", "timeframe": "1mo"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("items"))
        self.assertEqual(payload.get("ranking_timeframe", {}).get("key"), "1d")

    @override_settings(
        MARKET_LIST_TIMEFRAME_SUPPORT={
            "gainers": ["1d"],
            "losers": ["1d"],
            "most_active": ["1d"],
            "top_turnover": ["1d"],
        }
    )
    def test_returns_timeframe_not_supported_error_for_disallowed_combo(self):
        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ):
            response = self.client.get(self.market_url, {"list": "most_active", "timeframe": "1mo"})

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "timeframe_not_supported")
        self.assertIn("capabilities", payload)
        self.assertEqual(
            payload.get("capabilities", {}).get("supported_timeframes_by_list", {}).get("most_active"),
            ["1d"],
        )

    def test_market_response_includes_capabilities(self):
        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market.market_data.fetch_market_movers",
            return_value={
                "gainers": [
                    {
                        "symbol": "AAA",
                        "price": 10.0,
                        "change_pct_period": 1.2,
                        "change_pct_day": 1.2,
                        "volume": 900_000.0,
                        "dollar_volume": 9_000_000.0,
                    }
                ],
                "losers": [],
            },
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value=self._meta_map(),
        ):
            response = self.client.get(self.market_url, {"list": "gainers", "timeframe": "1d"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        capabilities = payload.get("capabilities", {}).get("supported_timeframes_by_list", {})
        self.assertIn("gainers", capabilities)
        self.assertEqual(capabilities.get("gainers"), ["1d", "5d", "1mo", "6mo"])

    def test_top_turnover_falls_back_when_window_rows_have_no_turnover(self):
        timeframe_rows = [
            {
                "symbol": "AAA",
                "price": 11.0,
                "change_pct_period": 1.0,
                "change_pct_day": 0.1,
                "volume": 1000.0,
                "dollar_volume": None,
            }
        ]
        snapshot_rows = [
            {
                "symbol": "BBB",
                "price": 20.0,
                "change_pct_period": 0.5,
                "change_pct_day": 0.2,
                "volume": 10_000.0,
                "dollar_volume": 200_000.0,
            }
        ]

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._resolve_universe_window_rankings",
            return_value=(timeframe_rows, "alpaca"),
        ), mock.patch(
            "trading.views.market._build_snapshot_rankings",
            return_value=snapshot_rows,
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value=self._meta_map(),
        ):
            response = self.client.get(self.market_url, {"list": "top_turnover", "timeframe": "1mo"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("items"))
        self.assertEqual(payload["items"][0]["symbol"], "BBB")
        self.assertEqual(payload.get("ranking_timeframe", {}).get("key"), "1d")

    def test_window_rankings_fetch_history_in_chunks(self):
        assets_payload = [
            {"symbol": "AAA", "name": "AAA Corp", "exchange": "NASDAQ", "status": "active", "class": "us_equity", "tradable": True},
            {"symbol": "BBB", "name": "BBB Corp", "exchange": "NYSE", "status": "active", "class": "us_equity", "tradable": True},
            {"symbol": "CCC", "name": "CCC Corp", "exchange": "NASDAQ", "status": "active", "class": "us_equity", "tradable": True},
        ]
        fetch_calls: list[list[str]] = []

        def _chunk_frame(symbols: list[str]) -> pd.DataFrame:
            rows = [
                {"open": 10.0, "high": 11.0, "low": 9.8, "close": 10.5, "volume": 1000},
                {"open": 10.6, "high": 11.2, "low": 10.2, "close": 11.0, "volume": 1200},
                {"open": 11.1, "high": 11.5, "low": 10.9, "close": 11.3, "volume": 900},
            ]
            frames = [_build_ohlcv_frame(symbol, rows) for symbol in symbols]
            return pd.concat(frames, axis=1)

        def _fetch_side_effect(symbols, **_kwargs):
            chunk = list(symbols)
            fetch_calls.append(chunk)
            if len(chunk) > 2:
                return pd.DataFrame()
            return _chunk_frame(chunk)

        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._load_assets_master",
            return_value=assets_payload,
        ), mock.patch(
            "trading.views.market.cache_memoize",
            side_effect=lambda _k, builder, _ttl, **_kwargs: builder(),
        ), mock.patch(
            "trading.views.market._build_snapshot_rankings",
            return_value=[],
        ), mock.patch(
            "trading.views.market._resolve_universe_rankings",
            return_value=([], "unknown"),
        ), mock.patch(
            "trading.views.market._build_asset_meta_map",
            return_value={
                "AAA": assets_payload[0],
                "BBB": assets_payload[1],
                "CCC": assets_payload[2],
            },
        ), mock.patch(
            "trading.views.market.market_data.fetch",
            side_effect=_fetch_side_effect,
        ), mock.patch(
            "trading.views.market.MARKET_UNIVERSE_CHUNK_SIZE",
            2,
        ), mock.patch(
            "trading.views.market.MARKET_UNIVERSE_CHUNK_WORKERS",
            1,
        ):
            response = self.client.get(self.market_url, {"list": "top_turnover", "timeframe": "1mo"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(len(payload.get("items", [])), 0)
        self.assertGreaterEqual(len(fetch_calls), 2)
        self.assertTrue(all(len(call) <= 2 for call in fetch_calls))
