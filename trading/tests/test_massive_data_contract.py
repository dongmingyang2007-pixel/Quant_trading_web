from __future__ import annotations

from unittest import mock

from django.test import TestCase

from trading import massive_data


class MassiveDataContractTests(TestCase):
    @mock.patch("trading.massive_data._resolve_massive_api_key", return_value="key")
    @mock.patch("trading.massive_data._request_json")
    def test_fetch_stock_bars_normalizes_polygon_payload(self, mock_request_json, _mock_key):
        mock_request_json.return_value = {
            "results": [
                {
                    "t": 1735689600000,
                    "o": 100.0,
                    "h": 102.0,
                    "l": 99.5,
                    "c": 101.5,
                    "v": 5000,
                    "n": 20,
                }
            ]
        }
        payload = massive_data.fetch_stock_bars(["SPY"], timeframe="1Day")
        self.assertIn("SPY", payload)
        row = payload["SPY"][0]
        self.assertIn("t", row)
        self.assertEqual(row["o"], 100.0)
        self.assertEqual(row["c"], 101.5)

    @mock.patch("trading.massive_data._resolve_massive_api_key", return_value="key")
    @mock.patch("trading.massive_data._request_json")
    def test_fetch_stock_snapshots_maps_snapshot_shape(self, mock_request_json, _mock_key):
        mock_request_json.return_value = {
            "ticker": {
                "day": {"o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 3000, "t": 1735689600000},
                "prevDay": {"c": 98.0, "v": 2500, "t": 1735603200000},
                "lastTrade": {"p": 101.2, "s": 10, "t": 1735689660000},
            }
        }
        snapshots = massive_data.fetch_stock_snapshots(["SPY"])
        self.assertIn("SPY", snapshots)
        snap = snapshots["SPY"]
        self.assertIn("dailyBar", snap)
        self.assertIn("prevDailyBar", snap)
        self.assertIn("latestTrade", snap)

    @mock.patch("trading.massive_data._resolve_massive_api_key", return_value="key")
    @mock.patch("trading.massive_data._request_json")
    def test_fetch_stock_snapshots_uses_full_market_snapshot_for_large_requests(self, mock_request_json, _mock_key):
        def _side_effect(path, **kwargs):
            if path == "/v2/snapshot/locale/us/markets/stocks/tickers":
                return {
                    "tickers": [
                        {
                            "ticker": "SPY",
                            "day": {"c": 501.0, "v": 1000},
                            "prevDay": {"c": 498.0, "v": 900},
                            "lastTrade": {"p": 501.2, "s": 10, "t": 1735689660000},
                        },
                        {
                            "ticker": "QQQ",
                            "day": {"c": 421.0, "v": 800},
                            "prevDay": {"c": 420.0, "v": 750},
                            "lastTrade": {"p": 421.1, "s": 5, "t": 1735689660000},
                        },
                    ]
                }
            raise AssertionError(f"unexpected path: {path}")

        mock_request_json.side_effect = _side_effect
        with mock.patch.object(massive_data, "FULL_SNAPSHOT_THRESHOLD", 2):
            snapshots = massive_data.fetch_stock_snapshots(["SPY", "QQQ"])
        self.assertEqual(set(snapshots.keys()), {"SPY", "QQQ"})
        self.assertEqual(mock_request_json.call_count, 1)
        self.assertEqual(mock_request_json.call_args.args[0], "/v2/snapshot/locale/us/markets/stocks/tickers")

    @mock.patch("trading.massive_data._resolve_massive_api_key", return_value="key")
    @mock.patch("trading.massive_data._request_json")
    def test_fetch_news_uses_ticker_any_of_for_related_news(self, mock_request_json, _mock_key):
        mock_request_json.return_value = {"results": []}
        massive_data.fetch_news(symbols=["TGCL", "TCGL"], limit=10)
        self.assertTrue(mock_request_json.called)
        call_kwargs = mock_request_json.call_args.kwargs
        params = call_kwargs.get("params") or {}
        self.assertIn("ticker.any_of", params)
        self.assertEqual(params.get("ticker.any_of"), "TGCL,TCGL")
        self.assertNotIn("ticker", params)

    @mock.patch("trading.massive_data._resolve_massive_api_key", return_value="key")
    @mock.patch("trading.massive_data._request_json")
    def test_fetch_company_overview_normalizes_fundamentals_fields(self, mock_request_json, _mock_key):
        mock_request_json.return_value = {
            "results": {
                "ticker": "AN",
                "name": "Adlai Nortye",
                "primary_exchange": "AMEX",
                "market_cap": 364_400_000,
                "sic_description": "Biotechnology",
                "address": {
                    "city": "Grand Cayman",
                    "state": "Cayman Islands",
                    "country": "KY",
                },
                "type": "CS",
            }
        }
        payload = massive_data.fetch_company_overview("AN")
        self.assertEqual(payload.get("symbol"), "AN")
        self.assertEqual(payload.get("exchange"), "AMEX")
        self.assertEqual(payload.get("market_cap"), 364_400_000)
        self.assertEqual(payload.get("industry"), "Biotechnology")
        self.assertEqual(payload.get("quote_type"), "EQUITY")
        self.assertIn("Grand Cayman", str(payload.get("hq") or ""))
