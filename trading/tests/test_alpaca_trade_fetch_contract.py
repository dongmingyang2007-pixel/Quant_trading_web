from __future__ import annotations

from unittest import mock

from django.test import SimpleTestCase

from trading.alpaca_data import fetch_stock_trades


class FetchStockTradesContractTests(SimpleTestCase):
    @mock.patch("trading.alpaca_data.resolve_alpaca_data_credentials", return_value=(None, None))
    def test_missing_credentials_returns_four_tuple(self, _mock_creds):
        result = fetch_stock_trades("AAPL", user_id="missing")
        self.assertEqual(result, ([], None, None, None))

        trades, next_token, downgrade_to, downgrade_message = result
        self.assertEqual(trades, [])
        self.assertIsNone(next_token)
        self.assertIsNone(downgrade_to)
        self.assertIsNone(downgrade_message)

    @mock.patch("trading.alpaca_data._alpaca_get")
    @mock.patch("trading.alpaca_data.resolve_alpaca_data_credentials", return_value=("key", "secret"))
    def test_empty_symbol_returns_four_tuple_without_network(self, _mock_creds, mock_alpaca_get):
        result = fetch_stock_trades("  ", user_id="user-1")
        self.assertEqual(result, ([], None, None, None))

        trades, next_token, downgrade_to, downgrade_message = result
        self.assertEqual(trades, [])
        self.assertIsNone(next_token)
        self.assertIsNone(downgrade_to)
        self.assertIsNone(downgrade_message)
        mock_alpaca_get.assert_not_called()
