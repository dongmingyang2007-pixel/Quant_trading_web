from __future__ import annotations

from unittest import mock

from django.test import SimpleTestCase

from trading.alpaca_data import fetch_news


class FetchNewsTests(SimpleTestCase):
    @mock.patch("trading.alpaca_data._alpaca_get")
    @mock.patch("trading.alpaca_data.resolve_alpaca_data_credentials", return_value=("key", "secret"))
    def test_fetch_news_paginates_until_requested_limit(self, _mock_creds, mock_alpaca_get):
        first_page = mock.Mock()
        first_page.json.return_value = {
            "news": [{"id": f"n{i:03d}"} for i in range(1, 51)],
            "next_page_token": "token-1",
        }
        second_page = mock.Mock()
        second_page.json.return_value = {
            "news": [{"id": f"n{i:03d}"} for i in range(51, 71)],
            "next_page_token": None,
        }
        mock_alpaca_get.side_effect = [first_page, second_page]

        rows = fetch_news(symbols=["NVDA"], limit=70, user_id="user-1")

        self.assertEqual(len(rows), 70)
        self.assertEqual(rows[0].get("id"), "n001")
        self.assertEqual(rows[-1].get("id"), "n070")
        self.assertEqual(mock_alpaca_get.call_count, 2)

        first_params = mock_alpaca_get.call_args_list[0].kwargs.get("params", {})
        second_params = mock_alpaca_get.call_args_list[1].kwargs.get("params", {})
        self.assertEqual(first_params.get("limit"), 50)
        self.assertEqual(second_params.get("limit"), 20)
        self.assertEqual(second_params.get("page_token"), "token-1")

