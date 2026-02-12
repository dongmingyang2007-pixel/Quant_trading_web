from __future__ import annotations

from unittest import mock

from django.test import SimpleTestCase

from trading.views import market as market_views


class MarketAiSummaryParserTests(SimpleTestCase):
    def test_parse_ai_json_accepts_code_fence_and_single_quotes(self):
        raw = """```json
{'event': '事件A', 'impact': '影响B', 'implication': '暗示C'}
```"""
        parsed = market_views._parse_ai_json(raw)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("event"), "事件A")
        self.assertEqual(parsed.get("impact"), "影响B")
        self.assertEqual(parsed.get("implication"), "暗示C")

    def test_normalize_ai_struct_accepts_synonyms(self):
        payload = {
            "key_event": "新品发布",
            "market_impact": "成交量上升",
            "trading_hint": "关注回踩",
        }
        normalized = market_views._normalize_ai_struct(payload)
        self.assertEqual(
            normalized,
            {
                "event": "新品发布",
                "impact": "成交量上升",
                "implication": "关注回踩",
            },
        )

    @mock.patch("trading.views.market.resolve_api_credential", return_value="fake-key")
    @mock.patch("trading.views.market._build_ai_summary")
    @mock.patch("trading.views.market._build_ai_summary_struct_llm", return_value=None)
    @mock.patch("trading.views.market._build_ai_summary_llm")
    def test_build_ai_summary_with_meta_falls_back_to_text_llm(
        self,
        mock_text_llm,
        _mock_struct_llm,
        mock_fallback_builder,
        _mock_resolve_key,
    ):
        mock_fallback_builder.return_value = "回退摘要"
        mock_text_llm.return_value = "核心事件：发布财报。市场影响：波动加大。交易暗示：控制仓位。"

        summary, meta, structured = market_views._build_ai_summary_with_meta(
            {"name": "TEST"},
            [{"title": "Headline"}],
            lang_prefix="zh",
            symbol="TEST",
            user=object(),
        )

        self.assertEqual(meta.get("status"), "llm")
        self.assertEqual(meta.get("source"), "bailian")
        self.assertNotIn("格式异常", meta.get("message", ""))
        self.assertIn("核心事件", summary)
        self.assertIsInstance(structured, dict)
        self.assertIn("event", structured)
