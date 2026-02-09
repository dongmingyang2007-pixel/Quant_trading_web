from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class AIChatStreamTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="ai_stream_user", password="secret123")
        self.client.force_login(self.user)
        self.url = reverse("trading:ai_chat_stream")

    def test_empty_payload_returns_400_instead_of_500(self):
        response = self.client.post(
            self.url,
            data=json.dumps({"context": {}, "message": "", "history": []}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertIn("error", payload)
        self.assertIn("request_id", payload)

    @mock.patch("trading.views.api.generate_ai_commentary")
    def test_stream_sanitizes_payload_before_invocation(self, mock_generate):
        mock_generate.return_value = {
            "answer": "ok",
            "selected_model": "mock-model",
            "timings_ms": {"total": 1},
        }
        raw_payload = {
            "context": {"symbol": "AAPL"},
            "message": "hello",
            "history": [
                {"role": "user", "content": "x" * 5005},
                "bad-entry",
                {"role": "assistant", "content": "short"},
            ],
            "show_thoughts": False,
            "enable_web": True,
            "web_query": 123,
            "web_max_results": "999",
            "tools": ["  search  ", "", 123],
            "tool_choice": 12,
            "response_schema": "invalid",
            "response_format": [],
            "rag_query": 88,
            "rag_top_k": "999",
            "rag_context": "r" * 5000,
            "images": ["img://1", "", 2, "img://3", "img://4", "img://5"],
            "extra_params": ["invalid"],
        }
        response = self.client.post(self.url, data=json.dumps(raw_payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)

        stream_text = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in response.streaming_content)
        self.assertIn("event: end", stream_text)
        mock_generate.assert_called_once()

        kwargs = mock_generate.call_args.kwargs
        self.assertIsNone(kwargs["web_query"])
        self.assertEqual(kwargs["web_max_results"], 12)
        self.assertEqual(kwargs["tools"], ["search", "123"])
        self.assertIsNone(kwargs["tool_choice"])
        self.assertIsNone(kwargs["response_schema"])
        self.assertIsNone(kwargs["response_format"])
        self.assertIsNone(kwargs["rag_query"])
        self.assertEqual(kwargs["rag_top_k"], 20)
        self.assertEqual(len(kwargs["rag_context"]), 2000)
        self.assertEqual(kwargs["images"], ["img://1", "2", "img://3", "img://4"])
        self.assertIsNone(kwargs["extra_params"])
        self.assertEqual(len(kwargs["history"]), 2)
        self.assertEqual(len(kwargs["history"][0]["content"]), 4000)
        self.assertEqual(kwargs["history"][0]["role"], "user")
