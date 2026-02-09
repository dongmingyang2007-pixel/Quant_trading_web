from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from trading.llm import LLMIntegrationError


class AIErrorContractTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="ai_error_user", password="secret123")
        self.client.force_login(self.user)
        self.url = reverse("trading:ai_chat")

    @mock.patch("trading.views.api.generate_ai_commentary")
    def test_ai_chat_does_not_expose_internal_exception_details(self, mock_generate):
        mock_generate.side_effect = LLMIntegrationError("secret-upstream-token")
        payload = {
            "context": {"ticker": "AAPL"},
            "message": "analyze this",
            "history": [],
        }
        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body.get("error_code"), "ai_chat_failed")
        self.assertIn("message", body)
        self.assertEqual(body.get("error"), body.get("message"))
        self.assertNotIn("secret-upstream-token", json.dumps(body, ensure_ascii=False))
