from __future__ import annotations

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class ApiAuthTests(TestCase):
    def test_task_status_requires_auth(self):
        url = reverse("trading:api_v1_task_status", kwargs={"task_id": "sync-test"})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 401)

    @mock.patch("trading.api.views_v1.get_task_status")
    def test_task_status_allows_authenticated(self, mock_status):
        user = get_user_model().objects.create_user(username="auth", password="secret123")
        self.client.force_login(user)
        mock_status.return_value = {"task_id": "sync-test", "state": "SUCCESS"}
        url = reverse("trading:api_v1_task_status", kwargs={"task_id": "sync-test"})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_id"], "sync-test")
