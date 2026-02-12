from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class AccountPageContractTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="account_contract_user", password="pass123")
        self.client.force_login(self.user)

    def test_account_page_keeps_tab_and_settings_contract_nodes(self):
        response = self.client.get(reverse("trading:account"))
        self.assertEqual(response.status_code, 200)

        html = response.content.decode("utf-8")
        required_fragments = [
            'class="account-page container-xl ac-scope ac-v2"',
            'data-role="account-tab-nav"',
            'data-tab-panel="overview"',
            'data-tab-panel="activity"',
            'data-tab-panel="history"',
            'data-tab-panel="settings"',
            'data-role="settings-nav"',
            'data-settings-panel="profile"',
            'data-settings-panel="security"',
            'data-settings-panel="api"',
            'data-role="settings-card"',
            'data-role="settings-body"',
            'data-role="api-secret-field"',
            'id="avatar-upload-modal"',
            'id="timeline-media-modal"',
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, html)
