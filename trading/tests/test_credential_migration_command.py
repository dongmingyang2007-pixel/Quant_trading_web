from __future__ import annotations

from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, override_settings

from trading.models import UserProfile
from trading.profile import load_api_credentials


@override_settings(SECRET_KEY="test-secret-key-for-credential-encryption")
class CredentialMigrationCommandTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="migrate_user", password="secret123")
        load_api_credentials(str(self.user.id))
        profile = UserProfile.objects.get(user=self.user)
        profile.api_credentials = {
            "alpaca_api_key_id": "legacy-id",
            "alpaca_api_secret_key": "legacy-secret",
        }
        profile.api_credentials_encrypted = ""
        profile.save(update_fields=["api_credentials", "api_credentials_encrypted", "updated_at"])

    def test_command_migrates_plaintext_credentials(self):
        stdout = StringIO()
        call_command("migrate_api_credentials_encryption", "--user-id", str(self.user.id), stdout=stdout)
        profile = UserProfile.objects.get(user=self.user)
        self.assertEqual(profile.api_credentials, {})
        self.assertTrue(profile.api_credentials_encrypted)
