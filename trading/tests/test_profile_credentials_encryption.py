from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from trading.models import UserProfile
from trading.profile import load_api_credentials, save_api_credentials


@override_settings(SECRET_KEY="test-secret-key-for-credential-encryption")
class ProfileCredentialEncryptionTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="cred_user", password="secret123")
        load_api_credentials(str(self.user.id))

    def test_save_api_credentials_persists_encrypted_payload(self):
        result = save_api_credentials(
            str(self.user.id),
            {
                "alpaca_api_key_id": "key-id-123",
                "alpaca_api_secret_key": "secret-key-123",
            },
        )
        profile = UserProfile.objects.get(user=self.user)
        self.assertEqual(result["alpaca_api_key_id"], "key-id-123")
        self.assertEqual(profile.api_credentials, {})
        self.assertTrue(profile.api_credentials_encrypted)
        self.assertNotIn("key-id-123", profile.api_credentials_encrypted)
        loaded = load_api_credentials(str(self.user.id))
        self.assertEqual(loaded["alpaca_api_key_id"], "key-id-123")
        self.assertEqual(loaded["alpaca_api_secret_key"], "secret-key-123")

    def test_load_api_credentials_migrates_legacy_plaintext(self):
        profile = UserProfile.objects.get(user=self.user)
        profile.api_credentials = {
            "alpaca_api_key_id": "legacy-key",
            "alpaca_api_secret_key": "legacy-secret",
        }
        profile.api_credentials_encrypted = ""
        profile.save(update_fields=["api_credentials", "api_credentials_encrypted", "updated_at"])

        loaded = load_api_credentials(str(self.user.id))
        self.assertEqual(loaded["alpaca_api_key_id"], "legacy-key")

        profile.refresh_from_db()
        self.assertEqual(profile.api_credentials, {})
        self.assertTrue(profile.api_credentials_encrypted)
        self.assertNotIn("legacy-key", profile.api_credentials_encrypted)
