from __future__ import annotations

import base64
from io import BytesIO
import shutil
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from PIL import Image

from trading.models import UserProfile
from django.conf import settings

TEMP_MEDIA_ROOT = Path(settings.BASE_DIR) / "test_media"


def _make_image(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 20), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


@override_settings(
    STATICFILES_STORAGE="django.contrib.staticfiles.storage.StaticFilesStorage",
    MEDIA_ROOT=TEMP_MEDIA_ROOT,
)
class AccountUploadTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        TEMP_MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA_ROOT, ignore_errors=True)

    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="tester", password="pass123")
        self.client = Client()
        self.client.force_login(self.user)

    def test_avatar_upload_persists_file(self):
        avatar_bytes = _make_image("green")
        response = self.client.post(
            reverse("trading:account"),
            {
                "action": "profile",
                "display_name": "Tester",
                "cover_color": "#123456",
                "bio": "hello",
            },
            format="multipart",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)

        avatar_file = SimpleUploadedFile("avatar.jpg", avatar_bytes, content_type="image/jpeg")
        response = self.client.post(
            reverse("trading:account"),
            {
                "action": "profile",
                "display_name": "Tester",
                "cover_color": "#123456",
                "bio": "hello",
                "avatar": avatar_file,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        profile = UserProfile.objects.get(user=self.user)
        self.assertTrue(profile.avatar_path)
        self.assertIn("/avatar-", f"/{profile.avatar_path}")
        avatar_file = Path(settings.MEDIA_ROOT) / profile.avatar_path
        self.assertTrue(avatar_file.exists())

    def test_feature_upload_via_cropped_data(self):
        raw = _make_image("blue")
        data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        response = self.client.post(
            reverse("trading:account"),
            {
                "action": "profile",
                "display_name": "Tester",
                "cover_color": "#654321",
                "bio": "world",
                "feature_cropped_data": data_url,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        profile = UserProfile.objects.get(user=self.user)
        self.assertTrue(profile.feature_image_path)
        self.assertIn("/feature-", f"/{profile.feature_image_path}")
        feature_file = Path(settings.MEDIA_ROOT) / profile.feature_image_path
        self.assertTrue(feature_file.exists())
