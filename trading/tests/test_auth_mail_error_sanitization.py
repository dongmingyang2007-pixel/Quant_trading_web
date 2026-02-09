from __future__ import annotations

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(
    DEBUG=False,
    EMAIL_BACKEND="django.core.mail.backends.smtp.EmailBackend",
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class ActivationMailErrorSanitizationTests(TestCase):
    def test_signup_mail_error_is_sanitized_and_activation_url_hidden(self):
        with mock.patch(
            "trading.views.auth.send_mail",
            side_effect=RuntimeError("smtp auth failed: secret=password123"),
        ):
            response = self.client.post(
                reverse("trading:signup"),
                data={
                    "username": "signup_mail_user",
                    "email": "signup_mail_user@example.com",
                    "password1": "StrongPass123!",
                    "password2": "StrongPass123!",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "registration/signup_done.html")
        self.assertIsNone(response.context.get("activation_url"))
        mail_error = response.context.get("mail_error")
        self.assertTrue(mail_error)
        self.assertNotIn("secret=password123", mail_error)
        created = get_user_model().objects.get(username="signup_mail_user")
        self.assertFalse(created.is_active)

    def test_resend_activation_mail_error_is_sanitized_and_activation_url_hidden(self):
        user_model = get_user_model()
        user_model.objects.create_user(
            username="resend_mail_user",
            email="resend_mail_user@example.com",
            password="StrongPass123!",
            is_active=False,
        )
        with mock.patch(
            "trading.views.auth.send_mail",
            side_effect=RuntimeError("smtp auth failed: secret=token456"),
        ):
            response = self.client.post(
                reverse("trading:resend_activation"),
                data={"email": "resend_mail_user@example.com"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "registration/signup_done.html")
        self.assertIsNone(response.context.get("activation_url"))
        mail_error = response.context.get("mail_error")
        self.assertTrue(mail_error)
        self.assertNotIn("secret=token456", mail_error)
