from __future__ import annotations

import logging

from django.conf import settings
from django.contrib.auth.tokens import default_token_generator
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.core.mail import send_mail

from ..forms import ResendActivationForm, SignupForm

LOGGER = logging.getLogger(__name__)


def _language_code(request) -> str:
    return (getattr(request, "LANGUAGE_CODE", "") or "").lower()


def _msg(language: str, english_text: str, chinese_text: str) -> str:
    return chinese_text if language.startswith("zh") else english_text


def _activation_mail_error(language: str) -> str:
    return _msg(
        language,
        "Activation email could not be sent. Please retry later.",
        "激活邮件发送失败，请稍后重试。",
    )


def signup(request):
    language = _language_code(request)
    if request.user.is_authenticated:
        return redirect("trading:backtest")
    if request.method == "POST":
        form = SignupForm(request.POST, language=language)
        if form.is_valid():
            user = form.save()
            user.is_active = False
            user.save()
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            url = request.build_absolute_uri(
                reverse("trading:activate", args=[uid, token])
            )
            subject = _msg(language, "Activate your account", "激活你的账户")
            message = _msg(
                language,
                f"Hello {user.username},\nPlease click the link below to activate your account:\n{url}\nIf this wasn't you, please ignore this email.",
                f"您好，{user.username}：\n请点击以下链接激活您的账户：\n{url}\n如果非本人操作，请忽略此邮件。",
            )
            mail_error = None
            try:
                send_mail(
                    subject,
                    message,
                    settings.DEFAULT_FROM_EMAIL,
                    [user.email],
                    fail_silently=False,
                )
            except Exception:
                LOGGER.exception("Failed to send activation email for user_id=%s", user.id)
                mail_error = _activation_mail_error(language)
            is_console = (
                settings.EMAIL_BACKEND.endswith("console.EmailBackend")
                if hasattr(settings, "EMAIL_BACKEND")
                else False
            )
            return render(
                request,
                "registration/signup_done.html",
                {
                    "email": user.email,
                    "activation_url": url if (is_console or settings.DEBUG) else None,
                    "mail_error": mail_error,
                },
            )
    else:
        form = SignupForm(language=language)
    return render(request, "registration/signup.html", {"form": form})


def activate(request, uidb64: str, token: str):
    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        from django.contrib.auth import get_user_model

        User = get_user_model()
        user = User.objects.get(pk=uid)
    except Exception:
        user = None
    ok = False
    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        ok = True
    return render(request, "registration/activate_result.html", {"ok": ok})


def resend_activation(request):
    language = _language_code(request)
    if request.user.is_authenticated:
        return redirect("trading:backtest")
    info = None
    error = None
    if request.method == "POST":
        form = ResendActivationForm(request.POST, language=language)
        if form.is_valid():
            email = form.cleaned_data["email"]
            from django.contrib.auth import get_user_model

            User = get_user_model()
            user = User.objects.filter(email__iexact=email).first()
            if not user:
                error = _msg(language, "No account is associated with this email.", "未找到该邮箱对应的账号。")
            elif user.is_active:
                info = _msg(language, "This account is already activated. Please sign in directly.", "该账号已激活，可直接登录。")
            else:
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                token = default_token_generator.make_token(user)
                url = request.build_absolute_uri(
                    reverse("trading:activate", args=[uid, token])
                )
                subject = _msg(language, "Activate your account", "激活你的账户")
                message = _msg(
                    language,
                    f"Hello {user.username},\nPlease click the link below to activate your account:\n{url}\nIf this wasn't you, please ignore this email.",
                    f"您好，{user.username}：\n请点击以下链接激活您的账户：\n{url}\n如果非本人操作，请忽略此邮件。",
                )
                mail_error = None
                try:
                    send_mail(
                        subject,
                        message,
                        settings.DEFAULT_FROM_EMAIL,
                        [user.email],
                        fail_silently=False,
                    )
                except Exception:
                    LOGGER.exception("Failed to resend activation email for user_id=%s", user.id)
                    mail_error = _activation_mail_error(language)
                is_console = (
                    settings.EMAIL_BACKEND.endswith("console.EmailBackend")
                    if hasattr(settings, "EMAIL_BACKEND")
                    else False
                )
                return render(
                    request,
                    "registration/signup_done.html",
                    {
                        "email": user.email,
                        "activation_url": url if (is_console or settings.DEBUG) else None,
                        "mail_error": mail_error,
                    },
                )
    else:
        form = ResendActivationForm(language=language)
    return render(
        request,
        "registration/resend_activation.html",
        {"form": form, "info": info, "error": error},
    )
