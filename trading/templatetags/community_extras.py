from __future__ import annotations

from django import template
from django.contrib.auth import get_user_model

from ..models import BacktestRecord

register = template.Library()


@register.simple_tag
def get_user_badge(user):
    if not user:
        return None
    resolved_user = user
    if not hasattr(user, "is_staff"):
        try:
            resolved_user = get_user_model().objects.filter(pk=user).only("id", "is_staff").first()
        except (TypeError, ValueError):
            resolved_user = None
    if not resolved_user:
        return None
    if getattr(resolved_user, "is_staff", False):
        return {"label": "Admin", "color": "danger"}
    backtest_count = BacktestRecord.objects.filter(user=resolved_user).count()
    if backtest_count > 5:
        return {"label": "Quant Analyst", "color": "primary"}
    return None
