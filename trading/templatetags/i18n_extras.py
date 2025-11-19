from __future__ import annotations

from django import template

register = template.Library()


def _current_language(context) -> str:
    request = context.get("request")
    lang = ""
    if request is not None:
        lang = getattr(request, "LANGUAGE_CODE", "") or request.COOKIES.get("django_language", "")
    return (lang or "").lower()


@register.simple_tag(takes_context=True)
def bilingual(context, english_text: str = "", chinese_text: str = "") -> str:
    """Return English or Chinese text based on current LANGUAGE_CODE.

    Falls back to English when request/locale is unavailable.
    """
    lang = _current_language(context)
    if lang.startswith("zh"):
        return chinese_text
    return english_text


HEAT_LEVEL_MAP = {
    "低": "low",
    "较低": "low",
    "中": "medium",
    "中等": "medium",
    "较高": "high",
    "高": "high",
    "极高": "very high",
    "极低": "very low",
}


@register.simple_tag(takes_context=True)
def heat_display(context, value) -> str:
    """Return localized 'Heat …' or '热度 …' text."""
    lang = _current_language(context)
    normalized = "" if value is None else str(value).strip()
    if lang.startswith("zh"):
        label = "热度"
        display_value = normalized or "—"
    else:
        label = "Heat"
        display_value = HEAT_LEVEL_MAP.get(normalized, normalized or "—")
    return f"{label} {display_value}".strip()
