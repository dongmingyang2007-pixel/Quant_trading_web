from __future__ import annotations

from django import template
from django.utils.safestring import mark_safe

from ..security import sanitize_html_fragment

register = template.Library()


@register.filter(name="sanitize_fragment")
def sanitize_fragment(value) -> str:
    """Sanitize potentially unsafe HTML snippets before rendering."""
    cleaned = sanitize_html_fragment(value)
    return mark_safe(cleaned)

