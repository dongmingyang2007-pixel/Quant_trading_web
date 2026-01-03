from __future__ import annotations

import math

from django import template
from django.utils.safestring import mark_safe

from ..security import sanitize_html_fragment

register = template.Library()


@register.filter(name="sanitize_fragment")
def sanitize_fragment(value) -> str:
    """Sanitize potentially unsafe HTML snippets before rendering."""
    cleaned = sanitize_html_fragment(value)
    return mark_safe(cleaned)


@register.filter(name="percent")
def percent(value, digits=1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    try:
        precision = int(digits)
    except (TypeError, ValueError):
        precision = 1
    fmt = f"{{:.{max(0, precision)}f}}"
    return f"{fmt.format(numeric * 100)}%"
