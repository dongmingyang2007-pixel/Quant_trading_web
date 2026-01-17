from __future__ import annotations

import re
import markdown as md
import bleach
from bleach.linkifier import Linker
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

ALLOWED_TAGS = [
    "p",
    "br",
    "b",
    "strong",
    "i",
    "em",
    "code",
    "pre",
    "ul",
    "ol",
    "li",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "a",
]
ALLOWED_ATTRS = {
    "a": ["href", "title", "rel", "target"],
    "code": ["class"],
    "pre": ["class"],
}

CASHTAG_PATTERN = re.compile(r"(?<![\\w])\\$([A-Z]{2,5})\\b")


def _add_rel(attrs, new=False):
    href = attrs.get("href", "")
    if href:
        attrs["rel"] = "noopener noreferrer nofollow"
        if href.startswith(("http://", "https://")):
            attrs["target"] = "_blank"
    return attrs


@register.filter(name="markdown_format")
def markdown_format(value) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    html = md.markdown(text, extensions=["fenced_code"])
    cleaned = bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
    linker = Linker(callbacks=[_add_rel])
    cleaned = linker.linkify(cleaned)
    return mark_safe(cleaned)


@register.filter(name="parse_cashtags")
def parse_cashtags(value) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)

    def replacer(match: re.Match) -> str:
        symbol = match.group(1)
        return (
            f'<a href="/market/analysis/{symbol}/" class="cashtag" '
            f'title="View {symbol} Market Data">${symbol}</a>'
        )

    return mark_safe(CASHTAG_PATTERN.sub(replacer, text))
