from __future__ import annotations

from typing import Any

import bleach
from bleach.sanitizer import Cleaner

ALLOWED_TAGS = [
    "b",
    "strong",
    "i",
    "em",
    "u",
    "small",
    "p",
    "br",
    "ul",
    "ol",
    "li",
    "blockquote",
    "code",
    "pre",
    "a",
]
ALLOWED_ATTRS = {
    "a": ["href", "title", "rel", "target"],
}

_CLEANER = Cleaner(tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)


def sanitize_html_fragment(value: Any) -> str:
    """
    Whitelist HTML fragments coming from LLM /外部输入，防止 XSS。
    非字符串输入将被转换为字符串后处理。
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)

    cleaned = _CLEANER.clean(value)

    def _add_rel(attrs, new=False):
        href = attrs.get("href", "")
        if href:
            attrs["rel"] = "noopener noreferrer nofollow"
            if href.startswith(("http://", "https://")):
                attrs["target"] = "_blank"
        return attrs

    linker = bleach.linkifier.Linker(callbacks=[_add_rel])
    return linker.linkify(cleaned)
