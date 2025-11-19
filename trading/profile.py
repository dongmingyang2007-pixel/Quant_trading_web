from __future__ import annotations

from typing import Any

from django.contrib.auth import get_user_model

from .models import UserProfile

DEFAULT_PROFILE = {
    "display_name": "",
    "cover_color": "#116e5f",
    "bio": "",
    "avatar_path": "",
    "feature_image_path": "",
    "gallery_paths": [],
}


def _get_or_create_profile(user_id: str) -> UserProfile:
    UserModel = get_user_model()
    user = UserModel.objects.get(pk=user_id)
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


def _normalize_gallery(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def load_profile(user_id: str) -> dict[str, Any]:
    profile_obj = _get_or_create_profile(user_id)
    payload = {
        "display_name": profile_obj.display_name or "",
        "cover_color": profile_obj.cover_color or DEFAULT_PROFILE["cover_color"],
        "bio": profile_obj.bio or "",
        "avatar_path": profile_obj.avatar_path or "",
        "feature_image_path": profile_obj.feature_image_path or "",
        "gallery_paths": _normalize_gallery(profile_obj.gallery_paths),
        "slug": str(profile_obj.slug),
    }
    return payload


def save_profile(user_id: str, profile: dict[str, Any]) -> None:
    profile_obj = _get_or_create_profile(user_id)
    for key in ("display_name", "cover_color", "bio", "avatar_path", "feature_image_path"):
        value = profile.get(key)
        if value is not None:
            setattr(profile_obj, key, value)
    if "gallery_paths" in profile:
        profile_obj.gallery_paths = _normalize_gallery(profile.get("gallery_paths"))
    profile_obj.save()
