from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordResetForm
from django.conf import settings
from django.http import Http404
from django.shortcuts import render
from django.urls import reverse
import uuid
from datetime import datetime
from typing import Any
from django.utils import timezone

from ..forms import ProfileForm
from ..history import load_history
from ..profile import load_profile, save_profile
from ..storage_utils import (
    delete_media_file,
    save_uploaded_file,
    describe_image_error,
    decode_data_url_image,
    resolve_media_url,
)
from ..community import list_posts
from ..models import UserProfile, CommunityPost as CommunityPostModel, CommunityPostComment, CommunityPostLike
from ..i18n_messages import translate_list


def _lang_helpers(request):
    language = (getattr(request, "LANGUAGE_CODE", "") or "").lower()
    lang_is_zh = language.startswith("zh")

    def _msg(english: str, chinese: str) -> str:
        return chinese if lang_is_zh else english

    return language, lang_is_zh, _msg


@login_required
def account(request):
    user = request.user
    language, lang_is_zh, _msg = _lang_helpers(request)
    success_message = None
    error_message = None
    profile_success = None
    profile_error = None

    history_runs = load_history(user_id=str(user.id))
    history_briefs: list[dict[str, Any]] = []
    for entry in history_runs:
        entry["warnings_localized"] = translate_list(entry.get("warnings") or [], language)
        history_briefs.append(
            {
                "record_id": entry.get("record_id"),
                "label": f"{entry.get('ticker', 'Strategy')} · {entry.get('engine', '')}",
                "ticker": entry.get("ticker"),
                "engine": entry.get("engine"),
                "period": f"{entry.get('start_date', '--')} → {entry.get('end_date', '--')}",
                "stats": entry.get("stats") or {},
            }
        )
    profile_data = load_profile(str(user.id))
    display_name = profile_data.get("display_name") or user.username
    avatar_path = profile_data.get("avatar_path") or ""
    feature_path = profile_data.get("feature_image_path") or ""
    gallery_paths = profile_data.get("gallery_paths") or []
    if isinstance(gallery_paths, str):
        gallery_paths = [gallery_paths] if gallery_paths else []
    elif not isinstance(gallery_paths, list):
        gallery_paths = []

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "password-reset":
            if not user.email:
                error_message = _msg(
                    "This account has no email on record, so we cannot send reset instructions.",
                    "账号未绑定邮箱，无法发送重设密码邮件。",
                )
            else:
                reset_form = PasswordResetForm({"email": user.email})
                if reset_form.is_valid():
                    try:
                        reset_form.save(
                            request=request,
                            from_email=settings.DEFAULT_FROM_EMAIL,
                            use_https=request.is_secure(),
                        )
                        success_message = _msg(
                            f"Password reset instructions have been emailed to {user.email}. Check spam if you don’t see them soon.",
                            f"已向 {user.email} 发送重设密码邮件，如未收到请检查垃圾箱或稍后重试。",
                        )
                    except Exception as exc:  # pragma: no cover - 邮件异常
                        error_message = _msg(f"Failed to send: {exc}", f"发送失败：{exc}")
                else:
                    error_message = _msg(
                        "We couldn’t send the email. Please try again later or contact support.",
                        "发送失败，请稍后再试或联系管理员。",
                    )
            profile_form = ProfileForm(initial=profile_data)
        elif action == "profile":
            profile_form = ProfileForm(request.POST, request.FILES)
            if profile_form.is_valid():
                cleaned = profile_form.cleaned_data
                profile_data["display_name"] = cleaned.get("display_name", "")
                profile_data["cover_color"] = cleaned.get("cover_color", "#116e5f")
                profile_data["bio"] = cleaned.get("bio", "")

                image_error = None

                avatar_file = cleaned.get("avatar")
                if avatar_file and not image_error:
                    try:
                        new_avatar_path = save_uploaded_file(
                            avatar_file,
                            subdir=f"profiles/{user.id}",
                            filename_prefix="avatar",
                        )
                    except ValueError as exc:
                        image_error = describe_image_error(exc)
                    else:
                        old_avatar = profile_data.get("avatar_path")
                        if old_avatar:
                            delete_media_file(old_avatar)
                        profile_data["avatar_path"] = new_avatar_path

                feature_file = None
                cropped_feature_data = profile_form.cleaned_data.get("feature_cropped_data")
                if cropped_feature_data:
                    feature_file = decode_data_url_image(
                        cropped_feature_data,
                        filename_prefix=f"feature-{user.id}",
                    )
                if not feature_file:
                    feature_file = cleaned.get("feature_image")
                if feature_file and not image_error:
                    try:
                        new_feature_path = save_uploaded_file(
                            feature_file,
                            subdir=f"profiles/{user.id}",
                            filename_prefix="feature",
                        )
                    except ValueError as exc:
                        image_error = describe_image_error(exc)
                    else:
                        old_feature = profile_data.get("feature_image_path")
                        if old_feature:
                            delete_media_file(old_feature)
                        profile_data["feature_image_path"] = new_feature_path

                profile_data["gallery_paths"] = gallery_paths
                save_profile(str(user.id), profile_data)

                success_notes: list[str] = []
                if image_error:
                    profile_error = image_error
                    success_notes.append(
                        _msg(
                            "Profile saved, but some images failed validation.",
                            "资料已保存，但部分图片未能通过校验。",
                        )
                    )
                else:
                    success_notes.append(_msg("Profile updated.", "个人资料已更新。"))
                profile_success = " ".join(success_notes).strip()

                display_name = profile_data.get("display_name") or user.username
                avatar_path = profile_data.get("avatar_path") or ""
                feature_path = profile_data.get("feature_image_path") or ""
                profile_form = ProfileForm(initial=profile_data)
            else:
                profile_error = _msg(
                    "Failed to save profile. Please check the form fields.",
                    "保存失败，请检查填写内容。",
                )
        elif action == "remove-feature":
            if feature_path:
                delete_media_file(feature_path)
                profile_data["feature_image_path"] = ""
                feature_path = ""
                save_profile(str(user.id), profile_data)
                profile_success = _msg("Feature image removed.", "已移除展示照片。")
            else:
                profile_error = _msg("No feature image to remove.", "当前没有展示照片。")
            profile_form = ProfileForm(initial=profile_data)
        elif action == "remove-gallery":
            target = request.POST.get("image")
            if target and target in gallery_paths:
                delete_media_file(target)
                gallery_paths = [item for item in gallery_paths if item != target]
                profile_data["gallery_paths"] = gallery_paths
                save_profile(str(user.id), profile_data)
                profile_success = _msg("Gallery image removed.", "已移除一张展示照片。")
            else:
                profile_error = _msg("Could not find the selected gallery image.", "未找到指定的展示照片。")
            profile_form = ProfileForm(initial=profile_data)
        else:
            profile_form = ProfileForm(initial=profile_data)
    else:
        profile_form = ProfileForm(initial=profile_data)

    def _localize(dt: datetime) -> tuple[datetime, str]:
        aware = dt if timezone.is_aware(dt) else timezone.make_aware(dt, timezone.utc)
        localized = timezone.localtime(aware)
        return localized, localized.strftime("%Y-%m-%d %H:%M")

    def _build_activity_entries() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        posts = (
            CommunityPostModel.objects.filter(author_id=user.id)
            .select_related("topic")
            .order_by("-created_at")[:120]
        )
        for post in posts:
            localized, display_time = _localize(post.created_at)
            topic_name = post.topic.name if getattr(post, "topic", None) else ""
            entries.append(
                {
                    "kind": "post",
                    "time": localized,
                    "display_time": display_time,
                    "title": _msg(
                        f"Published topic #{topic_name}" if topic_name else "Published a post",
                        f"发布了话题 #{topic_name}" if topic_name else "发布了社区动态",
                    ),
                    "content": post.content,
                    "excerpt": post.content,
                    "image": resolve_media_url(post.image_path),
                    "topic": topic_name,
                }
            )
        comments = (
            CommunityPostComment.objects.filter(author_id=user.id)
            .select_related("post__topic", "post__author")
            .order_by("-created_at")[:120]
        )
        for comment in comments:
            localized, display_time = _localize(comment.created_at)
            post = comment.post
            topic_name = post.topic.name if getattr(post, "topic", None) else ""
            author_name = post.author_display_name or _msg("a user", "用户")
            entries.append(
                {
                    "kind": "comment",
                    "time": localized,
                    "display_time": display_time,
                    "title": _msg(
                        f"Commented on {author_name}'s post",
                        f"评论了 {author_name} 的帖子",
                    ),
                    "content": comment.content,
                    "excerpt": comment.content,
                    "topic": topic_name,
                }
            )
        likes = (
            CommunityPostLike.objects.filter(user_id=user.id)
            .select_related("post__topic", "post__author")
            .order_by("-created_at")[:120]
        )
        for like in likes:
            localized, display_time = _localize(like.created_at)
            post = like.post
            topic_name = post.topic.name if getattr(post, "topic", None) else ""
            author_name = post.author_display_name or _msg("a user", "用户")
            entries.append(
                {
                    "kind": "like",
                    "time": localized,
                    "display_time": display_time,
                    "title": _msg(
                        f"Liked {author_name}'s post",
                        f"赞了 {author_name} 的帖子",
                    ),
                    "content": "",
                    "excerpt": post.content,
                    "topic": topic_name,
                    "image": resolve_media_url(post.image_path),
                }
            )
        entries.sort(key=lambda item: item["time"], reverse=True)
        return entries[:25]

    activity_entries = _build_activity_entries()

    # Activity statistics for the hero cards
    stat_counts = {"post": 0, "comment": 0, "like": 0}
    for item in activity_entries:
        if item.get("kind") in stat_counts:
            stat_counts[item["kind"]] += 1
    level_value = max(1, min(9, history_runs.__len__() // 3 + 1))
    level_progress = min(100, history_runs.__len__() * 4)
    account_stats = [
        {"label": _msg("Posts", "我的发帖"), "value": stat_counts["post"]},
        {"label": _msg("Replies", "我的回复"), "value": stat_counts["comment"]},
        {"label": _msg("Likes", "收到点赞"), "value": stat_counts["like"]},
    ]
    quick_actions = [
        {
            "icon": "📊",
            "label": _msg("Backtest hub", "策略回测"),
            "desc": _msg("Flagship configuration", "旗舰配置一键运行"),
            "url": reverse("trading:backtest"),
            "color": "action-blue",
        },
        {
            "icon": "📰",
            "label": _msg("Market pulse", "股市情报"),
            "desc": _msg("Gainers & decliners", "涨跌榜及时掌握"),
            "url": reverse("trading:market_insights"),
            "color": "action-orange",
        },
        {
            "icon": "🎓",
            "label": _msg("Learning hub", "学习中心"),
            "desc": _msg("Hypothesis playbooks", "知识卡片体系"),
            "url": reverse("trading:learning_center"),
            "color": "action-purple",
        },
        {
            "icon": "💬",
            "label": _msg("Community", "社区广场"),
            "desc": _msg("Share insights", "交流策略灵感"),
            "url": reverse("trading:community"),
            "color": "action-pink",
        },
    ]

    gallery_media = [
        {"url": resolve_media_url(path), "path": path}
        for path in gallery_paths
        if path
    ]
    profile_media = {
        "avatar": resolve_media_url(avatar_path),
        "feature": resolve_media_url(feature_path),
        "gallery": gallery_media,
    }

    context = {
        "user_info": {
            "username": user.username,
            "display_name": display_name,
            "email": user.email or _msg("Not linked", "未绑定"),
            "date_joined": user.date_joined,
        },
        "history_runs": history_runs,
        "password_success": success_message,
        "password_error": error_message,
        "profile": profile_data,
        "profile_form": profile_form,
        "profile_success": profile_success,
        "profile_error": profile_error,
        "profile_media": profile_media,
        "activity_entries": activity_entries,
        "account_stats": account_stats,
        "quick_actions": quick_actions,
        "level_value": level_value,
        "level_progress": level_progress,
        "history_delete_template": request.build_absolute_uri(
            reverse("trading:delete_history", kwargs={"record_id": "placeholder"})
        ),
        "history_load_url": request.build_absolute_uri(reverse("trading:backtest")),
        "history_compare_url": reverse("trading:history_compare"),
        "history_briefs": history_briefs,
    }
    return render(request, "trading/account.html", context)


@login_required
def profile_public(request, profile_slug: uuid.UUID):
    language, lang_is_zh, _msg = _lang_helpers(request)
    try:
        profile_obj = UserProfile.objects.select_related("user").get(slug=profile_slug)
    except UserProfile.DoesNotExist as exc:  # pragma: no cover
        raise Http404(_msg("User not found.", "用户不存在")) from exc
    target = profile_obj.user
    profile = load_profile(str(target.id))

    avatar_url = resolve_media_url(profile.get("avatar_path"))
    feature_url = resolve_media_url(profile.get("feature_image_path"))
    display_name = profile.get("display_name") or target.username

    all_posts = list_posts(limit=200, user_id=str(target.id))
    sanitized_posts: list[dict[str, Any]] = []
    for post in all_posts[:10]:
        sanitized_posts.append(
            {
                "content": post.get("content", ""),
                "created_at": post.get("created_at"),
                "image_url": resolve_media_url(post.get("image_path")),
                "topic": post.get("topic_name"),
            }
        )

    gallery_paths = profile.get("gallery_paths") or []
    if isinstance(gallery_paths, str):
        gallery_paths = [gallery_paths] if gallery_paths else []
    elif not isinstance(gallery_paths, list):
        gallery_paths = []
    gallery_media = [
        {"url": resolve_media_url(path), "path": path}
        for path in gallery_paths
        if path
    ]

    public_user = {
        "username": target.username,
        "display_name": display_name,
        "date_joined": target.date_joined,
        "email_bound": bool(target.email),
    }

    profile_public_view = {
        "cover_color": profile.get("cover_color", "#116e5f"),
        "bio": profile.get("bio", ""),
    }

    context = {
        "public_user": public_user,
        "profile": profile_public_view,
        "avatar_url": avatar_url,
        "feature_url": feature_url,
        "is_self": target == request.user,
        "posts": sanitized_posts,
        "stats": {
            "post_count": len(all_posts),
            "joined": target.date_joined,
        },
        "gallery_media": gallery_media,
    }
    return render(request, "trading/profile_public.html", context)
