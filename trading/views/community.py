from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode
from django.views.decorators.http import require_POST

from datetime import datetime
from typing import Any
import uuid

from ..community import (
    DEFAULT_TOPIC_ID,
    DEFAULT_TOPIC_NAME,
    CommunityComment,
    add_comment,
    append_post,
    build_backtest_summary,
    build_post,
    create_topic,
    get_topic,
    list_posts,
    list_topics,
    toggle_like,
    remove_post,
)
from ..forms import CommunityPostForm
from ..history import get_history_record
from ..profile import load_profile
from ..storage_utils import save_uploaded_file, describe_image_error, decode_data_url_image, resolve_media_url
from ..observability import record_metric


def _build_share_prefill(summary: dict[str, Any]) -> str:
    ticker = summary.get("ticker") or "—"
    engine = summary.get("engine") or "Strategy"
    start_date = summary.get("start_date") or "--"
    end_date = summary.get("end_date") or "--"
    header_zh = f"【回测摘要】{ticker} · {engine} · {start_date} → {end_date}"
    metrics_zh = (
        f"收益率：{summary.get('total_return')} | 夏普：{summary.get('sharpe')} | "
        f"最大回撤：{summary.get('max_drawdown')} | 波动率：{summary.get('volatility')}"
    )
    header_en = f"[Backtest Summary] {ticker} · {engine} · {start_date} → {end_date}"
    metrics_en = (
        f"Return: {summary.get('total_return')} | Sharpe: {summary.get('sharpe')} | "
        f"Max drawdown: {summary.get('max_drawdown')} | Volatility: {summary.get('volatility')}"
    )
    return "\n".join(
        [
            header_zh,
            metrics_zh,
            "我的思考：",
            "",
            header_en,
            metrics_en,
            "Notes:",
        ]
    )


@login_required
def community(request):
    user = request.user
    profile = load_profile(str(user.id))
    display_name = profile.get("display_name") or user.username
    avatar_path = profile.get("avatar_path") or ""
    language = (getattr(request, "LANGUAGE_CODE", "") or "").lower()
    is_zh = language.startswith("zh")

    def resolve_media(path: str | None) -> str:
        return resolve_media_url(path)

    topics = list_topics()
    topic_choices = [(topic["topic_id"], topic["name"]) for topic in topics]
    requested_topic = request.GET.get("topic") or DEFAULT_TOPIC_ID
    topic_meta = get_topic(requested_topic)
    share_record = None
    share_error = None
    share_history_id = (request.GET.get("share_history_id") or "").strip()
    if share_history_id:
        record = get_history_record(share_history_id, user_id=str(user.id))
        if record:
            share_record = build_backtest_summary(record)
        else:
            share_error = "未找到对应的回测记录，无法分享。" if is_zh else "Backtest record not found. Unable to share."

    post_success = None
    post_error = None

    if request.method == "POST":
        form = CommunityPostForm(request.POST, request.FILES, topics=topic_choices)
        target_topic = None
        if form.is_valid():
            content = form.cleaned_data["content"].strip()
            topic_id = form.cleaned_data.get("topic") or ""
            new_topic_name = (form.cleaned_data.get("new_topic_name") or "").strip()
            new_topic_description = (form.cleaned_data.get("new_topic_description") or "").strip()

            target_topic = None
            if new_topic_name:
                existing = next(
                    (topic for topic in topics if topic["name"].lower() == new_topic_name.lower()),
                    None,
                )
                if existing:
                    target_topic = existing
                else:
                    target_topic = create_topic(
                        new_topic_name,
                        new_topic_description,
                        creator_id=str(user.id),
                        creator_name=display_name,
                    )
                    topics.insert(0, target_topic)
                    topic_choices.insert(0, (target_topic["topic_id"], target_topic["name"]))
            elif topic_id:
                target_topic = get_topic(topic_id)
            else:
                target_topic = get_topic(DEFAULT_TOPIC_ID)

            image_file = None
            cropped_data = form.cleaned_data.get("image_cropped_data")
            if cropped_data:
                image_file = decode_data_url_image(cropped_data, filename_prefix=f"community-{user.id}")
            if (not image_file) and request.FILES:
                image_file = form.cleaned_data.get("image")
            image_path = None
            if image_file:
                try:
                    image_path = save_uploaded_file(
                        image_file,
                        subdir=f"community/{user.id}",
                        filename_prefix="post",
                    )
                except ValueError as exc:
                    post_error = describe_image_error(exc)
            backtest_record_id = (form.cleaned_data.get("backtest_record_id") or "").strip()
            if backtest_record_id and not post_error:
                record = get_history_record(backtest_record_id, user_id=str(user.id))
                if record:
                    backtest_record_id = record.get("record_id") or ""
                else:
                    post_error = "回测记录已不存在，无法关联发布。" if is_zh else "Backtest record is missing and cannot be attached."
                    backtest_record_id = ""
            if not content and not post_error:
                post_error = "内容不能为空。"
            if not post_error:
                append_post(
                    build_post(
                        topic_id=target_topic.get("topic_id", DEFAULT_TOPIC_ID),
                        topic_name=target_topic.get("name", DEFAULT_TOPIC_NAME),
                        user_id=str(user.id),
                        author=display_name,
                        content=content,
                        image_path=image_path,
                        backtest_record_id=backtest_record_id or None,
                    )
                )
                post_success = "已发布，与社区成员共同交流。"
                record_metric(
                    "community.post.success",
                    user_id=str(user.id),
                    topic_id=target_topic.get("topic_id", DEFAULT_TOPIC_ID),
                )
                requested_topic = target_topic.get("topic_id", DEFAULT_TOPIC_ID)
                topic_meta = target_topic
                form = CommunityPostForm(topics=topic_choices, initial={"topic": requested_topic})
        else:
            post_error = "提交失败，请检查内容后重试。"
        if post_error:
            topic_id = target_topic.get("topic_id") if target_topic else None
            record_metric(
                "community.post.failure",
                user_id=str(user.id),
                topic_id=topic_id,
            )
    else:
        initial = {"topic": topic_meta.get("topic_id")}
        if share_record:
            initial["content"] = _build_share_prefill(share_record)
            initial["backtest_record_id"] = share_record.get("record_id")
        form = CommunityPostForm(topics=topic_choices, initial=initial)

    if not post_success and request.GET.get("deleted") == "1":
        post_success = "帖子已删除。"
    if request.GET.get("delete_error"):
        reason = request.GET.get("delete_error")
        error_map = {
            "not_found": "未找到对应的帖子。",
            "forbidden": "只能删除自己发布的帖子。",
            "invalid": "请求无效，请刷新后重试。",
        }
        post_error = error_map.get(reason, "删除失败，请稍后再试。")
    if share_error and not post_error:
        post_error = share_error

    topic_filter = topic_meta.get("topic_id")
    topic_value = topic_filter if topic_filter and topic_filter != "all" else None
    posts = list_posts(limit=200, topic_id=topic_value)
    enriched_posts: list[dict[str, Any]] = []
    for post in posts:
        post = post.copy()
        post["avatar_url"] = resolve_media(post.get("avatar_path"))
        post["image_url"] = resolve_media(post.get("image_path"))
        post["display_name"] = post.get("author")
        post["user_id"] = post.get("user_id")
        post["profile_slug"] = post.get("user_slug")
        liked_by = post.get("liked_by") or []
        post["liked"] = str(user.id) in liked_by
        post["can_delete"] = post.get("user_id") == str(user.id)
        comment_entries: list[dict[str, Any]] = []
        for comment in post.get("comments") or []:
            comment_entry = comment.copy()
            comment_entry["avatar_url"] = resolve_media(comment_entry.get("avatar_path"))
            comment_entries.append(comment_entry)
        post["comments"] = comment_entries
        post["comment_count"] = len(comment_entries)
        enriched_posts.append(post)

    return render(
        request,
        "trading/community.html",
        {
            "profile": profile,
            "display_name": display_name,
            "avatar_url": resolve_media(avatar_path),
            "posts": enriched_posts,
            "post_form": form,
            "post_success": post_success,
            "post_error": post_error,
            "topics": topics,
            "active_topic": topic_meta,
            "share_record": share_record,
            "share_history_id": share_history_id,
        },
    )


@login_required
@require_POST
def community_like(request):
    post_id = request.POST.get("post_id", "")
    if not post_id:
        return JsonResponse({"error": "missing_post_id"}, status=400)
    result = toggle_like(post_id, str(request.user.id))
    if result is None:
        record_metric("community.like.failure", user_id=str(request.user.id), post_id=post_id, error="not_found")
        return JsonResponse({"error": "not_found"}, status=404)
    record_metric(
        "community.like.toggle",
        user_id=str(request.user.id),
        post_id=post_id,
        liked=result["liked"],
    )
    return JsonResponse({"liked": result["liked"], "like_count": result["like_count"]})


@login_required
@require_POST
def community_comment(request):
    post_id = request.POST.get("post_id", "")
    content = (request.POST.get("content") or "").strip()
    if not post_id:
        return JsonResponse({"error": "missing_post_id"}, status=400)
    if not content:
        record_metric("community.comment.failure", user_id=str(request.user.id), post_id=post_id, error="empty_content")
        return JsonResponse({"error": "empty_content"}, status=400)
    full_name = request.user.get_full_name()
    profile = load_profile(str(request.user.id))
    author_name = profile.get("display_name") or full_name or request.user.username
    comment = CommunityComment(
        comment_id=f"comment-{uuid.uuid4().hex[:10]}",
        user_id=str(request.user.id),
        author=author_name,
        content=content,
        created_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    added = add_comment(post_id, comment)
    if added is None:
        record_metric("community.comment.failure", user_id=str(request.user.id), post_id=post_id, error="not_found")
        return JsonResponse({"error": "not_found"}, status=404)
    record_metric(
        "community.comment.success",
        user_id=str(request.user.id),
        post_id=post_id,
    )
    avatar_url = resolve_media_url(profile.get("avatar_path"))
    return JsonResponse(
        {
            "comment": {
                "comment_id": added["comment_id"],
                "author": comment.author,
                "content": comment.content,
                "created_at": comment.created_at,
                "avatar_url": avatar_url,
            }
        }
    )


@login_required
@require_POST
def community_delete(request):
    post_id = (request.POST.get("post_id") or "").strip()
    topic_hint = request.POST.get("topic") or request.GET.get("topic") or DEFAULT_TOPIC_ID
    if not post_id:
        return JsonResponse({"error": "missing_post_id"}, status=400)
    success, payload = remove_post(post_id, str(request.user.id))
    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"
    if success:
        topic_hint = payload.get("topic_id") or topic_hint
        if is_ajax:
            record_metric("community.delete.success", user_id=str(request.user.id), post_id=post_id, topic_id=topic_hint)
            return JsonResponse({"deleted": True, "topic_id": topic_hint, "message": "帖子已删除"})
        query = {"deleted": "1"}
    else:
        error_code = payload.get("error") if isinstance(payload, dict) else "delete_failed"
        if is_ajax:
            status_map = {"not_found": 404, "forbidden": 403}
            record_metric("community.delete.failure", user_id=str(request.user.id), post_id=post_id, error=error_code)
            return JsonResponse({"error": error_code or "delete_failed"}, status=status_map.get(error_code, 400))
        query = {"delete_error": error_code or "delete_failed"}
    if not is_ajax:
        if success:
            record_metric("community.delete.success", user_id=str(request.user.id), post_id=post_id, topic_id=topic_hint)
        else:
            record_metric("community.delete.failure", user_id=str(request.user.id), post_id=post_id, error=error_code)
        query["topic"] = topic_hint
        url = f"{reverse('trading:community')}?{urlencode(query)}"
        return redirect(url)
