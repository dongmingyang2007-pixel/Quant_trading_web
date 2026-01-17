from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, F, Q
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

from datetime import datetime
from typing import Any
import json
import uuid
import bleach
from bleach.css_sanitizer import CSSSanitizer

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
    serialize_posts,
    toggle_like,
    remove_post,
)
from ..forms import CommunityPostForm
from ..history import get_history_record
from ..models import CommunityPost as CommunityPostModel
from ..models import CommunityPostComment
from ..models import CommunityTopic as CommunityTopicModel
from ..models import Notification
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


def _sanitize_post_content(content: str) -> str:
    allowed_tags = [
        "p",
        "b",
        "i",
        "u",
        "em",
        "strong",
        "a",
        "h1",
        "h2",
        "h3",
        "ul",
        "ol",
        "li",
        "blockquote",
        "pre",
        "span",
        "div",
        "br",
    ]
    allowed_attrs = {
        "*": ["style", "class"],
        "a": ["href", "target", "rel", "style", "class"],
    }
    css_sanitizer = CSSSanitizer(allowed_css_properties=["color", "background-color"])
    cleaned = bleach.clean(
        content or "",
        tags=allowed_tags,
        attributes=allowed_attrs,
        strip=True,
        css_sanitizer=css_sanitizer,
    )
    text_only = bleach.clean(cleaned, tags=[], strip=True)
    return cleaned if text_only.strip() else ""


@login_required
def community(request):
    user = request.user
    profile = load_profile(str(user.id))
    display_name = profile.get("display_name") or user.username
    avatar_path = profile.get("avatar_path") or ""
    language = (getattr(request, "LANGUAGE_CODE", "") or "").lower()
    is_zh = language.startswith("zh")
    is_htmx = bool(request.headers.get("HX-Request"))

    def resolve_media(path: str | None) -> str:
        return resolve_media_url(path)

    topics = list_topics()
    topic_choices = [(topic["topic_id"], topic["name"]) for topic in topics]
    requested_topic = request.GET.get("topic") or DEFAULT_TOPIC_ID
    topic_meta = get_topic(requested_topic)
    share_record = None
    share_error = None
    share_record_id = ""
    share_history_id = (request.GET.get("share_history_id") or "").strip()
    if share_history_id:
        record = get_history_record(share_history_id, user_id=str(user.id))
        if record:
            share_record = build_backtest_summary(record)
            share_record_id = share_record.get("record_id") or ""
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
    page_number = request.GET.get("page") or 1
    search_query = (request.GET.get("q") or "").strip()
    focus_post_id = (request.GET.get("post") or "").strip()
    sort = (request.GET.get("sort") or "latest").strip().lower()
    if sort in {"top", "trending"}:
        sort = "trending"
    else:
        sort = "latest"
    posts_qs = list_posts(limit=None, topic_id=topic_value, return_queryset=True)
    posts_qs = posts_qs.filter(status=CommunityPostModel.STATUS_PUBLISHED)
    if search_query:
        posts_qs = posts_qs.filter(
            Q(content__icontains=search_query) | Q(author_display_name__icontains=search_query)
        )
    if focus_post_id:
        posts_qs = posts_qs.filter(post_id=focus_post_id)
    if sort == "trending":
        posts_qs = posts_qs.annotate(
            like_count=Count("liked_by", distinct=True),
            comment_count=Count("comments", distinct=True),
        ).annotate(score=F("like_count") + F("comment_count")).order_by("-score", "-created_at")
    else:
        posts_qs = posts_qs.order_by("-created_at")
    paginator = Paginator(posts_qs, 20)
    page_obj = paginator.get_page(page_number)
    posts = serialize_posts(page_obj.object_list)
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

    next_page_url = None
    if page_obj.has_next():
        query = {"page": page_obj.next_page_number()}
        if requested_topic:
            query["topic"] = requested_topic
        if search_query:
            query["q"] = search_query
        if sort:
            query["sort"] = sort
        next_page_url = f"?{urlencode(query)}"

    show_empty_message = page_obj.number == 1

    if is_htmx:
        return render(
            request,
            "trading/includes/_post_list.html",
            {
                "posts": enriched_posts,
                "avatar_url": resolve_media(avatar_path),
                "display_name": display_name,
                "next_page_url": next_page_url,
                "show_empty_message": show_empty_message,
                "is_htmx": True,
            },
        )

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
            "share_record_id": share_record_id,
            "next_page_url": next_page_url,
            "show_empty_message": show_empty_message,
            "search_query": search_query,
            "sort": sort,
            "is_htmx": False,
        },
    )


@login_required
def community_notifications(request):
    notifications_qs = (
        Notification.objects.filter(recipient=request.user)
        .select_related("actor", "target_post")
        .order_by("-created_at")
    )
    page_number = request.GET.get("page") or 1
    paginator = Paginator(notifications_qs, 20)
    page_obj = paginator.get_page(page_number)
    notifications = list(page_obj.object_list)
    unread_ids = [notice.id for notice in notifications if not notice.is_read]
    if unread_ids:
        Notification.objects.filter(recipient=request.user, id__in=unread_ids).update(is_read=True)
    return render(
        request,
        "trading/community_notifications.html",
        {
            "notifications": notifications,
            "page_obj": page_obj,
        },
    )


@login_required
@ensure_csrf_cookie
def write_post(request, post_id: int | None = None):
    print(f"Received data: {request.body}")
    user = request.user
    profile = load_profile(str(user.id))
    display_name = profile.get("display_name") or user.username
    language = (getattr(request, "LANGUAGE_CODE", "") or "").lower()
    is_zh = language.startswith("zh")

    topics = list_topics()
    topic_choices = [(topic["topic_id"], topic["name"]) for topic in topics]
    topic_ids = {topic["topic_id"] for topic in topics}

    post = None
    requested_id = request.GET.get("id") or post_id
    if requested_id:
        if isinstance(requested_id, str) and not requested_id.isdigit():
            return redirect(reverse("trading:community"))
        post = (
            CommunityPostModel.objects.select_related("topic")
            .filter(pk=requested_id, author=user)
            .first()
        )
        if not post:
            return redirect(reverse("trading:community"))
        if post.status != CommunityPostModel.STATUS_DRAFT:
            return redirect(reverse("trading:community"))

    error_message = None
    selected_topic = post.topic.topic_id if post else DEFAULT_TOPIC_ID
    title_value = post.title if post else ""
    content_value = post.content if post else ""
    draft_id = str(post.pk) if post else ""

    if request.method == "POST":
        is_json = request.content_type and "application/json" in request.content_type
        payload = request.POST
        if is_json:
            try:
                payload = json.loads(request.body or "{}")
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
                return JsonResponse({"status": "error", "message": str(exc)}, status=400)
            if not isinstance(payload, dict):
                return JsonResponse({"status": "error", "message": "invalid_payload"}, status=400)

        action = (payload.get("action") or "").strip().lower()
        status = (payload.get("status") or "").strip().lower()
        if not action and status:
            if status in {"draft", "save_draft"}:
                action = "save_draft"
            elif status in {"published", "publish"}:
                action = "publish"
        title = (payload.get("title") or "").strip()
        raw_content = payload.get("content") or ""
        content = _sanitize_post_content(raw_content)
        topic_id = (payload.get("topic") or "").strip() or DEFAULT_TOPIC_ID
        if topic_id not in topic_ids:
            topic_id = DEFAULT_TOPIC_ID
        topic = CommunityTopicModel.objects.filter(topic_id=topic_id).first()
        if not topic:
            topic = CommunityTopicModel.objects.filter(topic_id=DEFAULT_TOPIC_ID).first()
        if not topic:
            topic = CommunityTopicModel.objects.create(
                topic_id=DEFAULT_TOPIC_ID,
                name=DEFAULT_TOPIC_NAME,
                description="",
                creator_name="system",
            )

        target_post = post
        if not target_post:
            draft_id = payload.get("post_id")
            if draft_id:
                target_post = (
                    CommunityPostModel.objects.select_related("topic")
                    .filter(pk=draft_id, author=user)
                    .first()
                )

        if action == "save_draft":
            try:
                if not target_post:
                    target_post = CommunityPostModel.objects.create(
                        topic=topic,
                        author=user,
                        author_display_name=display_name,
                        title=title,
                        content=content,
                        status=CommunityPostModel.STATUS_DRAFT,
                    )
                else:
                    target_post.topic = topic
                    target_post.title = title
                    target_post.content = content
                    target_post.status = CommunityPostModel.STATUS_DRAFT
                    target_post.save(update_fields=["topic", "title", "content", "status", "updated_at"])
            except Exception as exc:
                return JsonResponse({"status": "error", "message": str(exc)}, status=500)
            return JsonResponse({"status": "success", "post_id": target_post.pk})

        if action == "publish":
            if not title:
                error_message = "标题不能为空。" if is_zh else "Title is required."
            elif not content:
                error_message = "内容不能为空。" if is_zh else "Content is required."
            if error_message:
                if is_json:
                    return JsonResponse({"status": "error", "message": error_message}, status=400)
                selected_topic = topic.topic_id
                title_value = title
                content_value = raw_content
                if target_post:
                    draft_id = str(target_post.pk)
            else:
                try:
                    if not target_post:
                        target_post = CommunityPostModel.objects.create(
                            topic=topic,
                            author=user,
                            author_display_name=display_name,
                            title=title,
                            content=content,
                            status=CommunityPostModel.STATUS_PUBLISHED,
                        )
                    else:
                        target_post.topic = topic
                        target_post.title = title
                        target_post.content = content
                        target_post.status = CommunityPostModel.STATUS_PUBLISHED
                        target_post.save(update_fields=["topic", "title", "content", "status", "updated_at"])
                except Exception as exc:
                    return JsonResponse({"status": "error", "message": str(exc)}, status=500)
                redirect_url = f"{reverse('trading:community')}?topic={topic.topic_id}"
                if is_json:
                    return JsonResponse(
                        {"status": "success", "post_id": target_post.pk, "redirect_url": redirect_url}
                    )
                return redirect(redirect_url)
        else:
            if is_json:
                return JsonResponse({"status": "error", "message": "invalid_action"}, status=400)
            error_message = "Invalid action."

    return render(
        request,
        "trading/community_write.html",
        {
            "display_name": display_name,
            "topics": topics,
            "topic_choices": topic_choices,
            "selected_topic": selected_topic,
            "title_value": title_value,
            "content_value": content_value,
            "draft_id": draft_id,
            "post_error": error_message,
        },
    )


@login_required
def get_user_drafts(request):
    drafts = (
        CommunityPostModel.objects.filter(author=request.user, status=CommunityPostModel.STATUS_DRAFT)
        .order_by("-updated_at")
        .only("id", "title", "updated_at")
    )
    payload = [
        {
            "id": draft.pk,
            "title": draft.title or "",
            "updated_at": draft.updated_at.strftime("%Y-%m-%d %H:%M"),
        }
        for draft in drafts
    ]
    return JsonResponse({"status": "success", "drafts": payload})


@login_required
@require_POST
def delete_post(request, post_id: int):
    post = CommunityPostModel.objects.filter(pk=post_id).first()
    if not post:
        return JsonResponse({"status": "error", "message": "not_found"}, status=404)
    if post.author_id != request.user.id:
        return JsonResponse({"status": "error", "message": "forbidden"}, status=403)
    if post.status != CommunityPostModel.STATUS_DRAFT:
        return JsonResponse({"status": "error", "message": "not_draft"}, status=400)
    try:
        post.delete()
    except Exception as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=500)
    return JsonResponse({"status": "success"})


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
    parent_id = (request.POST.get("parent_id") or "").strip()
    if not post_id:
        return JsonResponse({"error": "missing_post_id"}, status=400)
    if not content:
        record_metric("community.comment.failure", user_id=str(request.user.id), post_id=post_id, error="empty_content")
        return JsonResponse({"error": "empty_content"}, status=400)
    parent = None
    if parent_id:
        parent = CommunityPostComment.objects.filter(post__post_id=post_id, comment_id=parent_id).first()
        if not parent:
            record_metric(
                "community.comment.failure",
                user_id=str(request.user.id),
                post_id=post_id,
                error="parent_not_found",
            )
            return JsonResponse({"error": "parent_not_found"}, status=404)
    full_name = request.user.get_full_name()
    profile = load_profile(str(request.user.id))
    author_name = profile.get("display_name") or full_name or request.user.username
    comment = CommunityComment(
        comment_id=f"comment-{uuid.uuid4().hex[:10]}",
        user_id=str(request.user.id),
        author=author_name,
        content=content,
        created_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        parent_id=parent_id or None,
    )
    added = add_comment(post_id, comment, parent=parent)
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
                "parent_id": added.get("parent_id", ""),
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
