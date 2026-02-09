from __future__ import annotations

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import CommunityPostComment, CommunityPostLike, Notification
from .notifications_cache import invalidate_unread_notifications_cache


@receiver(post_save, sender=CommunityPostLike)
def create_like_notification(sender, instance, created, **kwargs):
    if not created:
        return
    post = instance.post
    recipient = getattr(post, "author", None)
    actor = getattr(instance, "user", None)
    if not recipient or not actor or recipient.pk == actor.pk:
        return
    Notification.objects.create(
        recipient=recipient,
        actor=actor,
        verb=Notification.VERB_LIKED,
        target_post=post,
    )
    invalidate_unread_notifications_cache(recipient.pk)


@receiver(post_save, sender=CommunityPostComment)
def create_comment_notification(sender, instance, created, **kwargs):
    if not created:
        return
    post = instance.post
    recipient = getattr(post, "author", None)
    actor = getattr(instance, "author", None)
    if not recipient or not actor or recipient.pk == actor.pk:
        return
    Notification.objects.create(
        recipient=recipient,
        actor=actor,
        verb=Notification.VERB_COMMENTED,
        target_post=post,
    )
    invalidate_unread_notifications_cache(recipient.pk)
