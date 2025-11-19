
from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import render

from ..learning import get_learning_track, get_learning_tracks


@login_required
def learning_center(request):
    return render(
        request,
        "trading/learning_center.html",
        {"tracks": get_learning_tracks()},
    )


@login_required
def learning_module_detail(request, slug: str):
    module = get_learning_track(slug)
    if not module:
        raise Http404("课程不存在或已下线")

    return render(
        request,
        "trading/learning_detail.html",
        {"module": module},
    )
