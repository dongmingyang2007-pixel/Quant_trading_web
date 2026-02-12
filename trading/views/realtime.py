from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.urls import reverse
from django.views.decorators.http import require_http_methods


def _redirect_to_shortterm_panel():
    return redirect(f"{reverse('trading:backtest')}?workspace=trade")


@login_required
@require_http_methods(["GET", "POST"])
def realtime_settings(request):
    return _redirect_to_shortterm_panel()


@login_required
@require_http_methods(["GET", "POST"])
def realtime_monitor(request):
    return _redirect_to_shortterm_panel()
