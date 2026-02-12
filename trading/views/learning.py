
from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect


@login_required
def learning_center(request):
    return redirect("/backtest/?workspace=trade")


@login_required
def learning_module_detail(request, slug: str):
    return redirect("/backtest/?workspace=trade")
