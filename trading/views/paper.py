from __future__ import annotations

from datetime import date, timedelta

from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def paper_trading(request):
    """Standalone paper trading console (list/create sessions, view equity/trades)."""
    today = date.today()
    start_default = today - timedelta(days=365)
    return render(
        request,
        "trading/paper.html",
        {
            "paper_start_default": start_default.isoformat(),
            "paper_end_default": today.isoformat(),
        },
    )
