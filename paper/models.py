from __future__ import annotations

import uuid
from django.conf import settings
from django.db import models


def default_positions() -> dict:
    return {}


def default_curve() -> list:
    return []


class PaperTradingSession(models.Model):
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("running", "Running"),
        ("paused", "Paused"),
        ("stopped", "Stopped"),
        ("error", "Error"),
    ]

    session_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="paper_sessions")
    name = models.CharField(max_length=120, blank=True, default="")
    ticker = models.CharField(max_length=16)
    benchmark = models.CharField(max_length=16, blank=True, default="")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="running")
    config = models.JSONField(default=dict, blank=True)
    current_cash = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    initial_cash = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    current_positions = models.JSONField(default=default_positions, blank=True)
    equity_curve = models.JSONField(default=default_curve, blank=True)
    last_equity = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    interval_seconds = models.PositiveIntegerField(default=300)
    last_run_at = models.DateTimeField(null=True, blank=True)
    next_run_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "-updated_at"]),
            models.Index(fields=["status", "next_run_at"]),
        ]
        ordering = ["-updated_at"]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.ticker} ({self.status})"


class PaperTrade(models.Model):
    SIDE_CHOICES = [("buy", "Buy"), ("sell", "Sell")]

    session = models.ForeignKey(PaperTradingSession, on_delete=models.CASCADE, related_name="trades")
    symbol = models.CharField(max_length=16)
    side = models.CharField(max_length=8, choices=SIDE_CHOICES)
    quantity = models.DecimalField(max_digits=18, decimal_places=6)
    price = models.DecimalField(max_digits=18, decimal_places=6)
    notional = models.DecimalField(max_digits=18, decimal_places=2)
    executed_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-executed_at"]
        indexes = [models.Index(fields=["session", "-executed_at"])]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.side} {self.quantity} {self.symbol} @ {self.price}"
