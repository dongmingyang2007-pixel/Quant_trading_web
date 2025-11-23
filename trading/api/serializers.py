from __future__ import annotations

from typing import Any

from rest_framework import serializers

from ..forms import QuantStrategyForm
from paper.models import PaperTradingSession


class StrategyTaskSerializer(serializers.Serializer):
    """Serializer wrapper around QuantStrategyForm for DRF endpoints."""

    ticker = serializers.CharField(max_length=16)
    benchmark_ticker = serializers.CharField(max_length=16, allow_blank=True, required=False)
    start_date = serializers.DateField()
    end_date = serializers.DateField()
    capital = serializers.DecimalField(max_digits=12, decimal_places=2, required=False)
    ml_mode = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        form = QuantStrategyForm(attrs, language=self.context.get("language"))
        if not form.is_valid():
            raise serializers.ValidationError(form.errors)
        attrs["_cleaned"] = form.cleaned_data
        attrs["_form_warnings"] = getattr(form, "warnings", [])
        return attrs


class TrainingTaskSerializer(StrategyTaskSerializer):
    tickers = serializers.ListField(
        child=serializers.CharField(max_length=16),
        required=False,
        allow_empty=True,
    )
    engines = serializers.ListField(
        child=serializers.CharField(max_length=32),
        required=False,
        allow_empty=True,
    )

    def to_internal_value(self, data: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            tickers = data.get("tickers")
            if isinstance(tickers, str):
                data = dict(data)
                data["tickers"] = [item.strip() for item in tickers.split(",") if item.strip()]
            engines = data.get("engines")
            if isinstance(engines, str):
                data = dict(data)
                data["engines"] = [item.strip() for item in engines.split(",") if item.strip()]
        return super().to_internal_value(data)


class PaperSessionCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=120, allow_blank=True, required=False)
    initial_cash = serializers.DecimalField(max_digits=14, decimal_places=2, required=False)
    interval_seconds = serializers.IntegerField(required=False)
    params = serializers.DictField()

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        language = self.context.get("language")
        params = attrs.get("params") or {}
        form = QuantStrategyForm(params, language=language)
        if not form.is_valid():
            raise serializers.ValidationError(form.errors)
        attrs["_cleaned"] = form.cleaned_data
        attrs["_form_warnings"] = getattr(form, "warnings", [])
        return attrs
