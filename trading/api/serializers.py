from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from rest_framework import serializers

from ..forms import QuantStrategyForm


class StrategyTaskSerializer(serializers.Serializer):
    """Serializer wrapper around QuantStrategyForm for DRF endpoints."""

    ticker = serializers.CharField(max_length=16)
    benchmark_ticker = serializers.CharField(max_length=16, allow_blank=True, required=False)
    start_date = serializers.DateField()
    end_date = serializers.DateField()
    capital = serializers.DecimalField(max_digits=12, decimal_places=2, required=False)
    ml_mode = serializers.CharField(required=False, allow_blank=True)
    strategy_engine = serializers.CharField(required=False, allow_blank=True)
    risk_profile = serializers.CharField(required=False, allow_blank=True)
    short_window = serializers.IntegerField(required=False)
    long_window = serializers.IntegerField(required=False)
    rsi_period = serializers.IntegerField(required=False)
    volatility_target = serializers.FloatField(required=False)
    transaction_cost_bps = serializers.FloatField(required=False)
    slippage_bps = serializers.FloatField(required=False)
    min_holding_days = serializers.IntegerField(required=False)
    entry_threshold = serializers.FloatField(required=False)
    exit_threshold = serializers.FloatField(required=False)
    optimize_thresholds = serializers.BooleanField(required=False)
    train_window = serializers.IntegerField(required=False)
    test_window = serializers.IntegerField(required=False)
    val_ratio = serializers.FloatField(required=False)
    embargo_days = serializers.IntegerField(required=False)
    auto_apply_best_config = serializers.BooleanField(required=False)
    enable_hyperopt = serializers.BooleanField(required=False)
    hyperopt_trials = serializers.IntegerField(required=False)
    hyperopt_timeout = serializers.IntegerField(required=False)
    max_leverage = serializers.FloatField(required=False)
    max_drawdown_stop = serializers.FloatField(required=False)
    daily_exposure_limit = serializers.FloatField(required=False)
    allow_short = serializers.BooleanField(required=False)
    execution_delay_days = serializers.IntegerField(required=False)

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


def _serialize_json_payload(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_json_payload(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_payload(item) for item in value]
    return value


class StrategyPresetSerializer(serializers.Serializer):
    preset_id = serializers.UUIDField(read_only=True)
    name = serializers.CharField(max_length=80)
    description = serializers.CharField(max_length=200, allow_blank=True, required=False)
    payload = serializers.DictField()
    is_default = serializers.BooleanField(required=False)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)

    def validate_payload(self, value: dict[str, Any]) -> dict[str, Any]:
        form = QuantStrategyForm(value, language=self.context.get("language"))
        if not form.is_valid():
            raise serializers.ValidationError(form.errors)
        cleaned = form.cleaned_data
        return _serialize_json_payload(cleaned)


class HistoryMetaSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=120, required=False, allow_blank=True)
    tags = serializers.ListField(child=serializers.CharField(max_length=36), required=False)
    notes = serializers.CharField(required=False, allow_blank=True)
    starred = serializers.BooleanField(required=False)

    def to_internal_value(self, data: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            tags = data.get("tags")
            if isinstance(tags, str):
                normalized = [item.strip() for item in tags.replace("，", ",").split(",") if item.strip()]
                data = dict(data)
                data["tags"] = normalized
        return super().to_internal_value(data)

    def validate_tags(self, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise serializers.ValidationError("Invalid tags payload.")
        cleaned = []
        for item in value:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned
