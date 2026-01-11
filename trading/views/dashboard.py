from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.conf import settings
from django.urls import reverse

from ..forms import QuantStrategyForm
from ..headlines import get_global_headlines
from ..history import (
    BacktestRecord,
    append_history,
    get_history_record,
    load_history,
    sanitize_snapshot,
)
from ..backtest_logger import top_runs
from ..strategies import QuantStrategyError, StrategyInput, run_quant_pipeline
from ..observability import ensure_request_id
from ..i18n_messages import translate_list, translate_text
from ..strategy_defaults import ADVANCED_STRATEGY_DEFAULTS


FOCUS_PLACEHOLDER_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' width='320' height='180' viewBox='0 0 320 180'>
    <defs>
        <linearGradient id='grad' x1='0%' y1='0%' x2='100%' y2='100%'>
            <stop offset='0%' stop-color='#f5f5f7'/>
            <stop offset='100%' stop-color='#e8ebf2'/>
        </linearGradient>
    </defs>
    <rect width='320' height='180' fill='url(#grad)' rx='28'/>
    <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='#6e6e73' font-size='22' font-family='-apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial'>市场焦点</text>
</svg>
""".strip()

FOCUS_PLACEHOLDER_DATA = "data:image/svg+xml;charset=UTF-8," + quote(FOCUS_PLACEHOLDER_SVG)


def _ollama_model_choices() -> list[str]:
    env_choices = os.environ.get("OLLAMA_MODEL_CHOICES", "")
    choices: list[str] = []
    if env_choices:
        choices = [item.strip() for item in env_choices.split(",") if item.strip()]
    default_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
    fallbacks = [
        default_model,
        os.environ.get("OLLAMA_SECONDARY_MODEL"),
        "deepseek-r1:8b",
        "llama3.2",
        "llama3.2:3b",
        "qwen2:7b",
    ]
    bailian_env_choices = os.environ.get("BAILIAN_MODEL_CHOICES") or os.environ.get("DASHSCOPE_MODEL_CHOICES") or ""
    bailian_default = os.environ.get("BAILIAN_MODEL") or os.environ.get("DASHSCOPE_MODEL") or "qwen-max"
    bailian_candidates_raw = [bailian_default, "qwen-max", "qwen-plus", "qwen-turbo"]
    bailian_candidates: list[str] = []
    bailian_env_list = bailian_env_choices.split(",") if bailian_env_choices else []
    for item in bailian_env_list:
        normalized = item.strip()
        if normalized:
            if ":" not in normalized:
                normalized = f"bailian:{normalized}"
            if normalized not in bailian_candidates:
                bailian_candidates.append(normalized)
    for candidate in bailian_candidates_raw:
        if not candidate:
            continue
        normalized = str(candidate).strip()
        if not normalized:
            continue
        if ":" not in normalized:
            normalized = f"bailian:{normalized}"
        if normalized not in bailian_candidates:
            bailian_candidates.append(normalized)
    gemini_candidates = [
        os.environ.get("GEMINI_MODEL"),
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-latest",
        "gemini-3.0-pro",
    ]
    ai_provider = (os.environ.get("AI_PROVIDER") or "").strip().lower()
    if ai_provider == "gemini":
        gemini_candidates = [os.environ.get("GEMINI_MODEL", "gemini-3.0-pro")] + gemini_candidates
    if ai_provider == "bailian":
        preferred = bailian_default.strip() if isinstance(bailian_default, str) else str(bailian_default)
        if ":" not in preferred:
            preferred = f"bailian:{preferred}"
        bailian_candidates = [preferred] + [item for item in bailian_candidates if item != preferred]
    for candidate in fallbacks:
        if candidate and candidate.strip() and candidate not in choices:
            choices.append(candidate.strip())
    for candidate in gemini_candidates:
        if candidate and candidate.strip() and candidate not in choices:
            choices.append(candidate.strip())
    for candidate in bailian_candidates:
        if candidate and candidate.strip() and candidate not in choices:
            choices.append(candidate.strip())
    return choices


def _ensure_ai_model_metadata(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    base_choices = _ollama_model_choices()
    existing_choices = payload.get("ai_model_choices") or []
    merged: list[str] = []
    for candidate in (existing_choices if isinstance(existing_choices, list) else []):
        if isinstance(candidate, str):
            trimmed = candidate.strip()
            if trimmed and trimmed not in merged:
                merged.append(trimmed)
    for candidate in base_choices:
        if candidate and candidate not in merged:
            merged.append(candidate)
    selected = str(payload.get("ai_model") or "").strip()
    lowered = selected.lower()
    if lowered.startswith(("bailian:", "dashscope:", "aliyun:")):
        provider = "bailian"
    elif lowered.startswith("gemini"):
        provider = "gemini"
    else:
        provider = (os.environ.get("AI_PROVIDER") or "").strip().lower()
    if provider == "bailian":
        bailian_default = os.environ.get("BAILIAN_MODEL") or os.environ.get("DASHSCOPE_MODEL") or "qwen-max"
        default_model = bailian_default.strip() if isinstance(bailian_default, str) else str(bailian_default)
        if ":" not in default_model:
            default_model = f"bailian:{default_model}"
    elif provider == "gemini":
        default_model = os.environ.get("GEMINI_MODEL", "gemini-3.0-pro")
    else:
        default_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
    if not selected:
        selected = default_model or (merged[0] if merged else "")
    elif selected not in merged:
        merged.insert(0, selected)
    payload["ai_model_choices"] = merged
    payload["ai_model"] = selected or default_model
    payload["ai_provider"] = provider or "ollama"


def _prepare_result_payload(payload: dict[str, Any]) -> None:
    """Attach derived fields for templates (e.g. summary dicts without underscores)."""
    if not isinstance(payload, dict):
        return
    flows = payload.get("capital_flows")
    if isinstance(flows, dict):
        payload["capital_flows_summary"] = flows.get("_summary")


def _normalize_history_snapshot(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    stats = payload.get("stats")
    if isinstance(stats, dict):
        if "execution_stats" not in stats:
            execution = stats.get("execution")
            if isinstance(execution, dict):
                stats["execution_stats"] = execution
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        repro = payload.get("repro")
        if isinstance(repro, dict) and repro:
            metadata = dict(repro)
        else:
            metadata = {}
    params = payload.get("params")
    if not isinstance(params, dict):
        params = {}

    def _stringify_date(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value).strip()

    def _fallback_value(*keys: str) -> str:
        for key in keys:
            val = params.get(key) if key in params else payload.get(key)
            text = _stringify_date(val)
            if text:
                return text
        return ""

    if not metadata.get("requested_start"):
        fallback_start = _fallback_value("start_date")
        if fallback_start:
            metadata["requested_start"] = fallback_start
    if not metadata.get("requested_end"):
        fallback_end = _fallback_value("end_date")
        if fallback_end:
            metadata["requested_end"] = fallback_end
    if not metadata.get("effective_start") and metadata.get("requested_start"):
        metadata["effective_start"] = metadata["requested_start"]
    if not metadata.get("effective_end") and metadata.get("requested_end"):
        metadata["effective_end"] = metadata["requested_end"]

    payload["metadata"] = metadata


def build_strategy_input(cleaned: dict[str, Any], *, request_id: str, user) -> tuple[StrategyInput, dict[str, Any]]:
    config = ADVANCED_STRATEGY_DEFAULTS.copy()
    benchmark = cleaned.get("benchmark_ticker") or config["benchmark_ticker"]
    capital = float(cleaned.get("capital") or config["capital"])
    ml_mode = cleaned.get("ml_mode") or "light"
    ml_mapping = {
        "light": "lightgbm",
        "deep": "lstm",
        "transformer": "transformer",
        "fusion": "seq_hybrid",
    }
    config["ml_model"] = ml_mapping.get(ml_mode, config["ml_model"])
    config["benchmark_ticker"] = benchmark
    config["capital"] = capital

    def _override(key: str, cast=None) -> None:
        if key not in cleaned:
            return
        val = cleaned.get(key)
        if val is None or val == "":
            return
        config[key] = cast(val) if cast else val

    _override("strategy_engine", str)
    _override("risk_profile", str)
    _override("short_window", int)
    _override("long_window", int)
    _override("rsi_period", int)
    _override("volatility_target", float)
    _override("transaction_cost_bps", float)
    _override("slippage_bps", float)
    _override("return_path", str)
    _override("max_adv_participation", float)
    _override("execution_liquidity_buffer", float)
    _override("execution_penalty_bps", float)
    _override("limit_move_threshold", float)
    _override("borrow_cost_bps", float)
    _override("min_holding_days", int)
    _override("entry_threshold", float)
    _override("exit_threshold", float)
    _override("optimize_thresholds", bool)
    _override("train_window", int)
    _override("test_window", int)
    _override("val_ratio", float)
    _override("embargo_days", int)
    _override("auto_apply_best_config", bool)
    _override("enable_hyperopt", bool)
    _override("hyperopt_trials", int)
    _override("hyperopt_timeout", int)
    _override("max_leverage", float)
    _override("max_drawdown_stop", float)
    _override("daily_exposure_limit", float)
    _override("allow_short", bool)
    _override("execution_delay_days", int)

    user_id = str(user.id) if getattr(user, "is_authenticated", False) else None

    strategy_params = StrategyInput(
        ticker=cleaned["ticker"],
        benchmark_ticker=benchmark,
        start_date=cleaned["start_date"],
        end_date=cleaned["end_date"],
        short_window=config["short_window"],
        long_window=config["long_window"],
        rsi_period=config["rsi_period"],
        include_plots=config["include_plots"],
        show_ai_thoughts=config["show_ai_thoughts"],
        risk_profile=config["risk_profile"],
        capital=capital,
        strategy_engine=config["strategy_engine"],
        volatility_target=config["volatility_target"],
        transaction_cost_bps=config["transaction_cost_bps"],
        slippage_bps=config["slippage_bps"],
        execution_penalty_bps=config.get("execution_penalty_bps", 6.0),
        execution_liquidity_buffer=config.get("execution_liquidity_buffer", 0.05),
        max_adv_participation=config.get("max_adv_participation", 0.1),
        borrow_cost_bps=config.get("borrow_cost_bps", 0.0),
        min_holding_days=config["min_holding_days"],
        train_window=config["train_window"],
        test_window=config["test_window"],
        entry_threshold=config["entry_threshold"],
        exit_threshold=config["exit_threshold"],
        max_leverage=config["max_leverage"],
        ml_task=config["ml_task"],
        val_ratio=config["val_ratio"],
        embargo_days=config["embargo_days"],
        optimize_thresholds=config["optimize_thresholds"],
        ml_model=config["ml_model"],
        ml_params=config.get("ml_params"),
        auto_apply_best_config=config["auto_apply_best_config"],
        calibrate_proba=config["calibrate_proba"],
        early_stopping_rounds=config["early_stopping_rounds"],
        enable_hyperopt=config["enable_hyperopt"],
        hyperopt_trials=config["hyperopt_trials"],
        hyperopt_timeout=config["hyperopt_timeout"],
        max_drawdown_stop=config["max_drawdown_stop"],
        daily_exposure_limit=config["daily_exposure_limit"],
        dl_sequence_length=config["dl_sequence_length"],
        dl_hidden_dim=config["dl_hidden_dim"],
        dl_dropout=config["dl_dropout"],
        dl_epochs=config["dl_epochs"],
        dl_batch_size=config["dl_batch_size"],
        dl_num_layers=config["dl_num_layers"],
        rl_engine=config["rl_engine"],
        rl_params=config.get("rl_params"),
        label_style=config["label_style"],
        tb_up=config["tb_up"],
        tb_down=config["tb_down"],
        tb_max_holding=config["tb_max_holding"],
        interest_keywords=None,
        investment_horizon=config["investment_horizon"],
        experience_level=config["experience_level"],
        primary_goal=config["primary_goal"],
        return_path=config["return_path"],
        request_id=request_id,
        user_id=user_id,
        model_version=config["ml_model"],
        data_version=os.environ.get("MARKET_DATA_VERSION"),
        lot_size=config.get("lot_size", 1),
        max_weight=config.get("max_weight"),
        min_weight=config.get("min_weight"),
        max_holdings=config.get("max_holdings"),
        sector_caps=config.get("sector_caps"),
        turnover_cap=config.get("turnover_cap"),
        allow_short=config.get("allow_short", True),
        limit_move_threshold=config.get("limit_move_threshold"),
        execution_delay_days=config.get("execution_delay_days", 1),
    )
    return strategy_params, config


@login_required
def backtest(request):
    request_id = ensure_request_id(request)
    language = (getattr(request, "LANGUAGE_CODE", "") or "").lower()
    lang_is_zh = language.startswith("zh")
    default_start = date.today() - timedelta(days=365)
    default_end = date.today()
    default_initial = {
        "start_date": default_start,
        "end_date": default_end,
        "benchmark_ticker": ADVANCED_STRATEGY_DEFAULTS["benchmark_ticker"],
        "capital": ADVANCED_STRATEGY_DEFAULTS["capital"],
    }

    result: dict[str, Any] | None = None
    result_json = "{}"
    history_message = ""
    history_error = ""

    if request.method == "POST":
        form = QuantStrategyForm(request.POST, language=language)
        if form.is_valid():
            cleaned = form.cleaned_data
            try:
                strategy_params, config = build_strategy_input(cleaned, request_id=request_id, user=request.user)

                result = run_quant_pipeline(strategy_params)
                _normalize_history_snapshot(result)
                _ensure_ai_model_metadata(result)
                _prepare_result_payload(result)
                result["include_ai"] = True
                result["chat_history"] = []
                result["ai_summary"] = None
                result["ai_thinking"] = []
                result["ai_models"] = []
                result["show_ai_thoughts"] = True
                result["last_ai_answer"] = ""
                result.setdefault("params", {})
                result["params"].update(
                    {
                        "auto_mode": True,
                        "capital": strategy_params.capital,
                        "benchmark": strategy_params.benchmark_ticker,
                        "short_window": config["short_window"],
                        "long_window": config["long_window"],
                        "rsi_period": config["rsi_period"],
                        "volatility_target": config["volatility_target"],
                        "transaction_cost_bps": config["transaction_cost_bps"],
                        "slippage_bps": config["slippage_bps"],
                        "min_holding_days": config["min_holding_days"],
                        "label_style": config["label_style"],
                        "tb_up": config["tb_up"],
                        "tb_down": config["tb_down"],
                        "tb_max_holding": config["tb_max_holding"],
                        "strategy_engine": config["strategy_engine"],
                        "ml_model": config["ml_model"],
                        "ml_task": config["ml_task"],
                    }
                )
                combined_warnings = []
                combined_warnings.extend(getattr(form, "warnings", []))
                combined_warnings.extend(result.get("warnings", []))
                result["warnings"] = combined_warnings
                history_snapshot = sanitize_snapshot(result)
                try:
                    user_id = str(request.user.id) if request.user.is_authenticated else None
                    history_payload = dict(result)
                    history_payload["snapshot"] = history_snapshot
                    safe_payload = sanitize_snapshot(history_payload)
                    append_history(BacktestRecord.from_payload(safe_payload, user_id=user_id))
                except Exception:
                    pass
                result_json = json.dumps(history_snapshot, ensure_ascii=False, default=str)
                request.session["last_result"] = result_json
            except QuantStrategyError as exc:
                form.add_error(None, translate_text(str(exc), language))
                result = None
                result_json = "{}"
        else:
            result = None
            result_json = "{}"
    else:
        initial_values = default_initial.copy()
        ticker_query = (request.GET.get("ticker") or "").strip()
        if ticker_query:
            initial_values["ticker"] = ticker_query.upper()
        form = QuantStrategyForm(initial=initial_values, language=language)

        history_id = request.GET.get("history_id")
        if history_id:
            user_id = str(request.user.id) if request.user.is_authenticated else None
            record = get_history_record(history_id, user_id=user_id)
            if record and record.get("snapshot"):
                snapshot = record["snapshot"]
                _normalize_history_snapshot(snapshot)
                result = snapshot
                _ensure_ai_model_metadata(result)
                _prepare_result_payload(result)
                result["from_history"] = True
                result["history_record_id"] = record.get("record_id")
                result["last_ai_answer"] = snapshot.get("last_ai_answer", "")
                result_json = json.dumps(snapshot, ensure_ascii=False, default=str)
                params = record.get("params") or {}
                initial_values = default_initial.copy()
                initial_values.update(
                    {
                        "ticker": params.get("ticker") or snapshot.get("ticker"),
                        "benchmark_ticker": params.get("benchmark") or snapshot.get("benchmark_ticker") or default_initial["benchmark_ticker"],
                        "capital": params.get("capital") or snapshot.get("capital") or default_initial["capital"],
                    }
                )
                start_val = params.get("start_date")
                end_val = params.get("end_date")
                try:
                    if start_val:
                        initial_values["start_date"] = date.fromisoformat(start_val[:10])
                    if end_val:
                        initial_values["end_date"] = date.fromisoformat(end_val[:10])
                except ValueError:
                    pass
                form = QuantStrategyForm(initial=initial_values, language=getattr(request, "LANGUAGE_CODE", None))
                history_message = (
                    f"已载入历史回测：{record.get('ticker')}（{record.get('timestamp')}）"
                    if lang_is_zh
                    else f"Loaded historical backtest: {record.get('ticker')} ({record.get('timestamp')})"
                )
                request.session["last_result"] = result_json
            else:
                history_error = "未找到该历史记录或缺少快照。" if lang_is_zh else "The selected history record is missing or has no snapshot."

    history_runs = load_history(user_id=str(request.user.id)) if request.user.is_authenticated else []
    history_briefs: list[dict[str, Any]] = []
    history_tags: set[str] = set()
    for entry in history_runs:
        try:
            snapshot_payload = entry.get("snapshot") or entry
            entry["json_payload"] = json.dumps(snapshot_payload, ensure_ascii=False, indent=2)
            best = top_runs(entry.get("ticker", ""))
            entry["top_runs"] = best
        except (TypeError, ValueError):
            entry["json_payload"] = "{}"
        entry["warnings_localized"] = translate_list(entry.get("warnings") or [], language)
        for tag in entry.get("tags") or []:
            if isinstance(tag, str) and tag.strip():
                history_tags.add(tag.strip())
        ticker_label = entry.get("ticker", "Strategy")
        engine_label = entry.get("engine", "")
        default_label = f"{ticker_label} · {engine_label}"
        display_label = entry.get("title") or default_label
        history_briefs.append(
            {
                "record_id": entry.get("record_id"),
                "label": display_label,
                "ticker": entry.get("ticker"),
                "engine": entry.get("engine"),
                "period": f"{entry.get('start_date', '--')} → {entry.get('end_date', '--')}",
                "stats": entry.get("stats") or {},
            }
        )

    refresh_news = request.method == "POST" or request.GET.get("refresh_news") == "1"
    focus_feed = get_global_headlines(
        refresh=refresh_news,
        user_id=str(request.user.id) if request.user.is_authenticated else None,
    )
    for story in focus_feed:
        if "readers" not in story:
            story["readers"] = story.pop("heat", 0)
    seen_urls = {item.get("url") for item in focus_feed if item.get("url")}
    if result:
        market_focus = (result.get("market_context") or {}).get("focus_news") or []
        for story in market_focus:
            url = story.get("url")
            if url and url in seen_urls:
                continue
            merged = dict(story)
            basis = merged.get("id") or url or merged.get("title") or ""
            if basis:
                merged["id"] = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
            if not merged.get("image"):
                merged["image"] = ""
            if "readers" not in merged:
                merged["readers"] = merged.pop("heat", 0)
            focus_feed.append(merged)
            if url:
                seen_urls.add(url)

    focus_feed.sort(key=lambda item: item.get("readers", item.get("heat", 0)), reverse=True)

    result_warnings = translate_list(result.get("warnings", []) if result else [], language)
    form_warnings = getattr(form, "warnings", [])
    history_delete_template = request.build_absolute_uri(
        reverse("trading:delete_history", kwargs={"record_id": "placeholder"})
    )
    history_load_url = request.build_absolute_uri(reverse("trading:backtest"))
    history_compare_url = reverse("trading:history_compare") if request.user.is_authenticated else ""

    return render(
        request,
        "trading/backtest.html",
        {
            "form": form,
            "result": result,
            "warnings": form_warnings,
            "result_warnings": result_warnings,
            "result_json": result_json,
            "history_runs": history_runs,
            "history_message": history_message,
            "history_error": history_error,
            "focus_feed": focus_feed,
            "focus_placeholder": FOCUS_PLACEHOLDER_DATA,
            "ai_fetch_timeout_ms": getattr(settings, "AI_CHAT_FETCH_TIMEOUT_MS", getattr(settings, "AI_CHAT_TIMEOUT_SECONDS", 120) * 1000),
            "history_delete_template": history_delete_template,
            "history_load_url": history_load_url,
            "history_compare_url": history_compare_url,
            "history_briefs": history_briefs,
            "history_tags": sorted(history_tags),
            "advanced_defaults": ADVANCED_STRATEGY_DEFAULTS,
        },
    )
