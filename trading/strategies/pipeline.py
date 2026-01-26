"""
Pipeline facade module.

为后续继续拆分 pipeline 逻辑做准备，当前仅从 core 透出关键 API，
以便后续逐步迁移实现而不影响外部调用。
"""

from .core import (
    RISK_PROFILE_LABELS,
    _ensure_global_seed,
    _load_latest_walk_forward_report,
    extract_context_features,
    fetch_price_data,
    fetch_remote_strategy_overrides,
    run_rl_policy_backtest,
)
from .insights import (
    generate_recommendations,
    build_related_portfolios,
    build_statistical_baselines,
    build_multimodal_bundle,
    run_deep_signal_model,
    build_model_ensemble_view,
    analyze_factor_effectiveness,
    build_knowledge_graph_bundle,
    build_factor_scorecard,
    compute_model_weights,
    build_risk_dashboard,
    build_mlops_report,
    build_scenario_simulation,
    build_opportunity_radar,
    summarize_macro_highlight,
    build_executive_briefing,
    build_user_questions,
    build_advisor_playbook,
    build_flagship_research_bundle,
    build_key_takeaways,
    build_user_guidance,
)
from .market import fetch_market_context
from .metrics import build_core_metrics, build_metric, format_percentage
from .risk import calculate_max_drawdown
from .config import StrategyOutcome, StrategyInput, QuantStrategyError, DEFAULT_STRATEGY_SEED
from .indicators import compute_indicators
from .ma_cross import backtest_sma_strategy, format_table
from .mean_reversion import backtest_mean_reversion_strategy
from .ml_engine import (
    build_feature_matrix,
    load_best_ml_config,
    run_ml_backtest,
)
from ..optimization import PurgedWalkForwardSplit
from .charts import build_interactive_chart_payload, generate_charts
from ..portfolio import (
    apply_sector_caps,
    build_trade_list,
    cap_turnover,
)
from .metrics import (
    calculate_avg_gain_loss,
    calculate_beta,
    calculate_cagr,
    calculate_calmar,
    calculate_holding_periods,
    calculate_hit_ratio,
    calculate_sharpe,
    calculate_sortino,
    calculate_var_cvar,
    compute_validation_metrics,
    aggregate_oos_metrics,
    get_risk_free_rate_annual,
)
from ..preprocessing import sanitize_price_history
from ..data_sources import collect_auxiliary_data
from ..observability import record_metric
from ..cache_utils import build_cache_key, cache_get_object, cache_set_object
from ..reinforcement import build_reinforcement_playbook
from ..risk_stats import (
    calculate_cvar,
    compute_robust_sharpe,
    compute_spa_pvalue,
    compute_white_reality_check,
    compute_white_reality_check_bootstrap,
    recovery_period_days,
)
from ..validation import (
    compute_tail_risk_summary,
    build_walk_forward_report,
    build_purged_kfold_schedule,
    collect_repro_metadata,
    build_data_signature,
)
from ..security import sanitize_html_fragment
from django.utils.safestring import mark_safe
from sklearn.metrics import roc_auc_score
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from typing import Any, Optional
from django.utils.translation import gettext_lazy as _
from django.conf import settings

__all__ = [
    "run_quant_pipeline",
    "summarize_backtest",
    "combine_strategy_outcomes",
    "_run_quant_pipeline_inner",
    "_compute_oos_from_backtest",
]


def _run_quant_pipeline_inner(params: StrategyInput) -> dict[str, Any]:
    """Execute the end-to-end workflow and return context for rendering."""
    if params.start_date >= params.end_date:
        raise QuantStrategyError("Start date must be earlier than end date.")
    try:
        _ensure_global_seed(int(getattr(params, "random_seed", DEFAULT_STRATEGY_SEED)))
    except Exception:
        _ensure_global_seed(DEFAULT_STRATEGY_SEED)
    warnings: list[str] = []
    if params.strategy_engine in {"ml_momentum", "multi_combo", "rl_policy"} and params.auto_apply_best_config:
        engine, mlp = load_best_ml_config(params.ticker)
        if engine:
            try:
                params = replace(params, ml_model=engine, ml_params=mlp or params.ml_params)
                warnings.append(f"已根据训练缓存自动应用最优引擎：{engine}。可在表单取消自动应用或手动覆盖参数。")
            except Exception:
                pass
    prices, fetch_warnings = fetch_price_data(
        params.ticker,
        params.start_date,
        params.end_date,
        user_id=params.user_id,
    )
    warnings.extend(fetch_warnings)
    prices, quality_report = sanitize_price_history(prices)
    warnings.extend(quality_report.notes)
    min_required = max(params.long_window + params.rsi_period, params.long_window * 3, params.train_window + params.test_window, 200)
    if prices.shape[0] < min_required:
        buffer_days = max(params.long_window * 3, 365)
        extended_start = params.start_date - timedelta(days=buffer_days)
        warnings.append(f"原始区间内数据不足，已自动向前扩展至 {extended_start.isoformat()} 以满足指标计算所需的历史长度。")
        prices, extended_warnings = fetch_price_data(
            params.ticker,
            extended_start,
            params.end_date,
            user_id=params.user_id,
        )
        warnings.extend(extended_warnings)
        prices, extended_report = sanitize_price_history(prices)
        warnings.extend(extended_report.notes)
    prices = compute_indicators(prices, params.short_window, params.long_window, params.rsi_period)
    if prices.empty:
        raise QuantStrategyError("可用数据不足以计算指标，请尝试延长回测窗口或缩短均线周期。")
    market_context = fetch_market_context(params)
    auxiliary = collect_auxiliary_data(params, market_context or {}, user_id=params.user_id)
    context_features = extract_context_features(auxiliary)
    remote_overrides = fetch_remote_strategy_overrides(params)
    if remote_overrides.get("note"):
        warnings.append(str(remote_overrides["note"]))
    combo_details: list[dict[str, Any]] = []
    component_outcomes: list[StrategyOutcome] = []
    ensemble_weights: dict[str, float] = {}
    ml_context: tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]] | None = None
    if params.strategy_engine in {"ml_momentum", "rl_policy", "multi_combo"}:
        ml_context = run_ml_backtest(prices, params, context_features)
        ml_backtest, ml_metrics, ml_stats = ml_context
        warnings.extend(ml_stats.pop("runtime_warnings", []))
        ml_outcome = StrategyOutcome("机器学习动量", ml_backtest, ml_metrics, ml_stats)
        if params.strategy_engine == "ml_momentum":
            component_outcomes.append(ml_outcome)
            backtest, metrics, stats = ml_backtest, ml_metrics, ml_stats
        elif params.strategy_engine == "rl_policy":
            rl_backtest, rl_metrics, rl_stats = run_rl_policy_backtest(prices, params, ml_context, context_features)
            component_outcomes.append(StrategyOutcome("强化学习策略", rl_backtest, rl_metrics, rl_stats))
            backtest, metrics, stats = rl_backtest, rl_metrics, rl_stats
        else:
            component_outcomes.append(ml_outcome)
            try:
                sma_backtest, sma_metrics, sma_stats = backtest_sma_strategy(
                    prices, params, summarize_backtest_fn=summarize_backtest, compute_oos_report=_compute_oos_from_backtest
                )
                warnings.extend(sma_stats.pop("runtime_warnings", []))
                component_outcomes.append(StrategyOutcome("传统双均线", sma_backtest, sma_metrics, sma_stats))
            except QuantStrategyError as exc:
                warnings.append(f"组合策略中的双均线部分计算失败：{exc}")
            try:
                rl_backtest, rl_metrics, rl_stats = run_rl_policy_backtest(prices, params, ml_context, context_features)
                component_outcomes.append(StrategyOutcome("强化学习策略", rl_backtest, rl_metrics, rl_stats))
            except QuantStrategyError as exc:
                warnings.append(f"强化学习策略生成失败：{exc}")
            combined_outcome, ensemble_weights = combine_strategy_outcomes(component_outcomes, params, overrides=remote_overrides)
            backtest, metrics, stats = combined_outcome.backtest, combined_outcome.metrics, combined_outcome.stats
            combo_details = [
                {
                    "engine": outcome.engine,
                    "backtest": outcome.backtest,
                    "metrics": outcome.metrics,
                    "stats": outcome.stats,
                    "weight": ensemble_weights.get(outcome.engine, outcome.weight),
                }
                for outcome in component_outcomes
            ]
    else:
        if params.strategy_engine == "mean_reversion":
            backtest, metrics, stats = backtest_mean_reversion_strategy(
                prices, params, summarize_backtest_fn=summarize_backtest, compute_oos_report=_compute_oos_from_backtest
            )
            component_outcomes.append(StrategyOutcome("RSI 均值回归", backtest, metrics, stats))
        else:
            backtest, metrics, stats = backtest_sma_strategy(
                prices, params, summarize_backtest_fn=summarize_backtest, compute_oos_report=_compute_oos_from_backtest
            )
            component_outcomes.append(StrategyOutcome("传统双均线", backtest, metrics, stats))
    warnings.extend(stats.pop("runtime_warnings", []))
    if not ensemble_weights and component_outcomes:
        ensemble_weights = {component_outcomes[0].engine: 1.0}
    benchmark_metrics: list[dict[str, str]] = []
    benchmark_stats: dict[str, float] | None = None
    benchmark_series: pd.DataFrame | None = None
    benchmark_label = ""
    if params.benchmark_ticker:
        benchmark_label = params.benchmark_ticker.upper()
        benchmark_prices, bench_warnings = fetch_price_data(
            params.benchmark_ticker,
            params.start_date,
            params.end_date,
            user_id=params.user_id,
        )
        warnings.extend(bench_warnings)
        benchmark_prices, bench_quality = sanitize_price_history(benchmark_prices)
        warnings.extend(bench_quality.notes)
        if benchmark_prices.empty:
            warnings.append(f"未能获取基准 {benchmark_label} 的行情数据，已跳过对比分析。")
        else:
            benchmark_returns = benchmark_prices["adj close"].pct_change().fillna(0)
            combined = backtest[["strategy_return"]].join(benchmark_returns.rename("benchmark_return"), how="inner").dropna()
            if combined.empty:
                warnings.append("基准与策略的交易日无交集，基准对比已跳过。")
            else:
                combined["benchmark_cum"] = (1 + combined["benchmark_return"]).cumprod()
                annual_factor = stats["annual_factor"]
                benchmark_total_return = combined["benchmark_cum"].iloc[-1] - 1
                benchmark_vol = combined["benchmark_return"].std() * np.sqrt(annual_factor)
                rf = get_risk_free_rate_annual()
                benchmark_sharpe = calculate_sharpe(combined["benchmark_return"], trading_days=annual_factor, risk_free_rate=rf)
                correlation = combined["strategy_return"].corr(combined["benchmark_return"])
                beta = calculate_beta(combined["strategy_return"], combined["benchmark_return"])
                strat_excess_daily = combined["strategy_return"].mean() - rf / annual_factor
                bench_excess_daily = combined["benchmark_return"].mean() - rf / annual_factor
                alpha = annual_factor * (strat_excess_daily - beta * bench_excess_daily)
                relative = combined["strategy_return"] - combined["benchmark_return"]
                tracking_error = relative.std() * np.sqrt(annual_factor)
                info_ratio = relative.mean() * annual_factor / tracking_error if tracking_error != 0 else 0.0
                benchmark_metrics = [
                    build_metric("基准累计收益率", format_percentage(benchmark_total_return)),
                    build_metric("基准年化波动率", format_percentage(benchmark_vol)),
                    build_metric("基准夏普比率", f"{benchmark_sharpe:.2f}"),
                    build_metric("策略相对基准α", format_percentage(alpha)),
                    build_metric("β系数", f"{beta:.2f}"),
                    build_metric("与基准相关系数", f"{correlation:.2f}"),
                    build_metric("信息比率", f"{info_ratio:.2f}"),
                    build_metric("跟踪误差", format_percentage(tracking_error)),
                ]
                benchmark_stats = {
                    "total_return": benchmark_total_return,
                    "volatility": benchmark_vol,
                    "sharpe": benchmark_sharpe,
                    "alpha": alpha,
                    "beta": beta,
                    "correlation": correlation,
                    "info_ratio": info_ratio,
                    "tracking_error": tracking_error,
                }
                benchmark_series = combined
    charts = generate_charts(prices, backtest, benchmark_series, params) if params.include_plots else []
    if params.include_plots and stats.get("shap_img"):
        charts.append({"title": "特征重要性（SHAP）", "img": stats.get("shap_img")})

    def _safe_call(label: str, default: Any, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            warnings.append(f"{label} 生成失败：{exc}")
            return default

    recommendations = _safe_call("投资建议", [], generate_recommendations, stats, benchmark_stats, params, market_context)
    related_portfolios = _safe_call("相关组合", [], build_related_portfolios, params, market_context, params.capital)
    key_takeaways = _safe_call("关键结论", [], build_key_takeaways, stats, benchmark_stats, params)
    user_guidance = _safe_call("用户指引", {}, build_user_guidance, stats, benchmark_stats, params)
    advanced_research = _safe_call(
        "旗舰研究摘要",
        {},
        build_flagship_research_bundle,
        params=params,
        prices=prices,
        backtest=backtest,
        stats=stats,
        benchmark_stats=benchmark_stats,
        market_context=market_context or {},
        combo_details=combo_details,
    )
    try:
        feature_dataset_for_analysis, feature_columns_for_analysis = build_feature_matrix(prices, params)
    except Exception:
        feature_dataset_for_analysis, feature_columns_for_analysis = None, []
    statistical_bundle = _safe_call(
        "统计基线",
        {"arima": None, "var": None, "diagnostics": []},
        build_statistical_baselines,
        prices,
        params,
    )
    deep_signal_bundle = None
    if feature_dataset_for_analysis is not None and feature_columns_for_analysis:
        deep_signal_bundle = _safe_call(
            "深度信号",
            None,
            run_deep_signal_model,
            feature_dataset_for_analysis,
            feature_columns_for_analysis,
        )
    multimodal_bundle = _safe_call(
        "多模态摘要",
        {},
        build_multimodal_bundle,
        params,
        feature_dataset_for_analysis,
        market_context,
        fundamentals_override=auxiliary.fundamentals,
        macro_bundle=auxiliary.macro,
    )
    knowledge_bundle = _safe_call(
        "知识图谱",
        {"available": False, "message": "知识图谱生成失败。"},
        build_knowledge_graph_bundle,
        params,
        market_context,
        feature_dataset_for_analysis,
    )
    factor_scorecard = _safe_call(
        "因子得分",
        {"available": False, "message": "因子得分生成失败。"},
        build_factor_scorecard,
        prices,
        feature_dataset_for_analysis,
        auxiliary.fundamentals,
    )
    factor_effectiveness = _safe_call(
        "因子表现",
        {"available": False, "message": "因子表现分析失败。"},
        analyze_factor_effectiveness,
        feature_dataset_for_analysis,
        feature_columns_for_analysis,
    )
    risk_dashboard = _safe_call("风险看板", {}, build_risk_dashboard, stats, benchmark_stats)
    mlops_report = _safe_call("MLOps 报告", {}, build_mlops_report, params, stats)
    macro_highlight = _safe_call("宏观摘要", {}, summarize_macro_highlight, auxiliary.macro)
    scenario_simulation = _safe_call("情景模拟", {}, build_scenario_simulation, backtest, stats)
    opportunity_radar = _safe_call(
        "机会雷达",
        {},
        build_opportunity_radar,
        params,
        factor_effectiveness,
        knowledge_bundle,
    )
    ensemble_bundle = _safe_call(
        "模型集成视图",
        {},
        build_model_ensemble_view,
        statistical_bundle,
        stats,
        deep_signal_bundle,
        graph_bundle=knowledge_bundle,
        factor_bundle=factor_effectiveness,
    )
    model_weights = _safe_call(
        "模型权重",
        {},
        compute_model_weights,
        statistical_bundle,
        stats,
        deep_signal_bundle,
        knowledge_bundle,
        factor_effectiveness,
    )
    executive_briefing = _safe_call(
        "高管简报",
        {},
        build_executive_briefing,
        params,
        ensemble_bundle,
        model_weights,
        risk_dashboard,
        knowledge_bundle,
        factor_effectiveness,
        multimodal_bundle,
        deep_signal_bundle,
        scenario_simulation,
        opportunity_radar,
    )
    rl_playbook = _safe_call(
        "强化学习策略摘要",
        {},
        build_reinforcement_playbook,
        backtest,
        (params.transaction_cost_bps + params.slippage_bps) / 10000.0,
    )
    user_questions = _safe_call(
        "用户问题",
        [],
        build_user_questions,
        stats,
        recommendations,
        risk_dashboard,
        model_weights,
        ensemble_bundle,
        scenario_simulation,
        opportunity_radar,
    )
    advisor_playbook = _safe_call(
        "顾问行动手册",
        {},
        build_advisor_playbook,
        stats,
        user_guidance,
        recommendations,
        scenario_simulation,
        risk_dashboard,
        opportunity_radar,
        macro_highlight,
    )
    combo_results: list[dict[str, Any]] = []
    if params.strategy_engine == "multi_combo" and combo_details:
        for idx, entry in enumerate(combo_details):
            stats_item = entry["stats"]
            metrics_item = entry["metrics"]
            guidance_item = user_guidance if idx == 0 else build_user_guidance(stats_item, None, params)
            engine_name = entry["engine"] + ("（主）" if idx == 0 else "")
            combo_results.append(
                {
                    "engine": engine_name,
                    "metrics": metrics_item,
                    "stats": {
                        "total_return": stats_item.get("total_return"),
                        "cagr": stats_item.get("cagr"),
                        "sharpe": stats_item.get("sharpe"),
                        "max_drawdown": stats_item.get("max_drawdown"),
                        "volatility": stats_item.get("volatility"),
                        "annual_turnover": stats_item.get("annual_turnover"),
                        "average_holding_days": stats_item.get("average_holding_days"),
                        "optionality": stats_item.get("optionality"),
                    },
                    "weight": entry.get("weight"),
                    "quick_summary": guidance_item.get("quick_summary", []),
                    "confidence_label": guidance_item.get("confidence_label"),
                    "confidence_score": guidance_item.get("confidence_score"),
                    "action_plan": guidance_item.get("action_plan", []),
                    "risk_alerts": guidance_item.get("risk_alerts", []),
                }
            )
    remote_meta = {k: remote_overrides.get(k) for k in ("source", "version", "timestamp") if remote_overrides.get(k) is not None}
    if params.strategy_engine == "ml_momentum":
        engine_label = _("机器学习动量 + 风险控制")
    elif params.strategy_engine == "multi_combo":
        engine_label = _("组合策略（主策略：机器学习动量）")
    elif params.strategy_engine == "mean_reversion":
        engine_label = _("RSI 均值回归")
    elif params.strategy_engine == "rl_policy":
        engine_label = _("强化学习策略")
    else:
        engine_label = _("双均线动量框架")
    walk_forward_report = build_walk_forward_report(backtest.get("strategy_return"))
    purged_schedule = build_purged_kfold_schedule(
        feature_dataset_for_analysis.index if isinstance(feature_dataset_for_analysis, pd.DataFrame) else None,
        n_splits=max(2, params.validation_slices or 3),
        embargo=getattr(params, "embargo_days", 5),
    )
    stats["validation_report"] = {"walk_forward": walk_forward_report, "purged_kfold": purged_schedule}
    stats["tail_risk_summary"] = compute_tail_risk_summary(backtest.get("strategy_return"))
    metadata = collect_repro_metadata(params)
    metadata["requested_start"] = params.start_date.isoformat()
    metadata["requested_end"] = params.end_date.isoformat()
    if not prices.empty:
        meta_start = prices.index[0]
        meta_end = prices.index[-1]
        metadata["effective_start"] = str(meta_start.date()) if hasattr(meta_start, "date") else str(meta_start)
        metadata["effective_end"] = str(meta_end.date()) if hasattr(meta_end, "date") else str(meta_end)
    data_signature = build_data_signature(
        prices,
        columns=["adj close", "open", "high", "low", "close", "volume"],
    )
    if data_signature:
        metadata["data_signature"] = data_signature
        if data_signature.get("source"):
            metadata["data_source"] = data_signature.get("source")
    fetch_note = getattr(prices, "attrs", {}).get("data_fetch_note")
    if fetch_note:
        metadata["data_fetch_note"] = fetch_note
    data_quality = quality_report.to_dict() if quality_report else {}
    data_risks: list[str] = []
    if auxiliary.financials or auxiliary.fundamentals:
        data_risks.append("基本面/财务数据来自第三方快照，未做 point-in-time 校验，存在未来信息风险。")
    if data_quality.get("zero_volume_days"):
        data_risks.append("历史数据存在零成交量交易日，回测执行已自动跳过停牌日。")
    if data_quality.get("missing_ratio", 0.0) > 0.05:
        data_risks.append("行情数据缺失比例偏高，已补全但可能影响指标稳定性。")
    risk_controls = {
        "volatility_target": {"target": params.volatility_target, "realized": stats.get("volatility")},
        "tail_risk": stats.get("tail_risk_summary"),
        "max_drawdown_stop": params.max_drawdown_stop,
        "daily_exposure_limit": params.daily_exposure_limit,
    }
    if data_quality:
        stats["data_quality"] = data_quality
    if data_risks:
        stats["data_risks"] = data_risks
    reliability = _build_reliability(stats, metadata)
    if reliability:
        stats["reliability"] = reliability
    label_meta = {
        "label_style": params.label_style,
        "tb_up": params.tb_up,
        "tb_down": params.tb_down,
        "tb_dynamic": params.tb_dynamic,
        "tb_vol_multiplier": params.tb_vol_multiplier,
        "tb_vol_window": params.tb_vol_window,
        "tb_max_holding": params.tb_max_holding,
        "return_path": params.return_path,
    }
    if stats.get("tb_dynamic_summary"):
        label_meta["tb_dynamic_summary"] = stats.get("tb_dynamic_summary")
    if hasattr(params, "tb_up_applied") or hasattr(params, "tb_down_applied"):
        label_meta["tb_applied"] = {"up": getattr(params, "tb_up_applied", params.tb_up), "down": getattr(params, "tb_down_applied", params.tb_down)}
    walk_forward_training: dict[str, Any] | None = None
    if getattr(params, "include_walk_forward_report", False):
        walk_forward_training = _load_latest_walk_forward_report(params)
        if walk_forward_training:
            stats["walk_forward_training"] = walk_forward_training
        else:
            warnings.append("未找到 walk-forward 报告，请运行 trading.mlops.walk_forward_train 生成最新报告。")

    signal_snapshot: dict[str, Any] = {}
    target_weights: dict[str, float] = {}
    current_positions = params.current_positions or {}
    current_weights: dict[str, float] = {}
    trade_list: list[dict[str, Any]] = []
    if not backtest.empty:
        latest = backtest.iloc[-1]
        last_price = float(latest.get("adj close", np.nan)) if isinstance(latest, pd.Series) else np.nan
        signal_snapshot = {
            "as_of": str(backtest.index[-1].date()) if len(backtest.index) else "",
            "signal": float(latest.get("signal", 0.0)) if isinstance(latest, pd.Series) else 0.0,
            "position": float(latest.get("position", 0.0)) if isinstance(latest, pd.Series) else 0.0,
            "exposure": float(latest.get("exposure", latest.get("position", 0.0))) if isinstance(latest, pd.Series) else 0.0,
            "probability": float(latest.get("probability", np.nan)) if isinstance(latest, pd.Series) else np.nan,
            "price": None if np.isnan(last_price) else round(last_price, 4),
        }
        raw_weight = float(latest.get("exposure", latest.get("position", 0.0))) if isinstance(latest, pd.Series) else 0.0
        if not params.allow_short and raw_weight < 0:
            raw_weight = 0.0
        if params.max_weight is not None:
            raw_weight = float(np.clip(raw_weight, -abs(params.max_weight), abs(params.max_weight)))
        if params.min_weight is not None and params.min_weight > 0 and abs(raw_weight) < params.min_weight:
            raw_weight = 0.0
        target_weights = {params.ticker.upper(): raw_weight}
        if current_positions:
            capital = max(float(params.capital or 0.0), 1.0)
            if last_price and last_price > 0:
                current_value = float(current_positions.get(params.ticker.upper(), 0.0)) * last_price
                current_weights = {params.ticker.upper(): current_value / capital}
        if params.turnover_cap:
            target_weights = cap_turnover(current_weights, target_weights, params.turnover_cap)
        if params.sector_caps:
            sector_name = auxiliary.fundamentals.get("sector") if isinstance(auxiliary.fundamentals, dict) else None
            sector_map = {params.ticker.upper(): sector_name} if sector_name else {}
            target_weights = apply_sector_caps(target_weights, sector_map, params.sector_caps)
        if last_price and last_price > 0:
            trade_list = build_trade_list(
                target_weights,
                current_positions,
                {params.ticker.upper(): last_price},
                capital=float(params.capital or 0.0),
                lot_size=int(params.lot_size or 1),
            )
    target_portfolio = {
        "target_weights": target_weights,
        "current_weights": current_weights,
        "gross_exposure": float(sum(abs(v) for v in target_weights.values())) if target_weights else 0.0,
        "net_exposure": float(sum(target_weights.values())) if target_weights else 0.0,
        "turnover": float(sum(abs(target_weights.get(k, 0.0) - current_weights.get(k, 0.0)) for k in set(target_weights) | set(current_weights))),
    }
    interactive_chart = build_interactive_chart_payload(prices, backtest, params)
    return_series = _build_return_series(backtest)
    overview_series = _build_overview_series(backtest, params, stats)
    result_payload = {
        "ticker": params.ticker.upper(),
        "start_date": params.start_date.strftime("%Y-%m-%d"),
        "end_date": params.end_date.strftime("%Y-%m-%d"),
        "metrics": metrics,
        "benchmark_ticker": benchmark_label if benchmark_metrics else "",
        "benchmark_metrics": benchmark_metrics,
        "recent_rows": format_table(backtest),
        "return_series": return_series,
        "overview_series": overview_series,
        "warnings": warnings,
        "stats": stats,
        "benchmark_stats": benchmark_stats,
        "charts": charts,
        "interactive_chart": interactive_chart,
        "recommendations": recommendations,
        "related_portfolios": related_portfolios,
        "key_takeaways": key_takeaways,
        "market_context": market_context,
        "risk_profile": RISK_PROFILE_LABELS.get(params.risk_profile, params.risk_profile),
        "capital": params.capital,
        "engine": params.strategy_engine,
        "engine_label": engine_label,
        "data_quality": data_quality,
        "data_risks": data_risks,
        "signal_snapshot": signal_snapshot,
        "target_portfolio": target_portfolio,
        "trade_list": trade_list,
        "params": {
            "ticker": params.ticker.upper(),
            "benchmark": params.benchmark_ticker,
            "start_date": params.start_date.isoformat(),
            "end_date": params.end_date.isoformat(),
            "short_window": params.short_window,
            "long_window": params.long_window,
            "rsi_period": params.rsi_period,
            "volatility_target": params.volatility_target,
            "transaction_cost_bps": params.transaction_cost_bps,
            "slippage_bps": params.slippage_bps,
            "min_holding_days": params.min_holding_days,
            "strategy_engine": params.strategy_engine,
            "risk_profile": params.risk_profile,
            "capital": params.capital,
            "investment_horizon": params.investment_horizon,
            "experience_level": params.experience_level,
            "primary_goal": params.primary_goal,
            "interest_keywords": params.interest_keywords,
            "max_drawdown_stop": params.max_drawdown_stop,
            "daily_exposure_limit": params.daily_exposure_limit,
            "rl_engine": params.rl_engine,
            "validation_slices": params.validation_slices,
            "out_of_sample_ratio": params.out_of_sample_ratio,
            "execution_liquidity_buffer": params.execution_liquidity_buffer,
            "execution_penalty_bps": params.execution_penalty_bps,
            "execution_delay_days": getattr(params, "execution_delay_days", 1),
            "return_path": params.return_path,
            "slippage_model": params.slippage_model,
            "borrow_cost_bps": params.borrow_cost_bps,
            "long_borrow_cost_bps": params.long_borrow_cost_bps,
            "short_borrow_cost_bps": params.short_borrow_cost_bps,
            "max_adv_participation": params.max_adv_participation,
            "class_weight_mode": params.class_weight_mode,
            "focal_gamma": params.focal_gamma,
            "tb_dynamic": params.tb_dynamic,
            "tb_vol_multiplier": params.tb_vol_multiplier,
            "tb_vol_window": params.tb_vol_window,
            "lot_size": params.lot_size,
            "max_weight": params.max_weight,
            "min_weight": params.min_weight,
            "max_holdings": params.max_holdings,
            "sector_caps": params.sector_caps,
            "turnover_cap": params.turnover_cap,
            "allow_short": params.allow_short,
            "limit_move_threshold": params.limit_move_threshold,
            "include_walk_forward_report": params.include_walk_forward_report,
            "walk_forward_horizon_days": params.walk_forward_horizon_days,
            "walk_forward_step_days": params.walk_forward_step_days,
            "walk_forward_jobs": params.walk_forward_jobs,
            "validation_summary": stats.get("validation_summary_compact"),
            "threshold_scan_summary": stats.get("threshold_scan_summary"),
        },
        "guidance": user_guidance,
        "combo_results": combo_results,
        "include_plots": params.include_plots,
        "show_ai_thoughts": params.show_ai_thoughts,
        "advanced_research": advanced_research,
        "statistical_bundle": statistical_bundle,
        "deep_signal_bundle": deep_signal_bundle,
        "multimodal_bundle": multimodal_bundle,
        "ensemble_bundle": ensemble_bundle,
        "knowledge_bundle": knowledge_bundle,
        "factor_scorecard": factor_scorecard,
        "macro_bundle": auxiliary.macro,
        "event_bundle": auxiliary.events,
        "financial_snapshot": auxiliary.financials,
        "capital_flows": auxiliary.capital_flows,
        "news_sentiment": auxiliary.news_sentiment,
        "options_metrics": auxiliary.options_metrics,
        "global_macro_context": auxiliary.global_macro,
        "factor_effectiveness": factor_effectiveness,
        "model_weights": model_weights,
        "risk_dashboard": risk_dashboard,
        "mlops_report": mlops_report,
        "executive_briefing": executive_briefing,
        "user_questions": user_questions,
        "macro_highlight": macro_highlight,
        "scenario_simulation": scenario_simulation,
        "opportunity_radar": opportunity_radar,
        "advisor_playbook": advisor_playbook,
        "rl_playbook": rl_playbook,
        "validation_report": stats.get("validation_report"),
        "calibration": stats.get("calibration"),
        "ensemble_breakdown": {"weights": ensemble_weights, "available": params.strategy_engine == "multi_combo"},
        "remote_config": remote_meta,
        "metadata": metadata,
        "risk_controls": risk_controls,
        "return_path": params.return_path,
        "label_meta": label_meta,
        "repro": metadata,
        "walk_forward_training": walk_forward_training,
    }
    if stats.get("threshold_scan"):
        result_payload["params"]["threshold_scan"] = stats.get("threshold_scan")
    extra_risk_alerts = list(user_guidance.get("risk_alerts", []))
    if stats.get("risk_events"):
        extra_risk_alerts.extend(stats.get("risk_events", []))
        warnings.extend(stats.get("risk_events", []))
    result_payload.update(
        {
            "quick_summary": user_guidance.get("quick_summary", []),
            "action_plan": user_guidance.get("action_plan", []),
            "risk_alerts": extra_risk_alerts,
            "education_tips": user_guidance.get("education_tips", []),
            "confidence_label": user_guidance.get("confidence_label"),
            "confidence_score": user_guidance.get("confidence_score"),
            "experience_label": user_guidance.get("experience_label"),
            "investment_horizon_label": user_guidance.get("investment_horizon_label"),
            "primary_goal_label": user_guidance.get("primary_goal_label"),
            "disclaimer": user_guidance.get("disclaimer"),
        }
    )
    hyperopt_report = stats.get("hyperopt_report")
    if hyperopt_report:
        result_payload["hyperopt_report"] = hyperopt_report
    progress_plan = _build_progress_plan(params, result_payload)
    result_payload["progress_plan"] = progress_plan
    _sanitize_analysis_sections(result_payload)
    result_payload["task_feedback"] = {"remaining_steps": progress_plan["remaining"], "eta_seconds": progress_plan["eta_seconds"]}
    return result_payload


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _build_return_series(backtest: pd.DataFrame) -> list[dict[str, Any]]:
    if backtest.empty:
        return []
    returns: pd.Series | None = None
    if "strategy_return" in backtest.columns:
        returns = backtest["strategy_return"].astype(float)
    elif "daily_return" in backtest.columns:
        returns = backtest["daily_return"].astype(float)
    if returns is None:
        return []
    if "cum_strategy" in backtest.columns:
        cum_strategy = backtest["cum_strategy"].astype(float)
    else:
        cum_strategy = (1 + returns.fillna(0.0)).cumprod()
    cum_buy_hold = backtest["cum_buy_hold"].astype(float) if "cum_buy_hold" in backtest.columns else None
    rows: list[dict[str, Any]] = []
    for idx in backtest.index:
        date_value = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
        daily_return = _safe_float(returns.loc[idx])
        entry = {"date": date_value, "daily_return": daily_return}
        cum_value = _safe_float(cum_strategy.loc[idx])
        if cum_value is not None:
            entry["cum_strategy"] = cum_value
        if cum_buy_hold is not None:
            buy_hold_value = _safe_float(cum_buy_hold.loc[idx])
            if buy_hold_value is not None:
                entry["cum_buy_hold"] = buy_hold_value
        rows.append(entry)
    return rows


def _build_overview_series(
    backtest: pd.DataFrame,
    params: StrategyInput,
    stats: dict[str, Any] | None = None,
    *,
    max_points: int = 420,
) -> dict[str, Any]:
    if backtest.empty:
        return {"series": [], "events": [], "thresholds": {}}
    frame = backtest.tail(max_points).copy()
    series: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        date_value = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
        entry: dict[str, Any] = {"date": date_value}
        if "probability" in frame.columns:
            entry["probability"] = _safe_float(row.get("probability"))
        if "position" in frame.columns:
            entry["position"] = _safe_float(row.get("position"))
        elif "exposure" in frame.columns:
            entry["position"] = _safe_float(row.get("exposure"))
        if "fill_coverage" in frame.columns:
            entry["coverage"] = _safe_float(row.get("fill_coverage"))
        if "signal" in frame.columns:
            entry["signal"] = _safe_float(row.get("signal"))
        series.append(entry)

    events: list[dict[str, Any]] = []
    prev_pos = 0.0
    limit_threshold = getattr(params, "limit_move_threshold", None)
    prices = frame.get("adj close") if "adj close" in frame.columns else frame.get("close")
    price_change = None
    if prices is not None:
        try:
            price_change = prices.pct_change().abs()
        except Exception:
            price_change = None
    for idx, row in frame.iterrows():
        date_value = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
        pos_val = row.get("position") if "position" in frame.columns else row.get("exposure")
        pos = _safe_float(pos_val) or 0.0
        if prev_pos <= 0 < pos:
            events.append({"date": date_value, "type": "entry", "label": "Entry"})
        elif prev_pos >= 0 > pos:
            events.append({"date": date_value, "type": "entry", "label": "Short"})
        elif prev_pos > 0 >= pos:
            events.append({"date": date_value, "type": "exit", "label": "Exit"})
        elif prev_pos < 0 <= pos:
            events.append({"date": date_value, "type": "exit", "label": "Cover"})
        prev_pos = pos

        if "volume" in frame.columns and row.get("volume", 0) <= 0:
            events.append({"date": date_value, "type": "halt", "label": "Halt"})
        if price_change is not None and limit_threshold and limit_threshold > 0:
            try:
                change = price_change.get(idx)
                if change is not None and float(change) >= float(limit_threshold):
                    events.append({"date": date_value, "type": "limit", "label": "Limit"})
            except Exception:
                pass

    events = events[-30:]
    if stats:
        exec_stats = stats.get("execution_stats") if isinstance(stats, dict) else None
        adv_hits = None
        if isinstance(exec_stats, dict):
            adv_hits = exec_stats.get("adv_hard_cap_hits")
        if adv_hits is None and isinstance(stats, dict):
            adv_hits = stats.get("adv_hits")
        adv_hits_val = _safe_float(adv_hits)
        if adv_hits_val and series:
            events.append({"date": series[-1]["date"], "type": "adv", "label": f"ADV cap ×{int(adv_hits_val)}"})
    thresholds = {
        "entry": getattr(params, "entry_threshold", None),
        "exit": getattr(params, "exit_threshold", None),
    }
    return {"series": series, "events": events, "thresholds": thresholds}


def _clamp_score(value: Any) -> int:
    numeric = _safe_float(value)
    if numeric is None:
        return 0
    return max(0, min(100, int(round(numeric))))


def _build_reliability(stats: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    actions: list[dict[str, str]] = []

    def _penalize(score: float, amount: float, reason: str) -> float:
        if amount <= 0:
            return score
        if reason and reason not in reasons:
            reasons.append(reason)
        return score - amount

    def _parse_date_value(value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, str) and value:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None

    data_score = 100.0
    data_quality = stats.get("data_quality") if isinstance(stats, dict) else {}
    if not isinstance(data_quality, dict):
        data_quality = {}
    missing_ratio = _safe_float(data_quality.get("missing_ratio"))
    if missing_ratio is not None:
        if missing_ratio > 0.1:
            data_score = _penalize(data_score, 30, "缺失比例偏高，数据完整性不足。")
        elif missing_ratio > 0.05:
            data_score = _penalize(data_score, 18, "缺失比例偏高，建议关注数据质量。")
    zero_volume = _safe_float(data_quality.get("zero_volume_days"))
    if zero_volume is not None and zero_volume > 0:
        data_score = _penalize(data_score, min(12, zero_volume), "存在零成交量交易日，执行与信号可靠性受限。")
    stale_days = _safe_float(data_quality.get("stale_price_days"))
    if stale_days is not None and stale_days > 0:
        data_score = _penalize(data_score, min(10, stale_days / 5), "价格无变动日偏多，可能存在停牌或低流动性。")

    data_signature = metadata.get("data_signature") if isinstance(metadata, dict) else {}
    if not isinstance(data_signature, dict):
        data_signature = {}
    data_source = data_signature.get("source") or metadata.get("data_source")
    if data_source == "csv_cache":
        data_score = _penalize(data_score, 12, "已降级为本地缓存数据，线上行情不可用。")
    elif not data_source:
        data_score = _penalize(data_score, 8, "未检测到数据来源，可能影响可复现性。")

    requested_start = _parse_date_value(metadata.get("requested_start") if isinstance(metadata, dict) else None)
    requested_end = _parse_date_value(metadata.get("requested_end") if isinstance(metadata, dict) else None)
    effective_start = _parse_date_value(metadata.get("effective_start") if isinstance(metadata, dict) else None)
    effective_end = _parse_date_value(metadata.get("effective_end") if isinstance(metadata, dict) else None)
    if requested_start and effective_start and effective_start > requested_start:
        data_score = _penalize(data_score, 8, "实际数据起点晚于请求区间，样本外推风险上升。")
    if requested_end and effective_end and effective_end < requested_end:
        data_score = _penalize(data_score, 6, "实际数据终点早于请求区间，结果可能偏乐观。")

    model_score = 100.0
    model_signals = 0
    penalized_sharpe = _safe_float(stats.get("validation_penalized_sharpe")) if isinstance(stats, dict) else None
    if penalized_sharpe is not None:
        model_signals += 1
        if penalized_sharpe < 0:
            model_score = _penalize(model_score, 25, "样本外 Penalized Sharpe 低于 0，模型稳定性偏弱。")
        elif penalized_sharpe < 0.4:
            model_score = _penalize(model_score, 12, "样本外 Sharpe 较低，需谨慎看待。")
    validation_summary = stats.get("validation_summary_compact") if isinstance(stats, dict) else None
    sharpe_summary = validation_summary.get("sharpe") if isinstance(validation_summary, dict) else None
    if isinstance(sharpe_summary, dict):
        model_signals += 1
        mean_val = _safe_float(sharpe_summary.get("mean"))
        std_val = _safe_float(sharpe_summary.get("std"))
        if mean_val is not None and std_val is not None and mean_val - std_val < 0:
            model_score = _penalize(model_score, 10, "样本外 Sharpe 波动较大，稳定性不足。")
    auc = _safe_float(stats.get("auc")) if isinstance(stats, dict) else None
    if auc is not None:
        model_signals += 1
        if auc < 0.55:
            model_score = _penalize(model_score, 10, "AUC 偏低，方向预测可靠性不足。")
    calibration = stats.get("calibration") if isinstance(stats, dict) else None
    brier = _safe_float(calibration.get("brier")) if isinstance(calibration, dict) else None
    if brier is not None:
        model_signals += 1
        if brier > 0.25:
            model_score = _penalize(model_score, 8, "Brier 偏高，概率校准效果一般。")
    cpcv = stats.get("cpcv") if isinstance(stats, dict) else None
    if isinstance(cpcv, dict) and cpcv:
        model_signals += 1
        p10 = _safe_float(cpcv.get("p10_sharpe"))
        worst = _safe_float(cpcv.get("worst_sharpe"))
        if worst is not None and worst < 0:
            model_score = _penalize(model_score, 12, "CPCV 最差 Sharpe 为负，部分折表现失稳。")
        if p10 is not None and p10 < 0.2:
            model_score = _penalize(model_score, 8, "CPCV 10% 分位 Sharpe 较低，需控制风险。")
    drift = stats.get("drift") if isinstance(stats, dict) else None
    if isinstance(drift, dict):
        model_signals += 1
        psi_ret = _safe_float(drift.get("psi_returns")) or 0.0
        psi_prob = _safe_float(drift.get("psi_probabilities")) or 0.0
        psi_val = max(psi_ret, psi_prob)
        if psi_val > 0.25:
            model_score = _penalize(model_score, 10, "检测到分布漂移（PSI 偏高），建议重新训练。")
    threshold_stability = stats.get("threshold_stability") if isinstance(stats, dict) else None
    if isinstance(threshold_stability, dict):
        model_signals += 1
        worst = threshold_stability.get("worst") if isinstance(threshold_stability.get("worst"), dict) else {}
        worst_sharpe = _safe_float(worst.get("sharpe")) if isinstance(worst, dict) else None
        if worst_sharpe is not None and worst_sharpe < 0:
            model_score = _penalize(model_score, 8, "阈值敏感度较高，最差 Sharpe 为负。")

    if model_signals == 0:
        model_score = _penalize(model_score, 30, "模型验证指标不足，可信度需谨慎。")

    exec_score = 100.0
    exec_signals = 0
    exec_stats = stats.get("execution_stats") if isinstance(stats, dict) else None
    if isinstance(exec_stats, dict):
        exec_signals += 1
        avg_coverage = _safe_float(exec_stats.get("avg_coverage"))
        if avg_coverage is not None:
            if avg_coverage < 0.7:
                exec_score = _penalize(exec_score, 20, "成交覆盖率偏低，执行假设可能过乐观。")
            elif avg_coverage < 0.85:
                exec_score = _penalize(exec_score, 8, "成交覆盖率一般，建议收紧仓位。")
        unfilled = _safe_float(exec_stats.get("unfilled_ratio"))
        if unfilled is not None and unfilled > 0.25:
            exec_score = _penalize(exec_score, 10, "未成交比例偏高，需降低参与率。")
        adv_hits = _safe_float(exec_stats.get("adv_hard_cap_hits"))
        if adv_hits is not None and adv_hits > 0:
            exec_score = _penalize(exec_score, min(10, adv_hits * 0.5), "多次触发 ADV 上限，真实成交受限。")
        halt_days = _safe_float(exec_stats.get("halt_days"))
        if halt_days is not None and halt_days > 0:
            exec_score = _penalize(exec_score, 6, "存在停牌/无成交交易日。")
        limit_days = _safe_float(exec_stats.get("limit_days"))
        if limit_days is not None and limit_days > 0:
            exec_score = _penalize(exec_score, 4, "存在涨跌停交易日，成交受限。")
    cost_ratio = _safe_float(stats.get("cost_ratio")) if isinstance(stats, dict) else None
    if cost_ratio is not None:
        exec_signals += 1
        if cost_ratio > 0.12:
            exec_score = _penalize(exec_score, 10, "成本占比较高，收益可能被侵蚀。")

    if exec_signals == 0:
        exec_score = _penalize(exec_score, 25, "执行诊断指标不足，成交可信度偏低。")

    data_score = _clamp_score(data_score)
    model_score = _clamp_score(model_score)
    exec_score = _clamp_score(exec_score)
    score = _clamp_score(data_score * 0.4 + model_score * 0.4 + exec_score * 0.2)

    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 55:
        label = "Caution"
    else:
        label = "High risk"

    def _add_action(label_text: str, href: str) -> None:
        if not label_text or not href:
            return
        if any(action.get("href") == href for action in actions):
            return
        actions.append({"label": label_text, "href": href})

    if data_score < 80:
        _add_action("查看数据质量面板", "#data-quality-card")
    if exec_score < 75:
        _add_action("以更保守成本假设复跑", "#config-pane")
    if model_score < 75:
        _add_action("加入历史对比", "#history-pane")
    if not actions:
        _add_action("打开专业数据面板", "#expert-pane")
        _add_action("加入历史对比", "#history-pane")

    return {
        "score": score,
        "label": label,
        "subscores": {"data": data_score, "model": model_score, "execution": exec_score},
        "reasons": reasons[:6],
        "actions": actions[:3],
    }


def _sanitize_analysis_sections(payload: dict[str, Any]) -> None:
    global_macro = payload.get("global_macro_context")
    if isinstance(global_macro, dict):
        if global_macro.get("data"):
            global_macro["data"] = mark_safe(sanitize_html_fragment(global_macro["data"]))
        if global_macro.get("message"):
            global_macro["message"] = mark_safe(sanitize_html_fragment(global_macro["message"]))
        summary = global_macro.get("summary")
        if isinstance(summary, dict):
            for key, value in list(summary.items()):
                summary[key] = mark_safe(sanitize_html_fragment(value))
    events = payload.get("event_bundle")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("title"):
                event["title"] = sanitize_html_fragment(event["title"])
            if event.get("summary"):
                event["summary"] = sanitize_html_fragment(event["summary"])


def _build_progress_plan(params: StrategyInput, payload: dict[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [
        {"key": "stats", "label": _("核心统计"), "eta_seconds": 2, "ready": True},
        {"key": "visuals", "label": _("图表/风控面板"), "eta_seconds": 3, "ready": bool(payload.get("charts")) and bool(params.include_plots)},
        {"key": "ai_insights", "label": _("AI 结论"), "eta_seconds": 4, "ready": not params.show_ai_thoughts},
    ]
    remaining = sum(1 for step in steps if not step["ready"])
    eta_seconds = sum(step["eta_seconds"] for step in steps if not step["ready"])
    message = _("核心统计已就绪，后续将依次补充图表与 AI 解读。") if remaining else _("所有步骤已完成。")
    return {"steps": steps, "remaining": remaining, "eta_seconds": eta_seconds, "message": message}


_CACHE_EXCLUDED_FIELDS = {"request_id", "exec_latency_ms"}


def _params_cache_signature(params: StrategyInput) -> str:
    payload = asdict(params)
    for key in _CACHE_EXCLUDED_FIELDS:
        payload.pop(key, None)
    signature_blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(signature_blob.encode("utf-8")).hexdigest()


def run_quant_pipeline(params: StrategyInput) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    success = False
    error_message: str | None = None
    result_payload: dict[str, Any] | None = None
    cache_ttl = int(getattr(settings, "STRATEGY_RESULT_CACHE_TTL", 1800) or 0)
    enable_cache = cache_ttl > 0 and getattr(settings, "ENABLE_STRATEGY_RESULT_CACHE", True)
    cache_key: str | None = None
    try:
        if isinstance(params.start_date, str):
            params.start_date = datetime.fromisoformat(params.start_date).date()
        if isinstance(params.end_date, str):
            params.end_date = datetime.fromisoformat(params.end_date).date()
    except Exception:
        pass
    _ensure_global_seed(getattr(params, "random_seed", DEFAULT_STRATEGY_SEED))
    try:
        if enable_cache:
            signature = _params_cache_signature(params)
            cache_key = build_cache_key("strategy_result", params.ticker.upper(), signature)
            cached = cache_get_object(cache_key)
            if isinstance(cached, dict):
                result_payload = copy.deepcopy(cached)
                params_payload = result_payload.get("params")
                if not isinstance(params_payload, dict):
                    params_payload = {}
                    result_payload["params"] = params_payload
                if params.request_id:
                    params_payload["request_id"] = params.request_id
                if params.user_id is not None:
                    params_payload["user_id"] = params.user_id
                success = True
                record_metric("backtest.cache_hit", ticker=params.ticker.upper())
                return result_payload
        result = _run_quant_pipeline_inner(params)
        if isinstance(result, dict):
            result_payload = result
            if cache_key and enable_cache:
                cache_set_object(cache_key, result, cache_ttl)
        success = True
        return result
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - pipeline_started) * 1000.0
        try:
            params.exec_latency_ms = elapsed_ms
        except Exception:
            pass
        if result_payload is not None:
            metadata = result_payload.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["exec_latency_ms"] = round(elapsed_ms, 2)
        record_metric(
            "backtest.pipeline",
            ticker=params.ticker.upper(),
            engine=params.strategy_engine,
            user_id=params.user_id,
            request_id=params.request_id,
            duration_ms=round(elapsed_ms, 2),
            success=success,
            error=error_message,
        )


def summarize_backtest(
    backtest: pd.DataFrame,
    params: StrategyInput,
    *,
    include_prediction: bool = False,
    include_auc: bool = False,
    feature_columns: Optional[list[str]] = None,
    shap_img: Optional[str] = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if backtest.empty:
        raise QuantStrategyError("回测结果为空，无法生成统计指标。")
    annual_factor = 252
    rf = get_risk_free_rate_annual()
    net_returns = backtest.get("strategy_return")
    if net_returns is None:
        raise QuantStrategyError("回测结果缺少 strategy_return 列。")
    net_returns = net_returns.astype(float).fillna(0.0)
    asset_returns = backtest.get("asset_return")
    if asset_returns is None:
        price_series = backtest.get("adj close")
        if price_series is None:
            raise QuantStrategyError("回测结果缺少 asset_return 与 adj close 列。")
        asset_returns = price_series.astype(float).pct_change().fillna(0.0)
    else:
        asset_returns = asset_returns.astype(float).fillna(0.0)
    cum_strategy = (1 + net_returns).cumprod()
    cum_buy_hold = (1 + asset_returns).cumprod()
    total_return = float(cum_strategy.iloc[-1] - 1) if not cum_strategy.empty else 0.0
    buy_hold_return = float(cum_buy_hold.iloc[-1] - 1) if not cum_buy_hold.empty else 0.0
    max_drawdown = calculate_max_drawdown(cum_strategy)
    sharpe = calculate_sharpe(net_returns, trading_days=annual_factor, risk_free_rate=rf)
    sortino = calculate_sortino(net_returns, trading_days=annual_factor, risk_free_rate=rf)
    volatility = net_returns.std() * np.sqrt(annual_factor)
    cagr = calculate_cagr(cum_strategy, trading_days=annual_factor)
    calmar = calculate_calmar(cagr, max_drawdown)
    hit_ratio = calculate_hit_ratio(net_returns)
    avg_gain, avg_loss = calculate_avg_gain_loss(net_returns)
    position_series = backtest.get("position")
    position_series = position_series.astype(float) if position_series is not None else pd.Series(0.0, index=backtest.index)
    avg_exposure = float(position_series.abs().mean()) if not position_series.empty else 0.0
    leverage_series = backtest.get("leverage")
    leverage_series = leverage_series.astype(float) if leverage_series is not None else pd.Series(0.0, index=backtest.index)
    avg_leverage = float(leverage_series.mean()) if not leverage_series.empty else 0.0
    var_95, cvar_95 = calculate_var_cvar(net_returns, alpha=0.95)
    cumulative_curve = (1 + net_returns).cumprod()
    recovery_days = recovery_period_days(cumulative_curve)
    loss_streak = int((net_returns < 0).astype(int).groupby((net_returns >= 0).astype(int).cumsum()).sum().max()) if not net_returns.empty else 0
    es_95 = calculate_cvar(net_returns, alpha=0.95)
    exposure_series = backtest.get("exposure")
    if exposure_series is not None:
        exposure_series = exposure_series.astype(float)
        exposure_change = exposure_series.diff().abs().fillna(exposure_series.abs())
    else:
        exposure_change = position_series.diff().abs().fillna(position_series.abs())
    daily_turnover = float(exposure_change.mean()) if not exposure_change.empty else 0.0
    annual_turnover = daily_turnover * annual_factor
    transaction_cost = backtest.get("transaction_cost")
    transaction_cost = transaction_cost.astype(float).fillna(0.0) if transaction_cost is not None else pd.Series(0.0, index=backtest.index)
    execution_cost = backtest.get("execution_cost")
    execution_cost = execution_cost.astype(float).fillna(0.0) if execution_cost is not None else pd.Series(0.0, index=backtest.index)
    borrow_cost_series = backtest.get("borrow_cost")
    borrow_cost_series = borrow_cost_series.astype(float).fillna(0.0) if borrow_cost_series is not None else pd.Series(0.0, index=backtest.index)
    total_cost = float(transaction_cost.sum() + execution_cost.sum() + borrow_cost_series.sum())
    strategy_return_gross = backtest.get("strategy_return_gross")
    if strategy_return_gross is not None:
        strategy_return_gross = strategy_return_gross.astype(float).fillna(0.0)
    else:
        strategy_return_gross = net_returns + transaction_cost + execution_cost + borrow_cost_series
    cost_base = float(exposure_change.abs().sum()) if not exposure_change.empty else 0.0
    cost_ratio = total_cost / max(cost_base, 1e-9)
    avg_holding = calculate_holding_periods(position_series) if not position_series.empty else 0.0
    prediction_accuracy = None
    if include_prediction:
        signal_series = backtest.get("signal")
        if signal_series is not None:
            direction_prediction = np.sign(signal_series.astype(float).fillna(0.0))
            actual_direction = np.sign(asset_returns.shift(-1).fillna(0.0))
            align_mask = direction_prediction != 0
            prediction_accuracy = float((direction_prediction[align_mask] == actual_direction[align_mask]).sum()) / int(align_mask.sum()) if align_mask.any() else 0.0
    auc = float("nan")
    if include_auc and roc_auc_score is not None and "probability" in backtest:
        proba = backtest["probability"].astype(float)
        actual = (asset_returns.shift(-1) > 0).astype(int)
        mask = proba.notna() & actual.notna()
        if mask.sum() > 1 and actual[mask].nunique() > 1:
            try:
                auc = float(roc_auc_score(actual[mask], proba[mask].clip(0, 1)))
            except ValueError:
                auc = float("nan")
    recent_window = net_returns.tail(60)
    recent_sharpe_60d = float(np.sqrt(252) * recent_window.mean() / (recent_window.std() + 1e-12)) if not recent_window.empty and recent_window.std() != 0 else 0.0
    stats = {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "cagr": cagr,
        "calmar": calmar,
        "hit_ratio": hit_ratio,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "avg_exposure": avg_exposure,
        "avg_leverage": avg_leverage,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "es_95": es_95,
        "annual_turnover": annual_turnover,
        "average_holding_days": avg_holding,
        "cost_ratio": cost_ratio,
        "total_cost": total_cost,
        "transaction_cost_total": float(transaction_cost.sum()),
        "execution_cost_total": float(execution_cost.sum()),
        "borrow_cost_total": float(borrow_cost_series.sum()),
        "trading_days": int(backtest.shape[0]),
        "annual_factor": annual_factor,
        "recent_sharpe_60d": recent_sharpe_60d,
        "recovery_days": recovery_days,
        "loss_streak": loss_streak,
        "feature_columns": feature_columns or [],
        "feature_count": len(feature_columns or []),
        "shap_img": shap_img,
        "twr_days": recovery_days,
        "return_path": getattr(params, "return_path", "close_to_close"),
        "label_return_path": getattr(params, "label_return_path", None) or getattr(params, "return_path", "close_to_close"),
        "pfws_train_window": getattr(params, "train_window", None),
        "pfws_test_window": getattr(params, "test_window", None),
        "pfws_embargo": getattr(params, "embargo_days", None),
        "pfws_enforced": bool(getattr(params, "enforce_pfws_only", False)),
    }
    stats.setdefault(
        "cost_assumptions",
        {
            "slippage_model": getattr(params, "slippage_model", None),
            "cost_rate": (getattr(params, "transaction_cost_bps", 0.0) + getattr(params, "slippage_bps", 0.0)) / 10000.0,
            "long_borrow_bps": getattr(params, "long_borrow_cost_bps", None) or getattr(params, "borrow_cost_bps", None),
            "short_borrow_bps": getattr(params, "short_borrow_cost_bps", None) or getattr(params, "borrow_cost_bps", None),
            "adv_participation": getattr(params, "max_adv_participation", None),
            "execution_mode": getattr(params, "execution_mode", None),
        },
    )
    exec_mode = stats["return_path"]
    label_mode = stats["label_return_path"]
    pfws_policy = "enforced" if stats["pfws_enforced"] else "not_enforced"
    stats["execution_assumptions"] = {
        "return_path": exec_mode,
        "label_return_path": label_mode,
        "description": f"执行口径={exec_mode}，标签口径={label_mode}",
    }
    stats["pfws_policy"] = {
        "status": pfws_policy,
        "train_window": stats["pfws_train_window"],
        "test_window": stats["pfws_test_window"],
        "embargo": stats["pfws_embargo"],
        "note": "PFWS 强制仅对 ML 训练/验证生效；传统/确定性策略无样本外切分。" if not stats["pfws_enforced"] else "全局启用 PFWS 强制，禁止非 PFWS 切分。",
    }
    if include_prediction:
        stats["prediction_accuracy"] = prediction_accuracy if prediction_accuracy is not None else float("nan")
    if include_auc:
        stats["auc"] = auc
    robust = compute_robust_sharpe(net_returns, annual_factor=annual_factor, trials=max(1, int(getattr(params, "hyperopt_trials", 1) or 1)))
    if robust:
        stats["sharpe_std_error"] = robust.get("std_error")
        stats["sharpe_ci"] = robust.get("ci")
        if robust.get("ci") and isinstance(robust.get("ci"), tuple):
            stats["sharpe_ci_lower"] = robust["ci"][0]
            stats["sharpe_ci_upper"] = robust["ci"][1]
        if robust.get("deflated_sharpe") is not None:
            stats["deflated_sharpe"] = robust.get("deflated_sharpe")
    white_rc = compute_white_reality_check(net_returns, trials=max(1, int(getattr(params, "hyperopt_trials", 1) or 1)), annual_factor=annual_factor)
    if white_rc:
        stats["sharpe_pvalue"] = white_rc.get("p_value")
        stats["sharpe_pvalue_adjusted"] = white_rc.get("p_value_adjusted")
        stats["sharpe_zscore"] = white_rc.get("z_score")
    enable_bootstrap = bool(getattr(params, "stats_enable_bootstrap", True))
    bootstrap_samples = max(0, int(getattr(params, "stats_bootstrap_samples", 600) or 0))
    bootstrap_block = getattr(params, "stats_bootstrap_block", None)
    if enable_bootstrap and bootstrap_samples > 0:
        white_boot = compute_white_reality_check_bootstrap(
            net_returns,
            trials=max(1, int(getattr(params, "hyperopt_trials", 1) or 1)),
            block_size=bootstrap_block,
            bootstrap_samples=bootstrap_samples,
            annual_factor=annual_factor,
            seed=getattr(params, "random_seed", DEFAULT_STRATEGY_SEED),
        )
        if white_boot:
            stats["sharpe_pvalue_bootstrap"] = white_boot.get("p_value_bootstrap")
            stats["sharpe_bootstrap_mean"] = white_boot.get("sharpe_bootstrap_mean")
            stats["sharpe_bootstrap_std"] = white_boot.get("sharpe_bootstrap_std")
            stats["sharpe_bootstrap_block"] = white_boot.get("block_size")
        spa = compute_spa_pvalue(
            net_returns,
            block_size=bootstrap_block,
            bootstrap_samples=bootstrap_samples,
            annual_factor=annual_factor,
            seed=getattr(params, "random_seed", DEFAULT_STRATEGY_SEED),
        )
        if spa:
            stats["sharpe_pvalue_spa"] = spa.get("p_value_spa")
            stats["sharpe_spa_block"] = spa.get("block_size")
            stats["sharpe_spa_bootstrap_mean"] = spa.get("sharpe_bootstrap_mean")
            stats["sharpe_spa_bootstrap_std"] = spa.get("sharpe_bootstrap_std")
    metrics = build_core_metrics(stats, include_prediction=include_prediction, include_auc=include_auc)
    return metrics, stats


def combine_strategy_outcomes(
    outcomes: list[StrategyOutcome],
    params: StrategyInput,
    overrides: Optional[dict[str, Any]] = None,
) -> tuple[StrategyOutcome, dict[str, float]]:
    if not outcomes:
        raise QuantStrategyError("没有可合并的策略结果。")
    if len(outcomes) == 1:
        single = outcomes[0]
        return single, {single.engine: 1.0}
    overrides = overrides or {}
    override_weights = overrides.get("weights") if isinstance(overrides.get("weights"), dict) else {}
    raw_weights: list[float] = []
    for outcome in outcomes:
        key = outcome.engine
        weight = override_weights.get(key)
        if weight is None and "（" in key:
            weight = override_weights.get(key.split("（", 1)[0])
        if weight is None:
            weight = outcome.weight
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        raw_weights.append(max(weight, 0.0))
    total_weight = sum(raw_weights)
    if total_weight <= 0:
        raw_weights = [1.0 for _ in outcomes]
        total_weight = float(len(outcomes))
    normalized_weights = [w / total_weight for w in raw_weights]
    weights_map = {out.engine: w for out, w in zip(outcomes, normalized_weights)}
    base = outcomes[0].backtest.copy()
    index = base.index
    numeric_cols = ["strategy_return_gross", "transaction_cost", "signal", "position", "leverage", "exposure"]
    combined_series: dict[str, pd.Series] = {col: pd.Series(0.0, index=index, dtype=float) for col in numeric_cols}
    probability_values = pd.Series(0.0, index=index, dtype=float)
    probability_weights = pd.Series(0.0, index=index, dtype=float)
    for outcome, weight in zip(outcomes, normalized_weights):
        df = outcome.backtest.reindex(index)
        for col in numeric_cols:
            if col in df:
                combined_series[col] = combined_series[col] + df[col].astype(float).fillna(0.0) * weight
        if "probability" in df:
            proba = df["probability"].astype(float)
            mask = proba.notna()
            probability_values.loc[mask] += proba[mask] * weight
            probability_weights.loc[mask] += weight
    ensemble = base.copy()
    for col, series in combined_series.items():
        ensemble[col] = series
    if probability_weights.gt(0).any():
        prob_series = probability_values.copy()
        mask = probability_weights > 0
        prob_series.loc[mask] = probability_values.loc[mask] / probability_weights.loc[mask]
        prob_series.loc[~mask] = np.nan
        ensemble["probability"] = prob_series
    ensemble["signal"] = np.clip(ensemble.get("signal", pd.Series(0.0, index=index)), -1.0, 1.0)
    ensemble["position"] = np.clip(ensemble.get("position", pd.Series(0.0, index=index)), -1.0, 1.0)
    ensemble["leverage"] = ensemble.get("leverage", pd.Series(0.0, index=index)).clip(lower=0.0)
    ensemble["exposure"] = ensemble["position"] * ensemble["leverage"]
    ensemble["strategy_return_gross"] = ensemble.get("strategy_return_gross", pd.Series(0.0, index=index))
    ensemble["transaction_cost"] = ensemble.get("transaction_cost", pd.Series(0.0, index=index))
    ensemble["strategy_return"] = ensemble["strategy_return_gross"] - ensemble["transaction_cost"]
    ensemble["cum_strategy"] = (1 + ensemble["strategy_return"]).cumprod()
    if "asset_return" in ensemble:
        ensemble["cum_buy_hold"] = (1 + ensemble["asset_return"].fillna(0.0)).cumprod()
    has_probability = any("probability" in out.backtest.columns for out in outcomes)
    feature_union = sorted({col for out in outcomes for col in out.stats.get("feature_columns", [])})
    shap_img = next((out.stats.get("shap_img") for out in outcomes if out.stats.get("shap_img")), None)
    metrics, stats = summarize_backtest(
        ensemble, params, include_prediction=True, include_auc=has_probability, feature_columns=feature_union, shap_img=shap_img
    )
    component_breakdown = []
    for outcome, weight in zip(outcomes, normalized_weights):
        component_breakdown.append(
            {"engine": outcome.engine, "weight": weight, "sharpe": outcome.stats.get("sharpe"), "total_return": outcome.stats.get("total_return"), "max_drawdown": outcome.stats.get("max_drawdown")}
        )
    stats.update({"weights": weights_map, "component_breakdown": component_breakdown})
    combined_outcome = StrategyOutcome(engine="组合策略", backtest=ensemble, metrics=metrics, stats=stats, weight=1.0)
    return combined_outcome, weights_map


def _compute_oos_from_backtest(backtest: pd.DataFrame, params: StrategyInput) -> dict[str, Any] | None:
    if backtest.empty or "strategy_return" not in backtest:
        return None
    total = len(backtest)
    if total < params.train_window + params.test_window:
        return None
    splitter = PurgedWalkForwardSplit(train_window=params.train_window, test_window=max(params.test_window, 5), embargo=max(0, params.embargo_days))
    slices: list[dict[str, Any]] = []
    for fold_idx, (_train_idx, test_idx) in enumerate(splitter.split(total)):
        test_returns = backtest["strategy_return"].iloc[test_idx].fillna(0.0)
        if test_returns.empty:
            continue
        metrics = compute_validation_metrics(test_returns)
        metrics.update(
            {
                "fold": fold_idx + 1,
                "test_start": str(test_returns.index[0].date()) if hasattr(test_returns.index[0], "date") else str(test_returns.index[0]),
                "test_end": str(test_returns.index[-1].date()) if hasattr(test_returns.index[-1], "date") else str(test_returns.index[-1]),
            }
        )
        slices.append(metrics)
    if not slices:
        return None
    summary = aggregate_oos_metrics(slices)
    sharpe_stats = summary.get("sharpe") or {}
    penalized = (sharpe_stats.get("mean") or 0.0) - (sharpe_stats.get("std") or 0.0)
    distributions = {k: [float(entry.get(k, 0.0)) for entry in slices if entry.get(k) is not None] for k in ("sharpe", "cagr", "max_drawdown", "hit_ratio")}
    return {
        "slices": slices,
        "summary": summary,
        "folds": len(slices),
        "penalized_sharpe": penalized,
        "train_window": params.train_window,
        "test_window": params.test_window,
        "embargo": params.embargo_days,
        "distributions": distributions,
    }
