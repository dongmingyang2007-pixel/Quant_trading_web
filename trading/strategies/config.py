from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import pandas as pd

# 默认种子配置，集中管理以便各模块复用/注入
DEFAULT_STRATEGY_SEED = int(os.environ.get("STRATEGY_SEED", "42"))
DEFAULT_SEED_META = {"strategy_seed": DEFAULT_STRATEGY_SEED}


class QuantStrategyError(Exception):
    """Custom exception for strategy failures."""


@dataclass(slots=True)
class StrategyInput:
    ticker: str
    benchmark_ticker: Optional[str]
    start_date: date
    end_date: date
    short_window: int
    long_window: int
    rsi_period: int
    include_plots: bool
    show_ai_thoughts: bool
    risk_profile: str
    capital: float
    strategy_engine: str = "ml_momentum"
    volatility_target: float = 0.15
    transaction_cost_bps: float = 8.0
    slippage_bps: float = 5.0
    min_holding_days: int = 3
    train_window: int = 252
    test_window: int = 21
    entry_threshold: float = 0.55
    exit_threshold: float = 0.45
    max_leverage: float = 2.5
    # ML pipeline controls
    ml_task: str = "direction"  # 'direction' | 'hybrid'
    val_ratio: float = 0.15
    embargo_days: int = 5
    optimize_thresholds: bool = True
    ml_model: str = "seq_hybrid"  # 默认自动融合 LSTM+Transformer；如后端缺失将自动回退
    ml_params: dict[str, Any] | None = None
    auto_apply_best_config: bool = True
    calibrate_proba: bool = True
    early_stopping_rounds: int = 50
    use_feature_cache: bool = True
    enable_hyperopt: bool = False
    hyperopt_trials: int = 20
    hyperopt_timeout: int = 120
    max_drawdown_stop: float = 0.25
    daily_exposure_limit: float = 1.5
    dl_sequence_length: int = 32
    dl_hidden_dim: int = 64
    dl_dropout: float = 0.2
    dl_epochs: int = 12
    dl_batch_size: int = 64
    dl_num_layers: int = 2
    rl_engine: str = "value_iter"
    rl_params: dict[str, Any] | None = None
    validation_slices: int = 3
    out_of_sample_ratio: float = 0.2
    execution_liquidity_buffer: float = 0.05
    execution_penalty_bps: float = 6.0
    execution_mode: str = "adv"  # 'adv' or 'limit'
    class_weight_mode: str = "balanced"  # imbalance handling
    focal_gamma: float = 2.0
    dynamic_exposure_multiplier: float = 1.0
    label_return_path: str | None = None  # override label/future_return path; defaults to return_path
    intraday_loss_limit: float = 0.08
    slippage_model: dict[str, Any] | None = None
    borrow_cost_bps: float = 0.0
    long_borrow_cost_bps: float = 0.0
    short_borrow_cost_bps: float = 0.0
    max_adv_participation: float = 0.1
    target_vol: float | None = None
    vol_target_window: int = 60
    # triple barrier labeling (optional)
    label_style: str = "direction"  # 'direction' | 'triple_barrier'
    tb_up: float = 0.03
    tb_down: float = 0.03
    tb_max_holding: int = 10
    tb_dynamic: bool = False
    tb_vol_multiplier: float = 1.2
    tb_vol_window: int = 20
    interest_keywords: list[str] | None = None
    investment_horizon: str = "medium"
    experience_level: str = "novice"
    primary_goal: str = "growth"
    return_path: str = "close_to_close"  # 'close_to_close' | 'close_to_open'
    # 支持标签/回测收益口径：close_to_close | close_to_open | open_to_close
    # 若前端需要拆分盘中/隔夜，可用 label_return_path 切换
    request_id: str | None = None
    user_id: str | None = None
    model_version: str | None = None
    data_version: str | None = None
    exec_latency_ms: float | None = None
    include_walk_forward_report: bool = False
    walk_forward_horizon_days: int | None = None
    walk_forward_step_days: int | None = None
    walk_forward_jobs: int = 1
    force_pfws: bool = True
    # stats / reporting controls
    stats_enable_bootstrap: bool = True
    stats_bootstrap_samples: int = 600
    stats_bootstrap_block: int | None = None
    enforce_pfws_only: bool = False  # 若为 True，禁止非 PFWS 切分的训练/验证
    random_seed: int = DEFAULT_STRATEGY_SEED
    threshold_jobs: int = 1


@dataclass(slots=True)
class StrategyOutcome:
    engine: str
    backtest: pd.DataFrame
    metrics: list[dict[str, Any]]
    stats: dict[str, Any]
    weight: float = 1.0


__all__ = [
    "DEFAULT_STRATEGY_SEED",
    "DEFAULT_SEED_META",
    "QuantStrategyError",
    "StrategyInput",
    "StrategyOutcome",
]
