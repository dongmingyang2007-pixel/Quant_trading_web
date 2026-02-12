from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


HYPEROPT_ENABLED = _env_bool("ENABLE_HYPEROPT", True)
HYPEROPT_TRIALS = _env_int("HYPEROPT_TRIALS", 8)
HYPEROPT_TIMEOUT = _env_int("HYPEROPT_TIMEOUT", 90)

ADVANCED_STRATEGY_DEFAULTS = {
    "trading_focus": "intraday_retail",
    "benchmark_ticker": "SPY",
    "capital": 250000.0,
    "short_window": 35,
    "long_window": 126,
    "rsi_period": 21,
    "include_plots": True,
    "show_ai_thoughts": True,
    "risk_profile": "balanced",
    "strategy_engine": "multi_combo",
    "volatility_target": 0.14,
    "transaction_cost_bps": 6.0,
    "slippage_bps": 4.0,
    "execution_penalty_bps": 6.0,
    "execution_liquidity_buffer": 0.05,
    "max_adv_participation": 0.10,
    "borrow_cost_bps": 0.0,
    "min_holding_days": 3,
    "train_window": 504,
    "test_window": 21,
    "entry_threshold": 0.58,
    "exit_threshold": 0.42,
    "max_leverage": 3.0,
    "ml_task": "hybrid",
    "val_ratio": 0.2,
    "embargo_days": 7,
    "optimize_thresholds": True,
    "ml_model": "lightgbm",
    "ml_params": None,
    "auto_apply_best_config": True,
    "calibrate_proba": True,
    "early_stopping_rounds": 80,
    "dl_sequence_length": 36,
    "dl_hidden_dim": 96,
    "dl_dropout": 0.15,
    "dl_epochs": 14,
    "dl_batch_size": 64,
    "dl_num_layers": 2,
    "rl_engine": "value_iter",
    "rl_params": None,
    "label_style": "triple_barrier",
    "tb_up": 0.035,
    "tb_down": 0.03,
    "tb_max_holding": 15,
    "return_path": "close_to_close",
    "enable_hyperopt": HYPEROPT_ENABLED,
    "hyperopt_trials": HYPEROPT_TRIALS,
    "hyperopt_timeout": HYPEROPT_TIMEOUT,
    "max_drawdown_stop": 0.25,
    "daily_exposure_limit": 1.5,
    "investment_horizon": "medium",
    "experience_level": "advanced",
    "primary_goal": "growth",
    "lot_size": 1,
    "max_weight": None,
    "min_weight": None,
    "max_holdings": None,
    "sector_caps": None,
    "turnover_cap": None,
    "allow_short": True,
    "limit_move_threshold": None,
    "execution_delay_days": 1,
}

__all__ = ["ADVANCED_STRATEGY_DEFAULTS"]
