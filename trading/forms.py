from datetime import date

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.utils.translation import get_language

from .strategy_defaults import ADVANCED_STRATEGY_DEFAULTS


class ApiCredentialForm(forms.Form):
    alpaca_api_key_id = forms.CharField(
        required=False,
        label="Alpaca API Key ID",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于 Alpaca 行情/交易 API（Key ID）。",
    )
    alpaca_api_secret_key = forms.CharField(
        required=False,
        label="Alpaca API Secret",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于 Alpaca 行情/交易 API（Secret Key）。",
    )
    bailian_api_key = forms.CharField(
        required=False,
        label="BaiLian (DashScope) API Key",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于阿里云百炼（通义千问）API Key。",
    )
    aliyun_access_key_id = forms.CharField(
        required=False,
        label="Aliyun AccessKey ID",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于阿里云服务的 AccessKey ID。",
    )
    aliyun_access_key_secret = forms.CharField(
        required=False,
        label="Aliyun AccessKey Secret",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于阿里云服务的 AccessKey Secret。",
    )
    gemini_api_key = forms.CharField(
        required=False,
        label="Gemini API Key",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于 Gemini 模型调用。",
    )
    ollama_api_key = forms.CharField(
        required=False,
        label="Ollama API Key",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于 Ollama Web Search 或云端接口（可选）。",
    )
    strategy_update_auth_token = forms.CharField(
        required=False,
        label="Strategy Update Token",
        widget=forms.PasswordInput(render_value=True),
        help_text="用于远程策略覆写（可选）。",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self.fields:
            self.fields[name].widget.attrs.setdefault("class", "form-control")
            self.fields[name].widget.attrs.setdefault("autocomplete", "off")


class QuantStrategyForm(forms.Form):
    ML_MODE_CHOICES_ZH = [
        ("light", "轻量：梯度提升 (默认)"),
        ("deep", "深度：LSTM 序列模型"),
        ("transformer", "深度：Transformer 编码器"),
        ("fusion", "深度：自动融合 LSTM + Transformer"),
    ]
    ML_MODE_CHOICES_EN = [
        ("light", "Light · Gradient Boosting (default)"),
        ("deep", "Deep · LSTM sequence model"),
        ("transformer", "Deep · Transformer encoder"),
        ("fusion", "Deep · Fusion (auto LSTM + Transformer)"),
    ]
    STRATEGY_ENGINE_CHOICES_ZH = [
        ("multi_combo", "组合策略（ML + RL + 均线）"),
        ("ml_momentum", "机器学习动量"),
        ("rl_policy", "强化学习策略"),
        ("sma", "传统双均线"),
    ]
    STRATEGY_ENGINE_CHOICES_EN = [
        ("multi_combo", "Multi-combo (ML + RL + SMA)"),
        ("ml_momentum", "ML momentum"),
        ("rl_policy", "RL policy"),
        ("sma", "Classic SMA"),
    ]
    RISK_PROFILE_CHOICES_ZH = [
        ("conservative", "保守"),
        ("balanced", "均衡"),
        ("aggressive", "激进"),
    ]
    RISK_PROFILE_CHOICES_EN = [
        ("conservative", "Conservative"),
        ("balanced", "Balanced"),
        ("aggressive", "Aggressive"),
    ]
    RETURN_PATH_CHOICES_ZH = [
        ("close_to_close", "收盘→收盘（默认）"),
        ("close_to_open", "收盘→次日开盘（隔夜）"),
        ("open_to_close", "开盘→收盘（盘中）"),
    ]
    RETURN_PATH_CHOICES_EN = [
        ("close_to_close", "Close-to-close (default)"),
        ("close_to_open", "Close-to-open (overnight)"),
        ("open_to_close", "Open-to-close (intraday)"),
    ]

    ticker = forms.CharField(
        max_length=16,
        label="主要标的代码",
        help_text="支持 Yahoo Finance 代码，例如 NVDA、AAPL、600519.SS。",
    )
    benchmark_ticker = forms.CharField(
        max_length=16,
        required=False,
        label="对比基准（可选）",
        help_text="默认对比 SPY。若专注单一资产可留空。",
    )
    start_date = forms.DateField(
        label="开始日期",
        widget=forms.DateInput(attrs={"type": "date"}),
    )
    end_date = forms.DateField(
        label="结束日期",
        widget=forms.DateInput(attrs={"type": "date"}),
    )
    capital = forms.DecimalField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["capital"],
        label="模拟资金规模",
        help_text="用于资产配置建模，默认 250,000。",
    )
    ml_mode = forms.ChoiceField(
        label="模型类型",
        choices=ML_MODE_CHOICES_ZH,
        initial="light",
        help_text="轻量 GBDT 兼具速度与稳定性；深度模型可捕捉更多时序模式，Fusion 会自动融合 LSTM 与 Transformer。",
    )
    strategy_engine = forms.ChoiceField(
        label="策略引擎",
        choices=STRATEGY_ENGINE_CHOICES_ZH,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["strategy_engine"],
        help_text="选择核心策略引擎，默认组合策略。",
    )
    risk_profile = forms.ChoiceField(
        label="风险偏好",
        choices=RISK_PROFILE_CHOICES_ZH,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["risk_profile"],
        help_text="控制风险预算与仓位尺度。",
    )
    short_window = forms.IntegerField(
        min_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["short_window"],
        label="短期均线窗口",
        help_text="短期均线窗口（天）。",
    )
    long_window = forms.IntegerField(
        min_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["long_window"],
        label="长期均线窗口",
        help_text="长期均线窗口（天）。",
    )
    rsi_period = forms.IntegerField(
        min_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["rsi_period"],
        label="RSI 周期",
        help_text="相对强弱指标窗口长度。",
    )
    volatility_target = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["volatility_target"],
        label="目标波动率",
        help_text="年度目标波动率（0-1）。",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    transaction_cost_bps = forms.FloatField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["transaction_cost_bps"],
        label="交易成本（bps）",
        help_text="单次交易成本（bps）。",
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    slippage_bps = forms.FloatField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["slippage_bps"],
        label="滑点（bps）",
        help_text="滑点假设（bps）。",
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    return_path = forms.ChoiceField(
        label="执行口径",
        choices=RETURN_PATH_CHOICES_ZH,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["return_path"],
        help_text="回测收益使用的价格路径。",
    )
    max_adv_participation = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["max_adv_participation"],
        label="ADV 参与率上限",
        help_text="单日成交额参与率上限（0-1）。",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    execution_liquidity_buffer = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["execution_liquidity_buffer"],
        label="流动性缓冲",
        help_text="执行时使用的流动性缓冲比例（0-1）。",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    execution_penalty_bps = forms.FloatField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["execution_penalty_bps"],
        label="执行冲击惩罚（bps）",
        help_text="冲击成本惩罚（bps）。",
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    limit_move_threshold = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["limit_move_threshold"],
        label="涨跌幅限制阈值",
        help_text="触及涨跌停的阈值（0-1），为空则忽略。",
        widget=forms.NumberInput(attrs={"step": "0.001"}),
    )
    borrow_cost_bps = forms.FloatField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["borrow_cost_bps"],
        label="融券成本（bps）",
        help_text="空头借券成本（bps）。",
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    min_holding_days = forms.IntegerField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["min_holding_days"],
        label="最短持仓天数",
        help_text="最少持仓天数。",
    )
    entry_threshold = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["entry_threshold"],
        label="入场阈值",
        help_text="入场信号阈值（0-1）。",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    exit_threshold = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["exit_threshold"],
        label="离场阈值",
        help_text="离场信号阈值（0-1）。",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    optimize_thresholds = forms.BooleanField(
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["optimize_thresholds"],
        label="优化阈值",
        help_text="自动微调入场/离场阈值。",
    )
    train_window = forms.IntegerField(
        min_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["train_window"],
        label="训练窗口",
        help_text="训练样本长度。",
    )
    test_window = forms.IntegerField(
        min_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["test_window"],
        label="测试窗口",
        help_text="滚动测试窗口长度。",
    )
    val_ratio = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["val_ratio"],
        label="验证集比例",
        help_text="验证集比例（0-1）。",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    embargo_days = forms.IntegerField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["embargo_days"],
        label="隔离期",
        help_text="训练/测试之间的隔离天数。",
    )
    auto_apply_best_config = forms.BooleanField(
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["auto_apply_best_config"],
        label="自动套用最优配置",
        help_text="优先使用缓存中的最优引擎配置。",
    )
    enable_hyperopt = forms.BooleanField(
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["enable_hyperopt"],
        label="启用超参搜索",
        help_text="启用超参数搜索流程。",
    )
    hyperopt_trials = forms.IntegerField(
        min_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["hyperopt_trials"],
        label="超参搜索次数",
        help_text="超参数搜索迭代次数。",
    )
    hyperopt_timeout = forms.IntegerField(
        min_value=10,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["hyperopt_timeout"],
        label="超参搜索超时（秒）",
        help_text="单次搜索最大耗时（秒）。",
    )
    max_leverage = forms.FloatField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["max_leverage"],
        label="最大杠杆",
        help_text="单日最大杠杆倍数。",
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    max_drawdown_stop = forms.FloatField(
        min_value=0,
        max_value=1,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["max_drawdown_stop"],
        label="最大回撤止损",
        help_text="最大可承受回撤比例（0-1）。",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    daily_exposure_limit = forms.FloatField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["daily_exposure_limit"],
        label="日内暴露上限",
        help_text="单日最大暴露倍数。",
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    allow_short = forms.BooleanField(
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["allow_short"],
        label="允许做空",
        help_text="允许策略进行做空。",
    )
    execution_delay_days = forms.IntegerField(
        min_value=0,
        required=False,
        initial=ADVANCED_STRATEGY_DEFAULTS["execution_delay_days"],
        label="执行延迟（天）",
        help_text="信号执行的延迟天数。",
    )

    def __init__(self, *args, language=None, **kwargs):
        self.language = (language or get_language() or "").lower()
        self.lang_is_zh = self.language.startswith("zh")
        super().__init__(*args, **kwargs)
        def _msg(en: str, zh: str) -> str:
            return zh if self.lang_is_zh else en
        self._msg = _msg
        label_map = {
            "ticker": ("Primary ticker", "主要标的代码"),
            "benchmark_ticker": ("Benchmark (optional)", "对比基准（可选）"),
            "start_date": ("Start date", "开始日期"),
            "end_date": ("End date", "结束日期"),
            "capital": ("Capital base", "模拟资金规模"),
            "ml_mode": ("Model type", "模型类型"),
            "strategy_engine": ("Strategy engine", "策略引擎"),
            "risk_profile": ("Risk profile", "风险偏好"),
            "short_window": ("Short window", "短期均线窗口"),
            "long_window": ("Long window", "长期均线窗口"),
            "rsi_period": ("RSI period", "RSI 周期"),
            "volatility_target": ("Volatility target", "目标波动率"),
            "transaction_cost_bps": ("Transaction cost (bps)", "交易成本（bps）"),
            "slippage_bps": ("Slippage (bps)", "滑点（bps）"),
            "return_path": ("Return path", "执行口径"),
            "max_adv_participation": ("Max ADV participation", "ADV 参与率上限"),
            "execution_liquidity_buffer": ("Liquidity buffer", "流动性缓冲"),
            "execution_penalty_bps": ("Execution penalty (bps)", "执行冲击惩罚（bps）"),
            "limit_move_threshold": ("Limit move threshold", "涨跌幅限制阈值"),
            "borrow_cost_bps": ("Borrow cost (bps)", "融券成本（bps）"),
            "min_holding_days": ("Min holding days", "最短持仓天数"),
            "entry_threshold": ("Entry threshold", "入场阈值"),
            "exit_threshold": ("Exit threshold", "离场阈值"),
            "optimize_thresholds": ("Optimize thresholds", "优化阈值"),
            "train_window": ("Train window", "训练窗口"),
            "test_window": ("Test window", "测试窗口"),
            "val_ratio": ("Validation ratio", "验证集比例"),
            "embargo_days": ("Embargo days", "隔离期"),
            "auto_apply_best_config": ("Auto-apply best config", "自动套用最优配置"),
            "enable_hyperopt": ("Enable hyperopt", "启用超参搜索"),
            "hyperopt_trials": ("Hyperopt trials", "超参搜索次数"),
            "hyperopt_timeout": ("Hyperopt timeout (sec)", "超参搜索超时（秒）"),
            "max_leverage": ("Max leverage", "最大杠杆"),
            "max_drawdown_stop": ("Max drawdown stop", "最大回撤止损"),
            "daily_exposure_limit": ("Daily exposure limit", "日内暴露上限"),
            "allow_short": ("Allow short selling", "允许做空"),
            "execution_delay_days": ("Execution delay (days)", "执行延迟（天）"),
        }
        help_map = {
            "ticker": (
                "Accepts Yahoo Finance symbols, e.g., NVDA, AAPL, 600519.SS.",
                "支持 Yahoo Finance 代码，例如 NVDA、AAPL、600519.SS。",
            ),
            "benchmark_ticker": (
                "Defaults to SPY. Leave blank when focusing on a single asset.",
                "默认对比 SPY。若专注单一资产可留空。",
            ),
            "capital": (
                "Used for allocation modeling. Default 250,000.",
                "用于资产配置建模，默认 250,000。",
            ),
            "ml_mode": (
                "Light GBDT balances speed and stability; deep modes capture richer sequences. Fusion blends LSTM and Transformer automatically.",
                "轻量 GBDT 兼具速度与稳定性；深度模型可捕捉更多时序模式；Fusion 会自动融合 LSTM 与 Transformer。",
            ),
            "strategy_engine": (
                "Select the core engine powering this backtest.",
                "选择本次回测的核心策略引擎。",
            ),
            "risk_profile": (
                "Sets risk budgeting and position sizing style.",
                "控制风险预算与仓位尺度。",
            ),
            "short_window": (
                "Short SMA window in days.",
                "短期均线窗口（天）。",
            ),
            "long_window": (
                "Long SMA window in days.",
                "长期均线窗口（天）。",
            ),
            "rsi_period": (
                "RSI indicator window length.",
                "相对强弱指标窗口长度。",
            ),
            "volatility_target": (
                "Target annualized volatility (0-1).",
                "年度目标波动率（0-1）。",
            ),
            "transaction_cost_bps": (
                "Transaction cost assumption (bps).",
                "单次交易成本（bps）。",
            ),
            "slippage_bps": (
                "Slippage assumption (bps).",
                "滑点假设（bps）。",
            ),
            "return_path": (
                "Return path used for realized P&L.",
                "回测收益使用的价格路径。",
            ),
            "max_adv_participation": (
                "Max daily participation vs ADV (0-1).",
                "单日成交额参与率上限（0-1）。",
            ),
            "execution_liquidity_buffer": (
                "Liquidity buffer for execution impact (0-1).",
                "执行时使用的流动性缓冲比例（0-1）。",
            ),
            "execution_penalty_bps": (
                "Execution impact penalty in bps.",
                "冲击成本惩罚（bps）。",
            ),
            "limit_move_threshold": (
                "Daily move threshold to block fills (0-1). Leave blank to disable.",
                "触及涨跌停的阈值（0-1），为空则忽略。",
            ),
            "borrow_cost_bps": (
                "Borrow cost for short positions (bps).",
                "空头借券成本（bps）。",
            ),
            "min_holding_days": (
                "Minimum holding days per position.",
                "最少持仓天数。",
            ),
            "entry_threshold": (
                "Entry signal threshold (0-1).",
                "入场信号阈值（0-1）。",
            ),
            "exit_threshold": (
                "Exit signal threshold (0-1).",
                "离场信号阈值（0-1）。",
            ),
            "optimize_thresholds": (
                "Auto-tune entry/exit thresholds.",
                "自动微调入场/离场阈值。",
            ),
            "train_window": (
                "Training window length.",
                "训练样本长度。",
            ),
            "test_window": (
                "Rolling test window length.",
                "滚动测试窗口长度。",
            ),
            "val_ratio": (
                "Validation split ratio (0-1).",
                "验证集比例（0-1）。",
            ),
            "embargo_days": (
                "Embargo window between train/test.",
                "训练/测试之间的隔离天数。",
            ),
            "auto_apply_best_config": (
                "Prefer cached best engine configuration.",
                "优先使用缓存中的最优引擎配置。",
            ),
            "enable_hyperopt": (
                "Enable hyperparameter search pipeline.",
                "启用超参数搜索流程。",
            ),
            "hyperopt_trials": (
                "Hyperparameter search iterations.",
                "超参数搜索迭代次数。",
            ),
            "hyperopt_timeout": (
                "Max hyperopt runtime per trial (sec).",
                "单次搜索最大耗时（秒）。",
            ),
            "max_leverage": (
                "Maximum gross leverage.",
                "单日最大杠杆倍数。",
            ),
            "max_drawdown_stop": (
                "Stop when drawdown exceeds threshold (0-1).",
                "最大可承受回撤比例（0-1）。",
            ),
            "daily_exposure_limit": (
                "Max daily exposure multiplier.",
                "单日最大暴露倍数。",
            ),
            "allow_short": (
                "Allow short selling.",
                "允许策略进行做空。",
            ),
            "execution_delay_days": (
                "Signal execution delay in days.",
                "信号执行的延迟天数。",
            ),
        }
        for field, labels in label_map.items():
            self.fields[field].label = labels[1] if self.lang_is_zh else labels[0]
        for field, helps in help_map.items():
            self.fields[field].help_text = helps[1] if self.lang_is_zh else helps[0]
        self.fields["ml_mode"].choices = self.ML_MODE_CHOICES_ZH if self.lang_is_zh else self.ML_MODE_CHOICES_EN
        self.fields["strategy_engine"].choices = (
            self.STRATEGY_ENGINE_CHOICES_ZH if self.lang_is_zh else self.STRATEGY_ENGINE_CHOICES_EN
        )
        self.fields["risk_profile"].choices = (
            self.RISK_PROFILE_CHOICES_ZH if self.lang_is_zh else self.RISK_PROFILE_CHOICES_EN
        )
        self.fields["return_path"].choices = (
            self.RETURN_PATH_CHOICES_ZH if self.lang_is_zh else self.RETURN_PATH_CHOICES_EN
        )
        self.warnings: list[str] = []
        self.fields["ticker"].widget.attrs.update({"placeholder": "NVDA"})
        self.fields["benchmark_ticker"].widget.attrs.update({"placeholder": "SPY"})
        for name in (
            "start_date",
            "end_date",
            "capital",
            "short_window",
            "long_window",
            "rsi_period",
            "volatility_target",
            "transaction_cost_bps",
            "slippage_bps",
            "return_path",
            "max_adv_participation",
            "execution_liquidity_buffer",
            "execution_penalty_bps",
            "limit_move_threshold",
            "borrow_cost_bps",
            "min_holding_days",
            "entry_threshold",
            "exit_threshold",
            "train_window",
            "test_window",
            "val_ratio",
            "embargo_days",
            "hyperopt_trials",
            "hyperopt_timeout",
            "max_leverage",
            "max_drawdown_stop",
            "daily_exposure_limit",
            "execution_delay_days",
        ):
            self.fields[name].widget.attrs.setdefault("class", "form-control")
        for name in ("ticker", "benchmark_ticker", "ml_mode", "strategy_engine", "risk_profile"):
            self.fields[name].widget.attrs.setdefault("class", "form-control")
        for name in ("optimize_thresholds", "auto_apply_best_config", "enable_hyperopt", "allow_short"):
            self.fields[name].widget.attrs.setdefault("class", "form-check-input")

    def clean(self):
        cleaned = super().clean()
        start = cleaned.get("start_date")
        end = cleaned.get("end_date")
        if start and end:
            today = date.today()
            if end > today:
                cleaned["end_date"] = today
                self.warnings.append(
                    self._msg(
                        f"End date adjusted to {today} to avoid using future data.",
                        f"结束日期已自动校正为 {today}，避免使用未来数据。",
                    )
                )
                end = today
            if start >= end:
                raise ValidationError(self._msg("Start date must be earlier than end date.", "开始日期必须早于结束日期。"))
            if (end - start).days < 120:
                self.warnings.append(
                    self._msg(
                        "The backtest window is shorter than 120 days. Extend it for more stable statistics.",
                        "回测窗口少于 120 天，建议拉长周期以提高统计稳定性。",
                    )
                )
        ticker = (cleaned.get("ticker") or "").strip().upper()
        benchmark = (cleaned.get("benchmark_ticker") or "").strip().upper()
        if benchmark and benchmark == ticker:
            self.warnings.append(
                self._msg(
                    "Benchmark matches the primary ticker, so it was ignored.",
                    "基准代码与主要标的一致，已自动忽略基准。",
                )
            )
            cleaned["benchmark_ticker"] = ""
        capital = cleaned.get("capital")
        if capital is not None and capital < 10000:
            self.warnings.append(
                self._msg(
                    "Capital below 10,000 increases the impact of commissions and minimum lot sizes.",
                    "模拟资金低于 10,000，实际交易时需关注手续费与最小交易单位。",
                )
            )
        short_window = cleaned.get("short_window")
        long_window = cleaned.get("long_window")
        if short_window and long_window and short_window >= long_window:
            raise ValidationError(
                self._msg(
                    "Short window must be smaller than the long window.",
                    "短期均线窗口必须小于长期均线窗口。",
                )
            )
        return cleaned


class ProfileForm(forms.Form):
    display_name = forms.CharField(
        max_length=40,
        required=False,
        label="展示昵称",
        help_text="显示在个人主页上的名称，默认沿用账号用户名。",
    )
    cover_color = forms.CharField(
        max_length=16,
        required=False,
        initial="#116e5f",
        label="背景主色",
        help_text="",
        widget=forms.TextInput(attrs={"type": "color", "class": "form-control form-control-color"}),
    )
    bio = forms.CharField(
        required=False,
        label="个人简介",
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "介绍一下你的投资风格或兴趣..."}),
    )
    avatar = forms.ImageField(
        required=False,
        label="上传头像",
        help_text="支持 JPG/PNG，建议尺寸 400x400 以上。",
    )
    avatar_cropped_data = forms.CharField(required=False, widget=forms.HiddenInput)
    feature_image = forms.ImageField(
        required=False,
        label="展示照片",
        help_text="可选，用于展示在主页的代表性照片。",
    )
    feature_cropped_data = forms.CharField(required=False, widget=forms.HiddenInput)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("display_name", "bio"):
            self.fields[name].widget.attrs.setdefault("class", "form-control")
        cover_widget = self.fields["cover_color"].widget
        cover_widget.attrs.setdefault("class", "form-control form-control-color")
        self.fields["avatar"].widget.attrs.update(
            {
                "class": "visually-hidden avatar-file-input",
                "data-role": "avatar-input",
                "accept": "image/*",
                "form": "account-profile-form",
                "aria-hidden": "true",
                "tabindex": "-1",
            }
        )
        self.fields["avatar_cropped_data"].widget.attrs.update(
            {
                "data-role": "avatar-cropped",
                "form": "account-profile-form",
            }
        )
        # Featured image is hidden and controlled by JS uploader
        self.fields["feature_image"].widget.attrs.update(
            {
                "class": "d-none",
                "data-role": "feature-input",
                "accept": "image/*",
            }
        )
        self.fields["feature_cropped_data"].widget.attrs.update(
            {
                "data-role": "feature-cropped",
            }
        )


class CommunityPostForm(forms.Form):
    content = forms.CharField(
        label="想法",
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "分享你的策略心得、市场观察或经验..."}),
    )
    topic = forms.ChoiceField(
        label="话题",
        required=False,
        choices=[],
    )
    new_topic_name = forms.CharField(
        label="新话题名称",
        max_length=60,
        required=False,
        widget=forms.TextInput(attrs={"placeholder": "创建新话题…"}),
    )
    new_topic_description = forms.CharField(
        label="话题介绍",
        required=False,
        widget=forms.Textarea(attrs={"rows": 2, "placeholder": "描述话题的讨论方向…"}),
    )
    image = forms.ImageField(
        required=False,
        label="图片上传",
        help_text="可选，支持 JPG/PNG。",
    )
    image_cropped_data = forms.CharField(
        required=False,
        widget=forms.HiddenInput,
    )
    backtest_record_id = forms.CharField(
        required=False,
        widget=forms.HiddenInput,
    )

    def __init__(self, *args, topics: list[tuple[str, str]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["content"].widget.attrs.setdefault("class", "form-control")
        self.fields["topic"].widget.attrs.setdefault("class", "form-select")
        self.fields["new_topic_name"].widget.attrs.setdefault("class", "form-control")
        self.fields["new_topic_description"].widget.attrs.setdefault("class", "form-control")
        self.fields["image"].widget.attrs.setdefault("class", "form-control")
        if topics:
            self.fields["topic"].choices = topics
        else:
            self.fields["topic"].choices = []


class SignupForm(UserCreationForm):
    email = forms.EmailField(label="邮箱", help_text="用于找回密码与通知", required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def __init__(self, *args, language=None, **kwargs):
        self.language = (language or get_language() or "").lower()
        super().__init__(*args, **kwargs)
        is_zh = self.language.startswith("zh")
        self.fields["email"].label = "邮箱" if is_zh else "Email"
        self.fields["email"].help_text = "用于找回密码与通知" if is_zh else "Used for password reset and notifications."

    def clean_email(self):
        email = (self.cleaned_data.get("email") or "").strip()
        if not email:
            raise ValidationError("请输入邮箱地址" if self.language.startswith("zh") else "Please enter an email address.")
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("该邮箱已被注册" if self.language.startswith("zh") else "This email is already registered.")
        return email

    def save(self, commit: bool = True):
        user = super().save(commit=False)
        user.email = self.cleaned_data.get("email")
        if commit:
            user.save()
        return user


class ResendActivationForm(forms.Form):
    email = forms.EmailField(label="邮箱", required=True)

    def __init__(self, *args, language=None, **kwargs):
        self.language = (language or get_language() or "").lower()
        super().__init__(*args, **kwargs)
        label = "邮箱" if self.language.startswith("zh") else "Email"
        self.fields["email"].label = label
        self.fields["email"].widget.attrs.setdefault("placeholder", label)
        self.fields["email"].widget.attrs.setdefault("class", "form-control")
