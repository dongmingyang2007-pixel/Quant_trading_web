from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Match


def _identity(message: str) -> str:
    return message


@dataclass(frozen=True)
class TranslationRule:
    pattern: re.Pattern[str]
    render: Callable[[Match[str]], str]


def _rule(pattern: str, render: Callable[[Match[str]], str]) -> TranslationRule:
    return TranslationRule(re.compile(pattern), render)


RULES: list[TranslationRule] = [
    _rule(
        r"^本地缓存文件缺少列 (?P<missing>.+)\. 请确保包含 Close, Adj Close, Volume\.$",
        lambda m: f"The cached CSV is missing columns {m.group('missing')}. Include Close, Adj Close, and Volume.",
    ),
    _rule(
        r"^已从本地缓存 (?P<path>.+) 读取数据。若需最新行情，请联网后刷新缓存。$",
        lambda m: f"Loaded cached price data from {m.group('path')}. Connect to refresh quotes for the latest data.",
    ),
    _rule(
        r"^无法从 Yahoo Finance 下载行情，请确认网络可用或在 (?P<dir>.+) 放置名为 (?P<ticker>.+)\.csv 的历史数据文件。$",
        lambda m: (
            f"Yahoo Finance data download failed. Ensure the network is reachable or place a CSV named "
            f"{m.group('ticker')}.csv inside {m.group('dir')}."
        ),
    ),
    _rule(
        r"^关键价格字段缺失：(?P<fields>.+)。请尝试其他数据源或标的。$",
        lambda m: f"The following price fields are missing: {m.group('fields')}. Try another data source or ticker.",
    ),
    _rule(
        r"^原始区间内数据不足，已自动向前扩展至 (?P<date>[\d-]+) 以满足指标计算所需的历史长度。$",
        lambda m: (
            f"Not enough history in the selected window. The system extended the start date to {m.group('date')} "
            "to satisfy indicator requirements."
        ),
    ),
    _rule(
        r"^已根据训练缓存自动应用最优引擎：(?P<engine>.+)。可在表单取消自动应用或手动覆盖参数。$",
        lambda m: (
            f"Applied the cached optimal engine: {m.group('engine')}. Disable auto-apply or override the settings "
            "in the form if needed."
        ),
    ),
    _rule(
        r"^已应用 Optuna 搜索结果（score=(?P<score>[^,]+), trials=(?P<trials>[^)]+)）。$",
        lambda m: f"Applied Optuna search results (score={m.group('score')}, trials={m.group('trials')}).",
    ),
    _rule(
        r"^已启用 (?P<item>.+)。$",
        lambda m: f"Enabled {m.group('item')}.",
    ),
    _rule(
        r"^强化学习策略生成失败：(?P<reason>.+)$",
        lambda m: f"Failed to generate the reinforcement-learning strategy: {m.group('reason')}.",
    ),
    _rule(
        r"^组合策略中的双均线部分计算失败：(?P<reason>.+)$",
        lambda m: f"The dual moving-average component failed: {m.group('reason')}.",
    ),
    _rule(
        r"^自动超参搜索失败：(?P<reason>.+)$",
        lambda m: f"Automatic hyper-parameter search failed: {m.group('reason')}.",
    ),
    _rule(
        r"^未能获取基准 (?P<label>.+) 的行情数据，已跳过对比分析。$",
        lambda m: f"Benchmark data for {m.group('label')} is unavailable, so benchmark analysis was skipped.",
    ),
    _rule(
        r"^基准与策略的交易日无交集，基准对比已跳过。$",
        lambda _: "The benchmark and strategy have no overlapping trading days, so the comparison was skipped.",
    ),
    _rule(
        r"^scikit-learn 未安装，无法启用机器学习策略。请运行 pip install scikit-learn。$",
        lambda _: "scikit-learn is required for the ML strategy. Install it with `pip install scikit-learn`.",
    ),
    _rule(
        r"^历史样本数量不足以完成走期训练，建议延长回测区间或减少窗口。$",
        lambda _: "Historical samples are insufficient for walk-forward training. Extend the window or reduce the rolling spans.",
    ),
    _rule(
        r"^可用数据不足以计算指标，请尝试延长回测窗口或缩短均线周期。$",
        lambda _: "Not enough data to compute indicators. Extend the backtest window or shorten the moving-average periods.",
    ),
    _rule(
        r"^回测结果为空，无法生成统计指标。$",
        lambda _: "The backtest produced no rows, so statistics cannot be generated.",
    ),
    _rule(
        r"^回测结果缺少 strategy_return 列。$",
        lambda _: "The backtest result is missing the `strategy_return` column.",
    ),
    _rule(
        r"^回测结果缺少 asset_return 与 adj close 列。$",
        lambda _: "The backtest result is missing the `asset_return` and `adj close` columns.",
    ),
    _rule(
        r"^样本不足，无法构建机器学习特征矩阵。$",
        lambda _: "Samples are insufficient to build the machine-learning feature matrix.",
    ),
    _rule(
        r"^无法训练强化学习代理，可能缺少概率信号或样本不足。$",
        lambda _: "The reinforcement-learning agent cannot be trained due to missing probability signals or insufficient samples.",
    ),
    _rule(
        r"^没有可合并的策略结果。$",
        lambda _: "There are no strategy results available for aggregation.",
    ),
]


def translate_text(message: str | None, language: str | None) -> str:
    """Translate known Chinese warnings/errors into English when needed."""
    if not message:
        return ""
    lang = (language or "").lower()
    if lang.startswith("zh"):
        return message
    normalized = str(message).strip()
    for rule in RULES:
        match = rule.pattern.match(normalized)
        if match:
            return rule.render(match)
    return normalized


def translate_list(messages: Iterable[str] | None, language: str | None) -> list[str]:
    if not messages:
        return []
    return [translate_text(msg, language) for msg in messages]
