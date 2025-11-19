from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np
import pandas as pd

ACTION_LABELS = {
    "long": "加仓做多",
    "flat": "轻仓观望",
    "short": "对冲/做空",
}

ACTION_TO_SIGNAL = {
    "long": 1.0,
    "flat": 0.0,
    "short": -1.0,
}


def _bucket_probability(value: float) -> str:
    if value >= 0.65:
        return "高概率"
    if value <= 0.35:
        return "低概率"
    return "中性"


def _bucket_trend(spread: float) -> str:
    if spread >= 0.01:
        return "上行趋势"
    if spread <= -0.01:
        return "下行趋势"
    return "震荡"


def _format_pct(value: float) -> str:
    return f"{value*100:.2f}%"


class ValueIterationAgent:
    """轻量级 RL 代理，可作为 FinRL/Stable Baselines 接入前的默认实现。"""

    def __init__(self, cost_rate: float, gamma: float = 0.92, alpha: float = 0.35) -> None:
        self.cost_rate = cost_rate
        self.gamma = gamma
        self.alpha = alpha
        self.actions = ("long", "flat", "short")
        self.q_table: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(self.actions)))
        self._policies: list[dict[str, Any]] = []

    def _extract_series(self, backtest: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series] | None:
        if "probability" not in backtest.columns:
            return None
        prob = backtest["probability"].fillna(0.5)
        if "sma_short" in backtest.columns and "sma_long" in backtest.columns:
            trend = (backtest["sma_short"] - backtest["sma_long"]).fillna(0.0)
        else:
            trend = backtest.get("position", pd.Series(0.0, index=prob.index))
        asset_ret = backtest.get("asset_return")
        if asset_ret is None or asset_ret.dropna().empty:
            asset_ret = backtest.get("strategy_return")
        if asset_ret is None:
            return None
        asset_ret = asset_ret.fillna(0.0)
        if len(asset_ret.dropna()) < 30:
            return None
        return prob, trend, asset_ret

    def fit_from_backtest(self, backtest: pd.DataFrame) -> bool:
        series = self._extract_series(backtest)
        if series is None:
            return False
        prob, trend, asset_ret = series
        transitions: list[tuple[str, str, float]] = []
        for idx in range(len(prob) - 1):
            current_state = f"{_bucket_probability(float(prob.iloc[idx]))}|{_bucket_trend(float(trend.iloc[idx]))}"
            next_state = f"{_bucket_probability(float(prob.iloc[idx + 1]))}|{_bucket_trend(float(trend.iloc[idx + 1]))}"
            reward = float(asset_ret.iloc[idx + 1])
            transitions.append((current_state, next_state, reward))
        if not transitions:
            return False
        for state, next_state, reward in reversed(transitions):
            best_next = np.max(self.q_table[next_state]) if next_state in self.q_table else 0.0
            rewards = np.array(
                [
                    reward - self.cost_rate,
                    -abs(self.cost_rate) * 0.2,
                    -reward - self.cost_rate,
                ],
                dtype=float,
            )
            updates = rewards + self.gamma * best_next
            self.q_table[state] = (1 - self.alpha) * self.q_table[state] + self.alpha * updates
        self._policies = self._build_policies()
        return bool(self._policies)

    def _build_policies(self) -> list[dict[str, Any]]:
        scored: list[tuple[str, float, float]] = []
        for state, values in self.q_table.items():
            best_idx = int(np.argmax(values))
            second = float(np.partition(values, -2)[-2]) if len(values) > 1 else 0.0
            scored.append((state, float(values[best_idx]), second))
        scored.sort(key=lambda item: item[1], reverse=True)
        top_states = scored[:4]
        policies: list[dict[str, Any]] = []
        for state, best_value, second_value in top_states:
            prob_label, trend_label = state.split("|", 1)
            action_idx = int(np.argmax(self.q_table[state]))
            action = self.actions[action_idx]
            policies.append(
                {
                    "state": f"{prob_label} · {trend_label}",
                    "action": ACTION_LABELS[action],
                    "edge": _format_pct(best_value),
                    "edge_value": best_value,
                    "rationale": f"相较其他动作提升 {(best_value - second_value)*100:.1f} 个基点/日",
                }
            )
        return policies

    @property
    def playbook(self) -> dict[str, Any]:
        if not self._policies:
            return {"available": False, "message": "暂未形成明确的强化学习偏好。"}
        avg_edge = float(np.mean([item.get("edge_value", 0.0) for item in self._policies]))
        insight = (
            "当概率与趋势同向时，强化学习代理倾向放大多头；若概率走弱则建议迅速降低曝险。"
            if self._policies
            else "暂未形成明确的强化学习偏好。"
        )
        return {
            "available": True,
            "policies": self._policies,
            "edge": _format_pct(avg_edge),
            "insight": insight,
        }

    def best_action(self, state: str) -> str:
        values = self.q_table.get(state)
        if values is None or not len(values):
            return "flat"
        return self.actions[int(np.argmax(values))]

    def signal_series(self, backtest: pd.DataFrame) -> pd.Series:
        prob = backtest.get("probability", pd.Series(0.5, index=backtest.index)).fillna(0.5)
        if "sma_short" in backtest.columns and "sma_long" in backtest.columns:
            trend = (backtest["sma_short"] - backtest["sma_long"]).fillna(0.0)
        else:
            trend = backtest.get("position", pd.Series(0.0, index=backtest.index))
        states = [
            f"{_bucket_probability(float(prob.iloc[i]))}|{_bucket_trend(float(trend.iloc[i]))}"
            for i in range(len(prob))
        ]
        actions = [self.best_action(state) for state in states]
        signals = [ACTION_TO_SIGNAL.get(act, 0.0) for act in actions]
        return pd.Series(signals, index=prob.index, dtype=float)


def train_value_agent(backtest: pd.DataFrame, cost_rate: float) -> ValueIterationAgent | None:
    agent = ValueIterationAgent(cost_rate=cost_rate)
    ok = agent.fit_from_backtest(backtest)
    return agent if ok else None


def build_reinforcement_playbook(backtest: pd.DataFrame, cost_rate: float) -> dict[str, Any]:
    agent = train_value_agent(backtest, cost_rate)
    if agent is None:
        return {"available": False, "message": "样本不足或缺少概率信号，暂无法生成强化学习提示。"}
    return agent.playbook
