from __future__ import annotations

from typing import Any
import numpy as np

try:  # Prefer gymnasium, fallback到经典 gym
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore
        spaces = None  # type: ignore

IS_GYMNASIUM = bool(gym and getattr(gym, "__name__", "").startswith("gymnasium"))

SIGNAL_MAP = {
    0: -1.0,
    1: 0.0,
    2: 1.0,
}


class TradingHistoryEnv:
    """
    最简化的历史行情环境，供 Stable Baselines / FinRL 训练使用。

    - observation: [prob, sma_diff, rsi, volatility, momentum, current_position]
    - action: 0=做空，1=观望，2=做多
    - reward: 前一时刻仓位 * 当期资产收益 - 调仓成本
    """

    metadata: dict[str, Any] = {}

    def __init__(self, backtest, cost_rate: float) -> None:
        if gym is None or spaces is None:  # pragma: no cover - 运行时检查
            raise RuntimeError("需要安装 gym / gymnasium 才能启用强化学习环境。")
        self.df = backtest.copy()
        self.cost_rate = float(cost_rate)
        self.index = self.df.index
        self.length = len(self.df)
        if self.length < 3:
            raise ValueError("样本不足，无法构建 RL 环境。")
        self.pointer = 1
        self.position = 0.0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        self.pointer = 1
        self.position = 0.0
        obs = self._build_obs(self.pointer - 1)
        if IS_GYMNASIUM:
            return obs.astype(np.float32), {}
        return obs.astype(np.float32)

    def step(self, action: int):
        action = int(action)
        prev_pos = self.position
        self.position = SIGNAL_MAP.get(action, 0.0)
        ret = float(self.df["asset_return"].iloc[self.pointer])
        reward = prev_pos * ret - abs(self.position - prev_pos) * self.cost_rate
        self.pointer += 1
        done = self.pointer >= self.length - 1
        obs = self._build_obs(min(self.pointer - 1, self.length - 1)).astype(np.float32)
        info: dict[str, Any] = {}
        if IS_GYMNASIUM:
            return obs, reward, done, False, info
        return obs, reward, done, info

    def action_to_signal(self, action: int) -> float:
        return SIGNAL_MAP.get(int(action), 0.0)

    def _build_obs(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        prob = float(row.get("probability", 0.5))
        sma_short = float(row.get("sma_short", 0.0))
        sma_long = float(row.get("sma_long", 1.0))
        sma_diff = (sma_short - sma_long) / (abs(sma_long) + 1e-6)
        rsi = float(row.get("rsi", 50.0) / 100.0)
        vol = float(row.get("volatility", 0.0))
        momentum = float(row.get("momentum_short", 0.0))
        return np.array(
            [
                prob,
                sma_diff,
                rsi,
                vol,
                momentum,
                float(self.position),
            ],
            dtype=np.float32,
        )
