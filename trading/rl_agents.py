from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from .reinforcement import ValueIterationAgent, ACTION_TO_SIGNAL
from .rl_env import TradingHistoryEnv

try:  # Optional SB3 backend
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
except Exception:  # pragma: no cover - optional
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore

try:  # Optional FinRL wrapper
    from finrl.agents.stablebaselines3.models import DRLAgent  # type: ignore
except Exception:  # pragma: no cover - optional
    DRLAgent = None  # type: ignore


@dataclass(slots=True)
class SB3Config:
    timesteps: int = 20000
    policy: str = "MlpPolicy"
    gamma: float = 0.96
    learning_rate: float = 3e-4
    clip_range: float = 0.2


class SB3PPOAgent:
    def __init__(self, cost_rate: float, config: SB3Config | None = None) -> None:
        if PPO is None or DummyVecEnv is None:
            raise RuntimeError("需要安装 stable-baselines3 才能启用 FinRL/SB3 代理。请运行 pip install stable-baselines3 gymnasium。")
        self.cost_rate = cost_rate
        self.config = config or SB3Config()
        self.model: PPO | None = None
        self._last_backtest: pd.DataFrame | None = None

    def fit_from_backtest(self, backtest: pd.DataFrame) -> bool:
        env = DummyVecEnv([lambda: TradingHistoryEnv(backtest, self.cost_rate)])
        self.model = PPO(
            self.config.policy,
            env,
            verbose=0,
            gamma=self.config.gamma,
            learning_rate=self.config.learning_rate,
            clip_range=self.config.clip_range,
        )
        self.model.learn(total_timesteps=int(self.config.timesteps))
        self._last_backtest = backtest
        return True

    def signal_series(self, backtest: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("请先调用 fit_from_backtest 再生成信号。")
        env = TradingHistoryEnv(backtest, self.cost_rate)
        obs = env.reset()
        signals: list[float] = []
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            signal = ACTION_TO_SIGNAL.get(int(action), 0.0)
            signals.append(signal)
            obs, _, done, _ = env.step(int(action))
        if len(signals) < len(backtest):
            signals.extend([signals[-1] if signals else 0.0] * (len(backtest) - len(signals)))
        return pd.Series(signals[: len(backtest)], index=backtest.index)

    @property
    def playbook(self) -> dict[str, Any]:
        return {
            "available": True,
            "policies": [],
            "edge": "0.00%",
            "insight": "SB3 代理已训练完成，可在结果中查看强化学习策略表现。",
        }


class FinRLPPOAgent(SB3PPOAgent):
    def __init__(self, cost_rate: float, config: SB3Config | None = None) -> None:
        if DRLAgent is None:
            raise RuntimeError("需要安装 FinRL（pip install finrl）才能使用 FinRL 代理。")
        super().__init__(cost_rate, config)

    def fit_from_backtest(self, backtest: pd.DataFrame) -> bool:
        from functools import partial

        vec_env = DummyVecEnv([partial(TradingHistoryEnv, backtest, self.cost_rate)])
        agent = DRLAgent(env=vec_env)
        model = agent.get_model(
            "ppo",
            policy=self.config.policy,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            clip_range=self.config.clip_range,
            verbose=0,
        )
        self.model = agent.train_model(
            model=model,
            total_timesteps=int(self.config.timesteps),
        )
        return True


def build_rl_agent(engine: str, cost_rate: float, rl_params: dict[str, Any] | None = None):
    engine = (engine or "value_iter").lower()
    if engine in {"sb3", "ppo"}:
        cfg = SB3Config(
            timesteps=int(rl_params.get("timesteps", 20000)) if rl_params else 20000,
            policy=str(rl_params.get("policy", "MlpPolicy")) if rl_params else "MlpPolicy",
            gamma=float(rl_params.get("gamma", 0.96)) if rl_params else 0.96,
            learning_rate=float(rl_params.get("learning_rate", 3e-4)) if rl_params else 3e-4,
            clip_range=float(rl_params.get("clip_range", 0.2)) if rl_params else 0.2,
        )
        return SB3PPOAgent(cost_rate=cost_rate, config=cfg)
    if engine == "finrl":
        cfg = SB3Config(
            timesteps=int(rl_params.get("timesteps", 20000)) if rl_params else 20000,
            policy=str(rl_params.get("policy", "MlpPolicy")) if rl_params else "MlpPolicy",
            gamma=float(rl_params.get("gamma", 0.96)) if rl_params else 0.96,
            learning_rate=float(rl_params.get("learning_rate", 3e-4)) if rl_params else 3e-4,
            clip_range=float(rl_params.get("clip_range", 0.2)) if rl_params else 0.2,
        )
        return FinRLPPOAgent(cost_rate=cost_rate, config=cfg)
    return ValueIterationAgent(cost_rate=cost_rate)
