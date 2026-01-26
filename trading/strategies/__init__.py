"""
Strategy package entrypoint.

模块已拆分为配置/核心逻辑/指标等子模块，便于后续按策略类型继续拆分（如均线策略、ML 策略）。
公共 API 通过此处集中导出，保持向后兼容。
"""

from .config import *  # noqa: F401,F403
from .indicators import *  # noqa: F401,F403
from .risk import *  # noqa: F401,F403
from .execution import *  # noqa: F401,F403
from .ma_cross import *  # noqa: F401,F403
from .mean_reversion import *  # noqa: F401,F403
from .ml_engine import *  # noqa: F401,F403
from .charts import *  # noqa: F401,F403
from .pipeline import *  # noqa: F401,F403
from .insights import *  # noqa: F401,F403
from .market import *  # noqa: F401,F403
from .core import *  # noqa: F401,F403
from .core import _tb_summary_from_dataset  # noqa: F401

__all__ = [name for name in globals().keys() if not name.startswith("_")]
