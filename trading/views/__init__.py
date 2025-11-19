"""
Feature-area view package for the trading app.

Having a package allows us to keep related views together while exposing the
same public API that ``urls.py`` expects (e.g. ``from . import views``).
"""

from .dashboard import backtest
from .api import (
    ai_chat,
    ai_chat_stream,
    backtest_task_status,
    enqueue_backtest_task,
    enqueue_rl_task,
    enqueue_training_task,
    rl_task_status,
    screener_snapshot_api,
    training_task_status,
)
from .history import delete_history
from .reports import export_report
from .account import account
from .auth import activate, resend_activation, signup
from .market import market_insights, market_insights_data
from .community import community, community_like, community_comment, community_delete
from .account import profile_public
from .learning import learning_center, learning_module_detail
from .observability_dashboard import observability_dashboard

__all__ = [
    "backtest",
    "screener_snapshot_api",
    "ai_chat",
    "ai_chat_stream",
    "enqueue_backtest_task",
    "backtest_task_status",
    "enqueue_training_task",
    "training_task_status",
    "enqueue_rl_task",
    "rl_task_status",
    "delete_history",
    "export_report",
    "signup",
    "activate",
    "resend_activation",
    "account",
    "market_insights",
    "market_insights_data",
    "community",
    "community_like",
    "community_comment",
    "community_delete",
    "profile_public",
    "learning_center",
    "learning_module_detail",
    "observability_dashboard",
]
