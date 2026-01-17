"""
Feature-area view package for the trading app.

Having a package allows us to keep related views together while exposing the
same public API that ``urls.py`` expects (e.g. ``from . import views``).
"""

from .dashboard import backtest
from .api import (
    ai_chat,
    ai_chat_stream,
    ai_task_status,
    enqueue_ai_task,
    build_portfolio_api,
    backtest_task_status,
    enqueue_backtest_task,
    enqueue_rl_task,
    enqueue_training_task,
    rag_ingest,
    rag_query_api,
    rl_task_status,
    screener_snapshot_api,
    training_task_status,
)
from .history import delete_history, update_history_meta
from .reports import export_report
from .account import (
    account,
    update_profile,
    update_api_credentials,
    clear_api_credentials,
    reset_password,
    delete_avatar,
    remove_feature,
    remove_gallery,
)
from .auth import activate, resend_activation, signup
from .market import market_insights, market_insights_data
from .community import community, write_post, community_like, community_comment, community_delete
from .account import profile_public
from .learning import learning_center, learning_module_detail
from .observability_dashboard import observability_dashboard
from .history_compare import history_compare
from .paper import paper_trading
from .screen_analyzer import (
    screen_analyzer,
    screen_analyzer_api,
    screen_analyzer_sample_api,
    screen_analyzer_train_api,
)
from .realtime import realtime_settings, realtime_monitor

__all__ = [
    "backtest",
    "paper_trading",
    "screener_snapshot_api",
    "ai_chat",
    "ai_chat_stream",
    "enqueue_ai_task",
    "ai_task_status",
    "rag_ingest",
    "rag_query_api",
    "build_portfolio_api",
    "enqueue_backtest_task",
    "backtest_task_status",
    "enqueue_training_task",
    "training_task_status",
    "enqueue_rl_task",
    "rl_task_status",
    "delete_history",
    "update_history_meta",
    "export_report",
    "signup",
    "activate",
    "resend_activation",
    "account",
    "update_profile",
    "update_api_credentials",
    "clear_api_credentials",
    "reset_password",
    "delete_avatar",
    "remove_feature",
    "remove_gallery",
    "market_insights",
    "market_insights_data",
    "community",
    "write_post",
    "community_like",
    "community_comment",
    "community_delete",
    "profile_public",
    "learning_center",
    "learning_module_detail",
    "observability_dashboard",
    "history_compare",
    "screen_analyzer",
    "screen_analyzer_api",
    "screen_analyzer_sample_api",
    "screen_analyzer_train_api",
    "realtime_settings",
    "realtime_monitor",
]
