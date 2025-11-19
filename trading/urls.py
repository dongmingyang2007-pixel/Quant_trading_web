from django.urls import path
from django.views.generic import RedirectView

from . import views
from .api import views_v1 as api_v1


app_name = "trading"

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="trading:backtest", permanent=False)),
    path("backtest/", views.backtest, name="backtest"),
    path("api/screener/", views.screener_snapshot_api, name="screener_snapshot"),
    path("api/ai_chat/", views.ai_chat, name="ai_chat"),
    path("api/ai_chat/stream/", views.ai_chat_stream, name="ai_chat_stream"),
    path("api/backtest/task/", views.enqueue_backtest_task, name="enqueue_backtest_task"),
    path("api/backtest/task/<str:task_id>/", views.backtest_task_status, name="backtest_task_status"),
    path("api/v1/backtests/tasks/", api_v1.BacktestTaskView.as_view(), name="api_v1_backtest_tasks"),
    path("api/v1/training/tasks/", api_v1.TrainingTaskView.as_view(), name="api_v1_training_tasks"),
    path("api/v1/rl/tasks/", api_v1.RLTaskView.as_view(), name="api_v1_rl_tasks"),
    path("api/v1/tasks/<str:task_id>/", api_v1.TaskStatusView.as_view(), name="api_v1_task_status"),
    path("api/v1/screener/", api_v1.ScreenerSnapshotView.as_view(), name="api_v1_screener"),
    path("api/training/task/", views.enqueue_training_task, name="enqueue_training_task"),
    path("api/training/task/<str:task_id>/", views.training_task_status, name="training_task_status"),
    path("api/rl/task/", views.enqueue_rl_task, name="enqueue_rl_task"),
    path("api/rl/task/<str:task_id>/", views.rl_task_status, name="rl_task_status"),
    path("api/market/", views.market_insights_data, name="market_insights_data"),
    path("history/delete/<str:record_id>/", views.delete_history, name="delete_history"),
    path("export/report/", views.export_report, name="export_report"),
    path("accounts/signup/", views.signup, name="signup"),
    path("accounts/activate/<str:uidb64>/<str:token>/", views.activate, name="activate"),
    path("accounts/resend-activation/", views.resend_activation, name="resend_activation"),
    path("account/", views.account, name="account"),
    path("market/", views.market_insights, name="market_insights"),
    path("learning/", views.learning_center, name="learning_center"),
    path("learning/<slug:slug>/", views.learning_module_detail, name="learning_detail"),
    path("community/", views.community, name="community"),
    path("community/like/", views.community_like, name="community_like"),
    path("community/comment/", views.community_comment, name="community_comment"),
    path("community/delete/", views.community_delete, name="community_delete"),
    path("profile/<uuid:profile_slug>/", views.profile_public, name="profile_public"),
    path("observability/", views.observability_dashboard, name="observability"),
]
