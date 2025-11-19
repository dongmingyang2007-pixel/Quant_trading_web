from __future__ import annotations

from rest_framework.throttling import UserRateThrottle


class TaskBurstThrottle(UserRateThrottle):
    scope = "api_task"
