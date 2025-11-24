from __future__ import annotations

from django.conf import settings

from ..preprocessing import FeatureStore

# 共享数据/特征存储
DATA_CACHE_DIR = settings.DATA_CACHE_DIR
DATA_CACHE_DIR.mkdir(exist_ok=True)
FEATURE_STORE = FeatureStore()

__all__ = ["DATA_CACHE_DIR", "FEATURE_STORE"]
