# Quant Trading Web（当前版）

短线交易导向的 Django 平台，核心页面：
- `股市信息`（榜单 / 图表 / 新闻 / AI 摘要）
- `策略回测`（统一工作台：`trade / backtest / review`）
- `社区广场`
- `账户中心`（数据源与凭证管理）

## 1. 快速启动

```bash
cd app_bundle
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python3 -m daphne -b 127.0.0.1 -p 8000 quant_trading_site.asgi:application
```

访问：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## 2. 推荐完整运行（含后台任务）

```bash
# 终端1
redis-server

# 终端2
cd app_bundle && source ../.venv/bin/activate
celery -A quant_trading_site worker -l info -Q backtests,training,rl,paper_trading

# 终端3
cd app_bundle && source ../.venv/bin/activate
celery -A quant_trading_site beat -l info
```

## 3. 数据源（当前行为）
- 行情源：`Massive` / `Alpaca`（由用户在账户中心选择）
- 交易执行：固定 `Alpaca`
- 新闻源：`follow_data / massive / alpaca`
- 配置入口：`账户中心 -> 设置 -> API 凭证`

可配置凭证（按需）：
- Massive: `API Key`、`S3 Access Key ID`、`S3 Secret Access Key`
- Alpaca: `API Key ID`、`API Secret Key`

## 4. 常用环境变量（最小集）

```bash
DJANGO_SECRET_KEY=change-me
DJANGO_DEBUG=1
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
DJANGO_CSRF_TRUSTED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000

# 可选：缓存/队列
REDIS_URL=redis://127.0.0.1:6379/0
CELERY_BROKER_URL=redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0

# 可选：默认源后备（不覆盖账户中心中的用户选择）
MARKET_DATA_PROVIDER=alpaca
MARKET_NEWS_PROVIDER=follow_data

# 可选：Massive
MASSIVE_API_KEY=
MASSIVE_REST_URL=https://api.polygon.io
MASSIVE_WS_URL=wss://socket.polygon.io/stocks

# 可选：Alpaca
ALPACA_API_KEY_ID=
ALPACA_API_SECRET_KEY=
```

## 5. 常用开发检查

```bash
cd app_bundle
source ../.venv/bin/activate
ruff check .
python -m pytest -q
python manage.py check
```

## 6. 常用入口
- `/market/`
- `/backtest/`
- `/community/`
- `/account/`
- `/api/v1/strategy/workbench/`
- `/api/market/rankings/status/`

> 说明：`/realtime/monitor/` 与 `/realtime/settings/` 已重定向到 `/backtest/?workspace=trade`。
