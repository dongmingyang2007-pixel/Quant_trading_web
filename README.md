# Quant Trading Web / 量化交易模拟与实时分析平台

基于 Django 的量化交易研究与模拟实盘平台，包含回测、研究指标、策略管线、屏幕级图表分析与 LLM 洞察等能力。默认离线运行，可选接入外部服务（如 Redis、OCR、Ollama）。

## 目录结构
- `quant_trading_site/`：Django 项目配置与入口。
- `trading/`：核心业务（策略、回测、屏幕解析、LLM 等）。
- `paper/`：模拟实盘模块。
- `templates/`：页面模板。
- `storage_bundle/`：数据与运行产物（`db.sqlite3`、`data_cache/`、`media/`、`staticfiles/`，首次运行自动生成）。
- 根目录工具链：`Dockerfile`、`Procfile`、`Makefile`、`requirements-dev.txt`、`pyproject.toml`、`.github/workflows/ci.yml`。

## 快速开始（本地）
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 可选：开发工具
pip install -r requirements-dev.txt

python manage.py migrate
python manage.py runserver
```
访问 `http://127.0.0.1:8000/`。

可选创建管理员：
```bash
python manage.py createsuperuser
```

## 满血版本地启动（WebSocket / Redis / 实时引擎）
> 适用于需要 Market Insights WebSocket、实时引擎与队列能力的完整体验。

1) 安装完整依赖（包含 channels-redis）：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) 配置 `.env`（位置：项目根目录，与 `app_bundle/` 同级）：
```bash
DJANGO_DEBUG=1
DJANGO_SECRET_KEY=dev-local-change-me
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
DJANGO_CSRF_TRUSTED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000

REDIS_URL=redis://127.0.0.1:6379/0
CELERY_BROKER_URL=redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0
```

3) 启动 Redis（新终端）：
```bash
redis-server
```

4) 迁移数据库：
```bash
cd app_bundle
python manage.py migrate
```

5) 用 ASGI 启动服务（支持 WebSocket）：
```bash
python3 -m daphne -b 127.0.0.1 -p 8000 quant_trading_site.asgi:application
```

6) 可选：启动 Celery（新终端）：
```bash
celery -A quant_trading_site worker -l info -Q backtests,training,rl,paper_trading
celery -A quant_trading_site beat -l info
```

7) 可选：启动实时引擎（新终端）：
```bash
python manage.py realtime_run --user-id <你的用户ID>
python manage.py realtime_refresh_assets --user-id <你的用户ID>
```

8) 打开页面 `http://127.0.0.1:8000/`，在「账户中心 → 设置 → API 凭证」填入 Alpaca Key。

## 依赖分层（core / optional）
默认使用完整依赖：
```bash
pip install -r requirements.txt
```
轻量安装（仅核心能力）：
```bash
pip install -r requirements-core.txt
```
按需启用：
```bash
pip install -r requirements-ml.txt   # 机器学习相关
pip install -r requirements-rl.txt   # 强化学习相关
```

## 配置与环境变量
复制 `.env.example` 为 `.env`，按需修改；生产部署建议参考 `deploy/README.md`。

生产配置最小集合：
- `DJANGO_SECRET_KEY`：生产环境必须设置。
- `DJANGO_DEBUG=0`
- `DJANGO_ALLOWED_HOSTS`：逗号分隔域名/IP。
- `DJANGO_CSRF_TRUSTED_ORIGINS`：例如 `https://yourdomain.com`。

可选配置：
- `DJANGO_STORAGE_DIR`：自定义数据目录；默认使用 `storage_bundle/`，不存在时自动创建。
- `DJANGO_ALLOW_INSECURE_KEY=1`：仅限本地临时绕过密钥校验。
- `REDIS_URL`：启用 Redis cache（含限流跨进程一致）。
- `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND`：异步队列配置。
- `CELERY_ALWAYS_EAGER`：置为 `1` 可强制同步执行。
- `TASK_RETURN_SNAPSHOT=1`：仅在需要时让回测任务返回完整快照。
- `OLLAMA_ENDPOINT` / `OLLAMA_MODEL` / `AI_PROVIDER` / `GEMINI_API_KEY`：LLM 洞察与外部模型（可选）。
- `METRICS_MAX_BYTES`：遥测文件大小轮转阈值。

## 异步任务（Celery / Local runner）
推荐（生产或长回测）：启用 Redis + Celery。
```bash
celery -A quant_trading_site worker -l info -Q backtests,training,rl,paper_trading
celery -A quant_trading_site beat -l info
```
Celery beat 的调度文件默认写入 `storage_bundle/`。

本地开发（未配置 Redis 时）：回测会自动使用本地线程执行器（Local runner）。
- 任务状态写入数据库，刷新/跳转后仍可查询。
- 限制：仅限当前进程；服务重启会中断；不支持跨进程分发；并发由 `LOCAL_TASK_WORKERS` 控制（默认 2）。

如需强制同步（调试用，不建议长回测）：`CELERY_ALWAYS_EAGER=1`。

## 实时引擎（Realtime）
用于生成 Universe/Focus 列表、实时 bars 与信号，并通过监控页查看状态。

1) 在「账户中心 → 设置 → API 凭证」中填写 Alpaca Key（或通过环境变量注入）。  
2) 打开「实时引擎」页面创建/激活配置档案。  
3) 启动引擎：
```bash
python manage.py realtime_run --user-id <你的用户ID>
```
可选刷新资产主表（提升 Universe 覆盖）：
```bash
python manage.py realtime_refresh_assets --user-id <你的用户ID>
```

示例配置（Realtime Profile JSON）：
```json
{
  "universe": {
    "max_symbols": 1200,
    "top_n": 1000,
    "min_price": 2,
    "min_dollar_volume": 5000000,
    "min_volume": 200000
  },
  "focus": {
    "size": 200,
    "max_churn_per_refresh": 20,
    "min_residence_seconds": 300
  },
  "engine": {
    "stream_enabled": true,
    "feed": "sip",
    "bar_interval_seconds": 1,
    "bar_aggregate_seconds": 5,
    "stale_seconds": 2.5
  },
  "signals": {
    "lookback_bars": 3,
    "entry_threshold": 0.002,
    "max_spread_bps": 25,
    "min_volume": 10000
  }
}
```

常见问题：
- 监控页提示离线：检查 realtime_run 是否在独立终端运行、`stream_state.json` 是否更新。
- 无行情/信号：确认 Alpaca Key 生效；`feed` 与订阅类型是否与账户权限匹配。
- SIP 空数据：确认订阅绑定在当前 Key；测试 `feed=iex` 以排查权限问题。

## 实时引擎相关环境变量
- `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY`：Alpaca 行情/交易凭证。
- `ALPACA_DATA_FEED`：`sip` 或 `iex`，默认 `sip`。
- `ALPACA_DATA_WS_URL`：Alpaca WS 地址（默认 `wss://stream.data.alpaca.markets/v2/sip`）。
- `ALPACA_DATA_REST_URL`：Alpaca 数据 REST 根地址（默认 `https://data.alpaca.markets`）。
- `ALPACA_TRADING_REST_URL`：Alpaca 交易 REST 根地址（默认 `https://paper-api.alpaca.markets`）。
- `REALTIME_STATE_DIR` / `REALTIME_DATA_DIR`：实时引擎状态/数据目录（默认在 `storage_bundle/data_cache/realtime/`）。
- `REALTIME_NDJSON_MAX_BYTES`：NDJSON 文件轮转阈值（默认 10MB）。
- `METRICS_MAX_BYTES`：遥测文件轮转阈值。

## 屏幕图表分析（可选）
浏览器页面提供“屏幕波型分析”，使用本地屏幕捕获与图形解析。
- OCR 依赖 Tesseract（macOS: `brew install tesseract`）。
- 在页面中开启/关闭 OCR，避免不必要的资源消耗。

## 数据导入与清理
导入历史数据：
```bash
python manage.py import_legacy_cache --base-dir storage_bundle/data_cache
```
清理市场快照缓存：
```bash
python manage.py clean_market_cache --apply
```

## 测试与格式化
```bash
ruff check .
pytest -q
```
可选类型检查：
```bash
mypy .
```
可选：若要运行 WebSocket 集成测试，可设置 `ALPACA_TEST_WS_URL` 指向 FAKEPACA/本地 WS 服务。
如需安装 pre-commit：
```bash
pre-commit install
```

## Docker
```bash
docker build -t quant_trading_web .
docker run -p 8000:8000 \
  -e DJANGO_SECRET_KEY=change-me \
  -v $(pwd)/storage_bundle:/app/storage_bundle \
  quant_trading_web
```

## 备注
- 默认 Python 3.11（CI 与 Dockerfile 已统一）。
- `storage_bundle/` 作为运行数据目录（`db.sqlite3`、`data_cache/`、`media/`、`staticfiles/`），请确保可写并在 Docker 中做 volume 挂载。
