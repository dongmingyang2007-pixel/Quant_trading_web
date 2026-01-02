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
