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

## 配置与环境变量
常用环境变量（按需设置）：
- `DJANGO_STORAGE_DIR`：自定义数据目录；默认使用 `storage_bundle/`，不存在时自动创建 `storage_bundle/`。
- `DJANGO_DEBUG`：调试开关（默认测试时开启）。
- `DJANGO_SECRET_KEY`：生产环境必须设置。
- `DJANGO_ALLOW_INSECURE_KEY=1`：仅限本地临时绕过密钥校验。
- `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND`：异步队列配置。
- `CELERY_ALWAYS_EAGER`：置为 `1` 可强制同步执行。
- `METRICS_MAX_BYTES`：遥测文件大小轮转阈值。
- `OLLAMA_ENDPOINT` / `OLLAMA_MODEL`：本地 LLM 洞察（可选）。

## 异步任务（Celery）
同步模式（默认）：
- `CELERY_ALWAYS_EAGER=1` 或不配置 `CELERY_BROKER_URL`。

异步模式（需要 Redis）：
```bash
celery -A quant_trading_site worker -l info -Q backtests,training,rl,paper_trading
celery -A quant_trading_site beat -l info
```
Celery beat 的调度文件默认写入 `storage_bundle/`。

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
- `storage_bundle/` 作为运行数据目录，请确保可写。
