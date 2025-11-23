# Quant Trading Web / 量化交易模拟与模拟实盘一体化

结合经典动量/强化学习策略、AI 解读、社区与学习中心的 Django 应用。支持回测、Optuna 调参、RL 战术盘、模拟实盘（Paper Trading）以及本地 LLM 洞察。

## 目录结构
- `app_bundle/`：源代码、模板、依赖、部署文件（日常开发与部署用）。  
- `storage_bundle/`：数据与大文件（`db.sqlite3`、`data_cache/`、`media/`、`staticfiles/` 等）。  
- 顶层 `manage.py` / `Dockerfile` / `Procfile` 会代理到 `app_bundle/`，命令可在仓库根目录执行。

## 快速开始
```bash
cd quant_trading_web
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
浏览器访问 `http://127.0.0.1:8000/`。勾选“启用AI市场洞察”可调用本地 Ollama 模型（默认 deepseek-r1）。

## 异步队列 / Celery
- 同步模式（省事）：`CELERY_TASK_ALWAYS_EAGER=True`，无 Redis/队列。
- 异步模式（推荐并发/长任务）：  
  ```bash
  # worker：含回测/训练/RL/模拟盘队列
  PYTHONPATH=app_bundle celery -A quant_trading_site worker -l info -Q backtests,training,rl,paper_trading
  # beat：调度定时任务（含模拟盘心跳）
  PYTHONPATH=app_bundle celery -A quant_trading_site beat -l info
  ```
  环境变量示例：`CELERY_BROKER_URL=redis://localhost:6379/0`，`CELERY_RESULT_BACKEND` 同步设置。

## 模拟实盘（Paper Trading）
- API：`/api/v1/paper/sessions/` 创建/列出；`/api/v1/paper/sessions/<uuid>/` 查询、暂停、恢复、停止。
- 逻辑：复用策略引擎获取目标仓位 → 拉取最新行情 → 虚拟账户调仓 → 写入权益曲线与成交记录。
- 前端：账户中心新增“模拟实盘”页签，可创建会话、查看持仓/权益/成交（`trading/static/trading/js/paper_trading.js`）。
- 心跳：`trading.tasks.run_paper_trading_heartbeat` 默认由 Celery beat 按 `PAPER_TRADING_INTERVAL_SECONDS` 触发。

## 数据存储
- 可用 `DJANGO_STORAGE_DIR` 自定义读写目录；默认优先 `storage_bundle/`，不存在时自动创建 `app_bundle/storage_bundle/`。
- 确保目录可写；首次部署运行 `python manage.py migrate` 初始化 DB。

## 旧数据导入
- 一键导入社区/回测 JSON：`python manage.py import_legacy_cache --base-dir storage_bundle/data_cache`（可用 `--profiles/--community/--backtests` 拆分）。
- 专用命令：`migrate_community_json`、`migrate_backtests_json` 以更细粒度导入。

## 主要特性概览
- 回测：双均线+RSI、波动率目标、交易成本/滑点、风险指标（Sharpe、Sortino、VaR/CVaR、跟踪误差等）。
- 机器学习/深度序列：GBDT、LightGBM、CatBoost、LSTM、Transformer、Fusion 自动择优；Optuna + Walk-Forward 调参。
- 强化学习：战术盘与 RL 策略引擎（Value Iteration/FinRL/SB3）。
- 组合策略：多引擎加权、净值/指标对比、历史记录管理与导出。
- 宏观/资金流/基本面数据面板、学习中心与社区广场。
- LLM 洞察：DeepSeek + Qwen 协同，支持联网检索与思考过程输出。

## 批量/自动化示例
- 批量回测：`trading.batch.run_batch_backtests`。
- Optuna 搜索：`trading.optimization.run_optuna_search`（生成 `data_cache/training/hyperopt_*.json`）。

## 其它提示
- 默认 DEBUG 关闭，请在生产设置 `DJANGO_SECRET_KEY`（未设置将报错，可用 `DJANGO_ALLOW_INSECURE_KEY=1` 临时绕过）。
- LLM：安装 Ollama 并拉取模型（`deepseek-r1:8b`、`qwen3:8b`），设置 `OLLAMA_ENDPOINT`/`OLLAMA_MODEL` 等环境变量。
- 媒体/缓存清理：旧版市场快照可通过 `python manage.py clean_market_cache --apply` 清理；遥测文件 `telemetry.ndjson` 带尺寸轮转，可用 `METRICS_MAX_BYTES` 配置。
