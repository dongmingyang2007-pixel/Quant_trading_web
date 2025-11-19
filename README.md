# Quant Trading Web Demo

一个结合经典量化策略与本地 LLM 洞察的 Django 示例应用。核心功能包括：

- 通过 `yfinance` 下载标的历史行情（可从 `data_cache/` 导入离线 CSV）；
- 计算双均线 + RSI 指标并回测，叠加波动率目标管理（Volatility Targeting）与仓位控制；
- 输出收益、Sharpe、Sortino、Calmar、命中率、VaR/CVaR 等专业风险收益指标，并可与基准指数对比 α、β、信息比率、跟踪误差；
- 自动生成价格 / 净值 / 回撤 / 收益分布图，支持中文界面；
- 可选调用本机 Ollama `deepseek-r1:8b` 模型，展示“思考过程 + 结论”的中文量化解读。
- 新增“机器学习动量 + 风控”策略引擎：采用多因子特征 + 梯度提升模型的滚动训练、交易成本/滑点建模、波动率目标杠杆与最短持仓约束，输出专业级风险指标。
- 针对非专业用户设计的“执行清单 + 风险提醒 + 补充知识”看板，并支持按照投资期限、经验水平、核心目标定制建议。
- 组合策略模式一次对比“机器学习动量”与“传统双均线”，并支持一键导出 JSON 报告。
- 本地数据管线与特征商店支持多标的自动缓存、特征快照与批量训练配置管理（含 MLflow 记录）。
- 宏观 / 基本面 / 资金流数据面板：自动抓取 VIX、利率、美元、金油、信用、BTC 等宏观指标，以及 SPY/QQQ/TLT/HYG 等代表性 ETF 的动量与成交资金，并同步展示最新财报快照、事件雷达。
- 全新数据质量管线：自动去重、填补缺口、平滑极端价格跳变，并生成 `price_z` 标准化特征，同时把特征矩阵缓存到 `data_cache/feature_store/`，重复回测可直接复用。
- Optuna + Purged Walk-Forward 自动超参搜索：默认执行 8 次 TPE 试验，联合调优 LightGBM/GBDT 的学习率、树深与进/出场阈值，并把最佳配置写入 `data_cache/training/hyperopt_*.json`。
- PyTorch LSTM 深度动量模型：内置序列化特征构造、批量训练与梯度裁剪，适合捕捉非线性惯性。
- 强化学习战术盘：在回测页面实时展示“概率×趋势”状态下的最优动作与期望边际。
- 24 小时金融助手：实时巡检行情、AI 输出三段式提醒，并通过 `/api/realtime_snapshot`、`/api/assistant/*` 接口提供程序化推送能力。

## 快速开始

```bash
cd quant_trading_web
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

浏览器访问 `http://127.0.0.1:8000/`，填写参数并运行回测。如需生成 AI 洞察，请勾选“启用AI市场洞察”复选框。

## 导出报告

生成回测后，点击页面右上角“下载 JSON 报告”即可保存当前分析（含指标、执行清单、风险提醒等）。

## 界面快速上手（建议新手先照做一遍）

1. **启动服务**：完成“快速开始”中的 `runserver`，浏览器访问 `http://127.0.0.1:8000/backtest/`。
2. **填写表单**：只需输入标的与日期，其他参数使用默认值即可；若想体验深度模型，可在“模型类型”下拉中切换到 LSTM / Transformer，或直接选择 Fusion 让系统自动融合两套序列模型。
3. **解读结果**：
   - “核心亮点”“风险提醒”帮助你在一分钟内了解表现。
   - “强化学习战术盘”展示了概率 × 趋势下代理最推荐的动作（加仓/观望/对冲）及期望边际。
   - “自动调参结果”显示本次 Optuna 试验的最佳阈值与超参，方便复现。
   - 需要更多数据时点击“打开专业数据”或切换到“专业数据”标签查看完整图表、滚动指标与 SHAP 解释。
4. **导出/记录**：可在页面右上角下载 JSON 报告，或在“历史记录”中快速载入之前的回测。默认模式已切换为“多策略组合”，会同时考虑机器学习、强化学习与传统双均线，你无需额外调参即可体验最强配置。

## 批量回测（离线脚本）

```python
from datetime import date
from quant_trading_web.trading.batch import run_batch_backtests
from quant_trading_web.trading.strategies import StrategyInput

base = StrategyInput(
    ticker="AAPL",
    benchmark_ticker="SPY",
    start_date=date(2020, 1, 1),
    end_date=date.today(),
    short_window=20,
    long_window=60,
    rsi_period=14,
    include_plots=False,
    show_ai_thoughts=False,
    risk_profile="balanced",
    capital=100000,
    ml_model="lightgbm",        # 可选 'sk_gbdt' / 'lightgbm' / 'catboost' / 'lstm' / 'transformer' / 'seq_hybrid'
    # 若改为 'seq_hybrid' 可自动融合 LSTM + Transformer，省去手动选择；也可单独指定 'lstm' 或 'transformer'
    dl_sequence_length=36,
    dl_hidden_dim=96,
    max_drawdown_stop=0.25,     # 触发后强制平仓，守住最大回撤
    daily_exposure_limit=1.5,   # 限制单日总曝险，避免无限杠杆
    optimize_thresholds=True,    # 新增：验证集学习入/出场阈值
    val_ratio=0.15,              # 新增：验证占比
    embargo_days=5,              # 新增：防泄露隔离天数
)

df = run_batch_backtests(["AAPL", "MSFT", "NVDA"], base, engines=["ml_momentum", "sma_crossover"])
print(df)
```

执行后会在 `data_cache/batch_reports/` 下生成 CSV 文件，方便进一步分析或汇报。

## 自动超参搜索（Optuna + Walk-Forward）

- 后台默认启用 8 次 TPE 试验的 Optuna 搜索（可通过环境变量 `ENABLE_HYPEROPT=0/1`、`HYPEROPT_TRIALS`、`HYPEROPT_TIMEOUT` 调整），结合净值走期切片与防泄露隔离天数自动调节入/出场阈值和梯度提升模型超参。
- 每次搜索的最优结果会写入 `data_cache/training/hyperopt_<TICKER>.json`，并同步应用到随后的回测；若数据不足则在界面顶部给出提示。
- 如需离线批量调参，可直接复用策略模块：

```python
from datetime import date
from quant_trading_web.trading.strategies import StrategyInput, fetch_price_data, compute_indicators, build_feature_matrix
from quant_trading_web.trading.preprocessing import sanitize_price_history
from quant_trading_web.trading.optimization import run_optuna_search

params = StrategyInput(
    ticker="NVDA",
    benchmark_ticker="SPY",
    start_date=date(2020, 1, 1),
    end_date=date.today(),
    short_window=30,
    long_window=120,
    rsi_period=14,
    include_plots=False,
    show_ai_thoughts=False,
    risk_profile="balanced",
    capital=100000,
    enable_hyperopt=True,
    hyperopt_trials=20,
)
prices, _ = fetch_price_data(params.ticker, params.start_date, params.end_date)
prices, _ = sanitize_price_history(prices)
feature_frame, feature_cols = build_feature_matrix(compute_indicators(prices, params.short_window, params.long_window, params.rsi_period), params)
report = run_optuna_search(feature_frame, feature_cols, params)
print(report)
```

## 深度学习 / 强化学习 / 风控

- **LSTM 模型**：在 `StrategyInput` 中将 `ml_model="lstm"` 并设置 `dl_sequence_length`、`dl_hidden_dim`、`dl_epochs` 等参数即可启用。框架会自动将特征矩阵裁剪为张量序列，使用 PyTorch 训练并输出概率信号，仍可与阈值优化、校准和波动率目标等模块无缝衔接。
- **Fusion 序列融合**：当 `ml_model="seq_hybrid"` 时，系统会同时训练 LSTM 与 Transformer，自动在最近验证片段上挑选表现更好的模型或对两者概率做平均，用户无需在 UI 上手动决策；相关阈值（`validation_ratio`、`score_tolerance`、`ml_params.selection_mode`）可灵活调整。
- **强化学习战术盘**：回测结束后会根据“概率 × 均线趋势”构建有限状态空间，通过价值迭代给出“加仓 / 观望 / 对冲”最优动作及期望边际，结果在前端 `强化学习战术盘` 卡片展示，可作为盘中执行参考。
- **强化学习策略引擎**：将 `strategy_engine="rl_policy"` 即可直接以强化学习为主策略；该模式会先训练机器学习概率，再由 RL 代理（默认 Value Iteration，可通过 `rl_engine="finrl"`/`"sb3"` 启用 Stable Baselines3/FinRL PPO）转换成独立的仓位与回测结果，同步计入 `StrategyOutcome`。如需使用 SB3/FinRL，请运行 `pip install stable-baselines3 gymnasium finrl`。
- **最大回撤止损 (`max_drawdown_stop`)**：默认 25%，一旦触发系统会自动清仓，并在亏损收敛时恢复；触发记录会出现在“风险提醒”与 `warnings` 中。
- **日曝险上限 (`daily_exposure_limit`)**：默认 1.5（150% 资金），任何超过该阈值的仓位都会被裁剪；触发次数会在前端提示。
- **多切片验证 (`validation_slices`, `out_of_sample_ratio`)**：自动对样本进行多段滚动切割并输出 Sharpe/CAGR/最大回撤均值，帮助评估 out-of-sample 稳健性。
- **撮合模拟 (`execution_liquidity_buffer`, `execution_penalty_bps`)**：基于 ADV 与成交额建模执行冲击，为策略收益扣除额外的冲击成本，让结果更贴近实盘。

## 策略引擎切换

- `strategy_engine="ml_momentum"`：纯机器学习动量策略，支持 GBDT/LSTM/Transformer 与阈值调优。
- `strategy_engine="rl_policy"`：以强化学习代理生成信号；若 `rl_engine="value_iter"`（默认）则使用轻量值迭代代理，设置为 `"finrl"` 或 `"sb3"` 可在安装 Stable Baselines3+FinRL 后使用 PPO 训练的 DRL pipeline。
- `strategy_engine="multi_combo"`（默认）：组合机器学习、强化学习与传统双均线三套策略，自动计算加权净值与组件权重，属于“最强预测系统”的一体化方案。

## LLM 集成（DeepSeek + Qwen3）

应用支持“两模型协同”分析：主模型（默认 DeepSeek R1 8B）给出初稿与思考过程，副模型（默认 Qwen3 8B）在其基础上进行审阅与完善，输出最终执行方案。

1. 安装 [Ollama](https://ollama.com/)，拉取模型：
   - `ollama pull deepseek-r1:8b`
   - `ollama pull qwen3:8b`
2. 启动服务（通常在本地 `http://localhost:11434` 监听）；
3. 可用环境变量：
   - `OLLAMA_ENDPOINT`：默认 `http://localhost:11434/api/generate`
   - `OLLAMA_MODEL`：默认 `deepseek-r1:8b`
   - `OLLAMA_TEMPERATURE`：默认 `0.6`
   - `OLLAMA_SECONDARY_MODEL`：默认 `qwen3:8b`
   - `OLLAMA_SECONDARY_ENDPOINT`：可与主端点一致，默认沿用 `OLLAMA_ENDPOINT`
   - `OLLAMA_NUM_PREDICT`：主模型最大生成 token 数，默认 600
   - `OLLAMA_SECONDARY_NUM_PREDICT`：副模型最大生成 token 数，默认 600
   - `OLLAMA_NUM_CTX`：主模型上下文窗口，默认 4096
   - `OLLAMA_SECONDARY_NUM_CTX`：副模型上下文窗口，默认 4096
   - `OLLAMA_TIMEOUT_SECONDS`：主模型请求超时（秒），默认 60
   - `OLLAMA_SECONDARY_TIMEOUT_SECONDS`：副模型请求超时（秒），默认 75
   - `OLLAMA_RETRIES`：主模型失败重试次数，默认 2
   - `OLLAMA_SECONDARY_RETRIES`：副模型失败重试次数，默认 1
   - `OLLAMA_FOLLOWUP_MAX_CHARS`：传递给副模型的上一位结论最大字符数，默认 2400
   - `OLLAMA_FOLLOWUP_MAX_THOUGHT_LINES`：上一位思考过程行数上限，默认 16

当模型未启动时，前端会在表单顶部提示“AI分析失败”。

提示：如果你不需要看到“思考过程”，可在表单中关闭“展示AI思考过程”以获得更精炼的报告。

性能提示：较大模型（如 Qwen3 30B）首次加载和生成会显著慢于 8B/14B。默认使用 `qwen3:8b`，如需更强推理可切换为 14B/30B 并适当增大超时；本机资源有限时保持 8B 或更轻量模型可获得更快响应。

## 实时助手与巡检

运行成功后，仪表盘会在“实时助手”面板展示最新信号、风险提醒、预测展望、仓位建议以及关联资讯。核心模块包括：

- **RealtimeHub**：后台轮询（或接入第三方流式 API）实时更新行情，最新报价会直接写入特征缓存与助手引擎；
- **AdvancedForecaster**：使用梯度提升压缩版序列模型输出未来数日的期望收益、上涨概率与预测波动；
- **Portfolio Optimizer**：基于 Kelly fraction、波动约束与回测风险指标生成目标仓位/止损/止盈建议；
- **Ticker News Search**：通过 DuckDuckGo 检索最新资讯，助手与前端可即时展示相关新闻摘要；
- **改进版 LLM 提示**：会结合实时预测、仓位建议与外部资讯生成行动指南。

同时提供以下接口便于自动化集成：

- `GET /api/realtime_snapshot?ticker=xxx`：获取最近回测窗口的轻量快照（核心指标、近 5 条信号、风险摘要）。
- `GET /api/assistant/status?ticker=xxx`：触发助手巡检并返回三段式 AI 提醒、概率、仓位、指标等信息。
- `POST /api/assistant/ack`：确认当前提醒，避免重复提示。请求体示例：`{"ticker":"AAPL","signature":"..."}`。
- `GET /api/assistant/notifications?limit=20`：读取最近的提醒列表，便于推送或审计。

相关环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ASSISTANT_ENABLED` | `1` | 是否启用后台助手调度；设为 `0` 可完全关闭。
| `ASSISTANT_LOOKBACK_DAYS` | `365` | 助手巡检时的最短回溯天数，用于构造实时回测窗口。
| `ASSISTANT_SCAN_MINUTES` | `30` | 后台巡检间隔（分钟），依赖 APScheduler 或内置线程。
| `ASSISTANT_ACTION_PROB` | `0.65` | 当模型上涨概率高于该阈值且信号为多头时触发“加仓”提醒。
| `ASSISTANT_NEUTRAL_BAND` | `0.05` | 模型概率在 0.5±band 内视为中性，避免噪声提醒。
| `ASSISTANT_DRAWDOWN_ALERT` | `-0.12` | 最大回撤低于该值时标记风险提醒。
| `ASSISTANT_VOL_ALERT` | `0.35` | 年化波动超过阈值时给出风险提示。
| `ASSISTANT_STOP_BUFFER` | `0.03` | 助手提示中的波动容忍度，用于生成操作建议文字。
| `ASSISTANT_CHECK_MINUTES` | `30` | 前端面板默认建议的再次巡检时间，可被后端返回动态覆盖。
| `ASSISTANT_FORECAST_HORIZON` | `5` | 高级预测模型的时间跨度（交易日）。
| `ASSISTANT_MAX_KELLY` | `0.3` | 仓位建议中 Kelly fraction 的上限，避免过度杠杆。
| `REALTIME_PROVIDER` | `auto` | 实时行情提供方（`auto`/`yf`/`dummy`），可自定义接入第三方流式源。

若需自定义 watchlist，可在 `.env` 中设置 `PIPELINE_TICKERS="AAPL,MSFT,NVDA"`；助手会自动追加用户请求的标的并缓存最新行情与特征。

## 输出格式与长度控制

为提升可读性并避免“罗列一堆指标”，系统对最终报告进行约束与归一化：

- 段落固定为 4 个：盈利机会 / 操作计划 / 风险盲区 / 额外关注；
- 每段恰好 3 条要点、以 `- ` 开头，禁止表格/代码块/额外标题；
- 默认每条 ≤40 汉字，最终文本会被二次规整与截断；
- 可调参数：
- `AI_MAX_BULLETS_PER_SECTION`（默认 3）
- `AI_MAX_BULLET_CHARS`（默认 120，用于二次规整的最大字符数）
- `AI_MAX_TOTAL_CHARS`（默认 2600，总长度上限）
- `AI_QA_MAX_TOTAL_CHARS`（默认 2000，仅对“答疑模式”生效）

## AI 联网搜索（Ollama Web Search）

AI 聊天面板默认离线，只读取回测结果。若需要让 AI 在对话过程中拉取实时新闻，请完成以下配置：

1. 启动本地 Ollama 服务：`ollama serve`（确保 11434 端口可用），并给目标模型授权 Web Search（例如 `ollama run deepseek-r1:8b --keepalive 5m --web`，部分版本的参数可能为 `--web-search`，按本机提示为准）。
2. `.env` 至少包含：
   ```bash
   OLLAMA_ENDPOINT=http://127.0.0.1:11434/api/generate
   # 若需要使用云端，可保留 OLLAMA_API_KEY；本地模式可留空
   OLLAMA_API_KEY=<若使用 ollama.com Cloud，此处填入 token>
   ```
   可选地指定 `OLLAMA_WEB_SEARCH_URL`（默认自动根据 `OLLAMA_ENDPOINT` 推导，会按 “自定义 → 本地 11434 → ollama.com 云端” 的顺序重试）。
3. 前端 AI 表单右上角点击“🌐 联网”按钮，按钮高亮后再发送问题；系统会将按钮状态随请求一起提交。若提问中包含“新闻 / 监管 / 调查”等关键词，后端会直接返回实时资讯引用与摘要，确保显示的链接均来自真实抓取结果。

若联网失败，AI 面板会提示具体原因（如本地端口未启动或云端返回 401）。每次联网成功的查询会缓存十几分钟，重复问题会直接复用缓存结果以节省调用次数。

## 风险自由利率（RFR）

为了更严谨地度量 α/Sharpe/Sortino，系统支持设置年化无风险利率：

```bash
export RISK_FREE_RATE_ANNUAL=0.02  # 例如 2%
```

Sharpe/Sortino 将采用超额收益计算，CAPM α 也将以超额收益为基准；未设置时默认为 0。

## 行情缓存

成功从 Yahoo Finance 下载的行情会自动写入 `data_cache/TICKER.csv`（列：`date, close, adj close, volume`），后续运行优先使用缓存以加速并提升离线可用性。

## 注意事项

- 所有计算仅用于教学演示，未包含实盘交易、手续费、滑点等因素；
- 结束日期若晚于当前日期，会自动回退到今天并给出提示；建议窗口≥120 天以获得稳定指标；
- 若下载数据失败，请确认网络环境或改用其他标的符号（例如 `AAPL`、`MSFT`、`600519.SS`）。
- 离线环境可在 `data_cache` 目录中放入命名为 `TICKER.csv` 的历史行情文件（需包含 `Date, Close, Adj Close, Volume` 列）。
- 支持选择风险偏好（保守/均衡/进取），系统会生成不同风格的投资组合建议，并提供收益概率估算。
- 多模型 AI（DeepSeek + GPT-OSS）协同分析，展示各自的思考过程与合并后的最终建议。
- 机器学习引擎：支持 Sklearn GBDT / LightGBM / CatBoost；滚动训练内置 Embargo 与验证集阈值优化，默认开启。
- 若需联网获取市场快讯，可设置 `ENABLE_WEB_SEARCH=1` 并配置 DuckDuckGo 相关环境变量（如 `DDG_REGION`、`DDG_SAFESEARCH`、`DDG_PROXY`），未启用时会提示处于离线模式。

## 预测质量 · 怎么看是否“够好”？

我们给出一套“可执行”的质量标尺，并在结果页显示“质量评级”（高/中/低）。核心维度：

- Sharpe（年化超额收益/波动）：≥1.0 视为“良好”，≥1.5 “优秀”；<0 表示风险补偿不足。
- Calmar（年化收益/最大回撤）：≥0.6“稳健”，≥1.0“优异”；<0.3 需谨慎。
- 最大回撤（MDD）：>-20% 更友好；<-35% 需降杠杆或加对冲。
- 概率校准与提升：高概率组胜率显著高于低概率组，可靠性曲线接近 45°。

系统还会给出“信心评分/评级”（基于 Sharpe、CAGR、MDD 的组合）。你能在“绩效概览”看到“质量评级：高/中/低”。

## 一次实际全量训练（7年，联网）

我们用 yfinance 跑了三组标的的网格搜索（含 Embargo=5、验证=15%、阈值优化）：

- mega（AAPL/MSFT/NVDA/AMZN/META/GOOGL/TSLA）：最优 Sklearn GBDT（lr=0.03, n_estimators=500, max_depth=3, subsample=0.8）。集合均值 Sharpe≈0.14，CAGR≈1.3%，MDD≈-27.7%。结论：对大盘风格切换敏感，建议搭配“混合模式/对冲”提升稳定性。
- etf（SPY/QQQ/IWM/TLT/GLD）：最优 CatBoost（lr=0.05, iters=500, depth=6, subsample=0.8）。集合均值 Sharpe≈0.09，CAGR≈0.5%，MDD≈-29.3%。结论：窄窗难打败宽基，建议加“混合模式/再平衡”。
- sector（XLF/XLK/XLE/XLV/XLY）：整体偏弱（负分），推荐换用 `ml_task=hybrid` 或 `label_style=triple_barrier` 再训练。

实际单标例：`NVDA` 近 2 年在 `ml_model=lightgbm + ml_task=hybrid + 校准 + 早停` 下，Sharpe≈1.4、CAGR≈9.8%，达到“良好线”，可作为执行参考；宽基近 3 年表现一般属常见现象。

训练产物在 `data_cache/training/`：

- `report_*.csv`、`best_ml_config_*.json`、`best_ml_config_overall.json`
- 可用这些最优配置填入表单（或 `StrategyInput.ml_params`）作为默认引擎参数。

> 建议：
> - 高频换手/回撤升高 → 提高 `min_holding_days`、收紧阈值，或降低 `volatility_target`；
> - 可靠性图偏离/提升不显著 → 重新特征与引擎、改 `hybrid` 或 `triple_barrier`；
> - β 高位/波动上行 → 在“资产建议”里增配对冲（国债/黄金/指数）。
- 生成图表和 AI 思考过程可在表单中开关；若禁用则仅输出核心指标。
- 支持选择风险偏好（保守/均衡/进取），系统将基于回测数据给出三套中文策略建议，并展示关键指标解释。
- 若需联网获取市场快讯，可设置 `ENABLE_WEB_SEARCH=1` 并配置 `DDG_SEARCH_ENDPOINT` 指向 DuckDuckGo 代理/接口；默认在离线模式下给出提示。
