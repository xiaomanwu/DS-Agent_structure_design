# 后端接口、并发与部署 / API, concurrency & deployment

> 本文覆盖：FastAPI 端点清单、并发模型（哪里有锁、哪里可以水平扩展）、
> LLM 超时降级、以及三种部署方式（本地 / Docker / 多 worker）。

## 一、API 端点（`ds_agent/web.py`）

| 方法 | 路径 | 说明 | 阻塞性 |
| --- | --- | --- | --- |
| GET | `/` | 单页聊天界面（内嵌 index.html） | 即时 |
| GET | `/api/health` | 健康检查：数仓就绪状态 + LLM 配置状态 | 即时 |
| GET | `/api/capabilities` | 能力目录（含运行时注册的扩展） | 即时 |
| POST | `/api/chat` | 核心端点：自然语言 → 意图路由 → 执行 → 结构化回复 | **同步阻塞**，因果/建模类秒级 |
| POST | `/api/reset` | 清空当前进程会话（session/findings） | 即时 |
| GET/POST | `/api/llm/status` `/api/llm/config` `/api/llm/test` | LLM 运行时配置与连通性测试 | config/test 可能外呼 |
| GET | `/artifacts/{name}` | 下载生成的报告/看板/图片（路径穿越已防护） | 即时 |
| stdio | `ds-agent-mcp` | MCP server（stdio JSON-RPC），暴露 5 个工具给宿主 LLM | 按调用 |

`/api/chat` 返回体：`text`（结论先行的 Markdown）、`blocks`（KPI/表格/图表/
置信卡的有序列表）、`artifacts`（生成文件）、`intent`（命中的能力）。

## 二、并发模型

### 现状（单机工作台）
- **一个进程一个 Agent 实例**（`AgentHolder`，`threading.Lock` 保护懒加载）：
  会话状态（`session`/`findings`/指标字典）都在进程内存里；
- DuckDB 以**只读模式**被 Agent 持有，多个并发 `/api/chat` 共享只读连接——
  DuckDB 允许多读，但写（指标注册、临时表）会加锁；
- 长任务（PSM 安慰剂、因果森林）会占住请求线程直到完成。

### 安全并发上限的推导
- `uvicorn --workers N`：每个 worker 是独立进程、独立 Agent、独立 DuckDB 连接
  → **水平安全**，前提是数仓文件只读或各 worker 独立副本；
- 单 worker 内多线程：查询型请求并发安全（DuckDB 多读），但**指标注册、
  会话写操作**会竞争 → 生产部署建议每 worker 并发 ≤ 4（`--limit-concurrency`），
  或按 user 分片路由（sticky session）；
- CPU 密集任务（SHAP、bootstrap）受 GIL 限制，多 worker 才是真并行。

### LLM 调用的降级链
LLM 只承担意图识别/任务分发与结果润色，因此：
1. 每次调用带超时（默认 30s）与一次重试；
2. 失败 → 自动回退**规则路由**（`route()` 打分），功能零损失；
3. 执行失败 → **重规划循环**（次优能力重试，至多 2 次），与 LLM 无关。

生产建议再加：调用预算（每 user 每天 N 次）、并发信号量、慢调用熔断。

## 三、部署

### 1. 本地（默认形态）
```bash
pip install -e .
uvicorn ds_agent.web:app --host 127.0.0.1 --port 8000
# 首次启动自动构建演示数仓（data/wh.duckdb）
```

### 2. Docker（推荐基线）
```bash
docker build -t ds-agent .
docker run -p 8000:8000 \
  -v ds-agent-artifacts:/app/artifacts \
  ds-agent
# 可选：挂载自己的数仓 / 注入 LLM 配置
#   -v $PWD/demo_data/warehouse.duckdb:/app/demo_data/warehouse.duckdb:ro
#   -e DS_AGENT_LLM_BASE_URL=... -e DS_AGENT_LLM_API_KEY=... -e DS_AGENT_LLM_MODEL=...
```

### 3. 多 worker（小团队共享）
```bash
gunicorn ds_agent.web:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 --timeout 300 \
  --limit-concurrency 16 \
  -b 0.0.0.0:8000
```
注意：
- workers 前面放 nginx/Caddy 做统一入口与静态缓存；
- `artifacts/` 必须是**共享卷**（所有 worker 生成到同一目录才能被 `/artifacts` 命中）；
- 多 worker 各自持有指标字典副本，**注册指标会漂移**——多副本部署时指标字典
  应外置（见 `docs/production_design.md` §二.1）；
- 演示数仓默认按需构建，多 worker 同时冷启动可能竞态——先跑一次
  `python examples/00_build_demo_data.py` 再起 gunicorn；
- SSE/长轮询未使用，无需额外代理配置；反代超时需 ≥ 300s 以覆盖因果类长任务。

### 4. 健康检查与探针
- 存活：`GET /api/health` 恒 200；
- 就绪：检查响应里的 `warehouse_ready` 为 true 才接流量。

## 四、容量参考（演示数仓，4 核 CPU）

| 端点 | 单次耗时 | 并发 4 worker 吞吐 |
| --- | --- | --- |
| SQL 问数 | < 100ms | ~100 rps（受 DuckDB 读锁影响轻微） |
| 异动诊断 | 1–3 s | ~4 rps |
| 实验读出 | 2–5 s | ~3 rps |
| PSM（含安慰剂） | 5–10 s | ~1.5 rps |
| 建模闭环（含 SHAP） | 5–15 s | ~1 rps |
| 监控报告 | 2–4 s | ~3 rps |

> 生产数据量下的耗时主要取决于数仓查询下推，统计引擎本身的开销是常数级。
