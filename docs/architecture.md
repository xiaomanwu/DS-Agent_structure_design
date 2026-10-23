# Architecture

The design goal is simple: **every sentence the agent says must be traceable to a number a
statistician could recompute.** That constraint drives the whole layout — the LLM never touches
arithmetic, and the same semantic layer feeds every tool.

```
                       ┌──────────────────────────┐
  question ──► route() │  keyword scoring → intent │
                       └───────────┬──────────────┘
                                   │ slots: metric, dimensions, window, comparison, anchor
        ┌──────────────────────────┼───────────────────────────┬──────────────┐
        ▼                          ▼                           ▼              ▼
   tools/sql.py            tools/anomaly.py            tools/experiment.py  tools/causal.py
   (NL → SQL, guard)       (baseline, z, CUSUM)        (design, readout)    (DiD, PSM, SC)
        │                          │                           │              │
        └──────────────┬───────────┴─────────────┬─────────────┴──────────────┘
                       ▼                         ▼
             tools/contribution.py        stats/ (power, inference)
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                         agent.py (+ readouts.py): AgentReply(text, blocks, artifacts, data)
                                    │
                     ┌──────────────┴───────────────┐
                     ▼                              ▼
              cli.py (REPL / ask)            web.py + web_static/index.html
                     │                              │
                     └──────────► tools/report.py (Markdown + HTML)
```

## Layers

### 1. Semantic layer (`ds_agent/semantic.py`)

One `Table` object declares the time column, the dimensions (with Chinese/English aliases and an
optional code→label map) and the metrics. A `Metric` is either additive (`numerator`) or a ratio
(`numerator / denominator`), and knows its unit and whether higher is better.

Everything downstream reads from here:

- SQL generation uses `Metric.sql()` and `Table.dimension_expr()`.
- Anomaly detection uses `metric.is_ratio` to decide whether to pass denominators.
- Attribution uses `kind` to choose the additive or the rate/mix decomposition.
- Reports use `unit` and `higher_is_better` to phrase "good" vs "bad".

This is why "GMV", "成交额" and "revenue" cannot drift apart: there is exactly one definition of each.

### 2. Tool layer (`ds_agent/tools/`)

Each module is a pure function library plus a thin wrapper the agent calls. None of them import the
agent — they are usable (and tested) standalone.

| Module | Responsibility |
| --- | --- |
| `sql.py` | Guarded SQL: strip comments, allow a single `SELECT`/`WITH`, forbid DDL/DML keywords, wrap results in a `LIMIT`. Parses relative time windows (`近7天`, `上周`, `last 30 days`). |
| `anomaly.py` | Seasonal baseline, robust z-scores, CUSUM, episode merging, segment scans, window comparison. |
| `forecast.py` | Holt-Winters (additive/multiplicative), SARIMA, STL+ETS, seasonal naive, EWMA, trailing median; rolling-origin backtest, MASE, model selection. |
| `contribution.py` | Exact decompositions and multi-dimension attribution. |
| `experiment.py` | 15 experiment archetypes, power/sample-size planning, readout with SRM/CUPED/BH. |
| `experiment_designs.py` | The designs a plain A/B cannot cover: switchback (block-level HAC), reversal (sum test + persistence), parent-child (orthogonality + parent FE), two-sided 2×2, sensitivity analysis (specification curve + A/A + jackknife). |
| `causal.py` | DiD, event study, PSM (caliper matching, common-support + overlap diagnostics), IPW, AIPW, synthetic control, estimator comparison. |
| `causal_oss.py` | Open-source engine adapters: DoWhy backdoor re-estimation (linear/PSM/IPW), placebo-treatment and random-common-cause refutations, optional EconML causal forest, cross-engine agreement card. Every adapter degrades gracefully when its package is missing. |
| `dl_anomaly.py` | Deep-learning anomaly detection: autoencoder reconstruction error (stationary ratio input, quiet-window training, quantile threshold; torch LSTM-AE when available) and an isolation forest over level/DoD/WoW/trend-ratio features, with a day-level confidence card. |
| `uplift.py` | Causal tree, causal forest (honest split), T/S/X learners, Qini curve, decile targeting, cross-fitted AIPW. |
| `rdd.py` | Local linear sharp/fuzzy RDD, bandwidth selection and sensitivity, density manipulation test, covariate balance, placebo cutoffs. |
| `ts_causal.py` | ARIMA interrupted time series (log scale, trend + weekday + controls, placebo dates, specification curve) and a structural counterfactual with forecast intervals. |
| `root_cause.py` | Multi-level drill-down decision tree: per-node dimension selection, threshold branching, ranked root-cause paths, icicle chart. |
| `method_advisor.py` | Scenario parsing + a decision tree over the whole method catalogue: ranked recommendations with reasons, assumptions, pitfalls and the implementing tool. |
| `report.py` | `Section`/`Report` objects rendering to Markdown and self-contained HTML. |

### 2c. Confidence layer (`ds_agent/confidence.py`)

Every first-class readout ends with a standardized confidence card: named checks
(PASS/WARN/FAIL) plus an overall verdict (高置信 / 中等置信 / 低置信).

- **Anomaly**: window significance, baseline backtest quality (MASE), episode persistence,
  leave-one-out outlier dependence, segment agreement; chart = actual vs expected ± 2.5σ noise band.
- **AB readout**: SRM, BH-adjusted significance, label-permutation cross-check (histogram of the
  permuted null with the observed diff marked), CUPED direction agreement, adverse segments.
- **DiD**: parallel trends, pre-period leads, **in-space placebo** (the treated label is permuted
  across cities and TWFE re-run, seeded and deterministic; the real ATT must sit in the tail),
  level-vs-log sign agreement, treated coverage; chart = placebo-effect histogram.
- **PSM**: post-match SMD balance, common-support overlap, PSM-vs-AIPW agreement, significance.
- **Time-series causal**: Ljung-Box whiteness, placebo intervention dates, two-model agreement,
  counterfactual fit R², specification-curve stability.

### 2b. Knowledge layer (`ds_agent/knowledge.py`)

A RAG over `docs/business_knowledge.md` built on **LangChain + LangGraph** (with a pure-Python
fallback when those are not installed):

- **Chunking**: LangChain's `MarkdownHeaderTextSplitter` splits on `##`/`###`; each chunk title
  inherits its parents ("指标口径 > GMV").
- **Retrieval**: a custom LangChain `BaseRetriever` (`TitleBodyRetriever`) — TF-IDF over Chinese
  character bigrams + ASCII words, title and body scored as separate length-normalised fields (2:1).
- **Orchestration**: a LangGraph `StateGraph` runs `retrieve → grade → compose`; `KnowledgeBase.answer()`
  invokes the compiled graph and returns context, sources and scores.

Wired in three places:

- SQL answers append the relevant caliber note ("GMV includes unpaid orders, ignores refunds").
- Reports open with a business-context section built from the session's metrics.
- The `knowledge` intent answers glossary questions directly, with sources and relevance.

### 3. Stats primitives (`ds_agent/stats/`)

`inference.py` and `power.py` hold the maths: Wilson intervals, two-proportion z, Welch t, bootstrap,
delta-method ratio tests, SRM, CUPED, BH-FDR, O'Brien-Fleming, mSPRT, and the sample-size solvers.
Keeping them separate means the tools compose statistics instead of re-deriving formulas inline.

### 4. Orchestrator (`ds_agent/agent.py` + `ds_agent/readouts.py`)

- `route()` scores keyword families into eighteen intents, then extracts slots (metric, dimensions,
  window, comparison, anchor). An LLM, when configured, fills the same slots — the handlers are
  identical. The advanced intents (switchback/reversal/nested/two-sided/sensitivity/uplift/rdd/
  ts_causal/causal_oss/dl_anomaly) carry a heavier weight (+9) because their phrasings also contain
  generic words like "实验" or "因果" and would otherwise be split with the generic buckets:

  | Intent | Trigger examples | Handler |
  | --- | --- | --- |
  | `switchback` | 时间片 / 轮转 / switchback | `readouts.py::_handle_switchback` |
  | `reversal` | 反转实验 / 对调 / swap | `_handle_reversal` |
  | `nested` | 父子 / 嵌套实验 / 正交分层 | `_handle_nested` |
  | `two_sided` | 双边实验 / 2x2 / 买卖双方 | `_handle_two_sided` |
  | `sensitivity` | 敏感性 / 稳健性 / 规格曲线 | `_handle_sensitivity` |
  | `uplift` | uplift / 因果森林 / 因果树 / qini | `_handle_uplift` |
  | `rdd` | 断点回归 / RDD | `_handle_rdd` |
  | `ts_causal` | 时序因果 / 中断时间序列 / causal impact | `_handle_ts_causal` |
  | `method_select` | 该用什么方法 / 方法选型 / how to analyze | `_handle_method_select` (weight +12 — the question asks *what to use*, not for a result) |
  | `knowledge` | 口径 / 数据字典 / 是什么意思 | `_handle_knowledge` (RAG over the knowledge base) |
  | `diagnose` (baseline) | 时序基线 / holt-winters / arima | `agent.py::_handle_diagnose` with `_baseline_from_question` |

- Follow-up questions carry no intent keywords of their own, so `_is_followup()` detects them and
  reuses the previous intent, metric and window. An explicitly requested dimension is *pinned* as the
  attribution focus instead of being overridden by whichever dimension scores highest.
- Each handler returns an `AgentReply` with four parts:
  - `text` — the narration, localised (zh/en) and always backed by a tool call;
  - `blocks` — typed render payloads (`kpi`, `chart`, `table`) so the web UI and CLI never parse prose;
  - `artifacts` — file paths for reports;
  - `data` — raw structured results, so a caller can assert on numbers.
- Every step is appended to `tool_calls`, which is what makes the answer auditable.
- The advanced handlers live in `readouts.py` as a mixin (`ReadoutHandlers`) rather than in `agent.py`:
  they share one shape — load a table, call exactly one tool, compose KPI + chart + table + narrative —
  and each records a `Section`, so "write this up" reproduces them without re-running anything.

### 5. Entry points

- `cli.py`: `chat` (REPL), `ask` (one-shot), `serve` (web), `build-data` (rebuild the demo warehouse).
- `web.py`: FastAPI app with `/api/chat`, `/api/capabilities`, `/api/health` and a single-file
  chat frontend (`web_static/index.html`) that renders blocks directly.
- A lock around the agent instance keeps the REPL and the web UI safe against concurrent turns.

## Request lifecycle

1. Normalise the question, detect language (`zh` unless the text is clearly English).
2. `route()` → `(intent, slots, confidence)`.
3. Handler runs: queries the warehouse, computes statistics, builds charts and tables.
4. Narration is assembled from the computed values — no free-form generation over numbers.
5. If the intent is `report`, previously cached sections are rendered to Markdown + HTML.
6. Session state (`last_intent`, `last_window`, `last_metric`, `last_dimensions`) is updated so the
   next turn can be a fragment.

## Extension points

| To add… | Do this |
| --- | --- |
| A new metric | Add a `Metric` to the table in `semantic.py` (numerator, optional denominator, aliases, unit) — or register it at runtime through the metric dictionary (below). |
| A new dimension | Add it to `Table.dimensions`; if it is coded (0/1), add `value_labels`. |
| A new warehouse | Point `DS_AGENT_WAREHOUSE` at a DuckDB file with the same table shapes, or generalise `Warehouse` to another engine. |
| A new intent (code change) | Add an entry to `CAPABILITIES`, a scoring rule in `route()`, and a `_handle_*` method. |
| A new intent (runtime, no fork) | `register_extension(Extension(...))` — see `ds_agent/registry.py` and `examples/04_custom_extension.py`. |
| An LLM planner | Implement a `route_with_llm()` that returns the same `(intent, slots)` contract; nothing else changes. |
| An external data source | `关联数据源 <file>` / `连接数据库 <uri>` — the `DataSourceRegistry` attaches it into the session. |

### Runtime extensibility (`ds_agent/registry.py`)

An `Extension` bundles a handler (`(question, slots) -> AgentReply`), routing
keywords, and catalog metadata. Registered extensions are first-class:

- the deterministic router scores their keywords;
- the LLM planner sees them in the merged catalog (`agent.full_catalog()`);
- `help`, the web `/api/capabilities` list and task chains include them;
- `agent.register_extension(...)` / `unregister_extension(name)` manage the
  lifecycle; `registry.clear_extensions()` resets state (tests/hot reload).

### Metric dictionary (`ds_agent/metric_dict.py`)

User-defined metrics are registered with a formula (`注册指标：x = a / b`),
validated against the live warehouse (`SELECT <expr> FROM <table> LIMIT 1`),
activated into the semantic layer, and persisted to JSON with governance
metadata (owner, created/updated dates, deprecation instead of deletion).
Methodology and the registration checklist live in
`docs/metric_dictionary.md`, which is also part of the RAG corpus — caliber
questions retrieve it.

### Data sources (`ds_agent/tools/datasource.py`)

`DataSourceRegistry` attaches external data into the session's DuckDB:

- offline files (CSV/TSV/Parquet/JSON as `read_*_auto` views, Excel imported
  as a table) and databases (DuckDB `ATTACH`, SQLite, PostgreSQL/MySQL URIs);
- registrations persist to a JSON manifest and are re-applied on startup;
- attached tables are immediately usable from natural-language queries (the
  direct-table shortcut in `_llm_sql_fallback`), guarded SQL, diagnosis,
  modeling and dashboards — because they live in the same connection.

## Testing strategy

- `tests/test_anomaly.py` — planted dips are found, clean and purely seasonal series stay quiet,
  sliced frames keep their dates, zero-containing series fall back to the additive model.
- `tests/test_forecast.py` — forecasts stay causal (rewriting the future must not move the head), every
  method is finite, Holt-Winters beats seasonal naive on MASE, selection is deterministic.
- `tests/test_contribution.py` — decompositions sum exactly to the total move; the planted segment
  is ranked first.
- `tests/test_experiment.py` — sample-size monotonicity, the planted +6% lift is recovered, SRM is
  detected on lopsided traffic, the ratio test is a real delta method.
- `tests/test_experiment_designs.py` — switchback recovers +4.5% using blocks (not rows), reversal flips
  sign with persistence ≈ 0.7, nested finds the +6% child effect with orthogonality passing, two-sided
  separates +8% buyer from +6% seller with a cross-side ratio, sensitivity runs every meaningful spec.
- `tests/test_causal.py` — DiD / synthetic control recover the ~9% policy effect, PSM/IPW/AIPW land
  near the known ATE, event-study pre-trends pass on untreated data.
- `tests/test_uplift.py` — Qini ordering (perfect > random + 0.2, reversed ≈ −perfect), causal tree
  ATE within ±3 of the planted 12.5, monotone targeting shares, determinism.
- `tests/test_rdd.py` — sharp jump ≈ 22 with a stable bandwidth, no manipulation on clean scores,
  covariate balance, placebo cutoffs, fuzzy Wald ratio.
- `tests/test_ts_causal.py` — both estimators recover the planted +12% (and beat the naive +24% that the
  trend produces), Ljung-Box on the residuals, specification curve, placebo dates, determinism.
- `tests/test_agent.py` — routing, SQL guard, time-window parsing, baseline selection, end-to-end
  replies, report files.
- `tests/test_readouts.py` — all eight advanced intents route correctly and produce KPI + chart + table
  + a report section, with the expected numbers and both languages.

Fixtures build the demo warehouse once per session under `tests/.tmp` (not the OS temp dir, which is
often not writable in sandboxes).
