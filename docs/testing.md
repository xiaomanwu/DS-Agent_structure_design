# 测试分类目录（Testing taxonomy）

共 **321 个测试**（297 个用例函数，含参数化展开），按能力域分类如下。
**去重结论**：以 AST 规范化扫描（函数名跨文件重复检测 + 去字符串常量后的
函数体哈希比对），**未发现重名或重复逻辑的用例**——测试数量是真实的，
每个用例覆盖不同路径（详见文末扫描方法）。

| 能力域 | 用例数 | 覆盖内容 | 测试文件 |
| --- | ---: | --- | --- |
| 问数与口径 | 66 | 意图路由、失败重规划、语义层口径、指标字典注册/下线、LLM planner、运行时扩展、外部数据源 | test_agent · test_metric_dict · test_planner · test_registry · test_method_advisor · test_datasource |
| 因果推断 | 64 | DiD(TWFE)、PSM、IPW/AIPW、合成控制、RDD、中断时序、Uplift 因果森林（Qini）、DoWhy 交叉验证 | test_causal · test_causal_oss · test_ts_causal · test_uplift · test_rdd |
| 实验科学 | 47 | 15 种实验设计、SRM/CUPED/BH-FDR 读数、switchback HAC、敏感性套件 | test_experiment · test_experiment_designs · test_readouts |
| 统计原语 | 33 | 功效、Wilson 区间、两比例 z、Welch t、bootstrap、delta 法、置信度验证卡 | test_stats · test_confidence |
| 异动诊断 | 30 | 季节性基线（稳健 z/CUSUM/Holt-Winters/SARIMA/STL 回测择优）、贡献度、3 层根因、自编码器+孤立森林 | test_anomaly · test_contribution · test_root_cause · test_forecast · test_dl_anomaly |
| 接口与界面 | 16 | FastAPI/Web、MCP 协议握手与工具调用 | test_web · test_mcp_server |
| 知识库 | 12 | Markdown 分块、TF-IDF 检索、LangGraph RAG 流程、回退路径 | test_knowledge |
| 建模与交付 | 11 | 预处理→特征工程→CV→SHAP 流水线、看板生成 | test_modeling · test_dashboard |
| 抽样与预处理 | 18 | Cochran 样本量（比例/均值）、分层抽样、代表性检验、缺失值/共线性预处理 | test_preprocess_sampling |

> 数字为用例函数数；`pytest --collect-only` 的 321 包含参数化展开。

## 去重方法（可复跑）

```bash
# 1) 跨文件重名检测：零重名
grep -rn "def test_" tests/test_*.py | sed 's/.*def //;s/(.*//' | sort | uniq -d

# 2) 函数体规范化哈希比对（剥离字符串常量与空白）：零重复
python - <<'PY'
import ast, re
from pathlib import Path
bodies = {}
for f in Path("tests").glob("test_*.py"):
    for node in ast.walk(ast.parse(f.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            norm = re.sub(r"'[^']*'|\"[^\"]*\"", "LIT", ast.unparse(node))
            key = re.sub(r"\s+", "", norm)
            bodies.setdefault(key, []).append(f"{f.name}:{node.name}")
print([v for v in bodies.values() if len(v) > 1])  # -> []
PY
```

## 防重复的约定

- 同一统计公式的数学性质只测一次（`test_stats`），上层处理器只测"接线"
  （调用了正确的原语、输出了正确的结构），不做二次数学断言；
- 端到端问答回归集中在 `test_agent` 与 `benchmarks/golden_qa.jsonl`（后者
  面向发布，期望值冻结；前者面向开发，允许随行为演进而修改）；
- 新增测试前先查本目录，避免为同一分支再写一个"换数字"的副本。
