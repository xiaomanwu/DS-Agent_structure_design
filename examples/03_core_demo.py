"""Full demo of the agent's core feature set, one section per capability.

Runs every core capability end-to-end against the bundled demo warehouse —
including the newer platform features (metric dictionary, external data
sources) — and prints what each step produced plus the artifact manifest.

Run:  python examples/02_core_demo.py
"""

from pathlib import Path
import shutil

from ds_agent.agent import DSQueryAgent
from ds_agent.config import PROJECT_ROOT
from ds_agent.metric_dict import MetricDictionary
from ds_agent.semantic import semantic
from ds_agent.tools.datasource import DataSourceRegistry
from ds_agent.tools.sql import Warehouse


def show(title: str, reply) -> None:
    print(f"\n{'=' * 72}\n■ {title}\n{'-' * 72}")
    first = reply.text.split("\n")
    print("  " + "\n  ".join(first[:6]) + ("\n  …" if len(first) > 6 else ""))
    for tc in reply.tool_calls[1:4]:
        print(f"  [tool] {tc['tool']}")


def main() -> None:
    # A private workspace copy so registrations never touch the shipped demo data.
    workspace = Path("artifacts/core_demo_workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    warehouse = workspace / "warehouse.duckdb"
    if not warehouse.exists():
        shutil.copy(PROJECT_ROOT / "demo_data" / "warehouse.duckdb", warehouse)

    metric_dict = MetricDictionary(path=workspace / "metric_dictionary.json", semantic=semantic, warehouse=Warehouse(warehouse))
    datasources = DataSourceRegistry(Warehouse(warehouse), path=workspace / "data_sources.json")
    agent = DSQueryAgent(warehouse=warehouse, artifact_dir=Path("artifacts"), metric_dict=metric_dict, datasources=datasources)

    print("DS-Agent — core capability demo (deterministic mode, no LLM key needed)")

    show("1. 自然语言查询（语义层口径 + 只读 SQL）", agent.ask("近30天各渠道 GMV 排名"))
    show("2. 指标异动诊断（异常检测 + 贡献度 + 根因下钻 + 置信卡）", agent.ask("最近14天 GMV 为什么跌了？"))
    show("3. 实验设计（15 种实验类型的样本量/时效/分流方案）", agent.ask("支付转化率实验，基线3%，检测2%相对提升，每天5万流量，怎么设计？"))
    show("4. 实验分析（SRM + 显著性 + CUPED + 置信卡）", agent.ask("看下这个 AB 实验的结果"))
    show("5. 因果推断（DID / PSM / 开源引擎交叉验证）", agent.ask("用双重差分评估这次改版对 GMV 的影响"))
    show("6. Uplift 建模（因果森林 + Qini + 人群定向）", agent.ask("用因果森林看哪类用户对补贴最敏感"))
    show("7. 断点回归（RDD）", agent.ask("用断点回归评估会员门槛的效果"))
    show("8. 建模能力（特征工程 + 分类/回归 + 特征重要性 + 置信卡）", agent.ask("训练分类模型预测用户是否转化，看特征重要性"))
    show("9. 看板（KPI + 趋势 + 拆解 + 异动，独立 HTML）", agent.ask("给我生成一个 GMV 监控看板"))
    show("10. 指标字典：注册自定义指标（列校验 + 语义层生效）", agent.ask("注册指标：利润率 = (gmv - refund_amount) / gmv"))
    show("11. 新指标直接进入全链路分析", agent.ask("近7天各渠道 利润率 排名"))
    show("12. 指标字典：查看口径清单", agent.ask("指标字典"))

    # 13. external data source: write a small offline sheet, attach, query.
    import pandas as pd

    budget = workspace / "budget_2026H2.xlsx"
    pd.DataFrame({"month": ["2026-07", "2026-08", "2026-09"], "budget": [8_100_000, 8_300_000, 8_600_000]}).to_excel(budget, index=False)
    show("13. 数据源关联：离线表格接入", agent.ask(f"关联数据源 {budget} 叫 budget"))
    show("14. 外部表直接查询", agent.ask("查一下 budget 全部行"))

    show("15. 业务知识 RAG（含指标定义知识库）", agent.ask("GMV 的口径是什么？"))
    show("16. 方法选型（按场景推荐统计/因果方法）", agent.ask("没有对照组，只有上线前后的数据，该用什么方法评估效果？"))
    show("17. 报告生成（汇总本次会话全部发现）", agent.ask("把分析写成周报"))

    agent.ask("下线指标 利润率")
    agent.close()

    artifacts = sorted(p.name for p in Path("artifacts").glob("*") if p.is_file())
    print(f"\n{'=' * 72}\n■ Artifacts in artifacts/: {len(artifacts)} file(s)")
    for name in artifacts[-8:]:
        print("  -", name)
    print("\nWorkspace (metric dictionary + data source manifest):", workspace)


if __name__ == "__main__":
    main()
