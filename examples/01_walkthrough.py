"""End-to-end walkthrough: every capability the agent routes to.

Each block calls the same tool the agent calls when you type a question, so this
file doubles as documentation and as a smoke test.

    python examples/01_walkthrough.py

Takes about a minute: the causal forest, the two-sided bootstrap and the ARIMA
interrupted time series are the slow parts, and they are slow on purpose
(they are the estimators that need the work).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ds_agent.agent import DSQueryAgent
from ds_agent.config import settings

QUESTIONS = [
    # 1. governed SQL
    "近30天各渠道 GMV 排名",
    # 2. anomaly + contribution + attribution (the flagship path)
    "近7天 GMV 为什么跌了？",
    # 3. the same diagnosis with a Holt-Winters baseline instead of the decomposition
    "用 holt-winters 看下近7天 GMV 为什么跌了",
    # 4. follow-up inherits metric and window
    "按品类再拆一下",
    # 5. experiment design
    "设计一个支付转化率实验，基线3%，希望检测2%的相对提升，每天5万流量",
    # 6. experiment readout
    "看下这个 AB 实验的结果",
    # 7-10. designs a plain A/B cannot cover
    "看下时间片轮转实验的结果",
    "这个反转实验的效应能持续吗",
    "父子实验怎么验证两层互不干扰",
    "双边实验，买家侧和卖家侧同时干预",
    # 11. is the answer an artefact of one specification?
    "这个实验结论稳健吗，做个敏感性分析",
    # 12-13. heterogeneous effects and the threshold design
    "用因果森林看哪类用户对补贴最敏感",
    "用断点回归评估会员门槛的效果",
    # 14. one unit, before and after
    "这次改版对 GMV 的影响，用时序因果评估",
    # 15. panel causal inference
    "用双重差分评估补贴政策对城市 GMV 的影响",
    # 16. method selection: which estimator fits a described scenario
    "这个场景该用什么方法分析？促销是运营手动选人发的，没有做实验",
    # 17. open-source causal engines: DoWhy re-estimation + refutations
    "用 DoWhy 开源包交叉验证促销的因果效应",
    # 18. deep-learning anomaly detection
    "用自编码器/深度学习方法看下 GMV 有没有异常",
    # 19. business knowledge from the RAG knowledge base
    "GMV 的口径是什么？和 revenue 有什么区别？",
    # 20. modeling: classification + feature analysis
    "训练一个预测用户是否转化的分类模型，看特征重要性",
    # 21. modeling: regression
    "训练回归模型预测每日 GMV",
    # 22. dashboard
    "给我生成一个 GMV 监控看板",
    # 23. report
    "把分析写成周报",
]


def main() -> None:
    agent = DSQueryAgent(warehouse=settings.warehouse)
    try:
        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n{'=' * 72}\n[{i}] {question}\n{'=' * 72}")
            reply = agent.ask(question)
            print(reply.text[:900])
            for block in reply.blocks:
                if block["type"] == "table":
                    print(f"\n  [{block['title']}]")
                    if "markdown" in block:
                        print("  " + block["markdown"].replace("\n", "\n  "))
                    else:  # web-style table: columns + rows
                        rows = [block.get("columns", [])] + block.get("rows", [])[:10]
                        for row in rows:
                            print("  " + " | ".join(str(c) for c in row))
            for artifact in reply.artifacts:
                print(f"\n  artifact -> {artifact}")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
