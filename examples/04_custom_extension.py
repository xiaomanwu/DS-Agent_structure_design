"""Extend the agent with a custom capability at runtime.

DS teams always have house-specific analyses the framework cannot predict.
``ds_agent/registry.py`` makes those first-class citizens: register an
``Extension`` with a handler + routing keywords and it immediately

- joins the deterministic router (keyword scoring),
- is dispatchable by the LLM planner (it reads the merged catalog),
- appears in ``help`` / the web capability list,
- can be chained into LLM task flows.

Run:  python examples/04_custom_extension.py
"""

from pathlib import Path

import pandas as pd

from ds_agent.agent import DSQueryAgent
from ds_agent.registry import Extension, register_extension, unregister_extension
from ds_agent.reply import AgentReply, fmt_num as _fmt_num
from ds_agent.semantic import Metric, semantic, Table


def weekday_profile_handler(question: str, slots: dict) -> AgentReply:
    """Weekday seasonality profile for a metric: best/worst weekday + spread."""
    metric = slots.get("metric") or semantic.default_table().metrics["gmv"]
    table = semantic.default_table(metric.table or "orders_daily")
    sql = (
        f"SELECT dayofweek({table.time_column}) AS dow, {metric.sql()} AS value "
        f"FROM {table.name} GROUP BY 1 ORDER BY 1"
    )
    df = agent.sql.run(sql).dataframe  # noqa: F821 - bound below before any call
    names_zh = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]
    df["weekday"] = df["dow"].map(lambda i: names_zh[int(i)])
    best = df.loc[df["value"].idxmax()]
    worst = df.loc[df["value"].idxmin()]
    spread = float(best["value"]) / float(worst["value"]) - 1 if float(worst["value"]) else float("nan")
    text = (
        f"**{metric.name}** 的星期效应：最强为 {best['weekday']}（{_fmt_num(float(best['value']))}），"
        f"最弱为 {worst['weekday']}（{_fmt_num(float(worst['value']))}），峰谷差 {spread:+.1%}。\n\n"
        f"该能力由运行时注册的自定义扩展提供（见 examples/04_custom_extension.py），"
        f"与内置能力同样参与路由、规划与报告。"
    )
    block = {
        "type": "table",
        "title": "Weekday profile",
        "columns": ["weekday", "value"],
        "rows": df[["weekday", "value"]].round(4).to_dict(orient="records"),
        "markdown": df[["weekday", "value"]].to_string(index=False),
    }
    return AgentReply(
        text=text,
        blocks=[block],
        tool_calls=[{"tool": "weekday_profile", "args": {"metric": metric.name}, "result": {"spread": round(spread, 4)}}],
        data={"intent": "weekday_profile"},
    )


def main() -> None:
    global agent
    agent = DSQueryAgent(artifact_dir=Path("artifacts"))

    # 1. Register — one call makes it a full capability.
    register_extension(
        Extension(
            name="weekday_profile",
            handler=weekday_profile_handler,
            keywords=[r"星期效应|周内效应|weekday (?:effect|profile)|day[- ]of[- ]week"],
            zh="星期效应分析：指标按周几分解，输出峰谷差与最强/最弱日",
            en="Weekday seasonality: decompose a metric by day-of-week with peak/trough spread",
            example_zh="看一下 GMV 的星期效应",
            example_en="Show the weekday effect for GMV",
        )
    )

    # 2. It routes like a built-in.
    reply = agent.ask("看一下 GMV 的星期效应")
    print("[weekday_profile]", reply.text.split("\n")[0])

    # 3. It shows up in help / the capability catalog.
    assert "weekday_profile" in agent._help().text
    print("[catalog] listed in help alongside the built-ins")

    # 4. Cleanup (unregister when the extension is no longer needed).
    unregister_extension("weekday_profile")
    reply2 = agent.ask("看一下 GMV 的星期效应")
    print("[after unregister] intent:", reply2.data.get("intent"))
    agent.close()


if __name__ == "__main__":
    main()
