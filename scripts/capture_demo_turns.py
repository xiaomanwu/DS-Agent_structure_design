"""Capture one real agent turn per capability, for the demo animation.

Every scene is a genuine ``DSQueryAgent.ask()`` call against the demo
warehouse: the questions, KPIs, tables and charts that end up in the GIF are
the agent's own output, not hand-written mock-ups. Because a few capabilities
(causal forest, ARIMA, deep-learning detectors) are expensive and the sandbox
has a small memory ceiling, scenes are captured one process at a time:

    python scripts/capture_demo_turns.py --scene diagnose
    python scripts/capture_demo_turns.py --merge      # -> artifacts/demo_transcript.json

Without ``--scene`` the script lists the scene ids.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "demo_turns"
MERGED = ROOT / "artifacts" / "demo_transcript.json"

# (id, capability, label_zh, label_en, question_zh, question_en)
# The Chinese questions are the same strings the walkthrough uses, so a scene
# that renders also proves the router still lands on that capability.
SCENES: List[Dict[str, str]] = [
    {
        "id": "sampling",
        "label_zh": "统计抽样",
        "label_en": "Statistical sampling",
        "question_zh": "从用户表按渠道分层抽样 2000 个，检验样本代表性",
        "question_en": "Draw a stratified sample of 2000 users by channel and test representativeness",
    },
    {
        "id": "diagnose",
        "label_zh": "指标异常与归因",
        "label_en": "Anomaly & attribution",
        "question_zh": "最近14天 GMV 为什么跌了？",
        "question_en": "Why did GMV drop in the last 14 days?",
    },
    {
        "id": "causal",
        "label_zh": "因果推断",
        "label_en": "Causal inference",
        "question_zh": "用倾向得分匹配评估促销补贴的效应",
        "question_en": "Estimate the promo subsidy effect with propensity matching",
    },
    {
        "id": "modeling",
        "label_zh": "分类模型",
        "label_en": "Classification model",
        "question_zh": "训练分类模型预测用户是否转化，看特征重要性",
        "question_en": "Train a conversion classifier and show feature importance",
    },
    {
        "id": "dashboard",
        "label_zh": "看板生成",
        "label_en": "Dashboard",
        "question_zh": "给我生成一个 GMV 监控看板",
        "question_en": "Generate a GMV monitoring dashboard",
    },
    {
        "id": "metric",
        "label_zh": "指标定义",
        "label_en": "Metric definitions",
        "question_zh": "注册指标：利润率 = (gmv - refund_amount) / gmv",
        "question_en": "Register metric profit_rate = (gmv - refund_amount) / gmv",
    },
    # --- supporting scenes outside the focused demo reel (still capturable) ---
    {
        "id": "sql",
        "label_zh": "自然语言查询",
        "label_en": "NL query",
        "question_zh": "近30天各渠道 GMV 排名",
        "question_en": "GMV by channel over the last 30 days",
    },
    {
        "id": "tests",
        "label_zh": "统计检验清单",
        "label_en": "Test battery",
        "question_zh": "异常检测用了哪些检验？",
        "question_en": "Which statistical tests does the anomaly readout run?",
    },
    {
        "id": "exp_design",
        "label_zh": "实验设计",
        "label_en": "Experiment design",
        "question_zh": "支付转化率实验，基线3%，检测2%相对提升，每天5万流量，怎么设计？",
        "question_en": "Design a checkout test: 3% baseline, 2% relative lift, 50k daily traffic",
    },
    {
        "id": "exp_readout",
        "label_zh": "实验分析",
        "label_en": "Experiment readout",
        "question_zh": "看下这个 AB 实验的结果",
        "question_en": "Read out the AB test results",
    },
    {
        "id": "switchback",
        "label_zh": "时间片轮转",
        "label_en": "Switchback",
        "question_zh": "看下时间片轮转实验的结果",
        "question_en": "Read out the switchback experiment",
    },
    {
        "id": "reversal",
        "label_zh": "反转实验",
        "label_en": "Reversal",
        "question_zh": "做一个反转实验验证这个效应",
        "question_en": "Validate the effect with a reversal experiment",
    },
    {
        "id": "nested",
        "label_zh": "父子实验",
        "label_en": "Parent-child",
        "question_zh": "父子实验怎么验证互不干扰",
        "question_en": "Check that parent and child experiments do not interfere",
    },
    {
        "id": "two_sided",
        "label_zh": "双边实验",
        "label_en": "Two-sided",
        "question_zh": "双边实验，买家侧和卖家侧同时干预",
        "question_en": "Two-sided test treating buyers and sellers at once",
    },
    {
        "id": "sensitivity",
        "label_zh": "敏感性分析",
        "label_en": "Sensitivity",
        "question_zh": "这个实验结论稳健吗，做个敏感性分析",
        "question_en": "Is the readout robust? Run a sensitivity analysis",
    },
    {
        "id": "did",
        "label_zh": "双重差分",
        "label_en": "Difference-in-differences",
        "question_zh": "用双重差分评估这次改版对 GMV 的影响",
        "question_en": "Evaluate the relaunch effect with difference-in-differences",
    },
    {
        "id": "psm",
        "label_zh": "倾向得分匹配",
        "label_en": "Propensity matching",
        "question_zh": "用倾向得分匹配评估促销补贴的效应",
        "question_en": "Estimate the promo subsidy effect with propensity matching",
    },
    {
        "id": "ts_causal",
        "label_zh": "时序因果",
        "label_en": "Time-series causal",
        "question_zh": "这次改版对 GMV 的影响，用时序因果评估",
        "question_en": "Estimate the relaunch effect with a time-series causal model",
    },
    {
        "id": "uplift",
        "label_zh": "Uplift 建模",
        "label_en": "Uplift modelling",
        "question_zh": "用因果森林看哪类用户对补贴最敏感",
        "question_en": "Which users respond most to the subsidy? Fit a causal forest",
    },
    {
        "id": "rdd",
        "label_zh": "断点回归",
        "label_en": "Regression discontinuity",
        "question_zh": "用断点回归评估会员门槛的效果",
        "question_en": "Evaluate the membership threshold with a regression discontinuity",
    },
    {
        "id": "oss",
        "label_zh": "开源引擎交叉验证",
        "label_en": "OSS cross-validation",
        "question_zh": "用 DoWhy 开源包交叉验证促销的因果效应",
        "question_en": "Cross-validate the promo effect with the DoWhy package",
    },
    {
        "id": "dl_anomaly",
        "label_zh": "深度异常检测",
        "label_en": "Deep anomaly",
        "question_zh": "用自编码器看下 GMV 有没有异常",
        "question_en": "Check GMV anomalies with an autoencoder",
    },
    {
        "id": "modeling",
        "label_zh": "建模能力",
        "label_en": "Modelling",
        "question_zh": "训练分类模型预测用户是否转化，看特征重要性",
        "question_en": "Train a conversion classifier and show feature importance",
    },
    {
        "id": "dashboard",
        "label_zh": "监控看板",
        "label_en": "Dashboard",
        "question_zh": "给我生成一个 GMV 监控看板",
        "question_en": "Generate a GMV monitoring dashboard",
    },
    {
        "id": "metric",
        "label_zh": "指标字典",
        "label_en": "Metric dictionary",
        "question_zh": "注册指标：利润率 = (gmv - refund_amount) / gmv",
        "question_en": "Register metric profit_rate = (gmv - refund_amount) / gmv",
    },
    {
        "id": "datasource",
        "label_zh": "数据源接入",
        "label_en": "Data sources",
        "question_zh": "数据源有哪些？",
        "question_en": "Which data sources are supported?",
    },
    {
        "id": "knowledge",
        "label_zh": "业务知识 RAG",
        "label_en": "Knowledge RAG",
        "question_zh": "GMV 的口径是什么？",
        "question_en": "What is the GMV caliber?",
    },
    {
        "id": "method",
        "label_zh": "方法选型",
        "label_en": "Method selection",
        "question_zh": "没有对照组，只有上线前后的数据，该用什么方法评估效果？",
        "question_en": "No control group, only before/after data - which method applies?",
    },
    {
        "id": "report",
        "label_zh": "报告撰写",
        "label_en": "Report",
        "question_zh": "把分析写成周报",
        "question_en": "Write this up as a weekly report",
    },
]

_SCENE_IDS = [s["id"] for s in SCENES]

# The focused demo reel (what the GIF shows): sampling theory, anomaly +
# attribution, causal inference, classification, dashboard, metric definitions.
DEMO_IDS = ["sampling", "diagnose", "causal", "modeling", "dashboard", "metric"]

# Markdown noise that would be dropped anyway when drawing pixels.
_STRIP = re.compile(r"\*\*|`|^#+\s*|^>\s*")


def _clean(line: str) -> str:
    return _STRIP.sub("", line).strip()


def _compress_table(block: Dict[str, Any], max_rows: int = 4, max_cols: int = 4) -> Dict[str, Any]:
    cols = [str(c) for c in block.get("columns", [])][:max_cols]
    rows = [[str(c) for c in row][:max_cols] for row in block.get("rows", [])[:max_rows]]
    return {
        "title": _clean(str(block.get("title", ""))),
        "columns": cols,
        "rows": rows,
        "n_rows": len(block.get("rows", [])),
        "n_cols": len(block.get("columns", [])),
    }


def _save_chart(image: str, name: str, idx: int) -> str:
    """Decode the first chart data-URL to PNG so the renderer can paste it."""
    if not image or not image.startswith("data:image"):
        return ""
    import base64

    head, _, payload = image.partition(",")
    ext = "png" if "png" in head else "jpg"
    path = OUT_DIR / f"{name}_{idx}.{ext}"
    try:
        path.write_bytes(base64.b64decode(payload))
    except Exception:  # noqa: BLE001 - a broken chart must not kill the capture
        return ""
    return str(path.relative_to(ROOT))


def _summarise(reply: Any, scene: Dict[str, str], elapsed: float, lang: str) -> Dict[str, Any]:
    paragraphs = [p for p in re.split(r"\n\s*\n", reply.text or "") if p.strip()]
    headline = _clean(paragraphs[0].split("\n")[0]) if paragraphs else ""

    # Bullets: explicit list items first (they are the agent's own highlights),
    # then short single-line sentences that carry a number.
    bullets: List[str] = []
    for line in (reply.text or "").split("\n"):
        stripped = line.strip()
        if stripped.startswith(("- ", "• ", "* ")):
            bullets.append(_clean(stripped[2:]))
    if len(bullets) < 3:
        for para in paragraphs[1:6]:
            for line in para.split("\n"):
                text = _clean(line)
                if 8 <= len(text) <= 64 and re.search(r"\d", text):
                    bullets.append(text)
                if len(bullets) >= 4:
                    break
    bullets = [b[:64] for b in bullets[:4]]

    kpi: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    charts = 0
    chart_path = ""
    for block in reply.blocks or []:
        kind = block.get("type")
        if kind == "kpi":
            for item in block.get("items", [])[:3]:
                kpi.append(
                    {
                        "label": _clean(str(item.get("label", "")))[:18],
                        "value": _clean(str(item.get("value", "")))[:16],
                        "sub": _clean(str(item.get("sub", "")))[:22],
                        "good": item.get("good"),
                    }
                )
        elif kind == "table" and len(tables) < 2:
            tables.append(_compress_table(block))
        elif kind == "chart":
            charts += 1
            if not chart_path:
                chart_path = _save_chart(str(block.get("image", "")), scene["id"], charts)

    return {
        "id": scene["id"],
        "label_zh": scene["label_zh"],
        "label_en": scene["label_en"],
        "question_zh": scene["question_zh"],
        "question_en": scene["question_en"],
        "lang": lang,
        "intent": (reply.data or {}).get("intent", ""),
        "confidence": (reply.data or {}).get("confidence"),
        "planner": (reply.data or {}).get("planner", "rules"),
        "headline": headline,
        "bullets": bullets,
        "kpi": kpi[:3],
        "tables": tables,
        "n_charts": charts,
        "chart_png": chart_path,
        "n_blocks": len(reply.blocks or []),
        "n_tools": max(0, len(reply.tool_calls or []) - 1),
        "tools": [tc.get("tool") for tc in (reply.tool_calls or [])[1:7]],
        "artifacts": [Path(a).name for a in (reply.artifacts or [])],
        "text_chars": len(reply.text or ""),
        "elapsed_s": round(elapsed, 2),
    }


def capture(scene_id: str, lang: str = "zh") -> Dict[str, Any]:
    from ds_agent.agent import DSQueryAgent

    scene = next(s for s in SCENES if s["id"] == scene_id)
    question = scene["question_zh"] if lang == "zh" else scene["question_en"]

    agent = DSQueryAgent()
    if scene_id == "report":
        # A report summarises the session's findings, so give it a session.
        for seed in ("最近14天 GMV 为什么跌了？", "看下这个 AB 实验的结果", "用双重差分评估这次改版对 GMV 的影响"):
            try:
                agent.ask(seed)
            except Exception:  # noqa: BLE001 - seeds are best-effort context
                pass
    if scene_id == "datasource":
        # Attach a real offline sheet first so the screen shows a live
        # registration (with a preview table), not the "nothing attached" hint.
        import pandas as pd

        budget = ROOT / "artifacts" / "core_demo_workspace" / "budget_2026H2.csv"
        budget.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"month": ["2026-07", "2026-08", "2026-09"], "budget": [8_100_000, 8_300_000, 8_600_000]}
        ).to_csv(budget, index=False)
        question = f"关联数据源 {budget} 叫 budget"
        scene = {
            **scene,
            "question_zh": "关联数据源 budget_2026H2.csv 叫 budget",
            "question_en": "Attach data source budget_2026H2.csv as budget",
        }

    started = time.time()
    reply = agent.ask(question)
    elapsed = time.time() - started

    if scene_id == "metric":
        # The scene registers profit_rate; a leftover from an earlier run would
        # get the registration rejected, so start from a clean slate.
        for stale in ("利润率", "profit_rate", "register_metric_profit_rate"):
            try:
                agent.ask(f"下线指标 {stale}")
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass

    record = _summarise(reply, scene, elapsed, lang)
    record["text"] = reply.text or ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if lang == "zh" else "_en"
    path = OUT_DIR / f"{scene_id}{suffix}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"[{scene_id}{suffix}] intent={record['intent']} {elapsed:.1f}s "
        f"kpi={len(record['kpi'])} tables={len(record['tables'])} charts={record['n_charts']} "
        f"-> {path.relative_to(ROOT)}"
    )
    print(f"  headline: {record['headline'][:90]}")
    return record


def merge() -> None:
    records: List[Dict[str, Any]] = []
    en_records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for order, scene_id in enumerate(DEMO_IDS, start=1):
        scene = next(s for s in SCENES if s["id"] == scene_id)
        path = OUT_DIR / f"{scene_id}.json"
        if not path.exists():
            missing.append(scene_id)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["order"] = order
        records.append(data)
        en_path = OUT_DIR / f"{scene_id}_en.json"
        if en_path.exists():
            en = json.loads(en_path.read_text(encoding="utf-8"))
            en["order"] = order
            en_records.append(en)
        else:
            missing.append(f"{scene_id}_en")
    payload: Dict[str, Any] = {"scenes": records, "en_scenes": en_records}
    MERGED.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"wrote {MERGED.relative_to(ROOT)} "
        f"({len(records)} zh + {len(en_records)} en scenes of {len(DEMO_IDS)})"
    )
    if missing:
        print("missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="scene id to capture", choices=_SCENE_IDS)
    parser.add_argument("--lang", default="zh", choices=["zh", "en"])
    parser.add_argument("--merge", action="store_true", help="merge captured scenes into one transcript")
    parser.add_argument("--list", action="store_true", help="print the scene ids")
    args = parser.parse_args()

    if args.list or (not args.scene and not args.merge):
        for i, scene in enumerate(SCENES, start=1):
            print(f"{i:2d}. {scene['id']:<12} {scene['label_zh']:<12} {scene['question_zh']}")
        return
    if args.merge:
        merge()
        return
    capture(args.scene, args.lang)


if __name__ == "__main__":
    main()
