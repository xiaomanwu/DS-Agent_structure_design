"""Render the full-capability demo GIF (docs/demo.gif) with Pillow only.

The animation walks every capability the agent routes to, and every frame is
drawn from real agent output captured by ``scripts/capture_demo_turns.py``
(KPIs, tables, charts and prose come straight out of ``DSQueryAgent.ask``).
No ffmpeg is required: GIF is written by Pillow.

    python scripts/capture_demo_turns.py --scene sql      # one scene at a time
    python scripts/capture_demo_turns.py --merge
    python scripts/make_demo_gif.py

Scenes: NL query, diagnosis, test battery, experiment design + readout, the
four advanced designs (switchback / reversal / nested / two-sided), sensitivity,
DiD / PSM / time-series causal / uplift / RDD / OSS cross-validation, deep
anomaly, modelling, dashboard, metric dictionary, data sources, knowledge RAG,
method selection, report, plus the EN toggle and the LLM settings panel.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = ROOT / "artifacts" / "demo_transcript.json"
OUT = ROOT / "docs" / "demo.gif"

W, H = 900, 600
SIDE_W = 196
HEADER_H = 46
COMPOSER_H = 80
FPS = 8

INK = (15, 23, 42)
BODY = (51, 65, 85)
MUTED = (100, 116, 139)
FAINT = (148, 163, 184)
LINE = (226, 232, 240)
BRAND = (37, 99, 235)
BRAND_SOFT = (239, 246, 255)
BG = (245, 247, 251)
CARD = (255, 255, 255)
SOFT = (248, 250, 252)
RED = (220, 38, 38)
GREEN = (22, 163, 74)
PURPLE = (124, 58, 237)

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

GROUPS: Dict[str, Tuple[str, str]] = {
    "sampling": ("统计抽样", "Statistical sampling"),
    "diagnose": ("异常与归因", "Anomaly & attribution"),
    "causal": ("因果推断", "Causal inference"),
    "modeling": ("分类模型", "Classification model"),
    "dashboard": ("看板生成", "Dashboard"),
    "metric": ("指标定义", "Metric definitions"),
    "sql": ("查询与诊断", "Query & diagnosis"),
    "tests": ("查询与诊断", "Query & diagnosis"),
    "exp_design": ("实验设计与分析", "Experiment design & readout"),
    "exp_readout": ("实验设计与分析", "Experiment design & readout"),
    "switchback": ("实验设计与分析", "Experiment design & readout"),
    "reversal": ("实验设计与分析", "Experiment design & readout"),
    "nested": ("实验设计与分析", "Experiment design & readout"),
    "two_sided": ("实验设计与分析", "Experiment design & readout"),
    "sensitivity": ("实验设计与分析", "Experiment design & readout"),
    "did": ("因果推断", "Causal inference"),
    "psm": ("因果推断", "Causal inference"),
    "ts_causal": ("因果推断", "Causal inference"),
    "uplift": ("因果推断", "Causal inference"),
    "rdd": ("因果推断", "Causal inference"),
    "oss": ("因果推断", "Causal inference"),
    "dl_anomaly": ("异常与建模", "Anomaly & modelling"),
    "modeling": ("异常与建模", "Anomaly & modelling"),
    "dashboard": ("异常与建模", "Anomaly & modelling"),
    "metric": ("数据治理", "Data governance"),
    "datasource": ("数据治理", "Data governance"),
    "knowledge": ("数据治理", "Data governance"),
    "method": ("数据治理", "Data governance"),
    "report": ("交付", "Delivery"),
}

T = {
    "zh": {
        "tagline": "自然语言数据分析",
        "caps": "能力",
        "chat": "对话",
        "ph": "例如：近14天 GMV 为什么跌了？",
        "hint": "Enter 发送 · Shift+Enter 换行",
        "tools": "工具调用",
        "files": "生成文件",
        "thinking": "正在拆解问题、调用工具…",
        "built": "内置演示数仓 · DuckDB",
        "llm": "内置规则模式",
        "llm_on": "LLM 已连接",
        "intro_title": "DS-Agent",
        "intro": [
            "用自然语言提问，Agent 自己选工具、跑检验、写结论。",
            "{n} 个核心场景 · 先中文完整演示，再切换英文完整重演一遍。",
        ],
        "outro_title": "{n} 个核心场景 · 中英双语演示完毕",
        "outro_sub": "抽样 / 异动归因 / 因果推断 / 分类模型 / 看板 / 指标定义，全部真实运行输出",
        "scenes": "场景",
        "checks": "统计检验",
    },
    "en": {
        "tagline": "Analytics in natural language",
        "caps": "Capabilities",
        "chat": "Chat",
        "ph": "e.g. Why did GMV drop in the last 14 days?",
        "hint": "Enter to send · Shift+Enter for a new line",
        "tools": "Tool calls",
        "files": "Artifacts",
        "thinking": "Planning, calling tools…",
        "built": "Built-in demo warehouse · DuckDB",
        "llm": "Rules mode",
        "llm_on": "LLM connected",
        "intro_title": "DS-Agent",
        "intro": [
            "Ask in plain language; the agent picks tools, runs tests, writes the conclusion.",
            "{n} core scenarios - a full pass in Chinese first, then a full pass in English.",
        ],
        "outro_title": "{n} core scenarios demoed in both languages",
        "outro_sub": "Sampling / anomaly & attribution / causal / classification / dashboard / metric definitions",
        "scenes": "scenes",
        "checks": "tests",
    },
}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F = {size: load_font(size) for size in (7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20)}


def text(d: ImageDraw.ImageDraw, xy, s, size=12, fill=INK, anchor="la", bold=False) -> None:
    if bold:
        d.text(xy, s, font=F[size], fill=fill, anchor=anchor, stroke_width=1, stroke_fill=fill)
    else:
        d.text(xy, s, font=F[size], fill=fill, anchor=anchor)


def wrap(d: ImageDraw.ImageDraw, s: str, size: int, max_w: float) -> List[str]:
    """Character-level wrap: the content mixes CJK and Latin, so word wrap
    would leave huge gaps and (for Chinese) never break at all."""
    lines: List[str] = []
    cur = ""
    for ch in s:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if cur and d.textlength(cur + ch, font=F[size]) > max_w:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def fit(d: ImageDraw.ImageDraw, s: str, size: int, max_w: float, max_lines: int) -> List[str]:
    if not s:
        return [""]
    lines = wrap(d, s, size, max_w)
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    last = kept[-1]
    while last and d.textlength(last + "…", font=F[size]) > max_w:
        last = last[:-1]
    kept[-1] = last + "…"
    return kept


def rounded(d, box, radius, **kw) -> None:
    d.rounded_rectangle(box, radius=radius, **kw)


def delta_color(value: str) -> Tuple[int, int, int]:
    """Up is red, down is green - the same convention the product uses."""
    s = value.strip()
    if s.startswith("-"):
        return GREEN
    if s.startswith("+") or s.startswith("↑"):
        return RED
    return INK


# --------------------------------------------------------------------- chrome
def sidebar(d, lang: str, scenes: Sequence[Dict[str, Any]], current: Optional[int]) -> None:
    d.rectangle([0, 0, SIDE_W, H], fill=CARD)
    d.line([SIDE_W, 0, SIDE_W, H], fill=LINE)
    rounded(d, [16, 16, 46, 46], 9, fill=PURPLE)
    text(d, (31, 31), "DS", 12, CARD, "mm", bold=True)
    text(d, (54, 17), "DS-Agent", 14, bold=True)
    text(d, (54, 35), T[lang]["tagline"], 9, MUTED)

    text(d, (16, 68), T[lang]["caps"].upper(), 9, FAINT, bold=True)

    # The list is longer than the panel: show a window around the active scene
    # so the sidebar "scrolls" as the demo advances.
    rows = 20
    top, bottom = 82, H - 34
    visible = (bottom - top) // rows
    total = len(scenes)
    start = 0
    if current is not None:
        start = max(0, min(current - visible // 2, max(0, total - visible)))
    for i in range(start, min(total, start + visible)):
        scene = scenes[i]
        y = top + (i - start) * rows
        label = scene["label_zh"] if lang == "zh" else scene["label_en"]
        if current is not None and i == current:
            rounded(d, [10, y - 3, SIDE_W - 10, y + 15], 7, fill=BRAND_SOFT)
            d.rectangle([10, y - 3, 13, y + 15], fill=BRAND)
            text(d, (20, y), f"{i + 1:02d} {label}", 10, BRAND, bold=True)
        else:
            text(d, (20, y), f"{i + 1:02d} {label}", 10, BODY)
    if start > 0:
        text(d, (SIDE_W // 2, top - 12), "· · ·", 10, FAINT, "ma")
    if start + visible < total:
        text(d, (SIDE_W // 2, top + visible * rows + 2), "· · ·", 10, FAINT, "ma")

    text(d, (16, H - 26), T[lang]["built"], 9, FAINT)


def topbar(d, lang: str, llm_on: bool, title: str, sub: str, progress: float) -> None:
    d.rectangle([SIDE_W, 0, W, HEADER_H], fill=CARD)
    d.line([SIDE_W, HEADER_H, W, HEADER_H], fill=LINE)
    text(d, (SIDE_W + 18, 10), title, 13, bold=True)
    sub_x = SIDE_W + 18 + d.textlength(title, font=F[13]) + 8
    text(d, (sub_x, 15), sub.upper(), 8, FAINT)

    # language toggle
    rounded(d, [W - 214, 12, W - 158, 34], 8, fill=SOFT, outline=LINE)
    if lang == "zh":
        rounded(d, [W - 213, 13, W - 187, 33], 7, fill=BRAND)
        text(d, (W - 200, 23), "中文", 9, CARD, "mm", bold=True)
        text(d, (W - 172, 23), "EN", 9, MUTED, "mm")
    else:
        rounded(d, [W - 185, 13, W - 159, 33], 7, fill=BRAND)
        text(d, (W - 200, 23), "中文", 9, MUTED, "mm")
        text(d, (W - 172, 23), "EN", 9, CARD, "mm", bold=True)

    # LLM status
    label = T[lang]["llm_on"] if llm_on else T[lang]["llm"]
    d.ellipse([W - 140, 18, W - 132, 26], fill=GREEN if llm_on else FAINT)
    text(d, (W - 126, 17), label, 9, MUTED)

    # progress bar
    d.rectangle([SIDE_W, HEADER_H - 2, W, HEADER_H], fill=LINE)
    d.rectangle([SIDE_W, HEADER_H - 2, SIDE_W + int((W - SIDE_W) * progress), HEADER_H], fill=BRAND)


def composer(d, lang: str, draft: str, enabled: bool = True) -> None:
    top = H - COMPOSER_H
    d.rectangle([SIDE_W, top, W, H], fill=CARD)
    d.line([SIDE_W, top, W, top], fill=LINE)
    rounded(d, [SIDE_W + 16, top + 14, W - 16, top + 54], 12, fill=CARD, outline=LINE, width=1)
    lines = fit(d, draft, 11, W - SIDE_W - 100, 2) if draft else []
    if lines:
        for i, line in enumerate(lines):
            text(d, (SIDE_W + 30, top + 20 + i * 16), line, 11, INK if enabled else MUTED)
    else:
        text(d, (SIDE_W + 30, top + 26), T[lang]["ph"], 11, FAINT)
    rounded(d, [W - 56, top + 20, W - 26, top + 48], 9, fill=BRAND if draft else (203, 213, 225))
    text(d, (W - 41, top + 34), "↑", 13, CARD, "mm", bold=True)
    text(d, ((SIDE_W + W) // 2, H - 14), T[lang]["hint"], 9, FAINT, "ma")


def modal(d, lang: str, filled: float) -> None:
    mw, mh = 420, 292
    mx, my = (W - mw) // 2, (H - mh) // 2
    overlay = Image.new("RGBA", (W, H), (15, 23, 42, 96))
    merged = Image.alpha_composite(d._image.convert("RGBA"), overlay).convert("RGB")  # type: ignore[attr-defined]
    d._image.paste(merged)  # type: ignore[attr-defined]
    rounded(d, [mx, my, mx + mw, my + mh], 16, fill=CARD)
    title = "大模型接口设置" if lang == "zh" else "LLM API settings"
    text(d, (mx + 22, my + 18), title, 14, bold=True)
    sub = (
        "预留接口：规划、叙述、生成代码；留空则用内置规则引擎"
        if lang == "zh"
        else "Planner, narration and code generation; empty means the rule engine"
    )
    text(d, (mx + 22, my + 40), sub, 9, MUTED)
    fields = ["Base URL", "API Key", "Model"]
    values = ["https://api.deepseek.com/v1", "sk-********", "deepseek-chat"]
    for i, fname in enumerate(fields):
        fy = my + 64 + i * 58
        text(d, (mx + 22, fy), fname, 9, MUTED)
        rounded(d, [mx + 22, fy + 14, mx + mw - 22, fy + 44], 8, fill=SOFT, outline=LINE)
        reveal = min(max(filled * 3 - i, 0.0), 1.0)
        if reveal > 0:
            text(d, (mx + 32, fy + 28), values[i][: max(1, int(len(values[i]) * reveal))], 10, BODY)
    rounded(d, [mx + mw - 148, my + mh - 46, mx + mw - 92, my + mh - 16], 8, fill=BRAND)
    text(d, (mx + mw - 120, my + mh - 31), "保存" if lang == "zh" else "Save", 10, CARD, "mm", bold=True)
    rounded(d, [mx + mw - 214, my + mh - 46, mx + mw - 156, my + mh - 16], 8, fill=(241, 245, 249))
    text(d, (mx + mw - 185, my + mh - 31), "测试" if lang == "zh" else "Test", 10, BODY, "mm")


# ---------------------------------------------------------------- scene body
def kpi_row(d, y: int, kpis: Sequence[Dict[str, Any]]) -> int:
    if not kpis:
        return y
    x0, x1 = SIDE_W + 18, W - 18
    n = min(len(kpis), 3)
    gap = 8
    cw = (x1 - x0 - gap * (n - 1)) // n
    for i in range(n):
        item = kpis[i]
        x = x0 + i * (cw + gap)
        rounded(d, [x, y, x + cw, y + 54], 10, fill=SOFT, outline=LINE)
        text(d, (x + 10, y + 7), item.get("label", ""), 9, MUTED)
        value = item.get("value", "")
        size = 15 if len(value) <= 11 else 12
        text(d, (x + 10, y + 22), value, size, delta_color(value), bold=True)
        sub = item.get("sub", "")
        if sub:
            text(d, (x + 10, y + 40), fit(d, sub, 8, cw - 16, 1)[0], 8, FAINT)
    return y + 54 + 8


def table_card(d, box, table: Dict[str, Any], max_rows: int) -> None:
    x0, y0, x1, y1 = box
    rounded(d, [x0, y0, x1, y1], 10, fill=CARD, outline=LINE)
    title = table.get("title") or ""
    if title:
        text(d, (x0 + 10, y0 + 6), fit(d, title, 9, x1 - x0 - 20, 1)[0], 9, MUTED)
    header_y = y0 + 20
    cols = table.get("columns", [])
    rows = table.get("rows", [])[:max_rows]
    n = max(1, len(cols))
    col_w = (x1 - x0 - 16) / n
    d.rectangle([x0 + 1, header_y, x1 - 1, header_y + 17], fill=SOFT)
    for j, col in enumerate(cols):
        text(d, (x0 + 8 + j * col_w, header_y + 4), fit(d, str(col), 8, col_w - 8, 1)[0], 8, MUTED, bold=True)
    for i, row in enumerate(rows):
        ry = header_y + 18 + i * 16
        if ry + 14 > y1 - 2:
            break
        if i % 2:
            d.rectangle([x0 + 1, ry, x1 - 1, ry + 16], fill=(252, 253, 255))
        for j, cell in enumerate(row):
            cell = str(cell)
            color = INK
            if j > 0 and cell.startswith("-") and any(ch.isdigit() for ch in cell):
                color = GREEN
            elif j > 0 and cell.startswith("+") and any(ch.isdigit() for ch in cell):
                color = RED
            text(d, (x0 + 8 + j * col_w, ry + 2), fit(d, cell, 8, col_w - 8, 1)[0], 8, color)


def chart_card(d, box, chart_path: Path, caption: str) -> None:
    x0, y0, x1, y1 = box
    rounded(d, [x0, y0, x1, y1], 10, fill=CARD, outline=LINE)
    if caption:
        text(d, (x0 + 10, y0 + 6), fit(d, caption, 9, x1 - x0 - 20, 1)[0], 9, MUTED)
    try:
        img = Image.open(chart_path).convert("RGB")
    except Exception:  # noqa: BLE001 - a missing chart only drops the thumbnail
        return
    inner_w, inner_h = x1 - x0 - 12, y1 - y0 - 26
    ratio = min(inner_w / img.width, inner_h / img.height)
    img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.LANCZOS)
    d._image.paste(img, (x0 + 6, y0 + 20))  # type: ignore[attr-defined]


def strip(d, y: int, label: str, items: Sequence[str], color=MUTED) -> int:
    if not items:
        return y
    text(d, (SIDE_W + 18, y + 3), label, 8, FAINT)
    x = SIDE_W + 18 + d.textlength(label, font=F[8]) + 8
    for item in items:
        name = str(item)[:26]
        chip_w = d.textlength(name, font=F[8]) + 14
        if x + chip_w > W - 18:
            break
        rounded(d, [x, y, x + chip_w, y + 15], 7, fill=SOFT, outline=LINE)
        text(d, (x + 7, y + 4), name, 8, color)
        x += chip_w + 5
    return y + 18


def pick_headline(scene: Dict[str, Any]) -> str:
    """The first prose paragraph is not always the conclusion (e.g. the
    experiment readout opens with the SRM check); prefer the first paragraph
    that actually carries the finding."""
    import re

    fallback = scene.get("headline") or ""
    text = scene.get("text") or ""
    paras = [
        re.sub(r"\*\*|`|^#+\s*", "", p.strip())
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]
    if not paras:
        return fallback
    finding = re.compile(
        r"抽样|样本量|代表性|标准误|主指标|相对提升|相对下降|环比|效应|lift|ATT|Qini|建议|结论|显著"
        r"|最优模型|best model|accuracy|roc_auc|auc|看板|dashboard|已注册|registered"
    )
    for para in paras[:6]:
        line = para.split("\n")[0]
        if finding.search(line) and re.search(r"\d", line):
            return line
    return paras[0].split("\n")[0]


def _shorten_paths(s: str) -> str:
    """Collapse absolute paths so a filename stays readable on screen."""
    import re

    return re.sub(r"/[^\s：:，,]+/", "", s or "")


def answer_body(d, scene: Dict[str, Any], lang: str, stage: int) -> None:
    x0, x1 = SIDE_W + 18, W - 18
    q = scene["question_zh"] if lang == "zh" else scene["question_en"]
    qlines = fit(d, q, 11, 430, 2)
    qh = 12 + len(qlines) * 15
    qw = max(d.textlength(line, font=F[11]) for line in qlines) + 26
    qw = min(max(qw, 90), 460)
    rounded(d, [x1 - qw, 58, x1, 58 + qh], 11, fill=BRAND)
    for i, line in enumerate(qlines):
        text(d, (x1 - qw + 13, 64 + i * 15), line, 11, CARD)

    y = 58 + qh + 10
    intent = scene.get("intent") or ""
    text(d, (x0, y), "DS-Agent", 9, MUTED, bold=True)
    if intent:
        chip_w = d.textlength(intent, font=F[8]) + 12
        rounded(d, [x0 + 46, y - 1, x0 + 46 + chip_w, y + 13], 6, fill=BRAND_SOFT)
        text(d, (x0 + 52, y + 1), intent, 8, BRAND)
    meta = f"{scene.get('n_tools', 0)} tool calls · {scene.get('elapsed_s', 0):.1f}s"
    if scene.get("n_charts"):
        meta += f" · {scene['n_charts']} charts"
    text(d, (x1, y + 1), meta, 8, FAINT, "ra")
    y += 18

    if stage == 0:
        rounded(d, [x0, y, x0 + 76, y + 26], 12, fill=CARD, outline=LINE)
        for i in range(3):
            d.ellipse([x0 + 18 + i * 16, y + 10, x0 + 24 + i * 16, y + 16], fill=FAINT)
        text(d, (x0 + 88, y + 7), T[lang]["thinking"], 9, FAINT)
        return

    # Prose: the agent's own headline plus up to three numeric highlights.
    headline = _shorten_paths(pick_headline(scene))
    hlines = fit(d, headline, 11, x1 - x0 - 28, 2)
    bullets: List[str] = []
    head_key = re.sub(r"\W", "", headline)[:18]
    for bullet in scene.get("bullets", []):
        if len(bullets) == 3:
            break
        if re.sub(r"\W", "", bullet)[:18] == head_key:
            continue  # the capture heuristic can echo the headline; skip the dupe
        bullets.append(fit(d, _shorten_paths(bullet), 9, x1 - x0 - 44, 1)[0])
    card_h = 10 + len(hlines) * 16 + len(bullets) * 15 + 8
    rounded(d, [x0, y, x1, y + card_h], 11, fill=CARD, outline=LINE)
    d.rectangle([x0, y + 10, x0 + 3, y + 10 + len(hlines) * 16 - 4], fill=BRAND)
    for i, line in enumerate(hlines):
        text(d, (x0 + 12, y + 8 + i * 16), line, 11, INK, bold=(i == 0))
    by = y + 8 + len(hlines) * 16 + 2
    for i, line in enumerate(bullets):
        d.ellipse([x0 + 14, by + i * 15 + 5, x0 + 18, by + i * 15 + 9], fill=FAINT)
        text(d, (x0 + 24, by + i * 15), line, 9, BODY)
    y += card_h + 8

    if stage == 1:
        return

    y = kpi_row(d, y, scene.get("kpi", []))

    # Table (left) and chart thumbnail (right) share one row.
    table = (scene.get("tables") or [None])[0]
    chart_path = ROOT / scene["chart_png"] if scene.get("chart_png") else None
    has_chart = bool(chart_path and chart_path.exists())
    row_h = min(150, H - COMPOSER_H - 24 - y)
    if table and row_h >= 60:
        chart_w = 236 if has_chart else 0
        tx1 = x1 - chart_w - (10 if has_chart else 0)
        rows_fit = max(1, (row_h - 40) // 16)
        table_card(d, (x0, y, tx1, y + row_h), table, rows_fit)
        if has_chart:
            chart_card(d, (tx1 + 10, y, x1, y + row_h), chart_path, f"chart · {scene['n_charts']}")
        y += row_h + 6
    elif has_chart and row_h >= 60:
        chart_card(d, (x0, y, x1, y + row_h), chart_path, f"chart · {scene['n_charts']}")
        y += row_h + 6

    tools = scene.get("tools") or []
    y = strip(d, y + 2, T[lang]["tools"], tools)
    if scene.get("artifacts"):
        strip(d, y, T[lang]["files"], scene["artifacts"], BRAND)


# ------------------------------------------------------------------- frames
def frame(
    scenes: Sequence[Dict[str, Any]],
    index: Optional[int] = None,
    stage: int = 2,
    lang: str = "zh",
    llm_on: bool = False,
    modal_fill: float = 0.0,
    special: str = "",
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    current = index if index is not None else None
    if index is not None and index < len(scenes):
        group = GROUPS.get(scenes[index]["id"], ("", ""))
        title = scenes[index]["label_zh"] if lang == "zh" else scenes[index]["label_en"]
        sub = f"{group[0] if lang == 'zh' else group[1]} · {index + 1}/{len(scenes)}"
        progress = (index + 1) / len(scenes)
        draft = scenes[index]["question_zh"] if lang == "zh" else scenes[index]["question_en"]
    else:
        title, sub, progress, draft = "", "", 0.0, ""

    sidebar(d, lang, scenes, current)
    topbar(d, lang, llm_on, title, sub, progress if index is not None else 0.0)

    if special == "intro":
        y = 76
        rounded(d, [SIDE_W + 18, y, W - 18, y + 150], 12, fill=CARD, outline=LINE)
        rounded(d, [SIDE_W + 34, y + 18, SIDE_W + 66, y + 50], 9, fill=PURPLE)
        text(d, (SIDE_W + 50, y + 34), "DS", 12, CARD, "mm", bold=True)
        text(d, (SIDE_W + 80, y + 20), T[lang]["intro_title"], 15, bold=True)
        text(d, (SIDE_W + 80, y + 40), T[lang]["tagline"], 9, MUTED)
        for i, line in enumerate(T[lang]["intro"]):
            line = line.replace("{n}", str(len(scenes)))
            text(d, (SIDE_W + 34, y + 68 + i * 18), fit(d, line, 10, W - SIDE_W - 90, 1)[0], 10, BODY)
        chips = ["sampling", "anomaly+attribution", "causal", "classification", "dashboard", "metric dict"]
        x = SIDE_W + 34
        for chip in chips:
            cw = d.textlength(chip, font=F[9]) + 16
            rounded(d, [x, y + 112, x + cw, y + 132], 7, fill=BRAND_SOFT)
            text(d, (x + 8, y + 116), chip, 9, BRAND)
            x += cw + 6
        text(d, (SIDE_W + 18, y + 170), T[lang]["outro_sub"], 9, FAINT)
    elif special == "outro":
        y = 70
        text(d, (SIDE_W + 18, y), T[lang]["outro_title"].replace("{n}", str(len(scenes))), 15, bold=True)
        text(d, (SIDE_W + 18, y + 24), fit(d, T[lang]["outro_sub"], 9, W - SIDE_W - 40, 1)[0], 9, MUTED)
        gy = y + 52
        cols, gap = 4, 8
        cw = (W - 18 - SIDE_W - 18 - gap * (cols - 1)) // cols
        for i, scene in enumerate(scenes):
            r, c = divmod(i, cols)
            x = SIDE_W + 18 + c * (cw + gap)
            yy = gy + r * 40
            rounded(d, [x, yy, x + cw, yy + 34], 8, fill=CARD, outline=LINE)
            d.ellipse([x + 7, yy + 10, x + 21, yy + 24], fill=(220, 252, 231))
            text(d, (x + 14, yy + 17), "✓", 9, GREEN, "mm", bold=True)
            label = scene["label_zh"] if lang == "zh" else scene["label_en"]
            text(d, (x + 27, yy + 6), f"{i + 1:02d}", 7, FAINT)
            text(d, (x + 27, yy + 16), fit(d, label, 9, cw - 34, 1)[0], 9, INK)
    elif index is not None:
        answer_body(d, scenes[index], lang, stage)

    composer(d, lang, draft)
    if modal_fill > 0:
        modal(d, lang, modal_fill)
    return img


def load_scenes() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not TRANSCRIPT.exists():
        raise SystemExit(
            "missing artifacts/demo_transcript.json - run "
            "`python scripts/capture_demo_turns.py --scene <id>` for each scene, then `--merge`"
        )
    payload = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    scenes = payload.get("scenes") or []
    if not scenes:
        raise SystemExit("transcript has no scenes")
    en_scenes = payload.get("en_scenes") or []
    return scenes, en_scenes


def preview(ids: List[str]) -> None:
    """Dump PNG stills so the layout can be inspected without watching a GIF."""
    scenes, en_scenes = load_scenes()
    out_dir = ROOT / "artifacts" / "demo_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [i for i, s in enumerate(scenes) if not ids or s["id"] in ids]
    for i in wanted:
        frame(scenes, i, stage=2).save(out_dir / f"{i + 1:02d}_{scenes[i]['id']}.png")
        if i < len(en_scenes):
            frame(en_scenes, i, stage=2, lang="en").save(
                out_dir / f"{i + 1:02d}_{scenes[i]['id']}_en.png"
            )
    frame(scenes, special="intro").save(out_dir / "00_intro.png")
    frame(scenes, special="outro").save(out_dir / "99_outro.png")
    print(f"wrote previews to {out_dir.relative_to(ROOT)}")


def main() -> None:
    import sys

    if "--preview" in sys.argv:
        rest = sys.argv[sys.argv.index("--preview") + 1 :]
        preview([a for a in rest if not a.startswith("-")])
        return
    scenes, en_scenes = load_scenes()

    timeline: List[Tuple[Image.Image, int]] = []

    def add(img: Image.Image, ms: int) -> None:
        timeline.append((img, ms))

    # Pass 1: full Chinese demo of every scene.
    add(frame(scenes, special="intro"), 1500)
    for i in range(len(scenes)):
        add(frame(scenes, i, stage=0), 620)
        add(frame(scenes, i, stage=1), 700)
        add(frame(scenes, i, stage=2), 2100)

    # Toggle to English and replay every scene from a genuine EN capture.
    if en_scenes:
        add(frame(en_scenes, 0, stage=0, lang="en"), 700)
        for i in range(len(en_scenes)):
            add(frame(en_scenes, i, stage=1, lang="en"), 700)
            add(frame(en_scenes, i, stage=2, lang="en"), 2000)
        # LLM settings panel over the last EN screen.
        for fill in (0.34, 0.67, 1.0):
            add(frame(en_scenes, len(en_scenes) - 1, stage=2, lang="en", modal_fill=fill), 420)
        add(frame(en_scenes, len(en_scenes) - 1, stage=2, lang="en", llm_on=True), 1600)

    add(frame(scenes, special="outro"), 3200)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [img.convert("P", palette=Image.ADAPTIVE, colors=96) for img, _ in timeline]
    durations = [ms for _, ms in timeline]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"wrote {OUT} ({len(frames)} frames, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
