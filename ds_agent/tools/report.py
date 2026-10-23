"""Report composition: turns tool outputs into a Markdown file and a standalone HTML.

The HTML is single-file (charts are inline base64), print-friendly, and readable
without any assets, which makes it easy to paste into a wiki or send to a
stakeholder.
"""

from __future__ import annotations

import html as html_lib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CSS = """
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --brand:#2563eb; --up:#dc2626; --down:#16a34a; --bg:#ffffff; }
* { box-sizing: border-box; }
body { margin:0; background:#f6f8fb; color:var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  line-height:1.65; }
.page { max-width: 940px; margin: 28px auto 64px; background: var(--bg); padding: 44px 52px;
  box-shadow: 0 1px 3px rgba(15,23,42,.08), 0 12px 32px rgba(15,23,42,.06); border-radius: 14px; }
h1 { font-size: 27px; margin: 0 0 6px; letter-spacing: -.3px; }
h2 { font-size: 18px; margin: 34px 0 10px; padding-bottom: 7px; border-bottom: 1px solid var(--line); }
h3 { font-size: 15px; margin: 20px 0 8px; color:#1e293b; }
.meta { color: var(--muted); font-size: 12.5px; margin-bottom: 4px; }
.lead { background:#f8fafc; border:1px solid var(--line); border-left:3px solid var(--brand);
  border-radius:8px; padding:14px 18px; margin:18px 0 6px; }
.lead ul { margin:6px 0 0; padding-left: 20px; }
.lead li { margin: 5px 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 13px; }
th { background:#f1f5f9; text-align:left; font-weight:600; color:#334155; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--line); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover td { background:#fafcff; }
.chart { margin: 14px 0 20px; }
.chart img { width:100%; border:1px solid var(--line); border-radius:10px; }
.note { font-size: 12.5px; color: var(--muted); border-left: 2px solid var(--line); padding-left: 12px; margin: 12px 0; }
.tag { display:inline-block; font-size:11px; padding:1px 8px; border-radius:999px; background:#eff6ff; color:#1d4ed8; margin-right:6px;}
ul.plain { padding-left: 20px; }
ul.plain li { margin: 4px 0; }
.up { color: var(--up); font-weight:600; }
.down { color: var(--down); font-weight:600; }
.footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11.5px; }
@media print { body { background:#fff; } .page { box-shadow:none; margin:0; padding:0; max-width:none; } }
"""


@dataclass
class Section:
    """One block of the report."""

    title: str
    prose: Optional[str] = None
    bullets: List[str] = field(default_factory=list)
    table: Optional[pd.DataFrame] = None
    chart: Optional[str] = None
    note: Optional[str] = None
    tag: Optional[str] = None


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        av = abs(value)
        if av == 0:
            return "0"
        if av >= 1000:
            return f"{value:,.0f}"
        if av >= 1:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def _table_html(df: pd.DataFrame) -> str:
    headers = "".join(
        f'<th class="{"num" if pd.api.types.is_numeric_dtype(df[c]) else ""}">{html_lib.escape(str(c))}</th>'
        for c in df.columns
    )
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            numeric = pd.api.types.is_numeric_dtype(df[col])
            cells.append(f'<td class="{"num" if numeric else ""}">{html_lib.escape(_fmt(val))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _table_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"
    lines = []
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join([header, divider, *lines])


def to_markdown_table(df: pd.DataFrame) -> str:
    """Markdown table renderer that does not depend on the optional ``tabulate``."""
    return _table_md(df)


class Report:
    """Compose, then export to Markdown and standalone HTML."""
    def __init__(
        self, title: str, subtitle: str = "", meta: Optional[Dict[str, str]] = None, lang: str = "en"
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.meta = meta or {}
        self.lang = lang
        self.summary: List[str] = []
        self.sections: List[Section] = []
        self.methods: List[str] = []
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    @property
    def _h_summary(self) -> str:
        return "Executive summary" if self.lang == "zh" else "Executive summary"

    @property
    def _h_methods(self) -> str:
        return "Methodology & caveats" if self.lang == "zh" else "Methodology & caveats"

    @property
    def _footer(self) -> str:
        if self.lang == "zh":
            return "Generated by DS-Agent · all figures reproducible from the tool calls in this report."
        return "Generated by DS-Agent · all figures reproducible from the tool calls in this report."

    # ------------------------------------------------------------------ build
    def add_summary(self, bullets: Sequence[str]) -> "Report":
        self.summary.extend(bullets)
        return self

    def add_section(self, section: Section) -> "Report":
        self.sections.append(section)
        return self

    def add_method(self, text: str) -> "Report":
        self.methods.append(text)
        return self

    # ---------------------------------------------------------------- exports
    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.subtitle:
            lines += [f"*{self.subtitle}*", ""]
        for key, value in self.meta.items():
            lines.append(f"- **{key}**: {value}")
        lines += ["", f"*Generated by DS-Agent at {self.generated_at}*", ""]
        if self.summary:
            lines += [f"## {self._h_summary}", ""] + [f"- {b}" for b in self.summary] + [""]
        for sec in self.sections:
            head = f"## {sec.title}"
            if sec.tag:
                head += f"  `{sec.tag}`"
            lines += [head, ""]
            if sec.prose:
                lines += [sec.prose, ""]
            if sec.bullets:
                lines += [f"- {b}" for b in sec.bullets] + [""]
            if sec.table is not None and len(sec.table):
                lines += [_table_md(sec.table), ""]
            if sec.note:
                lines += [f"> {sec.note}", ""]
        if self.methods:
            lines += [f"## {self._h_methods}", ""] + [f"- {m}" for m in self.methods] + [""]
        lines += ["---", "", self._footer]
        return "\n".join(lines)

    def to_html(self) -> str:
        parts = [
            "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>{html_lib.escape(self.title)}</title><style>{CSS}</style></head><body><div class=\"page\">",
            f"<h1>{html_lib.escape(self.title)}</h1>",
        ]
        if self.subtitle:
            parts.append(f"<div class=\"meta\">{html_lib.escape(self.subtitle)}</div>")
        for key, value in self.meta.items():
            parts.append(f"<div class=\"meta\"><strong>{html_lib.escape(key)}</strong>: {html_lib.escape(str(value))}</div>")
        parts.append(f"<div class=\"meta\">Generated by DS-Agent · {self.generated_at}</div>")

        if self.summary:
            items = "".join(f"<li>{html_lib.escape(b)}</li>" for b in self.summary)
            parts.append(
                f"<div class=\"lead\"><strong>{self._h_summary}</strong><ul>{items}</ul></div>"
            )

        for sec in self.sections:
            tag = f'<span class="tag">{html_lib.escape(sec.tag)}</span>' if sec.tag else ""
            parts.append(f"<h2>{tag}{html_lib.escape(sec.title)}</h2>")
            if sec.prose:
                parts.append(f"<p>{html_lib.escape(sec.prose)}</p>")
            if sec.bullets:
                items = "".join(f"<li>{html_lib.escape(b)}</li>" for b in sec.bullets)
                parts.append(f"<ul class=\"plain\">{items}</ul>")
            if sec.table is not None and len(sec.table):
                parts.append(_table_html(sec.table))
            if sec.chart:
                parts.append(f'<div class="chart"><img src="{sec.chart}" alt="chart"></div>')
            if sec.note:
                parts.append(f"<div class=\"note\">{html_lib.escape(sec.note)}</div>")

        if self.methods:
            items = "".join(f"<li>{html_lib.escape(m)}</li>" for m in self.methods)
            parts.append(f"<h2>{html_lib.escape(self._h_methods)}</h2><ul class=\"plain\">{items}</ul>")

        parts.append(f"<div class=\"footer\">{html_lib.escape(self._footer)}</div>")
        parts.append("</div></body></html>")
        return "".join(parts)

    def save(self, directory: Path, name: str) -> Dict[str, str]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80]
        md_path = directory / f"{safe}.md"
        html_path = directory / f"{safe}.html"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        html_path.write_text(self.to_html(), encoding="utf-8")
        return {"markdown": str(md_path), "html": str(html_path)}
