"""Dashboard tool: a self-contained HTML metrics dashboard per warehouse.

Assembles the everyday DS monitoring view — KPI cards with week-over-week
deltas, daily trend, dimension breakdowns, refund quality and a table of
recently flagged anomaly days — into one standalone HTML file (charts inline
as base64, no server needed). Reuses the same SQL layer as every other tool,
so the numbers match the agent's answers by construction.
"""

from __future__ import annotations

import html as html_lib
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import charts
from ..config import settings


def _wow(cur: float, prev: float) -> Optional[float]:
    if prev is None or (isinstance(prev, float) and not np.isfinite(prev)) or prev == 0:
        return None
    return float(cur - prev) / prev


def _kpi(label: str, value: str, delta: Optional[float], lang: str) -> Dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "delta": delta,
        "good": (delta >= 0) if delta is not None else None,
    }


def _window_sql(table: str, time_col: str, select: str, start, end, group_extra: str = "") -> str:
    return (
        f"SELECT {select} FROM {table} WHERE {time_col} BETWEEN DATE '{start}' AND DATE '{end}'"
        + (f" GROUP BY {group_extra}" if group_extra else "")
    )


def build_dashboard(agent, metric: str = "gmv", lang: str = "zh") -> Dict[str, Any]:
    """Build the dashboard HTML; returns {path, kpis, sections} for the chat."""
    warehouse = agent.sql.warehouse
    table = "orders_daily"
    time_col = "date"

    con = warehouse.con
    max_row = con.execute(f"SELECT MAX({time_col}) FROM {table}").fetchone()
    end = max_row[0] if max_row and max_row[0] else pd.Timestamp.today().date()
    end = pd.Timestamp(end).date()
    start28, start14 = end - timedelta(days=27), end - timedelta(days=13)

    totals = con.execute(
        f"""SELECT
            SUM(CASE WHEN {time_col} > DATE '{start14}' THEN gmv END) AS gmv_cur,
            SUM(CASE WHEN {time_col} <= DATE '{start14}' THEN gmv END) AS gmv_prev,
            SUM(CASE WHEN {time_col} > DATE '{start14}' THEN orders END) AS orders_cur,
            SUM(CASE WHEN {time_col} <= DATE '{start14}' THEN orders END) AS orders_prev,
            SUM(CASE WHEN {time_col} > DATE '{start14}' THEN buyers END) AS buyers_cur,
            SUM(CASE WHEN {time_col} <= DATE '{start14}' THEN buyers END) AS buyers_prev,
            SUM(CASE WHEN {time_col} > DATE '{start14}' THEN sessions END) AS sess_cur,
            SUM(CASE WHEN {time_col} > DATE '{start14}' THEN refund_orders END) AS refund_cur
            FROM {table} WHERE {time_col} > DATE '{start28}'"""
    ).fetchone()
    gmv_cur, gmv_prev, orders_cur, orders_prev, buyers_cur, buyers_prev, sess_cur, refund_cur = totals
    conv_cur = orders_cur / sess_cur if sess_cur else float("nan")
    conv_prev = orders_prev / (con.execute(
        f"SELECT SUM(sessions) FROM {table} WHERE {time_col} <= DATE '{start14}' AND {time_col} > DATE '{start28}'"
    ).fetchone()[0] or 1)

    labels = (
        ["GMV", "订单量", "购买人数", "支付转化率"]
        if lang == "zh"
        else ["GMV", "Orders", "Buyers", "Conversion"]
    )
    kpis = [
        _kpi(labels[0], f"¥{gmv_cur:,.0f}", _wow(gmv_cur, gmv_prev), lang),
        _kpi(labels[1], f"{orders_cur:,.0f}", _wow(orders_cur, orders_prev), lang),
        _kpi(labels[2], f"{buyers_cur:,.0f}", _wow(buyers_cur, buyers_prev), lang),
        _kpi(labels[3], f"{conv_cur:.2%}", _wow(conv_cur, conv_prev), lang),
    ]

    daily = con.execute(
        f"SELECT {time_col} AS d, SUM(gmv) AS gmv, SUM(orders) AS orders FROM {table} "
        f"WHERE {time_col} > DATE '{start28}' GROUP BY 1 ORDER BY 1"
    ).df()
    daily["date"] = pd.to_datetime(daily["d"])
    daily["ma7"] = daily["gmv"].rolling(7, min_periods=1).mean()
    trend_img = charts.anomaly_chart(
        pd.DataFrame({"date": daily["date"], "actual": daily["gmv"], "expected": daily["ma7"], "flagged": False}),
        title="GMV 近28天趋势（含7日均线）" if lang == "zh" else "GMV, last 28 days (with 7-day moving average)",
    )

    by_channel = con.execute(
        f"SELECT channel AS segment, SUM(gmv) AS contribution FROM {table} WHERE {time_col} > DATE '{start14}' GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    channel_img = charts.contribution_bar(by_channel, "channel", title="渠道 GMV（近14天）" if lang == "zh" else "GMV by channel (last 14 days)")
    by_device = con.execute(
        f"SELECT device AS segment, SUM(gmv) AS contribution FROM {table} WHERE {time_col} > DATE '{start14}' GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    device_img = charts.contribution_bar(by_device, "device", title="设备 GMV（近14天）" if lang == "zh" else "GMV by device (last 14 days)")

    refund = con.execute(
        f"SELECT {time_col} AS d, SUM(refund_orders)::FLOAT / NULLIF(SUM(orders),0) AS rate FROM {table} "
        f"WHERE {time_col} > DATE '{start28}' GROUP BY 1 ORDER BY 1"
    ).df()
    refund["date"] = pd.to_datetime(refund["d"])
    refund_img = charts.anomaly_chart(
        pd.DataFrame({"date": refund["date"], "actual": refund["rate"], "expected": refund["rate"].rolling(7, min_periods=1).mean(), "flagged": False}),
        title="退款率近28天" if lang == "zh" else "Refund rate, last 28 days",
    )

    # Anomaly flags for the dashboard table: same detector as the agent's
    # diagnosis path, so the "recent anomalies" block always agrees with it.
    flagged_rows: List[Dict[str, Any]] = []
    try:
        from .anomaly import AnomalyDetector

        detection = AnomalyDetector().detect(
            daily["date"].tolist(), daily["gmv"].astype(float).tolist(), label="GMV"
        )
        for ep in detection["episodes"][:6]:
            flagged_rows.append(
                {
                    "date": f"{ep['start']} ~ {ep['end']}",
                    "metric": "GMV",
                    "delta": f"{ep['impact_pct']:+.1%} ({ep['direction']})",
                    "severity": str(ep["severity"]),
                }
            )
    except Exception:  # noqa: BLE001 - dashboard must render even without detections
        flagged_rows = []

    path = _render_html(kpis, trend_img, channel_img, device_img, refund_img, flagged_rows, lang, end)
    return {
        "path": str(path),
        "kpis": kpis,
        "charts": [trend_img, channel_img, device_img, refund_img],
        "flagged_rows": flagged_rows,
    }


def _render_html(
    kpis: List[Dict[str, Any]],
    trend_img: str,
    channel_img: str,
    device_img: str,
    refund_img: str,
    flagged_rows: List[Dict[str, Any]],
    lang: str,
    data_end,
) -> Path:
    up_word = "环比" if lang == "zh" else "WoW"
    t = {
        "title": "DS 监控看板" if lang == "zh" else "DS Monitoring Dashboard",
        "sub": f"数据截至 {data_end} · 内置演示数仓" if lang == "zh" else f"Data through {data_end} · built-in demo warehouse",
        "anomalies": "近期异动" if lang == "zh" else "Recent anomalies",
        "none": "近 28 天无显著异动" if lang == "zh" else "No significant anomalies in the last 28 days",
        "footer": "由 DS-Agent 生成 · 数字与 Agent 查询同源同口径" if lang == "zh" else "Generated by DS-Agent · same source and calibers as agent answers",
    }
    kpi_html = ""
    for k in kpis:
        delta = ""
        if k["delta"] is not None:
            cls = "up" if k["delta"] >= 0 else "down"
            arrow = "▲" if k["delta"] >= 0 else "▼"
            delta = f'<span class="{cls}">{arrow} {abs(k["delta"]):.1%} {up_word}</span>'
        kpi_html += f'<div class="kpi"><div class="lbl">{html_lib.escape(k["label"])}</div><div class="val">{html_lib.escape(k["value"])}</div>{delta}</div>'

    if flagged_rows:
        head = "<tr><th>" + "</th><th>".join(["date", "metric", "delta", "severity"]) + "</th></tr>"
        body = "".join(
            "<tr>" + "".join(f"<td>{html_lib.escape(str(r[c]))}</td>" for c in ("date", "metric", "delta", "severity")) + "</tr>"
            for r in flagged_rows
        )
        anomaly_html = f'<table class="tbl">{head}{body}</table>'
    else:
        anomaly_html = f'<div class="empty">{t["none"]}</div>'

    css = """
    body{margin:0;background:#f5f7fb;color:#0f172a;font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;}
    .page{max-width:1080px;margin:0 auto;padding:28px 24px;}
    h1{font-size:20px;margin:0 0 4px;}
    .sub{color:#64748b;font-size:12px;margin-bottom:22px;}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:22px;}
    .kpi{background:#fff;border:1px solid #e5e9f0;border-radius:14px;padding:16px 18px;}
    .kpi .lbl{font-size:12px;color:#64748b;}
    .kpi .val{font-size:26px;font-weight:650;letter-spacing:-.5px;margin:4px 0 2px;}
    .kpi .up{color:#dc2626;font-size:12px;} .kpi .down{color:#16a34a;font-size:12px;}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
    .card{background:#fff;border:1px solid #e5e9f0;border-radius:14px;padding:14px;}
    .card img{width:100%;display:block;border-radius:8px;}
    .wide{grid-column:1/-1;}
    .tbl{border-collapse:collapse;width:100%;font-size:13px;}
    .tbl th{background:#f8fafc;text-align:left;padding:8px 12px;border-bottom:1px solid #e5e9f0;}
    .tbl td{padding:7px 12px;border-bottom:1px solid #f1f5f9;}
    .empty{color:#64748b;font-size:13px;padding:10px 0;}
    .foot{color:#94a3b8;font-size:11px;margin-top:24px;text-align:center;}
    """
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{t['title']}</title><style>{css}</style></head><body><div class='page'>",
        f"<h1>{t['title']}</h1><div class='sub'>{t['sub']}</div>",
        f"<div class='kpis'>{kpi_html}</div>",
        "<div class='grid'>",
        f"<div class='card wide'><img src='{trend_img}'></div>",
        f"<div class='card'><img src='{channel_img}'></div>",
        f"<div class='card'><img src='{device_img}'></div>",
        f"<div class='card wide'><img src='{refund_img}'></div>",
        f"<div class='card wide'><div class='lbl' style='font-size:13px;font-weight:600;margin-bottom:8px'>{t['anomalies']}</div>{anomaly_html}</div>",
        "</div>",
        f"<div class='foot'>{t['footer']}</div></div></body></html>",
    ]
    out_dir = Path(settings.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dashboard_{data_end}.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path
