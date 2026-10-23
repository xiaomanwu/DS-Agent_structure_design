"""Build the demo warehouse (idempotent).

    python examples/00_build_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb

from ds_agent.config import settings
from ds_agent.data import build_demo_warehouse

path, orders = build_demo_warehouse(settings.warehouse, force=True)
con = duckdb.connect(str(path), read_only=True)
tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
con.close()

print(f"warehouse : {path}")
print(f"tables    : {', '.join(tables)}")
print(f"orders    : {len(orders):,} rows, {orders['date'].min()} .. {orders['date'].max()}")
print("\nPlanted incidents (useful for the demos):")
print("  1. feed channel tracking outage  ~10-14 Oct (GMV -38% for that channel)")
print("  2. iOS payment SDK regression    ~15-20 Oct (conversion -25% for iOS)")
