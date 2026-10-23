"""Data source registry: file attach, database attach, persistence, agent flows."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ds_agent.agent import DSQueryAgent
from ds_agent.tools.datasource import DataSourceError, DataSourceRegistry
from ds_agent.tools.sql import Warehouse

# Excel support is optional at runtime (pandas needs openpyxl for .xlsx);
# the Excel-specific cases skip cleanly when it is absent.
HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


@pytest.fixture()
def setup(workdir: Path, warehouse: Path):
    wh = workdir / "warehouse_datasource.duckdb"
    shutil.copy(warehouse, wh)
    reg = DataSourceRegistry(Warehouse(wh), path=workdir / "data_sources_test.json")
    reg.sources = {}
    agent = DSQueryAgent(warehouse=wh, artifact_dir=workdir / "artifacts_ds", datasources=reg)
    # A pair of offline files to attach.
    csv = workdir / "ds_test_sales.csv"
    pd.DataFrame(
        {"date": ["2026-10-20", "2026-10-21"], "channel": ["app", "web"], "sales": [120.0, 340.0]}
    ).to_csv(csv, index=False)
    xls = workdir / "ds_test_budget.xlsx"
    if HAS_OPENPYXL:
        pd.DataFrame({"month": ["2026-09", "2026-10"], "budget": [5000, 6000]}).to_excel(xls, index=False)
    yield agent, reg, csv, xls
    agent.close()
    reg.sources = {}


def test_register_csv_view_and_query(setup):
    agent, reg, csv, _ = setup
    info = reg.register_file("sales_extra", str(csv))
    assert info["row_count"] == 2
    assert set(info["columns"]) == {"date", "channel", "sales"}
    # The view is queryable through the guarded SQL tool.
    result = agent.sql.run("SELECT SUM(sales) AS total FROM sales_extra")
    assert float(result.dataframe["total"].iloc[0]) == 460.0


@pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl not installed - .xlsx registration needs it")
def test_register_excel_table(setup):
    _, reg, _, xls = setup
    info = reg.register_file("budget", str(xls))
    assert info["mode"] == "table"
    assert info["row_count"] == 2


def test_register_rejects_missing_file_and_bad_name(setup):
    _, reg, _, _ = setup
    with pytest.raises(DataSourceError):
        reg.register_file("nope", "/nowhere/thing.csv")
    with pytest.raises(DataSourceError):
        reg.register_file("bad name!", __file__)


def test_attach_external_duckdb(workdir: Path, warehouse: Path):
    wh = workdir / "warehouse_datasource.duckdb"
    shutil.copy(warehouse, wh)
    # The attached database must be a *separate* file: DuckDB is single-writer,
    # and attaching a file already open read-write by another connection fails.
    external = workdir / "external_dw.duckdb"
    shutil.copy(warehouse, external)
    reg = DataSourceRegistry(Warehouse(wh), path=workdir / "data_sources_test.json")
    reg.sources = {}
    info = reg.register_database("other_dw", str(external))
    assert info["mode"] == "attach"
    assert "orders_daily" in info["tables"]
    # Cross-database SELECT through the attached alias.
    row = Warehouse(wh).con.execute("SELECT COUNT(*) FROM src_other_dw.orders_daily").fetchone()
    assert row[0] > 0
    reg.drop("other_dw")


def test_persistence_and_reapply(setup, workdir: Path):
    _, reg, csv, _ = setup
    reg.register_file("sales_extra", str(csv))
    fresh = DataSourceRegistry(Warehouse(reg.warehouse.path), path=workdir / "data_sources_test.json")
    names = [s["name"] for s in fresh.list()]
    assert "sales_extra" in names
    assert not any(s.get("stale") for s in fresh.list())


@pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl not installed - .xlsx registration needs it")
def test_agent_natural_language_attach_and_query(setup):
    agent, reg, csv, xls = setup
    reply = agent.ask(f"关联数据源 {csv} 叫 sales_extra")
    assert reply.data["intent"] == "datasource"
    assert "sales_extra" in reply.text
    assert "sales_extra" in [s["name"] for s in reg.list()]
    # Direct table preview without any LLM (note: a name like "sales_extra"
    # would be stolen by the gmv alias "sales", so query `budget` here).
    reg.register_file("budget", str(xls))
    reply2 = agent.ask("查一下 budget 前10行")
    assert reply2.data.get("direct_table") == "budget"
    # Listing.
    reply3 = agent.ask("数据源列表")
    assert "sales_extra" in reply3.text


def test_agent_drop_source(setup):
    agent, reg, csv, _ = setup
    reg.register_file("sales_extra", str(csv))
    reply = agent.ask("移除数据源 sales_extra")
    assert "已移除" in reply.text
    assert [s["name"] for s in reg.list()] == []


def test_agent_lists_empty_state(setup):
    agent, _, _, _ = setup
    reply = agent.ask("数据源列表")
    assert reply.data["intent"] == "datasource"
    assert "还没有" in reply.text
