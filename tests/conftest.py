"""Shared fixtures: one demo warehouse per test session.

Temporary files live under ``tests/.tmp`` rather than the OS temp directory so
the suite works inside sandboxed environments where ``/tmp`` is not writable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ds_agent.data import build_demo_warehouse

TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


@pytest.fixture(scope="session")
def workdir() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TMP_ROOT


@pytest.fixture(scope="session")
def warehouse(workdir: Path) -> Path:
    """Build the demo DuckDB warehouse once for the whole test session."""
    path = workdir / "warehouse.duckdb"
    build_demo_warehouse(path, force=True)
    return path


@pytest.fixture(scope="session")
def agent(warehouse: Path, workdir: Path):
    """A ready-to-use agent bound to the demo warehouse."""
    from ds_agent.agent import DSQueryAgent

    artifacts = workdir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    instance = DSQueryAgent(warehouse=warehouse, artifact_dir=artifacts)
    yield instance
    instance.close()
    shutil.rmtree(artifacts, ignore_errors=True)
