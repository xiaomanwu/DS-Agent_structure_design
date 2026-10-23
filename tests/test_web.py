"""Web app: chat endpoints, bilingual page, and the runtime LLM API config."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import ds_agent.config as config_module
from ds_agent.web import create_app


@pytest.fixture(scope="module")
def client(warehouse, workdir):
    """Point the global settings at the shared warehouse and a sandboxed .env."""
    original = {
        "warehouse": config_module.settings.warehouse,
        "artifact_dir": config_module.settings.artifact_dir,
        "llm_api_key": config_module.settings.llm_api_key,
        "project_root": config_module.PROJECT_ROOT,
    }
    config_module.settings.warehouse = warehouse
    config_module.settings.artifact_dir = workdir / "artifacts_web"
    config_module.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    config_module.settings.llm_api_key = ""  # deterministic: rules mode unless a test opts in
    config_module.PROJECT_ROOT = workdir  # .env writes land in the test sandbox
    app = create_app()
    yield TestClient(app)
    config_module.settings.warehouse = original["warehouse"]
    config_module.settings.artifact_dir = original["artifact_dir"]
    config_module.settings.llm_api_key = original["llm_api_key"]
    config_module.PROJECT_ROOT = original["project_root"]


def test_index_is_bilingual(client):
    html = client.get("/").text
    assert "中文" in html and "EN" in html
    assert "setLang" in html and "localStorage" in html


def test_health_and_capabilities(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["warehouse_ready"] is True
    caps = client.get("/api/capabilities").json()
    assert "diagnose" in caps and "causal_oss" in caps


def test_chat_roundtrip(client):
    res = client.post("/api/chat", json={"message": "help"})
    assert res.status_code == 200
    data = res.json()
    assert data["text"]
    assert data["intent"] in (None, "help")


def test_chat_rejects_empty(client):
    res = client.post("/api/chat", json={"message": "   "})
    assert res.status_code == 400


def test_reset(client):
    assert client.post("/api/reset").json()["status"] == "reset"


def test_llm_status_and_runtime_config(client):
    status = client.get("/api/llm/status").json()
    assert status["enabled"] is False
    # Saving a key flips the agent into LLM mode immediately.
    res = client.post("/api/llm/config", json={"api_key": "sk-test-1234567890", "model": "test-model"})
    assert res.json()["enabled"] is True
    assert res.json()["model"] == "test-model"
    assert res.json()["api_key_masked"].endswith("7890")
    # Clearing the key returns to rules mode.
    res = client.post("/api/llm/config", json={"api_key": ""})
    assert res.json()["enabled"] is False


def test_llm_test_without_key(client):
    data = client.post("/api/llm/test").json()
    assert data["ok"] is False
