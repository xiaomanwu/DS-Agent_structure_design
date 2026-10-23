"""FastAPI web app: a single-page chat UI over the agent.

Endpoints
---------
GET  /                 chat page
POST /api/chat         {"message": "..."} -> {"text", "blocks", "artifacts"}
POST /api/reset        clear conversational state
GET  /api/health       liveness + whether the warehouse is loaded
GET  /api/capabilities capability catalog
GET  /artifacts/{name} download a generated report
GET  /api/llm/status   LLM endpoint config (key masked)
POST /api/llm/config   set base_url / api_key / model at runtime (persisted to .env)
POST /api/llm/test     round-trip a trivial completion to verify the endpoint
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .agent import DSQueryAgent, full_catalog
from .config import settings
from .data import build_demo_warehouse

STATIC_DIR = Path(__file__).parent / "web_static"


class ChatRequest(BaseModel):
    message: str
    reset: bool = False


class ChatResponse(BaseModel):
    text: str
    blocks: list
    artifacts: list
    intent: Optional[str] = None


class LLMConfigRequest(BaseModel):
    """Runtime LLM configuration (any OpenAI-compatible endpoint)."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


class AgentHolder:
    """One agent per process, guarded by a lock (single-user local tool)."""

    def __init__(self) -> None:
        self._agent: Optional[DSQueryAgent] = None
        self._lock = threading.Lock()

    def get(self) -> DSQueryAgent:
        with self._lock:
            if self._agent is None:
                if not settings.warehouse.exists():
                    build_demo_warehouse(settings.warehouse)
                self._agent = DSQueryAgent(warehouse=settings.warehouse)
            return self._agent

    def reset(self) -> None:
        with self._lock:
            if self._agent is not None:
                self._agent.session.clear()
                self._agent.findings.clear()
                self._agent.summary_bullets.clear()


def create_app() -> FastAPI:
    app = FastAPI(title="DS Agent", version="0.1.0")
    holder = AgentHolder()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "warehouse": str(settings.warehouse),
            "warehouse_ready": settings.warehouse.exists(),
            "llm": settings.llm_enabled,
        }

    @app.get("/api/capabilities")
    def capabilities() -> Dict[str, Any]:
        # Includes runtime-registered extensions (see ds_agent/registry.py).
        return full_catalog()

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if request.reset:
            holder.reset()
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="message is required")
        agent = holder.get()
        reply = agent.ask(request.message)
        return ChatResponse(
            text=reply.text,
            blocks=reply.blocks,
            artifacts=reply.artifacts,
            intent=reply.data.get("intent"),
        )

    @app.post("/api/reset")
    def reset() -> JSONResponse:
        holder.reset()
        return JSONResponse({"status": "reset"})

    # ------------------------------------------------------------- LLM config
    @app.get("/api/llm/status")
    def llm_status() -> Dict[str, Any]:
        return {
            "enabled": settings.llm_enabled,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "api_key_masked": settings.llm_key_masked(),
        }

    @app.post("/api/llm/config")
    def llm_config(request: LLMConfigRequest) -> Dict[str, Any]:
        settings.update_llm(
            base_url=request.base_url,
            api_key=request.api_key,
            model=request.model,
        )
        agent = holder.get()
        agent.reconfigure_llm()
        return llm_status()

    @app.post("/api/llm/test")
    def llm_test() -> Dict[str, Any]:
        """Round-trip a trivial completion to verify the endpoint + key."""
        if not settings.llm_enabled:
            return {"ok": False, "error": "no api key configured"}
        from .llm import LLMClient

        try:
            client = LLMClient()
            text = client.chat([{"role": "user", "content": "Reply with the single word: ok"}], temperature=0.0)
            return {"ok": True, "sample": text[:80]}
        except Exception as exc:  # noqa: BLE001 - surfaced to the settings panel
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @app.get("/artifacts/{name}")
    def artifact(name: str) -> FileResponse:
        path = (settings.artifact_dir / name).resolve()
        if not str(path).startswith(str(settings.artifact_dir.resolve())) or not path.exists():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path)

    return app


app = create_app()
