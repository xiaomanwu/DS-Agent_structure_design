"""Runtime configuration, loaded from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader so the project works without extra dependencies."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Real environment variables always win over the file.
        os.environ.setdefault(key, value)


_load_dotenv()


def _write_dotenv(values: Dict[str, str]) -> None:
    """Upsert keys into ``.env`` without touching unrelated lines.

    Failures are silently ignored: persistence is a convenience, runtime
    configuration already took effect on the settings object.
    """
    try:
        env_path = PROJECT_ROOT / ".env"
        lines: list = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
        pending = dict(values)
        kept: list = []
        for line in lines:
            key = line.split("=", 1)[0].strip() if "=" in line else None
            if key in pending:
                value = pending.pop(key)
                if value:  # drop cleared keys entirely
                    kept.append(f"{key}={value}")
            else:
                kept.append(line)
        for key, value in pending.items():
            if value:
                kept.append(f"{key}={value}")
        env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except OSError:
        pass


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return value if value not in ("", None) else default


@dataclass
class Settings:
    """All tunable knobs of the agent in one place."""

    llm_base_url: str = field(default_factory=lambda: _env("DS_AGENT_LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("DS_AGENT_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("DS_AGENT_LLM_MODEL", "gpt-4o-mini"))
    warehouse: Path = field(
        default_factory=lambda: Path(_env("DS_AGENT_WAREHOUSE", str(PROJECT_ROOT / "demo_data" / "warehouse.duckdb")))
    )
    artifact_dir: Path = field(
        default_factory=lambda: Path(_env("DS_AGENT_ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts")))
    )
    # Guard rails for ad-hoc SQL generated from natural language.
    sql_max_rows: int = 2000
    sql_timeout_seconds: int = 30
    # Statistical defaults used when the user does not specify them.
    alpha: float = 0.05
    power: float = 0.8

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    def update_llm(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """Runtime re-configuration of the LLM endpoint, persisted to ``.env``.

        ``None`` keeps the current value; an empty string clears the key.
        """
        if base_url is not None:
            self.llm_base_url = base_url.strip().rstrip("/") or self.llm_base_url
        if api_key is not None:
            self.llm_api_key = api_key.strip()
        if model is not None:
            self.llm_model = model.strip() or self.llm_model
        _write_dotenv(
            {
                "DS_AGENT_LLM_BASE_URL": self.llm_base_url,
                "DS_AGENT_LLM_API_KEY": self.llm_api_key,
                "DS_AGENT_LLM_MODEL": self.llm_model,
            }
        )

    @staticmethod
    def llm_key_masked() -> str:
        """Masked key for display: ``…abcd`` style, '' when unset."""
        key = settings.llm_api_key
        if not key:
            return ""
        tail = key[-4:] if len(key) > 8 else "***"
        return f"…{tail}"

    def ensure_dirs(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
