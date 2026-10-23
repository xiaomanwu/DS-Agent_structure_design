"""Optional LLM backend (any OpenAI-compatible chat completion endpoint).

The agent is fully functional without it: the LLM only upgrades intent parsing
and narrative quality. Every failure path degrades to the deterministic router.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import requests

from .config import settings


class LLMClient:
    """Thin chat-completions client with JSON helpers."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout = 60

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, json_mode: bool = False) -> str:
        if not self.available:
            raise RuntimeError("no LLM API key configured")
        payload: Dict[str, object] = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def extract_json(text: str) -> Optional[dict]:
        """Tolerate code fences and prose around the JSON object."""
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        candidate = fenced.group(1) if fenced else text
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
