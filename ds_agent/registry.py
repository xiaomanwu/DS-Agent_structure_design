"""Runtime extensibility: plug new capabilities into the agent without forking.

An :class:`Extension` is a self-contained capability bundle:

- ``handler(question, slots) -> AgentReply`` — the execution entry point;
- ``keywords`` — regex patterns the deterministic router scores against;
- catalog metadata (zh/en description + examples) surfaced in ``help``, the
  web capability list and the LLM planner's catalog.

Extensions live in a module-level registry so they can be registered from
anywhere (a plugin file, a notebook, ``examples/03_custom_extension.py``).
The agent consults the registry at route time and handler-dispatch time, so
registered extensions are first-class citizens: they can be planned by the
LLM planner, chained into task flows, and shown in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .reply import AgentReply

Handler = Callable[[str, Dict[str, Any]], AgentReply]

_EXTENSIONS: Dict[str, "Extension"] = {}


@dataclass
class Extension:
    """A pluggable capability bundle."""

    name: str
    handler: Handler
    keywords: List[str] = field(default_factory=list)
    weight: int = 9
    zh: str = ""
    en: str = ""
    example_zh: str = ""
    example_en: str = ""

    def catalog_entry(self) -> Dict[str, str]:
        """CAPABILITIES-shaped metadata (consumed by help/UI/planner)."""
        return {
            "zh": self.zh or self.name,
            "en": self.en or self.name,
            "example_zh": self.example_zh,
            "example_en": self.example_en,
        }


def register_extension(extension: Extension) -> Extension:
    """Register (or replace) an extension. Validates name and handler."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", extension.name):
        raise ValueError(f"extension name `{extension.name}` must be snake_case starting with a letter")
    if not callable(extension.handler):
        raise TypeError(f"extension `{extension.name}` handler must be callable(question, slots)")
    _EXTENSIONS[extension.name] = extension
    return extension


def unregister_extension(name: str) -> bool:
    return _EXTENSIONS.pop(name, None) is not None


def get_extension(name: str) -> Optional[Extension]:
    return _EXTENSIONS.get(name)


def list_extensions() -> List[Extension]:
    return list(_EXTENSIONS.values())


def clear_extensions() -> None:
    """Drop every extension (used by tests and hot reloads)."""
    _EXTENSIONS.clear()


def extension_catalog() -> Dict[str, Dict[str, str]]:
    """Extension metadata in the same shape as ``CAPABILITIES``."""
    return {ext.name: ext.catalog_entry() for ext in _EXTENSIONS.values()}


def extension_handlers() -> Dict[str, Handler]:
    return {ext.name: ext.handler for ext in _EXTENSIONS.values()}


def extension_scores(question_lower: str) -> Dict[str, int]:
    """Router scores for extensions whose keywords match the question."""
    scores: Dict[str, int] = {}
    if not question_lower:
        return scores
    for ext in _EXTENSIONS.values():
        if any(re.search(pattern, question_lower, re.I) for pattern in ext.keywords):
            scores[ext.name] = scores.get(ext.name, 0) + ext.weight
    return scores
