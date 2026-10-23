"""Reply container and number formatting shared by every handler.

Handlers live in two places - the core ones in ``agent.py`` and the advanced
experiment / causal readouts in ``readouts.py`` - so the object they all return
has to sit in a module neither of them owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AgentReply:
    """What every handler returns."""

    text: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "blocks": self.blocks,
            "tool_calls": self.tool_calls,
            "artifacts": self.artifacts,
        }


def fmt_num(value: Optional[float], digits: int = 2) -> str:
    """Format a number for prose: thousands separators, four decimals when small."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    av = abs(value)
    if av >= 1:
        return f"{value:,.{digits}f}"
    return f"{value:.4f}"


def fmt_pct(value: Optional[float], digits: int = 2) -> str:
    """Format a fraction as a signed percentage."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value * 100:+.{digits}f}%"
