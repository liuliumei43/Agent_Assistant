from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class CliState:
    workspace: str = "."
    max_steps: int = 8
    auto_approve: bool = False
    enable_rag: bool = True
    context_max_chars: int = 24_000
    verbose: bool = False
    raw_json: bool = False
    session_id: str = field(default_factory=lambda: uuid4().hex[:12])
    stream: bool = True
