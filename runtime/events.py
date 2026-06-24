from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class EventWriter:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "events.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **data: Any) -> None:
        event = {"type": event_type, "ts": utc_now(), **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
