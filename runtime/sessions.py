from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


class SessionStore:
    """Complete conversation checkpoints, not automatic replay of interrupted writes."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, "
                "workspace TEXT NOT NULL, profile TEXT NOT NULL, messages TEXT NOT NULL, "
                "updated TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
        os.chmod(path, 0o600)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def load(self, session_id: str, workspace: Path, profile: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            row = db.execute(
                "SELECT workspace,profile,messages FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            return []
        if row[0] != str(workspace) or row[1] != profile:
            raise ValueError("会话绑定了其他工作区或模型配置；请使用 /new 或新的 --session")
        messages: list[dict[str, Any]] = json.loads(row[2])
        return messages

    def save(
        self, session_id: str, workspace: Path, profile: str, messages: list[dict[str, Any]]
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO sessions(id,workspace,profile,messages) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET messages=excluded.messages,"
                "updated=CURRENT_TIMESTAMP",
                (session_id, str(workspace), profile, json.dumps(messages, ensure_ascii=False)),
            )
