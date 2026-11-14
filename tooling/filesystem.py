from __future__ import annotations

import fnmatch
import hashlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

MAX_FILE_BYTES = 2 * 1024 * 1024
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".runs",
    ".agent-state",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def resolve_in_workspace(workspace: Path, path_text: str) -> Path:
    root = workspace.resolve(strict=True)
    candidate = root / path_text
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"禁止访问工作区外路径: {path_text}")
    for part in [candidate, *candidate.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise PermissionError(f"禁止访问符号链接: {path_text}")
    relative = resolved.relative_to(root)
    if any(
        p in SKIP_DIRS or p == ".env" or p.startswith(".env.") or p.endswith((".pem", ".key"))
        for p in relative.parts
    ):
        raise PermissionError(f"受保护路径: {path_text}")
    return resolved


def read_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path.name}")
    with path.open("rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_BYTES} byte limit: {path.name}")
    return data


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_files(workspace: Path, limit: int = 5000) -> Iterator[Path]:
    root = workspace.resolve(strict=True)
    ignore_file = root / ".agentignore"
    patterns: list[str] = []
    if ignore_file.is_file() and not ignore_file.is_symlink():
        patterns = [
            x.strip()
            for x in read_bytes(ignore_file).decode("utf-8").splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        ]
    count = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d for d in dirs if d not in SKIP_DIRS and not (Path(directory) / d).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(relative, p) for p in patterns):
                continue
            try:
                path = resolve_in_workspace(root, relative)
            except (PermissionError, OSError, ValueError):
                continue
            if not path.is_file():
                continue
            yield path
            count += 1
            if count >= limit:
                return


def atomic_write(workspace: Path, relative: str, content: bytes, expected: str) -> None:
    path = resolve_in_workspace(workspace, relative)
    actual = digest(read_bytes(path)) if path.exists() else "missing"
    if actual != expected:
        raise ValueError("文件已变化：请重新读取并审批新的 diff")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temp_path, path.stat().st_mode & 0o777)
        # This guard is not an OS sandbox against a hostile concurrent process.
        resolve_in_workspace(workspace, relative)
        actual = digest(read_bytes(path)) if path.exists() else "missing"
        if actual != expected:
            raise ValueError("文件已变化：取消写入")
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
