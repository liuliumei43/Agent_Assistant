from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from Agent_Assistant.tooling.filesystem import (
    atomic_write,
    digest,
    iter_files,
    read_bytes,
    resolve_in_workspace,
)
from Agent_Assistant.tooling.rag import LocalRagIndex


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


def result_json(data: Any) -> ToolResult:
    return ToolResult(json.dumps(data, ensure_ascii=False))


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseTool:
    name: str
    description: str
    params_model: type[BaseModel]

    async def invoke(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        return await asyncio.to_thread(self.execute, params, workspace)

    def execute(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        raise NotImplementedError


class ListDirParams(Params):
    path: str = "."
    max_depth: int = Field(default=2, ge=1, le=4)


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "列出工作区文件，跳过符号链接、受保护目录及 .agentignore；最多 200 个。"
    params_model = ListDirParams

    def execute(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        parsed = ListDirParams.model_validate(params)
        root = resolve_in_workspace(workspace, parsed.path)
        if not root.is_dir():
            raise NotADirectoryError(parsed.path)
        paths = []
        for file in iter_files(workspace):
            if root not in file.parents:
                continue
            relative = file.relative_to(root)
            if len(relative.parts) <= parsed.max_depth:
                paths.append(file.relative_to(workspace).as_posix())
            if len(paths) >= 200:
                break
        return result_json({"files": paths, "limit": 200})


class ReadFileParams(Params):
    path: str
    start_line: int = Field(default=1, ge=1)
    line_count: int = Field(default=120, ge=1, le=300)
    start_column: int = Field(default=0, ge=0)
    max_chars: int = Field(default=8000, ge=100, le=24000)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "按行读取 UTF-8 文件，返回行号、sha256 和下一页位置。按 next_line/next_column 继续读取。"
    )
    params_model = ReadFileParams

    def execute(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        p = ReadFileParams.model_validate(params)
        raw = read_bytes(resolve_in_workspace(workspace, p.path))
        text = raw.decode("utf-8")
        lines = text.splitlines(keepends=True)
        output: list[dict[str, Any]] = []
        remaining = p.max_chars
        next_line, next_column = p.start_line, p.start_column
        for index in range(p.start_line - 1, min(len(lines), p.start_line - 1 + p.line_count)):
            column = p.start_column if index == p.start_line - 1 else 0
            if column > len(lines[index]):
                raise ValueError("start_column exceeds line length")
            segment = lines[index][column : column + remaining]
            output.append({"line": index + 1, "column": column, "text": segment})
            remaining -= len(segment)
            if column + len(segment) < len(lines[index]):
                next_line, next_column = index + 1, column + len(segment)
                break
            next_line, next_column = index + 2, 0
            if remaining <= 0:
                break
        return result_json(
            {
                "path": p.path,
                "sha256": digest(raw),
                "lines": output,
                "next_line": next_line if next_line <= len(lines) else None,
                "next_column": next_column,
                "total_lines": len(lines),
            }
        )


class SearchParams(Params):
    query: str = Field(min_length=1, max_length=200)
    glob: str = "*"
    case_sensitive: bool = False
    max_results: int = Field(default=30, ge=1, le=100)


class SearchTextTool(BaseTool):
    name = "search_text"
    description = "字面文本搜索（不是正则），返回相对文件名、行号和匹配片段。可用 glob 限定文件。"
    params_model = SearchParams

    def execute(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        p = SearchParams.model_validate(params)
        needle = p.query if p.case_sensitive else p.query.casefold()
        hits = []
        for file in iter_files(workspace):
            relative = file.relative_to(workspace).as_posix()
            if not fnmatch.fnmatch(relative, p.glob):
                continue
            try:
                text = read_bytes(file).decode("utf-8")
            except (OSError, ValueError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                compared = line if p.case_sensitive else line.casefold()
                position = compared.find(needle)
                if position >= 0:
                    start = max(0, position - 80)
                    hits.append({"path": relative, "line": n, "text": line[start : start + 300]})
                if len(hits) >= p.max_results:
                    return result_json({"hits": hits, "limited": True})
        return result_json({"hits": hits, "limited": False})


class GlobParams(Params):
    pattern: str = "*"


class GlobTool(BaseTool):
    name = "glob_files"
    description = "按 glob 模式查找工作区文件路径，最多返回 200 个。"
    params_model = GlobParams

    def execute(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        p = GlobParams.model_validate(params)
        files = []
        for file in iter_files(workspace):
            relative = file.relative_to(workspace).as_posix()
            if fnmatch.fnmatch(relative, p.pattern):
                files.append(relative)
            if len(files) >= 200:
                break
        return result_json({"files": files, "limit": 200})


@dataclass(frozen=True)
class PreparedWrite:
    path: str
    content: bytes
    expected_sha256: str
    diff: str

    def preview(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "diff": self.diff,
            "expected_sha256": self.expected_sha256,
            "new_sha256": digest(self.content),
            "bytes": len(self.content),
        }

    def commit(self, workspace: Path) -> ToolResult:
        atomic_write(workspace, self.path, self.content, self.expected_sha256)
        return result_json({"path": self.path, "sha256": digest(self.content), "written": True})


class WriteTool(BaseTool):
    def prepare(self, params: dict[str, Any], workspace: Path) -> PreparedWrite:
        raise NotImplementedError

    def execute(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        raise PermissionError("Write tools must pass through runtime approval")

    def prepare_content(
        self, workspace: Path, path_text: str, content: str, expected: str
    ) -> PreparedWrite:
        path = resolve_in_workspace(workspace, path_text)
        before = read_bytes(path) if path.exists() else None
        actual = digest(before) if before is not None else "missing"
        if actual != expected:
            raise ValueError("expected_sha256 不匹配；先读取文件，再修改")
        encoded = content.encode("utf-8")
        if len(encoded) > 512 * 1024:
            raise ValueError("写入内容上限为 512 KB")
        old = before.decode("utf-8") if before is not None else ""
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path_text}",
                tofile=f"b/{path_text}",
            )
        )
        if len(diff) > 100_000:
            raise ValueError("Diff 超过 100000 字符，请拆分修改")
        return PreparedWrite(path.relative_to(workspace).as_posix(), encoded, actual, diff)


class WriteFileParams(Params):
    path: str
    content: str
    expected_sha256: str = "missing"


class WriteFileTool(WriteTool):
    name = "write_file"
    description = (
        "创建或替换 UTF-8 文件，需审批 diff。覆盖时必须提供 read_file 的 sha256；新文件用 missing。"
    )
    params_model = WriteFileParams

    def prepare(self, params: dict[str, Any], workspace: Path) -> PreparedWrite:
        p = WriteFileParams.model_validate(params)
        return self.prepare_content(workspace, p.path, p.content, p.expected_sha256)


class PatchParams(Params):
    path: str
    old_text: str = Field(min_length=1)
    new_text: str
    expected_sha256: str


class ApplyPatchTool(WriteTool):
    name = "apply_patch"
    description = (
        "精确替换文件中唯一一处 old_text，需 read_file 返回的 sha256；先展示 diff 并审批。"
    )
    params_model = PatchParams

    def prepare(self, params: dict[str, Any], workspace: Path) -> PreparedWrite:
        p = PatchParams.model_validate(params)
        text = read_bytes(resolve_in_workspace(workspace, p.path)).decode("utf-8")
        if text.count(p.old_text) != 1:
            raise ValueError("old_text 必须精确匹配一处，请增加上下文")
        return self.prepare_content(
            workspace, p.path, text.replace(p.old_text, p.new_text, 1), p.expected_sha256
        )


class RagSearchParams(Params):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = "轻量 TF-IDF 文档检索，返回文件、行号与片段。精确符号定位可使用 search_text。"
    params_model = RagSearchParams

    def __init__(self) -> None:
        self.indexes: dict[Path, LocalRagIndex] = {}
        import threading

        self.lock = threading.Lock()

    def execute(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        p = RagSearchParams.model_validate(params)
        with self.lock:
            if workspace not in self.indexes:
                if len(self.indexes) >= 8:
                    self.indexes.pop(next(iter(self.indexes)))
                self.indexes[workspace] = LocalRagIndex(workspace)
            hits = self.indexes[workspace].search(p.query, p.top_k)
        return result_json(
            [
                {"path": h.path, "score": round(h.score, 4), "line": h.line, "snippet": h.snippet}
                for h in hits
            ]
        )


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def openai_schemas(self, *, enable_rag: bool = True) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.params_model.model_json_schema(),
                },
            }
            for t in self.tools.values()
            if enable_rag or t.name != "rag_search"
        ]


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [
        RagSearchTool(),
        ListDirTool(),
        ReadFileTool(),
        SearchTextTool(),
        GlobTool(),
        WriteFileTool(),
        ApplyPatchTool(),
    ]:
        registry.register(tool)
    return registry
