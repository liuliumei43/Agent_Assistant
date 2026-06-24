from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from Agent_Asistant.tooling.rag import LocalRagIndex


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class BaseTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    params_model: type[BaseModel]

    async def invoke(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        raise NotImplementedError


def resolve_in_workspace(workspace: Path, path_text: str) -> Path:
    candidate = (workspace / path_text).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError(f"禁止访问工作区外路径: {path_text}")
    return candidate


class ListDirParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str = "."
    max_depth: int = Field(default=2, ge=1, le=4)


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "以紧凑树形结构列出工作区内文件。"
    params_model = ListDirParams
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对目录路径。"},
            "max_depth": {"type": "integer", "description": "树深度，范围 1 到 4。"},
        },
    }

    async def invoke(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        parsed = self.params_model.model_validate(params)
        root = resolve_in_workspace(workspace, parsed.path)
        if not root.exists():
            raise FileNotFoundError(parsed.path)
        if not root.is_dir():
            raise NotADirectoryError(parsed.path)

        relative = root.relative_to(workspace.resolve())
        lines = [f"{relative or Path('.')}/"]
        count = 0

        def walk(directory: Path, depth: int, prefix: str) -> None:
            nonlocal count
            if depth > parsed.max_depth or count >= 200:
                return
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
            for index, entry in enumerate(entries):
                if count >= 200:
                    lines.append(f"{prefix}... truncated")
                    return
                connector = "`-- " if index == len(entries) - 1 else "|-- "
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{prefix}{connector}{entry.name}{suffix}")
                count += 1
                if entry.is_dir():
                    extension = "    " if index == len(entries) - 1 else "|   "
                    walk(entry, depth + 1, prefix + extension)

        walk(root, 1, "")
        return ToolResult("\n".join(lines))


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取工作区内的 UTF-8 文本文件，大文件会被截断。"
    params_model = ReadFileParams
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对文件路径。"}},
        "required": ["path"],
    }

    async def invoke(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        parsed = self.params_model.model_validate(params)
        path = resolve_in_workspace(workspace, parsed.path)
        raw = path.read_bytes()
        truncated = len(raw) > 256 * 1024
        text = raw[: 256 * 1024].decode("utf-8", errors="replace")
        if truncated:
            text += "\n[truncated]"
        return ToolResult(text)


class WriteFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    content: str


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "在权限审批后向工作区内写入 UTF-8 文本文件。"
    params_model = WriteFileParams
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对文件路径。"},
            "content": {"type": "string", "description": "文件内容。"},
        },
        "required": ["path", "content"],
    }

    async def invoke(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        parsed = self.params_model.model_validate(params)
        path = resolve_in_workspace(workspace, parsed.path)
        encoded = parsed.content.encode("utf-8")
        if len(encoded) > 512 * 1024:
            return ToolResult("内容过大：限制为 512 KB", is_error=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(parsed.content, encoding="utf-8")
        return ToolResult(f"已写入 {len(encoded)} 字节到 {path.relative_to(workspace.resolve())}")


class RagSearchParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    top_k: int = Field(default=5, ge=1, le=10)


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "使用轻量 TF-IDF 检索工作区内文档。需要阅读大量文件前优先使用。"
    )
    params_model = RagSearchParams
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索查询。"},
            "top_k": {"type": "integer", "description": "返回结果数，范围 1 到 10。"},
        },
        "required": ["query"],
    }

    async def invoke(self, params: dict[str, Any], workspace: Path) -> ToolResult:
        parsed = self.params_model.model_validate(params)
        index = LocalRagIndex(workspace)
        hits = index.search(parsed.query, parsed.top_k)
        if not hits:
            return ToolResult("未找到相关本地文档。")
        lines: list[str] = []
        for i, hit in enumerate(hits, start=1):
            lines.append(f"{i}. {hit.path} score={hit.score:.4f}\n{hit.snippet}")
        return ToolResult("\n\n".join(lines))


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def openai_schemas(self, *, enable_rag: bool = True) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools.values()
            if enable_rag or tool.name != "rag_search"
        ]


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(RagSearchTool())
    registry.register(ListDirTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    return registry
