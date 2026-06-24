from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from Agent_Assistant.config import load_llm_config
from Agent_Assistant.llm_clients.deepseek import DeepSeekClient
from Agent_Assistant.runtime.context import ContextBudget, compact_messages, should_compact
from Agent_Assistant.runtime.events import EventWriter
from Agent_Assistant.tooling.tools import ToolRegistry, build_default_registry

MAX_TOOL_RESULT_CHARS = 8_000

WORKSPACE_HINTS = (
    "当前项目",
    "这个项目",
    "项目结构",
    "代码",
    "文件",
    "目录",
    "模块",
    "仓库",
    "工作区",
    "读取",
    "修改",
    "创建",
    "写入",
    "实现",
    "测试",
    "运行",
    "bug",
    "readme",
    ".py",
    ".md",
    "workspace",
    "repo",
    "repository",
    "code",
    "file",
    "directory",
    "module",
    "implement",
    "refactor",
    "debug",
    "test",
)


@dataclass
class RunResult:
    run_id: str
    status: str
    result: str
    events_path: str
    steps: int
    elapsed_ms: int
    usage: dict[str, Any]


def new_run_id() -> str:
    return datetime.now(UTC).strftime("demo_%Y%m%d_%H%M%S")


def truncate_tool_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    keep = MAX_TOOL_RESULT_CHARS // 2
    omitted = len(text) - keep
    return text[:keep] + f"\n[... 已省略 {omitted} 个工具结果字符]"


def is_workspace_task(goal: str) -> bool:
    lowered = goal.lower()
    return any(hint in lowered for hint in WORKSPACE_HINTS)


def empty_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_tokens": 0,
        "total_tokens": 0,
        "estimated": False,
    }


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    text = json.dumps(messages, ensure_ascii=False)
    return max(1, len(text) // 4)


def add_usage(
    total: dict[str, Any],
    data: dict[str, Any],
    prompt_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
) -> None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        total["estimated"] = True
        total["input_tokens"] = int(total["input_tokens"]) + estimate_message_tokens(
            prompt_messages
        )
        total["output_tokens"] = int(total["output_tokens"]) + estimate_text_tokens(
            str(assistant_message.get("content") or "")
        )
        total["total_tokens"] = int(total["input_tokens"]) + int(total["output_tokens"])
        return

    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cache_tokens = int(usage.get("cache_read_input_tokens") or 0) + int(
        usage.get("cache_creation_input_tokens") or 0
    )
    total["input_tokens"] = int(total["input_tokens"]) + input_tokens
    total["output_tokens"] = int(total["output_tokens"]) + output_tokens
    total["cache_tokens"] = int(total["cache_tokens"]) + cache_tokens
    total["total_tokens"] = int(total["total_tokens"]) + int(
        usage.get("total_tokens") or input_tokens + output_tokens
    )


def approve_write(path: str, content: str, auto_approve: bool) -> bool:
    if auto_approve:
        return True
    preview = content[:160].replace("\n", "\\n")
    print("\n[权限] 工具请求写入文件")
    print(f"  路径: {path}")
    print(f"  字节数: {len(content.encode('utf-8'))}")
    print(f"  预览: {preview}")
    answer = input("允许本次写入吗？[y/N] ").strip().lower()
    return answer in {"y", "yes"}


async def execute_tool(
    call: dict[str, Any],
    registry: ToolRegistry,
    workspace: Path,
    events: EventWriter,
    auto_approve: bool,
) -> dict[str, Any]:
    function = call.get("function", {})
    name = str(function.get("name", ""))
    call_id = str(call.get("id") or uuid.uuid4())
    raw_args = str(function.get("arguments") or "{}")
    started = time.monotonic()

    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        tool = registry.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        tool.params_model.model_validate(args)

        if name == "write_file":
            path = str(args.get("path", ""))
            content = str(args.get("content", ""))
            events.emit("permission.requested", tool_name=name, path=path, risk_level="medium")
            if not approve_write(path, content, auto_approve):
                events.emit("permission.denied", tool_name=name, path=path)
                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "用户拒绝了写入权限。",
                }
            events.emit("permission.granted", tool_name=name, path=path)

        result = await tool.invoke(args, workspace)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        content = truncate_tool_result(result.content)
        event_type = "tool.failed" if result.is_error else "tool.finished"
        events.emit(event_type, tool_name=name, elapsed_ms=elapsed_ms, output_preview=content[:500])
        return {"role": "tool", "tool_call_id": call_id, "content": content}
    except (ValidationError, ValueError, KeyError, OSError, PermissionError) as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        events.emit("tool.failed", tool_name=name, elapsed_ms=elapsed_ms, error=str(exc))
        return {"role": "tool", "tool_call_id": call_id, "content": f"错误: {exc}"}


class AgentRunner:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or build_default_registry()
        self.client = DeepSeekClient(load_llm_config())
        self.context_budget = ContextBudget()

    def _format_goal(self, goal: str, verbose: bool, workspace_mode: bool) -> str:
        if verbose or not workspace_mode:
            return goal
        return (
            goal
            + "\n\n输出要求:\n"
            + "- 使用中文返回简洁、事实准确的工程总结。\n"
            + "- 只描述已经从文件中确认的能力。\n"
            + "- 不要提到 streaming、jieba、向量数据库、交互式聊天或 9090 端口，"
            + "除非文件中确实存在。\n"
            + "- 优先使用这些小标题：概览、核心模块、执行流程、简历亮点。\n"
            + "- 除非用户要求详细说明，否则控制在 1200 个中文字符以内。"
        )

    async def run(
        self,
        goal: str,
        workspace: Path,
        max_steps: int,
        auto_approve: bool,
        enable_rag: bool = True,
        context_max_chars: int = 24_000,
        verbose: bool = False,
    ) -> RunResult:
        run_id = new_run_id()
        started = time.monotonic()
        run_dir = workspace / ".runs" / run_id
        events = EventWriter(run_dir)
        usage_total = empty_usage()
        self.context_budget = ContextBudget(max_chars=context_max_chars)
        workspace_mode = is_workspace_task(goal)
        tools = self.registry.openai_schemas(enable_rag=enable_rag) if workspace_mode else []
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是 Agent_Assistant，一个通用 AI 助手和本地编码 Agent Runtime。"
                    "普通知识问答请直接根据自身知识回答，不要检查工作区。"
                    "当用户提出代码、项目、文件或仓库相关任务时，再使用可用工具，"
                    "并且只描述从文件中确认过的工作区能力。"
                    "如果用户询问实时信息，而当前没有联网工具，请说明回答可能不包含最新变化。"
                ),
            },
            {"role": "user", "content": self._format_goal(goal, verbose, workspace_mode)},
        ]

        events.emit(
            "run.started",
            run_id=run_id,
            model=self.client.config.model,
            workspace=str(workspace),
            enable_rag=enable_rag,
            context_max_chars=context_max_chars,
            workspace_mode=workspace_mode,
        )

        final_text = ""
        for step in range(1, max_steps + 1):
            events.emit("step.started", step=step)
            prompt_messages = list(messages)
            data = await self.client.chat(messages, tools)
            choice = data["choices"][0]
            assistant_message = choice["message"]
            add_usage(usage_total, data, prompt_messages, assistant_message)
            messages.append(assistant_message)

            final_text = str(assistant_message.get("content") or "")
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                events.emit(
                    "run.finished",
                    status="success",
                    steps=step,
                    elapsed_ms=elapsed_ms,
                    usage=usage_total,
                )
                return RunResult(
                    run_id,
                    "success",
                    final_text,
                    str(events.path),
                    step,
                    elapsed_ms,
                    usage_total,
                )

            for call in tool_calls:
                events.emit("tool.started", tool_name=call.get("function", {}).get("name", ""))
                tool_message = await execute_tool(
                    call,
                    self.registry,
                    workspace,
                    events,
                    auto_approve,
                )
                messages.append(tool_message)

            if should_compact(messages, self.context_budget):
                messages = await compact_messages(
                    self.client,
                    messages,
                    events,
                    self.context_budget,
                )

            events.emit("step.finished", step=step)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        events.emit(
            "run.finished",
            status="failed",
            reason="max_steps",
            steps=max_steps,
            elapsed_ms=elapsed_ms,
            usage=usage_total,
        )
        return RunResult(
            run_id,
            "failed",
            final_text,
            str(events.path),
            max_steps,
            elapsed_ms,
            usage_total,
        )

    async def stream_run(
        self,
        goal: str,
        workspace: Path,
        max_steps: int,
        auto_approve: bool,
        enable_rag: bool = True,
        context_max_chars: int = 24_000,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        run_id = new_run_id()
        started = time.monotonic()
        run_dir = workspace / ".runs" / run_id
        events = EventWriter(run_dir)
        self.context_budget = ContextBudget(max_chars=context_max_chars)
        workspace_mode = is_workspace_task(goal)
        tools = self.registry.openai_schemas(enable_rag=enable_rag) if workspace_mode else []
        usage_total = empty_usage()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是 Agent_Assistant，一个通用 AI 助手和本地编码 Agent Runtime。"
                    "普通知识问答请直接根据自身知识回答，不要检查工作区。"
                    "当用户提出代码、项目、文件或仓库相关任务时，再使用可用工具，"
                    "并将所有修改限制在工作区内。"
                ),
            },
            {"role": "user", "content": self._format_goal(goal, verbose, workspace_mode)},
        ]

        events.emit(
            "run.started",
            run_id=run_id,
            model=self.client.config.model,
            workspace=str(workspace),
            enable_rag=enable_rag,
            context_max_chars=context_max_chars,
            streaming=True,
            workspace_mode=workspace_mode,
        )
        yield {"type": "run.started", "run_id": run_id, "model": self.client.config.model}

        final_text = ""
        for step in range(1, max_steps + 1):
            events.emit("step.started", step=step)
            yield {"type": "step.started", "step": step}
            prompt_messages = list(messages)
            final_data: dict[str, Any] | None = None
            async for stream_event in self.client.chat_stream(messages, tools):
                if stream_event.get("type") == "text_delta":
                    text = str(stream_event.get("text") or "")
                    if text:
                        yield {"type": "text.delta", "text": text}
                elif stream_event.get("type") == "final":
                    data = stream_event.get("data")
                    if isinstance(data, dict):
                        final_data = data

            if final_data is None:
                raise RuntimeError("LLM stream ended without final message")
            choice = final_data["choices"][0]
            assistant_message = choice["message"]
            add_usage(usage_total, final_data, prompt_messages, assistant_message)
            messages.append(assistant_message)

            final_text = str(assistant_message.get("content") or "")
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                events.emit(
                    "run.finished",
                    status="success",
                    steps=step,
                    elapsed_ms=elapsed_ms,
                    usage=usage_total,
                )
                result = RunResult(
                    run_id,
                    "success",
                    final_text,
                    str(events.path),
                    step,
                    elapsed_ms,
                    usage_total,
                )
                yield {"type": "run.finished", "status": "success", "steps": step}
                yield {
                    "result": {
                        "run_id": result.run_id,
                        "status": result.status,
                        "result": result.result,
                        "events_path": result.events_path,
                        "steps": result.steps,
                        "elapsed_ms": result.elapsed_ms,
                        "usage": result.usage,
                    }
                }
                return

            for call in tool_calls:
                tool_name = str(call.get("function", {}).get("name", ""))
                events.emit("tool.started", tool_name=tool_name)
                yield {"type": "tool.started", "tool_name": tool_name}
                tool_message = await execute_tool(
                    call,
                    self.registry,
                    workspace,
                    events,
                    auto_approve,
                )
                messages.append(tool_message)
                yield {
                    "type": "tool.finished",
                    "tool_name": tool_name,
                    "preview": str(tool_message.get("content") or "")[:160],
                }

            if should_compact(messages, self.context_budget):
                yield {"type": "context.compacting"}
                messages = await compact_messages(
                    self.client,
                    messages,
                    events,
                    self.context_budget,
                )
                yield {"type": "context.compacted"}

            events.emit("step.finished", step=step)
            yield {"type": "step.finished", "step": step}

        elapsed_ms = int((time.monotonic() - started) * 1000)
        events.emit(
            "run.finished",
            status="failed",
            reason="max_steps",
            steps=max_steps,
            elapsed_ms=elapsed_ms,
            usage=usage_total,
        )
        result = RunResult(
            run_id,
            "failed",
            final_text,
            str(events.path),
            max_steps,
            elapsed_ms,
            usage_total,
        )
        yield {"type": "run.finished", "status": "failed", "steps": max_steps}
        yield {
            "result": {
                "run_id": result.run_id,
                "status": result.status,
                "result": result.result,
                "events_path": result.events_path,
                "steps": result.steps,
                "elapsed_ms": result.elapsed_ms,
                "usage": result.usage,
            }
        }
