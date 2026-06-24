from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Agent_Assistant.llm_clients.deepseek import DeepSeekClient
from Agent_Assistant.runtime.events import EventWriter


@dataclass(frozen=True)
class ContextBudget:
    max_chars: int = 24_000
    compact_at: float = 0.75


def estimate_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def should_compact(messages: list[dict[str, Any]], budget: ContextBudget) -> bool:
    return estimate_chars(messages) >= int(budget.max_chars * budget.compact_at)


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")
        parts.append(f"[{role}]\n{content}")
        if tool_calls:
            parts.append(f"[TOOL_CALLS]\n{tool_calls}")
    return "\n\n".join(parts)


async def compact_messages(
    client: DeepSeekClient,
    messages: list[dict[str, Any]],
    events: EventWriter,
    budget: ContextBudget,
) -> list[dict[str, Any]]:
    original_chars = estimate_chars(messages)
    if original_chars < int(budget.max_chars * budget.compact_at):
        return messages

    system_message = messages[0]
    latest_user_goal = next(
        (m for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)),
        {"role": "user", "content": ""},
    )
    prompt = (
        "请将这段 Agent 对话压缩为可接续的交接摘要。\n"
        "保留：原始目标、已完成步骤、关键工具结果、已修改文件、"
        "待办事项和准确错误信息。省略重复闲聊。\n\n"
        + _messages_to_text(messages[1:])
    )
    response = await client.chat(
        messages=[
            {
                "role": "system",
                "content": "你负责总结 Agent 执行状态，便于后续继续任务。",
            },
            {"role": "user", "content": prompt},
        ],
        tools=[],
    )
    summary = str(response["choices"][0]["message"].get("content") or "").strip()
    if not summary:
        events.emit("context.compact_skipped", reason="empty_summary")
        return messages

    compacted = [
        system_message,
        latest_user_goal,
        {
            "role": "assistant",
            "content": "上下文已压缩。我会基于这份状态摘要继续执行。",
        },
        {"role": "user", "content": "状态摘要:\n" + summary},
    ]
    compacted_chars = estimate_chars(compacted)
    events.emit(
        "context.compacted",
        original_chars=original_chars,
        compacted_chars=compacted_chars,
        compression_ratio=round(compacted_chars / original_chars, 4),
    )
    return compacted
