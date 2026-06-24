from __future__ import annotations

from typing import Any

from Agent_Asistant.cli_app.client import run_agent, stream_agent
from Agent_Asistant.cli_app.commands import handle_command
from Agent_Asistant.cli_app.output import (
    CYAN,
    RESET,
    YELLOW,
    print_banner,
    print_json,
    print_run_result,
    print_run_summary,
)
from Agent_Asistant.cli_app.state import CliState


async def run_streaming_turn(line: str, state: CliState) -> dict[str, Any] | None:
    final_result: dict[str, Any] | None = None
    async for event in stream_agent(line, state):
        if state.raw_json:
            print_json(event)
            continue
        event_type = str(event.get("type", ""))
        if event_type == "run.started":
            print(f"{CYAN}运行{RESET} {event.get('run_id', '')}")
        elif event_type == "step.started":
            print(f"{CYAN}步骤{RESET} {event.get('step', '')}")
        elif event_type == "text.delta":
            print(str(event.get("text") or ""), end="", flush=True)
        elif event_type == "tool.started":
            print()
            print(f"{CYAN}工具{RESET} {event.get('tool_name', '')} 执行中...")
        elif event_type == "tool.finished":
            print(f"{CYAN}工具{RESET} {event.get('tool_name', '')} 完成")
        elif event_type == "context.compacting":
            print(f"{CYAN}上下文{RESET} 压缩中...")
        elif event_type == "context.compacted":
            print(f"{CYAN}上下文{RESET} 已压缩")
        elif event_type == "result":
            data = event.get("data")
            if isinstance(data, dict):
                final_result = data
    return final_result


async def interactive_loop(state: CliState) -> None:
    print_banner(state)
    while True:
        try:
            prompt = f"{YELLOW}输入消息 >{RESET} "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith("/"):
            keep_running = await handle_command(line, state)
            if not keep_running:
                return
            continue
        try:
            print(f"{CYAN}运行{RESET} 启动中...")
            if state.stream:
                result = await run_streaming_turn(line, state)
            else:
                result = await run_agent(line, state)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            continue
        if result is None:
            print()
            continue
        if state.raw_json:
            print_json(result)
        elif state.stream:
            print_run_summary(result)
        else:
            print_run_result(result)
        print()
