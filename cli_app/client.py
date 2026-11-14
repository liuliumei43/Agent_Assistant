from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from Agent_Assistant.cli_app.state import CliState
from Agent_Assistant.config import load_core_config
from Agent_Assistant.rpc.transport import JsonRpcClient

ApprovalHandler = Callable[[dict[str, Any]], Awaitable[bool]]


async def call_core(method: str, params: dict[str, Any]) -> Any:
    config = load_core_config()
    return await JsonRpcClient(config.host, config.port).call(method, params)


async def terminal_approval(event: dict[str, Any]) -> bool:
    print(f"\n写入审批：{event['path']}\n{event['diff']}")
    if not sys.stdin.isatty():
        print("非交互输入：拒绝写入。可信自动化可显式使用 --yes。")
        return False
    print("允许这个 diff？[y/N] ", end="", flush=True)
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[str] = loop.create_future()
    fd = sys.stdin.fileno()

    def ready() -> None:
        if not answer.done():
            answer.set_result(sys.stdin.readline())

    loop.add_reader(fd, ready)
    try:
        text = await asyncio.wait_for(answer, 290)
        return text.strip().lower() in {"y", "yes"}
    except TimeoutError:
        print("\n审批已超时。")
        return False
    finally:
        loop.remove_reader(fd)


async def stream_agent(
    goal: str, state: CliState, approval_handler: ApprovalHandler | None = None
) -> AsyncIterator[dict[str, Any]]:
    config = load_core_config()
    client = JsonRpcClient(config.host, config.port)
    stream = client.stream(
        "agent.run_stream",
        {
            "goal": goal,
            "workspace": str(Path(state.workspace).expanduser().resolve()),
            "max_steps": state.max_steps,
            "auto_approve": state.auto_approve,
            "enable_rag": state.enable_rag,
            "context_max_chars": state.context_max_chars,
            "verbose": state.verbose,
            "session_id": state.session_id,
            "stream_llm": state.stream,
            "skill": state.skill,
        },
    )
    try:
        async for event in stream:
            yield event
            if event.get("type") == "permission.requested":
                handler = approval_handler or terminal_approval
                approved = False if state.raw_json else await handler(event)
                await call_core(
                    "approval.respond", {"approval_id": event["approval_id"], "approved": approved}
                )
    finally:
        await stream.aclose()


async def run_agent(goal: str, state: CliState) -> dict[str, Any]:
    async for event in stream_agent(goal, state):
        if event.get("type") == "result":
            result: dict[str, Any] = event["data"]
            return result
    raise RuntimeError("No result from Core")
