from __future__ import annotations

from typing import Any

from Agent_Asistant.cli_app.state import CliState
from Agent_Asistant.config import load_core_config
from Agent_Asistant.rpc.transport import JsonRpcClient


async def call_core(method: str, params: dict[str, Any]) -> Any:
    config = load_core_config()
    client = JsonRpcClient(config.host, config.port)
    return await client.call(method, params)


async def run_agent(goal: str, state: CliState) -> Any:
    return await call_core(
        "agent.run",
        {
            "goal": goal,
            "workspace": state.workspace,
            "max_steps": state.max_steps,
            "auto_approve": state.auto_approve,
            "enable_rag": state.enable_rag,
            "context_max_chars": state.context_max_chars,
            "verbose": state.verbose,
        },
    )


async def stream_agent(goal: str, state: CliState) -> Any:
    config = load_core_config()
    client = JsonRpcClient(config.host, config.port)
    async for event in client.stream(
        "agent.run_stream",
        {
            "goal": goal,
            "workspace": state.workspace,
            "max_steps": state.max_steps,
            "auto_approve": state.auto_approve,
            "enable_rag": state.enable_rag,
            "context_max_chars": state.context_max_chars,
            "verbose": state.verbose,
        },
    ):
        yield event
