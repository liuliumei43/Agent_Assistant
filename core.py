from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Agent_Asistant.config import load_core_config
from Agent_Asistant.rpc.protocol import AgentRunParams, PingParams, validate_params
from Agent_Asistant.rpc.transport import JsonRpcServer, StreamingResult
from Agent_Asistant.runtime.agent import AgentRunner


class MiniCore:
    def __init__(self) -> None:
        self.config = load_core_config()
        self.runner: AgentRunner | None = None
        self.server = JsonRpcServer(self.config.host, self.config.port)
        self.server.register("ping", self.handle_ping)
        self.server.register("agent.run", self.handle_run)
        self.server.register("agent.run_stream", self.handle_run_stream)

    async def handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        parsed = validate_params("ping", params)
        assert isinstance(parsed, PingParams)
        return {"message": parsed.message, "pong": True}

    async def handle_run(self, params: dict[str, Any]) -> dict[str, Any]:
        parsed = validate_params("agent.run", params)
        assert isinstance(parsed, AgentRunParams)
        if self.runner is None:
            self.runner = AgentRunner()
        result = await self.runner.run(
            goal=parsed.goal,
            workspace=Path(parsed.workspace).resolve(),
            max_steps=parsed.max_steps,
            auto_approve=parsed.auto_approve,
            enable_rag=parsed.enable_rag,
            context_max_chars=parsed.context_max_chars,
            verbose=parsed.verbose,
        )
        return {
            "run_id": result.run_id,
            "status": result.status,
            "result": result.result,
            "events_path": result.events_path,
            "steps": result.steps,
            "elapsed_ms": result.elapsed_ms,
            "usage": result.usage,
        }

    async def handle_run_stream(self, params: dict[str, Any]) -> StreamingResult:
        parsed = validate_params("agent.run_stream", params)
        assert isinstance(parsed, AgentRunParams)
        if self.runner is None:
            self.runner = AgentRunner()
        return StreamingResult(
            self.runner.stream_run(
                goal=parsed.goal,
                workspace=Path(parsed.workspace).resolve(),
                max_steps=parsed.max_steps,
                auto_approve=parsed.auto_approve,
                enable_rag=parsed.enable_rag,
                context_max_chars=parsed.context_max_chars,
                verbose=parsed.verbose,
            )
        )

    async def run(self) -> None:
        addr = await self.server.start()
        print(f"agent-asistant-core listening at {addr}")
        await self.server.serve_forever()


def main() -> None:
    asyncio.run(MiniCore().run())


if __name__ == "__main__":
    main()
