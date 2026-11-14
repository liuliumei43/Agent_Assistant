from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Agent_Assistant.config import load_core_config
from Agent_Assistant.rpc.protocol import AgentRunParams, ApprovalParams, PingParams, validate_params
from Agent_Assistant.rpc.transport import JsonRpcServer, StreamingResult
from Agent_Assistant.runtime.agent import AgentRunner
from Agent_Assistant.runtime.approvals import ApprovalBroker


class MiniCore:
    def __init__(self) -> None:
        self.config = load_core_config()
        self.runner: AgentRunner | None = None
        self.approvals = ApprovalBroker()
        self.server = JsonRpcServer(self.config.host, self.config.port)
        self.server.register("ping", self.handle_ping)
        self.server.register("agent.run", self.handle_run)
        self.server.register("agent.run_stream", self.handle_run_stream)
        self.server.register("approval.respond", self.handle_approval)

    async def handle_approval(self, params: dict[str, Any]) -> dict[str, Any]:
        parsed = validate_params("approval.respond", params)
        assert isinstance(parsed, ApprovalParams)
        self.approvals.resolve(parsed.approval_id, parsed.approved)
        return {"accepted": True}

    async def handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        parsed = validate_params("ping", params)
        assert isinstance(parsed, PingParams)
        return {"message": parsed.message, "pong": True}

    async def handle_run(self, params: dict[str, Any]) -> dict[str, Any]:
        parsed = validate_params("agent.run", params)
        assert isinstance(parsed, AgentRunParams)
        if self.runner is None:
            self.runner = AgentRunner(approvals=self.approvals)
        result = await self.runner.run(
            goal=parsed.goal,
            workspace=Path(parsed.workspace).resolve(),
            max_steps=parsed.max_steps,
            auto_approve=parsed.auto_approve,
            enable_rag=parsed.enable_rag,
            context_max_chars=parsed.context_max_chars,
            verbose=parsed.verbose,
            session_id=parsed.session_id,
        )
        return {
            "run_id": result.run_id,
            "status": result.status,
            "result": result.result,
            "events_path": result.events_path,
            "steps": result.steps,
            "elapsed_ms": result.elapsed_ms,
            "usage": result.usage,
            "error": result.error,
            "session_id": result.session_id,
        }

    async def handle_run_stream(self, params: dict[str, Any]) -> StreamingResult:
        parsed = validate_params("agent.run_stream", params)
        assert isinstance(parsed, AgentRunParams)
        if self.runner is None:
            self.runner = AgentRunner(approvals=self.approvals)
        return StreamingResult(
            self.runner.stream_run(
                goal=parsed.goal,
                workspace=Path(parsed.workspace).resolve(),
                max_steps=parsed.max_steps,
                auto_approve=parsed.auto_approve,
                enable_rag=parsed.enable_rag,
                context_max_chars=parsed.context_max_chars,
                verbose=parsed.verbose,
                session_id=parsed.session_id,
                stream_llm=parsed.stream_llm,
                skill=parsed.skill,
            )
        )

    async def run(self) -> None:
        addr = await self.server.start()
        print(f"agent-assistant-core listening at {addr}")
        try:
            await self.server.serve_forever()
        finally:
            await self.server.aclose()
            if self.runner:
                await self.runner.client.aclose()


def main() -> None:
    asyncio.run(MiniCore().run())


if __name__ == "__main__":
    main()
