import asyncio
import json

import pytest
import pytest_asyncio
from Agent_Assistant import core as core_module
from Agent_Assistant.cli_app import client as cli_client
from Agent_Assistant.cli_app.state import CliState
from Agent_Assistant.config import CoreConfig
from Agent_Assistant.core import MiniCore
from Agent_Assistant.rpc.transport import JsonRpcClient
from Agent_Assistant.runtime.agent import AgentRunner
from test_runtime import ScriptClient, call, reply


@pytest_asyncio.fixture
async def live_core(tmp_path, monkeypatch):
    monkeypatch.setattr(core_module, "load_core_config", lambda: CoreConfig("127.0.0.1", 0))
    core = MiniCore()
    client = ScriptClient([])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    core.runner = AgentRunner(client=client, state_dir=tmp_path / "state", approvals=core.approvals)
    await core.server.start()
    port = core.server.server.sockets[0].getsockname()[1]
    config = CoreConfig("127.0.0.1", port)
    monkeypatch.setattr(cli_client, "load_core_config", lambda: config)
    try:
        yield core, client, workspace, JsonRpcClient(config.host, port)
    finally:
        await core.server.aclose()


@pytest.mark.parametrize("approved", [True, False])
async def test_rpc_approval_ping_and_completion(live_core, approved):
    core, model, workspace, rpc = live_core
    model.responses = [
        reply("", [call("write_file", {"path": "a.txt", "content": "hello"})]),
        reply(),
    ]
    stream = rpc.stream(
        "agent.run_stream", {"goal": "创建说明", "workspace": str(workspace), "session_id": "demo"}
    )
    events = []
    async for event in stream:
        events.append(event)
        if event.get("type") == "permission.requested":
            assert (await asyncio.wait_for(rpc.call("ping", {}), 1))["pong"]
            await rpc.call(
                "approval.respond", {"approval_id": event["approval_id"], "approved": approved}
            )
    assert events[-1]["type"] == "result" and events[-1]["data"]["status"] == "success"
    assert (workspace / "a.txt").exists() == approved
    assert not core.approvals.pending


async def test_disconnect_cancels_pending_write(live_core):
    core, model, workspace, rpc = live_core
    model.responses = [reply("", [call("write_file", {"path": "a.txt", "content": "hello"})])]
    stream = rpc.stream(
        "agent.run_stream", {"goal": "create", "workspace": str(workspace), "session_id": "cancel"}
    )
    async for event in stream:
        if event.get("type") == "permission.requested":
            assert core.approvals.pending
            await stream.aclose()
            break
    async with asyncio.timeout(1):
        while core.approvals.pending or core.runner.active_sessions:
            await asyncio.sleep(0.005)
    assert not (workspace / "a.txt").exists()


async def test_cli_transport_handles_nonstream_model_and_session(live_core):
    _, model, workspace, _ = live_core
    model.responses = [reply("remembered"), reply("ALPHA")]
    state = CliState(workspace=str(workspace), stream=False, session_id="cli-demo")
    first = await cli_client.run_agent("remember ALPHA", state)
    second = await cli_client.run_agent("what was it", state)
    assert first["status"] == second["status"] == "success"
    assert "ALPHA" in json.dumps(model.requests[1])


async def test_cli_approval_callback(live_core):
    _, model, workspace, _ = live_core
    model.responses = [reply("", [call("write_file", {"path": "note", "content": "yes"})]), reply()]
    seen = []

    async def approve(event):
        seen.append(event["path"])
        return True

    state = CliState(workspace=str(workspace))
    events = [e async for e in cli_client.stream_agent("create", state, approve)]
    assert seen == ["note"] and (workspace / "note").read_text() == "yes"
    assert events[-1]["data"]["status"] == "success"


async def test_stream_error_does_not_break_client_error_decoding(live_core):
    _, model, workspace, rpc = live_core
    model.responses = [RuntimeError("fixture_failure")]
    events = [
        e
        async for e in rpc.stream(
            "agent.run_stream", {"goal": "hello", "workspace": str(workspace)}
        )
    ]
    assert events[-1]["data"]["status"] == "failed"
    assert "fixture_failure" in events[-1]["data"]["error"]


async def test_unknown_and_reused_approval_rejected(live_core):
    core, _, _, rpc = live_core
    approval_id = core.approvals.create()
    await rpc.call("approval.respond", {"approval_id": approval_id, "approved": False})
    with pytest.raises(RuntimeError, match="already answered"):
        await rpc.call("approval.respond", {"approval_id": approval_id, "approved": True})
    assert not await core.approvals.wait(approval_id)
