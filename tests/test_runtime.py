import asyncio
import copy
import json
from pathlib import Path

import pytest
from Agent_Assistant.config import LlmConfig
from Agent_Assistant.runtime.agent import AgentRunner, add_usage, empty_usage, new_run_id
from Agent_Assistant.runtime.approvals import ApprovalBroker
from Agent_Assistant.runtime.context import (
    ContextBudget,
    compact_messages,
    estimate_chars,
    should_compact,
)
from Agent_Assistant.runtime.events import EventWriter


def reply(content="done", calls=None):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": calls or []},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def call(name, args, call_id="call1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptClient:
    config = LlmConfig("fake", "https://example.invalid", "fixture", "chat_completions")

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def chat(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)

    async def chat_stream(self, messages, tools):
        result = await self.chat(messages, tools)
        yield {"type": "text_delta", "text": result["choices"][0]["message"]["content"]}
        yield {"type": "final", "data": result}

    async def aclose(self):
        pass


def make_runner(tmp_path, responses, broker=None):
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    client = ScriptClient(responses)
    runner = AgentRunner(client=client, state_dir=tmp_path / "state", approvals=broker)
    return runner, client, ws


async def test_session_continues_after_restart_and_isolates(tmp_path):
    runner, client, ws = make_runner(tmp_path, [reply("remembered"), reply("ALPHA"), reply("new")])
    first = await runner.run("记住代号 ALPHA", ws, 3, False, session_id="session-a")
    restarted = AgentRunner(client=client, state_dir=tmp_path / "state")
    second = await restarted.run("代号是什么", ws, 3, False, session_id="session-a")
    await restarted.run("代号是什么", ws, 3, False, session_id="session-b")
    assert first.status == second.status == "success"
    assert "ALPHA" in json.dumps(client.requests[1])
    assert "ALPHA" not in json.dumps(client.requests[2])
    assert first.run_id != second.run_id


async def test_profile_workspace_binding(tmp_path):
    runner, client, ws = make_runner(tmp_path, [reply()])
    await runner.run("hello", ws, 2, False, session_id="same")
    other = tmp_path / "other"
    other.mkdir()
    failed = await runner.run("hello", other, 2, False, session_id="same")
    assert failed.status == "failed" and "绑定" in failed.error
    assert len(client.requests) == 1


@pytest.mark.parametrize("approved", [True, False])
async def test_frontend_approval_controls_real_write(tmp_path, approved):
    broker = ApprovalBroker(timeout=1)
    runner, client, ws = make_runner(
        tmp_path,
        [reply("", [call("write_file", {"path": "note.txt", "content": "hello"})]), reply()],
        broker,
    )
    events = []
    async for event in runner.stream_run("创建说明", ws, 3, False, session_id="a"):
        events.append(event)
        if event.get("type") == "permission.requested":
            assert not (ws / "note.txt").exists()
            assert "+hello" in event["diff"]
            broker.resolve(event["approval_id"], approved)
    assert (ws / "note.txt").exists() == approved
    assert not broker.pending
    assert client.requests[1][-1]["is_error"] == (not approved)


async def test_approval_does_not_block_event_loop_and_times_out(tmp_path):
    broker = ApprovalBroker(timeout=0.01)
    runner, _, ws = make_runner(
        tmp_path, [reply("", [call("write_file", {"path": "x", "content": "x"})]), reply()], broker
    )
    heartbeat = []

    async def tick():
        await asyncio.sleep(0)
        heartbeat.append(True)

    ticker = asyncio.create_task(tick())
    events = [e async for e in runner.stream_run("write", ws, 3, False)]
    await ticker
    assert heartbeat and not (ws / "x").exists() and not broker.pending
    assert any(e.get("type") == "tool.failed" for e in events)


async def test_closing_stream_discards_pending_approval(tmp_path):
    broker = ApprovalBroker(timeout=1)
    runner, _, ws = make_runner(
        tmp_path, [reply("", [call("write_file", {"path": "x", "content": "x"})])], broker
    )
    stream = runner.stream_run("write", ws, 3, False, session_id="a")
    async for event in stream:
        if event.get("type") == "permission.requested":
            await stream.aclose()
            break
    assert not broker.pending and not runner.active_sessions
    assert not (ws / "x").exists()


async def test_direct_nonstream_run_denies_unapproved_write(tmp_path):
    runner, _, ws = make_runner(
        tmp_path, [reply("", [call("write_file", {"path": "x", "content": "x"})]), reply()]
    )
    result = await runner.run("create", ws, 3, False)
    assert result.status == "success" and not (ws / "x").exists()


async def test_stream_failure_records_failed_result(tmp_path):
    runner, _, ws = make_runner(tmp_path, [RuntimeError("synthetic stream failure")])
    events = [e async for e in runner.stream_run("hello", ws, 3, False)]
    result = events[-1]["result"]
    assert result["status"] == "failed" and "synthetic" in result["error"]
    log = [json.loads(line) for line in Path(result["events_path"]).read_text().splitlines()]
    assert log[-1]["type"] == "run.finished" and log[-1]["status"] == "failed"


async def test_current_turn_budget_counts_arguments_and_does_not_cross_contaminate(tmp_path):
    runner, _, ws = make_runner(tmp_path, [reply(), reply()])
    small, large = await asyncio.gather(
        runner.run("x" * 6000, ws, 2, False, context_max_chars=4000, session_id="small"),
        runner.run("hello", ws, 2, False, context_max_chars=100_000, session_id="large"),
    )
    assert small.status == "failed" and large.status == "success"
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("write_file", {"content": "x" * 100_000})],
        }
    ]
    assert should_compact(messages, ContextBudget(4000))
    assert len({new_run_id() for _ in range(1000)}) == 1000


async def test_disabled_tool_is_not_executed(tmp_path):
    runner, client, ws = make_runner(
        tmp_path, [reply("", [call("rag_search", {"query": "anything"})]), reply()]
    )
    await runner.run("hello", ws, 3, False, enable_rag=False)
    assert "not enabled" in client.requests[1][-1]["content"]


async def test_compaction_preserves_current_tool_group(tmp_path):
    current = [
        {"role": "user", "content": "new requirement"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "opaque",
            "tool_calls": [call("list_dir", {})],
        },
        {"role": "tool", "tool_call_id": "call1", "content": "files"},
    ]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old" * 3000},
        {"role": "assistant", "content": "old answer"},
        *current,
    ]
    client = ScriptClient([reply("old task summary")])
    compacted, response = await compact_messages(
        client, messages, EventWriter(tmp_path / "events"), ContextBudget(4000)
    )
    assert compacted[-3:] == current
    assert estimate_chars(compacted) < estimate_chars(messages)
    assert response["usage"]["prompt_tokens"] == 10


def test_usage_includes_anthropic_cache_without_double_counting_openai():
    total = empty_usage()
    add_usage(
        total,
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 30,
            }
        },
        [],
        {},
    )
    assert total["total_tokens"] == 65 and total["cache_tokens"] == 50
    add_usage(
        total,
        {
            "usage": {
                "prompt_tokens": 60,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 20},
            }
        },
        [],
        {},
    )
    assert total["total_tokens"] == 130 and total["cache_tokens"] == 70


async def test_explicit_skill_is_loaded_only_when_requested(tmp_path):
    runner, client, ws = make_runner(tmp_path, [reply()])
    _ = [e async for e in runner.stream_run("review", ws, 2, False, skill="repo-review")]
    assert "严重程度" in client.requests[0][1]["content"]
