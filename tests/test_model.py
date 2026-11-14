import json

import httpx
import pytest
from Agent_Assistant.config import LlmConfig
from Agent_Assistant.llm_clients.deepseek import ModelClient, ModelProtocolError


def sse(frames):
    return "".join("data: " + (f if isinstance(f, str) else json.dumps(f)) + "\n\n" for f in frames)


async def parse(frames, mode="chat_completions"):
    async def handler(request):
        return httpx.Response(200, text=sse(frames), headers={"content-type": "text/event-stream"})

    client = ModelClient(
        LlmConfig("fake", "https://example.invalid", "fixture", mode),
        transport=httpx.MockTransport(handler),
    )
    try:
        return [event async for event in client.chat_stream([], [])]
    finally:
        await client.aclose()


async def test_openai_stream_preserves_reasoning_and_tool_fragments():
    frames = [
        {"choices": [{"delta": {"reasoning_content": "opaque-state"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "function": {"name": "read_file", "arguments": '{"path":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.txt"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 5}},
        "[DONE]",
    ]
    events = await parse(frames)
    data = events[-1]["data"]
    msg = data["choices"][0]["message"]
    assert msg["reasoning_content"] == "opaque-state"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt"}
    assert data["usage"]["prompt_tokens"] == 20


@pytest.mark.parametrize(
    "frames",
    [
        [{"choices": [{"delta": {"content": "partial"}}]}],
        [{"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]}, "[DONE]"],
        [{"error": {"message": "synthetic"}}, "[DONE]"],
        [{"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]"],
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c",
                                    "function": {"name": "read_file", "arguments": '{"path":'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            "[DONE]",
        ],
    ],
)
async def test_incomplete_or_failed_stream_never_finalizes(frames):
    with pytest.raises(ModelProtocolError):
        await parse(frames)


async def test_anthropic_preserves_thinking_signature_and_tool_json():
    frames = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 8}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "opaque"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signed-fixture"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call1", "name": "list_dir", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"."}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    ]
    events = await parse(frames, "anthropic")
    msg = events[-1]["data"]["choices"][0]["message"]
    client = ModelClient(LlmConfig("fake", "https://example.invalid/v1", "fixture", "anthropic"))
    try:
        url, _, payload = client._request(
            [msg, {"role": "tool", "tool_call_id": "call1", "content": "denied", "is_error": True}],
            [],
        )
        assert url == "https://example.invalid/v1/messages"
        assert payload["messages"][0]["content"][0] == {
            "type": "thinking",
            "thinking": "opaque",
            "signature": "signed-fixture",
        }
        assert payload["messages"][1]["content"][0]["is_error"] is True
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "frames",
    [
        [{"type": "error", "error": {"type": "overloaded_error"}}],
        [{"type": "message_start", "message": {}}],
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "partial"},
            },
            {"type": "message_stop"},
        ],
    ],
)
async def test_anthropic_errors_and_eof_fail(frames):
    with pytest.raises(ModelProtocolError):
        await parse(frames, "anthropic")


async def test_nonstream_and_stream_keep_same_provider_state():
    message = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [],
        "reasoning_content": "opaque",
    }

    async def handler(request):
        assert "_anthropic_content" not in request.content.decode()
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    client = ModelClient(
        LlmConfig("fake", "https://example.invalid", "fixture", "chat_completions"),
        transport=httpx.MockTransport(handler),
    )
    try:
        data = await client.chat([{**message, "_anthropic_content": []}], [])
        events = await parse([{"choices": [{"delta": message, "finish_reason": "stop"}]}, "[DONE]"])
        assert data["choices"][0]["message"] == events[-1]["data"]["choices"][0]["message"]
    finally:
        await client.aclose()


async def test_http_errors_do_not_echo_provider_body():
    async def handler(request):
        return httpx.Response(401, text="SECRET_FIXTURE_DO_NOT_LOG")

    client = ModelClient(
        LlmConfig("fake", "https://example.invalid", "fixture", "chat_completions"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ModelProtocolError) as exc:
            await client.chat([], [])
        assert "401" in str(exc.value) and "SECRET_FIXTURE" not in str(exc.value)
    finally:
        await client.aclose()
