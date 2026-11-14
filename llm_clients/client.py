from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from Agent_Assistant.config import LlmConfig


class ModelProtocolError(RuntimeError):
    """A provider response cannot safely be treated as a completed model turn."""


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    parts: list[str] = []
    size = 0
    async for line in response.aiter_lines():
        if not line:
            if parts:
                yield "\n".join(parts)
                parts, size = [], 0
        elif line.startswith("data:"):
            value = line[5:].lstrip(" ")
            size += len(value)
            if size > 4 * 1024 * 1024:
                raise ModelProtocolError("SSE event exceeds size limit")
            parts.append(value)
    if parts:
        yield "\n".join(parts)


def validate_completion(data: dict[str, Any]) -> None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelProtocolError("Missing response choice")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason not in {"stop", "tool_calls"}:
        raise ModelProtocolError(f"Model did not complete normally: {reason}")
    message = choice.get("message", {})
    calls = message.get("tool_calls") or []
    if reason == "tool_calls" and not calls:
        raise ModelProtocolError("Tool-call finish without tool calls")
    ids: set[str] = set()
    for call in calls:
        call_id = call.get("id")
        function = call.get("function", {})
        if not call_id or call_id in ids or not function.get("name"):
            raise ModelProtocolError("Invalid or duplicate tool call identity")
        ids.add(call_id)
        try:
            args = json.loads(function.get("arguments", ""))
        except (ValueError, TypeError) as exc:
            raise ModelProtocolError(
                "Incomplete/invalid tool arguments; tool not executed"
            ) from exc
        if not isinstance(args, dict):
            raise ModelProtocolError("Tool arguments must be an object")
    if not calls and not str(message.get("content") or "").strip():
        raise ModelProtocolError("Model returned an empty answer")


class ModelClient:
    """Small protocol adapters preserving provider-specific assistant state."""

    def __init__(self, config: LlmConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15), transport=transport)

    async def aclose(self) -> None:
        await self.http.aclose()

    def _request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        if self.config.api_mode == "anthropic":
            base = self.config.base_url.removesuffix("/v1")
            return (
                f"{base}/v1/messages",
                {"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"},
                self._build_anthropic_payload(messages, tools),
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    k: v
                    for k, v in m.items()
                    if not k.startswith("_")
                    and k != "is_error"
                    and not (k == "tool_calls" and not v)
                }
                for m in messages
            ],
        }
        token_key = "max_completion_tokens" if self.config.provider == "openai" else "max_tokens"
        payload[token_key] = self.config.max_output_tokens
        if self.config.provider == "deepseek" and self.config.thinking != "default":
            payload["thinking"] = {"type": self.config.thinking}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        return (
            f"{self.config.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.config.api_key}"},
            payload,
        )

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_error:
            # Do not copy arbitrary provider error bodies (which can contain secrets)
            # into the terminal or persisted task logs.
            raise ModelProtocolError(
                f"Model HTTP {response.status_code}; check provider configuration"
            )

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        url, headers, payload = self._request(messages, tools)
        response = await self.http.post(url, headers=headers, json=payload)
        await self._raise_for_status(response)
        data = response.json()
        if not isinstance(data, dict):
            raise ModelProtocolError("Provider returned non-object JSON")
        if self.config.api_mode == "anthropic":
            data = self._anthropic_to_openai(data)
        validate_completion(data)
        result: dict[str, Any] = data
        return result

    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        url, headers, payload = self._request(messages, tools)
        if self.config.provider == "openai":
            payload["stream_options"] = {"include_usage": True}
        async with self.http.stream(
            "POST", url, headers=headers, json={**payload, "stream": True}
        ) as response:
            await self._raise_for_status(response)
            parser = (
                self._anthropic_stream(response)
                if self.config.api_mode == "anthropic"
                else self._chat_stream(response)
            )
            async for event in parser:
                yield event

    async def _chat_stream(self, response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        message: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": []}
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        done = False
        async for raw in sse_data(response):
            if raw == "[DONE]":
                done = True
                break
            chunk = json.loads(raw)
            if chunk.get("error"):
                raise ModelProtocolError("Provider returned a stream error")
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for field in ("content", "reasoning_content"):
                if delta.get(field) is not None:
                    message[field] = message.get(field, "") + str(delta[field])
            if delta.get("content"):
                yield {"type": "text_delta", "text": delta["content"]}
            self._merge_openai_tool_delta(message, delta)
        if not done:
            raise ModelProtocolError("Model stream ended before [DONE]")
        data: dict[str, Any] = {"choices": [{"message": message, "finish_reason": finish_reason}]}
        if usage is not None:
            data["usage"] = usage
        validate_completion(data)
        yield {"type": "final", "data": data}

    async def _anthropic_stream(self, response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        blocks: dict[int, dict[str, Any]] = {}
        active: set[int] = set()
        usage: dict[str, Any] = {}
        stop_reason: str | None = None
        done = False
        async for raw in sse_data(response):
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "error":
                raise ModelProtocolError("Anthropic returned a stream error")
            if kind == "message_start":
                usage.update(event.get("message", {}).get("usage", {}))
            elif kind == "content_block_start":
                index = int(event["index"])
                if index in blocks:
                    raise ModelProtocolError("Duplicate content block")
                blocks[index] = copy.deepcopy(event["content_block"])
                active.add(index)
            elif kind == "content_block_delta":
                index = int(event["index"])
                if index not in active:
                    raise ModelProtocolError("Delta without an active block")
                block, delta = blocks[index], event["delta"]
                field = {
                    "text_delta": "text",
                    "thinking_delta": "thinking",
                    "signature_delta": "signature",
                    "input_json_delta": "partial_json",
                }.get(delta["type"])
                if field is None:
                    raise ModelProtocolError(f"Unsupported content delta: {delta['type']}")
                block[field] = str(block.get(field, "")) + str(delta.get(field, ""))
                if field == "text":
                    yield {"type": "text_delta", "text": delta[field]}
            elif kind == "content_block_stop":
                active.discard(int(event["index"]))
            elif kind == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
                usage.update(event.get("usage", {}))
            elif kind == "message_stop":
                done = True
                break
        if not done or active:
            raise ModelProtocolError("Anthropic stream ended before message/block completion")
        normalized = [self._normalize_anthropic_block(blocks[i]) for i in sorted(blocks)]
        data = self._anthropic_to_openai(
            {"content": normalized, "usage": usage, "stop_reason": stop_reason}
        )
        validate_completion(data)
        yield {"type": "final", "data": data}

    def _merge_openai_tool_delta(self, message: dict[str, Any], delta: dict[str, Any]) -> None:
        calls = message["tool_calls"]
        for part in delta.get("tool_calls") or []:
            index = int(part.get("index", 0))
            if not 0 <= index < 64:
                raise ModelProtocolError("Invalid tool index")
            while len(calls) <= index:
                calls.append(
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
            call = calls[index]
            if part.get("id"):
                call["id"] = part["id"]
            for key in ("name", "arguments"):
                call["function"][key] += str((part.get("function") or {}).get(key) or "")

    def _normalize_anthropic_block(self, block: dict[str, Any]) -> dict[str, Any]:
        block = copy.deepcopy(block)
        if "partial_json" in block:
            try:
                block["input"] = json.loads(block.pop("partial_json"))
            except (ValueError, TypeError) as exc:
                raise ModelProtocolError("Incomplete Anthropic tool JSON") from exc
        return block

    def _assistant_to_anthropic(self, message: dict[str, Any]) -> dict[str, Any]:
        if "_anthropic_content" in message:
            return {"role": "assistant", "content": copy.deepcopy(message["_anthropic_content"])}
        blocks: list[dict[str, Any]] = []
        if message.get("content"):
            blocks.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": fn["name"],
                    "input": json.loads(fn["arguments"]),
                }
            )
        return {"role": "assistant", "content": blocks}

    def _build_anthropic_payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        system: list[str] = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                system.append(str(message["content"]))
            elif role == "assistant":
                converted.append(self._assistant_to_anthropic(message))
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                    "is_error": bool(message.get("is_error")),
                }
                if (
                    converted
                    and converted[-1]["role"] == "user"
                    and isinstance(converted[-1]["content"], list)
                ):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            else:
                converted.append({"role": "user", "content": str(message["content"])})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "messages": converted,
            "system": "\n".join(system),
        }
        if tools:
            payload["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
        if self.config.thinking == "disabled":
            payload["thinking"] = {"type": "disabled"}
        elif self.config.thinking == "enabled":
            raise ValueError(
                "Anthropic thinking configuration is model-specific; use default or disabled"
            )
        return payload

    def _anthropic_to_openai(self, data: dict[str, Any]) -> dict[str, Any]:
        blocks = data.get("content", [])
        calls = []
        text = []
        for block in blocks:
            if block["type"] == "text":
                text.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"], ensure_ascii=False),
                        },
                    }
                )
            elif block["type"] not in {"thinking", "redacted_thinking"}:
                raise ModelProtocolError(f"Unsupported Anthropic block: {block['type']}")
        reason = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls"}.get(
            str(data.get("stop_reason")), data.get("stop_reason")
        )
        return {
            "choices": [
                {
                    "finish_reason": reason,
                    "message": {
                        "role": "assistant",
                        "content": "".join(text),
                        "tool_calls": calls,
                        "_anthropic_content": copy.deepcopy(blocks),
                    },
                }
            ],
            "usage": data.get("usage", {}),
        }


# Import compatibility for integrations using the original class name.
DeepSeekClient = ModelClient
