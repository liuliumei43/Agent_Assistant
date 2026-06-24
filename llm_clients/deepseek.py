from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from Agent_Assistant.config import LlmConfig


class DeepSeekClient:
    def __init__(self, config: LlmConfig) -> None:
        self.config = config

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.config.api_mode == "anthropic":
            return await self._chat_anthropic(messages, tools)
        return await self._chat_completions(messages, tools)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        if self.config.api_mode == "anthropic":
            async for event in self._chat_anthropic_stream(messages, tools):
                yield event
            return
        async for event in self._chat_completions_stream(messages, tools):
            yield event

    async def _chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            await self._raise_for_status(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("LLM API returned a non-object JSON response")
            return data

    async def _chat_completions_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        final_message: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": []}
        usage: dict[str, Any] | None = None
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if raw == "[DONE]":
                        break
                    chunk = json.loads(raw)
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if not isinstance(delta, dict):
                        continue
                    text = str(delta.get("content") or "")
                    if text:
                        final_message["content"] += text
                        yield {"type": "text_delta", "text": text}
                    self._merge_openai_tool_delta(final_message, delta)
        result: dict[str, Any] = {"choices": [{"message": final_message}]}
        if usage is not None:
            result["usage"] = usage
        yield {"type": "final", "data": result}

    async def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_text = ""
        anthropic_messages: list[dict[str, Any]] = []
        pending_tool_results: dict[str, str] = {}

        for message in messages:
            role = str(message.get("role", ""))
            if role == "system":
                system_text += str(message.get("content", "")) + "\n"
                continue
            if role == "tool":
                tool_id = str(message.get("tool_call_id", ""))
                pending_tool_results[tool_id] = str(message.get("content", ""))
                continue
            if pending_tool_results:
                anthropic_messages.append(self._tool_results_to_anthropic(pending_tool_results))
                pending_tool_results = {}
            if role == "assistant":
                anthropic_messages.append(self._assistant_to_anthropic(message))
                continue
            if role == "user":
                anthropic_messages.append(self._user_to_anthropic(message))

        if pending_tool_results:
            anthropic_messages.append(self._tool_results_to_anthropic(pending_tool_results))

        anthropic_tools = [self._tool_to_anthropic(tool) for tool in tools]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/messages",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            await self._raise_for_status(response)
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("Anthropic API returned a non-object JSON response")
            return self._anthropic_to_openai(data)

    async def _chat_anthropic_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._build_anthropic_payload(messages, tools)
        content_blocks: list[dict[str, Any]] = []
        current_index: int | None = None
        usage: dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/v1/messages",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={**payload, "stream": True},
            ) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if not raw:
                        continue
                    event = json.loads(raw)
                    event_type = str(event.get("type", ""))
                    if event_type == "message_start":
                        msg_usage = event.get("message", {}).get("usage", {})
                        if isinstance(msg_usage, dict):
                            usage.update(msg_usage)
                    elif event_type == "content_block_start":
                        current_index = int(event.get("index") or 0)
                        block = event.get("content_block", {})
                        if isinstance(block, dict):
                            self._ensure_block(content_blocks, current_index)
                            content_blocks[current_index] = dict(block)
                    elif event_type == "content_block_delta" and current_index is not None:
                        delta = event.get("delta", {})
                        if not isinstance(delta, dict):
                            continue
                        self._ensure_block(content_blocks, current_index)
                        block = content_blocks[current_index]
                        if delta.get("type") == "text_delta":
                            text = str(delta.get("text") or "")
                            block["text"] = str(block.get("text") or "") + text
                            if text:
                                yield {"type": "text_delta", "text": text}
                        elif delta.get("type") == "input_json_delta":
                            block["partial_json"] = str(block.get("partial_json") or "") + str(
                                delta.get("partial_json") or ""
                            )
                    elif event_type == "content_block_stop":
                        current_index = None
                    elif event_type == "message_delta":
                        msg_usage = event.get("usage", {})
                        if isinstance(msg_usage, dict):
                            usage.update(msg_usage)

        normalized_blocks = [self._normalize_anthropic_block(block) for block in content_blocks]
        result = self._anthropic_to_openai({"content": normalized_blocks, "usage": usage})
        yield {"type": "final", "data": result}

    def _build_anthropic_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_text = ""
        anthropic_messages: list[dict[str, Any]] = []
        pending_tool_results: dict[str, str] = {}

        for message in messages:
            role = str(message.get("role", ""))
            if role == "system":
                system_text += str(message.get("content", "")) + "\n"
                continue
            if role == "tool":
                tool_id = str(message.get("tool_call_id", ""))
                pending_tool_results[tool_id] = str(message.get("content", ""))
                continue
            if pending_tool_results:
                anthropic_messages.append(self._tool_results_to_anthropic(pending_tool_results))
                pending_tool_results = {}
            if role == "assistant":
                anthropic_messages.append(self._assistant_to_anthropic(message))
                continue
            if role == "user":
                anthropic_messages.append(self._user_to_anthropic(message))

        if pending_tool_results:
            anthropic_messages.append(self._tool_results_to_anthropic(pending_tool_results))

        anthropic_tools = [self._tool_to_anthropic(tool) for tool in tools]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if anthropic_tools:
            payload["tools"] = anthropic_tools
        return payload

    async def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            await response.aread()
            body = response.text[:2_000]
            raise RuntimeError(
                f"LLM HTTP {response.status_code} from {response.url}: {body}"
            ) from exc

    def _merge_openai_tool_delta(
        self,
        final_message: dict[str, Any],
        delta: dict[str, Any],
    ) -> None:
        tool_deltas = delta.get("tool_calls") or []
        if not isinstance(tool_deltas, list):
            return
        tool_calls = final_message.setdefault("tool_calls", [])
        if not isinstance(tool_calls, list):
            return
        for tool_delta in tool_deltas:
            if not isinstance(tool_delta, dict):
                continue
            index = int(tool_delta.get("index") or 0)
            while len(tool_calls) <= index:
                tool_calls.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            target = tool_calls[index]
            if tool_delta.get("id"):
                target["id"] = str(tool_delta["id"])
            function = tool_delta.get("function")
            if isinstance(function, dict):
                target_function = target.setdefault("function", {"name": "", "arguments": ""})
                if function.get("name"):
                    target_function["name"] = str(target_function.get("name", "")) + str(
                        function["name"]
                    )
                if function.get("arguments"):
                    target_function["arguments"] = str(
                        target_function.get("arguments", "")
                    ) + str(function["arguments"])

    def _ensure_block(self, blocks: list[dict[str, Any]], index: int) -> None:
        while len(blocks) <= index:
            blocks.append({})

    def _normalize_anthropic_block(self, block: dict[str, Any]) -> dict[str, Any]:
        if block.get("type") == "tool_use" and "partial_json" in block:
            try:
                block["input"] = json.loads(str(block.get("partial_json") or "{}"))
            except json.JSONDecodeError:
                block["input"] = {}
        return block

    def _tool_to_anthropic(self, tool: dict[str, Any]) -> dict[str, Any]:
        function = tool.get("function", {})
        return {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
        }

    def _assistant_to_anthropic(self, message: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        text = str(message.get("content", ""))
        if text:
            content.append({"type": "text", "text": text})
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", {})
            raw_args = str(function.get("arguments") or "{}")
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "input": parsed_args,
                }
            )
        return {"role": "assistant", "content": content or [{"type": "text", "text": ""}]}

    def _user_to_anthropic(self, message: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        text = str(message.get("content", ""))
        if text:
            content.append({"type": "text", "text": text})
        return {"role": "user", "content": content or [{"type": "text", "text": ""}]}

    def _tool_results_to_anthropic(self, pending_tool_results: dict[str, str]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for tool_id, result in pending_tool_results.items():
            content.append({"type": "tool_result", "tool_use_id": tool_id, "content": result})
        return {"role": "user", "content": content}

    def _anthropic_to_openai(self, data: dict[str, Any]) -> dict[str, Any]:
        content = data.get("content", [])
        if not isinstance(content, list):
            content = []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    }
                )
        result: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "".join(text_parts),
                        "tool_calls": tool_calls,
                    }
                }
            ]
        }
        usage = data.get("usage")
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            result["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            }
        return result
