from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from Agent_Assistant.rpc.protocol import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcRequest,
    JsonRpcSuccess,
    ProtocolError,
    make_error,
)

CommandHandler = Callable[[dict[str, Any]], Awaitable[Any]]
MAX_FRAME_BYTES = 2 * 1024 * 1024
CLIENT_DISCONNECTED_ERRORS = (ConnectionError, BrokenPipeError)


@dataclass
class StreamingResult:
    events: Any


class JsonRpcServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.handlers: dict[str, CommandHandler] = {}
        self.server: asyncio.AbstractServer | None = None

    def register(self, method: str, handler: CommandHandler) -> None:
        self.handlers[method] = handler

    async def start(self) -> str:
        self.server = await asyncio.start_server(
            self.handle_connection,
            host=self.host,
            port=self.port,
            limit=MAX_FRAME_BYTES,
        )
        return f"{self.host}:{self.port}"

    async def serve_forever(self) -> None:
        if self.server is None:
            await self.start()
        assert self.server is not None
        async with self.server:
            await self.server.serve_forever()

    async def handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                await self.handle_line(line, writer)
        except CLIENT_DISCONNECTED_ERRORS:
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except CLIENT_DISCONNECTED_ERRORS:
                pass

    async def handle_line(self, line: bytes, writer: asyncio.StreamWriter) -> None:
        request_id: str | int | None = None
        try:
            raw = json.loads(line)
            request = JsonRpcRequest.model_validate(raw)
            request_id = request.id
            handler = self.handlers.get(request.method)
            if handler is None:
                raise ProtocolError(METHOD_NOT_FOUND, f"Method not found: {request.method}")
            result = await handler(request.params)
            if isinstance(result, StreamingResult):
                async for event in result.events:
                    writer.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                    await writer.drain()
                return
            result_data = result.model_dump() if isinstance(result, BaseModel) else result
            await self.send(writer, JsonRpcSuccess(id=request.id, result=result_data))
        except CLIENT_DISCONNECTED_ERRORS:
            return
        except json.JSONDecodeError as exc:
            await self.send(writer, make_error(None, PARSE_ERROR, f"Parse error: {exc}"))
        except ValidationError as exc:
            await self.send(
                writer,
                make_error(request_id, INVALID_REQUEST, "Invalid request", str(exc)),
            )
        except ProtocolError as exc:
            await self.send(writer, make_error(request_id, exc.code, str(exc), exc.data))
        except Exception as exc:
            await self.send(
                writer,
                make_error(request_id, INTERNAL_ERROR, "Internal error", str(exc)),
            )

    async def send(self, writer: asyncio.StreamWriter, msg: BaseModel) -> None:
        try:
            writer.write(msg.model_dump_json().encode("utf-8") + b"\n")
            await writer.drain()
        except CLIENT_DISCONNECTED_ERRORS:
            pass


class JsonRpcClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    async def call(self, method: str, params: dict[str, Any]) -> Any:
        reader, writer = await asyncio.open_connection(self.host, self.port, limit=MAX_FRAME_BYTES)
        request = JsonRpcRequest(id="1", method=method, params=params)
        writer.write(request.model_dump_json().encode("utf-8") + b"\n")
        await writer.drain()
        line = await reader.readline()
        writer.close()
        await writer.wait_closed()
        if not line:
            raise RuntimeError("no response from core")
        response = json.loads(line)
        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"[{error.get('code')}] {error.get('message')}: {error.get('data')}")
        return response.get("result")

    async def stream(self, method: str, params: dict[str, Any]) -> Any:
        reader, writer = await asyncio.open_connection(self.host, self.port, limit=MAX_FRAME_BYTES)
        request = JsonRpcRequest(id="1", method=method, params=params)
        writer.write(request.model_dump_json().encode("utf-8") + b"\n")
        await writer.drain()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    raise RuntimeError("stream closed before final response")
                response = json.loads(line)
                if "error" in response:
                    error = response["error"]
                    raise RuntimeError(
                        f"[{error.get('code')}] {error.get('message')}: {error.get('data')}"
                    )
                if "result" in response:
                    yield {"type": "result", "data": response.get("result")}
                    return
                yield response
        finally:
            writer.close()
            await writer.wait_closed()
