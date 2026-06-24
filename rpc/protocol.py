from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcErrorObject(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JsonRpcSuccess(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None
    result: Any


class JsonRpcError(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None
    error: JsonRpcErrorObject


class PingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = "ping"


class AgentRunParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str
    workspace: str = "."
    max_steps: int = Field(default=8, ge=1, le=30)
    auto_approve: bool = False
    enable_rag: bool = True
    context_max_chars: int = Field(default=24_000, ge=4_000, le=200_000)
    verbose: bool = False


class ProtocolError(ValueError):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


def make_error(
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any | None = None,
) -> JsonRpcError:
    return JsonRpcError(
        id=request_id,
        error=JsonRpcErrorObject(code=code, message=message, data=data),
    )


def validate_params(method: str, params: dict[str, Any]) -> BaseModel:
    try:
        if method == "ping":
            return PingParams.model_validate(params)
        if method in {"agent.run", "agent.run_stream"}:
            return AgentRunParams.model_validate(params)
    except ValidationError as exc:
        raise ProtocolError(INVALID_PARAMS, "Invalid params", str(exc)) from exc
    raise ProtocolError(METHOD_NOT_FOUND, f"Method not found: {method}")
