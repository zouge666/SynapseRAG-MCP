from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "synapserag-mcp", "version": "0.1.0"}


class JSONRPCError(ValueError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": dict(self.input_schema)}


class ProtocolHandler:
    def __init__(
        self,
        tools: Mapping[str, ToolDefinition] | None = None,
        protocol_version: str = PROTOCOL_VERSION,
        server_info: dict[str, str] | None = None,
    ) -> None:
        self.tools = dict(tools or {})
        self.protocol_version = protocol_version
        self.server_info = dict(server_info or SERVER_INFO)

    def handle_request(self, request: Any) -> dict[str, Any] | None:
        try:
            self._validate_request(request)
            request_id = request.get("id")
            method = request["method"]
            if method == "notifications/initialized":
                return None
            result = self._dispatch(method, request.get("params", {}))
            return self.result_response(request_id, result)
        except JSONRPCError as error:
            request_id = request.get("id") if isinstance(request, dict) else None
            return self.error_response(request_id, error.code, error.message)
        except Exception:
            request_id = request.get("id") if isinstance(request, dict) else None
            return self.error_response(request_id, -32603, "Internal error")

    def handle_initialize(self, params: Any) -> dict[str, Any]:
        if params is not None and not isinstance(params, dict):
            raise JSONRPCError(-32602, "Invalid params")
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        protocol_version = requested if isinstance(requested, str) and requested else self.protocol_version
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": dict(self.server_info),
        }

    def handle_tools_list(self) -> dict[str, Any]:
        return {"tools": [tool.schema() for tool in sorted(self.tools.values(), key=lambda item: item.name)]}

    def handle_tools_call(self, name: Any, arguments: Any) -> Any:
        if not isinstance(name, str) or not name:
            raise JSONRPCError(-32602, "Invalid params")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise JSONRPCError(-32602, "Invalid params")
        tool = self.tools.get(name)
        if tool is None:
            raise JSONRPCError(-32602, "Invalid params")
        return tool.handler(dict(arguments))

    def result_response(self, request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _validate_request(self, request: Any) -> None:
        if not isinstance(request, dict):
            raise JSONRPCError(-32600, "Invalid Request")
        if request.get("jsonrpc") != "2.0":
            raise JSONRPCError(-32600, "Invalid Request")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            raise JSONRPCError(-32600, "Invalid Request")

    def _dispatch(self, method: str, params: Any) -> Any:
        if method == "initialize":
            return self.handle_initialize(params)
        if method == "tools/list":
            if params not in ({}, None):
                raise JSONRPCError(-32602, "Invalid params")
            return self.handle_tools_list()
        if method == "tools/call":
            if not isinstance(params, dict):
                raise JSONRPCError(-32602, "Invalid params")
            return self.handle_tools_call(params.get("name"), params.get("arguments", {}))
        raise JSONRPCError(-32601, "Method not found")
