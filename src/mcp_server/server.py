from __future__ import annotations

import json
import sys
from typing import Any, TextIO


PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "synapserag-mcp", "version": "0.1.0"}


class MCPServerError(ValueError):
    pass


class MCPServer:
    def __init__(self, stderr: TextIO | None = None) -> None:
        self.stderr = stderr or sys.stderr

    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        active_stdin = stdin or sys.stdin
        active_stdout = stdout or sys.stdout
        self._log("mcp server started")
        for line in active_stdin:
            if not line.strip():
                continue
            response = self.handle_line(line)
            if response is not None:
                active_stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
                active_stdout.flush()
        self._log("mcp server stopped")
        return 0

    def handle_line(self, line: str) -> dict[str, Any] | None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            self._log(f"invalid json: {error.msg}")
            return self._error(None, -32700, "Parse error")
        return self.handle_request(request)

    def handle_request(self, request: Any) -> dict[str, Any] | None:
        if not isinstance(request, dict):
            return self._error(None, -32600, "Invalid Request")
        request_id = request.get("id")
        method = request.get("method")
        if method == "notifications/initialized":
            self._log("client initialized")
            return None
        if not isinstance(method, str) or request.get("jsonrpc") != "2.0":
            return self._error(request_id, -32600, "Invalid Request")
        if method == "initialize":
            self._log("initialize handled")
            return self._result(request_id, self._initialize_result(request.get("params", {})))
        return self._error(request_id, -32601, "Method not found")

    def _initialize_result(self, params: Any) -> dict[str, Any]:
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        protocol_version = requested if isinstance(requested, str) and requested else PROTOCOL_VERSION
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": dict(SERVER_INFO),
        }

    def _result(self, request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        self._log(f"jsonrpc error {code}: {message}")
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _log(self, message: str) -> None:
        print(message, file=self.stderr, flush=True)


def main(stdin: TextIO | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    try:
        return MCPServer(stderr=stderr).serve(stdin=stdin, stdout=stdout)
    except Exception as error:
        print(f"mcp server failed: {error}", file=stderr or sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
