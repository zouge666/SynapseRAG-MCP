from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.tools import default_tools


class MCPServerError(ValueError):
    pass


class MCPServer:
    def __init__(self, stderr: TextIO | None = None, handler: ProtocolHandler | None = None) -> None:
        self.stderr = stderr or sys.stderr
        self.handler = handler or ProtocolHandler(tools=default_tools())

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
            return self.handler.error_response(None, -32700, "Parse error")
        return self.handle_request(request)

    def handle_request(self, request: Any) -> dict[str, Any] | None:
        response = self.handler.handle_request(request)
        method = request.get("method") if isinstance(request, dict) else None
        if response is None and method == "notifications/initialized":
            self._log("client initialized")
        elif isinstance(response, dict) and "error" in response:
            error = response["error"]
            self._log(f"jsonrpc error {error['code']}: {error['message']}")
        elif method == "initialize":
            self._log("initialize handled")
        return response

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
