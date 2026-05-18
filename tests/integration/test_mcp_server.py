import json
import os
import subprocess
import sys
from pathlib import Path

from mcp_server.server import MCPServer


ROOT = Path(__file__).resolve().parents[2]


def run_server(input_text: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=5,
        cwd=str(ROOT),
        env=env,
        check=False,
    )


def test_mcp_server_initialize_over_stdio() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1.0"}},
    }

    completed = run_server(json.dumps(request) + "\n")

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == "2025-06-18"
    assert response["result"]["serverInfo"]["name"] == "synapserag-mcp"
    assert response["result"]["capabilities"] == {"tools": {}}
    assert "mcp server started" not in completed.stdout
    assert "mcp server started" in completed.stderr


def test_mcp_server_logs_to_stderr_without_polluting_stdout() -> None:
    request = {"jsonrpc": "2.0", "id": 2, "method": "unknown", "params": {}}

    completed = run_server(json.dumps(request) + "\n")

    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    response = json.loads(stdout_lines[0])
    assert response["error"] == {"code": -32601, "message": "Method not found"}
    assert "jsonrpc error -32601" in completed.stderr
    assert "jsonrpc error" not in completed.stdout


def test_mcp_server_handles_initialized_notification_without_response() -> None:
    request = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}

    completed = run_server(json.dumps(request) + "\n")

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert "client initialized" in completed.stderr


def test_mcp_server_parse_error_response() -> None:
    completed = run_server("{bad json}\n")

    response = json.loads(completed.stdout.strip())
    assert response["error"] == {"code": -32700, "message": "Parse error"}
    assert "invalid json" in completed.stderr


def test_mcp_server_handle_request_directly() -> None:
    server = MCPServer()

    response = server.handle_request({"jsonrpc": "2.0", "id": "abc", "method": "initialize", "params": {}})

    assert response["id"] == "abc"
    assert response["result"]["serverInfo"]["version"] == "0.1.0"
