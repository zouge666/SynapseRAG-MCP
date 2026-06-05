import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


SERVER_CODE = "\n".join(
    [
        "from core import RetrievalResult",
        "from mcp_server.protocol_handler import ProtocolHandler",
        "from mcp_server.server import MCPServer",
        "from mcp_server.tools.query_knowledge_hub import query_knowledge_hub_tool_definition",
        "class FakeSearch:",
        "    def search(self, query, top_k, filters=None, trace=None):",
        "        if trace is not None:",
        "            trace.record_stage('hybrid_search', {'count': 2})",
        "        return [",
        "            RetrievalResult(chunk_id='chunk-beta', score=0.92, text='beta answer text', metadata={'source_path': 'docs/beta.pdf', 'page': 4}),",
        "            RetrievalResult(chunk_id='chunk-alpha', score=0.41, text='alpha supporting text', metadata={'source_path': 'docs/alpha.pdf', 'page': 2}),",
        "        ][:top_k]",
        "class FakeReranker:",
        "    def rerank(self, query, candidates, trace=None):",
        "        if trace is not None:",
        "            trace.record_stage('reranker', {'count': len(candidates), 'fallback': False})",
        "        return sorted(candidates, key=lambda item: item.score, reverse=True)",
        "tool = query_knowledge_hub_tool_definition(settings={}, search_factory=lambda settings: FakeSearch(), reranker_factory=lambda settings: FakeReranker())",
        "server = MCPServer(handler=ProtocolHandler(tools={tool.name: tool}))",
        "raise SystemExit(server.serve())",
    ]
)


def run_client(requests: list[dict]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    payload = "\n".join(json.dumps(request, ensure_ascii=False) for request in requests) + "\n"
    return subprocess.run(
        [sys.executable, "-c", SERVER_CODE],
        input=payload,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        check=False,
        timeout=10,
    )


def response_by_id(stdout: str) -> dict:
    responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    return {response["id"]: response for response in responses}


def test_mcp_client_lists_and_calls_query_knowledge_hub_over_stdio() -> None:
    completed = run_client(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {"query": "find beta", "top_k": 2, "collection": "docs"},
                },
            },
        ]
    )

    assert completed.returncode == 0
    responses = response_by_id(completed.stdout)
    assert responses[1]["result"]["serverInfo"]["name"] == "synapserag-mcp"
    tools = {tool["name"]: tool for tool in responses[2]["result"]["tools"]}
    assert tools["query_knowledge_hub"]["inputSchema"]["required"] == ["query"]
    tool_result = responses[3]["result"]
    assert "beta answer text" in tool_result["content"][0]["text"]
    assert tool_result["structuredContent"]["citations"][0]["chunk_id"] == "chunk-beta"
    assert tool_result["structuredContent"]["citations"][0]["source"] == "docs/beta.pdf"
    assert tool_result["structuredContent"]["citations"][0]["page"] == 4
    assert "mcp server started" not in completed.stdout
    assert "mcp server started" in completed.stderr
    assert "client initialized" in completed.stderr
