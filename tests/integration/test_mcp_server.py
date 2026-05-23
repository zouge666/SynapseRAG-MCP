import base64
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

from core import RetrievalResult
from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.server import MCPServer
from mcp_server.tools.get_document_summary import get_document_summary_tool_definition
from mcp_server.tools.list_collections import list_collections_tool_definition
from mcp_server.tools.query_knowledge_hub import query_knowledge_hub_tool_definition


ROOT = Path(__file__).resolve().parents[2]


class FakeSearch:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters or {}, "trace": trace})
        return list(self.results[:top_k])


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "candidates": candidates, "trace": trace})
        return sorted(candidates, key=lambda item: (-item.score, item.chunk_id))


def result(chunk_id: str, score: float, page: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"{chunk_id} text",
        metadata={"source_path": f"docs/{chunk_id}.pdf", "page": page},
    )


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


def test_mcp_server_tools_list_exposes_query_knowledge_hub() -> None:
    request = {"jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": {}}

    completed = run_server(json.dumps(request) + "\n")

    response = json.loads(completed.stdout.strip())
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    assert "get_document_summary" in tools
    assert "list_collections" in tools
    assert "query_knowledge_hub" in tools
    assert tools["get_document_summary"]["inputSchema"]["required"] == ["doc_id"]
    assert tools["list_collections"]["inputSchema"] == {"type": "object", "properties": {}}
    assert tools["query_knowledge_hub"]["inputSchema"]["required"] == ["query"]


def test_mcp_server_handle_request_directly() -> None:
    server = MCPServer()

    response = server.handle_request({"jsonrpc": "2.0", "id": "abc", "method": "initialize", "params": {}})

    assert response["id"] == "abc"
    assert response["result"]["serverInfo"]["version"] == "0.1.0"


def test_mcp_server_query_knowledge_hub_tool_call_returns_citations() -> None:
    search = FakeSearch([result("a", 0.3, 1), result("b", 0.9, 2)])
    reranker = FakeReranker()
    tool = query_knowledge_hub_tool_definition(
        settings={},
        search_factory=lambda settings: search,
        reranker_factory=lambda settings: reranker,
    )
    server = MCPServer(
        stderr=StringIO(),
        handler=ProtocolHandler(tools={tool.name: tool}),
    )
    request = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "query_knowledge_hub", "arguments": {"query": "find beta", "top_k": 2, "collection": "docs"}},
    }
    output = StringIO()

    server.serve(stdin=StringIO(json.dumps(request) + "\n"), stdout=output)

    response = json.loads(output.getvalue())
    tool_result = response["result"]
    assert "[1] b text" in tool_result["content"][0]["text"]
    assert tool_result["structuredContent"]["citations"][0]["source"] == "docs/b.pdf"
    assert tool_result["structuredContent"]["citations"][0]["page"] == 2
    assert tool_result["structuredContent"]["citations"][0]["chunk_id"] == "b"
    assert search.calls[0]["filters"] == {"collection": "docs"}


def test_mcp_server_list_collections_tool_call_returns_document_stats(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    (root / "docs").mkdir(parents=True)
    (root / "notes").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("alpha", encoding="utf-8")
    (root / "notes" / "idea.txt").write_text("beta", encoding="utf-8")
    tool = list_collections_tool_definition(root=root)
    server = MCPServer(
        stderr=StringIO(),
        handler=ProtocolHandler(tools={tool.name: tool}),
    )
    request = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {"name": "list_collections", "arguments": {}},
    }
    output = StringIO()

    server.serve(stdin=StringIO(json.dumps(request) + "\n"), stdout=output)

    response = json.loads(output.getvalue())
    collections = response["result"]["structuredContent"]["collections"]
    assert [collection["name"] for collection in collections] == ["docs", "notes"]
    assert collections[0]["document_count"] == 1
    assert collections[0]["documents"] == ["guide.md"]
    assert collections[1]["document_count"] == 1
    assert collections[1]["documents"] == ["idea.txt"]


def test_mcp_server_get_document_summary_tool_call_returns_metadata(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    root.mkdir(parents=True)
    (root / "docs.json").write_text(
        json.dumps(
            {
                "collection": "docs",
                "records": [
                    {
                        "id": "vec-1",
                        "vector": [1.0, 0.0],
                        "text": "Alpha text",
                        "metadata": {
                            "source_path": "docs/alpha.pdf",
                            "title": "Alpha",
                            "summary": "Alpha summary.",
                            "tags": ["alpha"],
                            "chunk_id": "chunk-1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tool = get_document_summary_tool_definition(persist_path=root)
    server = MCPServer(
        stderr=StringIO(),
        handler=ProtocolHandler(tools={tool.name: tool}),
    )
    request = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {"name": "get_document_summary", "arguments": {"doc_id": "alpha"}},
    }
    output = StringIO()

    server.serve(stdin=StringIO(json.dumps(request) + "\n"), stdout=output)

    response = json.loads(output.getvalue())
    document = response["result"]["structuredContent"]["document"]
    assert document["title"] == "Alpha"
    assert document["summary"] == "Alpha summary."
    assert document["tags"] == ["alpha"]


def test_mcp_server_query_knowledge_hub_tool_call_returns_image_content(tmp_path: Path) -> None:
    image_path = tmp_path / "img-1.png"
    image_bytes = b"\x89PNG\r\n\x1a\nimage"
    image_path.write_bytes(image_bytes)
    search = FakeSearch(
        [
            RetrievalResult(
                chunk_id="chunk-with-image",
                score=0.95,
                text="Alpha [IMAGE: img-1] text",
                metadata={
                    "source_path": "docs/image.pdf",
                    "image_refs": ["img-1"],
                    "images": [{"id": "img-1", "path": str(image_path), "text_offset": 6, "text_length": 14}],
                },
            )
        ]
    )
    reranker = FakeReranker()
    tool = query_knowledge_hub_tool_definition(
        settings={},
        search_factory=lambda settings: search,
        reranker_factory=lambda settings: reranker,
    )
    server = MCPServer(
        stderr=StringIO(),
        handler=ProtocolHandler(tools={tool.name: tool}),
    )
    request = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {"name": "query_knowledge_hub", "arguments": {"query": "show image"}},
    }
    output = StringIO()

    server.serve(stdin=StringIO(json.dumps(request) + "\n"), stdout=output)

    response = json.loads(output.getvalue())
    content = response["result"]["content"]
    assert content[0]["type"] == "text"
    assert content[1] == {
        "type": "image",
        "mimeType": "image/png",
        "data": base64.b64encode(image_bytes).decode("ascii"),
    }
