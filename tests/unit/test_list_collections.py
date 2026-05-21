from pathlib import Path

import pytest

from mcp_server.tools import default_tools
from mcp_server.tools.list_collections import ListCollectionsError, list_collections, list_collections_tool_definition


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_list_collections_returns_collection_names_and_stats(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    write_file(root / "docs" / "guide.md", "alpha")
    write_file(root / "docs" / "nested" / "chapter.md", "beta")
    write_file(root / "notes" / "idea.txt", "gamma")

    response = list_collections(root)

    text = response["content"][0]["text"]
    collections = response["structuredContent"]["collections"]
    assert "docs" in text
    assert "notes" in text
    assert [collection["name"] for collection in collections] == ["docs", "notes"]
    assert collections[0]["document_count"] == 2
    assert collections[0]["documents"] == ["guide.md", "nested/chapter.md"]
    assert collections[0]["total_bytes"] == 9
    assert collections[1]["document_count"] == 1
    assert collections[1]["documents"] == ["idea.txt"]
    assert collections[1]["total_bytes"] == 5


def test_list_collections_ignores_files_and_hidden_paths(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    write_file(root / "docs" / "visible.md", "alpha")
    write_file(root / "docs" / ".hidden.md", "secret")
    write_file(root / ".hidden" / "ignored.md", "secret")
    write_file(root / "loose.md", "ignored")

    response = list_collections(root)

    collections = response["structuredContent"]["collections"]
    assert [collection["name"] for collection in collections] == ["docs"]
    assert collections[0]["documents"] == ["visible.md"]
    assert collections[0]["document_count"] == 1


def test_list_collections_returns_empty_result_for_missing_root(tmp_path: Path) -> None:
    response = list_collections(tmp_path / "missing")

    assert response == {
        "content": [{"type": "text", "text": "No document collections found."}],
        "structuredContent": {"collections": []},
    }


def test_list_collections_rejects_file_root(tmp_path: Path) -> None:
    root = tmp_path / "documents.txt"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ListCollectionsError, match="collections root must be a directory"):
        list_collections(root)


def test_list_collections_tool_definition_uses_bound_root(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    write_file(root / "docs" / "guide.md", "alpha")
    tool = list_collections_tool_definition(root=root)

    response = tool.handler({})

    assert tool.name == "list_collections"
    assert tool.input_schema == {"type": "object", "properties": {}}
    assert response["structuredContent"]["collections"][0]["name"] == "docs"


def test_default_tools_registers_list_collections() -> None:
    tools = default_tools()

    assert "list_collections" in tools
    assert "query_knowledge_hub" in tools
