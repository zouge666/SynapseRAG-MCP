from types import SimpleNamespace

import pytest

from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


def test_factory_routes_recursive_splitter_provider() -> None:
    splitter = SplitterFactory.create(SimpleNamespace(provider="recursive", chunk_size=80, chunk_overlap=10))

    assert isinstance(splitter, RecursiveSplitter)


def test_recursive_splitter_preserves_markdown_heading_boundaries() -> None:
    splitter = RecursiveSplitter(SimpleNamespace(provider="recursive", chunk_size=70, chunk_overlap=0))
    text = "# Intro\n\nSmall intro text.\n\n## Details\n\n" + "detail " * 18

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert chunks[0].startswith("# Intro")
    assert any(chunk.startswith("## Details") for chunk in chunks)
    assert all(len(chunk) <= 70 for chunk in chunks)


def test_recursive_splitter_keeps_fenced_code_block_together() -> None:
    splitter = RecursiveSplitter(SimpleNamespace(provider="recursive", chunk_size=45, chunk_overlap=0))
    code = "```python\nprint('alpha')\nprint('beta')\nprint('gamma')\n```"
    text = f"Before paragraph.\n\n{code}\n\nAfter paragraph."

    chunks = splitter.split_text(text)

    assert code in chunks
    assert all(not chunk.startswith("```python") or chunk.endswith("```") for chunk in chunks)


def test_recursive_splitter_uses_overlap_between_chunks() -> None:
    splitter = RecursiveSplitter(SimpleNamespace(provider="recursive", chunk_size=35, chunk_overlap=5))
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert chunks[0][-5:].strip()
    assert chunks[0][-5:].lstrip() in chunks[1]


def test_recursive_splitter_accepts_mapping_settings() -> None:
    splitter = RecursiveSplitter({"provider": "recursive", "chunk_size": 20, "overlap": 0})

    chunks = splitter.split_text("one two three four five six seven")

    assert chunks
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_recursive_splitter_rejects_invalid_text_and_overlap() -> None:
    splitter = RecursiveSplitter(SimpleNamespace(provider="recursive", chunk_size=20, chunk_overlap=20))

    with pytest.raises(ValueError, match="chunk_overlap"):
        splitter.split_text("hello")

    with pytest.raises(ValueError, match="text"):
        RecursiveSplitter(SimpleNamespace(provider="recursive")).split_text(123)
