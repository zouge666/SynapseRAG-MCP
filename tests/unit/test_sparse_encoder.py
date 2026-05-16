from core import Chunk, ChunkRecord
from core.trace import TraceContext
from ingestion.embedding import SparseEncoder


def chunk(text: str = "Hybrid search uses BM25 and dense search.", chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={"source_path": "docs/sample.pdf", "chunk_index": 0}, start_offset=0, end_offset=len(text), source_ref="doc-1")


def test_sparse_encoder_outputs_chunk_records() -> None:
    records = SparseEncoder().encode([chunk("BM25 search search")])

    assert len(records) == 1
    assert isinstance(records[0], ChunkRecord)
    assert records[0].id == "chunk-1"
    assert records[0].text == "BM25 search search"
    assert records[0].sparse_vector == {"bm25": 1.0, "search": 2.0}


def test_sparse_encoder_returns_one_record_per_chunk() -> None:
    chunks = [chunk("alpha beta", "chunk-1"), chunk("beta gamma", "chunk-2")]

    records = SparseEncoder(stop_words=set()).encode(chunks)

    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert [record.sparse_vector for record in records] == [{"alpha": 1.0, "beta": 1.0}, {"beta": 1.0, "gamma": 1.0}]


def test_sparse_encoder_filters_default_stop_words() -> None:
    record = SparseEncoder().encode([chunk("The retrieval is in the index")])[0]

    assert record.sparse_vector == {"index": 1.0, "retrieval": 1.0}


def test_sparse_encoder_accepts_custom_stop_words() -> None:
    record = SparseEncoder(stop_words={"alpha"}).encode([chunk("alpha beta beta")])[0]

    assert record.sparse_vector == {"beta": 2.0}


def test_sparse_encoder_handles_empty_text_explicitly() -> None:
    record = SparseEncoder().encode([chunk("", "chunk-empty")])[0]

    assert record.sparse_vector == {}
    assert record.metadata["sparse_token_count"] == 0
    assert record.metadata["sparse_unique_terms"] == 0


def test_sparse_encoder_preserves_metadata_copy() -> None:
    original = chunk("alpha beta")

    record = SparseEncoder(stop_words=set()).encode([original])[0]
    record.metadata["chunk_index"] = 99

    assert original.metadata["chunk_index"] == 0
    assert record.metadata["source_path"] == "docs/sample.pdf"


def test_sparse_encoder_adds_sparse_statistics_to_metadata() -> None:
    record = SparseEncoder(stop_words=set()).encode([chunk("alpha beta beta")])[0]

    assert record.metadata["sparse_token_count"] == 3
    assert record.metadata["sparse_unique_terms"] == 2


def test_sparse_encoder_tokenizes_hyphenated_and_numeric_terms() -> None:
    record = SparseEncoder(stop_words=set()).encode([chunk("GPT-4o BM25 v2 GPT-4o")])[0]

    assert record.sparse_vector == {"bm25": 1.0, "gpt-4o": 2.0, "v2": 1.0}


def test_sparse_encoder_tokenizes_cjk_characters() -> None:
    record = SparseEncoder(stop_words=set()).encode([chunk("检索 检索 RAG")])[0]

    assert record.sparse_vector == {"rag": 1.0, "检": 2.0, "索": 2.0}


def test_sparse_encoder_returns_empty_list_for_empty_input() -> None:
    records = SparseEncoder().encode([])

    assert records == []


def test_sparse_encoder_records_trace_stage() -> None:
    trace = TraceContext()

    SparseEncoder(stop_words=set()).encode([chunk("alpha beta"), chunk("beta gamma", "chunk-2")], trace=trace)

    assert trace.stages[0]["name"] == "sparse_encoder"
    assert trace.stages[0]["details"] == {"count": 2, "total_terms": 4, "vocabulary_size": 3}


def test_sparse_encoder_can_be_imported_from_package() -> None:
    from ingestion.embedding import SparseEncoder as ExportedSparseEncoder

    assert ExportedSparseEncoder is SparseEncoder
