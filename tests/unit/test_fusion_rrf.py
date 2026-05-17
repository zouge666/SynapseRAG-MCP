import pytest

from core import RetrievalResult
from core.query_engine import RRFusion, RRFusionError
from core.trace import TraceContext


def result(chunk_id: str, score: float, text: str | None = None) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, text=text or chunk_id, metadata={"source_path": f"docs/{chunk_id}.pdf"})


def test_rrf_fuses_dense_and_sparse_rankings() -> None:
    dense = [result("a", 0.9), result("b", 0.8), result("c", 0.7)]
    sparse = [result("c", 5.0), result("a", 4.0), result("d", 3.0)]

    fused = RRFusion(k=60).fuse(dense, sparse, top_k=4)

    assert [item.chunk_id for item in fused] == ["a", "c", "b", "d"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].metadata["rrf_ranks"] == {"dense": 1, "sparse": 2}
    assert fused[1].metadata["rrf_ranks"] == {"dense": 3, "sparse": 1}


def test_rrf_uses_best_source_record_for_duplicate_chunks() -> None:
    dense = [result("a", 0.4, "dense text")]
    sparse = [result("a", 0.9, "sparse text")]

    fused = RRFusion(k=60).fuse(dense, sparse, top_k=1)

    assert fused[0].text == "sparse text"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 61)


def test_rrf_output_is_deterministic_on_ties() -> None:
    dense = [result("b", 0.5), result("a", 0.5)]
    sparse = []

    fused = RRFusion(k=60).fuse(dense, sparse, top_k=2)

    assert [item.chunk_id for item in fused] == ["b", "a"]


def test_rrf_top_k_limits_results() -> None:
    dense = [result("a", 1.0), result("b", 0.9), result("c", 0.8)]

    fused = RRFusion(k=60).fuse(dense, [], top_k=2)

    assert [item.chunk_id for item in fused] == ["a", "b"]


def test_rrf_top_k_zero_returns_empty() -> None:
    assert RRFusion().fuse([result("a", 1.0)], [], top_k=0) == []


def test_rrf_k_parameter_changes_score() -> None:
    dense = [result("a", 1.0)]

    assert RRFusion(k=10).fuse(dense, [], top_k=1)[0].score == pytest.approx(1 / 11)
    assert RRFusion(k=60).fuse(dense, [], top_k=1)[0].score == pytest.approx(1 / 61)


def test_invalid_k_raises() -> None:
    with pytest.raises(RRFusionError, match="k"):
        RRFusion(k=0)


def test_invalid_result_list_raises() -> None:
    with pytest.raises(RRFusionError, match="dense_results"):
        RRFusion().fuse(["bad"], [], top_k=1)


def test_rrf_records_trace_stage() -> None:
    trace = TraceContext(trace_type="query")

    RRFusion(k=60).fuse([result("a", 1.0)], [result("b", 1.0)], top_k=2, trace=trace)

    assert trace.stages[0]["name"] == "fusion.rrf"
    assert trace.stages[0]["details"] == {"count": 2, "dense_count": 1, "sparse_count": 1, "k": 60}


def test_rrf_fusion_can_be_imported_from_package() -> None:
    from core.query_engine import RRFusion as ExportedRRFusion

    assert ExportedRRFusion is RRFusion
