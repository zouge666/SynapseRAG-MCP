from __future__ import annotations

from time import perf_counter
from typing import Any

from core import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessorError(ValueError):
    pass


class BatchProcessor:
    def __init__(
        self,
        settings: object,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.settings = settings
        self.dense_encoder = dense_encoder or DenseEncoder(settings)
        self.sparse_encoder = sparse_encoder or SparseEncoder()
        self.batch_size = batch_size if batch_size is not None else self._batch_size(settings)
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise BatchProcessorError("batch_size must be a positive integer")

    def process(self, chunks: list[Chunk], trace: object | None = None) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        total_start = perf_counter()
        for batch_index, batch in enumerate(self.iter_batches(chunks)):
            batch_start = perf_counter()
            dense_records = self.dense_encoder.encode(batch, trace=trace)
            sparse_records = self.sparse_encoder.encode(batch, trace=trace)
            records.extend(self._merge_records(batch, dense_records, sparse_records))
            self._record(
                trace,
                "batch_processor.batch",
                {
                    "batch_index": batch_index,
                    "size": len(batch),
                    "duration_ms": self._elapsed_ms(batch_start),
                },
            )
        self._record(
            trace,
            "batch_processor",
            {
                "count": len(records),
                "batch_size": self.batch_size,
                "batch_count": len(self.iter_batches(chunks)),
                "duration_ms": self._elapsed_ms(total_start),
            },
        )
        return records

    def iter_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[index : index + self.batch_size] for index in range(0, len(chunks), self.batch_size)]

    def _merge_records(self, chunks: list[Chunk], dense_records: list[ChunkRecord], sparse_records: list[ChunkRecord]) -> list[ChunkRecord]:
        if len(dense_records) != len(chunks) or len(sparse_records) != len(chunks):
            raise BatchProcessorError("encoder record counts must match batch size")
        merged = []
        for chunk, dense_record, sparse_record in zip(chunks, dense_records, sparse_records, strict=True):
            if dense_record.id != chunk.id or sparse_record.id != chunk.id:
                raise BatchProcessorError("encoder record ids must match chunk ids")
            metadata = dict(dense_record.metadata)
            metadata.update(sparse_record.metadata)
            merged.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    dense_vector=dense_record.dense_vector,
                    sparse_vector=sparse_record.sparse_vector,
                )
            )
        return merged

    def _batch_size(self, settings: object) -> int:
        ingestion = getattr(settings, "ingestion", None)
        value = getattr(ingestion, "batch_size", None) if ingestion is not None else None
        if value is None and isinstance(settings, dict):
            value = settings.get("ingestion", {}).get("batch_size", settings.get("batch_size"))
        return value if isinstance(value, int) else 100

    def _elapsed_ms(self, start: float) -> float:
        return round((perf_counter() - start) * 1000, 3)

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
