from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from core import Document
from core.trace import TraceContext
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessor
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from ingestion.transform import BaseTransform, ChunkRefiner, ImageCaptioner, MetadataEnricher
from libs.loader import BaseLoader, FileIntegrityChecker, PdfLoader, SQLiteIntegrityChecker


ProgressCallback = Callable[[str, int, int], None]


class IngestionPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestionResult:
    source_path: str
    collection: str
    file_hash: str
    status: str
    chunk_count: int = 0
    vector_ids: list[str] | None = None
    image_count: int = 0
    skipped: bool = False
    trace: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "file_hash": self.file_hash,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "vector_ids": list(self.vector_ids or []),
            "image_count": self.image_count,
            "skipped": self.skipped,
            "trace": self.trace,
        }


class IngestionPipeline:
    def __init__(
        self,
        settings: object,
        loader: BaseLoader | None = None,
        integrity_checker: FileIntegrityChecker | None = None,
        chunker: DocumentChunker | None = None,
        transforms: list[BaseTransform] | None = None,
        batch_processor: BatchProcessor | None = None,
        vector_upserter: VectorUpserter | None = None,
        bm25_indexer: BM25Indexer | None = None,
        image_storage: ImageStorage | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or PdfLoader(image_root=self._setting("image_root", "data/images"))
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker(self._setting("integrity_db_path", "data/db/ingestion_history.db"))
        self.chunker = chunker or DocumentChunker(settings)
        self.transforms = transforms if transforms is not None else [ChunkRefiner(settings), MetadataEnricher(settings), ImageCaptioner(settings)]
        self.batch_processor = batch_processor or BatchProcessor(settings)
        self.vector_upserter = vector_upserter or VectorUpserter(settings)
        self.bm25_indexer = bm25_indexer or BM25Indexer(self._setting("bm25_path", "data/db/bm25"))
        self.image_storage = image_storage or ImageStorage(
            image_root=self._setting("image_root", "data/images"),
            db_path=self._setting("image_db_path", "data/db/image_index.db"),
        )

    def run(
        self,
        source_path: str | Path,
        collection: str = "default",
        force: bool = False,
        on_progress: ProgressCallback | None = None,
        trace: TraceContext | None = None,
    ) -> IngestionResult:
        path = str(source_path)
        file_size = Path(path).stat().st_size if Path(path).exists() else None
        active_trace = trace or TraceContext(trace_type="ingestion", metadata={"source_path": path, "collection": collection})
        file_hash = ""
        total = 7
        try:
            file_hash = self._run_stage("integrity", lambda: self.integrity_checker.compute_sha256(path), active_trace, on_progress, 1, total)
            if self.integrity_checker.should_skip(file_hash) and not force:
                active_trace.record_stage("pipeline.skip", {"file_hash": file_hash})
                return IngestionResult(
                    source_path=path,
                    collection=collection,
                    file_hash=file_hash,
                    status="skipped",
                    skipped=True,
                    trace=active_trace.finish("skipped"),
                )
            document = self._run_stage("load", lambda: self.loader.load(path), active_trace, on_progress, 2, total)
            document = self._run_stage("image_store", lambda: self._store_document_images(document, collection, file_hash), active_trace, on_progress, 3, total)
            chunks = self._run_stage("split", lambda: self.chunker.split_document(document), active_trace, on_progress, 4, total)
            chunks = self._run_stage("transform", lambda: self._transform(chunks, active_trace), active_trace, on_progress, 5, total)
            records = self._run_stage("encode", lambda: self.batch_processor.process(chunks, trace=active_trace), active_trace, on_progress, 6, total, trace_stage="embed")
            vector_ids = self._run_stage(
                "store",
                lambda: self._store_records(records, active_trace, replace_source=path if force else None),
                active_trace,
                on_progress,
                7,
                total,
                trace_stage="upsert",
            )
            self.integrity_checker.mark_success(file_hash, path, file_size=file_size, chunk_count=len(records))
            active_trace.record_stage("pipeline", {"status": "success", "chunk_count": len(records), "image_count": self._image_count(document)})
            return IngestionResult(
                source_path=path,
                collection=collection,
                file_hash=file_hash,
                status="success",
                chunk_count=len(records),
                vector_ids=vector_ids,
                image_count=self._image_count(document),
                trace=active_trace.finish("success"),
            )
        except IngestionPipelineError as error:
            if file_hash:
                self.integrity_checker.mark_failed(file_hash, str(error), file_path=path, file_size=file_size)
            active_trace.record_stage("pipeline", {"status": "failed", "error": str(error)})
            active_trace.finish("failed")
            raise

    def _run_stage(
        self,
        stage: str,
        action: Callable[[], Any],
        trace: TraceContext,
        on_progress: ProgressCallback | None,
        current: int,
        total: int,
        trace_stage: str | None = None,
    ) -> Any:
        start = perf_counter()
        try:
            result = action()
        except Exception as error:
            raise IngestionPipelineError(f"{stage} failed: {error}") from error
        duration_ms = round((perf_counter() - start) * 1000, 3)
        details = self._stage_details(stage, result)
        trace.record_stage(stage, details, duration_ms=duration_ms)
        if trace_stage is not None and trace_stage != stage:
            trace.record_stage(trace_stage, self._stage_details(trace_stage, result), duration_ms=duration_ms)
        if on_progress is not None:
            on_progress(stage, current, total)
        return result

    def _store_document_images(self, document: Document, collection: str, file_hash: str) -> Document:
        images = document.metadata.get("images", [])
        if not isinstance(images, list) or not images:
            return document
        updated_images = []
        seen = set()
        for image in images:
            if not isinstance(image, dict):
                continue
            image_id = image.get("id")
            image_path = image.get("path")
            if not isinstance(image_id, str) or not isinstance(image_path, str) or image_id in seen:
                continue
            seen.add(image_id)
            updated = dict(image)
            page_num = image.get("page") if isinstance(image.get("page"), int) else None
            updated["path"] = self.image_storage.save_image(image_id, image_path, collection=collection, doc_hash=file_hash, page_num=page_num)
            updated_images.append(updated)
        metadata = dict(document.metadata)
        metadata["images"] = updated_images
        return Document(id=document.id, text=document.text, metadata=metadata)

    def _transform(self, chunks: list[Any], trace: TraceContext) -> list[Any]:
        transformed = chunks
        for transform in self.transforms:
            transformed = transform.transform(transformed, trace=trace)
        return transformed

    def _store_records(self, records: list[Any], trace: TraceContext, replace_source: str | None = None) -> list[str]:
        if replace_source:
            self._delete_existing_source(replace_source, trace)
        vector_ids = self.vector_upserter.upsert(records, trace=trace)
        self.bm25_indexer.upsert(records)
        self.bm25_indexer.save()
        return vector_ids

    def _delete_existing_source(self, source_path: str, trace: TraceContext) -> None:
        vector_count = 0
        delete_source = getattr(self.vector_upserter, "delete_source", None)
        if callable(delete_source):
            vector_count = int(delete_source(source_path, trace=trace))
        bm25_count = int(self.bm25_indexer.remove_document(source_path))
        if bm25_count:
            self.bm25_indexer.save()
        trace.record_stage("store.replace_source", {"source_path": source_path, "vector_count": vector_count, "bm25_count": bm25_count})

    def _stage_details(self, stage: str, result: Any) -> dict[str, Any]:
        if isinstance(result, list):
            details = {"count": len(result)}
        elif isinstance(result, Document):
            details = {"document_id": result.id, "image_count": self._image_count(result)}
        elif isinstance(result, str):
            details = {"value": result}
        else:
            details = {}
        method = self._stage_method(stage)
        if method:
            details["method"] = method
        return details

    def _stage_method(self, stage: str) -> str:
        if stage == "integrity":
            return self._method_name(self.integrity_checker)
        if stage == "load":
            return self._method_name(self.loader)
        if stage == "image_store":
            return self._method_name(self.image_storage)
        if stage == "split":
            return self._method_name(self.chunker)
        if stage == "transform":
            return "+".join(self._method_name(transform) for transform in self.transforms)
        if stage == "encode" or stage == "embed":
            return self._method_name(self.batch_processor)
        if stage == "store" or stage == "upsert":
            return self._method_name(self.vector_upserter)
        return "ingestion_pipeline"

    def _method_name(self, value: object) -> str:
        name = type(value).__name__
        parts: list[str] = []
        current = ""
        for index, char in enumerate(name):
            if char.isupper() and index and (not name[index - 1].isupper() or (index + 1 < len(name) and name[index + 1].islower())):
                parts.append(current)
                current = char.lower()
            else:
                current += char.lower()
        if current:
            parts.append(current)
        return "_".join(part for part in parts if part)

    def _image_count(self, document: Document) -> int:
        images = document.metadata.get("images", [])
        return len(images) if isinstance(images, list) else 0

    def _setting(self, name: str, default: str) -> str:
        ingestion = getattr(self.settings, "ingestion", None)
        value = getattr(ingestion, name, None) if ingestion is not None else None
        if value is None and isinstance(self.settings, dict):
            value = self.settings.get("ingestion", {}).get(name, self.settings.get(name))
        return value if isinstance(value, str) and value else default
