from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.settings import Settings, load_settings


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    provider: str
    detail: str
    status: str = "configured"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "detail": self.detail,
            "status": self.status,
            "metadata": dict(self.metadata or {}),
        }


class ConfigService:
    def __init__(self, settings_path: str | Path = "config/settings.yaml") -> None:
        self.settings_path = Path(settings_path)

    def load(self) -> Settings:
        return load_settings(str(self.settings_path))

    def app_summary(self, settings: Settings | None = None) -> dict[str, str]:
        active = settings or self.load()
        return {
            "name": active.app.name,
            "environment": active.app.environment,
            "settings_path": str(self.settings_path),
        }

    def component_configs(self, settings: Settings | None = None) -> list[ComponentConfig]:
        active = settings or self.load()
        components = [
            ComponentConfig("LLM", active.llm.provider, active.llm.model),
            ComponentConfig("Embedding", active.embedding.provider, active.embedding.model, metadata=self._optional_dimension(active.embedding.dimensions)),
            ComponentConfig("Vector Store", active.vector_store.backend, active.vector_store.collection, metadata={"persist_path": active.vector_store.persist_path}),
            ComponentConfig("Splitter", active.splitter.provider, f"{active.splitter.chunk_size}/{active.splitter.chunk_overlap}"),
            ComponentConfig("Sparse Retrieval", active.retrieval.sparse_backend, active.retrieval.fusion_algorithm),
            ComponentConfig("Reranker", active.rerank.backend, active.rerank.model or "disabled", status="enabled" if active.rerank.enabled else "disabled"),
            ComponentConfig("Evaluation", ",".join(active.evaluation.backends) or "none", "enabled" if active.evaluation.enabled else "disabled"),
            ComponentConfig("Observability", "jsonl", active.observability.trace_path, metadata={"log_path": active.observability.log_path}),
        ]
        if active.vision_llm is not None:
            components.append(ComponentConfig("Vision LLM", active.vision_llm.provider, active.vision_llm.model))
        return components

    def component_dicts(self, settings: Settings | None = None) -> list[dict[str, Any]]:
        return [component.to_dict() for component in self.component_configs(settings)]

    def _optional_dimension(self, dimensions: int | None) -> dict[str, Any]:
        return {"dimensions": dimensions} if dimensions is not None else {}
