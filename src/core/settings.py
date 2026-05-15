from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class AppSettings:
    name: str
    environment: str = "local"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    deployment_name: str = ""
    base_url: str = ""
    max_image_size: int = 2048


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int | None = None
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    deployment_name: str = ""
    base_url: str = ""


@dataclass(frozen=True)
class VectorStoreSettings:
    backend: str
    persist_path: str
    collection: str = "default"


@dataclass(frozen=True)
class SplitterSettings:
    provider: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass(frozen=True)
class RetrievalSettings:
    sparse_backend: str
    fusion_algorithm: str
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    backend: str
    model: str = ""
    top_m: int = 30


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    backends: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ObservabilitySettings:
    log_path: str
    trace_path: str


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    splitter: SplitterSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    vision_llm: LLMSettings | None = None


def load_settings(path: str = "config/settings.yaml") -> Settings:
    config_path = Path(path)
    if not config_path.exists():
        raise SettingsError(f"settings file not found: {path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise SettingsError("settings root must be a mapping")

    settings = _parse_settings(raw)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    required = {
        "app.name": settings.app.name,
        "llm.provider": settings.llm.provider,
        "llm.model": settings.llm.model,
        "embedding.provider": settings.embedding.provider,
        "embedding.model": settings.embedding.model,
        "vector_store.backend": settings.vector_store.backend,
        "vector_store.persist_path": settings.vector_store.persist_path,
        "splitter.provider": settings.splitter.provider,
        "retrieval.sparse_backend": settings.retrieval.sparse_backend,
        "retrieval.fusion_algorithm": settings.retrieval.fusion_algorithm,
        "rerank.backend": settings.rerank.backend,
        "observability.log_path": settings.observability.log_path,
        "observability.trace_path": settings.observability.trace_path,
    }

    missing = [path for path, value in required.items() if value in (None, "")]
    if missing:
        raise SettingsError(f"missing required setting: {missing[0]}")

    if settings.vision_llm is not None:
        vision_missing = [
            path
            for path, value in {
                "vision_llm.provider": settings.vision_llm.provider,
                "vision_llm.model": settings.vision_llm.model,
            }.items()
            if value in (None, "")
        ]
        if vision_missing:
            raise SettingsError(f"missing required setting: {vision_missing[0]}")

    positive_ints = {
        "retrieval.top_k_dense": settings.retrieval.top_k_dense,
        "retrieval.top_k_sparse": settings.retrieval.top_k_sparse,
        "retrieval.top_k_final": settings.retrieval.top_k_final,
        "splitter.chunk_size": settings.splitter.chunk_size,
    }

    for path, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise SettingsError(f"{path} must be a positive integer")
    if not isinstance(settings.splitter.chunk_overlap, int) or settings.splitter.chunk_overlap < 0:
        raise SettingsError("splitter.chunk_overlap must be a non-negative integer")
    if settings.splitter.chunk_overlap >= settings.splitter.chunk_size:
        raise SettingsError("splitter.chunk_overlap must be smaller than splitter.chunk_size")

    if settings.llm.max_image_size <= 0:
        raise SettingsError("llm.max_image_size must be a positive integer")
    if settings.vision_llm is not None and settings.vision_llm.max_image_size <= 0:
        raise SettingsError("vision_llm.max_image_size must be a positive integer")


def _parse_settings(raw: dict[str, Any]) -> Settings:
    app = _section(raw, "app")
    llm = _section(raw, "llm")
    embedding = _section(raw, "embedding")
    vector_store = _section(raw, "vector_store")
    splitter = raw.get("splitter", {})
    if not isinstance(splitter, dict):
        raise SettingsError("splitter must be a mapping")
    retrieval = _section(raw, "retrieval")
    rerank = _section(raw, "rerank")
    evaluation = _section(raw, "evaluation")
    observability = _section(raw, "observability")
    vision_llm = raw.get("vision_llm")
    if vision_llm is not None and not isinstance(vision_llm, dict):
        raise SettingsError("vision_llm must be a mapping")

    return Settings(
        app=AppSettings(
            name=_text(app, "name"),
            environment=_text(app, "environment", "local"),
        ),
        llm=LLMSettings(
            provider=_text(llm, "provider"),
            model=_text(llm, "model"),
            api_key=_text(llm, "api_key", ""),
            azure_endpoint=_text(llm, "azure_endpoint", ""),
            api_version=_text(llm, "api_version", ""),
            deployment_name=_text(llm, "deployment_name", ""),
            base_url=_text(llm, "base_url", ""),
            max_image_size=_integer(llm, "max_image_size", 2048),
        ),
        embedding=EmbeddingSettings(
            provider=_text(embedding, "provider"),
            model=_text(embedding, "model"),
            dimensions=_optional_int(embedding, "dimensions"),
            api_key=_text(embedding, "api_key", ""),
            azure_endpoint=_text(embedding, "azure_endpoint", ""),
            api_version=_text(embedding, "api_version", ""),
            deployment_name=_text(embedding, "deployment_name", ""),
            base_url=_text(embedding, "base_url", ""),
        ),
        vector_store=VectorStoreSettings(
            backend=_text(vector_store, "backend", _text(vector_store, "provider")),
            persist_path=_text(vector_store, "persist_path", _text(vector_store, "persist_directory")),
            collection=_text(vector_store, "collection", "default"),
        ),
        splitter=SplitterSettings(
            provider=_text(splitter, "provider", _text(splitter, "backend", "recursive")),
            chunk_size=_integer(splitter, "chunk_size", 1000),
            chunk_overlap=_integer(splitter, "chunk_overlap", _integer(splitter, "overlap", 200)),
        ),
        retrieval=RetrievalSettings(
            sparse_backend=_text(retrieval, "sparse_backend", "bm25"),
            fusion_algorithm=_text(retrieval, "fusion_algorithm", "rrf"),
            top_k_dense=_integer(retrieval, "top_k_dense", _integer(retrieval, "top_k", 20)),
            top_k_sparse=_integer(retrieval, "top_k_sparse", _integer(retrieval, "top_k", 20)),
            top_k_final=_integer(retrieval, "top_k_final", _integer(retrieval, "top_k", 10)),
        ),
        rerank=RerankSettings(
            enabled=_boolean(rerank, "enabled", False),
            backend=_text(rerank, "backend", _text(rerank, "provider", "none")),
            model=_text(rerank, "model", ""),
            top_m=_integer(rerank, "top_m", 30),
        ),
        evaluation=EvaluationSettings(
            enabled=_boolean(evaluation, "enabled", False),
            backends=_text_list(evaluation, "backends"),
        ),
        observability=ObservabilitySettings(
            log_path=_text(observability, "log_path"),
            trace_path=_text(observability, "trace_path"),
        ),
        vision_llm=_parse_optional_llm_settings(vision_llm),
    )


def _parse_optional_llm_settings(section: dict[str, Any] | None) -> LLMSettings | None:
    if section is None:
        return None
    return LLMSettings(
        provider=_text(section, "provider"),
        model=_text(section, "model"),
        api_key=_text(section, "api_key", ""),
        azure_endpoint=_text(section, "azure_endpoint", ""),
        api_version=_text(section, "api_version", ""),
        deployment_name=_text(section, "deployment_name", ""),
        base_url=_text(section, "base_url", ""),
        max_image_size=_integer(section, "max_image_size", 2048),
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        raise SettingsError(f"missing required setting: {key}")
    if not isinstance(value, dict):
        raise SettingsError(f"{key} must be a mapping")
    return value


def _text(section: dict[str, Any], key: str, default: str | None = None) -> str:
    value = section.get(key, default)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SettingsError(f"{key} must be a string")
    return value


def _integer(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if not isinstance(value, int):
        raise SettingsError(f"{key} must be an integer")
    return value


def _optional_int(section: dict[str, Any], key: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise SettingsError(f"{key} must be an integer")
    return value


def _boolean(section: dict[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise SettingsError(f"{key} must be a boolean")
    return value


def _text_list(section: dict[str, Any], key: str) -> list[str]:
    value = section.get(key, [])
    if not isinstance(value, list):
        raise SettingsError(f"{key} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise SettingsError(f"{key} must contain only strings")
    return value
