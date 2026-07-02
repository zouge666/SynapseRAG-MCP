from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from core import RetrievalResult
from core.response.citation_generator import Citation, CitationGenerator
from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class AnswerGeneratorError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    provider: str
    model: str


class AnswerGenerator:
    system_prompt = (
        "You are a knowledge base assistant. Answer the question using only the numbered context "
        "passages below. Cite the supporting passages inline with [n] markers. If the context does "
        "not contain the answer, say so plainly instead of inventing facts. Respond in the same "
        "language as the question."
    )
    max_chunk_chars = 1200

    def __init__(self, settings: Settings, llm: BaseLLM | None = None, citation_generator: CitationGenerator | None = None) -> None:
        self.settings = settings
        self.llm = llm
        self.citation_generator = citation_generator or CitationGenerator()

    def generate(self, question: str, results: list[RetrievalResult], trace: Any | None = None) -> GeneratedAnswer:
        if not isinstance(question, str) or not question.strip():
            raise AnswerGeneratorError("question must be a non-empty string")
        if not isinstance(results, list) or not all(isinstance(result, RetrievalResult) for result in results):
            raise AnswerGeneratorError("results must be a list of RetrievalResult")
        if not results:
            raise AnswerGeneratorError("results must not be empty; retrieve passages before generating an answer")
        llm = self._llm()
        citations = self.citation_generator.generate(results)
        started = perf_counter()
        try:
            answer = llm.chat(self._messages(question, citations))
        except Exception as error:
            raise AnswerGeneratorError(f"llm answer generation failed: {error}") from error
        duration_ms = (perf_counter() - started) * 1000
        if not isinstance(answer, str) or not answer.strip():
            raise AnswerGeneratorError("llm returned an empty answer")
        self._record(trace, llm, duration_ms, citations, answer)
        return GeneratedAnswer(answer=answer.strip(), citations=citations, provider=llm.settings.provider, model=llm.settings.model)

    def _llm(self) -> BaseLLM:
        if self.llm is not None:
            return self.llm
        llm_settings = self.settings.llm
        if not getattr(llm_settings, "api_key", ""):
            raise AnswerGeneratorError("LLM API key is not configured; add one on the Settings page or in .env")
        return LLMFactory.create(llm_settings)

    def _messages(self, question: str, citations: list[Citation]) -> list[dict[str, str]]:
        blocks = []
        for citation in citations:
            header = f"[{citation.id}] source: {citation.source}"
            if citation.page is not None:
                header += f", page: {citation.page}"
            blocks.append(f"{header}\n{self._truncate(citation.text)}")
        context = "\n\n".join(blocks)
        user = f"Context:\n{context}\n\nQuestion: {question.strip()}"
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    def _truncate(self, text: str) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= self.max_chunk_chars:
            return normalized
        return f"{normalized[: self.max_chunk_chars - 3]}..."

    def _record(self, trace: Any | None, llm: BaseLLM, duration_ms: float, citations: list[Citation], answer: str) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(
                "generation",
                {
                    "provider": llm.settings.provider,
                    "model": llm.settings.model,
                    "citations": len(citations),
                    "answer_chars": len(answer),
                },
                duration_ms=duration_ms,
            )
