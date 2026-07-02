from __future__ import annotations

import time
from dataclasses import dataclass

from core.settings import LLMSettings
from libs.llm.llm_factory import LLMFactory


KEY_REQUIRED_PROVIDERS = {"openai", "anthropic"}


@dataclass(frozen=True)
class LLMTestResult:
    ok: bool
    summary: str
    detail: str = ""
    latency_ms: float = 0.0


def test_llm_connection(
    *,
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    timeout: float = 20.0,
) -> LLMTestResult:
    provider_key = (provider or "").strip().lower()
    if not provider_key:
        return LLMTestResult(False, "No provider selected.")
    if not model.strip():
        return LLMTestResult(False, "Model is empty.", "Enter a model name, then test again.")
    if provider_key in KEY_REQUIRED_PROVIDERS and not api_key:
        return LLMTestResult(
            False,
            f"The {provider_key} provider needs an API key, and none is configured.",
            "Paste a key into the API Key field, or set it in .env, then test again.",
        )
    settings = LLMSettings(
        provider=provider_key,
        model=model.strip(),
        api_key=api_key,
        base_url=base_url.strip(),
    )
    started = time.perf_counter()
    try:
        client = LLMFactory.create(settings)
        if hasattr(client, "timeout"):
            client.timeout = timeout
        reply = client.chat([{"role": "user", "content": "Reply with the single word: ok"}])
    except Exception as error:
        summary = _friendly_failure(provider_key, error)
        return LLMTestResult(False, summary, str(error))
    latency_ms = (time.perf_counter() - started) * 1000
    return LLMTestResult(
        True,
        f"Connection successful — `{model.strip()}` replied in {latency_ms:.0f} ms through the project's LLM client.",
        f"The exact code path used by query answering works with this configuration. Reply: {reply[:80]!r}",
        latency_ms,
    )


def _friendly_failure(provider: str, error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if "http error: 401" in text or "http error: 403" in text:
        return "Authentication failed — the API key was rejected. Check that the key is correct and active."
    if "http error: 402" in text:
        return "The provider reports insufficient balance. Top up the account, then test again."
    if "http error: 404" in text:
        return "Endpoint or model not found — check Base URL and Model."
    if "http error: 429" in text:
        return "Rate limit reached — wait a moment and test again."
    if "connection error" in lowered:
        return "Cannot reach the server — check Base URL and your network connection."
    if "timed out" in lowered or "timeout" in lowered:
        return "The request timed out — the server is unreachable or too slow to respond."
    if "response error" in lowered:
        return "The server responded with an unexpected payload."
    if "unsupported" in lowered:
        return text
    return f"Connection test failed: {text}"
