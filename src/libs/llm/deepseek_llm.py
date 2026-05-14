from __future__ import annotations

from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    default_base_url = "https://api.deepseek.com/v1"


LLMFactory.register_provider("deepseek", DeepSeekLLM)
