"""Model client factory."""

from __future__ import annotations

from ..cache import ResponseCache
from ..config import Settings
from .base import ModelCall, ModelClient
from .http_clients import GeminiClient, GroqClient, OllamaClient, OpenAICompatClient

__all__ = ["ModelCall", "ModelClient", "build_client"]


def build_client(tier: str, s: Settings, cache: ResponseCache | None) -> ModelClient:
    cheap = tier == "cheap"
    provider = s.cheap_provider if cheap else s.expensive_provider
    model = s.cheap_model if cheap else s.expensive_model
    common = dict(
        concurrency=s.cheap_concurrency if cheap else s.expensive_concurrency,
        rpm=s.cheap_rpm if cheap else s.expensive_rpm,
        timeout_s=s.request_timeout_s,
        max_retries=s.max_retries,
        cache=cache,
    )
    match provider:
        case "gemini":
            return GeminiClient(model, tier, base_url=s.gemini_base_url,
                                api_key=s.gemini_api_key, **common)
        case "groq":
            return GroqClient(model, tier, base_url=s.groq_base_url, api_key=s.groq_api_key,
                              **common)
        case "openai_compat":
            return OpenAICompatClient(model, tier, base_url=s.openai_base_url,
                                      api_key=s.openai_api_key, **common)
        case "ollama":
            return OllamaClient(model, tier, url=s.ollama_url, **common)
    raise ValueError(f"unknown provider {provider!r}")
