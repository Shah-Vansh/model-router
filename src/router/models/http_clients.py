"""Real LLM providers: any OpenAI-compatible API (Groq, OpenRouter, Gemini ...) and Ollama."""

from __future__ import annotations

import httpx

from .base import ModelClient, RetryableError


class OpenAICompatClient(ModelClient):
    provider = "openai_compat"

    max_tokens = 8000

    def __init__(self, model: str, tier: str, *, base_url: str, api_key: str, **kw):
        super().__init__(model, tier, **kw)
        if not base_url or not api_key:
            raise ValueError(f"{self.provider} needs a base URL and an API key (see .env.example)")
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout_s,
        )

    async def _generate(self, system: str, prompt: str) -> str:
        r = await self._http.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        if r.status_code == 429:
            raise RetryableError(f"rate limited: {r.text[:200]}")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    async def aclose(self) -> None:
        await self._http.aclose()


class GroqClient(OpenAICompatClient):
    provider = "groq"


class GeminiClient(OpenAICompatClient):
    """Google Gemini through its OpenAI-compatible endpoint (free tier via AI Studio key)."""

    provider = "gemini"


class OllamaClient(ModelClient):
    provider = "ollama"

    def __init__(self, model: str, tier: str, *, url: str, **kw):
        super().__init__(model, tier, **kw)
        self._http = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=self.timeout_s)

    async def _generate(self, system: str, prompt: str) -> str:
        r = await self._http.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"] or ""

    async def aclose(self) -> None:
        await self._http.aclose()
