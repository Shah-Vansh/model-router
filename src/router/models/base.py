"""Common model interface plus retry, rate-limit and cache handling."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..cache import ResponseCache


class ModelCall(BaseModel):
    tier: str
    model: str
    raw: str
    latency_s: float
    cached: bool = False
    error: str | None = None


class RetryableError(Exception):
    """Rate limit, timeout or server error: worth trying again."""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, RetryableError | httpx.TimeoutException | httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 409, 429) or exc.response.status_code >= 500
    return False


class RateLimiter:
    """Spaces requests so we stay under a requests-per-minute quota."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class ModelClient(ABC):
    """One LLM endpoint. Subclasses only implement `_generate`."""

    provider: str = "base"

    def __init__(
        self,
        model: str,
        tier: str,
        *,
        concurrency: int,
        rpm: int,
        timeout_s: float,
        max_retries: int,
        cache: ResponseCache | None,
    ):
        self.model = model
        self.tier = tier
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache = cache
        self._sem = asyncio.Semaphore(concurrency)
        self._limiter = RateLimiter(rpm)

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    async def _generate(self, system: str, prompt: str) -> str:
        """Return the raw text reply."""

    async def generate(self, system: str, prompt: str) -> ModelCall:
        key = None
        if self.cache is not None:
            key = self.cache.key(self.model_id, system, prompt)
            hit = self.cache.get(key)
            if hit is not None:
                return ModelCall(
                    tier=self.tier, model=self.model_id, raw=hit, latency_s=0.0, cached=True
                )

        start = time.perf_counter()
        try:
            async with self._sem:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential_jitter(initial=1, max=30),
                    retry=retry_if_exception(_retryable),
                    reraise=True,
                ):
                    with attempt:
                        await self._limiter.wait()
                        raw = await asyncio.wait_for(
                            self._generate(system, prompt), timeout=self.timeout_s
                        )
        except Exception as e:  # noqa: BLE001 - any failure becomes a routed outcome
            return ModelCall(
                tier=self.tier,
                model=self.model_id,
                raw="",
                latency_s=time.perf_counter() - start,
                error=f"{type(e).__name__}: {e}"[:300],
            )

        if self.cache is not None and key is not None:
            self.cache.put(key, raw)
        return ModelCall(
            tier=self.tier, model=self.model_id, raw=raw, latency_s=time.perf_counter() - start
        )

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass
