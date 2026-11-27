"""All tunable settings. Values come from environment variables (prefix ROUTER_) or .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["gemini", "groq", "ollama", "openai_compat"]
Policy = Literal["target", "max_accuracy"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTER_", env_file=".env", extra="ignore")

    # --- models ---------------------------------------------------------
    cheap_provider: Provider = "gemini"
    cheap_model: str = "gemini-3.5-flash-lite"
    expensive_provider: Provider = "gemini"
    expensive_model: str = "gemini-3.5-flash"

    # Documents per request. Allowed by the rules; each document is still charged.
    cheap_batch_size: int = 10
    expensive_batch_size: int = 5
    cheap_concurrency: int = 8
    expensive_concurrency: int = 4
    cheap_rpm: int = 15  # free-tier request quotas; 0 = unlimited
    expensive_rpm: int = 10
    request_timeout_s: float = 120.0
    max_retries: int = 4

    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openai_base_url: str = ""
    openai_api_key: str = ""
    ollama_url: str = "http://localhost:11434"

    # --- cost / budget / time (from the task rules) -----------------------
    cost_cheap: float = 1.0
    cost_expensive: float = 20.0
    budget: float = 2100.0
    time_limit_s: float = 600.0

    # --- routing ----------------------------------------------------------
    target_accuracy: float = 0.92
    target_margin: float = 0.02
    policy: Policy = "max_accuracy"
    calibration_size: int = 20
    calibration_seed: int = 13
    expensive_accuracy_prior: float = 0.97
    calibration_prior_strength: float = 4.0

    # --- paths ------------------------------------------------------------
    documents_path: Path = Path("provided/candidate/documents.jsonl")
    output_dir: Path = Path("output")
    cache_path: Path = Path(".cache/llm_cache.sqlite")
    use_cache: bool = True
    max_doc_chars: int = 6000
