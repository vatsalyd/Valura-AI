"""
Centralised configuration — loaded once from .env at startup.

WHY a dedicated config module:
- Single source of truth for all env vars
- Pydantic validation catches misconfiguration at boot, not mid-request
- Easy to override in tests via monkeypatch
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings sourced from environment / .env file."""

    # --- LLM ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # --- App ---
    app_env: str = Field(default="development", alias="APP_ENV")

    # --- Timeouts ---
    pipeline_timeout_seconds: float = Field(default=30.0)
    llm_timeout_seconds: float = Field(default=15.0)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
