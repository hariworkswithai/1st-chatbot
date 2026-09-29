"""Central configuration. All secrets come from environment variables.

The backend owns the OpenRouter API key. The browser never sees it.
Copy `backend/.env.example` to `backend/.env` to configure locally.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider (OpenRouter-compatible) ---
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_fallback_model: str = "meta-llama/llama-3.1-8b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Reliability controls ---
    openrouter_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0  # exponential: 1s, 2s, 4s, ...

    # --- Context / token controls ---
    max_history_messages: int = 20  # max prior messages sent to the model
    max_input_chars: int = 4000  # per-message input guard
    max_output_tokens: int = 1024  # `max_tokens` sent to the provider

    # --- App-level rate limiting (per client IP, sliding window) ---
    app_rate_limit_per_minute: int = 30

    # --- Misc ---
    log_level: str = "INFO"
    # Comma-separated list, e.g. "http://localhost:5173,http://127.0.0.1:5173"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_api_key(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
