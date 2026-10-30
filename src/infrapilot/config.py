"""Runtime settings, read from environment variables (or a local ``.env`` file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 8192
    # Optional LLM-judge for eval answer quality; unset means the judge is disabled.
    judge_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
