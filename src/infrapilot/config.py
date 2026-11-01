"""Runtime settings, read from environment variables (or a local ``.env`` file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 4096
    database_url: str = "postgresql://infrapilot:infrapilot@127.0.0.1:5433/infrapilot"

    # Self-hosted Langfuse tracing; off unless explicitly enabled.
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://127.0.0.1:3100"

    # Optional LLM-judge for eval answer quality; unset means the judge is disabled.
    judge_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
