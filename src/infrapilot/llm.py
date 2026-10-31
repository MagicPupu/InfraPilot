"""Thin factory around Ollama. No other provider is supported by design."""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from infrapilot.config import Settings, get_settings


def get_chat_model(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> ChatOllama:
    s = settings or get_settings()
    return ChatOllama(
        model=model or s.llm_model,
        base_url=s.ollama_base_url,
        num_ctx=s.num_ctx,
        temperature=temperature,
        # Qwen3 "thinking" tokens break schema-constrained output and waste latency.
        reasoning=False,
        # Keep the model resident between graph steps (6 GB GPU: avoid reload churn).
        keep_alive="30m",
    )


def get_embeddings(settings: Settings | None = None) -> OllamaEmbeddings:
    s = settings or get_settings()
    return OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url, keep_alive=1800)
