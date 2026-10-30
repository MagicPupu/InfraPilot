"""Optional Langfuse tracing (self-hosted). Disabled unless LANGFUSE_ENABLED=true."""

import os
from typing import Any

from infrapilot.config import Settings


def get_callbacks(settings: Settings) -> list[Any]:
    if not settings.langfuse_enabled:
        return []
    # The Langfuse SDK reads its credentials from the process environment.
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]
