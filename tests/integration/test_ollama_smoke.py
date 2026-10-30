import urllib.request

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from infrapilot.config import get_settings
from infrapilot.llm import get_chat_model
from infrapilot.structured import invoke_structured


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(get_settings().ollama_base_url, timeout=1)
        return True
    except OSError:
        return False


class Pong(BaseModel):
    reply: str


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_up(), reason="Ollama is not reachable")
def test_structured_roundtrip_with_real_model():
    out = invoke_structured(
        get_chat_model(),
        [HumanMessage(content='Reply with JSON {"reply": "pong"}')],
        Pong,
    )
    assert out.reply
