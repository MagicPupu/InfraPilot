import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from infrapilot.structured import StructuredOutputError, invoke_structured


class Route(BaseModel):
    intent: str
    confidence: float


def fake(*outputs: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter(AIMessage(content=o) for o in outputs))


MSG = [HumanMessage(content="restart nginx")]


def test_valid_first_try():
    out = invoke_structured(fake('{"intent": "act", "confidence": 0.9}'), MSG, Route)
    assert out == Route(intent="act", confidence=0.9)


def test_retry_recovers():
    llm = fake("not json", '{"intent": "act", "confidence": 0.5}')
    assert invoke_structured(llm, MSG, Route).intent == "act"


def test_second_failure_raises():
    with pytest.raises(StructuredOutputError) as exc:
        invoke_structured(fake("nope", '{"intent": 1}'), MSG, Route)
    assert exc.value.schema is Route
