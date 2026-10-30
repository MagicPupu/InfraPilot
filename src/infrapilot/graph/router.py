from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from infrapilot.state import InfraState, RouteDecision
from infrapilot.structured import StructuredOutputError, invoke_structured

SYSTEM = """Classify the user's message for an infrastructure assistant.
- "question": asks for information, policies, conventions or how-to (answered from docs).
- "action": asks to create, scale, delete or list containers (changes or inspects the sandbox).
Reply with JSON: {"route": "question" | "action"}.

Examples:
"What is our naming convention for containers?" -> {"route": "question"}
"Create an nginx container called ip-web with 256 MB" -> {"route": "action"}
"How many replicas are we allowed to run?" -> {"route": "question"}
"Delete ip-cache" -> {"route": "action"}"""


def make_router(llm: BaseChatModel) -> Callable[[InfraState], dict[str, Any]]:
    def router(state: InfraState) -> dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM), HumanMessage(content=state["request"])]
        try:
            return {"route": invoke_structured(llm, messages, RouteDecision).route}
        except StructuredOutputError as exc:
            return {"error": str(exc)}

    return router
