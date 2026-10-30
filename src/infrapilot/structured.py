"""Schema-constrained generation with validation and a single retry.

Small local models are unreliable at native tool calling, so every structured
step (router, planner, grader) asks Ollama for JSON constrained by a Pydantic
schema (``format=<json schema>``) and validates the result here. On failure we
retry once with the validation error appended; a second failure raises
``StructuredOutputError`` so the graph can route to ``explain_failure`` and
never execute anything.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError


class StructuredOutputError(Exception):
    """The model failed to produce schema-valid output after one retry."""

    def __init__(self, schema: type[BaseModel], last_error: str, last_output: str):
        super().__init__(f"{schema.__name__}: invalid output after retry: {last_error}")
        self.schema = schema
        self.last_error = last_error
        self.last_output = last_output


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else "".join(map(str, content))


def invoke_structured[T: BaseModel](
    llm: BaseChatModel, messages: list[BaseMessage], schema: type[T]
) -> T:
    constrained = llm.bind(format=schema.model_json_schema())
    convo = list(messages)
    error = ""
    raw = ""
    for attempt in range(2):
        raw = _text(constrained.invoke(convo))
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            error = str(exc)
            if attempt == 0:
                convo += [
                    AIMessage(content=raw),
                    HumanMessage(
                        content=(
                            "Your previous output was invalid:\n"
                            f"{error}\n"
                            "Return only corrected JSON that matches the schema."
                        )
                    ),
                ]
    raise StructuredOutputError(schema, error, raw)
