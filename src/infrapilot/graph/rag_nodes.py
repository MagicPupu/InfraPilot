from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from infrapilot.rag.retrieve import Retriever
from infrapilot.state import (
    Chunk,
    CitedAnswer,
    InfraState,
    RelevanceGrade,
    RewrittenQuery,
)
from infrapilot.structured import StructuredOutputError, invoke_structured

INSUFFICIENT = "Insufficient evidence in the runbooks to answer this question."

GRADE_SYSTEM = """Decide whether the documents contain information that helps answer the question.
Reply with JSON: {"relevant": true | false}.
Example: question "What is the memory limit?" with a document about memory limits -> {"relevant": true}
Example: question "What is the memory limit?" with a document about log rotation -> {"relevant": false}"""

REWRITE_SYSTEM = """Rewrite the question as a short keyword-style search query for a runbook index.
Reply with JSON: {"query": str}.
Example: "can I run bigger boxes?" -> {"query": "container memory cpu limits maximum"}"""

ANSWER_SYSTEM = """Answer the question using ONLY the numbered documents below.
Cite the id of every document you rely on. If they do not answer the question, say so.
Reply with JSON: {"answer": str, "cited_ids": [str]}.
Example: {"answer": "Containers are limited to 1024 MB.", "cited_ids": ["a1b2c3d4e5"]}"""


def format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] ({c.source} > {c.heading_path})\n{c.text}" for c in chunks)


def invalid_citations(cited_ids: list[str], retrieved_ids: set[str]) -> list[str]:
    """Deterministic faithfulness check: cited ids must exist and be non-empty."""
    if not cited_ids:
        return ["<no citation>"]
    return [i for i in cited_ids if i not in retrieved_ids]


def make_retrieve(retriever: Retriever) -> Callable[[InfraState], dict[str, Any]]:
    def retrieve(state: InfraState) -> dict[str, Any]:
        query = state.get("search_query") or state["request"]
        chunks = retriever.retrieve(query)
        return {"search_query": query, "retrieved": [c.model_dump() for c in chunks]}

    return retrieve


def make_grade(llm: BaseChatModel) -> Callable[[InfraState], dict[str, Any]]:
    def grade_docs(state: InfraState) -> dict[str, Any]:
        chunks = [Chunk.model_validate(c) for c in state.get("retrieved", [])]
        if not chunks:
            return {"relevant": False}
        messages = [
            SystemMessage(content=GRADE_SYSTEM),
            HumanMessage(
                content=f"Question: {state['request']}\n\nDocuments:\n{format_chunks(chunks)}"
            ),
        ]
        try:
            return {"relevant": invoke_structured(llm, messages, RelevanceGrade).relevant}
        except StructuredOutputError as exc:
            return {"error": str(exc)}

    return grade_docs


def make_rewrite(llm: BaseChatModel) -> Callable[[InfraState], dict[str, Any]]:
    def rewrite_query(state: InfraState) -> dict[str, Any]:
        messages = [SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=state["request"])]
        try:
            query = invoke_structured(llm, messages, RewrittenQuery).query
        except StructuredOutputError as exc:
            return {"error": str(exc)}
        return {"search_query": query, "rewrites": state.get("rewrites", 0) + 1}

    return rewrite_query


def make_answer(llm: BaseChatModel) -> Callable[[InfraState], dict[str, Any]]:
    def answer_with_citations(state: InfraState) -> dict[str, Any]:
        chunks = [Chunk.model_validate(c) for c in state.get("retrieved", [])]
        if not state.get("relevant") or not chunks:
            return {"answer": INSUFFICIENT, "cited_ids": [], "final": INSUFFICIENT}
        by_id = {c.id: c for c in chunks}
        messages: list[BaseMessage] = [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=f"Question: {state['request']}\n\nDocuments:\n{format_chunks(chunks)}"
            ),
        ]
        for _ in range(2):  # one regeneration allowed
            try:
                out = invoke_structured(llm, messages, CitedAnswer)
            except StructuredOutputError as exc:
                return {"error": str(exc)}
            bad = invalid_citations(out.cited_ids, set(by_id))
            if not bad:
                sources = "\n".join(
                    f"- [{i}] {by_id[i].source} > {by_id[i].heading_path}" for i in out.cited_ids
                )
                return {
                    "answer": out.answer,
                    "cited_ids": out.cited_ids,
                    "final": f"{out.answer}\n\nSources:\n{sources}",
                }
            messages.append(
                HumanMessage(
                    content=f"Invalid or missing citations: {bad}. Cite only ids from the documents."
                )
            )
        return {"answer": INSUFFICIENT, "cited_ids": [], "final": INSUFFICIENT}

    return answer_with_citations
