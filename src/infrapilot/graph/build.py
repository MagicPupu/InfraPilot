from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from infrapilot.graph.approval import approval
from infrapilot.graph.executor import make_executor
from infrapilot.graph.failure import explain_failure, explain_refusal, rejected
from infrapilot.graph.planner import make_planner
from infrapilot.graph.policy import policy_guard
from infrapilot.graph.rag_nodes import (
    make_answer,
    make_grade,
    make_retrieve,
    make_rewrite,
)
from infrapilot.graph.router import make_router
from infrapilot.graph.verify import make_verify
from infrapilot.mcp_client import ToolInvoker
from infrapilot.rag.retrieve import Retriever
from infrapilot.state import InfraState

MAX_REWRITES = 1


@dataclass
class Deps:
    llm: BaseChatModel
    retriever: Retriever
    invoker: ToolInvoker


def _or_fail(mapping: Callable[[InfraState], str]) -> Callable[[InfraState], str]:
    """Any node that records an error sends the run to the safe failure node."""

    def edge(state: InfraState) -> str:
        return "explain_failure" if state.get("error") else mapping(state)

    return edge


def _after_grade(state: InfraState) -> str:
    if state.get("relevant"):
        return "answer_with_citations"
    return "rewrite_query" if state.get("rewrites", 0) < MAX_REWRITES else "answer_with_citations"


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
    g = StateGraph(InfraState)
    # LangGraph's add_node overloads do not accept a total=False TypedDict state.
    add_node: Any = g.add_node
    add_node("router", make_router(deps.llm))
    add_node("retrieve", make_retrieve(deps.retriever))
    add_node("grade_docs", make_grade(deps.llm))
    add_node("rewrite_query", make_rewrite(deps.llm))
    add_node("answer_with_citations", make_answer(deps.llm))
    add_node("planner", make_planner(deps.llm))
    add_node("policy_guard", policy_guard)
    add_node("approval", approval)
    add_node("executor", make_executor(deps.invoker))
    add_node("verify", make_verify(deps.invoker))
    add_node("explain_failure", explain_failure)
    add_node("explain_refusal", explain_refusal)
    add_node("rejected", rejected)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router", _or_fail(lambda s: "retrieve" if s["route"] == "question" else "planner")
    )
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", _or_fail(_after_grade))
    g.add_edge("rewrite_query", "retrieve")
    g.add_conditional_edges("answer_with_citations", _or_fail(lambda s: END))
    g.add_conditional_edges("planner", _or_fail(lambda s: "policy_guard"))
    g.add_conditional_edges(
        "policy_guard",
        lambda s: {
            "refuse": "explain_refusal",
            "approve": "executor",
            "needs_approval": "approval",
        }[s["policy"]["outcome"]],
    )
    g.add_conditional_edges(
        "approval", lambda s: "executor" if s["approval"]["approved"] else "rejected"
    )
    g.add_conditional_edges("executor", _or_fail(lambda s: "verify"))
    for terminal in ("verify", "explain_failure", "explain_refusal", "rejected"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer)
