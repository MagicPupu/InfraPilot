import asyncio
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from infrapilot.graph.build import Deps, build_graph
from infrapilot.state import Chunk
from tests.fakes import FakeInvoker, FakeRetriever, scripted_llm

CHUNK = Chunk(id="c1", source="limits.md", heading_path="Limits", text="Max memory is 1024 MB.")
CREATE = {
    "rationale": "Create web.",
    "calls": [
        {
            "tool": "create_container",
            "args": {"image": "nginx:1.27", "name": "ip-web", "mem_mb": 256, "cpus": 0.5},
        }
    ],
}


def run(
    llm: Any, request: str, chunks: list[Chunk] | None = None, invoker: FakeInvoker | None = None
):
    invoker = invoker or FakeInvoker()
    graph = build_graph(Deps(llm, FakeRetriever(chunks or [CHUNK]), invoker), InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    result = asyncio.run(graph.ainvoke({"request": request}, cfg))
    return graph, cfg, result, invoker


def resume(graph: Any, cfg: Any, approved: bool) -> dict[str, Any]:
    return asyncio.run(graph.ainvoke(Command(resume={"approved": approved}), cfg))  # type: ignore[no-any-return]


# --- question path ---------------------------------------------------------


def test_question_with_valid_citation():
    llm = scripted_llm(
        {"route": "question"}, {"relevant": True}, {"answer": "1024 MB.", "cited_ids": ["c1"]}
    )
    _, _, result, _ = run(llm, "What is the max memory?")
    assert result["cited_ids"] == ["c1"] and "limits.md" in result["final"]


def test_hallucinated_citation_regenerates_then_refuses():
    bad = {"answer": "x", "cited_ids": ["zzz"]}
    llm = scripted_llm({"route": "question"}, {"relevant": True}, bad, bad)
    _, _, result, _ = run(llm, "What is the max memory?")
    assert result["final"].startswith("Insufficient evidence") and result["cited_ids"] == []


def test_hallucinated_citation_recovers_on_regeneration():
    llm = scripted_llm(
        {"route": "question"},
        {"relevant": True},
        {"answer": "x", "cited_ids": ["zzz"]},
        {"answer": "1024 MB.", "cited_ids": ["c1"]},
    )
    _, _, result, _ = run(llm, "What is the max memory?")
    assert result["cited_ids"] == ["c1"]


def test_irrelevant_docs_rewrite_once_then_insufficient():
    llm = scripted_llm(
        {"route": "question"}, {"relevant": False}, {"query": "memory limits"}, {"relevant": False}
    )
    _, _, result, _ = run(llm, "Tell me about unicorns")
    assert result["rewrites"] == 1 and result["final"].startswith("Insufficient evidence")


# --- action path -----------------------------------------------------------


def test_action_interrupts_then_executes_after_approval():
    llm = scripted_llm({"route": "action"}, CREATE)
    graph, cfg, result, invoker = run(llm, "start nginx ip-web")
    assert "__interrupt__" in result and not invoker.calls
    assert "create_container" in result["__interrupt__"][0].value["preview"]
    final = resume(graph, cfg, approved=True)
    assert "ip-web" in invoker.containers and final["verification"]["ok"]


def test_rejection_executes_nothing():
    llm = scripted_llm({"route": "action"}, CREATE)
    graph, cfg, _, invoker = run(llm, "start nginx ip-web")
    final = resume(graph, cfg, approved=False)
    assert not invoker.calls and "rejected" in final["final"]


def test_policy_refusal_never_interrupts_or_executes():
    plan = {"rationale": "x", "calls": [{"tool": "drop_database", "args": {}}]}
    _, _, result, invoker = run(scripted_llm({"route": "action"}, plan), "wipe it")
    assert "__interrupt__" not in result and not invoker.calls
    assert "refused by policy" in result["final"]


def test_read_only_plan_skips_approval():
    plan = {"rationale": "list", "calls": [{"tool": "list_containers", "args": {}}]}
    _, _, result, invoker = run(scripted_llm({"route": "action"}, plan), "what is running?")
    assert "__interrupt__" not in result and result["verification"]["ok"]
    assert all(t == "list_containers" for t, _ in invoker.calls)


# --- failure handling ------------------------------------------------------


def test_invalid_router_output_routes_to_explain_failure():
    _, _, result, invoker = run(scripted_llm("garbage", "still garbage"), "hello")
    assert "Nothing was executed" in result["final"] and not invoker.calls
