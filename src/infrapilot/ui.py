"""Streamlit front-end: same graph as the CLI, with a button-based approval step.

Each interaction opens its own checkpointer and sandbox connection; the graph
state lives in Postgres under ``thread_id``, so an approval click can resume a
run started by an earlier script execution.
"""

import asyncio
import uuid
from typing import Any

import streamlit as st
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from infrapilot.config import get_settings
from infrapilot.graph.build import Deps, build_graph
from infrapilot.llm import get_chat_model, get_embeddings
from infrapilot.mcp_client import open_sandbox_invoker
from infrapilot.observability import get_callbacks
from infrapilot.rag.retrieve import PgVectorRetriever


async def _run(graph_input: Any, thread_id: str) -> dict[str, Any]:
    s = get_settings()
    async with AsyncPostgresSaver.from_conn_string(s.database_url) as saver:
        await saver.setup()
        async with open_sandbox_invoker() as invoker:
            deps = Deps(
                get_chat_model(s), PgVectorRetriever(s.database_url, get_embeddings(s)), invoker
            )
            graph = build_graph(deps, saver)
            config: Any = {
                "configurable": {"thread_id": thread_id},
                "callbacks": get_callbacks(s),
            }
            return dict(await graph.ainvoke(graph_input, config))


def _record(result: dict[str, Any]) -> None:
    """Store the outcome of a run: either a pending approval or a final answer."""
    if "__interrupt__" in result:
        st.session_state.pending = result["__interrupt__"][0].value
        return
    st.session_state.pending = None
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("final", "(no answer)"),
            "details": {
                k: result[k] for k in ("route", "plan", "policy", "results") if k in result
            },
        }
    )


def main() -> None:
    st.set_page_config(page_title="InfraPilot", page_icon="🛠️")
    st.title("InfraPilot")
    st.caption("Local, policy-gated infrastructure copilot.")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))

    with st.sidebar:
        s = get_settings()
        st.write(f"**Model:** `{s.llm_model}`")
        if st.button("New conversation"):
            for key in ("messages", "pending", "thread_id"):
                del st.session_state[key]
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("details"):
                with st.expander("Details"):
                    st.json(msg["details"])

    pending = st.session_state.pending
    if pending:
        with st.chat_message("assistant"):
            st.warning("Approval required")
            st.code(pending["preview"], language="text")
            col_ok, col_no = st.columns(2)
            decision = None
            if col_ok.button("Approve", type="primary"):
                decision = True
            if col_no.button("Reject"):
                decision = False
            if decision is not None:
                with st.spinner("Resuming..."):
                    result = asyncio.run(
                        _run(Command(resume={"approved": decision}), st.session_state.thread_id)
                    )
                _record(result)
                st.rerun()
        return

    if request := st.chat_input("Ask a question or request a change"):
        # Fresh thread per request so earlier runs never leak into this one.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages.append({"role": "user", "content": request})
        with st.spinner("Thinking..."):
            result = asyncio.run(_run({"request": request}, st.session_state.thread_id))
        _record(result)
        st.rerun()


main()
