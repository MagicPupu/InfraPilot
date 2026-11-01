"""Real Postgres, fake embeddings: checkpointer resume and the three retrieval modes."""

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from infrapilot.graph.build import Deps, build_graph
from infrapilot.rag.ingest import ingest_directory, load_chunks
from infrapilot.rag.retrieve import PgVectorRetriever
from tests.fakes import FakeInvoker, FakeRetriever, scripted_llm
from tests.integration.test_graph_flow import CHUNK, CREATE

pytestmark = pytest.mark.postgres
RUNBOOKS = Path(__file__).parents[2] / "docs" / "runbooks"


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors (768 dims, like nomic-embed-text). No model needed."""

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * 768
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 768] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def test_interrupted_run_resumes_from_postgres_after_rebuild(pg_dsn: str):
    cfg: Any = {"configurable": {"thread_id": "pg-resume"}}

    async def first_process() -> None:
        async with AsyncPostgresSaver.from_conn_string(pg_dsn) as saver:
            await saver.setup()
            graph = build_graph(
                Deps(
                    scripted_llm({"route": "action"}, CREATE), FakeRetriever([CHUNK]), FakeInvoker()
                ),
                saver,
            )
            result = await graph.ainvoke({"request": "start nginx"}, cfg)
            assert "__interrupt__" in result

    async def second_process() -> tuple[dict[str, Any], FakeInvoker]:
        invoker = FakeInvoker()
        async with AsyncPostgresSaver.from_conn_string(pg_dsn) as saver:  # new connection
            graph = build_graph(Deps(scripted_llm(), FakeRetriever([]), invoker), saver)
            return await graph.ainvoke(Command(resume={"approved": True}), cfg), invoker

    asyncio.run(first_process())
    final, invoker = asyncio.run(second_process())
    assert final["verification"]["ok"] and "ip-web" in invoker.containers


def test_retrieval_modes_against_real_postgres(pg_dsn: str):
    embeddings = HashEmbeddings()
    count = ingest_directory(RUNBOOKS, pg_dsn, embeddings)
    assert count == len(load_chunks(RUNBOOKS))
    retriever = PgVectorRetriever(pg_dsn, embeddings)
    corpus = {c.id: c for c in load_chunks(RUNBOOKS)}

    query = "container names must start with the ip- prefix"
    naming = next(i for i, c in corpus.items() if c.heading_path.endswith("Container names"))
    for mode in ("vector", "fulltext", "hybrid"):
        got = retriever.retrieve(query, k=3, mode=mode)
        assert 0 < len(got) <= 3 and all(c.id in corpus for c in got), mode
    assert naming in [c.id for c in retriever.retrieve(query, k=3, mode="fulltext")]
    assert naming in [c.id for c in retriever.retrieve(query, k=3, mode="hybrid")]
    assert retriever.retrieve("zzzqqq xxyyzz", k=3, mode="fulltext") == []  # no lexical match


def test_ingest_is_idempotent(pg_dsn: str):
    embeddings = HashEmbeddings()
    ingest_directory(RUNBOOKS, pg_dsn, embeddings)
    ingest_directory(RUNBOOKS, pg_dsn, embeddings)
    retriever = PgVectorRetriever(pg_dsn, embeddings)
    ids = [c.id for c in retriever.retrieve("memory", k=10, mode="hybrid")]
    assert len(ids) == len(set(ids))
