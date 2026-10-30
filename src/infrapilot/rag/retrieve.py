from typing import Protocol

import psycopg
from langchain_core.embeddings import Embeddings
from pgvector.psycopg import register_vector

from infrapilot.state import Chunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> list[Chunk]: ...


def rerank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Reranking hook. TODO: plug a cross-encoder; identity for now."""
    return chunks


class PgVectorRetriever:
    def __init__(self, dsn: str, embeddings: Embeddings) -> None:
        self._dsn = dsn
        self._embeddings = embeddings

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        vector = self._embeddings.embed_query(query)
        with psycopg.connect(self._dsn) as conn:
            register_vector(conn)
            rows = conn.execute(
                "SELECT id, source, heading_path, text FROM chunks "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, k),
            ).fetchall()
        return rerank(
            query, [Chunk(id=r[0], source=r[1], heading_path=r[2], text=r[3]) for r in rows]
        )
