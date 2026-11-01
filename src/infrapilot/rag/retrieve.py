"""Hybrid retrieval: pgvector cosine similarity plus Postgres full-text search, merged with
reciprocal rank fusion (RRF). No reranking model is involved.

Vector search finds paraphrases but can miss exact terms (``ip-`` prefix, ``1024``); full-text
search finds exact terms but misses paraphrases. RRF combines the two ranked lists using only
ranks, so the incompatible score scales (cosine distance vs ts_rank) never need calibrating.
"""

import re
from collections.abc import Sequence
from typing import Literal, Protocol

import psycopg
from langchain_core.embeddings import Embeddings
from pgvector.psycopg import register_vector

from infrapilot.state import Chunk

Mode = Literal["vector", "fulltext", "hybrid"]
RRF_K = 60  # standard constant from the original RRF paper
CANDIDATES = 10  # per-retriever depth fed to the fusion

_WORD = re.compile(r"[A-Za-z0-9]+")


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> list[Chunk]: ...


def rrf_merge(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Ties are broken by the best (lowest) rank in any list, then by id, so output is deterministic.
    """
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    return sorted(scores, key=lambda d: (-scores[d], best[d], d))


def or_tsquery(query: str) -> str:
    """'what is the limit?' -> 'what | is | the | limit'. OR semantics, safe for to_tsquery.

    plainto_tsquery ANDs every word, which returns nothing for a natural-language question.
    Stop words are dropped by the 'english' configuration inside Postgres.
    """
    return " | ".join(dict.fromkeys(w.lower() for w in _WORD.findall(query)))


class PgVectorRetriever:
    def __init__(self, dsn: str, embeddings: Embeddings, mode: Mode = "hybrid") -> None:
        self._dsn = dsn
        self._embeddings = embeddings
        self.mode: Mode = mode

    def _vector(self, conn: psycopg.Connection, query: str, n: int) -> list[Chunk]:
        register_vector(conn)
        vector = self._embeddings.embed_query(query)
        rows = conn.execute(
            "SELECT id, source, heading_path, text FROM chunks "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (vector, n),
        ).fetchall()
        return [Chunk(id=r[0], source=r[1], heading_path=r[2], text=r[3]) for r in rows]

    def _fulltext(self, conn: psycopg.Connection, query: str, n: int) -> list[Chunk]:
        tsquery = or_tsquery(query)
        if not tsquery:
            return []
        rows = conn.execute(
            "SELECT id, source, heading_path, text FROM chunks, "
            "to_tsquery('english', %s) q WHERE tsv @@ q "
            "ORDER BY ts_rank_cd(tsv, q) DESC, id LIMIT %s",
            (tsquery, n),
        ).fetchall()
        return [Chunk(id=r[0], source=r[1], heading_path=r[2], text=r[3]) for r in rows]

    def retrieve(self, query: str, k: int = 4, mode: Mode | None = None) -> list[Chunk]:
        mode = mode or self.mode
        depth = max(k, CANDIDATES)
        with psycopg.connect(self._dsn) as conn:
            vec = self._vector(conn, query, depth) if mode in ("vector", "hybrid") else []
            fts = self._fulltext(conn, query, depth) if mode in ("fulltext", "hybrid") else []
        if mode == "vector":
            return vec[:k]
        if mode == "fulltext":
            return fts[:k]
        by_id = {c.id: c for c in [*vec, *fts]}
        return [by_id[i] for i in rrf_merge([[c.id for c in vec], [c.id for c in fts]])[:k]]
