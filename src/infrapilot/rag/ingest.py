"""Markdown ingestion: heading-aware chunking, Ollama embeddings, pgvector upsert."""

import hashlib
import re
from importlib import resources
from pathlib import Path

import numpy as np
import psycopg
from langchain_core.embeddings import Embeddings
from pgvector.psycopg import register_vector

from infrapilot.state import Chunk

MAX_CHUNK_CHARS = 1200
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _split(text: str) -> list[str]:
    parts: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        parts.append(current)
    return parts


def parse_markdown(source: str, markdown: str) -> list[Chunk]:
    """Split a markdown document into chunks that carry their heading path."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            level = len(match.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, match.group(2))]
            sections.append((" > ".join(t for _, t in stack), []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for heading_path, lines in sections:
        for idx, piece in enumerate(_split("\n".join(lines).strip())):
            digest = hashlib.sha1(f"{source}|{heading_path}|{idx}".encode()).hexdigest()[:10]
            chunks.append(Chunk(id=digest, source=source, heading_path=heading_path, text=piece))
    return chunks


def load_chunks(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*.md")):
        chunks += parse_markdown(path.relative_to(directory).as_posix(), path.read_text())
    return chunks


def ingest_directory(directory: Path, dsn: str, embeddings: Embeddings) -> int:
    chunks = load_chunks(directory)
    if not chunks:
        return 0
    vectors = embeddings.embed_documents([f"{c.heading_path}\n{c.text}" for c in chunks])
    schema = resources.files("infrapilot.rag").joinpath("schema.sql").read_text()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(schema)  # type: ignore[arg-type]
        register_vector(conn)
        for chunk, vec in zip(chunks, vectors, strict=True):
            conn.execute(
                "INSERT INTO chunks (id, source, heading_path, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                "text = EXCLUDED.text, embedding = EXCLUDED.embedding",
                (chunk.id, chunk.source, chunk.heading_path, chunk.text, np.array(vec)),
            )
    return len(chunks)
