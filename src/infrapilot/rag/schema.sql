-- nomic-embed-text produces 768-dimensional vectors (verified against Ollama).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    source       text NOT NULL,
    heading_path text NOT NULL,
    text         text NOT NULL,
    embedding    vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text side of hybrid retrieval. Generated, so ingestion code does not change.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', heading_path || ' ' || text)) STORED;

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
