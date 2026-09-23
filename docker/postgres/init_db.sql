-- PostgreSQL Initialization DDL for Distributed Wikipedia Semantic Search Engine
-- Optimized for high-speed COPY bulk loading of 20M+ chunks
--
-- DESIGN DECISION: No foreign keys during bulk load.
-- The deterministic chunk_id = (doc_id << 16) | chunk_seq scheme
-- already encodes the document-chunk relationship. FK validation
-- during COPY of 20M rows adds ~30 min overhead with zero runtime benefit.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Documents table: 1 row per Wikipedia article (~6.5M rows)
CREATE TABLE IF NOT EXISTS documents (
    doc_id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    source_shard INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table: ~20M rows
-- NOTE: No FK on doc_id (intentional, see design decision above)
-- NOTE: tsvector column and GIN index are added AFTER bulk load via post_bulk_load.sql
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id BIGINT PRIMARY KEY,
    doc_id BIGINT NOT NULL,
    section_path TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INT NOT NULL,
    faiss_id BIGINT NOT NULL
);

-- B-Tree index on doc_id for analytical joins (optional, low overhead)
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- Query logs for evaluation and latency tracking
CREATE TABLE IF NOT EXISTS query_logs (
    query_id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    retrieved_chunk_ids BIGINT[] NOT NULL,
    reranked_chunk_ids BIGINT[] NOT NULL,
    dense_latency_ms REAL,
    sparse_latency_ms REAL,
    rerank_latency_ms REAL,
    total_latency_ms REAL NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
