-- Migration 001: Initial Relational Schema
-- Tables: documents, chunks, query_logs

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 1. Documents table: Stores article metadata.
-- Normalized to avoid duplicating title and URL across multiple chunks.
CREATE TABLE IF NOT EXISTS documents (
    doc_id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    source_shard INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index on document title for exact and prefix metadata lookups
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);

-- 2. Chunks table: Stores passage text and metadata.
-- chunk_id is a deterministic 64-bit ID: (doc_id << 16) | chunk_seq
-- faiss_id maps 1:1 to the FAISS vector index label.
-- NOTE: No FK constraint during bulk ingestion to maximize COPY ingestion throughput.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id BIGINT PRIMARY KEY,
    doc_id BIGINT NOT NULL,
    section_path TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INT NOT NULL,
    faiss_id BIGINT NOT NULL
);

-- B-tree index on doc_id for fetching all chunks belonging to a document
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- B-tree index on faiss_id for fast candidate hydration from vector retrieval results
CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id);

-- 3. Query logs: Records retrieval queries, candidate IDs, and stage latencies.
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

-- Index for ordering and analytics on recent queries
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at DESC);

