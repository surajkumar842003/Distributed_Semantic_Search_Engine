"""Asynchronous PostgreSQL client for Wikipedia document and chunk storage.

Features:
- Connection pooling with asyncpg.
- Strict parameterized SQL (preventing injection vulnerabilities).
- High-speed idempotent COPY via temporary staging tables.
- Normalized document metadata (avoiding duplicate chunk text storage).
- PostgreSQL Full-Text Search using tsvector and GIN index with ts_rank_cd.
- 64-bit chunk_id and faiss_id preservation.
- Query logging and latency profiling.
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Sequence
import asyncpg

from src.config import PostgresConfig
from src.common.logging import get_logger

logger = get_logger("storage.postgres_client")


def explain_embedding_storage_decision() -> str:
    """Explains why dense embeddings are managed by FAISS rather than stored in PostgreSQL."""
    return (
        "Architectural Rationale: Why Dense Embeddings are NOT Stored in PostgreSQL:\n"
        "1. Performance Separation: FAISS is purpose-built for approximate nearest neighbor (ANN) "
        "search using AVX-512 SIMD vector registers and optional GPU acceleration, achieving query "
        "latencies of 0.07-0.8 ms. Relational database engines cannot match this vector throughput.\n"
        "2. Storage & Cache Bloat: Storing 384-dimensional float32 vectors (1,536 bytes each) for 20M chunks "
        "would add ~30 GB of raw table bloat plus pgvector index overhead, evicting relational tables and "
        "full-text search indexes from PostgreSQL's shared_buffers cache.\n"
        "3. Clear Separation of Concerns: PostgreSQL serves as the relational system of record, lexical search "
        "engine (tsvector + GIN), and candidate hydrator (fetching text, title, and URL by chunk_id). "
        "FAISS serves as the dense vector index.\n"
        "4. 1:1 ID Mapping: Because chunk_id == faiss_id (64-bit integer), vector search results directly index "
        "into PostgreSQL's primary key index (chunks.chunk_id) in O(1) time without duplicating vector storage."
    )


class PostgresClient:
    """High-performance async PostgreSQL client for Wikipedia RAG search engine."""

    def __init__(self, config: Optional[PostgresConfig] = None):
        self.config = config or PostgresConfig()
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initializes the asyncpg connection pool."""
        if self.pool is not None:
            return

        logger.info(
            f"Connecting to PostgreSQL at {self.config.host}:{self.config.port}/{self.config.database} "
            f"(user={self.config.user}, pool={self.config.pool_size_min}-{self.config.pool_size_max})"
        )
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.pool_size_min,
            max_size=self.config.pool_size_max,
            command_timeout=60.0,
        )
        logger.info("PostgreSQL connection pool initialized successfully.")

    async def disconnect(self):
        """Closes the asyncpg connection pool."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed.")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def is_healthy(self) -> bool:
        """Verifies database connectivity."""
        if self.pool is None:
            return False
        try:
            async with self.pool.acquire() as conn:
                res = await conn.fetchval("SELECT 1;")
                return res == 1
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    # =========================================================================
    # Document Ingestion (Normalized, Idempotent)
    # =========================================================================

    async def insert_documents_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Idempotently inserts or updates document metadata in batches.

        Avoids duplicating article title and URL across multiple chunks.
        """
        if not documents:
            return 0

        query = """
            INSERT INTO documents (doc_id, title, url, source_shard)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (doc_id) DO UPDATE
            SET title = EXCLUDED.title,
                url = EXCLUDED.url,
                source_shard = EXCLUDED.source_shard;
        """
        tuples = [
            (
                int(d["doc_id"]),
                str(d["title"]),
                str(d["url"]),
                int(d.get("source_shard", 0)),
            )
            for d in documents
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(query, tuples)
        return len(tuples)

    # =========================================================================
    # Chunk Ingestion (Batch & High-Speed COPY)
    # =========================================================================

    async def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """Idempotently inserts chunk records in batches using parameterized SQL."""
        if not chunks:
            return 0

        query = """
            INSERT INTO chunks (chunk_id, doc_id, section_path, text, token_count, faiss_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (chunk_id) DO NOTHING;
        """
        tuples = [
            (
                int(c["chunk_id"]),
                int(c["doc_id"]),
                str(c.get("section_path", "")),
                str(c["text"]),
                int(c.get("token_count", 0)),
                int(c.get("faiss_id", c["chunk_id"])),
            )
            for c in chunks
        ]

        async with self.pool.acquire() as conn:
            await conn.executemany(query, tuples)
        return len(tuples)

    async def copy_chunks_idempotent(
        self,
        records: Sequence[Tuple[int, int, str, str, int, int]],
    ) -> int:
        """Streams chunk records via COPY into a staging table and performs an idempotent insert.

        Achieves raw COPY throughput (>100,000 rows/s) while preventing duplicate key collisions.

        Args:
            records: Sequence of (chunk_id, doc_id, section_path, text, token_count, faiss_id).

        Returns:
            int: Number of records processed.
        """
        if not records:
            return 0

        create_staging_sql = """
            CREATE TEMP TABLE temp_chunks_stage (
                chunk_id BIGINT,
                doc_id BIGINT,
                section_path TEXT,
                text TEXT,
                token_count INT,
                faiss_id BIGINT
            ) ON COMMIT DROP;
        """

        insert_from_stage_sql = """
            INSERT INTO chunks (chunk_id, doc_id, section_path, text, token_count, faiss_id)
            SELECT chunk_id, doc_id, section_path, text, token_count, faiss_id
            FROM temp_chunks_stage
            ON CONFLICT (chunk_id) DO NOTHING;
        """

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(create_staging_sql)
                await conn.copy_records_to_table(
                    "temp_chunks_stage",
                    records=records,
                    columns=["chunk_id", "doc_id", "section_path", "text", "token_count", "faiss_id"],
                )
                await conn.execute(insert_from_stage_sql)

        return len(records)

    # =========================================================================
    # Candidate Hydration & Lookups (Parameterized)
    # =========================================================================

    async def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieves chunks and joined document metadata for a list of chunk_ids.

        Used for candidate hydration after vector retrieval.
        """
        if not chunk_ids:
            return []

        query = """
            SELECT c.chunk_id, c.doc_id, c.section_path, c.text, c.token_count, c.faiss_id,
                   d.title, d.url
            FROM chunks c
            LEFT JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id = ANY($1::BIGINT[]);
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, [int(cid) for cid in chunk_ids])
            return [dict(r) for r in rows]

    async def get_chunks_by_doc_id(self, doc_id: int) -> List[Dict[str, Any]]:
        """Retrieves all chunks belonging to a document ordered by sequence."""
        query = """
            SELECT c.chunk_id, c.doc_id, c.section_path, c.text, c.token_count, c.faiss_id,
                   d.title, d.url
            FROM chunks c
            LEFT JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.doc_id = $1
            ORDER BY c.chunk_id ASC;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, int(doc_id))
            return [dict(r) for r in rows]

    # =========================================================================
    # Full-Text Search (tsvector + GIN)
    # =========================================================================

    async def search_fts(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Executes full-text keyword search using tsvector and GIN index with rank scoring."""
        sql = """
            SELECT c.chunk_id, c.doc_id, c.section_path, c.text, c.token_count, c.faiss_id,
                   d.title, d.url,
                   ts_rank_cd(c.text_tsv, plainto_tsquery('english', $1)) AS rank_score
            FROM chunks c
            LEFT JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.text_tsv @@ plainto_tsquery('english', $1)
            ORDER BY rank_score DESC
            LIMIT $2;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query_text, top_k)
            return [dict(r) for r in rows]

    # =========================================================================
    # Query Logging & Evaluation Tracking
    # =========================================================================

    async def log_query(
        self,
        query_text: str,
        retrieved_chunk_ids: List[int],
        reranked_chunk_ids: List[int],
        dense_latency_ms: Optional[float] = None,
        sparse_latency_ms: Optional[float] = None,
        rerank_latency_ms: Optional[float] = None,
        total_latency_ms: float = 0.0,
    ) -> int:
        """Records an executed retrieval query, candidate lists, and latencies."""
        sql = """
            INSERT INTO query_logs (
                query_text, retrieved_chunk_ids, reranked_chunk_ids,
                dense_latency_ms, sparse_latency_ms, rerank_latency_ms, total_latency_ms
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING query_id;
        """
        async with self.pool.acquire() as conn:
            qid = await conn.fetchval(
                sql,
                str(query_text),
                [int(x) for x in retrieved_chunk_ids],
                [int(x) for x in reranked_chunk_ids],
                float(dense_latency_ms) if dense_latency_ms is not None else None,
                float(sparse_latency_ms) if sparse_latency_ms is not None else None,
                float(rerank_latency_ms) if rerank_latency_ms is not None else None,
                float(total_latency_ms),
            )
            return int(qid)

    # =========================================================================
    # Telemetry & Table Statistics
    # =========================================================================

    async def get_table_counts(self) -> Dict[str, int]:
        """Returns row counts for documents, chunks, and query_logs."""
        async with self.pool.acquire() as conn:
            doc_count = await conn.fetchval("SELECT count(*) FROM documents;")
            chunk_count = await conn.fetchval("SELECT count(*) FROM chunks;")
            log_count = await conn.fetchval("SELECT count(*) FROM query_logs;")
            return {
                "documents": int(doc_count),
                "chunks": int(chunk_count),
                "query_logs": int(log_count),
            }

