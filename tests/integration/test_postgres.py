"""Integration tests for PostgreSQL storage layer.

Tests live database operations when PostgreSQL is accessible.
If the database service is not reachable, tests gracefully skip with instructions.
"""

import asyncio
import time
import pytest

from src.config import AppConfig, PostgresConfig
from src.storage.postgres_client import PostgresClient
from src.storage.migrator import DatabaseMigrator


def _run_async(coro):
    """Helper to run an async coroutine synchronously without external pytest plugins."""
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def pg_client():
    """Initializes and yields a PostgresClient if database is reachable."""
    cfg = AppConfig.load_from_dir("configs")
    client = PostgresClient(cfg.postgres)

    async def _init_client():
        try:
            await client.connect()
            healthy = await client.is_healthy()
            return healthy
        except Exception:
            return False

    is_healthy = _run_async(_init_client())
    if not is_healthy:
        pytest.skip(
            "PostgreSQL at localhost:5432 is not reachable. "
            "Start container via: sudo docker compose up -d postgres"
        )

    yield client

    _run_async(client.disconnect())


class TestPostgresIntegration:
    """Live database integration tests."""

    def test_apply_migrations(self, pg_client):
        """Verifies running the migration engine on the live database."""
        async def _test():
            migrator = DatabaseMigrator()
            applied = await migrator.apply_migrations(pg_client.pool)
            assert isinstance(applied, list)

        _run_async(_test())

    def test_idempotent_document_and_chunk_insertion(self, pg_client):
        """Verifies idempotent document and chunk ingestion."""
        async def _test():
            # 1. Insert documents
            doc = {
                "doc_id": 999901,
                "title": "Quantum Computing",
                "url": "https://en.wikipedia.org/wiki/Quantum_computing",
                "source_shard": 1,
            }
            added1 = await pg_client.insert_documents_batch([doc])
            assert added1 == 1

            # Re-insert should succeed idempotently
            added2 = await pg_client.insert_documents_batch([doc])
            assert added2 == 1

            # 2. Insert chunk via COPY
            chunk_record = (
                9999010001,  # chunk_id
                999901,      # doc_id
                "Introduction",  # section_path
                "Quantum computers utilize qubits in superposition to perform calculations.",  # text
                12,          # token_count
                9999010001,  # faiss_id
            )
            c_added1 = await pg_client.copy_chunks_idempotent([chunk_record])
            assert c_added1 == 1

            # Re-insert via COPY should be idempotent (no duplicate key violation)
            c_added2 = await pg_client.copy_chunks_idempotent([chunk_record])
            assert c_added2 == 1

            # 3. Retrieve chunk and verify document metadata is joined
            hydrated = await pg_client.get_chunks_by_ids([9999010001])
            assert len(hydrated) == 1
            res = hydrated[0]
            assert res["chunk_id"] == 9999010001
            assert res["doc_id"] == 999901
            assert res["title"] == "Quantum Computing"
            assert res["url"] == "https://en.wikipedia.org/wiki/Quantum_computing"
            assert "qubits" in res["text"]

        _run_async(_test())

    def test_full_text_search_fts(self, pg_client):
        """Verifies tsvector and GIN full-text search with ranking."""
        async def _test():
            # Insert test chunk with distinctive keywords
            chunk_record = (
                9999020001,
                999902,
                "Overview",
                "Photosynthesis is a biological process used by plants to synthesize nutrients.",
                15,
                9999020001,
            )
            await pg_client.copy_chunks_idempotent([chunk_record])

            # Search for keyword
            results = await pg_client.search_fts("photosynthesis", top_k=5)
            assert len(results) >= 1
            found_ids = [r["chunk_id"] for r in results]
            assert 9999020001 in found_ids
            assert results[0]["rank_score"] > 0.0

        _run_async(_test())

    def test_query_logging(self, pg_client):
        """Verifies logging query evaluation telemetry."""
        async def _test():
            qid = await pg_client.log_query(
                query_text="what is photosynthesis?",
                retrieved_chunk_ids=[9999020001],
                reranked_chunk_ids=[9999020001],
                dense_latency_ms=1.2,
                sparse_latency_ms=0.8,
                rerank_latency_ms=4.5,
                total_latency_ms=6.5,
            )
            assert qid > 0

        _run_async(_test())

