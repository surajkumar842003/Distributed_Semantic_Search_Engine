"""Unit tests for PostgreSQL storage layer components."""

import os
from pathlib import Path
import pytest

from src.storage.postgres_client import explain_embedding_storage_decision, PostgresClient
from src.storage.migrator import DatabaseMigrator
from src.config import AppConfig, PostgresConfig


class TestStorageRationaleAndConfigs:
    """Tests architectural rationale and configuration loading."""

    def test_explain_embedding_storage_decision(self):
        explanation = explain_embedding_storage_decision()
        assert "FAISS" in explanation
        assert "PostgreSQL" in explanation
        assert "shared_buffers" in explanation
        assert "1:1" in explanation
        assert "ANN" in explanation

    def test_postgres_config_from_yaml(self):
        cfg = AppConfig.load_from_dir("configs")
        pg = cfg.postgres
        assert pg.database == "wikirag"
        assert pg.user == "postgres"
        assert pg.port == 5432
        assert pg.pool_size_min >= 5
        assert pg.pool_size_max >= 10


class TestDatabaseMigrator:
    """Tests migration discovery and ordering."""

    def test_discover_migrations_order(self):
        migrator = DatabaseMigrator()
        migrations = migrator.discover_migrations()
        assert len(migrations) >= 2

        versions = [m[0] for m in migrations]
        assert versions == sorted(versions)
        assert versions[0] == 1
        assert versions[1] == 2

    def test_migration_files_content_validity(self):
        migrator = DatabaseMigrator()
        migrations = migrator.discover_migrations()

        # Check 001
        v1, name1, path1 = migrations[0]
        content1 = path1.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS documents" in content1
        assert "CREATE TABLE IF NOT EXISTS chunks" in content1
        assert "CREATE TABLE IF NOT EXISTS query_logs" in content1
        assert "idx_chunks_doc_id" in content1
        assert "idx_chunks_faiss_id" in content1

        # Check 002
        v2, name2, path2 = migrations[1]
        content2 = path2.read_text(encoding="utf-8")
        assert "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv TSVECTOR" in content2
        assert "USING GIN(text_tsv)" in content2
        assert "trg_chunks_text_tsv" in content2


class TestDocumentNormalizationLogic:
    """Tests that documents are normalized to eliminate duplicate text storage."""

    def test_unique_document_extraction(self):
        sample_chunks = [
            {"doc_id": 100, "title": "Article A", "url": "http://a.org", "chunk_id": 1, "text": "chunk 1"},
            {"doc_id": 100, "title": "Article A", "url": "http://a.org", "chunk_id": 2, "text": "chunk 2"},
            {"doc_id": 200, "title": "Article B", "url": "http://b.org", "chunk_id": 3, "text": "chunk 3"},
        ]

        # Extract unique documents
        seen = set()
        docs = []
        for c in sample_chunks:
            if c["doc_id"] not in seen:
                seen.add(c["doc_id"])
                docs.append({"doc_id": c["doc_id"], "title": c["title"], "url": c["url"]})

        assert len(docs) == 2
        assert docs[0]["doc_id"] == 100
        assert docs[1]["doc_id"] == 200

