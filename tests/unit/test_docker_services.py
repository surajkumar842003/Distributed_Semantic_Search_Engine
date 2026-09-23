"""Unit tests for Docker Compose container services and environment overrides.

Tests:
1. IngestionWorker healthcheck, heartbeat, signal handling, and batch execution.
2. EmbeddingService FastAPI app, /health probe, and /status endpoints.
3. AppConfig environment variable overrides for container networking and paths.
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.config import AppConfig
from src.ingestion.worker import IngestionWorker, check_health
from src.embedding.service import app as embedding_app, service_state, EmbedBatchRequest


# =============================================================================
# 1. Tests for IngestionWorker
# =============================================================================

class TestIngestionWorker:
    """Tests daemon worker heartbeat, healthcheck probe, and lifecycle."""

    def test_worker_initialization(self, tmp_path):
        watch_dir = tmp_path / "raw"
        out_dir = tmp_path / "chunks"
        heartbeat = tmp_path / "heartbeat"

        worker = IngestionWorker(
            watch_dir=str(watch_dir),
            output_dir=str(out_dir),
            watch_interval=5,
            heartbeat_file=str(heartbeat),
        )

        assert worker.watch_dir == watch_dir
        assert worker.output_dir == out_dir
        assert worker.watch_interval == 5
        assert not worker.stop_requested
        assert watch_dir.is_dir()
        assert out_dir.is_dir()

    def test_heartbeat_and_healthcheck(self, tmp_path):
        heartbeat = tmp_path / "worker_healthy"
        worker = IngestionWorker(
            watch_dir=str(tmp_path / "raw"),
            output_dir=str(tmp_path / "chunks"),
            heartbeat_file=str(heartbeat),
        )

        # Before touch, healthcheck fails
        assert check_health(str(heartbeat), max_age_sec=30) == 1

        # Touch heartbeat
        worker.touch_heartbeat()
        assert heartbeat.is_file()

        # Healthcheck passes with fresh heartbeat
        assert check_health(str(heartbeat), max_age_sec=30) == 0

        # Healthcheck fails when stale
        assert check_health(str(heartbeat), max_age_sec=0) == 1

    def test_signal_handling(self, tmp_path):
        import signal
        worker = IngestionWorker(
            watch_dir=str(tmp_path / "raw"),
            output_dir=str(tmp_path / "chunks"),
            heartbeat_file=str(tmp_path / "hb"),
        )
        assert not worker.stop_requested

        # Simulate SIGTERM
        worker._handle_signal(signal.SIGTERM, None)
        assert worker.stop_requested

    def test_single_pass_empty_dir(self, tmp_path):
        worker = IngestionWorker(
            watch_dir=str(tmp_path / "raw"),
            output_dir=str(tmp_path / "chunks"),
            heartbeat_file=str(tmp_path / "hb"),
        )
        processed = worker.run_single_pass()
        assert processed == 0


# =============================================================================
# 2. Tests for EmbeddingService
# =============================================================================

class TestEmbeddingService:
    """Tests FastAPI embedding microservice endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        # Mock embedder for clean testing without requiring CUDA in test runner
        mock_embedder = MagicMock()
        mock_embedder.expected_dim = 384
        mock_embedder.device = "cpu"
        import numpy as np
        mock_embedder.embed_batch.return_value = np.zeros((2, 384), dtype=np.float32)

        service_state["embedder"] = mock_embedder
        service_state["config"] = AppConfig.load_from_dir("configs")
        service_state["start_time"] = time.time()

        with TestClient(embedding_app) as test_client:
            yield test_client

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "uptime_seconds" in data
        assert "embedding_dim" in data
        assert data["embedding_dim"] == 384
        assert "cuda_available" in data

    def test_embed_batch_endpoint(self, client):
        payload = {
            "texts": ["First document passage.", "Second document passage."],
            "normalize": True,
        }
        resp = client.post("/embed", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 2
        assert data["dimension"] == 384
        assert data["count"] == 2
        assert "latency_ms" in data

    def test_status_endpoint(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "output_directory" in data
        assert "total_shards_on_disk" in data
        assert "completed_units_in_checkpoint" in data


# =============================================================================
# 3. Tests for AppConfig Environment Overrides
# =============================================================================

class TestConfigEnvironmentOverrides:
    """Tests that AppConfig.load_from_dir honors container environment variables."""

    def test_postgres_env_overrides(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "db.internal.host")
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        monkeypatch.setenv("POSTGRES_DB", "test_rag_db")
        monkeypatch.setenv("POSTGRES_USER", "rag_admin")
        monkeypatch.setenv("POSTGRES_PASSWORD", "super_secret_pw")

        cfg = AppConfig.load_from_dir("configs")
        assert cfg.postgres.host == "db.internal.host"
        assert cfg.postgres.port == 5433
        assert cfg.postgres.database == "test_rag_db"
        assert cfg.postgres.user == "rag_admin"
        assert cfg.postgres.password == "super_secret_pw"

    def test_storage_and_hardware_env_overrides(self, monkeypatch):
        monkeypatch.setenv("DATA_DIR", "/custom/data/mount")
        monkeypatch.setenv("CACHE_DIR", "/custom/cache/mount")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        monkeypatch.setenv("NUM_CPU_WORKERS", "16")

        cfg = AppConfig.load_from_dir("configs")
        assert cfg.paths.data_dir == "/custom/data/mount"
        assert cfg.paths.chunks_staging_dir == "/custom/data/mount/chunks"
        assert cfg.paths.embeddings_dir == "/custom/data/mount/embeddings"
        assert cfg.paths.cache_dir == "/custom/cache/mount"
        assert cfg.embedding.device == "cuda:1"
        assert cfg.retrieval.rerank_device == "cuda:1"
        assert cfg.ingestion.num_cpu_workers == 16

