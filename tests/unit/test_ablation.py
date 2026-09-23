"""Unit tests for the ablation study framework.

Tests:
1. ExperimentConfig generation and validation
2. Config YAML serialization round-trip
3. GPU monitor output parsing (mocked)
4. Result aggregation
5. Experiment ID uniqueness
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ablation.ablation_config import (
    ExperimentConfig,
    generate_all_experiments,
    generate_index_type_experiments,
    generate_nprobe_experiments,
    generate_chunk_size_experiments,
    generate_retrieval_mode_experiments,
    generate_reranking_experiments,
    generate_worker_count_experiments,
    save_all_configs,
)
from src.ablation.ablation_runner import ExperimentResult


# ============================================================================
# Test Config Generation
# ============================================================================

class TestExperimentConfigGeneration:
    """Tests for experiment config factory functions."""

    def test_index_type_experiments_count(self):
        configs = generate_index_type_experiments()
        assert len(configs) == 3
        types = {c.index_type for c in configs}
        assert types == {"FlatIP", "IVF-Flat", "IVF-PQ"}

    def test_nprobe_experiments_count(self):
        configs = generate_nprobe_experiments()
        assert len(configs) == 7
        nprobes = sorted(c.nprobe for c in configs)
        assert nprobes == [1, 4, 8, 16, 32, 64, 128]

    def test_chunk_size_experiments_count(self):
        configs = generate_chunk_size_experiments()
        assert len(configs) == 3
        sizes = sorted(c.chunk_target_tokens for c in configs)
        assert sizes == [128, 256, 512]

    def test_retrieval_mode_experiments_count(self):
        configs = generate_retrieval_mode_experiments()
        assert len(configs) == 4
        modes = {c.retrieval_mode for c in configs}
        assert modes == {"dense", "sparse", "hybrid_rrf", "hybrid_rerank"}

    def test_reranking_experiments_count(self):
        configs = generate_reranking_experiments()
        assert len(configs) == 2
        modes = {c.retrieval_mode for c in configs}
        assert "hybrid_rrf" in modes
        assert "hybrid_rerank" in modes

    def test_worker_count_experiments_count(self):
        configs = generate_worker_count_experiments()
        assert len(configs) == 5
        workers = sorted(c.num_workers for c in configs)
        assert workers == [1, 2, 4, 8, 16]

    def test_all_experiments_total(self):
        configs = generate_all_experiments()
        # 3 + 7 + 3 + 4 + 2 + 5 = 24
        assert len(configs) == 24

    def test_config_fields_valid_types(self):
        configs = generate_all_experiments()
        for c in configs:
            assert isinstance(c.experiment_id, str)
            assert len(c.experiment_id) > 0
            assert isinstance(c.dimension, str)
            assert c.dimension in {
                "index_type", "nprobe", "chunk_size",
                "retrieval_mode", "reranking", "worker_count",
            }
            assert isinstance(c.nlist, int)
            assert isinstance(c.nprobe, int)
            assert isinstance(c.chunk_target_tokens, int)
            assert isinstance(c.num_workers, int)

    def test_all_dimensions_represented(self):
        configs = generate_all_experiments()
        dimensions = {c.dimension for c in configs}
        assert dimensions == {
            "index_type", "nprobe", "chunk_size",
            "retrieval_mode", "reranking", "worker_count",
        }


# ============================================================================
# Test Config Serialization
# ============================================================================

class TestExperimentConfigSerialization:
    """Tests for YAML/JSON round-trip serialization."""

    def test_yaml_round_trip(self, tmp_path):
        original = ExperimentConfig(
            experiment_id="test_yaml_rt",
            dimension="index_type",
            description="Test YAML round-trip",
            index_type="IVF-Flat",
            nlist=128,
            nprobe=8,
        )
        yaml_path = tmp_path / "test.yaml"
        original.save_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.experiment_id == original.experiment_id
        assert loaded.index_type == original.index_type
        assert loaded.nlist == original.nlist
        assert loaded.nprobe == original.nprobe

    def test_json_serialization(self, tmp_path):
        config = ExperimentConfig(
            experiment_id="test_json",
            dimension="nprobe",
            description="Test JSON",
            nprobe=32,
        )
        json_path = tmp_path / "test.json"
        config.save_json(json_path)

        with open(json_path, "r") as f:
            data = json.load(f)
        assert data["experiment_id"] == "test_json"
        assert data["nprobe"] == 32

    def test_to_dict_completeness(self):
        config = ExperimentConfig(
            experiment_id="test_dict",
            dimension="chunk_size",
            description="Test dict",
        )
        d = config.to_dict()
        assert "experiment_id" in d
        assert "dimension" in d
        assert "index_type" in d
        assert "nlist" in d
        assert "chunk_target_tokens" in d

    def test_save_all_configs(self, tmp_path):
        configs = generate_all_experiments()
        save_all_configs(configs, tmp_path)

        # Check manifest
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.is_file()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["total_experiments"] == 24

        # Check YAML files
        yaml_files = list(tmp_path.glob("*.yaml"))
        assert len(yaml_files) == 24


# ============================================================================
# Test Experiment ID Uniqueness
# ============================================================================

class TestExperimentIDUniqueness:
    """Ensures all generated experiment IDs are globally unique."""

    def test_all_ids_unique(self):
        configs = generate_all_experiments()
        ids = [c.experiment_id for c in configs]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_ids_are_filename_safe(self):
        configs = generate_all_experiments()
        import re
        for c in configs:
            assert re.match(r"^[a-zA-Z0-9_-]+$", c.experiment_id), \
                f"ID contains unsafe characters: {c.experiment_id}"


# ============================================================================
# Test ExperimentResult
# ============================================================================

class TestExperimentResult:
    """Tests for result dataclass."""

    def test_result_serialization(self, tmp_path):
        result = ExperimentResult(
            experiment_id="test_result",
            config={"dimension": "test", "index_type": "FlatIP"},
            recall_at_10=0.8571,
            mrr=0.9048,
            ndcg_at_10=0.6131,
            query_p50_ms=12.5,
            query_p95_ms=15.0,
            index_memory_mb=50.26,
            index_build_time_sec=0.326,
        )

        path = tmp_path / "result.json"
        result.save(path)
        assert path.is_file()

        loaded = ExperimentResult.load(path)
        assert loaded.experiment_id == "test_result"
        assert loaded.recall_at_10 == 0.8571
        assert loaded.query_p50_ms == 12.5

    def test_result_to_dict(self):
        result = ExperimentResult(
            experiment_id="test_dict",
            config={"a": 1},
            recall_at_10=0.95,
        )
        d = result.to_dict()
        assert d["experiment_id"] == "test_dict"
        assert d["recall_at_10"] == 0.95
        assert "raw_query_latencies_ms" in d
        assert "errors" in d


# ============================================================================
# Test GPU Monitor (mocked)
# ============================================================================

class TestGPUMonitorParsing:
    """Tests GPU monitor output parsing logic with mocked data."""

    def test_monitor_properties_empty(self):
        from src.ablation.gpu_monitor import GPUUtilizationMonitor
        monitor = GPUUtilizationMonitor(device_id=0)
        # Without starting, should return safe defaults
        assert monitor.mean_utilization == 0.0
        assert monitor.peak_utilization == 0.0
        assert monitor.sample_count == 0
        assert monitor.samples == []

    def test_monitor_to_dict(self):
        from src.ablation.gpu_monitor import GPUUtilizationMonitor
        monitor = GPUUtilizationMonitor(device_id=0)
        d = monitor.to_dict()
        assert d["device_id"] == 0
        assert d["sample_count"] == 0
        assert d["mean_utilization_pct"] == 0.0

    def test_monitor_with_manual_samples(self):
        from src.ablation.gpu_monitor import GPUUtilizationMonitor
        monitor = GPUUtilizationMonitor(device_id=0)
        # Manually inject samples to test aggregation
        monitor._samples = [45.0, 72.0, 88.0, 50.0, 30.0]
        assert monitor.sample_count == 5
        assert monitor.mean_utilization == pytest.approx(57.0, abs=0.1)
        assert monitor.peak_utilization == 88.0

