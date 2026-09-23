"""Unit and integration tests for FAISS vector indexing module.

Verifies:
1. Metric explanation and L2 normalization logic.
2. Training requirement enforcement (IVF/PQ requires training, Flat does not).
3. Incremental shard addition and 64-bit chunk_id preservation.
4. Duplicate chunk_id prevention.
5. Atomic saving and reloading (both normal and MMAP mode).
6. Recall accuracy (FlatIP 100%, IVF-Flat high recall, IVF-PQ quantization trade-off).
7. Resumable indexing with PipelineCheckpoint.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Set
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import faiss

from src.indexing.faiss_indexer import (
    FAISSIndexer,
    explain_metric_choice,
    ensure_l2_normalized,
    extract_index_ids,
)
from src.common.checkpoint import PipelineCheckpoint


def _create_synthetic_embedding_shard(
    path: Path,
    num_vectors: int = 50,
    dim: int = 384,
    start_id: int = 268920000000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to generate a Parquet embedding shard with deterministic 64-bit chunk_ids."""
    rng = np.random.RandomState(seed)
    chunk_ids = np.array([start_id + i for i in range(num_vectors)], dtype=np.int64)
    raw_vecs = rng.randn(num_vectors, dim).astype(np.float32)
    faiss.normalize_L2(raw_vecs)

    flat_embeddings = raw_vecs.ravel()
    table = pa.Table.from_arrays(
        [
            pa.array(chunk_ids, type=pa.int64()),
            pa.FixedSizeListArray.from_arrays(flat_embeddings, dim),
        ],
        names=["chunk_id", "embedding"],
    )
    pq.write_table(table, str(path))
    return chunk_ids, raw_vecs


class TestMetricAndNormalization:
    """Tests similarity metric mathematical properties and normalization enforcement."""

    def test_metric_explanation_contains_key_properties(self):
        explanation = explain_metric_choice("INNER_PRODUCT")
        assert "INNER_PRODUCT" in explanation
        assert "Cosine Similarity" in explanation
        assert "cos(theta)" in explanation
        assert "Euclidean" in explanation

    def test_ensure_l2_normalized(self):
        rng = np.random.RandomState(123)
        # Unnormalized vectors
        vecs = rng.randn(10, 384).astype(np.float32) * 5.0
        assert not np.allclose(np.linalg.norm(vecs, axis=1), 1.0)

        norm_vecs = ensure_l2_normalized(vecs)
        assert np.allclose(np.linalg.norm(norm_vecs, axis=1), 1.0, atol=1e-5)

        # Already normalized vectors should not be altered
        norm_vecs_again = ensure_l2_normalized(norm_vecs)
        np.testing.assert_allclose(norm_vecs, norm_vecs_again, atol=1e-6)


class TestFAISSIndexerLifecycle:
    """Tests index creation, training, incremental ingestion, and duplicate prevention."""

    def test_flat_index_requires_no_training(self):
        indexer = FAISSIndexer(dim=384, index_type="FlatIP")
        assert indexer.is_trained
        assert indexer.ntotal == 0

    def test_ivf_requires_training_before_add(self):
        indexer = FAISSIndexer(dim=384, index_type="IVF-Flat", nlist=16)
        assert not indexer.is_trained

        rng = np.random.RandomState(42)
        dummy_vecs = rng.randn(5, 384).astype(np.float32)
        dummy_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        with pytest.raises(RuntimeError, match="not trained"):
            indexer.add_vectors(dummy_vecs, dummy_ids)

    def test_training_sample_validation(self):
        indexer = FAISSIndexer(dim=384, index_type="IVF-PQ", nlist=16, m_subquantizers=48)
        # Too few points for 8-bit PQ (< 256)
        too_few = np.random.randn(50, 384).astype(np.float32)
        with pytest.raises(ValueError, match="requires at least 256"):
            indexer.train(too_few)

    def test_incremental_shard_add_and_chunk_id_preservation(self, tmp_path):
        """Validates adding vectors shard by shard and preserving 64-bit chunk_ids."""
        shard1_path = tmp_path / "embeddings_0000.parquet"
        shard2_path = tmp_path / "embeddings_0001.parquet"

        ids1, vecs1 = _create_synthetic_embedding_shard(shard1_path, num_vectors=40, start_id=1000, seed=42)
        ids2, vecs2 = _create_synthetic_embedding_shard(shard2_path, num_vectors=30, start_id=2000, seed=43)

        # FlatIP index
        indexer = FAISSIndexer(dim=384, index_type="FlatIP")
        added1 = indexer.add_from_parquet_shard(shard1_path)
        assert added1 == 40
        assert indexer.ntotal == 40

        added2 = indexer.add_from_parquet_shard(shard2_path)
        assert added2 == 30
        assert indexer.ntotal == 70

        # Query with vector from shard 1
        scores1, query_ids1 = indexer.search(vecs1[:3], top_k=1)
        np.testing.assert_array_equal(query_ids1.ravel(), ids1[:3])
        np.testing.assert_allclose(scores1.ravel(), np.ones(3), atol=1e-4)

        # Query with vector from shard 2
        scores2, query_ids2 = indexer.search(vecs2[:3], top_k=1)
        np.testing.assert_array_equal(query_ids2.ravel(), ids2[:3])
        np.testing.assert_allclose(scores2.ravel(), np.ones(3), atol=1e-4)

    def test_duplicate_chunk_id_prevention(self):
        """Verifies duplicate chunk_id detection across and within batches."""
        indexer = FAISSIndexer(dim=384, index_type="FlatIP")
        rng = np.random.RandomState(99)
        vecs = rng.randn(4, 384).astype(np.float32)
        faiss.normalize_L2(vecs)
        ids = np.array([5001, 5002, 5003, 5004], dtype=np.int64)

        # First add: 4 vectors added
        added = indexer.add_vectors(vecs, ids)
        assert added == 4
        assert indexer.ntotal == 4

        # Second add with overlapping IDs (5003 and 5004 already exist, 5005 is new)
        vecs_overlap = rng.randn(3, 384).astype(np.float32)
        faiss.normalize_L2(vecs_overlap)
        ids_overlap = np.array([5003, 5004, 5005], dtype=np.int64)

        # With skip_duplicates=True: only 5005 should be added
        added_overlap = indexer.add_vectors(vecs_overlap, ids_overlap, skip_duplicates=True)
        assert added_overlap == 1
        assert indexer.ntotal == 5

        # With skip_duplicates=False: should raise ValueError
        with pytest.raises(ValueError, match="Duplicate chunk_id"):
            indexer.add_vectors(vecs_overlap, ids_overlap, skip_duplicates=False)


class TestPersistenceAndMMAP:
    """Tests atomic save, metadata generation, and zero-copy MMAP reloading."""

    def test_save_and_reload_mmap_equivalence(self, tmp_path):
        idx_path = tmp_path / "wiki_test.index"

        # Create and populate IVF-Flat index
        rng = np.random.RandomState(42)
        train_vecs = rng.randn(1000, 384).astype(np.float32)
        faiss.normalize_L2(train_vecs)

        indexer = FAISSIndexer(dim=384, index_type="IVF-Flat", nlist=16)
        indexer.train(train_vecs)

        data_vecs = train_vecs[:100]
        data_ids = np.array([9000 + i for i in range(100)], dtype=np.int64)
        indexer.add_vectors(data_vecs, data_ids)

        # Save index atomically
        indexer.save(idx_path)
        assert idx_path.exists()
        assert idx_path.with_suffix(".meta.json").exists()

        # Reload with standard heap loading
        loaded_heap = FAISSIndexer.load(idx_path, use_mmap=False)
        assert loaded_heap.ntotal == 100
        assert len(loaded_heap.seen_chunk_ids) == 100

        # Reload with memory mapping (IO_FLAG_MMAP)
        loaded_mmap = FAISSIndexer.load(idx_path, use_mmap=True)
        assert loaded_mmap.ntotal == 100
        assert len(loaded_mmap.seen_chunk_ids) == 100

        # Verify search results are identical between original, heap, and mmap
        query = data_vecs[:5]
        s_orig, id_orig = indexer.search(query, top_k=5, nprobe=8)
        s_heap, id_heap = loaded_heap.search(query, top_k=5, nprobe=8)
        s_mmap, id_mmap = loaded_mmap.search(query, top_k=5, nprobe=8)

        np.testing.assert_array_equal(id_orig, id_heap)
        np.testing.assert_array_equal(id_orig, id_mmap)
        np.testing.assert_allclose(s_orig, s_heap, atol=1e-5)
        np.testing.assert_allclose(s_orig, s_mmap, atol=1e-5)


class TestRecallAccuracy:
    """Tests search recall comparison across FlatIP, IVF-Flat, and IVF-PQ."""

    def test_recall_at_k_hierarchy(self):
        """Verifies FlatIP gives 1.0 self-recall, IVF-Flat gives high recall, and IVF-PQ reflects quantization."""
        dim = 384
        n_train = 1500
        n_data = 800
        n_queries = 20

        rng = np.random.RandomState(77)
        train_vecs = rng.randn(n_train, dim).astype(np.float32)
        faiss.normalize_L2(train_vecs)

        data_vecs = train_vecs[:n_data]
        chunk_ids = np.array([3000 + i for i in range(n_data)], dtype=np.int64)

        query_vecs = data_vecs[:n_queries]
        ground_truth_top1 = chunk_ids[:n_queries]

        # 1. Exact FlatIP baseline
        flat_idx = FAISSIndexer(dim=dim, index_type="FlatIP")
        flat_idx.add_vectors(data_vecs, chunk_ids)
        _, flat_res = flat_idx.search(query_vecs, top_k=1)
        np.testing.assert_array_equal(flat_res.ravel(), ground_truth_top1)

        # 2. IVF-Flat
        ivf_idx = FAISSIndexer(dim=dim, index_type="IVF-Flat", nlist=16)
        ivf_idx.train(train_vecs)
        ivf_idx.add_vectors(data_vecs, chunk_ids)
        # Search with nprobe=16 (scan all Voronoi cells -> should match exact)
        _, ivf_res_all = ivf_idx.search(query_vecs, top_k=1, nprobe=16)
        recall_ivf_all = np.mean(ivf_res_all.ravel() == ground_truth_top1)
        assert recall_ivf_all >= 0.95

        # 3. IVF-PQ
        pq_idx = FAISSIndexer(dim=dim, index_type="IVF-PQ", nlist=16, m_subquantizers=48, bits_per_code=8)
        pq_idx.train(train_vecs)
        pq_idx.add_vectors(data_vecs, chunk_ids)
        _, pq_res = pq_idx.search(query_vecs, top_k=1, nprobe=16)
        recall_pq = np.mean(pq_res.ravel() == ground_truth_top1)

        # Quantization trade-off: PQ recall should be reasonable (>0.6) but typically <= Flat
        assert recall_pq >= 0.60
        assert recall_pq <= 1.0


class TestResumableIndexing:
    """Tests checkpoint integration for resuming interrupted shard indexing."""

    def test_checkpoint_skips_completed_shards(self, tmp_path):
        shard1 = tmp_path / "embeddings_0000.parquet"
        shard2 = tmp_path / "embeddings_0001.parquet"
        _create_synthetic_embedding_shard(shard1, num_vectors=25, start_id=4000)
        _create_synthetic_embedding_shard(shard2, num_vectors=25, start_id=5000)

        ckpt_file = tmp_path / "ckpt_faiss.json"
        ckpt = PipelineCheckpoint(str(ckpt_file))

        indexer = FAISSIndexer(dim=384, index_type="FlatIP")

        # Run 1: process only shard 1
        stats1 = indexer.add_from_shards([shard1], checkpoint=ckpt)
        assert stats1["shards_processed"] == 1
        assert stats1["shards_skipped"] == 0
        assert indexer.ntotal == 25

        # Run 2: process shard 1 and shard 2 (shard 1 should be skipped)
        stats2 = indexer.add_from_shards([shard1, shard2], checkpoint=ckpt)
        assert stats2["shards_processed"] == 1
        assert stats2["shards_skipped"] == 1
        assert indexer.ntotal == 50
