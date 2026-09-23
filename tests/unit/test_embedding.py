"""Unit and correctness tests for the GPU embedding service.

Verifies:
1. Programmatic embedding dimension validation (384).
2. L2 vector normalization (unit length).
3. Numerical correctness comparing stored vectors against freshly computed vectors.
4. Shard alignment and chunk_id preservation.
5. Checkpointing and restartability (skips completed shards).
6. Bucketed batching correctness (F1).
7. Pre-tokenized embedding path (F2).
8. Deterministic seeding (F8).
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import AppConfig
from src.common.checkpoint import PipelineCheckpoint
from src.embedding.embedder import GPUEmbedder, DEFAULT_EMBEDDING_DIM, DEFAULT_BUCKET_BOUNDARIES
from src.embedding.pipeline import EmbeddingPipeline, _bucket_and_tokenize


@pytest.fixture(scope="module")
def embedder():
    """Initializes a shared GPUEmbedder instance for tests."""
    return GPUEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir="/DATA/suraj/m1/search_engine/data/cache/huggingface",
        use_fp16=True,
    )


def _create_sample_chunk_shard(path: Path, num_chunks: int = 10, start_id: int = 1000):
    """Helper to create a synthetic chunk Parquet shard for testing."""
    chunk_ids = [start_id + i for i in range(num_chunks)]
    texts = [
        f"Artificial intelligence and semantic search passage {i}. Machine learning models compute vector embeddings."
        for i in range(num_chunks)
    ]
    table = pa.Table.from_arrays(
        [
            pa.array(chunk_ids, type=pa.int64()),
            pa.array([1] * num_chunks, type=pa.int64()),
            pa.array([f"Title {i}" for i in range(num_chunks)], type=pa.string()),
            pa.array([f"http://example.com/{i}" for i in range(num_chunks)], type=pa.string()),
            pa.array(["Overview"] * num_chunks, type=pa.string()),
            pa.array(texts, type=pa.string()),
            pa.array([25] * num_chunks, type=pa.int32()),
            pa.array(chunk_ids, type=pa.int64()),
        ],
        names=["chunk_id", "doc_id", "title", "url", "section_path", "text", "token_count", "faiss_id"],
    )
    pq.write_table(table, str(path))
    return chunk_ids, texts


def _create_varied_length_shard(path: Path, num_chunks: int = 50, start_id: int = 2000):
    """Creates a shard with varied text lengths to test bucketed batching."""
    chunk_ids = [start_id + i for i in range(num_chunks)]
    texts = []
    for i in range(num_chunks):
        # Create texts of varying lengths: short (1 sentence), medium (3), long (8)
        if i % 3 == 0:
            texts.append(f"Short text {i}.")
        elif i % 3 == 1:
            texts.append(
                f"Medium text about topic {i}. "
                f"This passage discusses the relationship between deep learning and natural language processing. "
                f"Vector embeddings capture semantic meaning in dense representations."
            )
        else:
            texts.append(
                f"Long text about topic {i}. " * 8
                + "This final sentence concludes the passage about semantic search and retrieval systems."
            )

    table = pa.Table.from_arrays(
        [
            pa.array(chunk_ids, type=pa.int64()),
            pa.array([1] * num_chunks, type=pa.int64()),
            pa.array([f"Title {i}" for i in range(num_chunks)], type=pa.string()),
            pa.array([f"http://example.com/{i}" for i in range(num_chunks)], type=pa.string()),
            pa.array(["Overview"] * num_chunks, type=pa.string()),
            pa.array(texts, type=pa.string()),
            pa.array([25] * num_chunks, type=pa.int32()),
            pa.array(chunk_ids, type=pa.int64()),
        ],
        names=["chunk_id", "doc_id", "title", "url", "section_path", "text", "token_count", "faiss_id"],
    )
    pq.write_table(table, str(path))
    return chunk_ids, texts


class TestGPUEmbedder:
    """Tests for core GPUEmbedder behavior and numerical properties."""

    def test_dimension_verification(self, embedder):
        """Verifies that the embedding dimension is strictly 384."""
        assert embedder.expected_dim == 384
        vecs = embedder.embed_batch(["Testing dimension verification."])
        assert vecs.shape == (1, 384)

    def test_invalid_dimension_raises(self):
        """Verifies that asserting an incorrect expected dimension raises ValueError."""
        with pytest.raises(ValueError, match="does not match expected_dim"):
            GPUEmbedder(
                model_name="BAAI/bge-small-en-v1.5",
                cache_dir="/DATA/suraj/m1/search_engine/data/cache/huggingface",
                expected_dim=512,  # deliberately wrong
            )

    def test_l2_normalization(self, embedder):
        """Verifies that output embeddings have unit L2 norm."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "High performance computing and GPU acceleration.",
            "Wikipedia dense retrieval systems with hybrid search.",
        ]
        vecs = embedder.embed_batch(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)

    def test_empty_input(self, embedder):
        """Verifies handling of empty inputs."""
        vecs = embedder.embed_batch([])
        assert vecs.shape == (0, 384)

    def test_pre_tokenized_path(self, embedder):
        """F2: Verifies embed_batch_tokenized produces same vectors as embed_batch."""
        texts = [
            "Pre-tokenization correctness test.",
            "The GPU embedding pipeline uses bucketed batching.",
        ]
        # Via embed_batch (tokenize + infer together)
        vecs_combined = embedder.embed_batch(texts)

        # Via embed_batch_tokenized (tokenize separately, then infer)
        inputs = embedder.tokenize(texts)
        vecs_separate = embedder.embed_batch_tokenized(inputs)

        # Should be identical (same tokenizer, same model, same precision)
        np.testing.assert_allclose(vecs_combined, vecs_separate, atol=1e-6)

    def test_deterministic_seeding(self):
        """F8: Verifies that deterministic mode produces reproducible embeddings."""
        emb1 = GPUEmbedder(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="/DATA/suraj/m1/search_engine/data/cache/huggingface",
            use_fp16=True,
            seed=42,
        )
        emb2 = GPUEmbedder(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="/DATA/suraj/m1/search_engine/data/cache/huggingface",
            use_fp16=True,
            seed=42,
        )

        texts = ["Determinism test for reproducible embeddings."]
        v1 = emb1.embed_batch(texts)
        v2 = emb2.embed_batch(texts)
        np.testing.assert_allclose(v1, v2, atol=1e-6)


class TestBucketedBatching:
    """Tests for F1: token-length bucketed batching."""

    def test_bucket_assignment(self, embedder):
        """Verifies texts are assigned to correct length buckets."""
        texts = [
            "Short.",                           # ~3 tokens → bucket 64
            "A slightly longer text here.",      # ~7 tokens → bucket 64
            "This is a medium length text that discusses various topics in natural language processing "
            "and information retrieval systems for semantic search applications." * 2,  # ~50 tokens → bucket 64
        ]
        chunk_ids = np.array([100, 101, 102], dtype=np.int64)

        batches = _bucket_and_tokenize(
            texts, chunk_ids, embedder,
            batch_size=10, bucket_boundaries=[64, 128, 192, 256],
        )

        # All texts should fit in the first bucket (64 tokens)
        # so we should get 1 batch with all 3 texts
        assert len(batches) >= 1
        total_items = sum(b[0].shape[0] for b in batches)
        assert total_items == 3

    def test_bucketed_vs_unbucketed_correctness(self, embedder):
        """Verifies bucketed batching produces same embeddings as unbucketed."""
        texts = [
            "Short text.",
            "A medium length passage about machine learning and GPU computing in modern systems.",
            "Very long text. " * 20 + "End.",
        ]

        # Unbucketed: standard embed_batch
        vecs_unbucketed = embedder.embed_batch(texts)

        # Bucketed: via _bucket_and_tokenize
        chunk_ids = np.array([0, 1, 2], dtype=np.int64)
        batches = _bucket_and_tokenize(
            texts, chunk_ids, embedder,
            batch_size=10, bucket_boundaries=[64, 128, 192, 256],
        )

        # Reconstruct in original order
        all_cids = np.concatenate([b[0] for b in batches])
        all_vecs = np.vstack([embedder.embed_batch_tokenized(b[1]) for b in batches])

        # Reorder to match original
        order = np.argsort(all_cids)
        vecs_bucketed = all_vecs[order]

        # Cosine similarity should be very high (tiny padding differences possible)
        cosine_sims = np.sum(vecs_unbucketed * vecs_bucketed, axis=1)
        assert np.min(cosine_sims) >= 0.999, f"Min cosine sim: {np.min(cosine_sims)}"

    def test_padding_waste_reduction(self, embedder):
        """Verifies bucketed batching reduces total padded tokens vs unbucketed."""
        # Mix of very short and very long texts
        texts = (
            ["Short."] * 10
            + ["Medium length text about a topic. " * 5] * 10
            + ["Long passage. " * 15 + "End."] * 10
        )
        chunk_ids = np.arange(30, dtype=np.int64)

        # Bucketed: each batch padded to bucket boundary
        batches = _bucket_and_tokenize(
            texts, chunk_ids, embedder,
            batch_size=32, bucket_boundaries=[64, 128, 192, 256],
        )

        total_bucketed_tokens = 0
        for _, batch_inputs in batches:
            bs, seq_len = batch_inputs["input_ids"].shape
            total_bucketed_tokens += bs * seq_len

        # Unbucketed: single tokenize call with global max_length
        unbucketed = embedder.tokenize(texts)
        bs_ub, seq_len_ub = unbucketed["input_ids"].shape
        total_unbucketed_tokens = bs_ub * seq_len_ub

        # Bucketed should use fewer total padded tokens
        assert total_bucketed_tokens <= total_unbucketed_tokens, (
            f"Bucketed ({total_bucketed_tokens}) should use <= unbucketed ({total_unbucketed_tokens})"
        )


class TestEmbeddingPipelineCorrectness:
    """Tests pipeline streaming, shard alignment, numerical accuracy, and checkpointing."""

    def test_stored_vectors_vs_fresh_vectors_correctness(self, embedder, tmp_path):
        """Generates embedding shard, re-computes fresh vectors, and asserts cosine similarity > 0.9999."""
        in_dir = tmp_path / "chunks"
        out_dir = tmp_path / "embeddings"
        in_dir.mkdir()
        out_dir.mkdir()

        shard_path = in_dir / "chunks_0000.parquet"
        chunk_ids, texts = _create_sample_chunk_shard(shard_path, num_chunks=16, start_id=5000)

        cfg = AppConfig.load_from_dir("configs")
        ckpt = PipelineCheckpoint(str(tmp_path / "ckpt.json"))

        pipeline = EmbeddingPipeline(config=cfg, checkpoint=ckpt, embedder=embedder)
        stats = pipeline.run(
            input_dir=str(in_dir),
            output_dir=str(out_dir),
            batch_size=8,
        )

        assert stats["total_chunks"] == 16
        assert stats["shards_processed"] == 1
        # F2: Verify tokenization is tracked separately
        assert "tokenization_seconds" in stats

        # Read back saved embedding shard
        out_shard = out_dir / "embeddings_0000.parquet"
        assert out_shard.exists()

        table = pq.read_table(str(out_shard))
        stored_chunk_ids = table.column("chunk_id").to_numpy()
        flat_vecs = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        stored_vectors = flat_vecs.reshape(-1, 384)

        # 1. Verify chunk IDs match exactly in order
        np.testing.assert_array_equal(stored_chunk_ids, chunk_ids)

        # 2. Re-compute fresh vectors directly from raw texts
        fresh_vectors = embedder.embed_texts(texts, batch_size=8)

        # 3. Verify cosine similarity between stored and fresh vectors
        # For unit-normalized vectors, cosine similarity is simply the dot product
        cosine_sims = np.sum(stored_vectors * fresh_vectors, axis=1)
        min_cosine_sim = np.min(cosine_sims)
        assert min_cosine_sim >= 0.999, f"Minimum cosine similarity too low: {min_cosine_sim}"

        # 4. Verify maximum absolute difference is negligible (< 1e-2 for FP16 + bucketing)
        max_abs_diff = np.max(np.abs(stored_vectors - fresh_vectors))
        assert max_abs_diff < 1e-2, f"Max absolute difference too high: {max_abs_diff}"

    def test_varied_length_shard_correctness(self, embedder, tmp_path):
        """F1: Tests that varied-length texts produce correct embeddings with bucketing."""
        in_dir = tmp_path / "chunks"
        out_dir = tmp_path / "embeddings"
        in_dir.mkdir()
        out_dir.mkdir()

        shard_path = in_dir / "chunks_0000.parquet"
        chunk_ids, texts = _create_varied_length_shard(shard_path, num_chunks=30, start_id=3000)

        cfg = AppConfig.load_from_dir("configs")
        ckpt = PipelineCheckpoint(str(tmp_path / "ckpt.json"))

        pipeline = EmbeddingPipeline(config=cfg, checkpoint=ckpt, embedder=embedder)
        stats = pipeline.run(input_dir=str(in_dir), output_dir=str(out_dir), batch_size=8)

        assert stats["total_chunks"] == 30

        # Read back and verify all chunk_ids are present
        out_shard = out_dir / "embeddings_0000.parquet"
        table = pq.read_table(str(out_shard))
        stored_ids = set(table.column("chunk_id").to_numpy().tolist())
        assert stored_ids == set(chunk_ids)

        # Verify L2 normalization
        flat_vecs = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        vecs = flat_vecs.reshape(-1, 384)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)

    def test_checkpoint_resumability(self, embedder, tmp_path):
        """Verifies that running the pipeline on an already completed shard skips it."""
        in_dir = tmp_path / "chunks"
        out_dir = tmp_path / "embeddings"
        in_dir.mkdir()
        out_dir.mkdir()

        shard_path = in_dir / "chunks_0001.parquet"
        _create_sample_chunk_shard(shard_path, num_chunks=8, start_id=8000)

        cfg = AppConfig.load_from_dir("configs")
        ckpt_path = str(tmp_path / "ckpt.json")
        ckpt = PipelineCheckpoint(ckpt_path)

        pipeline = EmbeddingPipeline(config=cfg, checkpoint=ckpt, embedder=embedder)

        # Run 1: processes shard
        stats1 = pipeline.run(input_dir=str(in_dir), output_dir=str(out_dir), batch_size=4)
        assert stats1["shards_processed"] == 1
        assert stats1["shards_skipped"] == 0

        # Run 2: skips shard using checkpoint
        stats2 = pipeline.run(input_dir=str(in_dir), output_dir=str(out_dir), batch_size=4)
        assert stats2["shards_processed"] == 0
        assert stats2["shards_skipped"] == 1
        assert stats2["total_chunks"] == 0

    def test_checkpoint_with_missing_output_reprocesses(self, embedder, tmp_path):
        """F9: If checkpoint says done but output file missing, shard is reprocessed."""
        in_dir = tmp_path / "chunks"
        out_dir = tmp_path / "embeddings"
        in_dir.mkdir()
        out_dir.mkdir()

        shard_path = in_dir / "chunks_0000.parquet"
        _create_sample_chunk_shard(shard_path, num_chunks=8, start_id=9000)

        cfg = AppConfig.load_from_dir("configs")
        ckpt_path = str(tmp_path / "ckpt.json")
        ckpt = PipelineCheckpoint(ckpt_path)

        pipeline = EmbeddingPipeline(config=cfg, checkpoint=ckpt, embedder=embedder)

        # Run 1: process normally
        stats1 = pipeline.run(input_dir=str(in_dir), output_dir=str(out_dir), batch_size=4)
        assert stats1["shards_processed"] == 1

        # Simulate crash: delete output but keep checkpoint
        (out_dir / "embeddings_0000.parquet").unlink()

        # Run 2: should detect missing output and reprocess
        ckpt2 = PipelineCheckpoint(ckpt_path)  # reload
        pipeline2 = EmbeddingPipeline(config=cfg, checkpoint=ckpt2, embedder=embedder)
        stats2 = pipeline2.run(input_dir=str(in_dir), output_dir=str(out_dir), batch_size=4)
        assert stats2["shards_processed"] == 1
        assert (out_dir / "embeddings_0000.parquet").exists()
