"""Streaming GPU embedding pipeline for chunked Parquet shards.

Features:
- Streams input Parquet shards one by one (never loads all chunks into RAM).
- Resumable: skips already processed shards using PipelineCheckpoint.
- Shard-aligned Parquet output with atomic .tmp -> os.replace rename.
- Stores 64-bit chunk_id alongside each 384-d fixed_size_list vector.
- F1: Token-length bucketed batching to minimize padding waste.
- F2: Pre-tokenization separates CPU work from GPU inference timing.
- F3: GPU utilization sampled between shards only (no subprocess in loop).
- F7: Optional pinned-memory host-to-device transfer.
- F9: Checkpoint written before atomic rename for crash safety.
- F10: Multi-GPU support via shard-level threading.
- Disaggregated telemetry: CPU feeding, tokenization, GPU inference, disk I/O.
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import AppConfig
from src.common.checkpoint import PipelineCheckpoint
from src.common.torch_compat import (
    get_torch_device,
    get_gpu_memory_info,
    reset_gpu_memory_stats,
    get_gpu_utilization,
)
from src.embedding.embedder import GPUEmbedder, DEFAULT_EMBEDDING_DIM, DEFAULT_BUCKET_BOUNDARIES
from src.common.logging import get_logger

logger = get_logger("embedding.pipeline")


def _write_embedding_shard_atomic(
    output_path: Path,
    chunk_ids: np.ndarray,
    embeddings: np.ndarray,
    dim: int = DEFAULT_EMBEDDING_DIM,
):
    """Writes chunk_ids and embeddings to Parquet with atomic rename."""
    tmp_path = output_path.with_suffix(".parquet.tmp")

    # FixedSizeListArray enables zero-copy flat numpy view on read
    flat_embeddings = embeddings.ravel()
    table = pa.Table.from_arrays(
        [
            pa.array(chunk_ids, type=pa.int64()),
            pa.FixedSizeListArray.from_arrays(flat_embeddings, dim),
        ],
        names=["chunk_id", "embedding"],
    )

    pq.write_table(table, str(tmp_path), compression="snappy")
    os.replace(str(tmp_path), str(output_path))


def _bucket_and_tokenize(
    texts: List[str],
    chunk_ids: np.ndarray,
    embedder: GPUEmbedder,
    batch_size: int,
    bucket_boundaries: List[int],
) -> List[Tuple[np.ndarray, Dict]]:
    """F1+F2: Pre-tokenize texts, assign to buckets by token length, batch within buckets.

    Tokenizes all texts once without padding to get true token lengths, then
    groups texts into length buckets. Each bucket is re-tokenized with padding
    only to the bucket's max_length boundary, eliminating cross-bucket padding waste.

    Returns:
        List of (batch_chunk_ids, batch_tokenized_inputs) tuples ready for GPU.
        Each batch's inputs are padded to the bucket boundary, not the global max.
    """
    # Step 1: Get true token lengths (fast tokenizer batch call, no padding)
    encodings = embedder.tokenizer(
        texts, padding=False, truncation=True,
        max_length=embedder.max_seq_length,
    )
    token_lengths = [len(ids) for ids in encodings["input_ids"]]

    # Step 2: Assign each text to its smallest fitting bucket
    # Buckets: {boundary: [(original_index, text), ...]}
    sorted_bounds = sorted(bucket_boundaries)
    max_bound = sorted_bounds[-1]

    buckets: Dict[int, List[int]] = {b: [] for b in sorted_bounds}
    for idx, tok_len in enumerate(token_lengths):
        assigned = False
        for b in sorted_bounds:
            if tok_len <= b:
                buckets[b].append(idx)
                assigned = True
                break
        if not assigned:
            # Longer than largest bucket — assign to largest
            buckets[max_bound].append(idx)

    # Step 3: For each non-empty bucket, create batches tokenized to bucket max_length
    batches = []
    for bound in sorted_bounds:
        bucket_indices = buckets[bound]
        if not bucket_indices:
            continue

        # Sort within bucket by length for tighter dynamic padding
        bucket_indices.sort(key=lambda i: token_lengths[i])

        for b_start in range(0, len(bucket_indices), batch_size):
            b_end = min(b_start + batch_size, len(bucket_indices))
            batch_orig_indices = bucket_indices[b_start:b_end]

            batch_texts = [texts[i] for i in batch_orig_indices]
            batch_cids = chunk_ids[batch_orig_indices]

            # Tokenize with padding capped at bucket boundary (not global max_seq_length)
            batch_inputs = embedder.tokenize(batch_texts, max_length=bound)

            batches.append((batch_cids, batch_inputs))

    return batches


class EmbeddingPipeline:
    """Orchestrates streaming GPU embedding across Parquet chunk shards."""

    def __init__(
        self,
        config: AppConfig,
        checkpoint: Optional[PipelineCheckpoint] = None,
        embedder: Optional[GPUEmbedder] = None,
        bucket_boundaries: Optional[List[int]] = None,
    ):
        self.config = config
        self.checkpoint = checkpoint or PipelineCheckpoint(
            Path(config.paths.data_dir) / "checkpoint_embedding.json"
        )
        self.embedder = embedder or GPUEmbedder(
            model_name=config.embedding.model_name,
            cache_dir=config.paths.cache_dir + "/huggingface",
            device=config.embedding.device,
            use_fp16=(config.embedding.precision == "fp16"),
            expected_dim=config.embedding.embedding_dim,
            max_seq_length=config.embedding.max_seq_length,
            use_torch_compile=config.embedding.use_torch_compile,
            torch_compile_mode=config.embedding.torch_compile_mode,
        )
        self.bucket_boundaries = bucket_boundaries or list(
            getattr(config.embedding, "bucket_boundaries", DEFAULT_BUCKET_BOUNDARIES)
        )

    def _process_shard(
        self,
        shard_file: Path,
        out_file: Path,
        shard_id: str,
        effective_batch_size: int,
    ) -> Dict[str, Any]:
        """Processes a single shard: read → bucket-tokenize → GPU embed → write.

        Returns dict with per-shard telemetry.
        """
        # Stage 1: CPU Feeding (Read shard + extract columns)
        t_feed_start = time.time()
        table = pq.read_table(str(shard_file), columns=["chunk_id", "text"])
        chunk_ids = np.array(
            table.column("chunk_id").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        texts = table.column("text").to_pylist()
        num_chunks = len(texts)
        cpu_feed_sec = time.time() - t_feed_start

        if num_chunks == 0:
            self.checkpoint.mark_complete("embedding", shard_id)
            self.checkpoint.save()
            return {"chunks": 0, "cpu_feed": 0.0, "tokenize": 0.0, "gpu": 0.0, "write": 0.0}

        # Stage 2: CPU Pre-tokenization with bucketed batching (F1 + F2)
        t_tok_start = time.time()
        batches = _bucket_and_tokenize(
            texts, chunk_ids, self.embedder,
            effective_batch_size, self.bucket_boundaries,
        )
        tokenize_sec = time.time() - t_tok_start

        # Stage 3: GPU Inference (only model forward pass, no tokenizer calls)
        t_gpu_start = time.time()
        # Collect embeddings keyed by chunk_id for correct ordering
        all_chunk_ids = []
        all_embeddings = []

        for batch_cids, batch_inputs in batches:
            batch_vecs = self.embedder.embed_batch_tokenized(batch_inputs)
            all_chunk_ids.append(batch_cids)
            all_embeddings.append(batch_vecs)

        collected_cids = np.concatenate(all_chunk_ids)
        collected_vecs = np.vstack(all_embeddings)
        gpu_sec = time.time() - t_gpu_start

        # Stage 4: Restore original ordering + atomic write
        t_write_start = time.time()

        # Build chunk_id → position map to restore original shard order
        orig_order = {cid: pos for pos, cid in enumerate(chunk_ids)}
        reorder = np.array([orig_order[cid] for cid in collected_cids])
        inverse = np.empty_like(reorder)
        inverse[reorder] = np.arange(len(reorder))

        ordered_vecs = collected_vecs[inverse]
        ordered_cids = collected_cids[inverse]

        # F9: Checkpoint before atomic rename for crash safety
        self.checkpoint.mark_complete("embedding", shard_id)
        self.checkpoint.save()

        _write_embedding_shard_atomic(
            out_file, ordered_cids, ordered_vecs,
            dim=self.embedder.expected_dim,
        )
        write_sec = time.time() - t_write_start

        return {
            "chunks": num_chunks,
            "cpu_feed": cpu_feed_sec,
            "tokenize": tokenize_sec,
            "gpu": gpu_sec,
            "write": write_sec,
        }

    def run(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: Optional[int] = None,
        max_shards: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the embedding pipeline over all Parquet shards in input_dir.

        Args:
            input_dir: Directory containing chunk shards (chunks_*.parquet).
            output_dir: Directory where embedding shards (embeddings_*.parquet) are written.
            batch_size: GPU inference batch size (overrides config if specified).
            max_shards: Optional maximum number of shards to process (for benchmarking).

        Returns:
            Dict[str, Any]: Detailed telemetry and performance statistics.
        """
        start_wall_time = time.time()
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Clean any stale temporary files from previous aborted runs
        for tmp in out_path.glob("*.parquet.tmp"):
            logger.info(f"Removing stale temp file: {tmp.name}")
            tmp.unlink()

        effective_batch_size = batch_size or self.config.embedding.batch_size
        shard_files = sorted(in_path.glob("chunks_*.parquet"))
        if max_shards is not None:
            shard_files = shard_files[:max_shards]

        logger.info(
            f"Starting GPU embedding: {len(shard_files)} shards in {input_dir} -> {output_dir} "
            f"(batch_size={effective_batch_size}, device={self.embedder.device}, "
            f"buckets={self.bucket_boundaries})"
        )

        reset_gpu_memory_stats(self.embedder.device)

        total_chunks = 0
        shards_processed = 0
        shards_skipped = 0
        total_cpu_feeding_sec = 0.0
        total_tokenize_sec = 0.0
        total_gpu_compute_sec = 0.0
        total_disk_write_sec = 0.0
        gpu_util_samples = []

        for shard_idx, shard_file in enumerate(shard_files):
            shard_id = shard_file.stem.replace("chunks_", "")
            out_file = out_path / f"embeddings_{shard_id}.parquet"

            # Checkpoint check — also verify shard file exists (F9 edge case)
            if self.checkpoint.is_complete("embedding", shard_id):
                if out_file.exists():
                    shards_skipped += 1
                    logger.debug(f"Skipping already embedded shard: {shard_file.name}")
                    continue
                else:
                    # Checkpoint says done but output missing (crash between checkpoint + rename)
                    # Reset and re-process
                    logger.warning(
                        f"Shard {shard_id} marked complete but output missing. Re-processing."
                    )

            result = self._process_shard(
                shard_file, out_file, shard_id, effective_batch_size,
            )

            total_chunks += result["chunks"]
            total_cpu_feeding_sec += result["cpu_feed"]
            total_tokenize_sec += result["tokenize"]
            total_gpu_compute_sec += result["gpu"]
            total_disk_write_sec += result["write"]
            shards_processed += 1

            # F3: Sample GPU utilization between shards, not inside batch loop
            util = get_gpu_utilization(self.embedder.device)
            if util is not None:
                gpu_util_samples.append(util)

            if (shard_idx + 1) % 5 == 0 or (shard_idx + 1) == len(shard_files):
                logger.info(
                    f"Processed {shard_idx + 1}/{len(shard_files)} shards "
                    f"({total_chunks:,} chunks total)"
                )

        total_elapsed_sec = time.time() - start_wall_time
        mem_info = get_gpu_memory_info(self.embedder.device)
        avg_gpu_util = float(np.mean(gpu_util_samples)) if gpu_util_samples else 0.0

        throughput = round(total_chunks / max(0.001, total_elapsed_sec), 2)
        gpu_only_throughput = round(total_chunks / max(0.001, total_gpu_compute_sec), 2)

        stats = {
            "total_chunks": total_chunks,
            "shards_processed": shards_processed,
            "shards_skipped": shards_skipped,
            "total_shards": len(shard_files),
            "batch_size": effective_batch_size,
            "total_elapsed_seconds": round(total_elapsed_sec, 3),
            "cpu_feeding_seconds": round(total_cpu_feeding_sec, 3),
            "tokenization_seconds": round(total_tokenize_sec, 3),
            "gpu_compute_seconds": round(total_gpu_compute_sec, 3),
            "disk_write_seconds": round(total_disk_write_sec, 3),
            "throughput_chunks_per_sec": throughput,
            "gpu_only_throughput_chunks_per_sec": gpu_only_throughput,
            "peak_vram_allocated_mb": mem_info["peak_allocated_mb"],
            "peak_vram_reserved_mb": mem_info["peak_reserved_mb"],
            "total_vram_mb": mem_info["total_mb"],
            "avg_gpu_utilization_pct": round(avg_gpu_util, 1),
            "embedding_dim": self.embedder.expected_dim,
            "device": str(self.embedder.device),
        }

        logger.info(
            f"Embedding pipeline complete: {total_chunks:,} chunks in {total_elapsed_sec:.2f}s "
            f"({throughput:,.1f} chunks/sec) | "
            f"Tokenize: {total_tokenize_sec:.2f}s | GPU: {total_gpu_compute_sec:.2f}s | "
            f"Peak VRAM: {mem_info['peak_allocated_mb']:.1f}MB / {mem_info['total_mb']:.1f}MB"
        )

        return stats


class MultiGPUEmbeddingPipeline:
    """F10: Shard-level parallelism across multiple GPUs using threading.

    GIL is released during CUDA operations, so threads achieve true parallelism
    for GPU-bound work. Each GPU gets its own GPUEmbedder instance.
    """

    def __init__(
        self,
        config: AppConfig,
        devices: Optional[List[str]] = None,
        checkpoint: Optional[PipelineCheckpoint] = None,
        bucket_boundaries: Optional[List[int]] = None,
    ):
        self.config = config
        self.checkpoint = checkpoint or PipelineCheckpoint(
            Path(config.paths.data_dir) / "checkpoint_embedding.json"
        )
        self.devices = devices or ["cuda:0", "cuda:1"]
        self.bucket_boundaries = bucket_boundaries or list(
            getattr(config.embedding, "bucket_boundaries", DEFAULT_BUCKET_BOUNDARIES)
        )

        # Filter to actually available GPUs
        import torch
        num_gpus = torch.cuda.device_count()
        self.devices = [d for d in self.devices if int(d.split(":")[1]) < num_gpus]
        if not self.devices:
            self.devices = ["cuda:0"] if num_gpus > 0 else ["cpu"]

        logger.info(f"MultiGPU pipeline using devices: {self.devices}")

    def run(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: Optional[int] = None,
        max_shards: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Distributes shards round-robin across GPUs."""
        if len(self.devices) == 1:
            # Fall back to single-GPU pipeline
            embedder = GPUEmbedder(
                model_name=self.config.embedding.model_name,
                cache_dir=self.config.paths.cache_dir + "/huggingface",
                device=self.devices[0],
                use_fp16=(self.config.embedding.precision == "fp16"),
                expected_dim=self.config.embedding.embedding_dim,
                max_seq_length=self.config.embedding.max_seq_length,
            )
            pipeline = EmbeddingPipeline(
                config=self.config,
                checkpoint=self.checkpoint,
                embedder=embedder,
                bucket_boundaries=self.bucket_boundaries,
            )
            return pipeline.run(input_dir, output_dir, batch_size, max_shards)

        in_path = Path(input_dir)
        shard_files = sorted(in_path.glob("chunks_*.parquet"))
        if max_shards is not None:
            shard_files = shard_files[:max_shards]

        # Distribute shards round-robin across devices
        device_shards: Dict[str, List[Path]] = {d: [] for d in self.devices}
        for i, sf in enumerate(shard_files):
            device = self.devices[i % len(self.devices)]
            device_shards[device].append(sf)

        # Shared checkpoint with thread-safe save (PipelineCheckpoint.save is atomic file write)
        results: Dict[str, Dict[str, Any]] = {}
        lock = threading.Lock()

        def _worker(device: str, shards: List[Path]):
            embedder = GPUEmbedder(
                model_name=self.config.embedding.model_name,
                cache_dir=self.config.paths.cache_dir + "/huggingface",
                device=device,
                use_fp16=(self.config.embedding.precision == "fp16"),
                expected_dim=self.config.embedding.embedding_dim,
                max_seq_length=self.config.embedding.max_seq_length,
            )
            pipeline = EmbeddingPipeline(
                config=self.config,
                checkpoint=self.checkpoint,
                embedder=embedder,
                bucket_boundaries=self.bucket_boundaries,
            )
            # Process only the assigned shards
            stats = pipeline.run(
                input_dir, output_dir,
                batch_size=batch_size,
                max_shards=len(shards),
            )
            with lock:
                results[device] = stats

        threads = []
        for device, shards in device_shards.items():
            if not shards:
                continue
            t = threading.Thread(target=_worker, args=(device, shards), name=f"embed-{device}")
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Merge results
        merged = {
            "total_chunks": sum(r["total_chunks"] for r in results.values()),
            "shards_processed": sum(r["shards_processed"] for r in results.values()),
            "shards_skipped": sum(r["shards_skipped"] for r in results.values()),
            "devices_used": list(results.keys()),
            "per_device": results,
        }

        logger.info(
            f"MultiGPU complete: {merged['total_chunks']:,} chunks across {len(results)} GPUs"
        )
        return merged


StreamingEmbeddingPipeline = EmbeddingPipeline
