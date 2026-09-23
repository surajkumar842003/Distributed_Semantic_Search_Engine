"""Async producer-consumer pipeline for CPU ingestion — v2.

Architecture:
  Stage 1 (Reader)  – main process reads Parquet row-groups, sub-batches them
                       into configurable slices, pushes to queue_tasks
  Stage 2 (Workers) – N chunking workers pull from queue_tasks, chunk articles,
                       write Parquet shards directly to disk, report completion
                       via a lightweight result queue

Fixes applied (see cpu_ingestion_review.md):
  F1  Sub-batched work items: eliminates worker starvation
  F2  No output queue:        eliminates chunk pickle overhead
  F4  Per-worker shard writes: eliminates single-writer bottleneck
  F5  Clean shutdown:          eliminates deadlock vector
  F6  Atomic shard writes:     .tmp → os.replace prevents corrupt files
  F7  Pre-fork tokenizer:      avoids N redundant tokenizer loads
  F8  Disaggregated timing:    separates reader / compute / join phases
  F9  Resource cleanup:        explicit queue close + file handle release
"""

import multiprocessing as mp
import os
import pyarrow as pa
import pyarrow.parquet as pq
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any

from src.config import AppConfig
from src.common.checkpoint import PipelineCheckpoint
from src.common.schemas import RawArticle, DocumentChunk
from src.ingestion.chunker import HeadingAwareChunker
from src.common.logging import get_logger

logger = get_logger("ingestion.pipeline")

# ---------------------------------------------------------------------------
# Pre-fork tokenizer (F7)
# ---------------------------------------------------------------------------

_SHARED_TOKENIZER = None


def _preload_tokenizer(model_name: str):
    """Load tokenizer in the main process before forking workers.

    Workers inherit the loaded tokenizer via fork's copy-on-write pages,
    avoiding N independent disk reads and Rust heap allocations.
    """
    global _SHARED_TOKENIZER
    if _SHARED_TOKENIZER is None:
        from src.common.tokenizer import get_tokenizer
        _SHARED_TOKENIZER = get_tokenizer(model_name)
    return _SHARED_TOKENIZER


# ---------------------------------------------------------------------------
# Atomic shard writing (F6)
# ---------------------------------------------------------------------------

def _chunks_to_table(chunks: List[DocumentChunk]) -> pa.Table:
    """Convert DocumentChunk list to a PyArrow Table."""
    return pa.table({
        "chunk_id":     pa.array([int(c.chunk_id) for c in chunks], type=pa.int64()),
        "doc_id":       pa.array([int(c.doc_id) for c in chunks], type=pa.int64()),
        "title":        pa.array([str(c.title) for c in chunks], type=pa.string()),
        "url":          pa.array([str(c.url) for c in chunks], type=pa.string()),
        "section_path": pa.array([str(c.section_path) for c in chunks], type=pa.string()),
        "text":         pa.array([str(c.text) for c in chunks], type=pa.string()),
        "token_count":  pa.array([int(c.token_count) for c in chunks], type=pa.int32()),
        "faiss_id":     pa.array([int(c.faiss_id) for c in chunks], type=pa.int64()),
    })


def _write_shard_atomic(output_dir: str, batch_id: str, chunks: List[DocumentChunk]):
    """Write chunks to a Parquet shard with atomic rename.

    Writes to a .tmp file first; a crash during write leaves only
    the .tmp file, which is cleaned up on the next run.
    """
    out_path = Path(output_dir)
    file_path = out_path / f"chunks_{batch_id}.parquet"
    tmp_path = out_path / f"chunks_{batch_id}.parquet.tmp"

    table = _chunks_to_table(chunks)
    pq.write_table(table, str(tmp_path), compression="snappy")
    os.replace(str(tmp_path), str(file_path))   # atomic on POSIX

    logger.debug(f"Wrote shard {file_path.name} ({len(chunks):,} chunks)")


# ---------------------------------------------------------------------------
# Stage 2: Worker Process (F1/F2/F4/F6)
# ---------------------------------------------------------------------------

def _worker_process(
    worker_id: int,
    queue_tasks: mp.Queue,
    queue_done: mp.Queue,
    output_dir: str,
    target_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
):
    """Worker: pull sub-batches, chunk articles, write shards directly."""
    chunker = HeadingAwareChunker(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        tokenizer=_SHARED_TOKENIZER,     # inherited via fork (F7)
    )

    while True:
        item = queue_tasks.get()
        if item is None:
            break                         # sentinel received

        batch_id, articles = item
        all_chunks: List[DocumentChunk] = []
        failures = 0

        for art in articles:
            try:
                chunks = chunker.chunk_article(art)
                all_chunks.extend(chunks)
            except Exception:
                failures += 1
                logger.warning(
                    f"Worker {worker_id}: failed to chunk doc_id={art.id} "
                    f"title={art.title!r}: {traceback.format_exc(limit=2)}"
                )

        # Write shard atomically — one shard per sub-batch, deterministic
        # filename means re-processing the same batch on resume is idempotent
        if all_chunks:
            _write_shard_atomic(output_dir, batch_id, all_chunks)

        # Report completion: lightweight tuple, no bulk data (F2)
        queue_done.put((batch_id, len(articles), len(all_chunks), failures))

    # Sentinel: worker finished
    queue_done.put(None)
    logger.debug(f"Worker {worker_id} exiting.")


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """Multiprocessing ingestion pipeline with sub-batching and per-worker writes.

    Returns a comprehensive stats dictionary from ``run()`` including:
    - articles_processed, chunks_emitted, failed_records
    - elapsed_seconds, reader_seconds, compute_seconds
    - batches_enqueued, batches_skipped (checkpoint resumability)
    """

    def __init__(self, config: AppConfig, checkpoint: PipelineCheckpoint):
        self.config = config
        self.checkpoint = checkpoint

    def run(self, input_parquet: str, output_dir: str) -> Dict[str, Any]:
        """Run the pipeline.  Returns a stats dict."""
        start_time = time.time()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # F6: Remove stale .tmp files from previous crashes
        for tmp in out_path.glob("*.parquet.tmp"):
            logger.info(f"Removing stale temp file: {tmp.name}")
            tmp.unlink()

        # F7: Pre-load tokenizer before forking workers
        _preload_tokenizer(self.config.ingestion.chunking.tokenizer_name)

        num_workers = self.config.ingestion.num_cpu_workers
        sub_batch_size = self.config.ingestion.worker_chunksize

        # Queue sizing: just enough to keep workers fed, small enough
        # that backpressure kicks in before memory blows up.
        max_task_queue = max(num_workers * 4, 64)
        queue_tasks: mp.Queue = mp.Queue(maxsize=max_task_queue)
        queue_done: mp.Queue = mp.Queue()     # lightweight messages only

        # --- Start workers --------------------------------------------------
        workers: List[mp.Process] = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker_process,
                args=(
                    i, queue_tasks, queue_done, output_dir,
                    self.config.ingestion.chunking.target_token_count,
                    self.config.ingestion.chunking.overlap_token_count,
                    self.config.ingestion.chunking.min_token_count,
                ),
            )
            p.start()
            workers.append(p)

        # --- Stage 1: Reader (main process) — sub-batched (F1) --------------
        logger.info(f"Reading Parquet from {input_parquet}")
        pq_file = pq.ParquetFile(input_parquet)
        columns = ["id", "title", "url", "text"]
        has_categories = "categories" in pq_file.schema.names
        cols_to_read = columns + (["categories"] if has_categories else [])

        total_articles_enqueued = 0
        total_row_groups = pq_file.num_row_groups
        batches_enqueued = 0
        batches_skipped = 0

        for rg_idx in range(total_row_groups):
            table = pq_file.read_row_group(rg_idx, columns=cols_to_read)

            for offset in range(0, table.num_rows, sub_batch_size):
                batch_id = f"rg{rg_idx:03d}_b{offset:06d}"

                if self.checkpoint.is_complete("ingestion", batch_id):
                    batches_skipped += 1
                    continue

                # Zero-copy slice of the row-group Arrow table
                batch = table.slice(offset, min(sub_batch_size, table.num_rows - offset))
                ids    = batch.column("id").to_pylist()
                titles = batch.column("title").to_pylist()
                urls   = batch.column("url").to_pylist()
                texts  = batch.column("text").to_pylist()
                cats   = (batch.column("categories").to_pylist()
                          if has_categories else [[]] * len(ids))

                articles = [
                    RawArticle(
                        id=int(ids[i]) if (isinstance(ids[i], int)
                            or (isinstance(ids[i], str) and ids[i].isdigit()))
                            else (offset + i + 1),
                        title=titles[i] or "",
                        url=urls[i] or "",
                        text=texts[i] or "",
                        categories=cats[i] or [],
                    )
                    for i in range(len(ids))
                ]

                queue_tasks.put((batch_id, articles))
                total_articles_enqueued += len(articles)
                batches_enqueued += 1

        del pq_file                        # F9: release file handle
        reader_elapsed = time.time() - start_time

        # Send sentinels to workers
        for _ in range(num_workers):
            queue_tasks.put(None)

        # --- Collect results from workers -----------------------------------
        workers_finished = 0
        total_articles = 0
        total_chunks = 0
        total_failures = 0

        while workers_finished < num_workers:
            item = queue_done.get(timeout=600)
            if item is None:
                workers_finished += 1
                continue
            batch_id, n_articles, n_chunks, n_failures = item
            total_articles += n_articles
            total_chunks += n_chunks
            total_failures += n_failures
            # Checkpoint immediately after each batch (safe: shard is already
            # atomically on disk when the worker sent this message)
            self.checkpoint.mark_complete("ingestion", batch_id)
            self.checkpoint.save()

        compute_elapsed = time.time() - start_time - reader_elapsed

        # --- F5: Join workers with timeout ----------------------------------
        for p in workers:
            p.join(timeout=60)
            if p.is_alive():
                logger.error(f"Worker {p.pid} did not exit, terminating")
                p.terminate()
                p.join(timeout=5)

        # --- F9: Clean up queues --------------------------------------------
        queue_tasks.close()
        queue_tasks.join_thread()
        queue_done.close()
        queue_done.join_thread()

        elapsed = time.time() - start_time

        # --- Build stats dict -----------------------------------------------
        stats: Dict[str, Any] = {
            "elapsed_seconds":             round(elapsed, 3),
            "reader_seconds":              round(reader_elapsed, 3),
            "compute_seconds":             round(compute_elapsed, 3),
            "articles_enqueued":           total_articles_enqueued,
            "articles_processed":          total_articles,
            "chunks_emitted":              total_chunks,
            "failed_records":              total_failures,
            "row_groups_total":            total_row_groups,
            "batches_enqueued":            batches_enqueued,
            "batches_skipped":             batches_skipped,
            "num_workers":                 num_workers,
            "throughput_articles_per_sec":  round(total_articles / max(0.01, elapsed), 2),
            "throughput_chunks_per_sec":    round(total_chunks / max(0.01, elapsed), 2),
        }

        logger.info(
            f"Pipeline complete: {total_articles:,} articles → "
            f"{total_chunks:,} chunks in {elapsed:.1f}s "
            f"({stats['throughput_articles_per_sec']:,.1f} art/s, "
            f"{stats['throughput_chunks_per_sec']:,.1f} chunk/s) "
            f"| failures={total_failures}"
        )

        return stats
