"""Chunking Throughput & Worker Scaling Benchmark."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
from src.config import AppConfig
from src.ingestion.pipeline import IngestionPipeline
from src.common.checkpoint import PipelineCheckpoint
from src.common.logging import get_logger

logger = get_logger("scripts.benchmark_chunking")


def benchmark_chunking(
    input_parquet: Path,
    output_dir: Path,
    workers: int = 16,
    max_articles: int = 50000,
) -> Dict[str, Any]:
    """Runs a timed benchmark of the chunking pipeline on the pilot corpus."""
    logger.info(f"Starting chunking benchmark on {input_parquet} with {workers} CPU workers...")

    # Load configuration
    cfg = AppConfig.load_from_dir("configs")
    cfg.ingestion.num_cpu_workers = workers

    # Fresh checkpoint for isolated benchmarking
    bench_checkpoint_path = output_dir / "benchmark_checkpoint.json"
    if bench_checkpoint_path.exists():
        bench_checkpoint_path.unlink()
    checkpoint = PipelineCheckpoint(str(bench_checkpoint_path))

    pipeline = IngestionPipeline(config=cfg, checkpoint=checkpoint)

    start_time = time.time()
    stats = pipeline.run(input_parquet=str(input_parquet), output_dir=str(output_dir))
    elapsed = time.time() - start_time

    # Inspect emitted output shards
    shard_files = sorted(list(output_dir.glob("chunks_*.parquet")))
    total_chunks = 0
    total_tokens = 0
    total_bytes = 0

    for s in shard_files:
        total_bytes += s.stat().st_size
        pq_meta = pq.ParquetFile(str(s)).metadata
        total_chunks += pq_meta.num_rows

    articles_processed = stats.get("articles_processed", max_articles)
    art_rate = round(articles_processed / max(0.01, elapsed), 1)
    chunk_rate = round(total_chunks / max(0.01, elapsed), 1)

    result = {
        "benchmark_meta": {
            "input_corpus": input_parquet.name,
            "workers": workers,
            "target_tokens": cfg.ingestion.chunking.target_token_count,
            "overlap_tokens": cfg.ingestion.chunking.overlap_token_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "performance": {
            "elapsed_seconds": round(elapsed, 2),
            "articles_processed": articles_processed,
            "throughput_articles_per_sec": art_rate,
            "total_chunks_emitted": total_chunks,
            "throughput_chunks_per_sec": chunk_rate,
            "avg_chunks_per_article": round(total_chunks / max(1, articles_processed), 2),
            "total_output_mb": round(total_bytes / (1024 * 1024), 2),
            "output_shards_count": len(shard_files),
        },
    }

    # Save benchmark JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 70)
    logger.info("              CHUNKING PIPELINE THROUGHPUT BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Workers:             {workers} CPU cores (NUMA-aware)")
    logger.info(f"Elapsed Time:        {elapsed:.2f} seconds")
    logger.info(f"Articles Processed:  {articles_processed:,} ({art_rate:,} articles/sec)")
    logger.info(f"Chunks Emitted:      {total_chunks:,} ({chunk_rate:,} chunks/sec)")
    logger.info(f"Avg Chunks / Doc:    {total_chunks / max(1, articles_processed):.2f}")
    logger.info(f"Total Output Size:   {total_bytes / (1024*1024):.2f} MB across {len(shard_files)} shards")
    logger.info("=" * 70)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Wikipedia chunking pipeline throughput")
    parser.add_argument("--input", type=str, default="data/raw/wikipedia_en_pilot_50k.parquet", help="Path to input Parquet")
    parser.add_argument("--output-dir", type=str, default="data/chunks_bench", help="Output directory for chunks")
    parser.add_argument("--workers", type=int, default=32, help="Number of CPU worker processes")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_chunking(
        input_parquet=input_path,
        output_dir=output_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

