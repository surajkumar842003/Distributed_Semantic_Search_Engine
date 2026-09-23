"""GPU Embedding Benchmark across multiple batch sizes.

Measures:
- Chunks per second (end-to-end and GPU-only)
- Peak VRAM allocated and reserved
- CPU feeding time vs. GPU inference time vs. disk I/O time
- Average GPU utilization
- Generates JSON and CSV benchmark summaries

Usage:
    python scripts/benchmark_embedding.py \
        --input-dir data/chunks_pilot \
        --output-dir data/benchmark_embedding \
        --batch-sizes 64 128 256 512 \
        --max-shards 2
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.common.checkpoint import PipelineCheckpoint
from src.embedding.embedder import GPUEmbedder
from src.embedding.pipeline import EmbeddingPipeline
from src.common.logging import get_logger

logger = get_logger("scripts.benchmark_embedding")


def run_single_benchmark(
    embedder: GPUEmbedder,
    cfg: AppConfig,
    input_dir: Path,
    output_root: Path,
    batch_size: int,
    max_shards: int,
) -> Dict[str, Any]:
    """Runs embedding benchmark for a single batch size."""
    run_dir = output_root / f"bs_{batch_size:04d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "checkpoint.json"
    checkpoint = PipelineCheckpoint(str(ckpt_path))

    pipeline = EmbeddingPipeline(
        config=cfg,
        checkpoint=checkpoint,
        embedder=embedder,
    )

    logger.info(f"{'='*65}")
    logger.info(f"  BENCHMARK EMBEDDING: batch_size={batch_size}, max_shards={max_shards}")
    logger.info(f"{'='*65}")

    stats = pipeline.run(
        input_dir=str(input_dir),
        output_dir=str(run_dir),
        batch_size=batch_size,
        max_shards=max_shards,
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description="GPU Embedding Scaling Benchmark")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/chunks_pilot",
        help="Input directory containing Parquet chunks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark_embedding",
        help="Root directory for benchmark outputs and reports",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="List of batch sizes to benchmark",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=2,
        help="Number of shards to process per batch size run",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_sizes = sorted(args.batch_sizes)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    cfg = AppConfig.load_from_dir("configs")

    # Initialize shared embedder once
    logger.info("Initializing GPUEmbedder...")
    embedder = GPUEmbedder(
        model_name=cfg.embedding.model_name,
        cache_dir=cfg.paths.cache_dir + "/huggingface",
        device=cfg.embedding.device,
        use_fp16=(cfg.embedding.precision == "fp16"),
        expected_dim=cfg.embedding.embedding_dim,
        max_seq_length=cfg.embedding.max_seq_length,
    )

    # Warmup GPU
    logger.info("Running GPU warmup...")
    embedder.embed_batch(["Warmup sentence for GPU clock initialization."] * 32)

    all_results: List[Dict[str, Any]] = []

    for bs in batch_sizes:
        try:
            stats = run_single_benchmark(
                embedder=embedder,
                cfg=cfg,
                input_dir=input_dir,
                output_root=output_dir,
                batch_size=bs,
                max_shards=args.max_shards,
            )
            all_results.append(stats)
            logger.info(
                f"batch_size={bs:4d} | throughput={stats['throughput_chunks_per_sec']:,.1f} ch/s "
                f"(GPU-only: {stats['gpu_only_throughput_chunks_per_sec']:,.1f} ch/s) | "
                f"Peak VRAM: {stats['peak_vram_allocated_mb']:.1f} MB | "
                f"Elapsed: {stats['total_elapsed_seconds']:.2f}s"
            )
        except Exception as e:
            logger.error(f"Benchmark failed for batch_size={bs}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        logger.error("No benchmark results collected!")
        sys.exit(1)

    # --- Print Summary Table ---
    print("\n")
    print("=" * 115)
    print("                      GPU EMBEDDING SERVICE BENCHMARK (NVIDIA L4 24GB)")
    print("=" * 115)
    header = (
        f"{'Batch Size':>10} | {'Chunks':>8} | {'Elapsed(s)':>10} | {'Feeding(s)':>10} | "
        f"{'Tok(s)':>8} | {'GPU(s)':>8} | {'Total ch/s':>12} | {'GPU ch/s':>12} | "
        f"{'VRAM (MB)':>10} | {'GPU Util':>8}"
    )
    print(header)
    print("-" * 130)

    for r in all_results:
        print(
            f"{r['batch_size']:>10d} | "
            f"{r['total_chunks']:>8,d} | "
            f"{r['total_elapsed_seconds']:>10.2f} | "
            f"{r['cpu_feeding_seconds']:>10.2f} | "
            f"{r.get('tokenization_seconds', 0.0):>8.2f} | "
            f"{r['gpu_compute_seconds']:>8.2f} | "
            f"{r['throughput_chunks_per_sec']:>12,.1f} | "
            f"{r['gpu_only_throughput_chunks_per_sec']:>12,.1f} | "
            f"{r['peak_vram_allocated_mb']:>10.1f} | "
            f"{r['avg_gpu_utilization_pct']:>7.1f}%"
        )
    print("=" * 115)
    print()

    # --- Save Consolidated JSON and CSV ---
    json_path = output_dir / "embedding_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hardware": {
                    "gpu": "NVIDIA L4",
                    "device": str(embedder.device),
                    "vram_total_mb": all_results[0]["total_vram_mb"],
                },
                "model": {
                    "name": cfg.embedding.model_name,
                    "dimension": embedder.expected_dim,
                    "precision": cfg.embedding.precision,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved JSON report: {json_path}")

    csv_path = output_dir / "embedding_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "total_chunks",
                "total_elapsed_seconds",
                "cpu_feeding_seconds",
                "tokenization_seconds",
                "gpu_compute_seconds",
                "disk_write_seconds",
                "throughput_chunks_per_sec",
                "gpu_only_throughput_chunks_per_sec",
                "peak_vram_allocated_mb",
                "peak_vram_reserved_mb",
                "avg_gpu_utilization_pct",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "batch_size": r["batch_size"],
                "total_chunks": r["total_chunks"],
                "total_elapsed_seconds": r["total_elapsed_seconds"],
                "cpu_feeding_seconds": r["cpu_feeding_seconds"],
                "tokenization_seconds": r.get("tokenization_seconds", 0.0),
                "gpu_compute_seconds": r["gpu_compute_seconds"],
                "disk_write_seconds": r["disk_write_seconds"],
                "throughput_chunks_per_sec": r["throughput_chunks_per_sec"],
                "gpu_only_throughput_chunks_per_sec": r["gpu_only_throughput_chunks_per_sec"],
                "peak_vram_allocated_mb": r["peak_vram_allocated_mb"],
                "peak_vram_reserved_mb": r["peak_vram_reserved_mb"],
                "avg_gpu_utilization_pct": r["avg_gpu_utilization_pct"],
            })
    logger.info(f"Saved CSV summary: {csv_path}")


if __name__ == "__main__":
    main()

