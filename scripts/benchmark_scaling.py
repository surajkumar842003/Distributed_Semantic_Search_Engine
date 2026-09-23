"""CPU Ingestion Scaling Benchmark.

Iterates over multiple worker counts, runs the full ingestion pipeline for
each, collects throughput / timing / failure statistics, and writes a
consolidated JSON + CSV report with a scaling analysis.

Usage:
    python scripts/benchmark_scaling.py \
        --input data/raw/wikipedia_en_pilot_50k.parquet \
        --output-root data/benchmark_scaling \
        --workers 1 4 8 16 32 64 90
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

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
from src.config import AppConfig
from src.common.checkpoint import PipelineCheckpoint
from src.ingestion.pipeline import IngestionPipeline
from src.common.logging import get_logger

logger = get_logger("scripts.benchmark_scaling")


def _count_output_chunks(output_dir: Path) -> Dict[str, Any]:
    """Inspect emitted chunk shards and return summary."""
    shard_files = sorted(output_dir.glob("chunks_*.parquet"))
    total_chunks = 0
    total_bytes = 0
    for s in shard_files:
        total_bytes += s.stat().st_size
        try:
            meta = pq.ParquetFile(str(s)).metadata
            total_chunks += meta.num_rows
        except Exception:
            logger.warning(f"Could not read shard {s.name}, skipping")
    return {
        "total_chunks": total_chunks,
        "total_output_bytes": total_bytes,
        "total_output_mb": round(total_bytes / (1024 * 1024), 2),
        "output_shards": len(shard_files),
    }


def run_single_benchmark(
    input_parquet: Path,
    output_root: Path,
    num_workers: int,
) -> Dict[str, Any]:
    """Run a single pipeline benchmark at *num_workers* concurrency."""
    run_dir = output_root / f"workers_{num_workers:03d}"

    # Clean previous run artifacts for isolation
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load config and override worker count
    cfg = AppConfig.load_from_dir("configs")
    cfg.ingestion.num_cpu_workers = num_workers

    # Fresh checkpoint per run
    ckpt_path = run_dir / "checkpoint.json"
    checkpoint = PipelineCheckpoint(str(ckpt_path))

    logger.info(f"{'='*60}")
    logger.info(f"  BENCHMARK  workers={num_workers}")
    logger.info(f"{'='*60}")

    pipeline = IngestionPipeline(config=cfg, checkpoint=checkpoint)
    stats = pipeline.run(
        input_parquet=str(input_parquet),
        output_dir=str(run_dir),
    )

    # Augment with output inspection
    output_info = _count_output_chunks(run_dir)
    stats.update(output_info)

    # Persist per-run result
    with open(run_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def compute_scaling_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute speedup, efficiency, and scaling metrics relative to 1-worker baseline."""
    if not results:
        return {}

    # Find the single-worker baseline
    baseline = next((r for r in results if r["num_workers"] == 1), results[0])
    baseline_time = baseline["elapsed_seconds"]
    baseline_throughput = baseline["throughput_articles_per_sec"]

    analysis = []
    for r in results:
        w = r["num_workers"]
        speedup = baseline_time / max(0.01, r["elapsed_seconds"])
        efficiency = speedup / w * 100.0  # percentage
        analysis.append({
            "num_workers":                w,
            "elapsed_seconds":            r["elapsed_seconds"],
            "throughput_articles_per_sec": r["throughput_articles_per_sec"],
            "throughput_chunks_per_sec":   r["throughput_chunks_per_sec"],
            "speedup_vs_1_worker":        round(speedup, 2),
            "parallel_efficiency_pct":    round(efficiency, 1),
            "chunks_emitted":             r.get("chunks_emitted", r.get("total_chunks", 0)),
            "failed_records":             r.get("failed_records", 0),
        })

    return {
        "baseline_workers": baseline["num_workers"],
        "baseline_elapsed_seconds": baseline_time,
        "baseline_throughput_articles_per_sec": baseline_throughput,
        "per_worker_count": analysis,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CPU ingestion scaling benchmark across worker counts"
    )
    parser.add_argument(
        "--input", type=str,
        default="data/raw/wikipedia_en_pilot_50k.parquet",
        help="Path to input Parquet file",
    )
    parser.add_argument(
        "--output-root", type=str,
        default="data/benchmark_scaling",
        help="Root directory for benchmark outputs",
    )
    parser.add_argument(
        "--workers", type=int, nargs="+",
        default=[1, 4, 8, 16, 32, 64, 90],
        help="List of worker counts to benchmark",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    worker_counts = sorted(args.workers)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Pre-flight: show input file stats
    pf = pq.ParquetFile(str(input_path))
    logger.info(f"Input: {input_path} ({pf.metadata.num_rows:,} rows, "
                f"{pf.metadata.num_row_groups} row groups)")
    logger.info(f"Worker counts to benchmark: {worker_counts}")

    all_results: List[Dict[str, Any]] = []

    for wc in worker_counts:
        try:
            stats = run_single_benchmark(input_path, output_root, wc)
            all_results.append(stats)
            logger.info(
                f"workers={wc:3d}  elapsed={stats['elapsed_seconds']:.1f}s  "
                f"art/s={stats['throughput_articles_per_sec']:,.1f}  "
                f"chunk/s={stats['throughput_chunks_per_sec']:,.1f}  "
                f"failures={stats.get('failed_records', 0)}"
            )
        except Exception as e:
            logger.error(f"Benchmark failed for workers={wc}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        logger.error("No benchmark results collected!")
        sys.exit(1)

    # --- Scaling Analysis ------------------------------------------------
    analysis = compute_scaling_analysis(all_results)

    # --- Print Summary Table ---------------------------------------------
    print("\n")
    print("=" * 90)
    print("        CPU INGESTION SCALING BENCHMARK RESULTS")
    print("=" * 90)
    header = f"{'Workers':>8} | {'Elapsed(s)':>11} | {'Art/s':>10} | {'Chunk/s':>10} | {'Speedup':>8} | {'Efficiency':>11} | {'Failures':>8}"
    print(header)
    print("-" * 90)
    for row in analysis["per_worker_count"]:
        print(
            f"{row['num_workers']:>8d} | "
            f"{row['elapsed_seconds']:>11.1f} | "
            f"{row['throughput_articles_per_sec']:>10.1f} | "
            f"{row['throughput_chunks_per_sec']:>10.1f} | "
            f"{row['speedup_vs_1_worker']:>8.2f}x | "
            f"{row['parallel_efficiency_pct']:>10.1f}% | "
            f"{row['failed_records']:>8d}"
        )
    print("=" * 90)
    print()

    # --- Save consolidated report ----------------------------------------
    report = {
        "benchmark_meta": {
            "input_file": str(input_path),
            "input_rows": pf.metadata.num_rows,
            "input_row_groups": pf.metadata.num_row_groups,
            "worker_counts": worker_counts,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine_cores": os.cpu_count(),
        },
        "raw_results": all_results,
        "scaling_analysis": analysis,
    }

    json_path = output_root / "scaling_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved JSON report: {json_path}")

    # CSV summary
    csv_path = output_root / "scaling_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_workers",
                "elapsed_seconds",
                "throughput_articles_per_sec",
                "throughput_chunks_per_sec",
                "speedup_vs_1_worker",
                "parallel_efficiency_pct",
                "chunks_emitted",
                "failed_records",
            ],
        )
        writer.writeheader()
        for row in analysis["per_worker_count"]:
            writer.writerow(row)
    logger.info(f"Saved CSV summary: {csv_path}")


if __name__ == "__main__":
    main()

