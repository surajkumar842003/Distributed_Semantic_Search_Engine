#!/usr/bin/env python3
"""CLI entry point for the reproducible ablation study.

Usage:
    python scripts/run_ablation.py --dimensions all
    python scripts/run_ablation.py --dimensions index_type nprobe
    python scripts/run_ablation.py --dimensions chunk_size --chunk-sizes 128 256 512
    python scripts/run_ablation.py --dimensions nprobe --skip-completed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Module resolution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mask torchaudio ABI incompatibility
import src.common.torch_compat

# Set matplotlib config dir to avoid permission warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.common.logging import get_logger
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
from src.ablation.ablation_runner import AblationRunner
from src.ablation.ablation_plots import generate_all_plots, generate_report

logger = get_logger("scripts.run_ablation")

DIMENSION_GENERATORS = {
    "index_type": generate_index_type_experiments,
    "nprobe": generate_nprobe_experiments,
    "chunk_size": generate_chunk_size_experiments,
    "retrieval_mode": generate_retrieval_mode_experiments,
    "reranking": generate_reranking_experiments,
    "worker_count": generate_worker_count_experiments,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible ablation study for Wikipedia RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dimensions:
  index_type      Compare FlatIP, IVF-Flat, IVF-PQ
  nprobe          Sweep nprobe values for IVF-Flat
  chunk_size      Compare different chunk target token counts
  retrieval_mode  Compare dense, sparse, hybrid, reranked
  reranking       Compare with/without cross-encoder reranking
  worker_count    Compare ingestion throughput at different worker counts
  all             Run all dimensions
        """,
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["all"],
        choices=list(DIMENSION_GENERATORS.keys()) + ["all"],
        help="Which ablation dimensions to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/DATA/suraj/m1/search_engine/data/ablation",
        help="Output directory for results, plots, and configs",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip experiments that already have result files (default: True)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Force re-run of all experiments even if results exist",
    )
    parser.add_argument(
        "--gpu-device",
        type=str,
        default="cuda:0",
        help="GPU device for embedding and reranking",
    )
    parser.add_argument(
        "--source-parquet",
        type=str,
        default=None,
        help="Path to raw Wikipedia parquet (for worker-count experiments)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip experiments and only regenerate plots from existing results",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    skip_completed = args.skip_completed and not args.no_skip

    # --- Plots-only mode ---
    if args.plots_only:
        results_path = output_dir / "ablation_results.json"
        if not results_path.is_file():
            logger.error(f"Results file not found: {results_path}")
            sys.exit(1)
        generate_all_plots(results_path, plots_dir)
        generate_report(results_path, output_dir / "ablation_report.md")
        logger.info("Plots and report regenerated from existing results.")
        return

    # --- Generate experiment configs ---
    if "all" in args.dimensions:
        configs = generate_all_experiments()
    else:
        configs = []
        for dim in args.dimensions:
            generator = DIMENSION_GENERATORS[dim]
            configs.extend(generator())

    logger.info(
        f"Generated {len(configs)} experiment configs across "
        f"{len(set(c.dimension for c in configs))} dimensions"
    )

    # Save configs
    save_all_configs(configs, output_dir / "configs")

    # --- Run experiments ---
    runner = AblationRunner(
        output_dir=output_dir,
        source_parquet=args.source_parquet,
        gpu_device=args.gpu_device,
        skip_completed=skip_completed,
    )

    t_start = time.perf_counter()
    results = runner.run_all(configs)
    total_elapsed = time.perf_counter() - t_start

    # --- Generate plots and report ---
    results_path = output_dir / "ablation_results.json"
    if results_path.is_file():
        generate_all_plots(results_path, plots_dir)
        generate_report(results_path, output_dir / "ablation_report.md")

    # --- Summary ---
    successful = sum(1 for r in results if not r.errors)
    failed = sum(1 for r in results if r.errors)

    logger.info(
        f"\n{'='*60}\n"
        f"ABLATION STUDY COMPLETE\n"
        f"{'='*60}\n"
        f"Total experiments: {len(results)}\n"
        f"Successful:        {successful}\n"
        f"Failed:            {failed}\n"
        f"Total time:        {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n"
        f"Results:           {results_path}\n"
        f"Plots:             {plots_dir}\n"
        f"Report:            {output_dir / 'ablation_report.md'}\n"
        f"{'='*60}"
    )

    if failed > 0:
        logger.warning(f"{failed} experiments had errors:")
        for r in results:
            if r.errors:
                logger.warning(f"  {r.experiment_id}: {r.errors}")


if __name__ == "__main__":
    main()

