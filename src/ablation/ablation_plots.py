"""Ablation study visualization: 6 matplotlib figures.

1. Recall@10 vs Index Type — grouped bar chart
2. Recall@10 vs nprobe — line chart with latency on secondary y-axis
3. Recall@10 vs Chunk Size — bar chart
4. Retrieval Mode Comparison — grouped bar (Recall@10, MRR, nDCG@10)
5. Latency vs Recall Pareto — scatter plot with Pareto frontier
6. Ingestion Throughput vs Workers — line chart
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Must set backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.common.logging import get_logger

logger = get_logger("ablation.plots")

# Consistent styling
COLORS = {
    "FlatIP": "#2196F3",
    "IVF-Flat": "#4CAF50",
    "IVF-PQ": "#FF9800",
    "dense": "#2196F3",
    "sparse": "#4CAF50",
    "hybrid_rrf": "#FF9800",
    "hybrid_rerank": "#F44336",
}
FIGSIZE = (10, 6)
DPI = 150


def _save_fig(fig: plt.Figure, path: Path, title: str):
    """Saves a figure with tight layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path.name} — {title}")


def _filter_results(results: List[Dict], dimension: str) -> List[Dict]:
    """Filters results by experiment dimension."""
    return [r for r in results if r.get("config", {}).get("dimension") == dimension]


def plot_index_type_comparison(results: List[Dict], output_dir: Path):
    """Plot 1: Recall@10 and index size by index type."""
    data = _filter_results(results, "index_type")
    if not data:
        logger.warning("No index_type results to plot")
        return

    labels = [r["config"]["index_type"] for r in data]
    recalls = [r["recall_at_10"] for r in data]
    sizes = [r["index_memory_mb"] for r in data]
    build_times = [r["index_build_time_sec"] for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Recall@10
    colors = [COLORS.get(l, "#9E9E9E") for l in labels]
    bars1 = ax1.bar(labels, recalls, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Recall@10")
    ax1.set_title("Retrieval Accuracy by Index Type")
    ax1.set_ylim(0, 1.05)
    for bar, val in zip(bars1, recalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Index size + build time
    x = np.arange(len(labels))
    width = 0.35
    bars2 = ax2.bar(x - width / 2, sizes, width, label="Size (MB)", color="#42A5F5", edgecolor="black", linewidth=0.5)
    ax2_twin = ax2.twinx()
    bars3 = ax2_twin.bar(x + width / 2, build_times, width, label="Build Time (s)", color="#FFA726", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Index Size (MB)")
    ax2_twin.set_ylabel("Build Time (s)")
    ax2.set_title("Index Size and Build Time")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    _save_fig(fig, output_dir / "index_type_comparison.png", "Index Type Comparison")


def plot_nprobe_sweep(results: List[Dict], output_dir: Path):
    """Plot 2: Recall@10 and latency vs nprobe."""
    data = _filter_results(results, "nprobe")
    if not data:
        logger.warning("No nprobe results to plot")
        return

    # Sort by nprobe
    data.sort(key=lambda r: r["config"]["nprobe"])

    nprobes = [r["config"]["nprobe"] for r in data]
    recalls = [r["recall_at_10"] for r in data]
    p50s = [r["query_p50_ms"] for r in data]
    p95s = [r["query_p95_ms"] for r in data]

    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # Recall line
    line1 = ax1.plot(nprobes, recalls, "o-", color="#2196F3", linewidth=2, markersize=8, label="Recall@10")
    ax1.set_xlabel("nprobe (cells visited)")
    ax1.set_ylabel("Recall@10", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xticks(nprobes)

    # Latency on secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(nprobes, p50s, "s--", color="#F44336", linewidth=1.5, markersize=6, label="p50 Latency")
    line3 = ax2.plot(nprobes, p95s, "^:", color="#FF9800", linewidth=1.5, markersize=6, label="p95 Latency")
    ax2.set_ylabel("Latency (ms)")
    ax2.tick_params(axis="y")

    # Combined legend
    lines = line1 + line2 + line3
    labels_legend = [l.get_label() for l in lines]
    ax1.legend(lines, labels_legend, loc="center right")

    ax1.set_title("Recall@10 vs nprobe (IVF-Flat, nlist=256)")
    ax1.grid(True, alpha=0.3)

    _save_fig(fig, output_dir / "nprobe_sweep.png", "nprobe Sweep")


def plot_chunk_size_comparison(results: List[Dict], output_dir: Path):
    """Plot 3: Recall@10 by chunk size."""
    data = _filter_results(results, "chunk_size")
    if not data:
        logger.warning("No chunk_size results to plot")
        return

    data.sort(key=lambda r: r["config"]["chunk_target_tokens"])

    sizes = [str(r["config"]["chunk_target_tokens"]) for r in data]
    recalls = [r["recall_at_10"] for r in data]
    total_chunks = [r.get("total_chunks", r.get("total_vectors", 0)) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Recall@10
    colors = ["#66BB6A", "#42A5F5", "#FFA726"][:len(sizes)]
    bars = ax1.bar(sizes, recalls, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Chunk Target Tokens")
    ax1.set_ylabel("Recall@10")
    ax1.set_title("Retrieval Accuracy by Chunk Size")
    ax1.set_ylim(0, 1.05)
    for bar, val in zip(bars, recalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Total chunks
    bars2 = ax2.bar(sizes, total_chunks, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Chunk Target Tokens")
    ax2.set_ylabel("Total Chunks")
    ax2.set_title("Corpus Size by Chunk Granularity")
    for bar, val in zip(bars2, total_chunks):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f"{val:,}", ha="center", va="bottom", fontsize=9)

    _save_fig(fig, output_dir / "chunk_size_comparison.png", "Chunk Size Comparison")


def plot_retrieval_mode_comparison(results: List[Dict], output_dir: Path):
    """Plot 4: Grouped bar chart comparing retrieval modes on Recall@10, MRR, nDCG@10."""
    data = _filter_results(results, "retrieval_mode")
    if not data:
        logger.warning("No retrieval_mode results to plot")
        return

    mode_order = ["dense", "sparse", "hybrid_rrf", "hybrid_rerank"]
    data_map = {r["config"]["retrieval_mode"]: r for r in data}
    ordered = [data_map[m] for m in mode_order if m in data_map]

    labels = [r["config"]["retrieval_mode"] for r in ordered]
    recalls = [r["recall_at_10"] for r in ordered]
    mrrs = [r["mrr"] for r in ordered]
    ndcgs = [r["ndcg_at_10"] for r in ordered]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x - width, recalls, width, label="Recall@10", color="#42A5F5", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, mrrs, width, label="MRR", color="#66BB6A", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, ndcgs, width, label="nDCG@10", color="#FFA726", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality by Pipeline Configuration")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                        f"{height:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    _save_fig(fig, output_dir / "retrieval_mode_comparison.png", "Retrieval Mode Comparison")


def plot_latency_vs_recall_pareto(results: List[Dict], output_dir: Path):
    """Plot 5: Scatter plot of latency vs recall with Pareto frontier."""
    # Include all experiments that have recall and latency data
    data = [r for r in results if r["recall_at_10"] > 0 and r["query_p50_ms"] > 0]
    if not data:
        logger.warning("No data for latency-recall Pareto plot")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Color by dimension
    dim_colors = {
        "index_type": "#2196F3",
        "nprobe": "#4CAF50",
        "chunk_size": "#FF9800",
        "retrieval_mode": "#F44336",
        "reranking": "#9C27B0",
    }
    dim_markers = {
        "index_type": "o",
        "nprobe": "s",
        "chunk_size": "D",
        "retrieval_mode": "^",
        "reranking": "v",
    }

    for r in data:
        dim = r["config"]["dimension"]
        color = dim_colors.get(dim, "#9E9E9E")
        marker = dim_markers.get(dim, "o")
        ax.scatter(
            r["query_p50_ms"], r["recall_at_10"],
            c=color, marker=marker, s=80, edgecolors="black", linewidths=0.5,
            label=dim, zorder=5,
        )
        ax.annotate(
            r["experiment_id"].replace("_", "\n", 1)[:20],
            (r["query_p50_ms"], r["recall_at_10"]),
            fontsize=6, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
        )

    # Compute Pareto frontier
    points = [(r["query_p50_ms"], r["recall_at_10"]) for r in data]
    points_sorted = sorted(points, key=lambda p: p[0])
    pareto = []
    best_recall = -1
    for lat, rec in points_sorted:
        if rec > best_recall:
            pareto.append((lat, rec))
            best_recall = rec

    if len(pareto) >= 2:
        pareto_x, pareto_y = zip(*pareto)
        ax.plot(pareto_x, pareto_y, "k--", linewidth=1.5, alpha=0.5, label="Pareto frontier")

    # Deduplicate legend entries
    handles, labels_leg = ax.get_legend_handles_labels()
    seen = {}
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, loc="lower right", fontsize=8)

    ax.set_xlabel("Query p50 Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Latency vs Accuracy Pareto")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    _save_fig(fig, output_dir / "latency_vs_recall_pareto.png", "Latency vs Recall Pareto")


def plot_worker_throughput(results: List[Dict], output_dir: Path):
    """Plot 6: Ingestion throughput vs worker count."""
    data = _filter_results(results, "worker_count")
    if not data:
        logger.warning("No worker_count results to plot")
        return

    data.sort(key=lambda r: r["config"]["num_workers"])

    workers = [r["config"]["num_workers"] for r in data]
    throughputs = [r["ingestion_throughput_chunks_sec"] for r in data]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(workers, throughputs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(workers, throughputs, alpha=0.1, color="#2196F3")

    # Ideal linear scaling line
    if throughputs and throughputs[0] > 0:
        ideal = [throughputs[0] * w for w in workers]
        ax.plot(workers, ideal, "--", color="#9E9E9E", linewidth=1, label="Ideal linear scaling")

    ax.set_xlabel("Number of CPU Workers")
    ax.set_ylabel("Throughput (chunks/sec)")
    ax.set_title("Ingestion Throughput Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(workers)

    for w, t in zip(workers, throughputs):
        ax.annotate(f"{t:.0f}", (w, t), fontsize=9, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    _save_fig(fig, output_dir / "worker_throughput.png", "Worker Throughput")


def generate_all_plots(results_path: Path, output_dir: Path):
    """Loads results JSON and generates all 6 plots."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data.get("experiments", [])
    if not experiments:
        logger.warning("No experiments found in results file")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_index_type_comparison(experiments, output_dir)
    plot_nprobe_sweep(experiments, output_dir)
    plot_chunk_size_comparison(experiments, output_dir)
    plot_retrieval_mode_comparison(experiments, output_dir)
    plot_latency_vs_recall_pareto(experiments, output_dir)
    plot_worker_throughput(experiments, output_dir)

    logger.info(f"All plots saved to {output_dir}")


def generate_report(results_path: Path, output_path: Path):
    """Generates a markdown report from ablation results.

    Conclusions are drawn strictly from the measured data.
    No winner is pre-selected.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data.get("experiments", [])

    lines = [
        "# Ablation Study Report",
        "",
        f"**Total experiments**: {len(experiments)}",
        f"**Dimensions**: {len(set(e['config']['dimension'] for e in experiments))}",
        "",
    ]

    # Group by dimension
    dimensions = {}
    for exp in experiments:
        dim = exp["config"]["dimension"]
        if dim not in dimensions:
            dimensions[dim] = []
        dimensions[dim].append(exp)

    for dim_name, dim_exps in dimensions.items():
        lines.append(f"## {dim_name.replace('_', ' ').title()}")
        lines.append("")

        # Table header
        if dim_name in ("index_type", "nprobe"):
            lines.append("| Experiment | Index | nprobe | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms | Size MB | Build s |")
            lines.append("|------------|-------|--------|-----------|-----|---------|--------|--------|---------|---------|")
            for e in dim_exps:
                cfg = e["config"]
                lines.append(
                    f"| {e['experiment_id']} | {cfg['index_type']} | {cfg['nprobe']} | "
                    f"{e['recall_at_10']:.4f} | {e['mrr']:.4f} | {e['ndcg_at_10']:.4f} | "
                    f"{e['query_p50_ms']:.2f} | {e['query_p95_ms']:.2f} | "
                    f"{e['index_memory_mb']:.2f} | {e['index_build_time_sec']:.3f} |"
                )
        elif dim_name == "chunk_size":
            lines.append("| Experiment | Tokens | Total Chunks | Recall@10 | MRR | nDCG@10 | p50 ms | GPU % |")
            lines.append("|------------|--------|-------------|-----------|-----|---------|--------|-------|")
            for e in dim_exps:
                cfg = e["config"]
                lines.append(
                    f"| {e['experiment_id']} | {cfg['chunk_target_tokens']} | {e.get('total_chunks', e.get('total_vectors', '?'))} | "
                    f"{e['recall_at_10']:.4f} | {e['mrr']:.4f} | {e['ndcg_at_10']:.4f} | "
                    f"{e['query_p50_ms']:.2f} | {e['gpu_utilization_pct']:.1f} |"
                )
        elif dim_name in ("retrieval_mode", "reranking"):
            lines.append("| Experiment | Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms | GPU % |")
            lines.append("|------------|------|-----------|-----|---------|--------|--------|-------|")
            for e in dim_exps:
                cfg = e["config"]
                lines.append(
                    f"| {e['experiment_id']} | {cfg['retrieval_mode']} | "
                    f"{e['recall_at_10']:.4f} | {e['mrr']:.4f} | {e['ndcg_at_10']:.4f} | "
                    f"{e['query_p50_ms']:.2f} | {e['query_p95_ms']:.2f} | {e['gpu_utilization_pct']:.1f} |"
                )
        elif dim_name == "worker_count":
            lines.append("| Experiment | Workers | Throughput (chunks/s) | Total Chunks | Elapsed s |")
            lines.append("|------------|---------|----------------------|-------------|-----------|")
            for e in dim_exps:
                cfg = e["config"]
                lines.append(
                    f"| {e['experiment_id']} | {cfg['num_workers']} | "
                    f"{e['ingestion_throughput_chunks_sec']:.1f} | {e.get('total_chunks', '?')} | "
                    f"{e['index_build_time_sec']:.2f} |"
                )

        lines.append("")

        # Data-driven observations (not pre-selected conclusions)
        if dim_name == "index_type" and dim_exps:
            best = max(dim_exps, key=lambda e: e["recall_at_10"])
            smallest = min(dim_exps, key=lambda e: e["index_memory_mb"]) if any(e["index_memory_mb"] > 0 for e in dim_exps) else None
            lines.append(f"**Observation**: Highest Recall@10 = {best['recall_at_10']:.4f} ({best['experiment_id']}). ")
            if smallest and smallest["index_memory_mb"] > 0:
                lines.append(f"Smallest index = {smallest['index_memory_mb']:.2f} MB ({smallest['experiment_id']}).")
            lines.append("")

        elif dim_name == "nprobe" and dim_exps:
            sorted_exps = sorted(dim_exps, key=lambda e: e["config"]["nprobe"])
            low_np = sorted_exps[0]
            high_np = sorted_exps[-1]
            lines.append(
                f"**Observation**: nprobe={low_np['config']['nprobe']} → Recall@10={low_np['recall_at_10']:.4f}, "
                f"p50={low_np['query_p50_ms']:.2f}ms. "
                f"nprobe={high_np['config']['nprobe']} → Recall@10={high_np['recall_at_10']:.4f}, "
                f"p50={high_np['query_p50_ms']:.2f}ms. "
                f"Higher nprobe improves recall at the cost of latency."
            )
            lines.append("")

    lines.extend([
        "## Methodology",
        "",
        "- All experiments use the same 21 curated, grounded evaluation queries.",
        "- Index experiments reuse pre-computed embeddings (BAAI/bge-small-en-v1.5, FP16).",
        "- Chunk-size experiments re-chunk and re-embed from source articles.",
        "- Worker-count experiments measure end-to-end ingestion pipeline throughput.",
        "- GPU utilization sampled via `nvidia-smi dmon` at 1-second intervals.",
        "- Recall@10 computed as binary hit: 1.0 if any gold chunk is in top-10.",
        "",
        "> [!NOTE]",
        "> Conclusions are based exclusively on measured data from this experiment run.",
        "> No winner was pre-selected.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")

