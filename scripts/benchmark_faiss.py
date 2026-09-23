"""FAISS Indexing and Retrieval Benchmark.

Compares:
1. Exact IndexFlatIP (ground truth baseline)
2. IndexIVFFlat (inverted file with full precision)
3. IndexIVFPQ (inverted file with product quantization)

Measures:
- Index build time (training + incremental shard addition)
- Index size on disk and memory
- Query latency (mean, p50, p95, p99 ms) across nprobe values
- Recall@1, Recall@5, Recall@10, Recall@50 against exact FlatIP

Usage:
    python scripts/benchmark_faiss.py \
        --embeddings-dir data/benchmark_embedding_v2/bs_0064 \
        --output-dir data/benchmark_faiss \
        --num-queries 100 \
        --train-sample-size 20000 \
        --nlist 256 \
        --nprobe-values 1 4 16 32 64
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pyarrow.parquet as pq

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.indexing.faiss_indexer import (
    FAISSIndexer,
    explain_metric_choice,
    ensure_l2_normalized,
)
from src.common.logging import get_logger

logger = get_logger("scripts.benchmark_faiss")


def load_sample_vectors(
    shard_paths: List[Path],
    sample_size: int,
    dim: int = 384,
    seed: int = 42,
) -> np.ndarray:
    """Samples a representative set of vectors across shards for training."""
    collected: List[np.ndarray] = []
    total_needed = sample_size
    rng = np.random.RandomState(seed)

    for p in shard_paths:
        table = pq.read_table(str(p), columns=["embedding"])
        flat_vecs = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        vecs = flat_vecs.reshape(-1, dim).astype(np.float32)

        if len(vecs) <= total_needed:
            collected.append(vecs)
            total_needed -= len(vecs)
        else:
            sample_idx = rng.choice(len(vecs), size=total_needed, replace=False)
            collected.append(vecs[sample_idx])
            break

        if total_needed <= 0:
            break

    all_samples = np.vstack(collected)
    return ensure_l2_normalized(all_samples)


def sample_query_vectors(
    shard_paths: List[Path],
    num_queries: int = 100,
    dim: int = 384,
    seed: int = 99,
) -> Tuple[np.ndarray, np.ndarray]:
    """Samples query vectors and their chunk_ids from the corpus."""
    rng = np.random.RandomState(seed)
    # Held-out evaluation methodology: 
    # Sample from the second shard to prevent self-retrieval perfect recall
    query_shard = shard_paths[1] if len(shard_paths) > 1 else shard_paths[0]
    table = pq.read_table(str(query_shard), columns=["chunk_id", "embedding"])
    chunk_ids = np.array(table.column("chunk_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    flat_vecs = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
    vecs = flat_vecs.reshape(-1, dim).astype(np.float32)

    n_available = len(vecs)
    n_sample = min(num_queries, n_available)
    indices = rng.choice(n_available, size=n_sample, replace=False)

    queries = ensure_l2_normalized(vecs[indices])
    query_ids = chunk_ids[indices]
    return queries, query_ids


def compute_recall_at_k(
    approx_ids: np.ndarray,
    ground_truth_ids: np.ndarray,
    k: int,
) -> float:
    """Computes mean Recall@k between approximate and exact nearest neighbors."""
    recalls = []
    for i in range(len(approx_ids)):
        gt_set = set(ground_truth_ids[i, :k])
        approx_set = set(approx_ids[i, :k])
        intersection = len(gt_set.intersection(approx_set))
        recalls.append(intersection / max(1, k))
    return float(np.mean(recalls))


def measure_query_latencies(
    indexer: FAISSIndexer,
    queries: np.ndarray,
    top_k: int = 50,
    nprobe: Optional[int] = None,
    warmup_runs: int = 5,
    measured_runs: int = 20,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Measures single-query and batch latency percentiles in milliseconds."""
    # Warmup
    for _ in range(warmup_runs):
        indexer.search(queries[:5], top_k=top_k, nprobe=nprobe)

    # Per-query latencies
    latencies_ms = []
    last_scores = None
    last_ids = None

    for _ in range(measured_runs):
        for q_idx in range(len(queries)):
            q_single = queries[q_idx : q_idx + 1]
            t0 = time.perf_counter()
            scores, ids = indexer.search(q_single, top_k=top_k, nprobe=nprobe)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    # Final full batch search for recall evaluation
    last_scores, last_ids = indexer.search(queries, top_k=top_k, nprobe=nprobe)

    stats = {
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "qps": round(1000.0 / max(0.001, float(np.mean(latencies_ms))), 1),
    }
    return stats, last_scores, last_ids


def run_faiss_benchmark(
    embeddings_dir: Path,
    output_dir: Path,
    num_queries: int = 100,
    train_sample_size: int = 20000,
    nlist: int = 256,
    nprobe_values: Optional[List[int]] = None,
    m_subquantizers: int = 48,
    bits_per_code: int = 8,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Executes full benchmark comparing FlatIP, IVF-Flat, and IVF-PQ."""
    nprobes = nprobe_values or [1, 4, 16, 32, 64]
    shard_paths = sorted(embeddings_dir.glob("embeddings_*.parquet"))

    if not shard_paths:
        raise FileNotFoundError(f"No embeddings_*.parquet files found in {embeddings_dir}")

    # Use first shard only for indexing to ensure held-out evaluation split
    index_shard_paths = [shard_paths[0]]

    total_chunks = sum(pq.read_metadata(str(p)).num_rows for p in index_shard_paths)
    logger.info(
        f"Found {len(index_shard_paths)} shards ({total_chunks:,} vectors) for indexing in {embeddings_dir}"
    )

    # 1. Sample training vectors and query vectors
    logger.info(f"Sampling up to {train_sample_size:,} training vectors...")
    train_vecs = load_sample_vectors(index_shard_paths, sample_size=train_sample_size, seed=42)
    logger.info(f"Loaded {len(train_vecs):,} representative training vectors")

    logger.info(f"Sampling {num_queries} query probes from dataset...")
    queries, query_chunk_ids = sample_query_vectors(shard_paths, num_queries=num_queries, seed=99)

    dim = train_vecs.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    # =========================================================================
    # 2. Benchmark Exact Baseline: IndexFlatIP
    # =========================================================================
    logger.info("=" * 70)
    logger.info("  BENCHMARK 1: Exact IndexFlatIP (Ground Truth Baseline)")
    logger.info("=" * 70)
    flat_path = output_dir / "wiki_exact_flat.index"

    flat_indexer = FAISSIndexer(dim=dim, index_type="FlatIP")
    t0 = time.time()
    add_stats_flat = flat_indexer.add_from_shards(index_shard_paths)
    flat_build_time = time.time() - t0

    flat_indexer.save(flat_path)
    flat_disk_mb = round(flat_indexer.get_disk_size_bytes(flat_path) / (1024 * 1024), 2)

    # Test safe reload
    reloaded_flat = FAISSIndexer.load(flat_path, use_mmap=False)
    assert reloaded_flat.ntotal == flat_indexer.ntotal

    flat_latencies, gt_scores, gt_ids = measure_query_latencies(
        reloaded_flat, queries, top_k=top_k, nprobe=None
    )

    flat_result = {
        "index_type": "IndexFlatIP",
        "configuration": "exact brute-force",
        "nprobe": "N/A",
        "train_time_sec": 0.0,
        "add_time_sec": round(flat_build_time, 3),
        "total_build_time_sec": round(flat_build_time, 3),
        "total_vectors": flat_indexer.ntotal,
        "index_disk_mb": flat_disk_mb,
        "bytes_per_vector": round((flat_disk_mb * 1024 * 1024) / max(1, flat_indexer.ntotal), 1),
        "recall_at_1": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "recall_at_50": 1.0,
        **flat_latencies,
    }
    all_results.append(flat_result)
    logger.info(
        f"FlatIP | Build: {flat_build_time:.2f}s | Disk: {flat_disk_mb} MB | "
        f"Latency: {flat_latencies['mean_latency_ms']:.2f} ms | Recall@10: 100.0%"
    )

    # =========================================================================
    # 3. Benchmark IndexIVFFlat
    # =========================================================================
    logger.info("=" * 70)
    logger.info(f"  BENCHMARK 2: IndexIVFFlat (nlist={nlist})")
    logger.info("=" * 70)
    ivf_path = output_dir / "wiki_ivf_flat.index"

    ivf_indexer = FAISSIndexer(dim=dim, index_type="IVF-Flat", nlist=nlist)
    ivf_train_time = ivf_indexer.train(train_vecs)

    t0 = time.time()
    add_stats_ivf = ivf_indexer.add_from_shards(index_shard_paths)
    ivf_add_time = time.time() - t0
    ivf_total_build = ivf_train_time + ivf_add_time

    ivf_indexer.save(ivf_path)
    ivf_disk_mb = round(ivf_indexer.get_disk_size_bytes(ivf_path) / (1024 * 1024), 2)

    reloaded_ivf = FAISSIndexer.load(ivf_path, use_mmap=False)

    for np_val in nprobes:
        latencies, _, approx_ids = measure_query_latencies(
            reloaded_ivf, queries, top_k=top_k, nprobe=np_val
        )
        r1 = round(compute_recall_at_k(approx_ids, gt_ids, k=1), 4)
        r5 = round(compute_recall_at_k(approx_ids, gt_ids, k=5), 4)
        r10 = round(compute_recall_at_k(approx_ids, gt_ids, k=10), 4)
        r50 = round(compute_recall_at_k(approx_ids, gt_ids, k=50), 4)

        res = {
            "index_type": "IndexIVFFlat",
            "configuration": f"nlist={ivf_indexer.nlist}",
            "nprobe": np_val,
            "train_time_sec": round(ivf_train_time, 3),
            "add_time_sec": round(ivf_add_time, 3),
            "total_build_time_sec": round(ivf_total_build, 3),
            "total_vectors": ivf_indexer.ntotal,
            "index_disk_mb": ivf_disk_mb,
            "bytes_per_vector": round((ivf_disk_mb * 1024 * 1024) / max(1, ivf_indexer.ntotal), 1),
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "recall_at_50": r50,
            **latencies,
        }
        all_results.append(res)
        logger.info(
            f"IVF-Flat (nprobe={np_val:2d}) | Latency: {latencies['mean_latency_ms']:.2f} ms "
            f"(p95: {latencies['p95_latency_ms']:.2f} ms) | Recall@10: {r10*100:.1f}% | Recall@1: {r1*100:.1f}%"
        )

    # =========================================================================
    # 4. Benchmark IndexIVFPQ
    # =========================================================================
    logger.info("=" * 70)
    logger.info(f"  BENCHMARK 3: IndexIVFPQ (nlist={nlist}, m={m_subquantizers}, bits={bits_per_code})")
    logger.info("=" * 70)
    pq_path = output_dir / "wiki_ivf_pq.index"

    pq_indexer = FAISSIndexer(
        dim=dim,
        index_type="IVF-PQ",
        nlist=nlist,
        m_subquantizers=m_subquantizers,
        bits_per_code=bits_per_code,
    )
    pq_train_time = pq_indexer.train(train_vecs)

    t0 = time.time()
    add_stats_pq = pq_indexer.add_from_shards(index_shard_paths)
    pq_add_time = time.time() - t0
    pq_total_build = pq_train_time + pq_add_time

    pq_indexer.save(pq_path)
    pq_disk_mb = round(pq_indexer.get_disk_size_bytes(pq_path) / (1024 * 1024), 2)

    reloaded_pq = FAISSIndexer.load(pq_path, use_mmap=False)

    for np_val in nprobes:
        latencies, _, approx_ids = measure_query_latencies(
            reloaded_pq, queries, top_k=top_k, nprobe=np_val
        )
        r1 = round(compute_recall_at_k(approx_ids, gt_ids, k=1), 4)
        r5 = round(compute_recall_at_k(approx_ids, gt_ids, k=5), 4)
        r10 = round(compute_recall_at_k(approx_ids, gt_ids, k=10), 4)
        r50 = round(compute_recall_at_k(approx_ids, gt_ids, k=50), 4)

        res = {
            "index_type": "IndexIVFPQ",
            "configuration": f"nlist={pq_indexer.nlist}, m={m_subquantizers}, 8-bit",
            "nprobe": np_val,
            "train_time_sec": round(pq_train_time, 3),
            "add_time_sec": round(pq_add_time, 3),
            "total_build_time_sec": round(pq_total_build, 3),
            "total_vectors": pq_indexer.ntotal,
            "index_disk_mb": pq_disk_mb,
            "bytes_per_vector": round((pq_disk_mb * 1024 * 1024) / max(1, pq_indexer.ntotal), 1),
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "recall_at_50": r50,
            **latencies,
        }
        all_results.append(res)
        logger.info(
            f"IVF-PQ   (nprobe={np_val:2d}) | Latency: {latencies['mean_latency_ms']:.2f} ms "
            f"(p95: {latencies['p95_latency_ms']:.2f} ms) | Recall@10: {r10*100:.1f}% | Recall@1: {r1*100:.1f}%"
        )

    # =========================================================================
    # 5. Output Summary Table & Explanations
    # =========================================================================
    print("\n")
    print("=" * 135)
    print("                        FAISS DENSE RETRIEVAL BENCHMARK REPORT")
    print("=" * 135)
    print(explain_metric_choice("INNER_PRODUCT"))
    print("-" * 135)

    header = (
        f"{'Index Type':<13} | {'Config / nprobe':<20} | {'Build(s)':>8} | {'Disk MB':>8} | "
        f"{'Bytes/Vec':>9} | {'Mean ms':>8} | {'p95 ms':>8} | {'QPS':>7} | "
        f"{'R@1 (%)':>8} | {'R@10 (%)':>8} | {'R@50 (%)':>8}"
    )
    print(header)
    print("-" * 135)

    for r in all_results:
        cfg_str = f"nprobe={r['nprobe']}" if r['nprobe'] != "N/A" else r['configuration']
        print(
            f"{r['index_type']:<13} | "
            f"{cfg_str:<20} | "
            f"{r['total_build_time_sec']:>8.2f} | "
            f"{r['index_disk_mb']:>8.1f} | "
            f"{r['bytes_per_vector']:>9.1f} | "
            f"{r['mean_latency_ms']:>8.2f} | "
            f"{r['p95_latency_ms']:>8.2f} | "
            f"{r['qps']:>7.0f} | "
            f"{r['recall_at_1']*100:>7.1f}% | "
            f"{r['recall_at_10']*100:>7.1f}% | "
            f"{r['recall_at_50']*100:>7.1f}%"
        )
    print("=" * 135)

    print("\nANALYSIS & ARCHITECTURAL TRADE-OFFS:")
    print("1. Metric: INNER_PRODUCT computes Cosine Similarity directly because all vectors are L2-normalized.")
    print("2. Memory vs Accuracy: IVF-PQ achieves a ~30x reduction in vector storage (48 bytes vs 1,536 bytes),")
    print("   drastically cutting memory footprint. However, lossy product quantization reduces Recall@10.")
    print("3. Why IVF-PQ is NOT automatically superior: In two-stage retrieval with a cross-encoder reranker,")
    print("   the initial dense search must provide high candidate recall (Recall@50). If IVF-PQ drops relevant")
    print("   documents due to quantization error, downstream reranking cannot recover them. For high-precision")
    print("   search over pilot subsets, IVF-Flat provides near-100% recall with low search latency.\n")

    # =========================================================================
    # 6. Save JSON & CSV Reports
    # =========================================================================
    json_path = output_dir / "faiss_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embeddings_dir": str(embeddings_dir),
                "total_vectors": total_chunks,
                "num_queries": num_queries,
                "dimension": dim,
                "metric": "INNER_PRODUCT",
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved benchmark JSON report: {json_path}")

    csv_path = output_dir / "faiss_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    logger.info(f"Saved benchmark CSV summary: {csv_path}")

    return {
        "results": all_results,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="FAISS Indexing and Retrieval Benchmark")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/benchmark_embedding_v2/bs_0064",
        help="Directory containing Parquet embedding shards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark_faiss",
        help="Directory where indexes and benchmark reports are saved",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of query probes to sample from dataset",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=20000,
        help="Number of vectors to sample for training IVF/PQ",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=256,
        help="Number of Voronoi cells / inverted lists",
    )
    parser.add_argument(
        "--nprobe-values",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32, 64],
        help="List of nprobe values to benchmark",
    )
    parser.add_argument(
        "--m-subquantizers",
        type=int,
        default=48,
        help="Number of PQ subquantizers (must divide dim)",
    )
    parser.add_argument(
        "--bits-per-code",
        type=int,
        default=8,
        help="Bits per subquantizer code (default: 8 for 256 centroids)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k candidates retrieved for evaluation",
    )
    args = parser.parse_args()

    run_faiss_benchmark(
        embeddings_dir=Path(args.embeddings_dir),
        output_dir=Path(args.output_dir),
        num_queries=args.num_queries,
        train_sample_size=args.train_sample_size,
        nlist=args.nlist,
        nprobe_values=args.nprobe_values,
        m_subquantizers=args.m_subquantizers,
        bits_per_code=args.bits_per_code,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

