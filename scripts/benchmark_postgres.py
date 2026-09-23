"""PostgreSQL Storage Layer Throughput and Latency Benchmark.

Measures:
1. Document batch insert throughput (docs/sec).
2. Chunk batch insert vs high-speed COPY throughput (chunks/sec).
3. Candidate hydration latency by chunk_id batch (mean, p50, p95, p99 ms).
4. Full-Text Search (tsvector + GIN) query latency (mean, p50, p95, p99 ms).
5. Exports reports to data/benchmark_postgres/.

Usage:
    python scripts/benchmark_postgres.py \
        --chunks-dir data/chunks_pilot \
        --output-dir data/benchmark_postgres \
        --max-chunks 10000
"""

import argparse
import asyncio
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
from src.storage.postgres_client import PostgresClient, explain_embedding_storage_decision
from src.storage.migrator import DatabaseMigrator
from src.storage.ingest_chunks import PostgresIngester
from src.common.logging import get_logger

logger = get_logger("scripts.benchmark_postgres")


def load_test_chunks(chunks_dir: Path, max_chunks: int = 10000) -> Tuple[List[Dict[str, Any]], List[Tuple]]:
    """Loads a slice of chunk Parquet data for benchmarking."""
    shard_files = sorted(chunks_dir.glob("chunks_*.parquet"))
    if not shard_files:
        logger.warning(f"No chunk Parquet files in {chunks_dir}; generating synthetic data.")
        docs = [
            {"doc_id": i // 10, "title": f"Article {i // 10}", "url": f"http://wiki/{i // 10}", "source_shard": 0}
            for i in range(max_chunks)
        ]
        records = [
            (i, i // 10, "Section", f"Sample passage content for chunk {i} discussing technology.", 10, i)
            for i in range(max_chunks)
        ]
        return docs, records

    table = pq.read_table(str(shard_files[0]))
    n = min(table.num_rows, max_chunks)

    doc_ids = table.column("doc_id").to_numpy(zero_copy_only=False)[:n]
    titles = table.column("title").to_pylist()[:n]
    urls = table.column("url").to_pylist()[:n]
    chunk_ids = table.column("chunk_id").to_numpy(zero_copy_only=False)[:n]
    sections = table.column("section_path").to_pylist()[:n]
    texts = table.column("text").to_pylist()[:n]
    tokens = table.column("token_count").to_numpy(zero_copy_only=False)[:n]
    faiss_ids = table.column("faiss_id").to_numpy(zero_copy_only=False)[:n] if "faiss_id" in table.column_names else chunk_ids

    seen_docs = set()
    docs = []
    records = []
    for i in range(n):
        did = int(doc_ids[i])
        if did not in seen_docs:
            seen_docs.add(did)
            docs.append({"doc_id": did, "title": titles[i], "url": urls[i], "source_shard": 0})

        records.append((
            int(chunk_ids[i]),
            did,
            str(sections[i]),
            str(texts[i]),
            int(tokens[i]),
            int(faiss_ids[i]),
        ))

    return docs, records


async def run_benchmark(
    chunks_dir: Path,
    output_dir: Path,
    max_chunks: int = 10000,
) -> Dict[str, Any]:
    """Executes the database throughput and latency benchmark."""
    cfg = AppConfig.load_from_dir("configs")
    client = PostgresClient(cfg.postgres)

    try:
        await client.connect()
        healthy = await client.is_healthy()
        if not healthy:
            raise ConnectionError("Database is not reachable on localhost:5432.")
    except Exception as e:
        logger.error(
            f"Cannot connect to PostgreSQL at {cfg.postgres.host}:{cfg.postgres.port}: {e}\n"
            f"Make sure the container is running: sudo docker compose up -d postgres"
        )
        return {"error": str(e)}

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run migrations
    migrator = DatabaseMigrator()
    applied = await migrator.apply_migrations(client.pool)
    logger.info(f"Database migrations applied: {applied}")

    # 2. Load benchmark test data
    docs, records = load_test_chunks(chunks_dir, max_chunks=max_chunks)
    logger.info(f"Loaded {len(docs):,} documents and {len(records):,} chunks for benchmarking")

    results = []

    # 3. Document Ingestion Benchmark
    t0 = time.time()
    await client.insert_documents_batch(docs)
    doc_time = time.time() - t0
    doc_rate = len(docs) / max(0.001, doc_time)
    results.append({
        "operation": "Documents Batch Insert",
        "count": len(docs),
        "elapsed_sec": round(doc_time, 3),
        "throughput_ops_per_sec": round(doc_rate, 1),
        "mean_latency_ms": round((doc_time / len(docs)) * 1000, 3) if docs else 0.0,
    })

    # 4. Chunk COPY Benchmark (Idempotent COPY)
    t0 = time.time()
    await client.copy_chunks_idempotent(records)
    copy_time = time.time() - t0
    copy_rate = len(records) / max(0.001, copy_time)
    results.append({
        "operation": "Chunks Idempotent COPY",
        "count": len(records),
        "elapsed_sec": round(copy_time, 3),
        "throughput_ops_per_sec": round(copy_rate, 1),
        "mean_latency_ms": round((copy_time / len(records)) * 1000, 4) if records else 0.0,
    })

    # 5. Candidate Hydration Latency (batch of 50 IDs)
    sample_chunk_ids = [r[0] for r in records[:50]]
    latencies_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        hydrated = await client.get_chunks_by_ids(sample_chunk_ids)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    results.append({
        "operation": "Candidate Hydration (50 IDs)",
        "count": 50,
        "elapsed_sec": round(sum(latencies_ms) / 1000.0, 3),
        "throughput_ops_per_sec": round(1000.0 / np.mean(latencies_ms), 1),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 3),
    })

    # 6. Full-Text Search (tsvector + GIN) Latency
    test_queries = ["history", "science", "technology", "quantum", "war"]
    fts_latencies_ms = []
    for q in test_queries:
        for _ in range(10):
            t0 = time.perf_counter()
            fts_res = await client.search_fts(q, top_k=20)
            t1 = time.perf_counter()
            fts_latencies_ms.append((t1 - t0) * 1000.0)

    results.append({
        "operation": "Full-Text Search (tsvector + GIN)",
        "count": len(fts_latencies_ms),
        "elapsed_sec": round(sum(fts_latencies_ms) / 1000.0, 3),
        "throughput_ops_per_sec": round(1000.0 / np.mean(fts_latencies_ms), 1),
        "mean_latency_ms": round(float(np.mean(fts_latencies_ms)), 3),
        "p50_latency_ms": round(float(np.percentile(fts_latencies_ms, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(fts_latencies_ms, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(fts_latencies_ms, 99)), 3),
    })

    await client.disconnect()

    # Print summary table
    print("\n" + "=" * 95)
    print("                    POSTGRESQL STORAGE LAYER BENCHMARK")
    print("=" * 95)
    print(explain_embedding_storage_decision())
    print("-" * 95)
    header = f"{'Operation':<35} | {'Count':>8} | {'Elapsed(s)':>10} | {'Ops / Sec':>12} | {'Mean (ms)':>10}"
    print(header)
    print("-" * 95)
    for r in results:
        print(
            f"{r['operation']:<35} | "
            f"{r['count']:>8,d} | "
            f"{r['elapsed_sec']:>10.3f} | "
            f"{r['throughput_ops_per_sec']:>12,.1f} | "
            f"{r['mean_latency_ms']:>10.3f}"
        )
    print("=" * 95 + "\n")

    # Save JSON & CSV reports
    json_path = output_dir / "postgres_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    csv_path = output_dir / "postgres_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    logger.info(f"Saved benchmark results to {json_path} and {csv_path}")
    return {"results": results, "json_path": str(json_path), "csv_path": str(csv_path)}


def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Storage Benchmark")
    parser.add_argument("--chunks-dir", type=str, default="data/chunks_pilot", help="Chunks directory")
    parser.add_argument("--output-dir", type=str, default="data/benchmark_postgres", help="Report directory")
    parser.add_argument("--max-chunks", type=int, default=10000, help="Max chunks to load")
    args = parser.parse_args()

    asyncio.run(run_benchmark(Path(args.chunks_dir), Path(args.output_dir), max_chunks=args.max_chunks))


if __name__ == "__main__":
    main()

