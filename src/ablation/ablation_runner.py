"""Ablation Study Runner Engine.

Executes individual experiment configurations and collects measurements:
- Recall@10 (using 21 curated evaluation queries)
- Index memory (file size on disk)
- Index construction time (train + add, seconds)
- Query p50 and p95 latency (ms)
- Ingestion throughput (chunks/sec, for worker-count experiments)
- GPU utilization (%, for embedding/reranking experiments)

Each experiment is self-contained and produces a JSON result file.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import src.common.torch_compat
from src.common.logging import get_logger
from src.ablation.ablation_config import ExperimentConfig
from src.ablation.gpu_monitor import GPUUtilizationMonitor
from src.evaluation.metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k, compute_latency_stats
from src.evaluation.dataset import (
    QASample,
    GroundTruthEvidence,
    EvidenceMapper,
    get_curated_pilot_benchmark_samples,
)

logger = get_logger("ablation.runner")

# Paths
_DATA_DIR = Path("/DATA/suraj/m1/search_engine/data")
_EXISTING_EMBEDDINGS_DIR = _DATA_DIR / "benchmark_embedding_v2" / "bs_0064"
_EXISTING_CHUNKS_DIR = _DATA_DIR / "chunks_pilot"
_BM25_INDEX_DIR = _DATA_DIR / "benchmark_bm25"
_CACHE_DIR = _DATA_DIR / "cache" / "huggingface"


@dataclass
class ExperimentResult:
    """Collected measurements for a single experiment."""
    experiment_id: str
    config: Dict[str, Any]
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    index_memory_mb: float = 0.0
    index_build_time_sec: float = 0.0
    query_p50_ms: float = 0.0
    query_p95_ms: float = 0.0
    query_mean_ms: float = 0.0
    ingestion_throughput_chunks_sec: float = 0.0
    gpu_utilization_pct: float = 0.0
    total_vectors: int = 0
    total_chunks: int = 0
    num_queries_evaluated: int = 0
    raw_query_latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path):
        """Saves result to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class AblationRunner:
    """Orchestrates ablation experiments across all dimensions."""

    def __init__(
        self,
        output_dir: Path,
        source_parquet: Optional[str] = None,
        gpu_device: str = "cuda:0",
        skip_completed: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.configs_dir = self.output_dir / "configs"
        self.indexes_dir = self.output_dir / "indexes"
        self.source_parquet = source_parquet or str(_DATA_DIR / "raw" / "wikipedia.parquet")
        self.gpu_device = gpu_device
        self.skip_completed = skip_completed

        # Lazy-loaded components
        self._embedder = None
        self._reranker = None
        self._bm25_searcher = None
        self._ground_truth: Optional[List[GroundTruthEvidence]] = None
        self._chunk_store: Dict[int, Dict[str, Any]] = {}
        self._default_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Create directories
        for d in [self.results_dir, self.configs_dir, self.indexes_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Lazy component loading
    # -----------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-loads the GPU embedder."""
        if self._embedder is None:
            from src.embedding.embedder import GPUEmbedder
            self._embedder = GPUEmbedder(
                model_name="BAAI/bge-small-en-v1.5",
                cache_dir=str(_CACHE_DIR),
                device=self.gpu_device,
                use_fp16=True,
                expected_dim=384,
            )
        return self._embedder

    def _get_reranker(self):
        """Lazy-loads the cross-encoder reranker."""
        if self._reranker is None:
            from src.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                model_name="BAAI/bge-reranker-base",
                cache_dir=str(_CACHE_DIR),
                device=self.gpu_device,
                use_fp16=True,
            )
        return self._reranker

    def _get_bm25_searcher(self):
        """Lazy-loads the BM25 searcher."""
        if self._bm25_searcher is None:
            from src.retrieval.bm25_searcher import BM25Searcher
            if _BM25_INDEX_DIR.is_dir() and (_BM25_INDEX_DIR / "matrix.npz").is_file():
                self._bm25_searcher = BM25Searcher(index_dir=str(_BM25_INDEX_DIR))
                logger.info(f"BM25 searcher loaded from {_BM25_INDEX_DIR}")
            else:
                logger.warning("BM25 index not found; sparse retrieval will be unavailable")
        return self._bm25_searcher

    def _load_default_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads the pre-computed embeddings from the default directory."""
        if self._default_embeddings is not None:
            return self._default_embeddings

        all_vecs = []
        all_ids = []
        for shard in sorted(_EXISTING_EMBEDDINGS_DIR.glob("embeddings_*.parquet")):
            table = pq.read_table(str(shard), columns=["chunk_id", "embedding"])
            chunk_ids = np.array(table.column("chunk_id").to_numpy(zero_copy_only=False), dtype=np.int64)
            flat = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
            vecs = flat.reshape(-1, 384).astype(np.float32)
            all_vecs.append(vecs)
            all_ids.append(chunk_ids)

        embeddings = np.concatenate(all_vecs, axis=0)
        chunk_ids = np.concatenate(all_ids, axis=0)
        self._default_embeddings = (embeddings, chunk_ids)
        logger.info(f"Loaded {len(chunk_ids):,} pre-computed embeddings from {_EXISTING_EMBEDDINGS_DIR}")
        return self._default_embeddings

    def _load_chunk_store(self, chunks_dir: Path) -> Dict[int, Dict[str, Any]]:
        """Loads chunk metadata into an in-memory store for hydration."""
        store: Dict[int, Dict[str, Any]] = {}
        shard_files = sorted(chunks_dir.glob("chunks_*.parquet"))[:2]
        for shard in shard_files:
            table = pq.read_table(str(shard))
            for i in range(table.num_rows):
                cid = int(table.column("chunk_id")[i].as_py())
                store[cid] = {
                    "chunk_id": cid,
                    "doc_id": int(table.column("doc_id")[i].as_py()),
                    "title": str(table.column("title")[i].as_py()),
                    "url": str(table.column("url")[i].as_py()),
                    "section_path": str(table.column("section_path")[i].as_py()),
                    "text": str(table.column("text")[i].as_py()),
                }
        logger.info(f"Loaded {len(store):,} chunks from {chunks_dir}")
        return store

    def _prepare_ground_truth(self, chunks_dir: Path) -> List[GroundTruthEvidence]:
        """Maps curated QA samples to ground-truth evidence in the given chunk corpus."""
        import pandas as pd

        # Load chunks from the first 2 shards only (matching evaluation benchmark scope)
        dfs = []
        shard_files = sorted(chunks_dir.glob("chunks_*.parquet"))[:2]
        for shard in shard_files:
            dfs.append(pq.read_table(str(shard)).to_pandas())

        if not dfs:
            raise ValueError(f"No chunk shards found in {chunks_dir}")

        chunks_df = pd.concat(dfs, ignore_index=True)
        samples = get_curated_pilot_benchmark_samples()

        mapper = EvidenceMapper(min_answer_len_for_span=3)
        evidence_list, diagnostics = mapper.map_dataset(samples, chunks_df)

        grounded = [ev for ev in evidence_list if ev.is_grounded_in_corpus]
        logger.info(
            f"Ground truth: {diagnostics.grounded_queries}/{diagnostics.total_queries} grounded, "
            f"{diagnostics.total_positive_chunks} positive chunks"
        )
        return grounded

    # -----------------------------------------------------------------------
    # Index building
    # -----------------------------------------------------------------------

    def _build_index(
        self,
        config: ExperimentConfig,
        embeddings: np.ndarray,
        chunk_ids: np.ndarray,
    ) -> Tuple[Any, float, float]:
        """Builds a FAISS index per the experiment config.

        Returns: (indexer, build_time_sec, disk_size_mb)
        """
        from src.indexing.faiss_indexer import FAISSIndexer

        indexer = FAISSIndexer(
            dim=384,
            index_type=config.index_type,
            metric="INNER_PRODUCT",
            nlist=config.nlist,
            m_subquantizers=config.m_subquantizers,
            bits_per_code=config.bits_per_code,
            auto_nlist=True,
        )

        t0 = time.perf_counter()

        # Train if needed (IVF-based indexes)
        if not indexer.is_trained:
            indexer.train(embeddings)

        # Add vectors
        indexer.add_vectors(embeddings, chunk_ids, skip_duplicates=True)
        build_time = time.perf_counter() - t0

        # Save and measure disk size
        index_path = self.indexes_dir / f"{config.experiment_id}.index"
        indexer.save(index_path)
        disk_mb = index_path.stat().st_size / (1024 * 1024)

        return indexer, build_time, disk_mb

    # -----------------------------------------------------------------------
    # Query evaluation
    # -----------------------------------------------------------------------

    def _evaluate_queries_dense(
        self,
        indexer,
        config: ExperimentConfig,
        ground_truth: List[GroundTruthEvidence],
    ) -> Tuple[float, float, float, float, float, float, List[float]]:
        """Evaluates dense retrieval queries against ground truth.

        Returns: (recall@10, mrr, ndcg@10, p50_ms, p95_ms, mean_ms, latencies)
        """
        embedder = self._get_embedder()
        top_k = 50
        recalls = []
        mrrs = []
        ndcgs = []
        latencies = []

        for ev in ground_truth:
            t0 = time.perf_counter()
            q_vec = embedder.embed_batch([ev.question])

            # Set nprobe for IVF indexes
            if hasattr(indexer.index, "nprobe"):
                indexer.index.nprobe = config.nprobe

            scores, ids = indexer.search(q_vec, top_k=top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            retrieved = [int(x) for x in ids[0] if int(x) >= 0]
            recalls.append(compute_recall_at_k(retrieved, ev.gold_chunk_ids, k=10))
            mrrs.append(compute_mrr(retrieved, ev.gold_chunk_ids))
            ndcgs.append(compute_ndcg_at_k(retrieved, ev.relevance_scores, k=10))

        stats = compute_latency_stats(latencies)
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

        return mean_recall, mean_mrr, mean_ndcg, stats.p50_ms, stats.p95_ms, stats.mean_ms, latencies

    async def _evaluate_queries_pipeline(
        self,
        config: ExperimentConfig,
        indexer,
        ground_truth: List[GroundTruthEvidence],
        chunk_store: Dict[int, Dict[str, Any]],
    ) -> Tuple[float, float, float, float, float, float, List[float]]:
        """Evaluates queries using the full retrieval pipeline (for hybrid/reranking modes)."""
        from src.retrieval.pipeline import HybridRetrievalPipeline
        from src.config import RetrievalConfig
        from src.evaluation.runner import EvaluationRunner, RetrievalMode

        retrieval_config = RetrievalConfig(
            dense_top_k=config.dense_top_k,
            sparse_top_k=config.sparse_top_k,
            rrf_k=config.rrf_k,
            final_rerank_top_k=config.final_rerank_top_k,
        )

        # Set nprobe on the indexer
        if hasattr(indexer.index, "nprobe"):
            indexer.index.nprobe = config.nprobe

        pipeline = HybridRetrievalPipeline(
            embedder=self._get_embedder(),
            faiss_indexer=indexer,
            reranker=self._get_reranker() if config.retrieval_mode == "hybrid_rerank" else None,
            bm25_searcher=self._get_bm25_searcher(),
            chunk_store=chunk_store,
            config=retrieval_config,
        )

        mode_map = {
            "dense": RetrievalMode.DENSE,
            "sparse": RetrievalMode.SPARSE,
            "hybrid_rrf": RetrievalMode.HYBRID_RRF,
            "hybrid_rerank": RetrievalMode.HYBRID_RERANK,
        }
        mode = mode_map[config.retrieval_mode]

        runner = EvaluationRunner(
            pipeline=pipeline,
            k_values=(1, 5, 10, 20, 50),
            top_k_retrieve=50,
            final_top_k=config.final_rerank_top_k,
        )

        recalls = []
        mrrs = []
        ndcgs = []
        latencies = []

        for ev in ground_truth:
            result = await runner.evaluate_single_query(ev, mode)
            recalls.append(result.recall_at_k.get(10, 0.0))
            mrrs.append(result.mrr)
            ndcgs.append(result.ndcg_at_k.get(10, 0.0))
            total_lat = result.latency_ms.get("total_latency_ms", 0.0)
            latencies.append(total_lat)

        stats = compute_latency_stats(latencies)
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

        return mean_recall, mean_mrr, mean_ndcg, stats.p50_ms, stats.p95_ms, stats.mean_ms, latencies

    # -----------------------------------------------------------------------
    # Chunk-size experiment helpers
    # -----------------------------------------------------------------------

    def _rechunk_articles(
        self,
        config: ExperimentConfig,
        output_chunks_dir: Path,
    ) -> int:
        """Re-chunks source articles with a different target token count.

        Reads the first 2 shards of the existing pilot chunks (which contain the
        raw text) and re-chunks them. This avoids needing the original raw Parquet.
        """
        from src.ingestion.chunker import HeadingAwareChunker
        from src.common.schemas import RawArticle

        output_chunks_dir.mkdir(parents=True, exist_ok=True)

        chunker = HeadingAwareChunker(
            target_tokens=config.chunk_target_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
            min_tokens=config.chunk_min_tokens,
        )

        # Read existing chunks to extract unique articles
        articles: Dict[int, RawArticle] = {}
        shard_files = sorted(_EXISTING_CHUNKS_DIR.glob("chunks_*.parquet"))[:2]

        for shard_path in shard_files:
            table = pq.read_table(str(shard_path))
            for i in range(table.num_rows):
                doc_id = int(table.column("doc_id")[i].as_py())
                if doc_id not in articles:
                    # Extract a "raw" article from the chunk — use the text minus the header
                    text = str(table.column("text")[i].as_py())
                    title = str(table.column("title")[i].as_py())
                    url = str(table.column("url")[i].as_py())
                    articles[doc_id] = RawArticle(
                        id=doc_id,
                        title=title,
                        url=url,
                        text="",  # Will accumulate
                        categories=[],
                    )
                # Accumulate text (strip header prefix if present)
                raw_text = str(table.column("text")[i].as_py())
                # Remove the header line "Title - Section > Path\n"
                lines = raw_text.split("\n", 1)
                body = lines[1] if len(lines) > 1 else lines[0]
                articles[doc_id] = RawArticle(
                    id=articles[doc_id].id,
                    title=articles[doc_id].title,
                    url=articles[doc_id].url,
                    text=articles[doc_id].text + "\n\n" + body,
                    categories=[],
                )

        # Re-chunk each article
        all_chunks = []
        for doc_id, article in sorted(articles.items()):
            chunks = chunker.chunk_article(article)
            all_chunks.extend(chunks)

        # Write to parquet shard
        if all_chunks:
            table = pa.table({
                "chunk_id": pa.array([int(c.chunk_id) for c in all_chunks], type=pa.int64()),
                "doc_id": pa.array([int(c.doc_id) for c in all_chunks], type=pa.int64()),
                "title": pa.array([str(c.title) for c in all_chunks], type=pa.string()),
                "url": pa.array([str(c.url) for c in all_chunks], type=pa.string()),
                "section_path": pa.array([str(c.section_path) for c in all_chunks], type=pa.string()),
                "text": pa.array([str(c.text) for c in all_chunks], type=pa.string()),
                "token_count": pa.array([int(c.token_count) for c in all_chunks], type=pa.int32()),
                "faiss_id": pa.array([int(c.faiss_id) for c in all_chunks], type=pa.int64()),
            })
            pq.write_table(table, str(output_chunks_dir / "chunks_0000.parquet"), compression="snappy")

        logger.info(
            f"Re-chunked {len(articles)} articles → {len(all_chunks)} chunks "
            f"(target={config.chunk_target_tokens} tokens)"
        )
        return len(all_chunks)

    def _embed_chunks(self, chunks_dir: Path, output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Embeds all chunks in a directory using the GPU embedder."""
        output_dir.mkdir(parents=True, exist_ok=True)
        embedder = self._get_embedder()

        all_texts = []
        all_ids = []
        for shard in sorted(chunks_dir.glob("chunks_*.parquet")):
            table = pq.read_table(str(shard))
            texts = table.column("text").to_pylist()
            ids = np.array(table.column("chunk_id").to_numpy(zero_copy_only=False), dtype=np.int64)
            all_texts.extend(texts)
            all_ids.append(ids)

        chunk_ids = np.concatenate(all_ids, axis=0) if all_ids else np.array([], dtype=np.int64)

        # Embed in batches
        batch_size = 64
        all_vecs = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            vecs = embedder.embed_batch(batch)
            all_vecs.append(vecs)

        embeddings = np.concatenate(all_vecs, axis=0) if all_vecs else np.empty((0, 384), dtype=np.float32)

        # Save to parquet
        flat_emb = embeddings.flatten().tolist()
        emb_list = [flat_emb[i * 384:(i + 1) * 384] for i in range(len(chunk_ids))]
        table = pa.table({
            "chunk_id": pa.array(chunk_ids.tolist(), type=pa.int64()),
            "embedding": pa.array(emb_list, type=pa.list_(pa.float32())),
        })
        pq.write_table(table, str(output_dir / "embeddings_0000.parquet"), compression="snappy")

        logger.info(f"Embedded {len(chunk_ids):,} chunks → {output_dir}")
        return embeddings, chunk_ids

    # -----------------------------------------------------------------------
    # Experiment executors
    # -----------------------------------------------------------------------

    def _run_index_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Runs an index-type or nprobe experiment using existing embeddings."""
        result = ExperimentResult(experiment_id=config.experiment_id, config=config.to_dict())

        try:
            embeddings, chunk_ids = self._load_default_embeddings()

            # Build index
            indexer, build_time, disk_mb = self._build_index(config, embeddings, chunk_ids)
            result.index_build_time_sec = round(build_time, 3)
            result.index_memory_mb = round(disk_mb, 2)
            result.total_vectors = indexer.ntotal

            # Load chunk store and ground truth
            if not self._chunk_store:
                self._chunk_store = self._load_chunk_store(_EXISTING_CHUNKS_DIR)
            if self._ground_truth is None:
                self._ground_truth = self._prepare_ground_truth(_EXISTING_CHUNKS_DIR)

            # Evaluate
            recall, mrr, ndcg, p50, p95, mean_lat, latencies = self._evaluate_queries_dense(
                indexer, config, self._ground_truth
            )
            result.recall_at_10 = round(recall, 4)
            result.mrr = round(mrr, 4)
            result.ndcg_at_10 = round(ndcg, 4)
            result.query_p50_ms = round(p50, 3)
            result.query_p95_ms = round(p95, 3)
            result.query_mean_ms = round(mean_lat, 3)
            result.num_queries_evaluated = len(self._ground_truth)
            result.raw_query_latencies_ms = [round(x, 3) for x in latencies]

        except Exception as e:
            logger.error(f"Experiment {config.experiment_id} failed: {e}")
            result.errors.append(str(e))

        return result

    def _run_retrieval_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Runs a retrieval-mode or reranking experiment using the full pipeline."""
        result = ExperimentResult(experiment_id=config.experiment_id, config=config.to_dict())

        try:
            embeddings, chunk_ids = self._load_default_embeddings()

            # Build FlatIP index (exact baseline for retrieval experiments)
            from src.indexing.faiss_indexer import FAISSIndexer
            indexer = FAISSIndexer(dim=384, index_type="FlatIP", metric="INNER_PRODUCT")
            indexer.add_vectors(embeddings, chunk_ids)
            result.total_vectors = indexer.ntotal

            # Load chunk store and ground truth
            if not self._chunk_store:
                self._chunk_store = self._load_chunk_store(_EXISTING_CHUNKS_DIR)
            if self._ground_truth is None:
                self._ground_truth = self._prepare_ground_truth(_EXISTING_CHUNKS_DIR)

            # Evaluate with the full pipeline
            with GPUUtilizationMonitor(device_id=0) as gpu_mon:
                recall, mrr, ndcg, p50, p95, mean_lat, latencies = asyncio.run(
                    self._evaluate_queries_pipeline(
                        config, indexer, self._ground_truth, self._chunk_store
                    )
                )

            result.recall_at_10 = round(recall, 4)
            result.mrr = round(mrr, 4)
            result.ndcg_at_10 = round(ndcg, 4)
            result.query_p50_ms = round(p50, 3)
            result.query_p95_ms = round(p95, 3)
            result.query_mean_ms = round(mean_lat, 3)
            result.num_queries_evaluated = len(self._ground_truth)
            result.raw_query_latencies_ms = [round(x, 3) for x in latencies]
            result.gpu_utilization_pct = gpu_mon.mean_utilization

        except Exception as e:
            logger.error(f"Experiment {config.experiment_id} failed: {e}")
            result.errors.append(str(e))

        return result

    def _run_chunk_size_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Runs a chunk-size experiment: re-chunk, re-embed, build index, evaluate."""
        result = ExperimentResult(experiment_id=config.experiment_id, config=config.to_dict())

        try:
            chunks_dir = self.output_dir / f"chunks_{config.chunk_target_tokens}"
            embed_dir = self.output_dir / f"embeddings_{config.chunk_target_tokens}"

            # Check if 256 is the default — reuse existing data
            if config.chunk_target_tokens == 256:
                chunks_dir = _EXISTING_CHUNKS_DIR
                embeddings, chunk_ids = self._load_default_embeddings()
                total_chunks = len(chunk_ids)
            else:
                # Re-chunk
                with GPUUtilizationMonitor(device_id=0) as gpu_mon:
                    total_chunks = self._rechunk_articles(config, chunks_dir)
                    # Re-embed
                    embeddings, chunk_ids = self._embed_chunks(chunks_dir, embed_dir)
                result.gpu_utilization_pct = gpu_mon.mean_utilization

            result.total_chunks = total_chunks

            # Build FlatIP index
            indexer, build_time, disk_mb = self._build_index(config, embeddings, chunk_ids)
            result.index_build_time_sec = round(build_time, 3)
            result.index_memory_mb = round(disk_mb, 2)
            result.total_vectors = indexer.ntotal

            # Prepare ground truth for this chunk set
            ground_truth = self._prepare_ground_truth(chunks_dir)

            if not ground_truth:
                result.errors.append("No grounded queries found for this chunk configuration")
                return result

            # Evaluate
            recall, mrr, ndcg, p50, p95, mean_lat, latencies = self._evaluate_queries_dense(
                indexer, config, ground_truth
            )
            result.recall_at_10 = round(recall, 4)
            result.mrr = round(mrr, 4)
            result.ndcg_at_10 = round(ndcg, 4)
            result.query_p50_ms = round(p50, 3)
            result.query_p95_ms = round(p95, 3)
            result.query_mean_ms = round(mean_lat, 3)
            result.num_queries_evaluated = len(ground_truth)
            result.raw_query_latencies_ms = [round(x, 3) for x in latencies]

        except Exception as e:
            logger.error(f"Experiment {config.experiment_id} failed: {e}")
            result.errors.append(str(e))

        return result

    def _run_worker_count_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Runs a worker-count experiment: measures ingestion throughput."""
        result = ExperimentResult(experiment_id=config.experiment_id, config=config.to_dict())

        try:
            from src.config import AppConfig, IngestionConfig, ChunkingConfig
            from src.common.checkpoint import PipelineCheckpoint
            from src.ingestion.pipeline import IngestionPipeline

            worker_output_dir = self.output_dir / f"worker_test_{config.num_workers}"
            worker_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean any previous output
            for f in worker_output_dir.glob("chunks_*.parquet"):
                f.unlink()
            for f in worker_output_dir.glob("*.tmp"):
                f.unlink()

            # Configure with the specified worker count
            chunking = ChunkingConfig(
                target_token_count=config.chunk_target_tokens,
                overlap_token_count=config.chunk_overlap_tokens,
                min_token_count=config.chunk_min_tokens,
            )
            ingestion_cfg = IngestionConfig(
                num_cpu_workers=config.num_workers,
                worker_chunksize=500,
                chunking=chunking,
            )
            app_config = AppConfig(ingestion=ingestion_cfg)

            checkpoint = PipelineCheckpoint(
                checkpoint_path=str(worker_output_dir / "checkpoint.json")
            )

            pipeline = IngestionPipeline(config=app_config, checkpoint=checkpoint)

            # Use the raw parquet if available, otherwise use existing chunks
            input_path = self.source_parquet
            if not Path(input_path).is_file():
                # Fallback: use first existing chunk shard as input
                # This measures chunking overhead at different worker counts
                shard_files = sorted(_EXISTING_CHUNKS_DIR.glob("chunks_*.parquet"))
                if shard_files:
                    input_path = str(shard_files[0])
                else:
                    result.errors.append("No input data available for worker-count experiment")
                    return result

            t0 = time.perf_counter()
            stats = pipeline.run(input_path, str(worker_output_dir))
            elapsed = time.perf_counter() - t0

            result.ingestion_throughput_chunks_sec = stats.get("throughput_chunks_per_sec", 0.0)
            result.total_chunks = stats.get("chunks_emitted", 0)
            result.index_build_time_sec = round(elapsed, 3)

        except Exception as e:
            logger.error(f"Worker experiment {config.experiment_id} failed: {e}")
            result.errors.append(str(e))

        return result

    # -----------------------------------------------------------------------
    # Main execution
    # -----------------------------------------------------------------------

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Dispatches a single experiment to the appropriate executor."""
        result_path = self.results_dir / f"{config.experiment_id}.json"

        # Skip if already completed
        if self.skip_completed and result_path.is_file():
            logger.info(f"Skipping completed experiment: {config.experiment_id}")
            return ExperimentResult.load(result_path)

        # Save config
        config.save_yaml(self.configs_dir / f"{config.experiment_id}.yaml")

        logger.info(f"▶ Running experiment: {config.experiment_id} ({config.description})")
        t0 = time.perf_counter()

        if config.dimension in ("index_type", "nprobe"):
            result = self._run_index_experiment(config)
        elif config.dimension in ("retrieval_mode", "reranking"):
            result = self._run_retrieval_experiment(config)
        elif config.dimension == "chunk_size":
            result = self._run_chunk_size_experiment(config)
        elif config.dimension == "worker_count":
            result = self._run_worker_count_experiment(config)
        else:
            result = ExperimentResult(
                experiment_id=config.experiment_id,
                config=config.to_dict(),
                errors=[f"Unknown dimension: {config.dimension}"],
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"✓ Experiment {config.experiment_id} complete in {elapsed:.1f}s "
            f"(Recall@10={result.recall_at_10:.4f}, p50={result.query_p50_ms:.1f}ms)"
        )

        # Save result
        result.save(result_path)
        return result

    def run_all(
        self,
        configs: List[ExperimentConfig],
    ) -> List[ExperimentResult]:
        """Runs all experiments sequentially, saving results incrementally."""
        results = []
        total = len(configs)

        for i, config in enumerate(configs, 1):
            logger.info(f"\n{'='*60}\nExperiment {i}/{total}: {config.experiment_id}\n{'='*60}")
            result = self.run_experiment(config)
            results.append(result)

        # Save aggregated results
        self._save_aggregated(results)
        return results

    def _save_aggregated(self, results: List[ExperimentResult]):
        """Saves aggregated results as JSON and CSV."""
        # JSON
        agg_path = self.output_dir / "ablation_results.json"
        agg_data = {
            "total_experiments": len(results),
            "experiments": [r.to_dict() for r in results],
        }
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg_data, f, indent=2)

        # CSV summary
        csv_path = self.output_dir / "ablation_results.csv"
        header = (
            "experiment_id,dimension,index_type,nprobe,chunk_tokens,"
            "retrieval_mode,num_workers,recall@10,mrr,ndcg@10,"
            "index_mb,build_sec,p50_ms,p95_ms,gpu_util_pct,throughput_chunks_sec"
        )
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for r in results:
                cfg = r.config
                line = (
                    f"{r.experiment_id},{cfg.get('dimension','')},{cfg.get('index_type','')},{cfg.get('nprobe','')},{cfg.get('chunk_target_tokens','')},"
                    f"{cfg.get('retrieval_mode','')},{cfg.get('num_workers','')},{r.recall_at_10},{r.mrr},{r.ndcg_at_10},"
                    f"{r.index_memory_mb},{r.index_build_time_sec},{r.query_p50_ms},{r.query_p95_ms},{r.gpu_utilization_pct},{r.ingestion_throughput_chunks_sec}"
                )
                f.write(line + "\n")

        logger.info(f"Aggregated results saved to {agg_path} and {csv_path}")
