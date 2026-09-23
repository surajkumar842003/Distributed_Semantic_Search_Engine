"""Controlled Retrieval Architecture Evaluation Runner.

Compares four distinct retrieval paradigms on identical query sets:
1. Dense Retrieval (FAISS vector search with BAAI/bge-small-en-v1.5)
2. Sparse Retrieval (PostgreSQL FTS or BM25 lexical search)
3. Hybrid Retrieval (Reciprocal Rank Fusion with k=60)
4. Hybrid Retrieval + Cross-Encoder Reranking (BAAI/bge-reranker-base)

Collects granular per-query metrics, stage-by-stage latencies, and aggregated summaries.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Sequence, Set
from dataclasses import dataclass, asdict
import time
import asyncio

from src.common.logging import get_logger
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.retrieval.rrf import reciprocal_rank_fusion
from src.evaluation.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    aggregate_retrieval_metrics,
)
from src.evaluation.dataset import QASample, GroundTruthEvidence

logger = get_logger("evaluation.runner")


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class QueryRunResult:
    """Individual query execution result for one retrieval mode."""
    query_id: str
    question: str
    mode: str
    retrieved_chunk_ids: List[int]
    scores: List[float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    latency_ms: Dict[str, float]
    is_grounded_in_corpus: bool


class EvaluationRunner:
    """Orchestrates controlled retrieval experiments across the 4 configurations."""

    def __init__(
        self,
        pipeline: HybridRetrievalPipeline,
        k_values: Sequence[int] = (1, 5, 10, 20, 50),
        top_k_retrieve: int = 50,
        final_top_k: int = 10,
    ):
        self.pipeline = pipeline
        self.k_values = tuple(k_values)
        self.top_k_retrieve = top_k_retrieve
        self.final_top_k = final_top_k

    async def evaluate_single_query(
        self,
        evidence: GroundTruthEvidence,
        mode: RetrievalMode,
    ) -> QueryRunResult:
        """Executes a query under a specified retrieval mode and evaluates metrics."""
        query = evidence.question
        gt_chunks = evidence.gold_chunk_ids
        rel_map = evidence.relevance_scores

        latencies: Dict[str, float] = {}
        t_start = time.perf_counter()
        retrieved_ids: List[int] = []
        retrieved_scores: List[float] = []

        if mode == RetrievalMode.DENSE:
            # Dense Only (FAISS)
            dense_res, dense_ms = self.pipeline._retrieve_dense(query, top_k=self.top_k_retrieve)
            latencies["dense_latency_ms"] = dense_ms
            latencies["total_latency_ms"] = dense_ms
            retrieved_ids = [cid for cid, _ in dense_res]
            retrieved_scores = [score for _, score in dense_res]

        elif mode == RetrievalMode.SPARSE:
            # Sparse Only (PostgreSQL FTS or BM25)
            sparse_res, _, sparse_ms = await self.pipeline._retrieve_sparse(query, top_k=self.top_k_retrieve)
            latencies["sparse_latency_ms"] = sparse_ms
            latencies["total_latency_ms"] = sparse_ms
            retrieved_ids = [cid for cid, _ in sparse_res]
            retrieved_scores = [score for _, score in sparse_res]

        elif mode == RetrievalMode.HYBRID_RRF:
            # Hybrid Dense + Sparse via RRF
            dense_res, dense_ms = self.pipeline._retrieve_dense(query, top_k=self.top_k_retrieve)
            sparse_res, _, sparse_ms = await self.pipeline._retrieve_sparse(query, top_k=self.top_k_retrieve)

            t_f0 = time.perf_counter()
            fused = reciprocal_rank_fusion(dense_res, sparse_res, rrf_k=self.pipeline.config.rrf_k)
            fusion_ms = (time.perf_counter() - t_f0) * 1000.0

            latencies["dense_latency_ms"] = dense_ms
            latencies["sparse_latency_ms"] = sparse_ms
            latencies["fusion_latency_ms"] = round(fusion_ms, 3)
            latencies["total_latency_ms"] = round(dense_ms + sparse_ms + fusion_ms, 3)

            retrieved_ids = [item["chunk_id"] for item in fused[:self.top_k_retrieve]]
            retrieved_scores = [float(item["rrf_score"]) for item in fused[:self.top_k_retrieve]]

        elif mode == RetrievalMode.HYBRID_RERANK:
            # Hybrid Dense + Sparse + Hydration + Cross-Encoder Reranking
            search_resp = await self.pipeline.search_async(
                query=query,
                dense_top_k=self.top_k_retrieve,
                sparse_top_k=self.top_k_retrieve,
                final_top_k=self.final_top_k,
            )
            latencies.update(search_resp.latency_ms)
            retrieved_ids = [c.chunk_id for c in search_resp.results]
            retrieved_scores = [
                float(c.rerank_score if c.rerank_score is not None else c.rrf_score)
                for c in search_resp.results
            ]

        # Calculate retrieval metrics
        recalls = {
            k: compute_recall_at_k(retrieved_ids, gt_chunks, k=k)
            for k in self.k_values
        }
        mrr = compute_mrr(retrieved_ids, gt_chunks)
        ndcgs = {
            k: compute_ndcg_at_k(retrieved_ids, rel_map, k=k)
            for k in self.k_values
        }

        return QueryRunResult(
            query_id=evidence.query_id,
            question=query,
            mode=mode.value,
            retrieved_chunk_ids=retrieved_ids,
            scores=retrieved_scores,
            recall_at_k=recalls,
            mrr=mrr,
            ndcg_at_k=ndcgs,
            latency_ms=latencies,
            is_grounded_in_corpus=evidence.is_grounded_in_corpus,
        )

    async def run_benchmark(
        self,
        ground_truth_list: Sequence[GroundTruthEvidence],
        modes: Optional[Sequence[RetrievalMode]] = None,
        filter_grounded_only: bool = False,
    ) -> Dict[str, Any]:
        """Runs controlled evaluation across all queries and configured retrieval modes.

        Args:
            ground_truth_list: List of mapped GroundTruthEvidence items.
            modes: Sequence of RetrievalMode values to compare. Defaults to all 4.
            filter_grounded_only: If True, evaluates only queries with positive evidence
                                  in the pilot corpus.

        Returns:
            Dict containing detailed per-query evaluations and aggregated comparisons.
        """
        target_modes = modes or [
            RetrievalMode.DENSE,
            RetrievalMode.SPARSE,
            RetrievalMode.HYBRID_RRF,
            RetrievalMode.HYBRID_RERANK,
        ]

        queries_to_eval = (
            [ev for ev in ground_truth_list if ev.is_grounded_in_corpus]
            if filter_grounded_only
            else list(ground_truth_list)
        )

        logger.info(
            f"Starting retrieval benchmark: {len(queries_to_eval)} queries, "
            f"modes: {[m.value for m in target_modes]}"
        )

        results_by_mode: Dict[str, List[QueryRunResult]] = {m.value: [] for m in target_modes}

        for ev in queries_to_eval:
            for mode in target_modes:
                res = await self.evaluate_single_query(ev, mode)
                results_by_mode[mode.value].append(res)

        # Aggregate metrics per mode
        summaries: Dict[str, Any] = {}
        for mode in target_modes:
            mode_runs = results_by_mode[mode.value]
            dicts = [
                {
                    "recall_at_k": r.recall_at_k,
                    "mrr": r.mrr,
                    "ndcg_at_k": r.ndcg_at_k,
                    "latency_ms": r.latency_ms,
                }
                for r in mode_runs
            ]
            agg = aggregate_retrieval_metrics(dicts, k_values=self.k_values)
            agg["mode"] = mode.value
            summaries[mode.value] = agg

        return {
            "total_queries_evaluated": len(queries_to_eval),
            "filter_grounded_only": filter_grounded_only,
            "summaries": summaries,
            "detailed_runs": {
                mode: [asdict(r) for r in runs]
                for mode, runs in results_by_mode.items()
            },
        }

