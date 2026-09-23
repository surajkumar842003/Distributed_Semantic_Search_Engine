"""Deterministic Retrieval Evaluation Metrics.

Computes standard Information Retrieval (IR) metrics:
- Recall@k (k in {1, 5, 10, 20, 50})
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain with binary and graded relevance)
- Latency statistical profiles (mean, std, p50, p90, p95, p99)

Strictly deterministic - no LLM hallucinations or heuristic scoring.
"""

from typing import Sequence, Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass
import math
import numpy as np


def compute_hit_rate_at_k(
    retrieved_ids: Sequence[int],
    ground_truth_ids: Union[Sequence[int], Set[int]],
    k: int,
) -> float:
    """Computes Hit Rate@k (a.k.a. Success@k) for a single query.

    Hit Rate@k = 1.0 if any ground-truth document ID is present in top-k retrieved candidates,
    else 0.0. This is the standard open-domain QA retrieval metric, where retrieving at least
    one gold evidence passage allows the downstream generator to answer.

    Note: This is distinct from set-based Recall@k, which measures the proportion of all
    relevant documents retrieved. See compute_set_recall_at_k for that metric.

    Args:
        retrieved_ids: Ranked sequence of candidate IDs.
        ground_truth_ids: Set or sequence of relevant candidate IDs.
        k: Cutoff rank (e.g., 1, 5, 10, 20, 50).

    Returns:
        float: 1.0 if a relevant item is in top-k, 0.0 otherwise.
    """
    if not retrieved_ids or not ground_truth_ids or k <= 0:
        return 0.0

    gt_set = ground_truth_ids if isinstance(ground_truth_ids, set) else set(ground_truth_ids)
    if not gt_set:
        return 0.0

    top_k_candidates = retrieved_ids[:k]
    for cid in top_k_candidates:
        if cid in gt_set:
            return 1.0
    return 0.0


# Backward-compatible alias — evaluation code references this name
compute_recall_at_k = compute_hit_rate_at_k


def compute_set_recall_at_k(
    retrieved_ids: Sequence[int],
    ground_truth_ids: Union[Sequence[int], Set[int]],
    k: int,
) -> float:
    """Computes set-based Recall@k for a single query.

    Set Recall@k = |relevant ∩ top-k| / |relevant|
    Measures the proportion of all relevant documents retrieved in the top-k.

    Args:
        retrieved_ids: Ranked sequence of candidate IDs.
        ground_truth_ids: Set or sequence of relevant candidate IDs.
        k: Cutoff rank.

    Returns:
        float: Fraction of relevant documents found in top-k, between 0.0 and 1.0.
    """
    if not retrieved_ids or not ground_truth_ids or k <= 0:
        return 0.0

    gt_set = ground_truth_ids if isinstance(ground_truth_ids, set) else set(ground_truth_ids)
    if not gt_set:
        return 0.0

    top_k_set = set(retrieved_ids[:k])
    return len(gt_set.intersection(top_k_set)) / len(gt_set)


def compute_mrr(
    retrieved_ids: Sequence[int],
    ground_truth_ids: Union[Sequence[int], Set[int]],
    k: Optional[int] = None,
) -> float:
    """Computes Reciprocal Rank (RR) for a single query.

    RR = 1 / rank of the first relevant document in retrieved_ids (1-based).
    If no relevant document is found in top-k (or the entire list), RR = 0.0.

    Args:
        retrieved_ids: Ranked sequence of candidate IDs.
        ground_truth_ids: Set or sequence of relevant candidate IDs.
        k: Optional cutoff rank. If None, considers the entire retrieved list.

    Returns:
        float: Reciprocal rank between 0.0 and 1.0.
    """
    if not retrieved_ids or not ground_truth_ids:
        return 0.0

    gt_set = ground_truth_ids if isinstance(ground_truth_ids, set) else set(ground_truth_ids)
    if not gt_set:
        return 0.0

    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    for rank, cid in enumerate(candidates, start=1):
        if cid in gt_set:
            return 1.0 / float(rank)

    return 0.0


def compute_dcg_at_k(
    relevance_scores: Sequence[float],
    k: int,
) -> float:
    """Computes Discounted Cumulative Gain (DCG@k).

    Formula:
        DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log_2(i + 1)

    Args:
        relevance_scores: Sequence of numerical relevance values in retrieved order.
        k: Cutoff rank.

    Returns:
        float: Discounted cumulative gain.
    """
    if not relevance_scores or k <= 0:
        return 0.0

    dcg = 0.0
    cutoff = min(k, len(relevance_scores))
    for i in range(cutoff):
        rel = float(relevance_scores[i])
        if rel > 0.0:
            # (2^rel - 1) / log2(rank + 1)
            rank = i + 1
            gain = (math.pow(2.0, rel) - 1.0) / math.log2(rank + 1.0)
            dcg += gain
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: Sequence[int],
    relevance_map: Dict[int, float],
    k: int,
) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@k) for a single query.

    nDCG@k = DCG@k / IDCG@k
    where IDCG@k is the ideal DCG if candidates were sorted by ideal relevance descending.

    Args:
        retrieved_ids: Ranked sequence of candidate IDs.
        relevance_map: Mapping from candidate ID to true relevance score (e.g. 1.0 or 2.0).
        k: Cutoff rank.

    Returns:
        float: nDCG@k score between 0.0 and 1.0.
    """
    if not retrieved_ids or not relevance_map or k <= 0:
        return 0.0

    # Obtain relevance scores for retrieved items
    retrieved_relevances = [
        float(relevance_map.get(cid, 0.0)) for cid in retrieved_ids[:k]
    ]

    dcg = compute_dcg_at_k(retrieved_relevances, k=k)

    # Calculate Ideal DCG (IDCG@k) using all available positive relevance scores
    ideal_relevances = sorted(
        [float(rel) for rel in relevance_map.values() if rel > 0.0],
        reverse=True,
    )

    if not ideal_relevances or ideal_relevances[0] <= 0.0:
        return 0.0

    idcg = compute_dcg_at_k(ideal_relevances, k=k)
    if idcg <= 0.0:
        return 0.0

    return min(1.0, max(0.0, dcg / idcg))


@dataclass
class LatencyStats:
    """Latency distribution statistics in milliseconds."""
    count: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": float(self.count),
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def compute_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Computes comprehensive latency percentile statistics from a list of measurements."""
    if not latencies_ms:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    return LatencyStats(
        count=len(arr),
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
    )


def aggregate_retrieval_metrics(
    query_evaluations: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = (1, 5, 10, 20, 50),
) -> Dict[str, Any]:
    """Aggregates per-query retrieval metrics into mean benchmark results.

    Args:
        query_evaluations: List of per-query evaluation dicts containing:
            - 'recall_at_k': Dict[int, float]
            - 'mrr': float
            - 'ndcg_at_k': Dict[int, float]
            - 'latency_ms': Dict[str, float]
        k_values: Tuple of cutoff k values to aggregate.

    Returns:
        Dict[str, Any]: Summary dictionary with mean Recall@k, Mean MRR, Mean nDCG@k,
        and latency breakdown percentiles.
    """
    num_queries = len(query_evaluations)
    if num_queries == 0:
        return {
            "num_queries": 0,
            "mean_recall": {k: 0.0 for k in k_values},
            "mean_mrr": 0.0,
            "mean_ndcg": {k: 0.0 for k in k_values},
            "latency": {},
        }

    # Aggregate Recall@k
    mean_recall = {}
    for k in k_values:
        recalls = [
            float(q["recall_at_k"].get(k, 0.0)) for q in query_evaluations if "recall_at_k" in q
        ]
        mean_recall[f"recall@{k}"] = round(float(np.mean(recalls)), 4) if recalls else 0.0

    # Aggregate MRR
    mrrs = [float(q.get("mrr", 0.0)) for q in query_evaluations]
    mean_mrr = round(float(np.mean(mrrs)), 4) if mrrs else 0.0

    # Aggregate nDCG@k
    mean_ndcg = {}
    for k in k_values:
        ndcgs = [
            float(q["ndcg_at_k"].get(k, 0.0)) for q in query_evaluations if "ndcg_at_k" in q
        ]
        mean_ndcg[f"ndcg@{k}"] = round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0

    # Aggregate Latencies per stage
    stage_names = ["dense_latency_ms", "sparse_latency_ms", "fusion_latency_ms", 
                   "hydration_latency_ms", "rerank_latency_ms", "total_latency_ms"]
    latency_summary = {}
    for stage in stage_names:
        stage_times = [
            float(q["latency_ms"].get(stage, 0.0))
            for q in query_evaluations
            if "latency_ms" in q and stage in q["latency_ms"]
        ]
        if stage_times:
            stats = compute_latency_stats(stage_times)
            latency_summary[stage] = stats.to_dict()

    return {
        "num_queries": num_queries,
        "mean_recall": mean_recall,
        "mean_mrr": mean_mrr,
        "mean_ndcg": mean_ndcg,
        "latency": latency_summary,
    }

