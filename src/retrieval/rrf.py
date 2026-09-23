"""Deterministic Reciprocal Rank Fusion (RRF) for Hybrid Retrieval.

Mathematical Formulation:
Given ranked candidate lists from multiple heterogeneous retrieval systems
(e.g., M = {dense, sparse}), Reciprocal Rank Fusion computes a consensus score:

    RRF_Score(d) = SUM_{m in M} [ 1 / (k_rrf + r_m(d)) ]

where:
- r_m(d) >= 1 is the 1-based positional rank of document d in retrieval system m.
- k_rrf is a smoothing constant (default: 60) that balances high-ranked documents
  against moderate ranks in multiple systems.
- If document d does not appear in system m's top-K result list, its reciprocal rank
  contribution for that system is 0.

Key Properties:
1. Scale Invariance: Eliminates calibration issues between non-comparable score
   distributions (e.g. cosine similarity [-1, 1] vs uncalibrated BM25/ts_rank_cd scores).
2. Monotonicity: A document appearing near the top of both systems always outranks
   a document appearing near the top of only one system.
3. Deterministic: Purely mathematical ranking without LLM non-determinism.
"""

from typing import List, Tuple, Dict, Any, Optional, Sequence
from dataclasses import dataclass


def explain_rrf_formula(k_rrf: int = 60) -> str:
    """Returns mathematical explanation of the Reciprocal Rank Fusion formula."""
    return (
        f"Reciprocal Rank Fusion (RRF) Formula:\n"
        f"    RRF_Score(d) = SUM_{{m in M}} [ 1 / ({k_rrf} + r_m(d)) ]\n"
        f"where:\n"
        f"  - M = {{dense, sparse}}\n"
        f"  - r_m(d) is the 1-based rank of document d in system m (r >= 1)\n"
        f"  - {k_rrf} is the smoothing parameter (k_rrf) that prevents top-1 dominance\n"
        f"  - If document d does not appear in system m's results, 1 / ({k_rrf} + r_m(d)) = 0\n"
        f"Passages retrieved by both dense and sparse systems receive additive boosts, "
        f"rewarding consensus between lexical matching and semantic embedding similarity."
    )


def reciprocal_rank_fusion(
    dense_results: Sequence[Tuple[int, float]],
    sparse_results: Sequence[Tuple[int, float]],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Performs deterministic Reciprocal Rank Fusion across dense and sparse results.

    Args:
        dense_results: Ordered list of (chunk_id, score) tuples from dense retrieval (highest first).
        sparse_results: Ordered list of (chunk_id, score) tuples from sparse retrieval (highest first).
        rrf_k: Smoothing constant (default: 60).

    Returns:
        List[Dict[str, Any]]: Deduplicated list of candidates sorted descending by rrf_score.
            Each dict contains:
            - chunk_id: int
            - dense_score: Optional[float]
            - dense_rank: Optional[int] (1-based)
            - sparse_score: Optional[float]
            - sparse_rank: Optional[int] (1-based)
            - rrf_score: float
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")

    candidates: Dict[int, Dict[str, Any]] = {}

    # 1. Process dense retrieval results
    for rank_0, (cid, score) in enumerate(dense_results):
        cid_int = int(cid)
        rank_1 = rank_0 + 1
        contribution = 1.0 / (rrf_k + rank_1)

        if cid_int not in candidates:
            candidates[cid_int] = {
                "chunk_id": cid_int,
                "dense_score": float(score),
                "dense_rank": rank_1,
                "sparse_score": None,
                "sparse_rank": None,
                "rrf_score": contribution,
            }
        else:
            candidates[cid_int]["dense_score"] = float(score)
            candidates[cid_int]["dense_rank"] = rank_1
            candidates[cid_int]["rrf_score"] += contribution

    # 2. Process sparse retrieval results
    for rank_0, (cid, score) in enumerate(sparse_results):
        cid_int = int(cid)
        rank_1 = rank_0 + 1
        contribution = 1.0 / (rrf_k + rank_1)

        if cid_int not in candidates:
            candidates[cid_int] = {
                "chunk_id": cid_int,
                "dense_score": None,
                "dense_rank": None,
                "sparse_score": float(score),
                "sparse_rank": rank_1,
                "rrf_score": contribution,
            }
        else:
            candidates[cid_int]["sparse_score"] = float(score)
            candidates[cid_int]["sparse_rank"] = rank_1
            candidates[cid_int]["rrf_score"] += contribution

    # 3. Sort candidates descending by rrf_score
    # In case of tie, prefer candidates that appeared in dense retrieval with lower rank
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda c: (
            -c["rrf_score"],
            c["dense_rank"] if c["dense_rank"] is not None else 999999,
            c["sparse_rank"] if c["sparse_rank"] is not None else 999999,
            c["chunk_id"],
        ),
    )

    return sorted_candidates

