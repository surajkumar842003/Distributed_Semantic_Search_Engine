"""Wikipedia RAG Evaluation Module.

Provides:
- Deterministic IR metrics: Recall@k, MRR, nDCG@k, latency profiling
- Downstream generation metrics: Exact Match, Token F1, citation precision, context insufficiency
- Dataset loading and pilot corpus ground-truth evidence mapping
- Controlled 4-way retrieval benchmarking (Dense, Sparse, Hybrid RRF, Hybrid+Reranker)
"""

from src.evaluation.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_latency_stats,
    aggregate_retrieval_metrics,
    LatencyStats,
)
from src.evaluation.generation_metrics import (
    normalize_answer,
    compute_exact_match,
    compute_token_f1,
    compute_citation_precision,
    evaluate_generation_quality,
)
from src.evaluation.dataset import (
    QASample,
    GroundTruthEvidence,
    MappingDiagnostics,
    EvidenceMapper,
    load_qa_dataset,
    get_curated_pilot_benchmark_samples,
)
from src.evaluation.runner import (
    RetrievalMode,
    QueryRunResult,
    EvaluationRunner,
)

__all__ = [
    "compute_recall_at_k",
    "compute_mrr",
    "compute_dcg_at_k",
    "compute_ndcg_at_k",
    "compute_latency_stats",
    "aggregate_retrieval_metrics",
    "LatencyStats",
    "normalize_answer",
    "compute_exact_match",
    "compute_token_f1",
    "compute_citation_precision",
    "evaluate_generation_quality",
    "QASample",
    "GroundTruthEvidence",
    "MappingDiagnostics",
    "EvidenceMapper",
    "load_qa_dataset",
    "get_curated_pilot_benchmark_samples",
    "RetrievalMode",
    "QueryRunResult",
    "EvaluationRunner",
]

