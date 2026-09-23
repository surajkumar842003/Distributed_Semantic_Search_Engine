"""Unit tests for the Wikipedia RAG Evaluation Framework.

Tests:
1. Recall@k mathematical precision and edge cases.
2. MRR (Mean Reciprocal Rank) first-hit behavior.
3. nDCG@k calculation (DCG, IDCG, normalized ratio).
4. Latency percentiles calculation.
5. Aggregate retrieval metrics computation.
6. Text normalization, Exact Match (EM), and Token F1.
7. Citation precision and context insufficiency detection.
8. EvidenceMapper: answer span matching, title matching, limitation flagging.
9. EvaluationRunner 4-mode execution using mock pipeline.
"""

import math
import pytest
import pandas as pd

from src.evaluation.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_latency_stats,
    aggregate_retrieval_metrics,
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
    EvidenceMapper,
    get_curated_pilot_benchmark_samples,
)
from src.evaluation.runner import (
    EvaluationRunner,
    RetrievalMode,
)
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.common.schemas import RetrievalCandidate, SearchResponse


class TestRetrievalMetrics:
    """Tests deterministic information retrieval metrics."""

    def test_recall_at_k_basic(self):
        retrieved = [101, 102, 103, 104, 105]
        ground_truth = {103, 999}

        # Candidate 103 is at rank 3 (1-indexed)
        assert compute_recall_at_k(retrieved, ground_truth, k=1) == 0.0
        assert compute_recall_at_k(retrieved, ground_truth, k=2) == 0.0
        assert compute_recall_at_k(retrieved, ground_truth, k=3) == 1.0
        assert compute_recall_at_k(retrieved, ground_truth, k=5) == 1.0
        assert compute_recall_at_k(retrieved, ground_truth, k=10) == 1.0

    def test_recall_at_k_empty(self):
        assert compute_recall_at_k([], {101}, k=5) == 0.0
        assert compute_recall_at_k([101], set(), k=5) == 0.0
        assert compute_recall_at_k([101], {101}, k=0) == 0.0

    def test_mrr_first_hit(self):
        retrieved = [101, 102, 103, 104, 105]

        # Rank 1 hit -> 1.0
        assert compute_mrr(retrieved, {101}) == 1.0

        # Rank 2 hit -> 0.5
        assert compute_mrr(retrieved, {102}) == 0.5

        # Rank 4 hit -> 0.25
        assert compute_mrr(retrieved, {104}) == 0.25

        # Multiple hits: only first hit matters
        # 102 (rank 2) and 105 (rank 5) -> MRR must be 1/2 = 0.5
        assert compute_mrr(retrieved, {102, 105}) == 0.5

        # No hit -> 0.0
        assert compute_mrr(retrieved, {999}) == 0.0
        assert compute_mrr([], {101}) == 0.0

    def test_ndcg_at_k_perfect_ranking(self):
        retrieved = [101, 102, 103]
        relevance_map = {101: 2.0, 102: 1.0, 103: 0.0}

        # Retrieved order matches ideal order -> nDCG must be 1.0
        ndcg_3 = compute_ndcg_at_k(retrieved, relevance_map, k=3)
        assert math.isclose(ndcg_3, 1.0, rel_tol=1e-5)

    def test_ndcg_at_k_inverted_ranking(self):
        # Retrieved order is inverted compared to ideal
        retrieved = [102, 101]  # 102 has rel=1.0, 101 has rel=2.0
        relevance_map = {101: 2.0, 102: 1.0}

        # DCG@2:
        # rank 1: (2^1 - 1)/log2(2) = 1.0 / 1.0 = 1.0
        # rank 2: (2^2 - 1)/log2(3) = 3.0 / 1.5849625 = 1.892789
        # DCG@2 = 1.0 + 1.892789 = 2.892789
        # IDCG@2 (ideal order: 101 then 102):
        # rank 1: (2^2 - 1)/log2(2) = 3.0
        # rank 2: (2^1 - 1)/log2(3) = 1.0 / 1.5849625 = 0.6309297
        # IDCG@2 = 3.0 + 0.6309297 = 3.6309297
        # Expected nDCG = 2.892789 / 3.6309297 = 0.7967
        ndcg = compute_ndcg_at_k(retrieved, relevance_map, k=2)
        expected_ndcg = (1.0 + 3.0 / math.log2(3.0)) / (3.0 + 1.0 / math.log2(3.0))
        assert math.isclose(ndcg, expected_ndcg, rel_tol=1e-4)
        assert 0.0 < ndcg < 1.0

    def test_latency_stats(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_latency_stats(data)
        assert stats.count == 5
        assert stats.mean_ms == 30.0
        assert stats.p50_ms == 30.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 50.0

    def test_aggregate_retrieval_metrics(self):
        query_runs = [
            {
                "recall_at_k": {1: 1.0, 5: 1.0},
                "mrr": 1.0,
                "ndcg_at_k": {5: 1.0},
                "latency_ms": {"total_latency_ms": 10.0},
            },
            {
                "recall_at_k": {1: 0.0, 5: 1.0},
                "mrr": 0.5,
                "ndcg_at_k": {5: 0.8},
                "latency_ms": {"total_latency_ms": 20.0},
            },
        ]
        agg = aggregate_retrieval_metrics(query_runs, k_values=(1, 5))
        assert agg["num_queries"] == 2
        assert agg["mean_recall"]["recall@1"] == 0.5
        assert agg["mean_recall"]["recall@5"] == 1.0
        assert agg["mean_mrr"] == 0.75
        assert agg["mean_ndcg"]["ndcg@5"] == 0.9
        assert agg["latency"]["total_latency_ms"]["mean_ms"] == 15.0


class TestGenerationMetrics:
    """Tests answer generation and grounded citation evaluation."""

    def test_normalize_answer(self):
        assert normalize_answer("The United States of America.") == "united states of america"
        assert normalize_answer("  An apple!  ") == "apple"
        assert normalize_answer("A 1994 summer.") == "1994 summer"

    def test_compute_exact_match(self):
        gts = ["San Francisco 49ers", "The 49ers"]
        assert compute_exact_match("San Francisco 49ers", gts) == 1.0
        assert compute_exact_match("san francisco 49ers", gts) == 1.0
        assert compute_exact_match("49ers", gts) == 1.0  # matches 'The 49ers' after article stripping
        assert compute_exact_match("Oakland Raiders", gts) == 0.0

    def test_compute_token_f1(self):
        gts = ["San Francisco 49ers"]
        # Perfect match
        res = compute_token_f1("San Francisco 49ers", gts)
        assert res["f1"] == 1.0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0

        # Partial match: predicted '49ers team' (common: '49ers')
        # pred has 2 tokens ('49ers', 'team'), gt has 3 tokens ('san', 'francisco', '49ers')
        # precision: 1/2 = 0.5, recall: 1/3 = 0.3333
        # f1: 2 * 0.5 * (1/3) / (0.5 + 1/3) = (1/3) / (5/6) = 2/5 = 0.4
        res2 = compute_token_f1("49ers team", gts)
        assert math.isclose(res2["f1"], 0.4, rel_tol=1e-3)
        assert res2["precision"] == 0.5
        assert math.isclose(res2["recall"], 0.3333, rel_tol=1e-3)

    def test_citation_precision(self):
        gt_chunks = {101, 102}
        assert compute_citation_precision([101, 102], gt_chunks) == 1.0
        assert compute_citation_precision([101, 999], gt_chunks) == 0.5
        assert compute_citation_precision([888, 999], gt_chunks) == 0.0

    def test_evaluate_generation_quality(self):
        quality = evaluate_generation_quality(
            prediction="piano",
            gold_answers=["piano", "jazz piano"],
            cited_chunk_ids=[101],
            ground_truth_chunk_ids={101},
            insufficient_context_flag=False,
        )
        assert quality["exact_match"] == 1.0
        assert quality["token_f1"] == 1.0
        assert quality["citation_precision"] == 1.0
        assert quality["correct_abstention"] == 1.0


class TestEvidenceMapper:
    """Tests corpus ground-truth mapping and limitation tagging."""

    def test_mapping_with_exact_span_and_title(self):
        chunks_data = [
            {
                "chunk_id": 1,
                "doc_id": 10,
                "title": "Brad Mehldau",
                "text": "Brad Mehldau is an American jazz pianist. He plays the piano in trios.",
            },
            {
                "chunk_id": 2,
                "doc_id": 20,
                "title": "Oscar Peterson",
                "text": "Oscar Peterson was a Canadian jazz pianist.",
            },
        ]
        chunks_df = pd.DataFrame(chunks_data)

        sample = QASample(
            query_id="q1",
            question="What instrument does Brad Mehldau play?",
            answers=["piano"],
            entity_titles=["Brad Mehldau"],
        )

        mapper = EvidenceMapper(min_answer_len_for_span=3)
        evidence = mapper.map_query_to_chunks(sample, chunks_df)

        assert evidence.is_grounded_in_corpus is True
        assert 1 in evidence.gold_chunk_ids
        # Chunk 1 matches both title and answer span -> relevance 2.0
        assert evidence.relevance_scores[1] == 2.0

    def test_limitation_flagging_short_answers(self):
        chunks_df = pd.DataFrame([
            {"chunk_id": 1, "doc_id": 1, "title": "History", "text": "In 94 BC, events occurred."}
        ])
        sample = QASample(
            query_id="q2",
            question="What year did it happen?",
            answers=["94"],  # Short answer < 3 chars
            entity_titles=["History"],
        )

        mapper = EvidenceMapper(min_answer_len_for_span=3)
        evidence = mapper.map_query_to_chunks(sample, chunks_df)

        assert len(evidence.limitations_flagged) > 0
        assert "Potential lexical false positive" in evidence.limitations_flagged[0]


class TestEvaluationRunnerMock:
    """Tests controlled 4-mode execution using a mock retrieval pipeline."""

    def test_runner_execution_across_all_modes(self):
        import asyncio

        # Create synthetic pipeline with mock retrieval methods
        pipeline = HybridRetrievalPipeline()

        # Mock dense retrieval
        pipeline._retrieve_dense = lambda query, top_k=50: ([(101, 0.95), (102, 0.85)], 5.0)

        # Mock sparse retrieval
        async def mock_sparse(query, top_k=50):
            return [(102, 12.0), (103, 10.0)], {}, 8.0
        pipeline._retrieve_sparse = mock_sparse

        # Mock end-to-end search_async (for hybrid_rerank)
        async def mock_search_async(**kwargs):
            return SearchResponse(
                query=kwargs.get("query", ""),
                results=[
                    RetrievalCandidate(
                        chunk_id=102, doc_id=1, title="T", url="U", section_path="S",
                        text="txt", rrf_score=0.03, rerank_score=0.98
                    ),
                    RetrievalCandidate(
                        chunk_id=101, doc_id=1, title="T", url="U", section_path="S",
                        text="txt", rrf_score=0.016, rerank_score=0.72
                    ),
                ],
                total_candidates=2,
                latency_ms={"total_latency_ms": 25.0, "dense_latency_ms": 5.0, "sparse_latency_ms": 8.0, "rerank_latency_ms": 10.0}
            )
        pipeline.search_async = mock_search_async

        runner = EvaluationRunner(pipeline=pipeline, k_values=(1, 5), top_k_retrieve=5, final_top_k=2)

        evidence = GroundTruthEvidence(
            query_id="test_q1",
            question="Where is Columbus located?",
            gold_answers=["Ohio"],
            gold_chunk_ids={102},
            relevance_scores={102: 1.0},
            is_grounded_in_corpus=True,
        )

        # Run benchmark across all 4 modes
        report = asyncio.run(runner.run_benchmark([evidence]))

        assert report["total_queries_evaluated"] == 1
        summaries = report["summaries"]
        assert "dense" in summaries
        assert "sparse" in summaries
        assert "hybrid_rrf" in summaries
        assert "hybrid_rerank" in summaries

        # In dense mode: candidates are [101, 102] -> rank 1 is 101, rank 2 is 102
        # Recall@1 = 0, Recall@5 = 1.0, MRR = 0.5
        assert summaries["dense"]["mean_recall"]["recall@1"] == 0.0
        assert summaries["dense"]["mean_recall"]["recall@5"] == 1.0
        assert summaries["dense"]["mean_mrr"] == 0.5

        # In sparse mode: candidates are [102, 103] -> rank 1 is 102
        # Recall@1 = 1.0, Recall@5 = 1.0, MRR = 1.0
        assert summaries["sparse"]["mean_recall"]["recall@1"] == 1.0
        assert summaries["sparse"]["mean_mrr"] == 1.0

        # In hybrid_rerank: top-1 is 102 -> Recall@1 = 1.0
        assert summaries["hybrid_rerank"]["mean_recall"]["recall@1"] == 1.0
        assert summaries["hybrid_rerank"]["mean_mrr"] == 1.0
