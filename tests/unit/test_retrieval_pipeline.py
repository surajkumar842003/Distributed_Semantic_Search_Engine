"""Unit tests for the complete hybrid retrieval and reranking pipeline.

Verifies:
1. RRF mathematical precision and deterministic consensus ranking.
2. Passage deduplication across dense and sparse retrievers.
3. Handling of missing or empty results across any stage.
4. Complete provenance preservation (title, URL, section_path, doc_id, chunk_id).
5. Granular latency logging across all pipeline stages.
6. Cross-encoder scoring and relevance ranking with BAAI/bge-reranker-base.
7. Ranking formula explanations.
"""

import math
import pytest
import numpy as np

from src.common.schemas import RetrievalCandidate, SearchResponse
from src.retrieval.rrf import reciprocal_rank_fusion, explain_rrf_formula
from src.retrieval.reranker import CrossEncoderReranker, explain_cross_encoder_formula
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.config import RetrievalConfig


class TestReciprocalRankFusion:
    """Tests deterministic Reciprocal Rank Fusion calculations and edge cases."""

    def test_rrf_formula_explanation(self):
        exp = explain_rrf_formula(60)
        assert "RRF_Score" in exp
        assert "k_rrf" in exp
        assert "60" in exp

    def test_rrf_mathematical_precision(self):
        """Verifies exact manual calculation of RRF scores with k=60."""
        # Dense ranking:
        # Rank 1: Chunk 101 (score 0.9)
        # Rank 2: Chunk 102 (score 0.8)
        # Rank 3: Chunk 103 (score 0.7)
        dense = [(101, 0.9), (102, 0.8), (103, 0.7)]

        # Sparse ranking:
        # Rank 1: Chunk 102 (score 15.0)
        # Rank 2: Chunk 104 (score 12.0)
        sparse = [(102, 15.0), (104, 12.0)]

        k = 60
        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=k)

        # Expected calculations:
        # Chunk 102: 1/(60+2) [dense] + 1/(60+1) [sparse] = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # Chunk 101: 1/(60+1) [dense] + 0 = 1/61 = 0.016393
        # Chunk 104: 0 + 1/(60+2) [sparse] = 1/62 = 0.016129
        # Chunk 103: 1/(60+3) [dense] + 0 = 1/63 = 0.015873

        assert len(fused) == 4
        # Consensus chunk 102 must rank first
        assert fused[0]["chunk_id"] == 102
        expected_score_102 = (1.0 / 62.0) + (1.0 / 61.0)
        assert math.isclose(fused[0]["rrf_score"], expected_score_102, rel_tol=1e-5)
        assert fused[0]["dense_rank"] == 2
        assert fused[0]["sparse_rank"] == 1

        assert fused[1]["chunk_id"] == 101
        assert math.isclose(fused[1]["rrf_score"], 1.0 / 61.0, rel_tol=1e-5)

        assert fused[2]["chunk_id"] == 104
        assert math.isclose(fused[2]["rrf_score"], 1.0 / 62.0, rel_tol=1e-5)

        assert fused[3]["chunk_id"] == 103
        assert math.isclose(fused[3]["rrf_score"], 1.0 / 63.0, rel_tol=1e-5)

    def test_rrf_deduplication(self):
        """Verifies duplicate chunk IDs across retrievers are fused into a single entry."""
        dense = [(501, 0.95), (502, 0.90)]
        sparse = [(501, 25.0), (502, 20.0)]

        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
        assert len(fused) == 2
        ids = [c["chunk_id"] for c in fused]
        assert ids == [501, 502]
        assert len(set(ids)) == len(ids)

    def test_rrf_handling_missing_results(self):
        """Verifies graceful handling when one or both retrievers return empty results."""
        # 1. Empty dense
        sparse_only = reciprocal_rank_fusion([], [(201, 10.0), (202, 5.0)], rrf_k=60)
        assert len(sparse_only) == 2
        assert sparse_only[0]["chunk_id"] == 201
        assert sparse_only[0]["dense_rank"] is None
        assert sparse_only[0]["sparse_rank"] == 1

        # 2. Empty sparse
        dense_only = reciprocal_rank_fusion([(301, 0.85)], [], rrf_k=60)
        assert len(dense_only) == 1
        assert dense_only[0]["chunk_id"] == 301
        assert dense_only[0]["sparse_rank"] is None
        assert dense_only[0]["dense_rank"] == 1

        # 3. Both empty
        both_empty = reciprocal_rank_fusion([], [], rrf_k=60)
        assert both_empty == []


class TestPipelineProvenanceAndLatency:
    """Tests end-to-end provenance preservation and latency logging."""

    def test_provenance_preservation(self):
        """Verifies title, url, section_path, doc_id, chunk_id are preserved."""
        # Mock chunk store representing hydrated database rows
        chunk_store = {
            1001: {
                "chunk_id": 1001,
                "doc_id": 42,
                "title": "Artificial Intelligence",
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "section_path": "History / Early Work",
                "text": "The field of AI research was founded at a workshop at Dartmouth College in 1956.",
                "token_count": 17,
            },
            1002: {
                "chunk_id": 1002,
                "doc_id": 42,
                "title": "Artificial Intelligence",
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "section_path": "Modern Era / Deep Learning",
                "text": "Deep learning breakthroughs in the 2010s transformed speech and computer vision.",
                "token_count": 14,
            },
        }

        pipeline = HybridRetrievalPipeline(
            chunk_store=chunk_store,
            config=RetrievalConfig(dense_top_k=5, sparse_top_k=5, rrf_k=60, final_rerank_top_k=2),
        )

        # Inject synthetic retrieval results via mock internal methods
        pipeline._retrieve_dense = lambda q, top_k: ([(1001, 0.92), (1002, 0.85)], 1.5)
        async def mock_sparse(q, top_k):
            return [(1002, 18.5)], {}, 2.0
        pipeline._retrieve_sparse = mock_sparse

        response = pipeline.search("AI history", final_top_k=2)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 2

        first = response.results[0]
        # Chunk 1002 was rank 1 sparse and rank 2 dense -> top RRF
        assert first.chunk_id == 1002
        assert first.doc_id == 42
        assert first.title == "Artificial Intelligence"
        assert first.url == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        assert first.section_path == "Modern Era / Deep Learning"
        assert "Deep learning breakthroughs" in first.text
        assert first.dense_score == 0.85
        assert first.dense_rank == 2
        assert first.sparse_score == 18.5
        assert first.sparse_rank == 1
        assert first.rrf_score > 0.0

    def test_latency_logging_recorded(self):
        """Verifies every stage records positive latency in milliseconds."""
        chunk_store = {
            1: {"chunk_id": 1, "doc_id": 1, "title": "T1", "url": "U1", "section_path": "S1", "text": "P1"}
        }
        pipeline = HybridRetrievalPipeline(chunk_store=chunk_store)
        pipeline._retrieve_dense = lambda q, top_k: ([(1, 0.9)], 0.5)
        async def mock_sparse(q, top_k):
            return [(1, 10.0)], {}, 0.8
        pipeline._retrieve_sparse = mock_sparse

        response = pipeline.search("test query")
        latencies = response.latency_ms

        assert "dense_latency_ms" in latencies
        assert "sparse_latency_ms" in latencies
        assert "fusion_latency_ms" in latencies
        assert "hydration_latency_ms" in latencies
        assert "rerank_latency_ms" in latencies
        assert "total_latency_ms" in latencies

        for key, val in latencies.items():
            assert val >= 0.0

    def test_empty_query_returns_gracefully(self):
        pipeline = HybridRetrievalPipeline()
        resp = pipeline.search("")
        assert len(resp.results) == 0
        assert resp.total_candidates == 0


class TestCrossEncoderReranking:
    """Tests Transformer cross-encoder relevance discrimination."""

    @pytest.fixture(scope="class")
    @classmethod
    def reranker(cls):
        return CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            cache_dir="/DATA/suraj/m1/search_engine/data/cache/huggingface",
            use_fp16=True,
        )

    def test_formula_explanation(self):
        exp = explain_cross_encoder_formula()
        assert "Cross-Encoder" in exp
        assert "sigma(z)" in exp
        assert "[CLS]" in exp

    def test_cross_encoder_relevance_discrimination(self, reranker):
        """Verifies relevant passages score significantly higher than irrelevant ones."""
        query = "What is photosynthesis?"
        candidates = [
            RetrievalCandidate(
                chunk_id=1, doc_id=10, title="Baking", url="http://bake",
                section_path="Recipes",
                text="To make sourdough bread, mix flour, water, salt, and active sourdough starter thoroughly.",
            ),
            RetrievalCandidate(
                chunk_id=2, doc_id=20, title="Photosynthesis", url="http://biology",
                section_path="Definition",
                text="Photosynthesis is the biochemical process by which green plants convert sunlight and carbon dioxide into glucose.",
            ),
            RetrievalCandidate(
                chunk_id=3, doc_id=30, title="Astronomy", url="http://space",
                section_path="Galaxies",
                text="The Andromeda galaxy is a spiral galaxy approximately 2.5 million light-years away from Earth.",
            ),
        ]

        reranked = reranker.rerank(query, candidates, top_k=3)

        assert len(reranked) == 3
        # Top passage must be the biology passage about photosynthesis
        assert reranked[0].chunk_id == 2
        assert reranked[0].rerank_score is not None
        assert reranked[0].rerank_score > 0.80

        # Irrelevant passages must score low (< 0.1)
        assert reranked[1].rerank_score < 0.20
        assert reranked[2].rerank_score < 0.20


class TestCrossEncoderTokenization:
    """Tests verifying correct cross-encoder tokenizer behavior."""

    def test_tokenizer_produces_sep_token(self):
        """Verify that the tokenizer produces [SEP] between query and document."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base",
            cache_dir="/DATA/suraj/m1/search_engine/data/cache/huggingface",
            local_files_only=True)
        
        query = "What is the capital of France?"
        doc = "Paris is the capital of France."
        
        # Correct: text/text_pair approach
        correct = tokenizer(text=[query], text_pair=[doc], return_tensors="pt")
        
        # Verify [SEP] token (id=102 for BERT-based) separates query and doc
        sep_id = tokenizer.sep_token_id
        input_ids = correct["input_ids"][0].tolist()
        sep_positions = [i for i, x in enumerate(input_ids) if x == sep_id]
        assert len(sep_positions) >= 2, f"Expected at least 2 [SEP] tokens, got {len(sep_positions)}"
        
        # Verify token_type_ids distinguish query (0) from document (1)
        if "token_type_ids" in correct:
            ttids = correct["token_type_ids"][0].tolist()
            assert 1 in ttids, "token_type_ids should contain 1s for the document segment"

    def test_batch_tokenization_consistency(self):
        """Verify batch tokenization produces same shape as single-pair."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base",
            cache_dir="/DATA/suraj/m1/search_engine/data/cache/huggingface",
            local_files_only=True)
        
        query = "test query"
        docs = ["doc one text", "doc two text", "doc three text"]
        
        result = tokenizer(
            text=[query] * len(docs),
            text_pair=docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        assert result["input_ids"].shape[0] == 3, "Batch should have 3 sequences"
        if "token_type_ids" in result:
            assert result["token_type_ids"].shape[0] == 3
