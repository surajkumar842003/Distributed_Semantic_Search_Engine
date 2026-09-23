"""Complete Hybrid Retrieval Pipeline.

Stages:
1. Dense Retrieval using FAISS vector search with query embedding.
2. Sparse Retrieval using PostgreSQL full-text search (with BM25 fallback).
3. Reciprocal Rank Fusion (RRF) combining dense and sparse scores.
4. Candidate Hydration: populating passage text, title, url, doc_id, chunk_id.
5. Cross-Encoder Reranking using BAAI/bge-reranker-base.

Features:
- Configurable top-k, RRF constant (k_rrf), and reranker.
- Graceful handling of empty or missing results.
- Full provenance preservation across all stages.
- Automatic passage deduplication.
- Granular latency profiling recorded for every stage.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Sequence, Union
from pathlib import Path

from src.config import AppConfig, RetrievalConfig
from src.common.schemas import RetrievalCandidate, SearchRequest, SearchResponse
from src.common.logging import get_logger
from src.embedding.embedder import GPUEmbedder
from src.indexing.faiss_indexer import FAISSIndexer
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.storage.postgres_client import PostgresClient

logger = get_logger("retrieval.pipeline")


class HybridRetrievalPipeline:
    """End-to-end multi-stage hybrid retrieval and reranking pipeline."""

    def __init__(
        self,
        embedder: Optional[GPUEmbedder] = None,
        faiss_indexer: Optional[FAISSIndexer] = None,
        postgres_client: Optional[PostgresClient] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        bm25_searcher: Optional[Any] = None,
        chunk_store: Optional[Dict[int, Dict[str, Any]]] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.config = config or RetrievalConfig()
        self.embedder = embedder
        self.faiss_indexer = faiss_indexer
        self.postgres_client = postgres_client
        self.reranker = reranker
        self.bm25_searcher = bm25_searcher
        self.chunk_store = chunk_store or {}

        logger.info(
            f"Initialized HybridRetrievalPipeline "
            f"(dense_k={self.config.dense_top_k}, sparse_k={self.config.sparse_top_k}, "
            f"rrf_k={self.config.rrf_k}, final_k={self.config.final_rerank_top_k})"
        )

    # =========================================================================
    # Stage 1: Dense Retrieval
    # =========================================================================

    def _retrieve_dense(
        self,
        query: str,
        top_k: int = 50,
    ) -> Tuple[List[Tuple[int, float]], float]:
        """Encodes query and retrieves top-k candidates from FAISS index."""
        if self.embedder is None or self.faiss_indexer is None or self.faiss_indexer.ntotal == 0:
            return [], 0.0

        t0 = time.perf_counter()
        q_vec = self.embedder.embed_batch([query])
        scores, ids = self.faiss_indexer.search(q_vec, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        if len(ids) > 0:
            for rank, (cid, score) in enumerate(zip(ids[0], scores[0])):
                cid_int = int(cid)
                if cid_int >= 0:  # FAISS padding returns -1
                    results.append((cid_int, float(score)))

        return results, round(elapsed_ms, 3)

    # =========================================================================
    # Stage 2: Sparse Retrieval
    # =========================================================================

    async def _retrieve_sparse(
        self,
        query: str,
        top_k: int = 50,
    ) -> Tuple[List[Tuple[int, float]], Dict[int, Dict[str, Any]], float]:
        """Retrieves top-k candidates using PostgreSQL FTS (or BM25 fallback)."""
        t0 = time.perf_counter()
        results: List[Tuple[int, float]] = []
        hydrated_map: Dict[int, Dict[str, Any]] = {}

        # Option A: PostgreSQL tsvector + GIN search
        if self.postgres_client is not None and await self.postgres_client.is_healthy():
            try:
                rows = await self.postgres_client.search_fts(query, top_k=top_k)
                for r in rows:
                    cid = int(r["chunk_id"])
                    score = float(r.get("rank_score", 0.0))
                    results.append((cid, score))
                    hydrated_map[cid] = r
            except Exception as e:
                logger.warning(f"PostgreSQL FTS search failed: {e}; attempting fallback.")

        # Option B: Fallback to BM25 searcher if available and no results yet
        if not results and self.bm25_searcher is not None:
            try:
                bm25_res = self.bm25_searcher.search(query, top_k=top_k)
                results = [(int(cid), float(s)) for cid, s in bm25_res]
            except Exception as e:
                logger.warning(f"BM25 fallback search failed: {e}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, hydrated_map, round(elapsed_ms, 3)

    # =========================================================================
    # Stage 4: Candidate Hydration
    # =========================================================================

    async def _hydrate_candidates(
        self,
        fused_items: List[Dict[str, Any]],
        pre_hydrated: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[RetrievalCandidate], float]:
        """Fetches metadata and passage text for all candidate chunk_ids."""
        t0 = time.perf_counter()
        needed_ids = [item["chunk_id"] for item in fused_items if item["chunk_id"] not in pre_hydrated]

        db_hydrated = {}
        if needed_ids:
            # Try PostgreSQL
            if self.postgres_client is not None and await self.postgres_client.is_healthy():
                try:
                    rows = await self.postgres_client.get_chunks_by_ids(needed_ids)
                    for r in rows:
                        db_hydrated[int(r["chunk_id"])] = r
                except Exception as e:
                    logger.warning(f"PostgreSQL candidate hydration failed: {e}")

            # Try in-memory chunk_store fallback
            for cid in needed_ids:
                if cid not in db_hydrated and cid in self.chunk_store:
                    db_hydrated[cid] = self.chunk_store[cid]

        candidates: List[RetrievalCandidate] = []
        for item in fused_items:
            cid = item["chunk_id"]
            data = pre_hydrated.get(cid) or db_hydrated.get(cid) or {}

            # Create candidate with complete provenance
            cand = RetrievalCandidate(
                chunk_id=cid,
                doc_id=int(data.get("doc_id", cid >> 16)),
                title=str(data.get("title", f"Article {cid >> 16}")),
                url=str(data.get("url", "")),
                section_path=str(data.get("section_path", "Overview")),
                text=str(data.get("text", "")),
                dense_score=item.get("dense_score"),
                dense_rank=item.get("dense_rank"),
                sparse_score=item.get("sparse_score"),
                sparse_rank=item.get("sparse_rank"),
                rrf_score=float(item.get("rrf_score", 0.0)),
                rerank_score=None,
            )
            candidates.append(cand)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return candidates, round(elapsed_ms, 3)

    # =========================================================================
    # End-to-End Search Pipeline
    # =========================================================================

    async def search_async(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> SearchResponse:
        """Executes full 5-stage hybrid retrieval and reranking asynchronously.

        Args:
            query: Query string.
            dense_top_k: Number of dense candidates from FAISS.
            sparse_top_k: Number of sparse candidates from PostgreSQL/BM25.
            rrf_k: Reciprocal rank fusion smoothing parameter.
            final_top_k: Final number of reranked passages to return.

        Returns:
            SearchResponse: High-precision candidates with full provenance and stage latencies.
        """
        start_wall = time.perf_counter()
        k_dense = dense_top_k or self.config.dense_top_k
        k_sparse = sparse_top_k or self.config.sparse_top_k
        k_rrf = rrf_k or self.config.rrf_k
        k_final = final_top_k or self.config.final_rerank_top_k

        latencies: Dict[str, float] = {}

        if not query or not query.strip():
            return SearchResponse(query=query, results=[], total_candidates=0, latency_ms={"total_latency_ms": 0.0})

        # Stage 1: Dense Retrieval (sync GPU work, offloaded to threadpool)
        dense_results, dense_ms = await asyncio.to_thread(
            self._retrieve_dense, query, k_dense
        )
        latencies["dense_latency_ms"] = dense_ms

        # Stage 2: Sparse Retrieval
        sparse_results, pre_hydrated, sparse_ms = await self._retrieve_sparse(query, top_k=k_sparse)
        latencies["sparse_latency_ms"] = sparse_ms

        # Stage 3: Reciprocal Rank Fusion & Deduplication
        t_f0 = time.perf_counter()
        fused_items = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=k_rrf,
        )
        latencies["fusion_latency_ms"] = round((time.perf_counter() - t_f0) * 1000.0, 3)

        if not fused_items:
            latencies["total_latency_ms"] = round((time.perf_counter() - start_wall) * 1000.0, 3)
            return SearchResponse(query=query, results=[], total_candidates=0, latency_ms=latencies)

        # Stage 4: Candidate Hydration
        # Hydrate up to max(k_dense, k_sparse) candidates for reranking
        max_to_hydrate = max(k_dense, k_sparse)
        candidates, hydration_ms = await self._hydrate_candidates(
            fused_items[:max_to_hydrate], pre_hydrated
        )
        latencies["hydration_latency_ms"] = hydration_ms

        # Stage 5: Cross-Encoder Reranking (sync GPU work, offloaded to threadpool)
        t_r0 = time.perf_counter()
        if self.reranker is not None and candidates:
            final_results = await asyncio.to_thread(
                self.reranker.rerank, query, candidates, k_final
            )
        else:
            final_results = candidates[:k_final]
        latencies["rerank_latency_ms"] = round((time.perf_counter() - t_r0) * 1000.0, 3)

        total_wall_ms = (time.perf_counter() - start_wall) * 1000.0
        latencies["total_latency_ms"] = round(total_wall_ms, 3)

        # Optional query logging to PostgreSQL
        if self.postgres_client is not None and await self.postgres_client.is_healthy():
            try:
                retrieved_ids = [c.chunk_id for c in candidates]
                reranked_ids = [c.chunk_id for c in final_results]
                await self.postgres_client.log_query(
                    query_text=query,
                    retrieved_chunk_ids=retrieved_ids,
                    reranked_chunk_ids=reranked_ids,
                    dense_latency_ms=dense_ms,
                    sparse_latency_ms=sparse_ms,
                    rerank_latency_ms=latencies["rerank_latency_ms"],
                    total_latency_ms=latencies["total_latency_ms"],
                )
            except Exception as e:
                logger.debug(f"Failed to log query telemetry: {e}")

        logger.info(
            f"Query '{query[:30]}...' -> {len(final_results)} passages "
            f"(dense: {len(dense_results)}, sparse: {len(sparse_results)}, fused: {len(fused_items)}) "
            f"in {latencies['total_latency_ms']} ms"
        )

        return SearchResponse(
            query=query,
            results=final_results,
            total_candidates=len(fused_items),
            latency_ms=latencies,
        )

    def search(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> SearchResponse:
        """Synchronous convenience wrapper around search_async."""
        return asyncio.run(
            self.search_async(
                query=query,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                rrf_k=rrf_k,
                final_top_k=final_top_k,
            )
        )

