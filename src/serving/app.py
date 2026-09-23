"""FastAPI Serving Application for Semantic Search and Grounded RAG.

Endpoints:
- POST /search: High-speed hybrid semantic retrieval.
- POST /rag/answer: Grounded RAG question answering with inline citations.
- GET /ingest/status: Relational and vector index telemetry.
- GET /health: Component readiness and system health check.
- GET /metrics: Operational latency and query throughput metrics.
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import AppConfig
from src.common.logging import get_logger
from src.embedding.embedder import GPUEmbedder
from src.indexing.faiss_indexer import FAISSIndexer
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.storage.postgres_client import PostgresClient
from src.serving.schemas import (
    SearchQueryRequest,
    SearchQueryResponse,
    CandidateDTO,
    RAGRequest,
    RAGResponse,
    HealthResponse,
    MetricsResponse,
    IngestStatusResponse,
)
from src.serving.metrics import ServiceMetrics
from src.serving.generator import RAGGenerator

logger = get_logger("serving.app")

# Shared state container
app_state: Dict[str, Any] = {
    "config": None,
    "embedder": None,
    "faiss_indexer": None,
    "postgres_client": None,
    "reranker": None,
    "pipeline": None,
    "generator": None,
    "metrics": None,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for model and index initialization."""
    logger.info("Initializing FastAPI serving service components...")
    cfg = AppConfig.load_from_dir("configs")
    app_state["config"] = cfg
    app_state["metrics"] = ServiceMetrics()
    app_state["start_time"] = time.time()

    # 1. Initialize GPU Embedder
    try:
        app_state["embedder"] = GPUEmbedder(
            model_name=cfg.embedding.model_name,
            cache_dir=cfg.paths.hf_home,
            device=cfg.embedding.device,
            use_fp16=(cfg.embedding.precision == "fp16"),
        )
        logger.info("GPU Embedder initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize GPUEmbedder: {e}")

    # 2. Initialize or Load FAISS Index
    index_path = Path(cfg.faiss.index_path)
    if not index_path.is_file():
        # Check benchmark index locations as fallback
        alt_paths = [
            Path("data/benchmark_faiss/wiki_ivf_flat.index"),
            Path("data/benchmark_faiss/wiki_exact_flat.index"),
        ]
        for ap in alt_paths:
            if ap.is_file():
                index_path = ap
                break

    if index_path.is_file():
        try:
            app_state["faiss_indexer"] = FAISSIndexer.load(index_path, use_mmap=cfg.faiss.use_mmap)
            logger.info(f"Loaded FAISS index from {index_path} ({app_state['faiss_indexer'].ntotal:,} vectors).")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index from {index_path}: {e}")
    else:
        logger.info(f"No existing FAISS index found at {index_path}; initializing in-memory FlatIP.")
        app_state["faiss_indexer"] = FAISSIndexer(dim=cfg.embedding.embedding_dim, index_type="FlatIP")

    # 3. Initialize PostgreSQL Client
    try:
        pg = PostgresClient(cfg.postgres)
        await pg.connect()
        app_state["postgres_client"] = pg
        logger.info("PostgreSQL client connected.")
    except Exception as e:
        logger.warning(f"PostgreSQL connection offline: {e}; will operate in decoupled mode.")

    # 3b. Initialize BM25 Searcher (fallback & standalone sparse search)
    bm25_searcher = None
    bm25_dir = Path("data/benchmark_bm25")
    if bm25_dir.is_dir():
        try:
            from src.retrieval.bm25_searcher import BM25Searcher
            bm25_searcher = BM25Searcher(str(bm25_dir))
            logger.info(f"Loaded BM25 index from {bm25_dir} ({len(bm25_searcher.chunk_ids):,} chunks).")
        except Exception as e:
            logger.warning(f"Could not load BM25 index: {e}")

    # 3c. Load chunk store fallback for text hydration when PostgreSQL is offline
    chunk_store: Dict[int, Dict[str, Any]] = {}
    chunks_dir = Path("data/chunks_pilot")
    if chunks_dir.is_dir():
        try:
            import pyarrow.parquet as pq
            shard_files = sorted(chunks_dir.glob("chunks_*.parquet"))
            for sf in shard_files[:2]:
                table = pq.read_table(str(sf), columns=["chunk_id", "doc_id", "title", "url", "section_path", "text", "token_count"])
                for r in table.to_pylist():
                    chunk_store[r["chunk_id"]] = r
            logger.info(f"Loaded in-memory chunk store with {len(chunk_store):,} chunks for hydration.")
        except Exception as e:
            logger.warning(f"Could not load chunk store: {e}")

    # 4. Initialize Cross-Encoder Reranker
    try:
        app_state["reranker"] = CrossEncoderReranker(
            model_name=cfg.retrieval.rerank_model_name,
            cache_dir=cfg.paths.hf_home,
            device=cfg.retrieval.rerank_device,
            use_fp16=True,
        )
        logger.info("Cross-Encoder Reranker initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize CrossEncoderReranker: {e}")

    # 5. Build Hybrid Retrieval Pipeline
    app_state["pipeline"] = HybridRetrievalPipeline(
        embedder=app_state["embedder"],
        faiss_indexer=app_state["faiss_indexer"],
        postgres_client=app_state["postgres_client"],
        bm25_searcher=bm25_searcher,
        chunk_store=chunk_store,
        reranker=app_state["reranker"],
        config=cfg.retrieval,
    )

    # 6. Initialize RAG Generator
    app_state["generator"] = RAGGenerator(
        provider="api",
        model="gemini-1.5-flash",
        temperature=0.2,
        max_tokens=1024,
        fallback_to_offline=True,
    )

    logger.info("FastAPI service startup complete and ready to serve traffic.")
    yield

    # Clean shutdown
    logger.info("Shutting down FastAPI serving service...")
    if app_state["postgres_client"] is not None:
        await app_state["postgres_client"].disconnect()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Distributed Semantic Search & RAG Service",
    description="Multi-stage hybrid retrieval and grounded RAG question answering over Wikipedia.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the search engine UI
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the search engine UI."""
    index_file = _static_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file), media_type="text/html")
    return JSONResponse({"message": "WikiRAG API", "docs": "/docs"})


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns component readiness and operational status."""
    embedder_ok = app_state["embedder"] is not None
    faiss_ok = app_state["faiss_indexer"] is not None
    reranker_ok = app_state["reranker"] is not None
    pg_ok = app_state["postgres_client"] is not None and await app_state["postgres_client"].is_healthy()
    llm_ok = app_state["generator"] is not None

    components = {
        "embedder": "healthy" if embedder_ok else "unhealthy",
        "faiss_index": "healthy" if faiss_ok else "unhealthy",
        "reranker": "healthy" if reranker_ok else "unhealthy",
        "postgres": "healthy" if pg_ok else "offline",
        "generator": "healthy" if llm_ok else "unhealthy",
    }

    overall = "healthy" if (embedder_ok and faiss_ok and reranker_ok) else "degraded"
    uptime = time.time() - app_state["start_time"]

    import torch
    gpu_available = torch.cuda.is_available()

    return HealthResponse(
        status=overall,
        uptime_seconds=round(uptime, 1),
        components=components,
        gpu_available=gpu_available,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Returns latency percentiles, stage breakdowns, and query counts."""
    metrics: ServiceMetrics = app_state["metrics"]
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics system not initialized")
    summary = metrics.get_summary()
    return MetricsResponse(
        total_queries=summary["total_queries"],
        total_search_requests=summary["total_search_requests"],
        total_rag_requests=summary["total_rag_requests"],
        error_count=summary["error_count"],
        latency_p50_ms=summary["latency_p50_ms"],
        latency_p95_ms=summary["latency_p95_ms"],
        latency_p99_ms=summary["latency_p99_ms"],
        average_stage_latency_ms=summary["average_stage_latency_ms"],
    )


@app.get("/ingest/status", response_model=IngestStatusResponse)
async def get_ingest_status():
    """Returns ingestion counts from PostgreSQL and vector count from FAISS."""
    pg_client: Optional[PostgresClient] = app_state["postgres_client"]
    faiss_idx: Optional[FAISSIndexer] = app_state["faiss_indexer"]

    docs_count = 0
    chunks_count = 0
    if pg_client is not None and await pg_client.is_healthy():
        try:
            counts = await pg_client.get_table_counts()
            docs_count = counts.get("documents", 0)
            chunks_count = counts.get("chunks", 0)
        except Exception as e:
            logger.warning(f"Could not read table counts: {e}")

    vectors_count = faiss_idx.ntotal if faiss_idx else 0
    index_type = faiss_idx.index_type if faiss_idx else "None"
    dim = faiss_idx.dim if faiss_idx else 384

    return IngestStatusResponse(
        status="ready" if (vectors_count > 0 or chunks_count > 0) else "empty",
        total_documents=docs_count,
        total_chunks=chunks_count,
        faiss_vectors_indexed=vectors_count,
        index_type=index_type,
        dimension=dim,
    )


@app.post("/search", response_model=SearchQueryResponse)
async def search_endpoint(req: SearchQueryRequest):
    """Executes multi-stage hybrid search (dense + sparse -> RRF -> reranking)."""
    pipeline: HybridRetrievalPipeline = app_state["pipeline"]
    metrics: ServiceMetrics = app_state["metrics"]

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Retrieval pipeline not initialized")

    query_id = f"q_{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()

    try:
        resp = await pipeline.search_async(
            query=req.query,
            dense_top_k=req.dense_top_k,
            sparse_top_k=req.sparse_top_k,
            rrf_k=req.rrf_k,
            final_top_k=req.final_top_k,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        candidates_dto = [
            CandidateDTO(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                url=c.url,
                section_path=c.section_path,
                text=c.text,
                token_count=len(c.text.split()),
                dense_score=c.dense_score,
                dense_rank=c.dense_rank,
                sparse_score=c.sparse_score,
                sparse_rank=c.sparse_rank,
                rrf_score=c.rrf_score,
                rerank_score=c.rerank_score,
            )
            for c in resp.results
        ]

        if metrics:
            metrics.record_request(
                endpoint_type="search",
                latency_ms=total_ms,
                stage_latencies=resp.latency_ms,
            )

        return SearchQueryResponse(
            query_id=query_id,
            query=req.query,
            results=candidates_dto,
            total_candidates=resp.total_candidates,
            latency_ms=resp.latency_ms,
        )

    except Exception as e:
        logger.error(f"Search query {query_id} failed: {e}", exc_info=True)
        if metrics:
            metrics.record_request("search", 0.0, {}, is_error=True)
        raise HTTPException(status_code=500, detail="Search execution error")


@app.post("/rag/answer", response_model=RAGResponse)
async def rag_answer_endpoint(req: RAGRequest):
    """Answers a user question using grounded RAG with inline citations."""
    pipeline: HybridRetrievalPipeline = app_state["pipeline"]
    generator: RAGGenerator = app_state["generator"]
    metrics: ServiceMetrics = app_state["metrics"]

    if pipeline is None or generator is None:
        raise HTTPException(status_code=503, detail="RAG system components not initialized")

    query_id = f"rag_{uuid.uuid4().hex[:12]}"
    t_start = time.perf_counter()

    try:
        # 1. Retrieve top candidates using hybrid pipeline
        search_resp = await pipeline.search_async(
            query=req.query,
            dense_top_k=req.dense_top_k,
            sparse_top_k=req.sparse_top_k,
            rrf_k=req.rrf_k,
            final_top_k=req.final_top_k,
        )

        # 2. Generate answer with prompt injection defenses
        t_g0 = time.perf_counter()
        answer, citations, insufficient = await generator.generate_answer(
            query=req.query,
            candidates=search_resp.results,
        )
        gen_ms = (time.perf_counter() - t_g0) * 1000.0

        total_wall_ms = (time.perf_counter() - t_start) * 1000.0
        latencies = dict(search_resp.latency_ms)
        latencies["generation_latency_ms"] = round(gen_ms, 3)
        latencies["total_latency_ms"] = round(total_wall_ms, 3)

        if metrics:
            metrics.record_request(
                endpoint_type="rag",
                latency_ms=total_wall_ms,
                stage_latencies=latencies,
            )

        return RAGResponse(
            query_id=query_id,
            query=req.query,
            answer=answer,
            citations=citations,
            insufficient_context=insufficient,
            latency_ms=latencies,
            provider=generator.provider,
            model=generator.model,
        )

    except Exception as e:
        logger.error(f"RAG query {query_id} failed: {e}", exc_info=True)
        if metrics:
            metrics.record_request("rag", 0.0, {}, is_error=True)
        raise HTTPException(status_code=500, detail="RAG generation error")

