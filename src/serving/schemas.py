"""Pydantic Data Transfer Objects for the FastAPI Serving Layer."""

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class SearchQueryRequest(BaseModel):
    """Request schema for hybrid semantic search."""
    query: Optional[str] = Field(None, min_length=1, max_length=1000, description="Search query string")
    question: Optional[str] = Field(None, min_length=1, max_length=1000, description="Alias for query")
    dense_top_k: int = Field(50, ge=1, le=500, description="Number of dense candidates from FAISS")
    sparse_top_k: int = Field(50, ge=1, le=500, description="Number of sparse candidates from PostgreSQL/BM25")
    rrf_k: int = Field(60, ge=1, le=500, description="Reciprocal Rank Fusion smoothing parameter")
    final_top_k: int = Field(5, ge=1, le=100, description="Number of reranked passages to return")

    @model_validator(mode="before")
    @classmethod
    def resolve_query_or_question(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("query") and data.get("question"):
                data["query"] = data["question"]
            elif not data.get("query") and not data.get("question"):
                raise ValueError("Field 'query' or 'question' is required.")
            if "top_k" in data and "final_top_k" not in data:
                data["final_top_k"] = data["top_k"]
        return data


class CandidateDTO(BaseModel):
    """Passage candidate with full provenance and ranking scores."""
    chunk_id: int
    doc_id: int
    title: str
    url: str
    section_path: str
    text: str
    token_count: int
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    rrf_score: float
    rerank_score: Optional[float] = None


class SearchQueryResponse(BaseModel):
    """Response schema for hybrid semantic search."""
    query_id: str
    query: str
    results: List[CandidateDTO]
    total_candidates: int
    latency_ms: Dict[str, float]
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class RAGRequest(BaseModel):
    """Request schema for RAG question answering."""
    query: Optional[str] = Field(None, min_length=1, max_length=1000, description="User question to answer")
    question: Optional[str] = Field(None, min_length=1, max_length=1000, description="Alias for query")
    dense_top_k: int = Field(50, ge=1, le=500, description="Number of dense candidates")
    sparse_top_k: int = Field(50, ge=1, le=500, description="Number of sparse candidates")
    rrf_k: int = Field(60, ge=1, le=500, description="RRF constant")
    final_top_k: int = Field(5, ge=1, le=50, description="Top-k context passages provided to LLM")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM sampling temperature")
    max_tokens: int = Field(1024, ge=50, le=4096, description="Maximum tokens for generated response")

    @model_validator(mode="before")
    @classmethod
    def resolve_query_or_question(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("query") and data.get("question"):
                data["query"] = data["question"]
            elif not data.get("query") and not data.get("question"):
                raise ValueError("Field 'query' or 'question' is required.")
            if "top_k" in data and "final_top_k" not in data:
                data["final_top_k"] = data["top_k"]
        return data


class CitationDTO(BaseModel):
    """Inline citation reference mapping to document provenance."""
    citation_index: int = Field(..., description="1-based numerical index in answer text (e.g. [1])")
    chunk_id: int
    doc_id: int
    title: str
    url: str
    section_path: str
    snippet: str
    relevance_score: Optional[float] = None


class RAGResponse(BaseModel):
    """Response schema for RAG question answering with citations."""
    query_id: str
    query: str
    answer: str
    citations: List[CitationDTO]
    insufficient_context: bool
    latency_ms: Dict[str, float]
    provider: str
    model: str
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class HealthResponse(BaseModel):
    """System and component health status."""
    status: str
    uptime_seconds: float
    components: Dict[str, str]
    gpu_available: bool
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class MetricsResponse(BaseModel):
    """Operational latency and query throughput metrics."""
    total_queries: int
    total_search_requests: int
    total_rag_requests: int
    error_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    average_stage_latency_ms: Dict[str, float]


class IngestStatusResponse(BaseModel):
    """Relational and vector index ingestion telemetry."""
    status: str
    total_documents: int
    total_chunks: int
    faiss_vectors_indexed: int
    index_type: str
    dimension: int

