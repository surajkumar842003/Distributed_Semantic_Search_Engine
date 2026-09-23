"""Core data contracts and schemas across the distributed pipeline."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


def make_chunk_id(doc_id: int, chunk_seq: int) -> int:
    """
    Generates a deterministic 64-bit signed integer ID combining doc_id and sequence:
    High 48 bits: doc_id (supports up to 281 trillion documents)
    Low 16 bits: chunk_seq (supports up to 65,535 chunks per document)
    """
    return (int(doc_id) << 16) | (int(chunk_seq) & 0xFFFF)


def parse_chunk_id(chunk_id: int) -> tuple[int, int]:
    """Recovers (doc_id, chunk_seq) from a composite chunk_id."""
    doc_id = chunk_id >> 16
    chunk_seq = chunk_id & 0xFFFF
    return doc_id, chunk_seq


@dataclass
class RawArticle:
    """Raw Wikipedia document row from Parquet."""
    id: int
    title: str
    url: str
    text: str
    categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentChunk:
    """
    Chunk emitted by heading-aware chunker.
    Deterministic 64-bit chunk_id maps 1:1 to FAISS index vector ID and PostgreSQL primary key.
    """
    chunk_id: int
    doc_id: int
    title: str
    url: str
    section_path: str
    text: str
    token_count: int
    faiss_id: int

    @classmethod
    def create(
        cls,
        doc_id: int,
        chunk_seq: int,
        title: str,
        url: str,
        section_path: str,
        text: str,
        token_count: int,
    ) -> "DocumentChunk":
        cid = make_chunk_id(doc_id, chunk_seq)
        return cls(
            chunk_id=cid,
            doc_id=doc_id,
            title=title,
            url=url,
            section_path=section_path,
            text=text,
            token_count=token_count,
            faiss_id=cid,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCandidate:
    """Candidate chunk returned during multi-stage retrieval."""
    chunk_id: int
    doc_id: int
    title: str
    url: str
    section_path: str
    text: str
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchRequest:
    query: str
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    final_top_k: int = 5


@dataclass
class SearchResponse:
    query: str
    results: List[RetrievalCandidate]
    total_candidates: int
    latency_ms: Dict[str, float]


@dataclass
class RAGAnswerResponse:
    query: str
    answer: str
    citations: List[RetrievalCandidate]
    latency_ms: Dict[str, float]

