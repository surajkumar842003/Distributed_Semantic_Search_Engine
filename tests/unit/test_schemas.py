"""Unit tests for pipeline schemas and deterministic chunk ID generation."""
from src.common.schemas import (
    make_chunk_id,
    parse_chunk_id,
    DocumentChunk,
    RawArticle,
    RetrievalCandidate,
)


def test_make_chunk_id_and_parse():
    doc_id = 42109
    chunk_seq = 7

    chunk_id = make_chunk_id(doc_id, chunk_seq)
    assert isinstance(chunk_id, int)
    assert chunk_id > 0

    recovered_doc_id, recovered_seq = parse_chunk_id(chunk_id)
    assert recovered_doc_id == doc_id
    assert recovered_seq == chunk_seq


def test_make_chunk_id_bounds():
    # Test large 48-bit doc_id and 16-bit sequence limit
    doc_id = 10_000_000  # 10 million (well over 6.5M Wikipedia docs)
    chunk_seq = 65535    # Max 16-bit unsigned integer

    chunk_id = make_chunk_id(doc_id, chunk_seq)
    rec_doc, rec_seq = parse_chunk_id(chunk_id)
    assert rec_doc == doc_id
    assert rec_seq == chunk_seq


def test_document_chunk_creation():
    chunk = DocumentChunk.create(
        doc_id=101,
        chunk_seq=2,
        title="Albert Einstein",
        url="https://en.wikipedia.org/wiki/Albert_Einstein",
        section_path="Albert Einstein > Early life",
        text="Albert Einstein - Albert Einstein > Early life\nEinstein was born in Ulm.",
        token_count=18,
    )

    assert chunk.doc_id == 101
    assert chunk.chunk_id == make_chunk_id(101, 2)
    assert chunk.faiss_id == chunk.chunk_id
    assert "Early life" in chunk.section_path

    d = chunk.to_dict()
    assert d["chunk_id"] == chunk.chunk_id
    assert d["title"] == "Albert Einstein"

