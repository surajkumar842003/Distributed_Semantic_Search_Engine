"""Unit tests for heading-aware chunker."""
from src.common.schemas import RawArticle
from src.ingestion.chunker import HeadingAwareChunker


def test_chunk_article_heading_context():
    chunker = HeadingAwareChunker(target_tokens=50, overlap_tokens=10, min_tokens=5)
    
    raw_text = """== History ==
In 1905, Albert Einstein published four groundbreaking papers. These papers explained the photoelectric effect, Brownian motion, special relativity, and mass-energy equivalence.

== Legacy ==
Einstein received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect.
"""
    article = RawArticle(
        id=999,
        title="Albert Einstein",
        url="https://en.wikipedia.org/wiki/Albert_Einstein",
        text=raw_text,
    )

    chunks = chunker.chunk_article(article)
    assert len(chunks) >= 2

    # Check first chunk has correct doc_id, deterministic chunk_id, and heading prefix
    c0 = chunks[0]
    assert c0.doc_id == 999
    assert c0.title == "Albert Einstein"
    assert "Albert Einstein - Albert Einstein > History" in c0.text
    assert "photoelectric effect" in c0.text

    # Check second section chunk
    c_legacy = [c for c in chunks if "Legacy" in c.section_path][0]
    assert "Albert Einstein - Albert Einstein > Legacy" in c_legacy.text
    assert "Nobel Prize" in c_legacy.text


def test_chunk_sequence_and_ids():
    chunker = HeadingAwareChunker(target_tokens=20, overlap_tokens=5, min_tokens=5)
    
    long_text = "Word " * 200
    article = RawArticle(
        id=77,
        title="Repetitive Article",
        url="https://en.wikipedia.org/wiki/Repetitive",
        text=long_text,
    )

    chunks = chunker.chunk_article(article)
    assert len(chunks) > 5

    # Check that chunk_ids are strictly unique and sequentially ordered
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

