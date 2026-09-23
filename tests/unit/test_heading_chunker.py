"""Unit tests for HeadingAwareChunker sentence preservation, tokenizer integration, and heading handling."""
import pytest
from src.common.schemas import RawArticle, parse_chunk_id
from src.ingestion.chunker import HeadingAwareChunker
from src.common.tokenizer import get_tokenizer


@pytest.fixture(scope="module")
def bge_tokenizer():
    return get_tokenizer("BAAI/bge-small-en-v1.5")


def test_chunking_with_headings_and_section_paths(bge_tokenizer):
    chunker = HeadingAwareChunker(
        target_tokens=100,
        overlap_tokens=20,
        tokenizer=bge_tokenizer,
    )

    text = """
The introductory paragraph provides a broad summary of the historical subject.

== Early Life ==
Albert was born in Ulm, Germany. He showed an early interest in mathematics and science.
His family moved to Italy where he continued his independent self-study.

== Academic Career ==
In 1905, often referred to as his Annus Mirabilis, he published four groundbreaking papers.
These papers revolutionized theoretical physics and introduced special relativity.
"""
    article = RawArticle(
        id=1234,
        title="Albert Einstein",
        url="https://en.wikipedia.org/wiki/Albert_Einstein",
        text=text,
    )

    chunks = chunker.chunk_article(article)
    assert len(chunks) >= 2

    # Check section paths
    section_paths = [c.section_path for c in chunks]
    assert any("Early Life" in p for p in section_paths)
    assert any("Academic Career" in p for p in section_paths)

    # Check headers and deterministic IDs
    for seq, chunk in enumerate(chunks):
        assert chunk.doc_id == 1234
        doc_id_rec, chunk_seq_rec = parse_chunk_id(chunk.chunk_id)
        assert doc_id_rec == 1234
        assert chunk_seq_rec == seq
        assert chunk.title == "Albert Einstein"
        assert chunk.url == "https://en.wikipedia.org/wiki/Albert_Einstein"
        assert chunk.token_count > 0


def test_article_without_headings(bge_tokenizer):
    """Verifies articles without headings fall back gracefully without being dropped."""
    chunker = HeadingAwareChunker(target_tokens=150, tokenizer=bge_tokenizer)

    text = (
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics. "
        "Classical computers encode information in binary bits that can either be 0 or 1. "
        "Quantum computers use qubits which can exist in quantum superposition."
    )
    article = RawArticle(
        id=5678,
        title="Quantum Computing",
        url="https://en.wikipedia.org/wiki/Quantum_computing",
        text=text,
    )

    chunks = chunker.chunk_article(article)
    assert len(chunks) == 1
    assert "Overview" in chunks[0].section_path
    assert chunks[0].title == "Quantum Computing"
    assert chunks[0].doc_id == 5678


def test_short_article_handling(bge_tokenizer):
    """Short stub articles should produce exactly one chunk."""
    chunker = HeadingAwareChunker(target_tokens=256, tokenizer=bge_tokenizer)

    article = RawArticle(
        id=999,
        title="Albedo",
        url="https://en.wikipedia.org/wiki/Albedo",
        text="Albedo is the measure of the diffuse reflection of solar radiation out of the total solar radiation.",
    )

    chunks = chunker.chunk_article(article)
    assert len(chunks) == 1
    assert chunks[0].doc_id == 999
    assert chunks[0].token_count < 256


def test_sentence_boundary_preservation(bge_tokenizer):
    """Sentences must not be split across chunks unless an individual sentence exceeds target."""
    chunker = HeadingAwareChunker(
        target_tokens=40,
        overlap_tokens=10,
        tokenizer=bge_tokenizer,
    )

    sentences = [
        "Dr. Smith visited the U.S. capital yesterday to attend a scientific conference.",
        "The conference focused on artificial intelligence and distributed computing.",
        "Prof. Davis presented a paper on vector indexing algorithms.",
        "Several researchers from approx. ten universities participated in the workshop.",
    ]
    text = " ".join(sentences)

    article = RawArticle(
        id=42,
        title="AI Workshop",
        url="https://en.wikipedia.org/wiki/AI_Workshop",
        text=text,
    )

    chunks = chunker.chunk_article(article)
    assert len(chunks) >= 2

    # Verify no sentence is sliced mid-word
    for chunk in chunks:
        # Check that sentences ending with '.' are complete sentences
        body = chunk.text.split("\n", 1)[-1]
        assert body.endswith(".") or body.endswith("!") or body.endswith("?")


def test_exact_tokenizer_counting(bge_tokenizer):
    """Verifies that token_count matches exact bge-small tokenizer output."""
    chunker = HeadingAwareChunker(target_tokens=256, tokenizer=bge_tokenizer)

    article = RawArticle(
        id=100,
        title="Token Test",
        url="https://en.wikipedia.org/wiki/Token_Test",
        text="This is a test of the exact token counting mechanism using BGE small tokenizer.",
    )

    chunks = chunker.chunk_article(article)
    assert len(chunks) == 1

    chunk = chunks[0]
    expected_tokens = len(bge_tokenizer(chunk.text, add_special_tokens=False)["input_ids"])
    assert chunk.token_count == expected_tokens

