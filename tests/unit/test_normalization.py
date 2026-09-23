"""Unit tests for dataset normalization and article validation."""
from src.ingestion.normalizer import ArticleNormalizer, NormalizedArticle


def test_normalized_article_fields():
    article = NormalizedArticle(
        doc_id=101,
        title="Albert Einstein",
        url="https://en.wikipedia.org/wiki/Albert_Einstein",
        text="Albert Einstein was a theoretical physicist.",
        source_shard=0,
    )
    assert article.doc_id == 101
    assert article.title == "Albert Einstein"
    assert article.source_shard == 0

    d = article.to_dict()
    assert d["doc_id"] == 101
    assert "source_shard" in d


def test_extract_shard_id():
    normalizer = ArticleNormalizer()

    assert normalizer.extract_shard_id("train-00000-of-00041.parquet") == 0
    assert normalizer.extract_shard_id("train-00003-of-00041.parquet") == 3
    assert normalizer.extract_shard_id("/DATA/raw/shard_0012.parquet") == 12
    assert normalizer.extract_shard_id("chunks_0005.parquet") == 5
    assert normalizer.extract_shard_id("wikipedia_sample.parquet") == 0


def test_parse_doc_id():
    normalizer = ArticleNormalizer()

    assert normalizer.parse_doc_id(42) == 42
    assert normalizer.parse_doc_id("42") == 42
    assert normalizer.parse_doc_id("  10092  ") == 10092
    assert normalizer.parse_doc_id("wiki_998877") == 998877
    assert normalizer.parse_doc_id(None, fallback_index=7) == 7
    assert normalizer.parse_doc_id("invalid_no_digits", fallback_index=99) == 99


def test_normalize_record():
    normalizer = ArticleNormalizer(min_chars=50)

    raw = {
        "id": "12345",
        "title": "  Quantum Computing  ",
        "url": "https://en.wikipedia.org/wiki/Quantum_computing",
        "text": "  Quantum computing is a rapidly-emerging technology that harnesses quantum mechanics.  ",
    }

    norm, flags = normalizer.normalize_record(raw, source_shard=2)

    assert norm.doc_id == 12345
    assert norm.title == "Quantum Computing"
    assert norm.source_shard == 2
    assert norm.text.startswith("Quantum computing")
    assert not flags["has_empty_text"]
    assert not flags["is_extremely_short"]
    assert flags["has_valid_url"]


def test_article_validation():
    normalizer = ArticleNormalizer(min_chars=50)

    valid_article = NormalizedArticle(
        doc_id=1,
        title="Valid Title",
        url="https://en.wikipedia.org/wiki/Valid",
        text="A" * 60,
        source_shard=0,
    )
    assert normalizer.is_valid(valid_article)

    # Empty text
    empty_article = NormalizedArticle(
        doc_id=2,
        title="Valid Title",
        url="https://en.wikipedia.org/wiki/Valid",
        text="",
        source_shard=0,
    )
    assert not normalizer.is_valid(empty_article)

    # Too short (<50 chars)
    short_article = NormalizedArticle(
        doc_id=3,
        title="Valid Title",
        url="https://en.wikipedia.org/wiki/Valid",
        text="Too short text.",
        source_shard=0,
    )
    assert not normalizer.is_valid(short_article)

    # Invalid URL
    bad_url_article = NormalizedArticle(
        doc_id=4,
        title="Valid Title",
        url="ftp://invalid-scheme",
        text="A" * 60,
        source_shard=0,
    )
    assert not normalizer.is_valid(bad_url_article)

