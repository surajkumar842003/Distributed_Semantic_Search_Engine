import os
import tempfile
import pytest
from src.common.schemas import DocumentChunk
from src.indexing.bm25_builder import BM25IndexBuilder, Posting, ENGLISH_STOP_WORDS
from src.retrieval.bm25_searcher import BM25Searcher

def test_map_phase_term_extraction():
    builder = BM25IndexBuilder(num_workers=1)
    chunk = DocumentChunk.create(
        doc_id=1, chunk_seq=1, title="Test", url="url", section_path="",
        text="The quick brown fox jumps over the lazy dog.", token_count=10
    )
    term_postings = builder._map_phase([chunk])
    
    assert "quick" in term_postings
    assert "brown" in term_postings
    assert "fox" in term_postings
    assert "the" not in term_postings
    assert "over" not in term_postings
    
    assert len(term_postings["quick"]) == 1
    assert term_postings["quick"][0].term_frequency == 1
    assert term_postings["quick"][0].chunk_id == chunk.chunk_id

def test_bm25_scores_basic(tmp_path):
    builder = BM25IndexBuilder(num_workers=1, output_dir=str(tmp_path))
    c1 = DocumentChunk.create(1, 1, "T", "U", "S", "apple apple apple", 3)
    c2 = DocumentChunk.create(2, 1, "T", "U", "S", "apple banana", 2)
    c3 = DocumentChunk.create(3, 1, "T", "U", "S", "banana orange", 2)
    
    index = builder.build([c1, c2, c3])
    
    searcher = BM25Searcher(str(tmp_path))
    results = searcher.search("apple", top_k=5)
    
    assert len(results) == 2
    assert results[0][0] == c1.chunk_id
    assert results[1][0] == c2.chunk_id
    assert results[0][1] > results[1][1]

def test_bm25_search_returns_ranked(tmp_path):
    builder = BM25IndexBuilder(num_workers=1, output_dir=str(tmp_path))
    c1 = DocumentChunk.create(1, 1, "T", "U", "S", "cat dog", 2)
    c2 = DocumentChunk.create(2, 1, "T", "U", "S", "cat cat dog", 3)
    c3 = DocumentChunk.create(3, 1, "T", "U", "S", "dog", 1)
    
    builder.build([c1, c2, c3])
    searcher = BM25Searcher(str(tmp_path))
    results = searcher.search("cat", top_k=10)
    
    assert len(results) == 2
    assert results[0][0] == c2.chunk_id
    assert results[1][0] == c1.chunk_id

def test_bm25_idf_weighting(tmp_path):
    builder = BM25IndexBuilder(num_workers=1, output_dir=str(tmp_path))
    c1 = DocumentChunk.create(1, 1, "T", "U", "S", "rare common", 2)
    c2 = DocumentChunk.create(2, 1, "T", "U", "S", "common", 1)
    c3 = DocumentChunk.create(3, 1, "T", "U", "S", "common", 1)
    
    builder.build([c1, c2, c3])
    searcher = BM25Searcher(str(tmp_path))
    
    results_rare = searcher.search("rare", top_k=10)
    results_common = searcher.search("common", top_k=10)
    
    assert results_rare[0][1] > results_common[0][1]

def test_roundtrip_save_load(tmp_path):
    builder = BM25IndexBuilder(num_workers=1, output_dir=str(tmp_path))
    c1 = DocumentChunk.create(1, 1, "T", "U", "S", "test roundtrip", 2)
    builder.build([c1])
    
    searcher = BM25Searcher(str(tmp_path))
    results = searcher.search("test")
    
    assert len(results) == 1
    assert results[0][0] == c1.chunk_id
    assert results[0][1] > 0
