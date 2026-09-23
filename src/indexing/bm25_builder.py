"""
Distributed MapReduce BM25 Inverted Index Builder.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import multiprocessing
import os
import re
import math
import pickle
import numpy as np
from scipy import sparse
from collections import defaultdict, Counter

from src.common.schemas import DocumentChunk
from src.common.logging import get_logger

logger = get_logger(__name__)

ENGLISH_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "do", "does", "did", "will", "would", "could", 
    "should", "may", "might", "shall", "can", "to", "of", "in", "for", "on", 
    "with", "at", "by", "from", "as", "into", "through", "during", "before", 
    "after", "above", "below", "between", "out", "off", "over", "under", "again", 
    "further", "then", "once", "here", "there", "where", "when", "why", "how", 
    "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", 
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "and", 
    "but", "or", "if", "that", "this", "it", "its", "i", "me", "my", "we", "our", 
    "you", "your", "he", "him", "his", "she", "her", "they", "them", "their", 
    "what", "which", "who", "whom"
}

@dataclass
class Posting:
    """Represents a term occurrence in a document chunk."""
    chunk_id: int
    term_frequency: int

@dataclass
class BM25Index:
    """Container for the BM25 Index."""
    matrix: sparse.csr_matrix
    vocabulary: Dict[str, int]
    chunk_ids: np.ndarray
    metadata: Dict[str, Any]

def _tokenize(text: str) -> List[str]:
    """Tokenize text into valid lowercase alphanumeric terms."""
    tokens = re.split(r'[^a-z0-9]+', text.lower())
    return [t for t in tokens if t and t not in ENGLISH_STOP_WORDS]

def _map_worker(chunk: DocumentChunk) -> List[Tuple[str, Posting]]:
    """Worker function for Map Phase. Needs to be at top level for pickle."""
    tokens = _tokenize(chunk.text)
    tf_counts = Counter(tokens)
    
    postings = []
    for term, freq in tf_counts.items():
        postings.append((term, Posting(chunk_id=chunk.chunk_id, term_frequency=freq)))
        
    return postings

class BM25IndexBuilder:
    """Builds a BM25 inverted index using the MapReduce paradigm."""
    
    def __init__(self, num_workers: int = 4, k1: float = 1.2, b: float = 0.75, output_dir: str = "bm25_index"):
        """
        Initialize builder.
        
        Args:
            num_workers: Number of processes for parallel map
            k1: BM25 term frequency scaling parameter
            b: BM25 document length scaling parameter
            output_dir: Directory to save the index outputs
        """
        self.num_workers = num_workers
        self.k1 = k1
        self.b = b
        self.output_dir = output_dir

    def build(self, chunks: List[DocumentChunk]) -> BM25Index:
        """
        Run the complete MapReduce pipeline to build the index.
        """
        logger.info(f"Starting BM25 MapReduce build on {len(chunks)} chunks with {self.num_workers} workers.")
        
        # 1. Map Phase & Shuffle
        term_postings = self._map_phase(chunks)
        
        # Calculate lengths
        num_chunks = len(chunks)
        chunk_lengths = defaultdict(int)
        for postings in term_postings.values():
            for p in postings:
                chunk_lengths[p.chunk_id] += p.term_frequency
                
        # 2. Reduce Phase
        index = self._reduce_phase(term_postings, num_chunks, dict(chunk_lengths))
        
        # 3. Save to disk
        self._save(index)
        
        return index

    def _map_phase(self, chunks: List[DocumentChunk]) -> Dict[str, List[Posting]]:
        """
        Execute Map phase in parallel and shuffle results.
        """
        logger.info("Executing Map phase...")
        term_postings = defaultdict(list)
        
        if self.num_workers > 1:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                results = pool.map(_map_worker, chunks)
        else:
            results = [_map_worker(c) for c in chunks]
            
        # Shuffle (simulated via defaultdict)
        logger.info("Executing Shuffle phase...")
        for chunk_postings in results:
            for term, posting in chunk_postings:
                term_postings[term].append(posting)
                
        return dict(term_postings)

    def _reduce_phase(self, term_postings: Dict[str, List[Posting]], num_chunks: int, chunk_lengths: Dict[int, int]) -> BM25Index:
        """
        Execute Reduce phase to compute BM25 weights and build CSR matrix.
        """
        logger.info("Executing Reduce phase...")
        
        # Compute avgdl
        total_length = sum(chunk_lengths.values())
        avgdl = total_length / max(1, num_chunks)
        
        # Assign indices mapping
        # Sort vocabulary alphabetically
        vocabulary = {term: idx for idx, term in enumerate(sorted(term_postings.keys()))}
        # Create chunk_id -> matrix column index mapping
        unique_chunk_ids = sorted(list(chunk_lengths.keys()))
        chunk_id_to_idx = {cid: idx for idx, cid in enumerate(unique_chunk_ids)}
        chunk_ids_arr = np.array(unique_chunk_ids, dtype=np.int64)
        
        # CSR matrix components
        data = []
        rows = []
        cols = []
        
        # Reduce each term
        for term, postings in term_postings.items():
            row_idx = vocabulary[term]
            
            # Document frequency
            df = len(postings)
            
            # IDF using standard BM25 formula
            idf = math.log((num_chunks - df + 0.5) / (df + 0.5) + 1.0)
            
            # Calculate BM25 score for each posting
            for p in postings:
                tf = p.term_frequency
                dl = chunk_lengths[p.chunk_id]
                
                # Numerator & Denominator
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (dl / max(1, avgdl)))
                
                score = idf * (numerator / denominator)
                
                if score > 0:
                    data.append(score)
                    rows.append(row_idx)
                    cols.append(chunk_id_to_idx[p.chunk_id])
                    
        # Build CSR
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(vocabulary), len(unique_chunk_ids)), dtype=np.float32)
        
        metadata = {
            "N": num_chunks,
            "avgdl": avgdl,
            "k1": self.k1,
            "b": self.b
        }
        
        logger.info(f"Built BM25 Index. Vocab size: {len(vocabulary)}, Chunks: {num_chunks}")
        
        return BM25Index(
            matrix=matrix,
            vocabulary=vocabulary,
            chunk_ids=chunk_ids_arr,
            metadata=metadata
        )

    def _save(self, index: BM25Index) -> None:
        """Save the index components to output_dir."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save CSR matrix
        sparse.save_npz(os.path.join(self.output_dir, "matrix.npz"), index.matrix)
        
        # Save vocabulary
        with open(os.path.join(self.output_dir, "vocabulary.pkl"), "wb") as f:
            pickle.dump(index.vocabulary, f)
            
        # Save chunk_ids
        np.save(os.path.join(self.output_dir, "chunk_ids.npy"), index.chunk_ids)
        
        # Save metadata
        with open(os.path.join(self.output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(index.metadata, f)
            
        logger.info(f"Saved BM25 Index to {self.output_dir}")
