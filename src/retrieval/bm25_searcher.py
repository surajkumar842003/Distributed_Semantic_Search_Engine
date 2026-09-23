"""BM25 Searcher Module."""
from typing import List, Tuple
import os
import pickle
import numpy as np
import re
from scipy import sparse

from src.indexing.bm25_builder import ENGLISH_STOP_WORDS

class BM25Searcher:
    """Searches a pre-built BM25 inverted index."""
    
    def __init__(self, index_dir: str):
        """
        Load the CSR matrix, vocabulary, and chunk_id mapping from disk.
        """
        self.index_dir = index_dir
        self.matrix = sparse.load_npz(os.path.join(index_dir, "matrix.npz"))
        
        with open(os.path.join(index_dir, "vocabulary.pkl"), "rb") as f:
            self.vocabulary = pickle.load(f)
            
        self.chunk_ids = np.load(os.path.join(index_dir, "chunk_ids.npy"))
        
    def _tokenize(self, text: str) -> List[str]:
        tokens = re.split(r'[^a-z0-9]+', text.lower())
        return [t for t in tokens if t and t not in ENGLISH_STOP_WORDS]

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search the BM25 index for the given query.
        Returns top_k (chunk_id, score) pairs sorted descending.
        """
        tokens = self._tokenize(query)
        if not tokens:
            return []
            
        # Initialize scores array for all chunks
        scores = np.zeros(self.matrix.shape[1], dtype=np.float32)
        
        for term in set(tokens):
            if term in self.vocabulary:
                row_idx = self.vocabulary[term]
                # Get the row from the CSR matrix (1 x num_chunks)
                term_scores = self.matrix.getrow(row_idx).toarray()[0]
                scores += term_scores
                
        # Find non-zero scores
        nonzero_indices = np.nonzero(scores)[0]
        if len(nonzero_indices) == 0:
            return []
            
        # Get chunk_ids and scores for non-zero entries
        valid_scores = scores[nonzero_indices]
        valid_chunk_ids = self.chunk_ids[nonzero_indices]
        
        # Sort descending
        sorted_indices = np.argsort(-valid_scores)
        
        results = []
        for i in sorted_indices[:top_k]:
            results.append((int(valid_chunk_ids[i]), float(valid_scores[i])))
            
        return results
