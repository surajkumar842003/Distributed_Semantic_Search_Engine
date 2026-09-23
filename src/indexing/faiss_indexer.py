"""FAISS Vector Indexer for Wikipedia Chunk Embeddings.

Features:
- Supports exact IndexFlatIP/IndexFlatL2, IndexIVFFlat, and IndexIVFPQ.
- Explains and enforces Cosine Similarity via Inner Product (METRIC_INNER_PRODUCT).
- Dynamic cluster/subquantizer configuration with representative sample training.
- Incremental shard-by-shard ingestion with 64-bit chunk_id preservation.
- Duplicate chunk_id insertion prevention via in-memory and index-extracted ID sets.
- Atomic index persistence and safe reload with zero-copy memory mapping (MMAP).
- Integration with PipelineCheckpoint for crash-resilient shard processing.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import numpy as np
import pyarrow.parquet as pq

try:
    import faiss
except ImportError as e:
    raise ImportError("FAISS is required. Install faiss-cpu or faiss-gpu.") from e

from src.common.logging import get_logger
from src.common.checkpoint import PipelineCheckpoint

logger = get_logger("indexing.faiss_indexer")

DEFAULT_DIM = 384
DEFAULT_NLIST = 256
DEFAULT_M_SUBQUANTIZERS = 48
DEFAULT_BITS_PER_CODE = 8


def explain_metric_choice(metric: str = "INNER_PRODUCT") -> str:
    """Returns mathematical explanation of similarity metric choice for semantic search."""
    if metric.upper() in ("INNER_PRODUCT", "IP"):
        return (
            "Choice of Metric: INNER_PRODUCT (Cosine Similarity)\n"
            "For dense retrieval models such as BAAI/bge-small-en-v1.5, embeddings are trained "
            "to maximize cosine similarity between queries and relevant passages.\n"
            "When vectors are L2-normalized (||u||_2 = ||v||_2 = 1.0), the Inner Product is "
            "mathematically identical to Cosine Similarity:\n"
            "    <u, v> = ||u|| * ||v|| * cos(theta) = cos(theta)\n"
            "Furthermore, squared Euclidean distance is strictly monotonic with respect to Inner Product:\n"
            "    ||u - v||_2^2 = ||u||^2 + ||v||^2 - 2<u, v> = 2 - 2<u, v>\n"
            "Maximizing <u, v> is strictly equivalent to minimizing ||u - v||_2. Using METRIC_INNER_PRODUCT "
            "avoids extra O(d) Euclidean subtraction operations per distance evaluation and directly "
            "yields cosine similarity scores in the range [-1.0, 1.0]."
        )
    return (
        f"Choice of Metric: {metric}\n"
        "Computes Euclidean (L2) distance. Note that for normalized embeddings, minimizing L2 "
        "produces identical ranking to maximizing Inner Product."
    )


def ensure_l2_normalized(vectors: np.ndarray, tolerance: float = 1e-4) -> np.ndarray:
    """Validates and ensures vectors are L2-normalized.

    If any vector norm differs from 1.0 by more than tolerance, normalizes in-place or returns
    a normalized copy.
    """
    if vectors.shape[0] == 0:
        return vectors

    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=tolerance):
        logger.debug("Vectors deviate from unit norm; applying faiss.normalize_L2")
        vecs_norm = vectors.copy()
        faiss.normalize_L2(vecs_norm)
        return vecs_norm
    return vectors


def extract_index_ids(index: faiss.Index) -> Set[int]:
    """Extracts all 64-bit IDs stored inside a FAISS index across Flat/IVF/PQ variants."""
    ids: Set[int] = set()

    # Case 1: IndexIDMap or IndexIDMap2
    if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        arr = faiss.vector_to_array(index.id_map)
        ids.update(int(x) for x in arr)
        return ids

    # Case 2: Inverted file index (IndexIVFFlat, IndexIVFPQ)
    if hasattr(index, "invlists") and index.invlists is not None:
        invlists = index.invlists
        for list_no in range(index.nlist):
            list_size = invlists.list_size(list_no)
            if list_size > 0:
                list_ids = faiss.rev_swig_ptr(invlists.get_ids(list_no), list_size)
                ids.update(int(x) for x in list_ids)
        return ids

    # Case 3: Raw index without explicit ID map (0 .. ntotal - 1)
    if index.ntotal > 0:
        ids.update(range(index.ntotal))

    return ids


class FAISSIndexer:
    """High-performance FAISS vector index manager supporting Flat, IVF-Flat, and IVF-PQ."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        index_type: str = "IVF-Flat",
        metric: str = "INNER_PRODUCT",
        nlist: int = DEFAULT_NLIST,
        m_subquantizers: int = DEFAULT_M_SUBQUANTIZERS,
        bits_per_code: int = DEFAULT_BITS_PER_CODE,
        auto_nlist: bool = True,
        index: Optional[faiss.Index] = None,
    ):
        self.dim = dim
        self.index_type = index_type.strip()
        self.metric_name = metric.upper()
        self.nlist = nlist
        self.m_subquantizers = m_subquantizers
        self.bits_per_code = bits_per_code
        self.auto_nlist = auto_nlist

        # Resolve FAISS metric
        if self.metric_name in ("INNER_PRODUCT", "IP"):
            self.faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric_name in ("L2", "EUCLIDEAN"):
            self.faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {self.metric_name}. Use 'INNER_PRODUCT' or 'L2'.")

        self.index: Optional[faiss.Index] = index
        self.seen_chunk_ids: Set[int] = set()

        if self.index is not None:
            self.seen_chunk_ids = extract_index_ids(self.index)
        elif self._is_exact_type():
            # Flat exact index does not require training
            self._create_empty_index(self.nlist)

        logger.info(
            f"Initialized FAISSIndexer (type={self.index_type}, dim={self.dim}, "
            f"metric={self.metric_name}, nlist={self.nlist}, "
            f"seen_ids={len(self.seen_chunk_ids)})"
        )

    def _is_exact_type(self) -> bool:
        """Returns True if index type is exact brute-force without training."""
        norm = self.index_type.upper().replace("-", "").replace("_", "")
        return norm in ("FLAT", "FLATIP", "FLATL2", "INDEXFLATIP", "INDEXFLATL2")

    def _create_empty_index(self, effective_nlist: int):
        """Constructs the underlying FAISS index object."""
        norm = self.index_type.upper().replace("-", "").replace("_", "")

        if norm in ("FLAT", "FLATIP", "INDEXFLATIP"):
            base_index = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap2(base_index)
        elif norm in ("FLATL2", "INDEXFLATL2"):
            base_index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIDMap2(base_index)
        elif norm in ("IVFFLAT", "INDEXIVFFLAT"):
            quantizer = faiss.IndexFlat(self.dim, self.faiss_metric)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, effective_nlist, self.faiss_metric
            )
        elif norm in ("IVFPQ", "INDEXIVFPQ"):
            # Verify subquantizer divisibility
            if self.dim % self.m_subquantizers != 0:
                raise ValueError(
                    f"Dimension {self.dim} must be divisible by m_subquantizers {self.m_subquantizers}!"
                )
            quantizer = faiss.IndexFlat(self.dim, self.faiss_metric)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                effective_nlist,
                self.m_subquantizers,
                self.bits_per_code,
                self.faiss_metric,
            )
        else:
            raise ValueError(
                f"Unknown index_type: {self.index_type}. "
                f"Supported: 'FlatIP', 'FlatL2', 'IVF-Flat', 'IVF-PQ'."
            )

    @property
    def is_trained(self) -> bool:
        """Checks if underlying index is trained and ready to accept vectors."""
        if self.index is None:
            return False
        return bool(self.index.is_trained)

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal if self.index is not None else 0

    def train(self, training_vectors: np.ndarray) -> float:
        """Trains the IVF or PQ index using a representative sample of vectors.

        Args:
            training_vectors: 2D array of shape (N, dim), float32.

        Returns:
            float: Training duration in seconds.
        """
        if self.is_trained:
            logger.info("Index is already trained; skipping training step.")
            return 0.0

        if training_vectors.ndim != 2 or training_vectors.shape[1] != self.dim:
            raise ValueError(
                f"Training vectors shape {training_vectors.shape} does not match expected (N, {self.dim})!"
            )

        n_train = len(training_vectors)
        if n_train == 0:
            raise ValueError("Training vectors cannot be empty!")

        # Normalize if inner product
        if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
            training_vectors = ensure_l2_normalized(training_vectors)
        training_vectors = np.ascontiguousarray(training_vectors, dtype=np.float32)

        # Dynamic cluster sizing if auto_nlist is enabled
        effective_nlist = self.nlist
        if self.auto_nlist and not self._is_exact_type():
            # FAISS recommends at least 39 points per centroid for k-means
            max_safe_nlist = max(16, n_train // 39)
            if effective_nlist > max_safe_nlist:
                logger.warning(
                    f"Configured nlist={self.nlist} exceeds safe centroid ratio for {n_train} training samples. "
                    f"Clamping nlist to {max_safe_nlist} to avoid underspecified k-means clustering."
                )
                effective_nlist = max_safe_nlist
                self.nlist = effective_nlist

        # PQ training requirement: 2^bits_per_code points per subquantizer
        if "PQ" in self.index_type.upper():
            min_pq_points = 1 << self.bits_per_code  # 256 for 8-bit
            if n_train < min_pq_points:
                raise ValueError(
                    f"IVF-PQ requires at least {min_pq_points} training points, but got {n_train}!"
                )

        if self.index is None:
            self._create_empty_index(effective_nlist)

        t0 = time.time()
        logger.info(f"Training {self.index_type} on {n_train:,} sample vectors...")
        self.index.train(training_vectors)
        elapsed = time.time() - t0
        logger.info(f"Index training complete in {elapsed:.3f}s (is_trained={self.is_trained})")
        return elapsed

    def add_vectors(
        self,
        embeddings: np.ndarray,
        chunk_ids: np.ndarray,
        skip_duplicates: bool = True,
    ) -> int:
        """Adds a batch of vectors with corresponding 64-bit chunk_ids to the index.

        Args:
            embeddings: 2D array of shape (N, dim), float32.
            chunk_ids: 1D array of length N, int64.
            skip_duplicates: If True, filters out IDs already present in the index.
                             If False, raises ValueError on duplicate detection.

        Returns:
            int: Number of unique vectors added.
        """
        if not self.is_trained:
            raise RuntimeError(f"Cannot add vectors: index of type {self.index_type} is not trained!")

        if len(embeddings) == 0:
            return 0

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embeddings shape {embeddings.shape} incompatible with index dimension {self.dim}!"
            )

        if len(embeddings) != len(chunk_ids):
            raise ValueError(
                f"Mismatched counts: {len(embeddings)} embeddings vs {len(chunk_ids)} chunk_ids!"
            )

        # Normalize if inner product
        if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
            embeddings = ensure_l2_normalized(embeddings)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        chunk_ids = np.ascontiguousarray(chunk_ids, dtype=np.int64)

        # Duplicate detection and prevention
        keep_mask = np.ones(len(chunk_ids), dtype=bool)
        duplicates_found = 0
        batch_seen: Set[int] = set()

        for idx, cid in enumerate(chunk_ids):
            cid_int = int(cid)
            if cid_int in self.seen_chunk_ids or cid_int in batch_seen:
                duplicates_found += 1
                if not skip_duplicates:
                    raise ValueError(f"Duplicate chunk_id detected: {cid_int}")
                keep_mask[idx] = False
            else:
                batch_seen.add(cid_int)

        if duplicates_found > 0:
            logger.warning(
                f"Detected {duplicates_found:,} duplicate chunk_ids in batch of {len(chunk_ids):,}; "
                f"skipping duplicates to prevent index corruption."
            )

        valid_count = int(np.sum(keep_mask))
        if valid_count == 0:
            return 0

        valid_embeddings = embeddings[keep_mask]
        valid_ids = chunk_ids[keep_mask]

        self.index.add_with_ids(valid_embeddings, valid_ids)
        self.seen_chunk_ids.update(int(x) for x in valid_ids)

        return valid_count

    def add_from_parquet_shard(
        self,
        shard_path: Union[str, Path],
        skip_duplicates: bool = True,
    ) -> int:
        """Reads a single Parquet embedding shard and adds vectors incrementally."""
        path = Path(shard_path)
        if not path.is_file():
            raise FileNotFoundError(f"Embedding shard not found: {path}")

        table = pq.read_table(str(path), columns=["chunk_id", "embedding"])
        if table.num_rows == 0:
            return 0

        chunk_ids = np.array(
            table.column("chunk_id").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        flat_vecs = table.column("embedding").combine_chunks().values.to_numpy(zero_copy_only=False)
        embeddings = flat_vecs.reshape(-1, self.dim).astype(np.float32)

        added = self.add_vectors(embeddings, chunk_ids, skip_duplicates=skip_duplicates)
        logger.debug(f"Added {added:,} vectors from shard {path.name} (total={self.ntotal:,})")
        return added

    def add_from_shards(
        self,
        shard_paths: List[Union[str, Path]],
        checkpoint: Optional[PipelineCheckpoint] = None,
        checkpoint_stage: str = "faiss_indexing",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """Streams multiple Parquet shards incrementally into the index with checkpointing.

        Args:
            shard_paths: List of paths to Parquet embedding shards.
            checkpoint: PipelineCheckpoint to track and resume shard progress.
            checkpoint_stage: Stage name in the checkpoint.
            skip_duplicates: Whether to ignore duplicate IDs.

        Returns:
            Dict containing ingestion statistics.
        """
        t0 = time.time()
        shards_processed = 0
        shards_skipped = 0
        total_vectors_added = 0

        for shard_path in shard_paths:
            path = Path(shard_path)
            shard_id = path.stem.replace("embeddings_", "").replace("chunks_", "")

            if checkpoint is not None and checkpoint.is_complete(checkpoint_stage, shard_id):
                shards_skipped += 1
                logger.debug(f"Skipping already indexed shard: {path.name}")
                continue

            added = self.add_from_parquet_shard(path, skip_duplicates=skip_duplicates)
            total_vectors_added += added
            shards_processed += 1

            if checkpoint is not None:
                checkpoint.mark_complete(checkpoint_stage, shard_id)
                checkpoint.save()

        elapsed = time.time() - t0
        rate = total_vectors_added / max(0.001, elapsed)

        stats = {
            "shards_processed": shards_processed,
            "shards_skipped": shards_skipped,
            "total_shards": len(shard_paths),
            "vectors_added": total_vectors_added,
            "ntotal": self.ntotal,
            "elapsed_seconds": round(elapsed, 3),
            "vectors_per_second": round(rate, 1),
        }
        logger.info(
            f"Ingestion complete: added {total_vectors_added:,} vectors from {shards_processed} shards "
            f"in {elapsed:.2f}s ({rate:,.1f} vec/s) | Total index size: {self.ntotal:,}"
        )
        return stats

    def search(
        self,
        queries: np.ndarray,
        top_k: int = 50,
        nprobe: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Searches the index for top-k nearest neighbors.

        Args:
            queries: 2D array of shape (N, dim), float32.
            top_k: Number of nearest neighbors to retrieve.
            nprobe: Number of Voronoi cells to visit (for IVF indexes).

        Returns:
            Tuple of (scores, chunk_ids):
                - scores: float32 array of shape (N, top_k)
                - chunk_ids: int64 array of shape (N, top_k)
        """
        if self.ntotal == 0:
            return np.empty((len(queries), 0), dtype=np.float32), np.empty((len(queries), 0), dtype=np.int64)

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self.dim:
            raise ValueError(f"Query dim {queries.shape[1]} does not match index dim {self.dim}!")

        if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
            queries = ensure_l2_normalized(queries)

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # Set nprobe if supported
        if nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        scores, ids = self.index.search(queries, top_k)
        return scores, ids

    def save(self, index_path: Union[str, Path]):
        """Persists the FAISS index to disk atomically with metadata sidecar."""
        if self.index is None:
            raise RuntimeError("Cannot save an uninitialized index!")

        out_path = Path(index_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".index.tmp")

        faiss.write_index(self.index, str(tmp_path))
        os.replace(str(tmp_path), str(out_path))

        # Save metadata sidecar
        meta = {
            "index_type": self.index_type,
            "metric": self.metric_name,
            "dim": self.dim,
            "nlist": self.nlist,
            "m_subquantizers": self.m_subquantizers,
            "bits_per_code": self.bits_per_code,
            "ntotal": self.ntotal,
            "seen_ids_count": len(self.seen_chunk_ids),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path = out_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved FAISS index to {out_path} ({self.ntotal:,} vectors, {self.get_disk_size_bytes(out_path)/(1024*1024):.2f} MB)")

    @classmethod
    def load(
        cls,
        index_path: Union[str, Path],
        use_mmap: bool = False,
    ) -> "FAISSIndexer":
        """Loads a FAISS index from disk safely, supporting memory mapping."""
        path = Path(index_path)
        if not path.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        flags = faiss.IO_FLAG_MMAP if use_mmap else 0
        logger.info(f"Loading FAISS index from {path} (mmap={use_mmap})...")
        loaded_index = faiss.read_index(str(path), flags)

        # Inspect sidecar metadata if available
        meta_path = path.with_suffix(".meta.json")
        index_type = "IVF-Flat"
        metric = "INNER_PRODUCT"
        nlist = DEFAULT_NLIST
        m_subquantizers = DEFAULT_M_SUBQUANTIZERS
        bits_per_code = DEFAULT_BITS_PER_CODE

        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                index_type = meta.get("index_type", index_type)
                metric = meta.get("metric", metric)
                nlist = meta.get("nlist", nlist)
                m_subquantizers = meta.get("m_subquantizers", m_subquantizers)
                bits_per_code = meta.get("bits_per_code", bits_per_code)
            except Exception as e:
                logger.warning(f"Failed to read metadata sidecar {meta_path}: {e}")

        indexer = cls(
            dim=loaded_index.d,
            index_type=index_type,
            metric=metric,
            nlist=nlist,
            m_subquantizers=m_subquantizers,
            bits_per_code=bits_per_code,
            index=loaded_index,
        )
        logger.info(f"Successfully loaded FAISS index: {indexer.ntotal:,} vectors, dim={indexer.dim}")
        return indexer

    def get_disk_size_bytes(self, index_path: Optional[Union[str, Path]] = None) -> int:
        """Returns the file size in bytes of the serialized index."""
        if index_path is not None and Path(index_path).is_file():
            return Path(index_path).stat().st_size
        return 0

