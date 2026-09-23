"""Experiment configuration definitions and factory functions for ablation study.

Generates reproducible experiment configurations across 6 dimensions:
1. Index type:     FlatIP vs IVF-Flat vs IVF-PQ
2. nprobe:         1, 4, 8, 16, 32, 64, 128
3. Chunk size:     128, 256, 512 tokens
4. Retrieval mode: dense, sparse, hybrid_rrf, hybrid_rerank
5. Reranking:      with vs without cross-encoder
6. Worker count:   1, 2, 4, 8, 16 workers

Each config is a frozen snapshot of all parameters needed to reproduce the experiment.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class ExperimentConfig:
    """Immutable experiment configuration for a single ablation run."""

    experiment_id: str
    dimension: str  # "index_type" | "nprobe" | "chunk_size" | "retrieval_mode" | "reranking" | "worker_count"
    description: str

    # Index parameters
    index_type: str = "FlatIP"         # "FlatIP" | "IVF-Flat" | "IVF-PQ"
    nlist: int = 256
    nprobe: int = 16
    m_subquantizers: int = 48
    bits_per_code: int = 8

    # Chunking parameters
    chunk_target_tokens: int = 256
    chunk_overlap_tokens: int = 40
    chunk_min_tokens: int = 25

    # Retrieval parameters
    retrieval_mode: str = "dense"      # "dense" | "sparse" | "hybrid_rrf" | "hybrid_rerank"
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    final_rerank_top_k: int = 10

    # Ingestion parameters
    num_workers: int = 4

    # Paths (set at runtime)
    embeddings_dir: str = ""
    chunks_dir: str = ""
    index_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Path):
        """Saves config to a YAML file for reproducibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Loads config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_json(self, path: Path):
        """Saves config to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Default shared parameters (held constant when not the ablation variable)
# ---------------------------------------------------------------------------

_DEFAULT_NLIST = 256
_DEFAULT_NPROBE = 16
_DEFAULT_M_SUB = 48
_DEFAULT_BITS = 8
_DEFAULT_CHUNK_TOKENS = 256
_DEFAULT_CHUNK_OVERLAP = 40
_DEFAULT_CHUNK_MIN = 25


# ---------------------------------------------------------------------------
# Dimension 1: Index Type
# ---------------------------------------------------------------------------

def generate_index_type_experiments() -> List[ExperimentConfig]:
    """Compares FlatIP (exact), IVF-Flat (approximate), and IVF-PQ (compressed)."""
    configs = []

    configs.append(ExperimentConfig(
        experiment_id="idx_flat_ip",
        dimension="index_type",
        description="Exact brute-force IndexFlatIP (baseline)",
        index_type="FlatIP",
        nprobe=1,  # N/A for flat, but set for consistency
        retrieval_mode="dense",
    ))

    configs.append(ExperimentConfig(
        experiment_id="idx_ivf_flat",
        dimension="index_type",
        description=f"IVF-Flat with nlist={_DEFAULT_NLIST}, nprobe={_DEFAULT_NPROBE}",
        index_type="IVF-Flat",
        nlist=_DEFAULT_NLIST,
        nprobe=_DEFAULT_NPROBE,
        retrieval_mode="dense",
    ))

    configs.append(ExperimentConfig(
        experiment_id="idx_ivf_pq",
        dimension="index_type",
        description=f"IVF-PQ with nlist={_DEFAULT_NLIST}, m={_DEFAULT_M_SUB}, nprobe={_DEFAULT_NPROBE}",
        index_type="IVF-PQ",
        nlist=_DEFAULT_NLIST,
        nprobe=_DEFAULT_NPROBE,
        m_subquantizers=_DEFAULT_M_SUB,
        bits_per_code=_DEFAULT_BITS,
        retrieval_mode="dense",
    ))

    return configs


# ---------------------------------------------------------------------------
# Dimension 2: nprobe Sweep
# ---------------------------------------------------------------------------

def generate_nprobe_experiments() -> List[ExperimentConfig]:
    """Sweeps nprobe for IVF-Flat to show recall-latency tradeoff."""
    nprobe_values = [1, 4, 8, 16, 32, 64, 128]
    configs = []

    for np_val in nprobe_values:
        configs.append(ExperimentConfig(
            experiment_id=f"nprobe_ivf_flat_{np_val:03d}",
            dimension="nprobe",
            description=f"IVF-Flat with nprobe={np_val}",
            index_type="IVF-Flat",
            nlist=_DEFAULT_NLIST,
            nprobe=np_val,
            retrieval_mode="dense",
        ))

    return configs


# ---------------------------------------------------------------------------
# Dimension 3: Chunk Size
# ---------------------------------------------------------------------------

def generate_chunk_size_experiments() -> List[ExperimentConfig]:
    """Compares different chunk target token counts."""
    chunk_sizes = [128, 256, 512]
    configs = []

    for cs in chunk_sizes:
        overlap = max(20, cs // 6)  # ~16% overlap, scaled
        configs.append(ExperimentConfig(
            experiment_id=f"chunk_{cs:04d}",
            dimension="chunk_size",
            description=f"Chunk target={cs} tokens, overlap={overlap}",
            index_type="FlatIP",
            chunk_target_tokens=cs,
            chunk_overlap_tokens=overlap,
            chunk_min_tokens=max(15, cs // 10),
            retrieval_mode="dense",
        ))

    return configs


# ---------------------------------------------------------------------------
# Dimension 4: Retrieval Mode
# ---------------------------------------------------------------------------

def generate_retrieval_mode_experiments() -> List[ExperimentConfig]:
    """Compares dense, sparse, hybrid RRF, and hybrid + reranking."""
    modes = [
        ("dense", "Dense retrieval via FAISS only"),
        ("sparse", "Sparse retrieval via BM25 only"),
        ("hybrid_rrf", "Hybrid RRF (dense + sparse, k=60)"),
        ("hybrid_rerank", "Hybrid RRF + cross-encoder reranking"),
    ]
    configs = []

    for mode, desc in modes:
        configs.append(ExperimentConfig(
            experiment_id=f"ret_{mode}",
            dimension="retrieval_mode",
            description=desc,
            index_type="FlatIP",
            retrieval_mode=mode,
        ))

    return configs


# ---------------------------------------------------------------------------
# Dimension 5: Reranking
# ---------------------------------------------------------------------------

def generate_reranking_experiments() -> List[ExperimentConfig]:
    """Compares hybrid retrieval with and without cross-encoder reranking."""
    configs = []

    configs.append(ExperimentConfig(
        experiment_id="rerank_off",
        dimension="reranking",
        description="Hybrid RRF without reranking",
        index_type="FlatIP",
        retrieval_mode="hybrid_rrf",
    ))

    configs.append(ExperimentConfig(
        experiment_id="rerank_on",
        dimension="reranking",
        description="Hybrid RRF with cross-encoder reranking",
        index_type="FlatIP",
        retrieval_mode="hybrid_rerank",
    ))

    return configs


# ---------------------------------------------------------------------------
# Dimension 6: Worker Count
# ---------------------------------------------------------------------------

def generate_worker_count_experiments() -> List[ExperimentConfig]:
    """Compares ingestion throughput across different worker counts."""
    worker_counts = [1, 2, 4, 8, 16]
    configs = []

    for wc in worker_counts:
        configs.append(ExperimentConfig(
            experiment_id=f"workers_{wc:02d}",
            dimension="worker_count",
            description=f"Ingestion pipeline with {wc} CPU workers",
            num_workers=wc,
            index_type="FlatIP",
            retrieval_mode="dense",
        ))

    return configs


# ---------------------------------------------------------------------------
# All experiments
# ---------------------------------------------------------------------------

def generate_all_experiments() -> List[ExperimentConfig]:
    """Returns all 24 experiment configurations across all 6 dimensions."""
    all_configs = []
    all_configs.extend(generate_index_type_experiments())
    all_configs.extend(generate_nprobe_experiments())
    all_configs.extend(generate_chunk_size_experiments())
    all_configs.extend(generate_retrieval_mode_experiments())
    all_configs.extend(generate_reranking_experiments())
    all_configs.extend(generate_worker_count_experiments())

    # Validate uniqueness
    ids = [c.experiment_id for c in all_configs]
    if len(ids) != len(set(ids)):
        dupes = [x for x in ids if ids.count(x) > 1]
        raise ValueError(f"Duplicate experiment IDs detected: {set(dupes)}")

    return all_configs


def save_all_configs(configs: List[ExperimentConfig], output_dir: Path):
    """Saves all experiment configs to YAML files in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        cfg.save_yaml(output_dir / f"{cfg.experiment_id}.yaml")

    # Also save a manifest
    manifest = {
        "total_experiments": len(configs),
        "dimensions": list(set(c.dimension for c in configs)),
        "experiment_ids": [c.experiment_id for c in configs],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

