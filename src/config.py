"""Configuration loader and schema definition."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class PathsConfig:
    data_dir: str = "/DATA/suraj/m1/search_engine/data"
    raw_parquet_path: str = "/DATA/suraj/m1/search_engine/data/raw/wikipedia.parquet"
    chunks_staging_dir: str = "/DATA/suraj/m1/search_engine/data/chunks"
    index_dir: str = "/DATA/suraj/m1/search_engine/data/indexes"
    cache_dir: str = "/DATA/suraj/m1/search_engine/data/cache"
    embeddings_dir: str = "/DATA/suraj/m1/search_engine/data/embeddings"
    shm_staging_dir: str = "/dev/shm/search_engine_staging"

    @property
    def hf_home(self) -> str:
        return os.environ.get("HF_HOME", os.path.join(self.cache_dir, "huggingface"))

    @property
    def torch_home(self) -> str:
        return os.environ.get("TORCH_HOME", os.path.join(self.cache_dir, "torch"))


@dataclass
class ChunkingConfig:
    target_token_count: int = 256
    min_token_count: int = 40
    overlap_token_count: int = 40
    prepend_section_headers: bool = True
    preserve_heading_path: bool = True
    tokenizer_name: str = "BAAI/bge-small-en-v1.5"


@dataclass
class IngestionConfig:
    num_cpu_workers: int = 90
    worker_chunksize: int = 500
    pin_to_numa: bool = True
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    use_shm: bool = True
    max_queue_memory_mb: int = 8192
    flush_batch_size: int = 5000


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    batch_size: int = 64
    precision: str = "fp16"
    max_seq_length: int = 256
    normalize_embeddings: bool = True
    device: str = "cuda:0"
    use_torch_compile: bool = False
    torch_compile_mode: str = "default"
    use_onnx: bool = False
    bucket_by_length: bool = True
    bucket_boundaries: List[int] = field(default_factory=lambda: [64, 128, 192, 256])
    pinned_memory: bool = True
    num_workers: int = 4


@dataclass
class FaissConfig:
    index_type: str = "IVF-Flat"
    metric: str = "INNER_PRODUCT"
    nlist: int = 32768
    nprobe: int = 64
    train_sample_size: int = 1000000
    index_path: str = "/DATA/suraj/m1/search_engine/data/indexes/wiki_ivf_flat.index"
    use_mmap: bool = True
    m_subquantizers: int = 48
    bits_per_code: int = 8
    auto_nlist: bool = True


@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "wikirag"
    user: str = "postgres"
    password: str = "postgres_secure_password"
    pool_size_min: int = 10
    pool_size_max: int = 40
    shared_buffers: str = "64GB"
    maintenance_work_mem: str = "32GB"
    max_parallel_maintenance_workers: int = 16


@dataclass
class RetrievalConfig:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    final_rerank_top_k: int = 5
    rerank_model_name: str = "BAAI/bge-reranker-base"
    rerank_device: str = "cuda:0"


@dataclass
class AppConfig:
    project_name: str = "distributed_rag_wikipedia"
    version: str = "0.1.0"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def load_from_dir(cls, config_dir: str | Path) -> "AppConfig":
        dir_path = Path(config_dir)
        base = load_yaml(dir_path / "base.yaml") if (dir_path / "base.yaml").is_file() else {}
        ingest_data = load_yaml(dir_path / "ingestion.yaml") if (dir_path / "ingestion.yaml").is_file() else {}
        embed_data = load_yaml(dir_path / "embedding.yaml") if (dir_path / "embedding.yaml").is_file() else {}
        index_data = load_yaml(dir_path / "index.yaml") if (dir_path / "index.yaml").is_file() else {}
        pg_data = load_yaml(dir_path / "postgres.yaml") if (dir_path / "postgres.yaml").is_file() else {}
        serve_data = load_yaml(dir_path / "serving.yaml") if (dir_path / "serving.yaml").is_file() else {}

        paths = PathsConfig(**base.get("paths", {}))
        
        chunking_dict = ingest_data.get("ingestion", {}).get("chunking", {})
        chunking = ChunkingConfig(**chunking_dict) if chunking_dict else ChunkingConfig()
        
        ingest_dict = ingest_data.get("ingestion", {})
        ingest_filtered = {k: v for k, v in ingest_dict.items() if k not in ("chunking", "buffer")}
        if "buffer" in ingest_dict:
            ingest_filtered.update(ingest_dict["buffer"])
        ingestion = IngestionConfig(chunking=chunking, **ingest_filtered)

        embed_dict = embed_data.get("embedding", {})
        opt_dict = embed_dict.pop("optimization", {})
        embed_dict.update(opt_dict)
        embedding = EmbeddingConfig(**embed_dict) if embed_dict else EmbeddingConfig()

        faiss_dict = index_data.get("faiss", {}).get("primary", {})
        faiss = FaissConfig(**faiss_dict) if faiss_dict else FaissConfig()

        pg_dict = pg_data.get("postgres", {})
        tuning_dict = pg_dict.pop("tuning", {})
        pg_dict.pop("bulk_load", None)
        pg_dict.update(tuning_dict)
        # remove keys not in PostgresConfig
        valid_pg_keys = {"host", "port", "database", "user", "password", "pool_size_min", "pool_size_max", "shared_buffers", "maintenance_work_mem", "max_parallel_maintenance_workers"}
        pg_clean = {k: v for k, v in pg_dict.items() if k in valid_pg_keys}
        postgres = PostgresConfig(**pg_clean)

        serve_ret = serve_data.get("serving", {}).get("retrieval", {})
        rerank_dict = serve_data.get("serving", {}).get("reranker", {})
        retrieval = RetrievalConfig(
            dense_top_k=serve_ret.get("dense_top_k", 50),
            sparse_top_k=serve_ret.get("sparse_top_k", 50),
            rrf_k=serve_ret.get("rrf_k", 60),
            final_rerank_top_k=serve_ret.get("final_rerank_top_k", 5),
            rerank_model_name=rerank_dict.get("model_name", "BAAI/bge-reranker-base"),
            rerank_device=rerank_dict.get("device", "cuda:0"),
        )

        # Environment Variable Overrides for Container and Cloud Deployments
        if os.environ.get("DATA_DIR"):
            paths.data_dir = os.environ["DATA_DIR"]
            paths.chunks_staging_dir = os.path.join(paths.data_dir, "chunks")
            paths.index_dir = os.path.join(paths.data_dir, "indexes")
            paths.embeddings_dir = os.path.join(paths.data_dir, "embeddings")
        if os.environ.get("CACHE_DIR"):
            paths.cache_dir = os.environ["CACHE_DIR"]
        if os.environ.get("RAW_PARQUET_PATH"):
            paths.raw_parquet_path = os.environ["RAW_PARQUET_PATH"]

        if os.environ.get("POSTGRES_HOST"):
            postgres.host = os.environ["POSTGRES_HOST"]
        if os.environ.get("POSTGRES_PORT"):
            postgres.port = int(os.environ["POSTGRES_PORT"])
        if os.environ.get("POSTGRES_DB"):
            postgres.database = os.environ["POSTGRES_DB"]
        if os.environ.get("POSTGRES_USER"):
            postgres.user = os.environ["POSTGRES_USER"]
        if os.environ.get("POSTGRES_PASSWORD"):
            postgres.password = os.environ["POSTGRES_PASSWORD"]

        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            embedding.device = f"cuda:{os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]}"
            retrieval.rerank_device = f"cuda:{os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]}"
        if os.environ.get("EMBEDDING_DEVICE"):
            embedding.device = os.environ["EMBEDDING_DEVICE"]
        if os.environ.get("RERANK_DEVICE"):
            retrieval.rerank_device = os.environ["RERANK_DEVICE"]
        if os.environ.get("FAISS_INDEX_PATH"):
            faiss.index_path = os.environ["FAISS_INDEX_PATH"]
        if os.environ.get("NUM_CPU_WORKERS"):
            ingestion.num_cpu_workers = int(os.environ["NUM_CPU_WORKERS"])

        return cls(
            project_name=base.get("project_name", "distributed_rag_wikipedia"),
            version=base.get("version", "0.1.0"),
            seed=base.get("seed", 42),
            paths=paths,
            ingestion=ingestion,
            embedding=embedding,
            faiss=faiss,
            postgres=postgres,
            retrieval=retrieval,
        )
