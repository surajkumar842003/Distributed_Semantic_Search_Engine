"""Unit tests for configuration loader."""
from pathlib import Path
from src.config import AppConfig


def test_load_config():
    config_dir = Path("/DATA/suraj/m1/search_engine/configs")
    cfg = AppConfig.load_from_dir(config_dir)

    assert cfg.project_name == "distributed_rag_wikipedia"
    assert cfg.ingestion.num_cpu_workers == 90
    assert cfg.ingestion.chunking.target_token_count == 256
    assert cfg.embedding.model_name == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.embedding_dim == 384
    assert cfg.faiss.index_type == "IVF-Flat"
    assert cfg.faiss.nlist == 32768
    assert cfg.postgres.port == 5432
    assert cfg.retrieval.dense_top_k == 50
    assert cfg.retrieval.rrf_k == 60

