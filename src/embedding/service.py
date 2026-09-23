"""GPU Embedding Service for Distributed Wikipedia Semantic Search & RAG.

Endpoints:
- GET  /health: GPU memory telemetry, model status, and readiness probe.
- POST /embed: High-throughput batch vector embedding generation.
- POST /embed/shards: Triggers StreamingEmbeddingPipeline for Parquet chunk shards.
- GET  /status: Ingestion checkpoint telemetry and processed vector statistics.

Designed for NVIDIA L4 (24 GB VRAM) with FP16 and dynamic bucketed batching.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import src.common.torch_compat
import torch

from src.config import AppConfig
from src.common.checkpoint import PipelineCheckpoint
from src.common.logging import get_logger
from src.embedding.embedder import GPUEmbedder, DEFAULT_EMBEDDING_DIM
from src.embedding.pipeline import StreamingEmbeddingPipeline

logger = get_logger("embedding.service")


# =============================================================================
# Request / Response Schemas
# =============================================================================

class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of raw texts to embed")
    normalize: bool = Field(default=True, description="Whether to L2-normalize vectors")


class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int
    latency_ms: float
    device: str


class ShardEmbedRequest(BaseModel):
    input_dir: Optional[str] = Field(default=None, description="Directory with chunks_*.parquet")
    output_dir: Optional[str] = Field(default=None, description="Directory to write embeddings_*.parquet")
    batch_size: Optional[int] = Field(default=None, description="Override GPU batch size")


class ShardEmbedResponse(BaseModel):
    status: str
    message: str
    input_dir: str
    output_dir: str


# =============================================================================
# Application State & Lifespan
# =============================================================================

service_state: Dict[str, Any] = {
    "config": None,
    "embedder": None,
    "pipeline": None,
    "start_time": time.time(),
    "auto_embed_task": None,
}


async def _background_shard_monitor(pipeline: StreamingEmbeddingPipeline, input_dir: str, output_dir: str, interval: int = 15):
    """Background task monitoring for newly arrived chunk shards to embed."""
    logger.info(f"Auto-embed shard monitor started (watching: {input_dir}, interval: {interval}s)")
    while True:
        try:
            p_in = Path(input_dir)
            if p_in.is_dir() and list(p_in.glob("chunks_*.parquet")):
                stats = pipeline.run(input_dir=input_dir, output_dir=output_dir)
                if stats.get("shards_processed", 0) > 0:
                    logger.info(f"Auto-embedded {stats['shards_processed']} new shards ({stats['vectors_added']} vectors).")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in auto-embed monitor: {e}", exc_info=True)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the GPU embedder and background embedding worker."""
    logger.info("Starting GPU Embedding Service...")
    cfg = AppConfig.load_from_dir("configs")
    service_state["config"] = cfg
    service_state["start_time"] = time.time()

    # Determine device
    device_str = os.environ.get("EMBEDDING_DEVICE", cfg.embedding.device)
    if not torch.cuda.is_available() and device_str.startswith("cuda"):
        logger.warning(f"CUDA requested ({device_str}) but not available; falling back to CPU.")
        device_str = "cpu"

    try:
        embedder = GPUEmbedder(
            model_name=cfg.embedding.model_name,
            cache_dir=cfg.paths.hf_home,
            device=device_str,
            use_fp16=(cfg.embedding.precision == "fp16" and device_str != "cpu"),
            expected_dim=cfg.embedding.embedding_dim,
        )
        service_state["embedder"] = embedder
        logger.info(f"GPUEmbedder ready on {device_str} (dim={embedder.expected_dim})")
    except Exception as e:
        logger.error(f"Failed to initialize GPUEmbedder: {e}", exc_info=True)
        service_state["embedder"] = None

    # Initialize streaming pipeline
    chunks_dir = os.environ.get("CHUNKS_OUTPUT_DIR", str(Path(cfg.paths.data_dir) / "chunks"))
    embeddings_dir = os.environ.get("EMBEDDINGS_OUTPUT_DIR", str(Path(cfg.paths.data_dir) / "embeddings"))
    checkpoint_path = Path(embeddings_dir) / "embedding_checkpoint.json"
    checkpoint = PipelineCheckpoint(str(checkpoint_path))

    pipeline = StreamingEmbeddingPipeline(
        config=cfg,
        checkpoint=checkpoint,
        embedder=service_state["embedder"],
    )
    service_state["pipeline"] = pipeline

    # Optional auto-embed daemon
    auto_embed = os.environ.get("AUTO_EMBED_SHARDS", "false").lower() in ("true", "1", "yes")
    if auto_embed and service_state["embedder"] is not None:
        service_state["auto_embed_task"] = asyncio.create_task(
            _background_shard_monitor(pipeline, chunks_dir, embeddings_dir)
        )

    logger.info("Embedding service initialization complete.")
    yield

    # Clean shutdown
    logger.info("Shutting down GPU Embedding Service...")
    if service_state["auto_embed_task"] is not None:
        service_state["auto_embed_task"].cancel()
        try:
            await service_state["auto_embed_task"]
        except asyncio.CancelledError:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="GPU Embedding Service",
    description="High-throughput GPU vector embedding microservice for Wikipedia RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Component health and GPU VRAM telemetry probe."""
    embedder = service_state.get("embedder")
    is_healthy = embedder is not None
    uptime = time.time() - service_state.get("start_time", time.time())

    cuda_available = torch.cuda.is_available()
    gpu_info = {}

    if cuda_available:
        try:
            device_idx = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            allocated_bytes = torch.cuda.memory_allocated(device_idx)
            reserved_bytes = torch.cuda.memory_reserved(device_idx)

            gpu_info = {
                "device_name": torch.cuda.get_device_name(device_idx),
                "device_index": device_idx,
                "total_vram_gb": round(total_bytes / (1024**3), 2),
                "free_vram_gb": round(free_bytes / (1024**3), 2),
                "allocated_vram_mb": round(allocated_bytes / (1024**2), 1),
                "reserved_vram_mb": round(reserved_bytes / (1024**2), 1),
            }
        except Exception as e:
            gpu_info = {"error": str(e)}

    return {
        "status": "healthy" if is_healthy else "degraded",
        "uptime_seconds": round(uptime, 1),
        "model_loaded": is_healthy,
        "embedding_dim": embedder.expected_dim if embedder else DEFAULT_EMBEDDING_DIM,
        "cuda_available": cuda_available,
        "gpu": gpu_info,
    }


@app.post("/embed", response_model=EmbedBatchResponse)
async def generate_embeddings(req: EmbedBatchRequest):
    """Generates vector embeddings for a list of input texts."""
    embedder = service_state.get("embedder")
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPUEmbedder model is not loaded or initialization failed.",
        )

    t0 = time.perf_counter()
    vecs = embedder.embed_batch(req.texts)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return EmbedBatchResponse(
        embeddings=vecs.tolist(),
        dimension=embedder.expected_dim,
        count=len(req.texts),
        latency_ms=round(latency_ms, 2),
        device=str(embedder.device),
    )


@app.post("/embed/shards", response_model=ShardEmbedResponse)
async def embed_shards(req: ShardEmbedRequest, background_tasks: BackgroundTasks):
    """Triggers the streaming GPU embedding pipeline over chunk Parquet shards."""
    pipeline = service_state.get("pipeline")
    cfg = service_state.get("config")
    if pipeline is None or cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding pipeline is not ready.",
        )

    data_dir = Path(cfg.paths.data_dir)
    input_dir = req.input_dir or os.environ.get("CHUNKS_OUTPUT_DIR", str(data_dir / "chunks"))
    output_dir = req.output_dir or os.environ.get("EMBEDDINGS_OUTPUT_DIR", str(data_dir / "embeddings"))

    def _run():
        logger.info(f"Manual shard embedding started: {input_dir} -> {output_dir}")
        pipeline.run(input_dir=input_dir, output_dir=output_dir, batch_size=req.batch_size)

    background_tasks.add_task(_run)

    return ShardEmbedResponse(
        status="accepted",
        message="Shard embedding pipeline launched in background.",
        input_dir=input_dir,
        output_dir=output_dir,
    )


@app.get("/status")
async def get_embedding_status():
    """Returns telemetry on embedding shards and checkpoint state."""
    cfg = service_state.get("config")
    data_dir = Path(cfg.paths.data_dir) if cfg else Path("data")
    embeddings_dir = Path(os.environ.get("EMBEDDINGS_OUTPUT_DIR", str(data_dir / "embeddings")))

    shards = list(embeddings_dir.glob("embeddings_*.parquet")) if embeddings_dir.is_dir() else []
    checkpoint_file = embeddings_dir / "embedding_checkpoint.json"

    completed_units = []
    if checkpoint_file.is_file():
        try:
            cp = PipelineCheckpoint(str(checkpoint_file))
            completed_units = cp.get_completed_units("embedding")
        except Exception:
            pass

    return {
        "output_directory": str(embeddings_dir),
        "total_shards_on_disk": len(shards),
        "completed_units_in_checkpoint": len(completed_units),
        "shard_files": [s.name for s in sorted(shards)[:20]],
    }
