"""Ingestion runner: executes Map stage chunking and cleaning over Wikipedia Parquet."""
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
from src.config import AppConfig
from src.ingestion.reader import ParquetCorpusReader
from src.ingestion.mapper import IngestionMapper
from src.ingestion.pipeline import IngestionPipeline
from src.common.checkpoint import PipelineCheckpoint
from src.common.logging import get_logger

logger = get_logger("scripts.run_ingest")


def run_ingestion(input_parquet: str, output_dir: str, config_dir: str = "configs", max_workers: int = None, legacy: bool = False):
    start_time = time.time()
    cfg = AppConfig.load_from_dir(config_dir)
    
    if max_workers is not None:
        cfg.ingestion.num_cpu_workers = max_workers
        
    workers = cfg.ingestion.num_cpu_workers
    logger.info(f"Starting Ingestion pipeline on {input_parquet} (legacy={legacy})")
    logger.info(f"Worker count: {workers}, Target chunk tokens: {cfg.ingestion.chunking.target_token_count}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if legacy:
        reader = ParquetCorpusReader(input_parquet, batch_size=500)
        logger.info(f"Loaded Parquet file with {reader.total_rows} articles across {reader.num_row_groups} row groups")

        mapper = IngestionMapper(
            num_workers=workers,
            target_tokens=cfg.ingestion.chunking.target_token_count,
            overlap_tokens=cfg.ingestion.chunking.overlap_token_count,
            min_tokens=cfg.ingestion.chunking.min_token_count,
            pin_numa=cfg.ingestion.pin_to_numa,
        )

        total_chunks = 0
        total_tokens = 0
        shard_idx = 0

        accum_chunks = []
        FLUSH_SIZE = 10000

        for chunk_batch in mapper.process_batches(reader.iter_batches(), sub_batch_size=100):
            accum_chunks.extend(chunk_batch)
            total_chunks += len(chunk_batch)
            total_tokens += sum(c.token_count for c in chunk_batch)

            if len(accum_chunks) >= FLUSH_SIZE:
                _flush_shard(accum_chunks, out_path / f"chunks_{shard_idx:04d}.parquet")
                shard_idx += 1
                accum_chunks.clear()

        if accum_chunks:
            _flush_shard(accum_chunks, out_path / f"chunks_{shard_idx:04d}.parquet")
            shard_idx += 1
            accum_chunks.clear()

        elapsed = time.time() - start_time
        art_rate = reader.total_rows / elapsed if elapsed > 0 else 0
        chunk_rate = total_chunks / elapsed if elapsed > 0 else 0

        logger.info("=" * 60)
        logger.info(f"Ingestion Completed in {elapsed:.2f} seconds")
        logger.info(f"Total Articles Processed: {reader.total_rows:,} ({art_rate:.1f} articles/sec)")
        logger.info(f"Total Chunks Emitted:    {total_chunks:,} ({chunk_rate:.1f} chunks/sec)")
        logger.info(f"Average Chunks / Article: {total_chunks / max(1, reader.total_rows):.2f}")
        logger.info(f"Average Tokens / Chunk:   {total_tokens / max(1, total_chunks):.1f}")
        logger.info(f"Output Shards Written:    {shard_idx} to {out_path}")
        logger.info("=" * 60)
    else:
        checkpoint_path = out_path / "ingestion_checkpoint.json"
        checkpoint = PipelineCheckpoint(str(checkpoint_path))
        pipeline = IngestionPipeline(config=cfg, checkpoint=checkpoint)
        
        stats = pipeline.run(input_parquet, str(out_path))
        
        logger.info("=" * 60)
        logger.info(f"Ingestion Completed in {stats['elapsed_seconds']:.2f} seconds")
        logger.info(f"Total Articles Processed: {stats['articles_processed']:,}")
        logger.info("=" * 60)


def _flush_shard(chunks: list, file_path: Path):
    table = pa.Table.from_arrays(
        [
            pa.array([c.chunk_id for c in chunks], type=pa.int64()),
            pa.array([c.doc_id for c in chunks], type=pa.int64()),
            pa.array([c.title for c in chunks], type=pa.string()),
            pa.array([c.url for c in chunks], type=pa.string()),
            pa.array([c.section_path for c in chunks], type=pa.string()),
            pa.array([c.text for c in chunks], type=pa.string()),
            pa.array([c.token_count for c in chunks], type=pa.int32()),
            pa.array([c.faiss_id for c in chunks], type=pa.int64()),
        ],
        names=["chunk_id", "doc_id", "title", "url", "section_path", "text", "token_count", "faiss_id"],
    )
    pq.write_table(table, str(file_path), compression="snappy")
    logger.info(f"Flushed shard {file_path.name} with {len(chunks):,} chunks ({file_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wikipedia Ingestion Map Stage")
    parser.add_argument("--input", type=str, default="data/raw/wikipedia_sample.parquet")
    parser.add_argument("--output", type=str, default="data/chunks")
    parser.add_argument("--config", type=str, default="configs")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--legacy", action="store_true", help="Use legacy IngestionMapper")
    args = parser.parse_args()
    run_ingestion(args.input, args.output, args.config, args.workers, args.legacy)
