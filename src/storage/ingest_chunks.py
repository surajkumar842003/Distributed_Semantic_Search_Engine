"""Streaming ingestion pipeline from Parquet chunk shards into PostgreSQL."""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pyarrow.parquet as pq

from src.config import AppConfig, PostgresConfig
from src.storage.postgres_client import PostgresClient
from src.common.checkpoint import PipelineCheckpoint
from src.common.logging import get_logger

logger = get_logger("storage.ingest_chunks")


class PostgresIngester:
    """Streams Parquet chunk shards into PostgreSQL using COPY with checkpointing."""

    def __init__(
        self,
        client: PostgresClient,
        checkpoint: Optional[PipelineCheckpoint] = None,
        checkpoint_stage: str = "postgres_ingestion",
        batch_size: int = 10000,
    ):
        self.client = client
        self.checkpoint = checkpoint
        self.checkpoint_stage = checkpoint_stage
        self.batch_size = batch_size

    async def ingest_shard(self, shard_path: Union[str, Path]) -> Dict[str, int]:
        """Reads a single chunk Parquet file and ingests documents and chunks."""
        path = Path(shard_path)
        if not path.is_file():
            raise FileNotFoundError(f"Shard file not found: {path}")

        table = pq.read_table(str(path))
        num_rows = table.num_rows
        if num_rows == 0:
            return {"documents": 0, "chunks": 0}

        # 1. Extract unique documents for normalized metadata table
        doc_ids = table.column("doc_id").to_numpy(zero_copy_only=False)
        titles = table.column("title").to_pylist()
        urls = table.column("url").to_pylist()

        seen_docs = set()
        unique_docs = []
        for did, t, u in zip(doc_ids, titles, urls):
            did_int = int(did)
            if did_int not in seen_docs:
                seen_docs.add(did_int)
                unique_docs.append({"doc_id": did_int, "title": t, "url": u, "source_shard": 0})

        # Insert unique documents
        docs_added = await self.client.insert_documents_batch(unique_docs)

        # 2. Extract chunk records for COPY
        chunk_ids = table.column("chunk_id").to_numpy(zero_copy_only=False)
        sections = table.column("section_path").to_pylist()
        texts = table.column("text").to_pylist()
        tokens = table.column("token_count").to_numpy(zero_copy_only=False)
        faiss_ids = table.column("faiss_id").to_numpy(zero_copy_only=False) if "faiss_id" in table.column_names else chunk_ids

        records = [
            (
                int(chunk_ids[i]),
                int(doc_ids[i]),
                str(sections[i]),
                str(texts[i]),
                int(tokens[i]),
                int(faiss_ids[i]),
            )
            for i in range(num_rows)
        ]

        # Ingest chunks via high-speed COPY
        chunks_added = await self.client.copy_chunks_idempotent(records)

        return {"documents": docs_added, "chunks": chunks_added}

    async def ingest_directory(
        self,
        chunks_dir: Union[str, Path],
        max_shards: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Ingests all chunk Parquet shards from a directory with checkpoint resumption."""
        in_path = Path(chunks_dir)
        shard_files = sorted(in_path.glob("chunks_*.parquet"))
        if max_shards is not None:
            shard_files = shard_files[:max_shards]

        if not shard_files:
            logger.warning(f"No chunk Parquet files found in {chunks_dir}")
            return {"shards_processed": 0, "total_chunks": 0, "total_documents": 0}

        logger.info(f"Starting PostgreSQL ingestion: {len(shard_files)} shards in {chunks_dir}")
        t0 = time.time()
        total_chunks = 0
        total_docs = 0
        shards_processed = 0
        shards_skipped = 0

        for shard_file in shard_files:
            shard_id = shard_file.stem.replace("chunks_", "")

            if self.checkpoint is not None and self.checkpoint.is_complete(self.checkpoint_stage, shard_id):
                shards_skipped += 1
                logger.debug(f"Skipping already ingested shard: {shard_file.name}")
                continue

            stats = await self.ingest_shard(shard_file)
            total_docs += stats["documents"]
            total_chunks += stats["chunks"]
            shards_processed += 1

            if self.checkpoint is not None:
                self.checkpoint.mark_complete(self.checkpoint_stage, shard_id)
                self.checkpoint.save()

            logger.info(
                f"Ingested shard {shard_file.name}: {stats['chunks']:,} chunks, "
                f"{stats['documents']:,} docs (running total: {total_chunks:,} chunks)"
            )

        elapsed = time.time() - t0
        rate = total_chunks / max(0.001, elapsed)

        return {
            "shards_processed": shards_processed,
            "shards_skipped": shards_skipped,
            "total_shards": len(shard_files),
            "total_chunks": total_chunks,
            "total_documents": total_docs,
            "elapsed_seconds": round(elapsed, 3),
            "chunks_per_second": round(rate, 1),
        }

