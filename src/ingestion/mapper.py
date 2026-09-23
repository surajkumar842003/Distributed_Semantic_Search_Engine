"""Multiprocessing Map stage manager with NUMA awareness."""
import os
import multiprocessing as mp
from typing import List, Generator, Tuple
from src.common.schemas import RawArticle, DocumentChunk
from src.ingestion.chunker import HeadingAwareChunker
from src.common.logging import get_logger

logger = get_logger("ingestion.mapper")


def _init_worker(pin_numa: bool):
    """Initializes worker process and optionally binds CPU affinity to NUMA node."""
    if pin_numa:
        try:
            # Dual EPYC: Node 0 = CPUs 0-63,128-191; Node 1 = CPUs 64-127,192-255
            # Alternate workers across NUMA nodes
            proc_name = mp.current_process().name
            worker_num = int(proc_name.split('-')[-1]) if '-' in proc_name else 0
            node = worker_num % 2
            if hasattr(os, "sched_setaffinity"):
                if node == 0:
                    cpus = set(range(0, 64)) | set(range(128, 192))
                else:
                    cpus = set(range(64, 128)) | set(range(192, 256))
                os.sched_setaffinity(0, cpus)
        except Exception as e:
            # Fallback if unprivileged or affinity unsupported
            pass


def _process_article_batch(args: Tuple[List[RawArticle], int, int, int]) -> List[DocumentChunk]:
    """Worker function executed inside multiprocessing pool."""
    articles, target_tokens, overlap_tokens, min_tokens = args
    chunker = HeadingAwareChunker(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    )
    all_chunks: List[DocumentChunk] = []
    for art in articles:
        chunks = chunker.chunk_article(art)
        all_chunks.extend(chunks)
    return all_chunks


class IngestionMapper:
    """
    Orchestrates the parallel Map stage across CPU cores.
    Takes article batches from ParquetReader and maps them through the cleaner/chunker.
    """
    def __init__(
        self,
        num_workers: int = 90,
        target_tokens: int = 256,
        overlap_tokens: int = 40,
        min_tokens: int = 30,
        pin_numa: bool = True,
    ):
        self.num_workers = min(num_workers, mp.cpu_count())
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.pin_numa = pin_numa
        logger.info(f"Initialized IngestionMapper with {self.num_workers} workers (NUMA binding: {self.pin_numa})")

    def process_batches(
        self,
        batch_generator: Generator[List[RawArticle], None, None],
        sub_batch_size: int = 100,
    ) -> Generator[List[DocumentChunk], None, None]:
        """
        Streams articles through the multiprocessing pool and yields chunks in batches.
        """
        # Pool with initializer
        with mp.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.pin_numa,),
        ) as pool:
            for article_batch in batch_generator:
                # Partition article batch into sub-batches for workers
                sub_batches = [
                    (
                        article_batch[i:i + sub_batch_size],
                        self.target_tokens,
                        self.overlap_tokens,
                        self.min_tokens,
                    )
                    for i in range(0, len(article_batch), sub_batch_size)
                ]

                # Map across worker pool
                results = pool.map(_process_article_batch, sub_batches)
                for chunk_list in results:
                    if chunk_list:
                        yield chunk_list

