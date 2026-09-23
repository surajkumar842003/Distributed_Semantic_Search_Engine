"""Streaming Parquet reader using PyArrow."""
from pathlib import Path
from typing import Generator, List, Optional
import pyarrow.parquet as pq
from src.common.schemas import RawArticle


class ParquetCorpusReader:
    """
    Streams Wikipedia articles from Parquet row groups without loading the
    entire dataset into RAM.
    """
    def __init__(self, file_path: str | Path, batch_size: int = 1000):
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        if not self.file_path.exists():
            raise FileNotFoundError(f"Parquet source file not found: {self.file_path}")
        self._parquet_file = pq.ParquetFile(str(self.file_path))

    @property
    def total_rows(self) -> int:
        return self._parquet_file.metadata.num_rows

    @property
    def num_row_groups(self) -> int:
        return self._parquet_file.num_row_groups

    def iter_batches(self) -> Generator[List[RawArticle], None, None]:
        """
        Yields batches of RawArticle objects iteratively.
        """
        columns = ["id", "title", "url", "text"]
        # Check if categories column is in schema
        schema_names = self._parquet_file.schema.names
        has_categories = "categories" in schema_names
        cols_to_read = columns + (["categories"] if has_categories else [])

        for batch in self._parquet_file.iter_batches(batch_size=self.batch_size, columns=cols_to_read):
            ids = batch.column("id").to_pylist()
            titles = batch.column("title").to_pylist()
            urls = batch.column("url").to_pylist()
            texts = batch.column("text").to_pylist()
            cats = batch.column("categories").to_pylist() if has_categories else [[]] * len(ids)

            articles = [
                RawArticle(
                    id=int(ids[i]) if (isinstance(ids[i], int) or (isinstance(ids[i], str) and ids[i].isdigit())) else (i + 1),
                    title=titles[i] or "",
                    url=urls[i] or "",
                    text=texts[i] or "",
                    categories=cats[i] or [],
                )
                for i in range(len(ids))
            ]
            yield articles

