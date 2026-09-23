"""Dataset normalization module for Wikipedia articles."""
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


@dataclass
class NormalizedArticle:
    """
    Standard normalized schema for Wikipedia articles across all downstream pipeline stages:
    doc_id: 64-bit integer identifier
    title: Canonical article title
    url: Canonical article URL
    text: Cleaned article body text
    source_shard: Integer identifying the source Parquet shard (e.g. 0 for shard 0)
    """
    doc_id: int
    title: str
    url: str
    text: str
    source_shard: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArticleNormalizer:
    """
    Normalizes raw Wikipedia Parquet records into the standardized schema:
    [doc_id: int64, title: string, url: string, text: string, source_shard: int32].
    Handles string-to-int conversion, whitespace stripping, URL verification, and shard identification.
    """
    def __init__(self, min_chars: int = 50, default_shard: int = 0):
        self.min_chars = min_chars
        self.default_shard = default_shard
        self._url_pattern = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
        self._shard_pattern = re.compile(r"(\d+)(?:-of-\d+)?(?:\.parquet)?$", re.IGNORECASE)

    def extract_shard_id(self, source_path: str | Path) -> int:
        """
        Extracts integer shard index from filename.
        Examples:
          'train-00003-of-00041.parquet' -> 3
          'shard_0012.parquet'           -> 12
          'wikipedia_pilot_50k.parquet'  -> 0 (default fallback)
        """
        name = Path(source_path).stem
        # Match pattern like train-00003-of-00041
        m = re.search(r"(\d{4,6})-of-(\d{4,6})", name)
        if m:
            return int(m.group(1))

        # Match pattern like shard_0012 or chunks_0005
        m2 = re.search(r"(?:shard|chunks|part)?[_-]?(\d+)", name, re.IGNORECASE)
        if m2:
            return int(m2.group(1))

        return self.default_shard

    def parse_doc_id(self, raw_id: Any, fallback_index: int = 1) -> int:
        """Safely parses document ID from int, float, or string."""
        if raw_id is None:
            return fallback_index
        if isinstance(raw_id, (int, float)):
            return int(raw_id)
        if isinstance(raw_id, str):
            clean = raw_id.strip()
            if clean.isdigit():
                return int(clean)
            # Try to extract numbers if format is e.g. "wiki_12345"
            digits = "".join(filter(str.isdigit, clean))
            if digits:
                return int(digits)
        return fallback_index

    def normalize_record(
        self,
        raw_record: Dict[str, Any],
        source_shard: Optional[int] = None,
        fallback_index: int = 1,
    ) -> Tuple[NormalizedArticle, Dict[str, Any]]:
        """
        Transforms a raw record into NormalizedArticle while returning quality flags.
        """
        raw_id = raw_record.get("id") or raw_record.get("doc_id")
        doc_id = self.parse_doc_id(raw_id, fallback_index=fallback_index)

        title = str(raw_record.get("title") or "").strip()
        url = str(raw_record.get("url") or "").strip()
        text = str(raw_record.get("text") or raw_record.get("content") or "").strip()
        shard_id = self.default_shard if source_shard is None else int(source_shard)

        # Quality diagnostics
        flags = {
            "has_empty_text": len(text) == 0,
            "is_extremely_short": len(text) < self.min_chars,
            "has_empty_title": len(title) == 0,
            "has_empty_url": len(url) == 0,
            "has_valid_url": bool(self._url_pattern.match(url)),
            "char_length": len(text),
        }

        normalized = NormalizedArticle(
            doc_id=doc_id,
            title=title,
            url=url,
            text=text,
            source_shard=shard_id,
        )

        return normalized, flags

    def is_valid(self, article: NormalizedArticle) -> bool:
        """Determines if a normalized article meets quality thresholds for downstream indexing."""
        if not article.title or not article.text:
            return False
        if len(article.text) < self.min_chars:
            return False
        if not article.url.startswith("http"):
            return False
        return True

