"""Wikipedia Dataset Audit & Quality Diagnostic Suite."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from src.common.logging import get_logger
from src.ingestion.normalizer import ArticleNormalizer, NormalizedArticle

logger = get_logger("scripts.audit_dataset")


def is_latin_char(char: str) -> bool:
    """Checks if character belongs to Latin script, ASCII, or common punctuation."""
    code = ord(char)
    if code < 128:  # Standard ASCII
        return True
    # Latin-1 Supplement, Latin Extended-A, Latin Extended-B
    if 0x00A0 <= code <= 0x024F:
        return True
    # Common mathematical / typographic punctuation
    if 0x2000 <= code <= 0x206F:
        return True
    return False


def run_audit(
    parquet_path: Path,
    output_dir: Path,
    export_normalized: bool = False,
    normalized_output_path: Optional[Path] = None,
    batch_size: int = 5000,
    sample_reservoir_size: int = 50000,
) -> Dict[str, Any]:
    """
    Performs an exhaustive audit on the Wikipedia Parquet dataset and generates
    both JSON and CSV quality reports.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input file not found: {parquet_path}")

    start_time = time.time()
    pq_file = pq.ParquetFile(str(parquet_path))
    schema = pq_file.schema_arrow
    num_rows = pq_file.metadata.num_rows
    num_row_groups = pq_file.num_row_groups
    file_size_mb = round(parquet_path.stat().st_size / (1024 * 1024), 2)

    logger.info(f"Starting audit on {parquet_path.name} ({num_rows:,} rows, {file_size_mb} MB, {num_row_groups} row groups)")

    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = ArticleNormalizer(min_chars=50)
    source_shard = normalizer.extract_shard_id(parquet_path)

    # 1. Raw Schema Detection
    raw_fields = {f.name: str(f.type) for f in schema}
    col_names = [f.name for f in schema]
    id_col = "id" if "id" in col_names else col_names[0]
    title_col = "title" if "title" in col_names else "title"
    url_col = "url" if "url" in col_names else "url"
    text_col = "text" if "text" in col_names else "text"

    # Regex patterns for markup & malformed text detection
    rx_html = re.compile(r"<(?:div|span|p|a|table|tr|td|th|ul|ol|li|b|i|strong|em|br|hr)\b[^>]*>", re.IGNORECASE)
    rx_templates = re.compile(r"\{\{.*?\}\}", re.DOTALL)
    rx_wikitables = re.compile(r"\{\|.*?\|\}", re.DOTALL)
    rx_headings = re.compile(r"^={2,5}\s*.*?\s*={2,5}", re.MULTILINE)
    rx_wikilinks = re.compile(r"\[\[.*?\]\]")
    rx_refs = re.compile(r"<ref\b[^>]*>", re.IGNORECASE)
    rx_control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    rx_url_valid = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)

    # Accumulators
    seen_ids: Set[Any] = set()
    duplicate_ids: List[Any] = []

    seen_title_hashes: Dict[int, int] = defaultdict(int)
    seen_text_hashes: Dict[str, int] = defaultdict(int)
    seen_prefix_hashes: Dict[str, int] = defaultdict(int)  # First 200 chars normalized

    empty_text_count = 0
    whitespace_only_text_count = 0
    short_lt_50_chars = 0
    short_lt_100_chars = 0
    short_lt_250_chars = 0
    tokens_lt_10 = 0
    tokens_lt_25 = 0

    missing_title_count = 0
    empty_title_count = 0
    missing_url_count = 0
    empty_url_count = 0
    malformed_url_count = 0

    char_lengths: List[int] = []
    token_lengths: List[int] = []
    title_lengths: List[int] = []

    markup_stats = {
        "articles_with_html": 0,
        "articles_with_templates": 0,
        "articles_with_wikitables": 0,
        "articles_with_headings": 0,
        "articles_with_wikilinks": 0,
        "articles_with_refs": 0,
    }

    text_integrity = {
        "articles_with_unicode_replacement": 0,
        "articles_with_control_chars": 0,
        "predominantly_non_latin_articles": 0,  # >50% non-Latin characters
        "low_alphabetic_content_articles": 0,   # <30% letter characters
    }

    # Optional normalized output writer
    norm_writer = None
    if export_normalized:
        norm_output_path = normalized_output_path or (output_dir / "wikipedia_en_pilot_normalized.parquet")
        norm_schema = pa.schema([
            ("doc_id", pa.int64()),
            ("title", pa.string()),
            ("url", pa.string()),
            ("text", pa.string()),
            ("source_shard", pa.int32()),
        ])
        norm_writer = pq.ParquetWriter(str(norm_output_path), schema=norm_schema, compression="snappy")
        logger.info(f"Initialized normalized exporter -> {norm_output_path}")

    # Streaming Batch Profiling
    processed_count = 0
    reservoir_step = max(1, num_rows // sample_reservoir_size) if num_rows > 0 else 1

    with tqdm(total=num_rows, desc="Auditing dataset", unit="docs") as pbar:
        for batch in pq_file.iter_batches(batch_size=batch_size):
            b_len = batch.num_rows
            ids = batch.column(id_col).to_pylist() if id_col in col_names else list(range(processed_count + 1, processed_count + b_len + 1))
            titles = batch.column(title_col).to_pylist() if title_col in col_names else [""] * b_len
            urls = batch.column(url_col).to_pylist() if url_col in col_names else [""] * b_len
            texts = batch.column(text_col).to_pylist() if text_col in col_names else [""] * b_len

            norm_batch_records = []

            for idx in range(b_len):
                doc_idx = processed_count + idx + 1
                raw_id = ids[idx]
                raw_title = titles[idx]
                raw_url = urls[idx]
                raw_text = texts[idx]

                # 1. ID Audit
                if raw_id is None:
                    duplicate_ids.append("NULL")
                elif raw_id in seen_ids:
                    duplicate_ids.append(raw_id)
                else:
                    seen_ids.add(raw_id)

                # 2. Title Audit
                if raw_title is None:
                    missing_title_count += 1
                    t_str = ""
                else:
                    t_str = str(raw_title).strip()
                    if not t_str:
                        empty_title_count += 1
                title_lengths.append(len(t_str))

                if t_str:
                    t_hash = hash(t_str.lower())
                    seen_title_hashes[t_hash] += 1

                # 3. URL Audit
                if raw_url is None:
                    missing_url_count += 1
                    u_str = ""
                else:
                    u_str = str(raw_url).strip()
                    if not u_str:
                        empty_url_count += 1
                    elif not rx_url_valid.match(u_str):
                        malformed_url_count += 1

                # 4. Text Length & Stubs Audit
                if raw_text is None:
                    empty_text_count += 1
                    txt_str = ""
                    c_len = 0
                    t_len = 0
                else:
                    txt_raw = str(raw_text)
                    txt_str = txt_raw.strip()
                    c_len = len(txt_str)
                    t_len = max(1, int(len(txt_str.split()) * 1.33)) if c_len > 0 else 0

                    if len(txt_raw) > 0 and len(txt_str) == 0:
                        whitespace_only_text_count += 1
                    elif c_len == 0:
                        empty_text_count += 1

                if c_len < 50:
                    short_lt_50_chars += 1
                if c_len < 100:
                    short_lt_100_chars += 1
                if c_len < 250:
                    short_lt_250_chars += 1

                if t_len < 10:
                    tokens_lt_10 += 1
                if t_len < 25:
                    tokens_lt_25 += 1

                # Reservoir sampling for distribution
                if (processed_count + idx) % reservoir_step == 0 and len(char_lengths) < sample_reservoir_size:
                    char_lengths.append(c_len)
                    token_lengths.append(t_len)

                # 5. Duplicate Content Hashes
                if c_len >= 50:
                    txt_md5 = hashlib.md5(txt_str.encode("utf-8")).hexdigest()
                    seen_text_hashes[txt_md5] += 1

                    # Normalized prefix hash (first 250 chars lowercased alphanum)
                    prefix_clean = re.sub(r"[^a-z0-9]", "", txt_str[:250].lower())
                    if len(prefix_clean) >= 40:
                        prefix_hash = hashlib.md5(prefix_clean.encode("utf-8")).hexdigest()
                        seen_prefix_hashes[prefix_hash] += 1

                # 6. Markup Detection
                if c_len > 0:
                    if rx_html.search(txt_str):
                        markup_stats["articles_with_html"] += 1
                    if rx_templates.search(txt_str):
                        markup_stats["articles_with_templates"] += 1
                    if rx_wikitables.search(txt_str):
                        markup_stats["articles_with_wikitables"] += 1
                    if rx_headings.search(txt_str):
                        markup_stats["articles_with_headings"] += 1
                    if rx_wikilinks.search(txt_str):
                        markup_stats["articles_with_wikilinks"] += 1
                    if rx_refs.search(txt_str):
                        markup_stats["articles_with_refs"] += 1

                    # 7. Non-English & Malformed Text Diagnostics
                    if "\ufffd" in txt_str:
                        text_integrity["articles_with_unicode_replacement"] += 1
                    if rx_control_chars.search(txt_str):
                        text_integrity["articles_with_control_chars"] += 1

                    # Script analysis on sample (first 1000 chars)
                    sample_slice = txt_str[:1000]
                    total_chars_in_sample = len(sample_slice)
                    if total_chars_in_sample > 0:
                        latin_chars = sum(1 for ch in sample_slice if is_latin_char(ch))
                        letter_chars = sum(1 for ch in sample_slice if ch.isalpha())

                        if (latin_chars / total_chars_in_sample) < 0.5:
                            text_integrity["predominantly_non_latin_articles"] += 1
                        if (letter_chars / total_chars_in_sample) < 0.30:
                            text_integrity["low_alphabetic_content_articles"] += 1

                # 8. Schema Normalization Accumulator
                if export_normalized:
                    raw_dict = {"id": raw_id, "title": raw_title, "url": raw_url, "text": raw_text}
                    norm_obj, _ = normalizer.normalize_record(raw_dict, source_shard=source_shard, fallback_index=doc_idx)
                    norm_batch_records.append(norm_obj)

            # Write normalized batch
            if export_normalized and norm_writer is not None and norm_batch_records:
                b_table = pa.Table.from_arrays(
                    [
                        pa.array([n.doc_id for n in norm_batch_records], type=pa.int64()),
                        pa.array([n.title for n in norm_batch_records], type=pa.string()),
                        pa.array([n.url for n in norm_batch_records], type=pa.string()),
                        pa.array([n.text for n in norm_batch_records], type=pa.string()),
                        pa.array([n.source_shard for n in norm_batch_records], type=pa.int32()),
                    ],
                    names=["doc_id", "title", "url", "text", "source_shard"],
                )
                norm_writer.write_table(b_table)

            processed_count += b_len
            pbar.update(b_len)

    if norm_writer is not None:
        norm_writer.close()
        logger.info(f"Successfully exported normalized Parquet to {norm_output_path}")

    # Compute Statistical Summaries
    def compute_distribution(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {"min": 0, "q25": 0, "median": 0, "q75": 0, "mean": 0, "std": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
        np_vals = np.array(vals)
        return {
            "min": int(np.min(np_vals)),
            "q25": int(np.percentile(np_vals, 25)),
            "median": int(np.percentile(np_vals, 50)),
            "q75": int(np.percentile(np_vals, 75)),
            "mean": round(float(np.mean(np_vals)), 1),
            "std": round(float(np.std(np_vals)), 1),
            "p90": int(np.percentile(np_vals, 90)),
            "p95": int(np.percentile(np_vals, 95)),
            "p99": int(np.percentile(np_vals, 99)),
            "max": int(np.max(np_vals)),
        }

    char_dist = compute_distribution(char_lengths)
    token_dist = compute_distribution(token_lengths)
    title_dist = compute_distribution(title_lengths)

    # Calculate Duplicates
    duplicate_title_count = sum(cnt - 1 for cnt in seen_title_hashes.values() if cnt > 1)
    exact_duplicate_text_count = sum(cnt - 1 for cnt in seen_text_hashes.values() if cnt > 1)
    near_duplicate_text_count = sum(cnt - 1 for cnt in seen_prefix_hashes.values() if cnt > 1)

    elapsed = time.time() - start_time

    # Build Complete Audit Report Dictionary
    audit_report = {
        "dataset_audit_meta": {
            "file_name": parquet_path.name,
            "file_path": str(parquet_path),
            "file_size_mb": file_size_mb,
            "total_documents": num_rows,
            "source_shard_detected": source_shard,
            "audit_duration_seconds": round(elapsed, 2),
            "audit_speed_docs_per_sec": round(num_rows / max(0.01, elapsed), 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "raw_schema_detected": raw_fields,
        "normalized_schema_target": {
            "doc_id": "int64 (64-bit integer identifier)",
            "title": "string (cleaned article title)",
            "url": "string (valid canonical URL)",
            "text": "string (article body text)",
            "source_shard": "int32 (source partition identifier)",
        },
        "document_id_uniqueness": {
            "unique_ids": len(seen_ids),
            "duplicate_ids_count": len(duplicate_ids),
            "is_unique": len(duplicate_ids) == 0,
            "sample_duplicate_ids": duplicate_ids[:5],
        },
        "empty_and_short_documents": {
            "empty_text_count": empty_text_count,
            "empty_text_pct": round((empty_text_count / max(1, num_rows)) * 100, 3),
            "whitespace_only_text_count": whitespace_only_text_count,
            "chars_lt_50_count": short_lt_50_chars,
            "chars_lt_50_pct": round((short_lt_50_chars / max(1, num_rows)) * 100, 2),
            "chars_lt_100_count": short_lt_100_chars,
            "chars_lt_100_pct": round((short_lt_100_chars / max(1, num_rows)) * 100, 2),
            "chars_lt_250_count": short_lt_250_chars,
            "chars_lt_250_pct": round((short_lt_250_chars / max(1, num_rows)) * 100, 2),
            "tokens_lt_10_count": tokens_lt_10,
            "tokens_lt_10_pct": round((tokens_lt_10 / max(1, num_rows)) * 100, 2),
            "tokens_lt_25_count": tokens_lt_25,
            "tokens_lt_25_pct": round((tokens_lt_25 / max(1, num_rows)) * 100, 2),
        },
        "title_and_url_availability": {
            "missing_titles": missing_title_count,
            "empty_titles": empty_title_count,
            "title_availability_pct": round(((num_rows - (missing_title_count + empty_title_count)) / max(1, num_rows)) * 100, 2),
            "missing_urls": missing_url_count,
            "empty_urls": empty_url_count,
            "malformed_urls": malformed_url_count,
            "url_availability_pct": round(((num_rows - (missing_url_count + empty_url_count + malformed_url_count)) / max(1, num_rows)) * 100, 2),
            "title_length_stats": title_dist,
        },
        "text_length_distributions": {
            "character_distribution": char_dist,
            "token_estimate_distribution": token_dist,
        },
        "duplicate_detection": {
            "duplicate_titles_count": duplicate_title_count,
            "duplicate_titles_pct": round((duplicate_title_count / max(1, num_rows)) * 100, 3),
            "exact_duplicate_text_count": exact_duplicate_text_count,
            "exact_duplicate_text_pct": round((exact_duplicate_text_count / max(1, num_rows)) * 100, 3),
            "near_duplicate_text_count": near_duplicate_text_count,
            "near_duplicate_text_pct": round((near_duplicate_text_count / max(1, num_rows)) * 100, 3),
        },
        "markup_and_html_contamination": {
            "counts": markup_stats,
            "percentages": {
                k: round((v / max(1, num_rows)) * 100, 2)
                for k, v in markup_stats.items()
            },
        },
        "language_and_character_integrity": {
            "counts": text_integrity,
            "percentages": {
                k: round((v / max(1, num_rows)) * 100, 3)
                for k, v in text_integrity.items()
            },
            "clean_english_ratio_pct": round(
                ((num_rows - (text_integrity["predominantly_non_latin_articles"] + text_integrity["low_alphabetic_content_articles"])) / max(1, num_rows)) * 100,
                2
            ),
        },
    }

    # Save JSON Report
    json_path = output_dir / "audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    logger.info(f"Saved complete JSON audit report -> {json_path}")

    # Save CSV Summary Report
    csv_path = output_dir / "audit_summary.csv"
    _save_csv_summary(audit_report, csv_path)
    logger.info(f"Saved CSV audit summary -> {csv_path}")

    # Save Text Distribution Percentiles CSV
    dist_csv_path = output_dir / "text_length_percentiles.csv"
    _save_distribution_csv(char_dist, token_dist, dist_csv_path)
    logger.info(f"Saved text distribution CSV -> {dist_csv_path}")

    return audit_report


def _save_csv_summary(report: Dict[str, Any], csv_path: Path):
    """Writes key scalar metrics into a clean, flat CSV table."""
    m = report["dataset_audit_meta"]
    u = report["document_id_uniqueness"]
    s = report["empty_and_short_documents"]
    t = report["title_and_url_availability"]
    d = report["duplicate_detection"]
    c = report["text_length_distributions"]["character_distribution"]
    tok = report["text_length_distributions"]["token_estimate_distribution"]
    mk = report["markup_and_html_contamination"]["percentages"]
    lg = report["language_and_character_integrity"]["percentages"]

    rows = [
        ("Metric Category", "Metric Name", "Value", "Unit / Notes"),
        ("Metadata", "File Name", m["file_name"], "-"),
        ("Metadata", "Total Documents", str(m["total_documents"]), "documents"),
        ("Metadata", "File Size MB", str(m["file_size_mb"]), "MB"),
        ("Metadata", "Source Shard", str(m["source_shard_detected"]), "shard_id"),
        ("Identity", "Unique Document IDs", str(u["unique_ids"]), "IDs"),
        ("Identity", "Duplicate Document IDs", str(u["duplicate_ids_count"]), "IDs"),
        ("Identity", "ID Uniqueness Verified", str(u["is_unique"]), "boolean"),
        ("Completeness", "Empty Texts", str(s["empty_text_count"]), f"{s['empty_text_pct']}%"),
        ("Completeness", "Whitespace Only Texts", str(s["whitespace_only_text_count"]), "docs"),
        ("Completeness", "Short Docs (< 50 chars)", str(s["chars_lt_50_count"]), f"{s['chars_lt_50_pct']}%"),
        ("Completeness", "Short Docs (< 100 chars)", str(s["chars_lt_100_count"]), f"{s['chars_lt_100_pct']}%"),
        ("Completeness", "Short Docs (< 250 chars)", str(s["chars_lt_250_count"]), f"{s['chars_lt_250_pct']}%"),
        ("Completeness", "Tokens < 10 (Stubs)", str(s["tokens_lt_10_count"]), f"{s['tokens_lt_10_pct']}%"),
        ("Completeness", "Title Availability", f"{t['title_availability_pct']}%", f"{t['missing_titles']} missing"),
        ("Completeness", "URL Availability", f"{t['url_availability_pct']}%", f"{t['missing_urls']} missing, {t['malformed_urls']} malformed"),
        ("Distribution", "Text Chars Min", str(c["min"]), "characters"),
        ("Distribution", "Text Chars Median (Q2)", str(c["median"]), "characters"),
        ("Distribution", "Text Chars Mean", str(c["mean"]), "characters"),
        ("Distribution", "Text Chars p95", str(c["p95"]), "characters"),
        ("Distribution", "Text Chars Max", str(c["max"]), "characters"),
        ("Distribution", "Tokens Median (Q2)", str(tok["median"]), "tokens (est.)"),
        ("Distribution", "Tokens Mean", str(tok["mean"]), "tokens (est.)"),
        ("Distribution", "Tokens p95", str(tok["p95"]), "tokens (est.)"),
        ("Duplicates", "Duplicate Titles", str(d["duplicate_titles_count"]), f"{d['duplicate_titles_pct']}%"),
        ("Duplicates", "Exact Duplicate Texts", str(d["exact_duplicate_text_count"]), f"{d['exact_duplicate_text_pct']}%"),
        ("Duplicates", "Near Duplicate Texts", str(d["near_duplicate_text_count"]), f"{d['near_duplicate_text_pct']}%"),
        ("Markup", "Articles with HTML", f"{mk['articles_with_html']}%", "residual HTML tags"),
        ("Markup", "Articles with Templates", f"{mk['articles_with_templates']}%", "wikitext templates"),
        ("Markup", "Articles with Tables", f"{mk['articles_with_wikitables']}%", "wikitables"),
        ("Markup", "Articles with Headings", f"{mk['articles_with_headings']}%", "wiki headings"),
        ("Markup", "Articles with Refs", f"{mk['articles_with_refs']}%", "citation tags"),
        ("Integrity", "Unicode Replacement (U+FFFD)", f"{lg['articles_with_unicode_replacement']}%", "encoding errors"),
        ("Integrity", "Control Characters", f"{lg['articles_with_control_chars']}%", "unprintable control chars"),
        ("Integrity", "Predominantly Non-Latin", f"{lg['predominantly_non_latin_articles']}%", "non-English / non-Latin"),
        ("Integrity", "Low Alphabetic Content", f"{lg['low_alphabetic_content_articles']}%", "noise / lists (<30% letters)"),
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _save_distribution_csv(char_dist: Dict[str, Any], token_dist: Dict[str, Any], csv_path: Path):
    """Writes detailed text length percentiles into a dedicated CSV."""
    percentiles = ["min", "q25", "median", "q75", "mean", "std", "p90", "p95", "p99", "max"]
    rows = [("Percentile / Statistic", "Character Count", "Estimated Token Count")]
    for p in percentiles:
        rows.append((p.upper(), str(char_dist.get(p, 0)), str(token_dist.get(p, 0))))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def print_audit_summary_table(report: Dict[str, Any]):
    """Renders formatted audit tables to stdout."""
    m = report["dataset_audit_meta"]
    u = report["document_id_uniqueness"]
    s = report["empty_and_short_documents"]
    t = report["title_and_url_availability"]
    d = report["duplicate_detection"]
    c = report["text_length_distributions"]["character_distribution"]
    tok = report["text_length_distributions"]["token_estimate_distribution"]
    mk = report["markup_and_html_contamination"]["percentages"]
    lg = report["language_and_character_integrity"]["percentages"]

    print("\n" + "=" * 80)
    print("                WIKIPEDIA PILOT DATASET COMPREHENSIVE AUDIT")
    print("=" * 80)
    print(f"File:               {m['file_name']} ({m['file_size_mb']} MB)")
    print(f"Total Documents:    {m['total_documents']:,} articles | Source Shard: {m['source_shard_detected']}")
    print(f"Audit Speed:        {m['audit_speed_docs_per_sec']:,} docs/sec in {m['audit_duration_seconds']}s")
    print("-" * 80)

    print("1. IDENTITY & INTEGRITY:")
    id_status = "✓ Clean (0 duplicates)" if u["is_unique"] else f"⚠ {u['duplicate_ids_count']} duplicates!"
    print(f"   • Document IDs Unique : {id_status} ({u['unique_ids']:,} unique)")
    print(f"   • Title Availability  : {t['title_availability_pct']}% ({t['missing_titles']} missing, {t['empty_titles']} empty)")
    print(f"   • URL Availability    : {t['url_availability_pct']}% ({t['missing_urls']} missing, {t['malformed_urls']} malformed)")

    print("\n2. EMPTY & STUB DOCUMENTS:")
    print(f"   • Empty Text (0 chars): {s['empty_text_count']} ({s['empty_text_pct']}%)")
    print(f"   • Stubs (< 50 chars)  : {s['chars_lt_50_count']} ({s['chars_lt_50_pct']}%)")
    print(f"   • Stubs (< 100 chars) : {s['chars_lt_100_count']} ({s['chars_lt_100_pct']}%)")
    print(f"   • Tokens < 10         : {s['tokens_lt_10_count']} ({s['tokens_lt_10_pct']}%)")

    print("\n3. DUPLICATE DETECTION:")
    print(f"   • Exact Duplicate Titles: {d['duplicate_titles_count']:,} ({d['duplicate_titles_pct']}%)")
    print(f"   • Exact Duplicate Texts : {d['exact_duplicate_text_count']:,} ({d['exact_duplicate_text_pct']}%)")
    print(f"   • Near Duplicate Texts  : {d['near_duplicate_text_count']:,} ({d['near_duplicate_text_pct']}%)")

    print("\n4. TEXT LENGTH PERCENTILES:")
    print(f"   {'Stat':<8} | {'Characters':<15} | {'Estimated Tokens':<15}")
    print(f"   {'-'*8}-|-{'-'*15}-|-{'-'*15}")
    for p in ["min", "q25", "median", "q75", "mean", "p90", "p95", "p99", "max"]:
        c_val = c.get(p, 0)
        t_val = tok.get(p, 0)
        c_str = f"{c_val:,.1f}" if isinstance(c_val, float) else f"{c_val:,}"
        t_str = f"{t_val:,.1f}" if isinstance(t_val, float) else f"{t_val:,}"
        print(f"   {p.upper():<8} | {c_str:<15} | {t_str:<15}")

    print("\n5. MARKUP & HTML CONTAMINATION:")
    for k, pct in mk.items():
        name = k.replace("articles_with_", "").capitalize()
        print(f"   • Articles with {name:<12} : {pct:>5.2f}%")

    print("\n6. LANGUAGE & ENCODING INTEGRITY:")
    print(f"   • Unicode Replacement (\\ufffd) : {lg['articles_with_unicode_replacement']:>5.3f}%")
    print(f"   • Unprintable Control Chars   : {lg['articles_with_control_chars']:>5.3f}%")
    print(f"   • Predominantly Non-Latin     : {lg['predominantly_non_latin_articles']:>5.3f}%")
    print(f"   • Low Alphabetic (<30% letters): {lg['low_alphabetic_content_articles']:>5.3f}%")
    print(f"   • Clean English Ratio         : {report['language_and_character_integrity']['clean_english_ratio_pct']}%")

    print("\n7. NORMALIZED SCHEMA SPECIFICATION:")
    for col, spec in report["normalized_schema_target"].items():
        print(f"   • {col:<14} -> {spec}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit and profile Wikipedia Parquet dataset")
    parser.add_argument("--input", type=str, default="data/raw/wikipedia_en_pilot_50k.parquet", help="Path to input Parquet file")
    parser.add_argument("--output-dir", type=str, default="data/audit", help="Directory to store audit reports")
    parser.add_argument("--export-normalized", action="store_true", help="Export clean Parquet in normalized schema")
    parser.add_argument("--normalized-output", type=str, default=None, help="Custom path for normalized Parquet output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    norm_out = Path(args.normalized_output) if args.normalized_output else None

    report = run_audit(
        parquet_path=input_path,
        output_dir=output_dir,
        export_normalized=args.export_normalized,
        normalized_output_path=norm_out,
    )
    print_audit_summary_table(report)


if __name__ == "__main__":
    main()
