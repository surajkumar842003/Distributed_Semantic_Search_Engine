"""Dataset Inspection Module: Programmatic Schema, Quality, and Markup Profiling."""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from src.common.logging import get_logger

logger = get_logger("scripts.inspect_dataset")


def detect_nested_fields(schema: pa.Schema) -> Dict[str, str]:
    """Detects any complex or nested data structures (lists, structs, maps)."""
    nested = {}
    for field in schema:
        t = field.type
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            nested[field.name] = f"List<{t.value_type}>"
        elif pa.types.is_struct(t):
            nested[field.name] = f"Struct<{len(t)} fields>"
        elif pa.types.is_map(t):
            nested[field.name] = f"Map<{t.key_type}, {t.item_type}>"
        elif pa.types.is_nested(t):
            nested[field.name] = str(t)
    return nested


def inspect_parquet_dataset(parquet_path: Path, sample_reservoir_size: int = 50000, batch_size: int = 5000) -> Dict[str, Any]:
    """
    Performs streaming inspection of Parquet dataset without loading all rows into RAM.
    Detects schema, nulls, duplicates, text distributions, and markup presence.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")

    start_time = time.time()
    pq_file = pq.ParquetFile(str(parquet_path))
    schema = pq_file.schema_arrow
    num_rows = pq_file.metadata.num_rows
    num_row_groups = pq_file.num_row_groups
    file_size_bytes = parquet_path.stat().st_size

    # Calculate compressed vs uncompressed sizes from row group metadata
    total_uncompressed_bytes = 0
    total_compressed_bytes = 0
    for rg_idx in range(num_row_groups):
        rg = pq_file.metadata.row_group(rg_idx)
        total_uncompressed_bytes += rg.total_byte_size
        total_compressed_bytes += sum(rg.column(col_idx).total_compressed_size for col_idx in range(rg.num_columns))

    compression_ratio = (
        total_uncompressed_bytes / max(1, total_compressed_bytes)
        if total_compressed_bytes > 0
        else 1.0
    )

    # 1. Programmatic Schema Detection
    columns_info = {}
    for field in schema:
        columns_info[field.name] = {
            "type": str(field.type),
            "nullable": field.nullable,
            "is_nested": pa.types.is_nested(field.type),
        }
    nested_fields = detect_nested_fields(schema)

    # Select ID column and Text column programmatically
    col_names = [f.name for f in schema]
    id_col = "id" if "id" in col_names else col_names[0]
    text_col = "text" if "text" in col_names else ("content" if "content" in col_names else None)
    if text_col is None:
        # Fallback to first string column
        for f in schema:
            if pa.types.is_string(f.type) or pa.types.is_large_string(f.type):
                text_col = f.name
                break

    # 2. Streaming Counters & Statistical Accumulators
    null_counts = {name: 0 for name in col_names}
    seen_ids: Set[Any] = set()
    duplicate_id_count = 0
    
    # Text length reservoirs (fixed max size for memory safety)
    char_lengths: List[int] = []
    token_lengths: List[int] = []
    total_char_count = 0
    total_token_count = 0

    # Markup detection regexes
    rx_templates = re.compile(r"\{\{.*?\}\}", re.DOTALL)
    rx_tables = re.compile(r"\{\|.*?\|\}", re.DOTALL)
    rx_headings = re.compile(r"^={2,5}\s*.*?\s*={2,5}", re.MULTILINE)
    rx_wikilinks = re.compile(r"\[\[.*?\]\]")
    rx_refs = re.compile(r"<ref[^>]*>", re.IGNORECASE)
    rx_html = re.compile(r"<(div|span|table|tr|td|p|b|i)[^>]*>", re.IGNORECASE)

    markup_counts = {
        "articles_with_templates": 0,
        "articles_with_tables": 0,
        "articles_with_headings": 0,
        "articles_with_wikilinks": 0,
        "articles_with_refs": 0,
        "articles_with_html": 0,
    }

    processed_rows = 0
    reservoir_step = max(1, num_rows // sample_reservoir_size) if num_rows > 0 else 1

    logger.info(f"Streaming through {num_rows:,} rows in batches of {batch_size:,}...")
    with tqdm(total=num_rows, desc="Inspecting dataset", unit="rows") as pbar:
        for batch in pq_file.iter_batches(batch_size=batch_size):
            batch_len = batch.num_rows

            # Null counts
            for col_name in col_names:
                col_array = batch.column(col_name)
                null_counts[col_name] += col_array.null_count

            # Duplicate ID checking
            if id_col in col_names:
                id_list = batch.column(id_col).to_pylist()
                for item_id in id_list:
                    if item_id in seen_ids:
                        duplicate_id_count += 1
                    else:
                        seen_ids.add(item_id)

            # Text profiling & markup detection
            if text_col in col_names:
                text_list = batch.column(text_col).to_pylist()
                for i, raw_text in enumerate(text_list):
                    if raw_text is None:
                        continue
                    c_len = len(raw_text)
                    t_len = max(1, int(len(raw_text.split()) * 1.33))
                    total_char_count += c_len
                    total_token_count += t_len

                    # Reservoir sample for percentile computation
                    if (processed_rows + i) % reservoir_step == 0 and len(char_lengths) < sample_reservoir_size:
                        char_lengths.append(c_len)
                        token_lengths.append(t_len)

                    # Markup tests
                    if rx_templates.search(raw_text):
                        markup_counts["articles_with_templates"] += 1
                    if rx_tables.search(raw_text):
                        markup_counts["articles_with_tables"] += 1
                    if rx_headings.search(raw_text):
                        markup_counts["articles_with_headings"] += 1
                    if rx_wikilinks.search(raw_text):
                        markup_counts["articles_with_wikilinks"] += 1
                    if rx_refs.search(raw_text):
                        markup_counts["articles_with_refs"] += 1
                    if rx_html.search(raw_text):
                        markup_counts["articles_with_html"] += 1

            processed_rows += batch_len
            pbar.update(batch_len)

    # 3. Compute Length Statistics
    def compute_stats(arr: List[int], total_sum: int, count: int) -> Dict[str, float]:
        if not arr or count == 0:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "p99": 0}
        np_arr = np.array(arr)
        return {
            "min": int(np.min(np_arr)),
            "max": int(np.max(np_arr)),
            "mean": round(total_sum / count, 1),
            "median": int(np.percentile(np_arr, 50)),
            "p90": int(np.percentile(np_arr, 90)),
            "p95": int(np.percentile(np_arr, 95)),
            "p99": int(np.percentile(np_arr, 99)),
        }

    valid_text_count = num_rows - null_counts.get(text_col, 0)
    char_stats = compute_stats(char_lengths, total_char_count, valid_text_count)
    token_stats = compute_stats(token_lengths, total_token_count, valid_text_count)

    # Markup percentages
    markup_percentages = {
        k: round((v / max(1, valid_text_count)) * 100, 2)
        for k, v in markup_counts.items()
    }

    # Format missing values
    missing_values = {
        col: {
            "null_count": null_counts[col],
            "null_percentage": round((null_counts[col] / max(1, num_rows)) * 100, 2),
        }
        for col in col_names
    }

    elapsed = time.time() - start_time

    # Determine markup verdict
    if markup_percentages.get("articles_with_headings", 0) > 30 or markup_percentages.get("articles_with_templates", 0) > 30:
        markup_verdict = "Raw Wikitext (templates, infoboxes, and heading markup present; requires wikitext cleaner)"
    else:
        markup_verdict = "Pre-cleaned Text (no raw Wikitext markup detected)"

    report = {
        "file_info": {
            "path": str(parquet_path),
            "filename": parquet_path.name,
            "size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "num_rows": num_rows,
            "num_row_groups": num_row_groups,
            "compression_ratio": round(compression_ratio, 2),
            "inspection_duration_sec": round(elapsed, 2),
        },
        "schema": columns_info,
        "nested_fields": nested_fields,
        "missing_values": missing_values,
        "duplicate_detection": {
            "id_column": id_col,
            "unique_ids_count": len(seen_ids),
            "duplicate_ids_count": duplicate_id_count,
            "has_duplicates": duplicate_id_count > 0,
        },
        "text_length_distributions": {
            "text_column": text_col,
            "character_length": char_stats,
            "token_count_estimate": token_stats,
        },
        "markup_presence": {
            "verdict": markup_verdict,
            "percentages": markup_percentages,
            "raw_counts": markup_counts,
        },
    }

    return report


def print_formatted_report(report: Dict[str, Any]):
    """Renders a clean human-readable summary of the inspection report."""
    f = report["file_info"]
    dup = report["duplicate_detection"]
    c_stat = report["text_length_distributions"]["character_length"]
    t_stat = report["text_length_distributions"]["token_count_estimate"]
    mp = report["markup_presence"]

    print("\n" + "=" * 78)
    print("           WIKIPEDIA DATASET INSPECTION & PROFILING REPORT")
    print("=" * 78)
    print(f"File:                   {f['filename']} ({f['size_mb']} MB)")
    print(f"Total Rows:             {f['num_rows']:,} articles across {f['num_row_groups']} row groups")
    print(f"Compression Ratio:      {f['compression_ratio']:.2f}x")
    print(f"Inspection Speed:       {f['num_rows'] / max(0.01, f['inspection_duration_sec']):,.1f} rows/sec ({f['inspection_duration_sec']}s)")
    print("-" * 78)

    print("1. DETECTED SCHEMA & TYPES:")
    for col, info in report["schema"].items():
        nullable_str = "nullable" if info["nullable"] else "not null"
        print(f"   • {col:<18} : {info['type']:<22} ({nullable_str})")

    if report["nested_fields"]:
        print(f"\n   Nested Fields Detected: {report['nested_fields']}")
    else:
        print("\n   Nested Fields:          None (Flat relational schema)")

    print("\n2. MISSING VALUES & INTEGRITY:")
    for col, m in report["missing_values"].items():
        status = "✓ Clean" if m["null_count"] == 0 else f"⚠ {m['null_count']:,} NULLs ({m['null_percentage']}%)"
        print(f"   • {col:<18} : {status}")

    print(f"\n   ID Uniqueness ({dup['id_column']}):    {dup['unique_ids_count']:,} unique IDs, {dup['duplicate_ids_count']} duplicates")
    if dup["has_duplicates"]:
        print("   ⚠ WARNING: Duplicate IDs detected in dataset!")
    else:
        print("   ✓ All IDs are strictly unique (Primary Key constraint verified)")

    print("-" * 78)
    print("3. TEXT LENGTH DISTRIBUTIONS (Article Body):")
    print(f"   {'Metric':<10} | {'Characters':<15} | {'Tokens (Est.)':<15}")
    print(f"   {'-'*10}-|-{'-'*15}-|-{'-'*15}")
    print(f"   {'Min':<10} | {c_stat['min']:<15,d} | {t_stat['min']:<15,d}")
    print(f"   {'Median':<10} | {c_stat['median']:<15,d} | {t_stat['median']:<15,d}")
    print(f"   {'Mean':<10} | {c_stat['mean']:<15,.1f} | {t_stat['mean']:<15,.1f}")
    print(f"   {'p90':<10} | {c_stat['p90']:<15,d} | {t_stat['p90']:<15,d}")
    print(f"   {'p95':<10} | {c_stat['p95']:<15,d} | {t_stat['p95']:<15,d}")
    print(f"   {'p99':<10} | {c_stat['p99']:<15,d} | {t_stat['p99']:<15,d}")
    print(f"   {'Max':<10} | {c_stat['max']:<15,d} | {t_stat['max']:<15,d}")

    print("-" * 78)
    print("4. MARKUP & FORMAT VERDICT:")
    print(f"   Verdict: {mp['verdict']}")
    for k, pct in mp["percentages"].items():
        name = k.replace("articles_with_", "").capitalize()
        print(f"   • Articles with {name:<12} : {pct:>6.2f}%")

    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect and profile Wikipedia Parquet dataset")
    parser.add_argument("--input", type=str, default="data/raw/wikipedia_en_pilot_50k.parquet", help="Path to Parquet file")
    parser.add_argument("--report", type=str, default="data/raw/dataset_inspection_report.json", help="Path to write JSON report")
    parser.add_argument("--sample-size", type=int, default=50000, help="Reservoir sample size for quantile estimation")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Check if fallback or sample exists
        fallback = Path("data/raw/wikipedia_sample.parquet")
        if fallback.exists():
            logger.info(f"Target {input_path} not found. Inspecting available sample at {fallback}")
            input_path = fallback
        else:
            logger.error(f"Cannot find dataset file: {input_path}")
            sys.exit(1)

    report = inspect_parquet_dataset(input_path, sample_reservoir_size=args.sample_size)
    print_formatted_report(report)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved JSON inspection report to {report_path}")


if __name__ == "__main__":
    main()

