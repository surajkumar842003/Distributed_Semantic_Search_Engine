"""Wikipedia Dataset Acquisition Module: Resumable Parquet Download and Pilot Slicing."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.error

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from src.config import load_yaml
from src.common.logging import get_logger

logger = get_logger("scripts.download_dataset")

# Default Hugging Face direct Parquet shard URL for English Wikipedia
DEFAULT_SHARD_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00000-of-00041.parquet"
)
HF_DATASET_API = "https://huggingface.co/api/datasets/wikimedia/wikipedia"


def inspect_remote_sources(dataset_id: str = "wikimedia/wikipedia") -> Dict[str, Any]:
    """Inspects available remote configs and metadata from Hugging Face Hub."""
    logger.info(f"Querying Hugging Face API for dataset: {dataset_id}")
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "DistributedWikipediaRAG/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            info = {
                "id": data.get("id"),
                "author": data.get("author"),
                "sha": data.get("sha"),
                "last_modified": data.get("lastModified"),
                "tags": data.get("tags", []),
                "description": data.get("description", "")[:200] + "...",
            }
            logger.info(f"Found remote dataset commit: {info.get('sha')[:10]} (last modified: {info.get('last_modified')})")
            return info
    except Exception as e:
        logger.warning(f"Could not reach Hugging Face API ({e}). Proceeding with direct Parquet acquisition.")
        return {"id": dataset_id, "error": str(e)}


def compute_sha256(file_path: Path, buffer_size: int = 65536) -> str:
    """Computes SHA-256 hash of a local file in memory-efficient chunks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file_resumable(
    url: str,
    target_path: Path,
    chunk_size: int = 10 * 1024 * 1024,  # 10 MB
    timeout: int = 60,
    max_retries: int = 5,
) -> Path:
    """
    Downloads a remote file with HTTP Range resumption.
    Saves to a temporary '.part' file until download is complete and verified.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = Path(str(target_path) + ".part")

    if target_path.exists():
        logger.info(f"File already exists: {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return target_path

    existing_bytes = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "DistributedWikipediaRAG/0.1.0"}

    # Query total size with HEAD request
    total_size = None
    try:
        head_req = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(head_req, timeout=timeout) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length:
                total_size = int(content_length)
    except Exception as e:
        logger.warning(f"HEAD request failed ({e}). Total size will be detected dynamically.")

    for attempt in range(1, max_retries + 1):
        try:
            req_headers = headers.copy()
            if existing_bytes > 0:
                req_headers["Range"] = f"bytes={existing_bytes}-"
                logger.info(f"Resuming download from byte offset {existing_bytes:,} (Attempt {attempt}/{max_retries})")
            else:
                logger.info(f"Starting fresh download: {url} (Attempt {attempt}/{max_retries})")

            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.status
                is_partial = (status == 206)

                if existing_bytes > 0 and not is_partial:
                    logger.warning("Server does not support HTTP Range requests. Restarting download from byte 0.")
                    existing_bytes = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing_bytes > 0 else "wb"

                expected_total = total_size or (
                    existing_bytes + int(response.headers.get("Content-Length", 0))
                )

                with open(part_path, mode) as out_f, tqdm(
                    total=expected_total,
                    initial=existing_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=target_path.name,
                ) as pbar:
                    while True:
                        buffer = response.read(chunk_size)
                        if not buffer:
                            break
                        out_f.write(buffer)
                        pbar.update(len(buffer))
                        existing_bytes += len(buffer)

            # Atomically rename .part to final destination
            part_path.rename(target_path)
            logger.info(f"Successfully downloaded {target_path.name} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
            return target_path

        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as err:
            logger.warning(f"Network error during download: {err}. Retrying in {attempt * 3}s...")
            time.sleep(attempt * 3)
            existing_bytes = part_path.stat().st_size if part_path.exists() else 0

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


def slice_pilot_subset(
    source_parquet: Path,
    output_parquet: Path,
    target_rows: int = 50000,
    batch_size: int = 2000,
) -> int:
    """
    Streams row batches from a Parquet shard and writes exactly target_rows
    into a dedicated pilot Parquet file without loading all rows into RAM.
    """
    logger.info(f"Slicing pilot subset of {target_rows:,} rows from {source_parquet.name} -> {output_parquet.name}")
    reader = pq.ParquetFile(str(source_parquet))
    total_available = reader.metadata.num_rows
    logger.info(f"Source shard contains {total_available:,} articles across {reader.num_row_groups} row groups")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    schema = reader.schema_arrow

    writer: Optional[pq.ParquetWriter] = None
    rows_written = 0

    try:
        with tqdm(total=min(target_rows, total_available), desc="Slicing pilot", unit="rows") as pbar:
            for batch in reader.iter_batches(batch_size=batch_size):
                if rows_written >= target_rows:
                    break

                needed = target_rows - rows_written
                if batch.num_rows > needed:
                    batch = batch.slice(0, needed)

                table_chunk = pa.Table.from_batches([batch], schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_parquet), schema=schema, compression="snappy")

                writer.write_table(table_chunk)
                rows_written += batch.num_rows
                pbar.update(batch.num_rows)
    finally:
        if writer is not None:
            writer.close()

    logger.info(f"Finished slicing: wrote {rows_written:,} rows to {output_parquet} ({output_parquet.stat().st_size / (1024*1024):.2f} MB)")
    return rows_written


def save_provenance_metadata(
    output_metadata_path: Path,
    pilot_parquet_path: Path,
    source_url: str,
    dataset_cfg: Dict[str, Any],
    row_count: int,
    remote_info: Dict[str, Any],
):
    """Records complete dataset version, URL, license, and SHA-256 checksum."""
    sha256_hash = compute_sha256(pilot_parquet_path)
    file_size_bytes = pilot_parquet_path.stat().st_size

    metadata = {
        "dataset_name": dataset_cfg.get("dataset_id", "wikimedia/wikipedia"),
        "dataset_config": dataset_cfg.get("dataset_config", "20231101.en"),
        "license": dataset_cfg.get("license", "CC BY-SA 3.0 / CC BY-SA 4.0"),
        "source_url": source_url,
        "remote_hub_info": remote_info,
        "pilot_file": {
            "path": str(pilot_parquet_path),
            "filename": pilot_parquet_path.name,
            "row_count": row_count,
            "size_bytes": file_size_bytes,
            "size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "sha256": sha256_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved dataset provenance metadata to {output_metadata_path}")
    logger.info(f"File SHA-256: {sha256_hash}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Wikipedia pilot dataset")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config YAML")
    parser.add_argument("--target-rows", type=int, default=None, help="Override target rows for pilot subset (e.g. 50000)")
    parser.add_argument("--skip-slice", action="store_true", help="Keep full downloaded shard without slicing")
    args = parser.parse_args()

    cfg_data = load_yaml(args.config)
    ds_cfg = cfg_data.get("dataset", {})
    storage_cfg = ds_cfg.get("storage", {})
    pilot_cfg = ds_cfg.get("pilot", {})

    target_rows = args.target_rows or pilot_cfg.get("target_rows", 50000)
    raw_dir = Path(storage_cfg.get("raw_dir", "/DATA/suraj/m1/search_engine/data/raw"))
    shard_filename = pilot_cfg.get("preferred_shard", "train-00000-of-00041.parquet")
    shard_url = pilot_cfg.get("shard_url", DEFAULT_SHARD_URL)
    output_filename = storage_cfg.get("output_filename", "wikipedia_en_pilot_50k.parquet")

    raw_dir.mkdir(parents=True, exist_ok=True)
    local_shard_path = raw_dir / shard_filename
    pilot_output_path = raw_dir / output_filename
    metadata_path = raw_dir / storage_cfg.get("metadata_filename", "dataset_metadata.json")

    # Step 1: Inspect remote source metadata
    remote_info = inspect_remote_sources(ds_cfg.get("dataset_id", "wikimedia/wikipedia"))

    # Step 2: Resumable download of preferred Parquet shard (~450MB)
    logger.info(f"Step 1/3: Downloading preferred Parquet shard ({shard_filename})...")
    download_file_resumable(shard_url, local_shard_path)

    # Step 3: Slice pilot subset (e.g. 50k rows)
    if args.skip_slice:
        pilot_output_path = local_shard_path
        row_count = pq.ParquetFile(str(local_shard_path)).metadata.num_rows
    else:
        logger.info(f"Step 2/3: Slicing {target_rows:,} articles into pilot file...")
        row_count = slice_pilot_subset(local_shard_path, pilot_output_path, target_rows=target_rows)

    # Step 4: Record provenance metadata and checksum
    logger.info("Step 3/3: Recording dataset provenance and checksums...")
    save_provenance_metadata(
        output_metadata_path=metadata_path,
        pilot_parquet_path=pilot_output_path,
        source_url=shard_url,
        dataset_cfg=ds_cfg,
        row_count=row_count,
        remote_info=remote_info,
    )
    logger.info("Dataset acquisition complete!")


if __name__ == "__main__":
    main()

