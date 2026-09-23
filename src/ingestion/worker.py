"""Ingestion Worker Service for Wikipedia Ingestion Pipeline.

Supports:
- Daemon watch mode: scans $RAW_INPUT_DIR for Parquet files and ingests chunks to $CHUNKS_OUTPUT_DIR.
- Checkpointing: resumes interrupted work units safely via PipelineCheckpoint.
- Graceful shutdown: catches SIGTERM and SIGINT, finishing current batch before terminating.
- Healthcheck: maintains /tmp/worker_healthy heartbeat file and supports --health-check CLI probe.
- Single-pass batch mode: can be invoked as a one-off job container.
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, List

from src.config import AppConfig
from src.common.checkpoint import PipelineCheckpoint
from src.ingestion.pipeline import IngestionPipeline
from src.common.logging import get_logger

logger = get_logger("ingestion.worker")

DEFAULT_HEARTBEAT_FILE = "/tmp/worker_healthy"
DEFAULT_HEARTBEAT_TIMEOUT = 90  # seconds


class IngestionWorker:
    """Daemon worker managing chunking and normalization over raw article batches."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        watch_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        watch_interval: int = 10,
        heartbeat_file: str = DEFAULT_HEARTBEAT_FILE,
    ):
        self.config = config or AppConfig.load_from_dir("configs")
        data_dir = Path(self.config.paths.data_dir)

        self.watch_dir = Path(watch_dir or os.environ.get("RAW_INPUT_DIR", str(data_dir / "raw")))
        self.output_dir = Path(output_dir or os.environ.get("CHUNKS_OUTPUT_DIR", str(data_dir / "chunks")))
        self.watch_interval = int(os.environ.get("INGESTION_WATCH_INTERVAL", watch_interval))
        self.heartbeat_file = Path(heartbeat_file)

        self.stop_requested = False
        self._setup_signals()

        # Ensure directories exist
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self.output_dir / "ingestion_checkpoint.json"
        self.checkpoint = PipelineCheckpoint(str(checkpoint_path))
        self.pipeline = IngestionPipeline(config=self.config, checkpoint=self.checkpoint)

        logger.info(
            f"Initialized IngestionWorker (watch_dir={self.watch_dir}, "
            f"output_dir={self.output_dir}, workers={self.config.ingestion.num_cpu_workers}, "
            f"interval={self.watch_interval}s)"
        )

    def _setup_signals(self):
        """Registers POSIX signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.info(f"Received {sig_name}; initiating graceful worker shutdown...")
        self.stop_requested = True

    def touch_heartbeat(self):
        """Updates the healthcheck heartbeat file with current epoch timestamp."""
        try:
            self.heartbeat_file.write_text(str(time.time()), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to touch heartbeat file: {e}")

    def find_pending_files(self) -> List[Path]:
        """Finds all Parquet files in the watch directory."""
        if not self.watch_dir.is_dir():
            return []
        files = sorted(self.watch_dir.glob("*.parquet"))
        return [f for f in files if not f.name.endswith(".tmp")]

    def process_file(self, file_path: Path) -> dict:
        """Processes a single raw Parquet file through the chunking pipeline."""
        logger.info(f"Processing raw file: {file_path.name}")
        self.touch_heartbeat()
        stats = self.pipeline.run(str(file_path), str(self.output_dir))
        self.touch_heartbeat()
        return stats

    def run_single_pass(self) -> int:
        """Processes all currently pending files once and exits. Returns processed file count."""
        pending = self.find_pending_files()
        if not pending:
            logger.info("No pending raw Parquet files found.")
            return 0

        logger.info(f"Single pass: found {len(pending)} file(s) to process.")
        processed = 0
        for f in pending:
            if self.stop_requested:
                logger.info("Stop requested; terminating single pass early.")
                break
            self.process_file(f)
            processed += 1

        self.touch_heartbeat()
        return processed

    def run_daemon(self):
        """Runs the continuous worker loop until SIGTERM or SIGINT is caught."""
        logger.info("Starting IngestionWorker daemon loop...")
        self.touch_heartbeat()

        while not self.stop_requested:
            try:
                pending = self.find_pending_files()
                if pending:
                    for f in pending:
                        if self.stop_requested:
                            break
                        self.process_file(f)
                else:
                    logger.debug(f"No pending files in {self.watch_dir}; waiting {self.watch_interval}s...")

                self.touch_heartbeat()

                # Sleep in short increments for responsive signal handling
                for _ in range(self.watch_interval):
                    if self.stop_requested:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in ingestion worker loop: {e}", exc_info=True)
                time.sleep(self.watch_interval)

        logger.info("IngestionWorker shutdown clean.")
        if self.heartbeat_file.exists():
            try:
                self.heartbeat_file.unlink()
            except Exception:
                pass


def check_health(heartbeat_file: str = DEFAULT_HEARTBEAT_FILE, max_age_sec: int = DEFAULT_HEARTBEAT_TIMEOUT) -> int:
    """CLI healthcheck probe for Docker Compose healthcheck.

    Returns:
        0 if healthy (heartbeat exists and updated recently), 1 otherwise.
    """
    path = Path(heartbeat_file)
    if not path.is_file():
        print(f"Healthcheck FAIL: Heartbeat file {heartbeat_file} does not exist.")
        return 1

    try:
        content = path.read_text(encoding="utf-8").strip()
        last_heartbeat = float(content)
        age = time.time() - last_heartbeat
        if age > max_age_sec:
            print(f"Healthcheck FAIL: Heartbeat is stale ({age:.1f}s > {max_age_sec}s).")
            return 1
        print(f"Healthcheck OK: Heartbeat is fresh ({age:.1f}s old).")
        return 0
    except Exception as e:
        print(f"Healthcheck FAIL: Could not read heartbeat ({e}).")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Wikipedia Ingestion Worker Service")
    parser.add_argument("--watch-dir", type=str, default=None, help="Directory to watch for raw Parquet files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write chunked Parquet shards")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--single-pass", action="store_true", help="Process pending files once and exit")
    parser.add_argument("--health-check", action="store_true", help="Run Docker healthcheck probe and exit")
    parser.add_argument("--heartbeat-file", type=str, default=DEFAULT_HEARTBEAT_FILE)
    args = parser.parse_args()

    if args.health_check:
        sys.exit(check_health(args.heartbeat_file))

    worker = IngestionWorker(
        watch_dir=args.watch_dir,
        output_dir=args.output_dir,
        watch_interval=args.interval,
        heartbeat_file=args.heartbeat_file,
    )

    if args.single_pass:
        count = worker.run_single_pass()
        logger.info(f"Single pass finished. Processed {count} file(s).")
    else:
        worker.run_daemon()


if __name__ == "__main__":
    main()

