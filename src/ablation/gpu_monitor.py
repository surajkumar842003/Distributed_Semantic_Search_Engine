"""GPU utilization monitor using nvidia-smi dmon subprocess.

Provides a context manager that samples GPU utilization in a background thread
by parsing nvidia-smi output. No additional pip dependencies required.

Usage:
    with GPUUtilizationMonitor(device_id=0, sample_interval_sec=1) as monitor:
        # ... GPU-heavy work ...
        pass
    print(f"Mean GPU util: {monitor.mean_utilization:.1f}%")
"""

import subprocess
import threading
import time
from typing import List, Optional

from src.common.logging import get_logger

logger = get_logger("ablation.gpu_monitor")


class GPUUtilizationMonitor:
    """Samples GPU SM utilization from nvidia-smi dmon in a background thread.

    Spawns `nvidia-smi dmon -s u -d <interval> -i <device>` and parses the
    `sm` (streaming multiprocessor) utilization column every sample interval.
    """

    def __init__(self, device_id: int = 0, sample_interval_sec: int = 1):
        self.device_id = device_id
        self.sample_interval_sec = sample_interval_sec
        self._samples: List[float] = []
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def __enter__(self) -> "GPUUtilizationMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Starts the nvidia-smi dmon subprocess and reader thread."""
        self._samples.clear()
        self._stop_event.clear()

        try:
            cmd = [
                "nvidia-smi", "dmon",
                "-s", "u",                          # utilization metrics
                "-d", str(self.sample_interval_sec), # sample interval
                "-i", str(self.device_id),           # GPU device
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered
            )
            self._thread = threading.Thread(
                target=self._reader_loop,
                daemon=True,
                name="gpu-util-monitor",
            )
            self._thread.start()
            logger.debug(f"GPU monitor started (device={self.device_id}, interval={self.sample_interval_sec}s)")
        except FileNotFoundError:
            logger.warning("nvidia-smi not found; GPU utilization will not be monitored")
            self._process = None
        except Exception as e:
            logger.warning(f"Failed to start GPU monitor: {e}")
            self._process = None

    def stop(self):
        """Stops the monitor and collects final samples."""
        self._stop_event.set()

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass

        if self._thread is not None:
            self._thread.join(timeout=5)

        logger.debug(
            f"GPU monitor stopped: {len(self._samples)} samples, "
            f"mean={self.mean_utilization:.1f}%"
        )

    def _reader_loop(self):
        """Background thread: reads nvidia-smi dmon output line by line."""
        if self._process is None or self._process.stdout is None:
            return

        for line in self._process.stdout:
            if self._stop_event.is_set():
                break

            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # dmon -s u output format:
            # # gpu    sm   mem    enc    dec    jpg    ofa
            #     0    45    12      0      0      0      0
            parts = line.split()
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0])
                    sm_util = float(parts[1])
                    if gpu_id == self.device_id:
                        self._samples.append(sm_util)
                except (ValueError, IndexError):
                    pass

    @property
    def samples(self) -> List[float]:
        """Returns a copy of all collected GPU utilization samples (%)."""
        return list(self._samples)

    @property
    def mean_utilization(self) -> float:
        """Returns mean GPU SM utilization (%), or 0.0 if no samples."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def peak_utilization(self) -> float:
        """Returns peak GPU SM utilization (%), or 0.0 if no samples."""
        if not self._samples:
            return 0.0
        return max(self._samples)

    @property
    def sample_count(self) -> int:
        """Returns number of collected samples."""
        return len(self._samples)

    def to_dict(self) -> dict:
        """Returns summary statistics as a dictionary."""
        return {
            "device_id": self.device_id,
            "sample_count": self.sample_count,
            "mean_utilization_pct": round(self.mean_utilization, 1),
            "peak_utilization_pct": round(self.peak_utilization, 1),
            "sample_interval_sec": self.sample_interval_sec,
        }

