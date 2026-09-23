"""PyTorch compatibility layer, CUDA utilities, and telemetry helpers.

Handles:
1. Registering lightweight dummy torchaudio module to prevent loading broken ABI3 extension.
2. Safe CUDA device selection and property queries.
3. Telemetry metrics: allocated/reserved VRAM and GPU utilization.
"""

import sys
import os
import types
import importlib.machinery
from typing import Dict, Any, Optional

# Mask broken ABI3 torchaudio on torch 2.10 nightly with dummy ModuleSpec
if "torchaudio" not in sys.modules or sys.modules["torchaudio"] is None:
    _dummy_torchaudio = types.ModuleType("torchaudio")
    _dummy_torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    _dummy_torchaudio.__version__ = "0.0.0"
    sys.modules["torchaudio"] = _dummy_torchaudio

import torch

# Default HuggingFace cache directory on the 5.9 TB partition
DEFAULT_HF_CACHE = "/DATA/suraj/m1/search_engine/data/cache/huggingface"
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = DEFAULT_HF_CACHE


def get_torch_device(device_str: Optional[str] = None) -> torch.device:
    """Resolves and returns a torch.device.

    Defaults to 'cuda:0' if CUDA is available, otherwise 'cpu'.
    """
    if device_str:
        dev = torch.device(device_str)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        return dev

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Returns GPU VRAM usage in MB (allocated, reserved, peak allocated, peak reserved)."""
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
            "total_mb": 0.0,
        }

    dev_idx = device.index if (device and device.type == "cuda" and device.index is not None) else 0
    total_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
    allocated_bytes = torch.cuda.memory_allocated(dev_idx)
    reserved_bytes = torch.cuda.memory_reserved(dev_idx)
    max_alloc_bytes = torch.cuda.max_memory_allocated(dev_idx)
    max_res_bytes = torch.cuda.max_memory_reserved(dev_idx)

    return {
        "allocated_mb": round(allocated_bytes / (1024 * 1024), 2),
        "reserved_mb": round(reserved_bytes / (1024 * 1024), 2),
        "peak_allocated_mb": round(max_alloc_bytes / (1024 * 1024), 2),
        "peak_reserved_mb": round(max_res_bytes / (1024 * 1024), 2),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def reset_gpu_memory_stats(device: Optional[torch.device] = None):
    """Resets the peak memory stats for the active CUDA device."""
    if torch.cuda.is_available():
        dev_idx = device.index if (device and device.type == "cuda" and device.index is not None) else 0
        torch.cuda.reset_peak_memory_stats(dev_idx)


def get_gpu_utilization(device: Optional[torch.device] = None) -> Optional[int]:
    """Queries NVIDIA GPU utilization percentage via nvidia-smi CLI."""
    if not torch.cuda.is_available():
        return None
    try:
        import subprocess
        dev_idx = device.index if (device and device.type == "cuda" and device.index is not None) else 0
        cmd = [
            "nvidia-smi",
            f"--id={dev_idx}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, timeout=1.0)
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return None
