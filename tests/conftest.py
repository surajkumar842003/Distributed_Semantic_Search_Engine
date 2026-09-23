"""Pytest root configuration and test fixtures."""
import sys
import types
import importlib.machinery

# Ensure torchaudio dummy is registered before any tests import transformers
if "torchaudio" not in sys.modules or sys.modules["torchaudio"] is None:
    dummy = types.ModuleType("torchaudio")
    dummy.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    dummy.__version__ = "0.0.0"
    sys.modules["torchaudio"] = dummy

