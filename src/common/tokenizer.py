"""Hugging Face Tokenizer utility with local offline caching."""
import os
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_DIR = "/DATA/suraj/m1/search_engine/data/cache/huggingface"

_TOKENIZER_REGISTRY = {}


def get_tokenizer(model_name: str = DEFAULT_MODEL, cache_dir: str = DEFAULT_CACHE_DIR):
    """
    Retrieves the fast tokenizer for the specified embedding model.
    Prioritizes local offline files to avoid external network calls.
    """
    if model_name in _TOKENIZER_REGISTRY:
        return _TOKENIZER_REGISTRY[model_name]

    os.environ["HF_HOME"] = cache_dir
    try:
        # Try offline load first
        tok = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            use_fast=True,
        )
    except Exception:
        # Fallback to standard load
        tok = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
        )

    _TOKENIZER_REGISTRY[model_name] = tok
    return tok

