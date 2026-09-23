"""GPU embedding engine using HuggingFace Transformers and PyTorch.

Features:
- BAAI/bge-small-en-v1.5 (384-dimensional, cosine normalized)
- FP16 inference via torch.amp.autocast on NVIDIA L4 / CUDA
- Programmatic dimension validation on initialization
- Token-length bucketed batching to minimize padding waste
- Normalized CLS pooling with numerically stable float32 normalization
- Optional torch.compile for kernel fusion speedup
- Deterministic seeding for reproducibility
"""

import os
from typing import List, Optional, Tuple, Dict, Any, Sequence
import numpy as np

# Ensure torch_compat masks torchaudio before transformers import
from src.common.torch_compat import get_torch_device, get_gpu_memory_info
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from src.common.logging import get_logger

logger = get_logger("embedding.embedder")

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_DIR = "/DATA/suraj/m1/search_engine/data/cache/huggingface"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_BUCKET_BOUNDARIES = [64, 128, 192, 256]


class GPUEmbedder:
    """High-throughput GPU embedding engine for Wikipedia text passages."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
        use_fp16: bool = True,
        expected_dim: int = DEFAULT_EMBEDDING_DIM,
        max_seq_length: int = 256,
        use_torch_compile: bool = False,
        torch_compile_mode: str = "default",
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.expected_dim = expected_dim
        self.max_seq_length = max_seq_length
        self.use_fp16 = use_fp16

        # F8: Deterministic seeding
        if seed is not None:
            self._set_deterministic(seed)

        os.environ["HF_HOME"] = self.cache_dir
        self.device = get_torch_device(device)
        logger.info(f"Initializing GPUEmbedder on device={self.device} (FP16={self.use_fp16})")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
                use_fast=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_fast=True,
            )

        # Load model
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

        # F4: Keep model in FP32 — let autocast handle per-op precision.
        # This saves the permanent .half() conversion and lets autocast
        # promote numerically sensitive ops (layernorm, softmax) to FP32.
        self.model.eval()
        self.model.to(self.device)

        # F5: Optional torch.compile for kernel fusion
        if use_torch_compile and hasattr(torch, "compile"):
            # "default" is safer than "reduce-overhead" with variable-length inputs
            # dynamic=True avoids recompilation on every unique (batch, seq_len) shape
            logger.info(f"Compiling model with torch.compile(mode={torch_compile_mode!r}, dynamic=True)")
            self.model = torch.compile(self.model, mode=torch_compile_mode, dynamic=True)
            # Trigger compilation with warmup forward pass
            self._warmup_compile()

        # Programmatic dimension verification
        self._verify_embedding_dimension()

    @staticmethod
    def _set_deterministic(seed: int):
        """Sets seeds and deterministic flags for reproducible inference."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Deterministic mode enabled with seed={seed}")

    def _warmup_compile(self):
        """Runs a dummy forward pass to trigger torch.compile graph capture."""
        dummy = self.tokenizer(
            ["Compilation warmup sentence."] * 4,
            padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        dummy = {k: v.to(self.device) for k, v in dummy.items()}
        with torch.no_grad():
            if self.device.type == "cuda" and self.use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    self.model(**dummy)
            else:
                self.model(**dummy)
        logger.info("torch.compile warmup complete")

    def _verify_embedding_dimension(self):
        """Programmatically verifies that the model produces the expected embedding dimension."""
        # 1. Config attribute check
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size != self.expected_dim:
            raise ValueError(
                f"Model config hidden_size ({hidden_size}) does not match expected_dim ({self.expected_dim})!"
            )

        # 2. Execution check: forward pass with dummy input
        dummy_inputs = self.tokenizer(
            ["dimension verification probe"],
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}

        with torch.no_grad():
            if self.device.type == "cuda" and self.use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(**dummy_inputs)
            else:
                outputs = self.model(**dummy_inputs)
            cls_rep = outputs.last_hidden_state[:, 0]
            norm_rep = F.normalize(cls_rep.float(), p=2, dim=1)

        actual_dim = norm_rep.shape[-1]
        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model output dimension ({actual_dim}) does not match expected_dim ({self.expected_dim})!"
            )

        logger.info(
            f"Embedding dimension verified programmatically: {actual_dim}d (hidden_size={hidden_size})"
        )

    def tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """Tokenizes texts on CPU. Returns dict of tensors (NOT moved to device).

        Args:
            texts: Texts to tokenize.
            max_length: Per-batch max_length override (for bucketed batching).
                        Defaults to self.max_seq_length.
        """
        effective_max = max_length or self.max_seq_length
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=effective_max,
            return_tensors="pt",
        )

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds a single batch of texts with dynamic padding.

        Returns:
            np.ndarray: L2-normalized float32 vectors of shape (len(texts), expected_dim).
        """
        if not texts:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        inputs = self.tokenize(texts)
        return self.embed_batch_tokenized(inputs)

    def embed_batch_tokenized(self, inputs: Dict[str, "torch.Tensor"]) -> np.ndarray:
        """Embeds a pre-tokenized batch (F2: separated tokenization from GPU inference).

        Args:
            inputs: Dict of tensors from tokenizer (input_ids, attention_mask, etc.).
                    Can be on CPU — will be moved to device with non_blocking=True.

        Returns:
            np.ndarray: L2-normalized float32 vectors, shape (batch, expected_dim).
        """
        if inputs["input_ids"].shape[0] == 0:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        device_inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad():
            if self.device.type == "cuda" and self.use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(**device_inputs)
            else:
                outputs = self.model(**device_inputs)

            # BGE models use normalized [CLS] representation (token 0)
            cls_embeddings = outputs.last_hidden_state[:, 0]
            # Normalize in float32 for numerical stability
            normalized = F.normalize(cls_embeddings.float(), p=2, dim=1)

        return normalized.cpu().numpy().astype(np.float32)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embeds a list of texts in batches.

        Returns:
            np.ndarray: 2D array of shape (N, expected_dim), float32, L2-normalized.
        """
        if not texts:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        num_texts = len(texts)
        all_embeddings = []

        iterator = range(0, num_texts, batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Embedding batches", unit="batch")

        for start_idx in iterator:
            batch_texts = texts[start_idx : start_idx + batch_size]
            batch_vecs = self.embed_batch(batch_texts)
            all_embeddings.append(batch_vecs)

        return np.vstack(all_embeddings)
