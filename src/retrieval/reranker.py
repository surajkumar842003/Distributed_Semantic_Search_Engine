"""Cross-Encoder Reranker using BAAI/bge-reranker-base.

Features:
- Joint cross-attention over (query, passage) pairs.
- Sigmoid activation over logits to yield calibrated [0.0, 1.0] relevance scores.
- FP16 inference on CUDA with dynamic padding.
- Batch processing for high throughput.
- Graceful handling of empty or missing candidates.
"""

import os
from typing import List, Tuple, Dict, Any, Optional, Sequence
import numpy as np

# Mask broken ABI3 torchaudio before transformers import
import src.common.torch_compat
from src.common.torch_compat import get_torch_device
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.common.schemas import RetrievalCandidate
from src.common.logging import get_logger

logger = get_logger("retrieval.reranker")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", None)


def explain_cross_encoder_formula() -> str:
    """Returns mathematical explanation of the cross-encoder reranking formula."""
    return (
        "Cross-Encoder Reranking Formulation:\n"
        "Unlike bi-encoders (which encode query and document into separate vectors and compute a dot product),\n"
        "a cross-encoder feeds the concatenated query and passage text into a deep Transformer model with full\n"
        "bidirectional cross-attention across all query tokens and document tokens:\n"
        "    Input = [CLS] + Query_Tokens + [SEP] + Document_Tokens + [SEP]\n"
        "Every token in the query attends to every token in the passage across all 12 Transformer layers.\n"
        "The [CLS] representation is passed to a classification head producing a raw relevance logit z.\n"
        "The calibrated relevance probability S_rerank(q, d) is computed via standard logistic sigmoid:\n"
        "    S_rerank(q, d) = sigma(z) = 1 / (1 + exp(-z)) in [0.0, 1.0]\n"
        "Candidates are sorted in strictly descending order of S_rerank(q, d)."
    )


class CrossEncoderReranker:
    """High-precision Transformer cross-encoder reranker for passage candidate scoring."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 512,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size

        os.environ["HF_HOME"] = self.cache_dir
        self.device = get_torch_device(device)
        logger.info(f"Initializing CrossEncoderReranker on device={self.device} (FP16={self.use_fp16})")

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
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
        except Exception:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

        self.model.eval()
        self.model.to(self.device)
        logger.info(f"CrossEncoderReranker model loaded successfully ({self.model_name}).")

    def compute_scores(
        self,
        query: str,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Computes calibrated sigmoid relevance scores for (query, text) pairs.

        Args:
            query: Query string.
            texts: Sequence of candidate passage texts.
            batch_size: Optional override for inference batch size.

        Returns:
            List[float]: Relevance scores in [0.0, 1.0] for each passage.
        """
        if not texts:
            return []

        eff_batch_size = batch_size or self.batch_size
        text_list = [str(t) for t in texts]
        all_scores: List[float] = []

        for b_start in range(0, len(text_list), eff_batch_size):
            batch_texts = text_list[b_start : b_start + eff_batch_size]

            # Use text/text_pair to ensure proper [CLS] query [SEP] doc [SEP]
            # tokenization with correct token_type_ids for cross-encoder scoring
            inputs = self.tokenizer(
                text=[query] * len(batch_texts),
                text_pair=batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad():
                if self.device.type == "cuda" and self.use_fp16:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                logits = outputs.logits.view(-1)
                sigmoids = torch.sigmoid(logits).cpu().tolist()

                # Handle scalar output for single pair
                if isinstance(sigmoids, float):
                    sigmoids = [sigmoids]
                all_scores.extend([round(float(s), 6) for s in sigmoids])

        return all_scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """Reranks retrieval candidates and returns the top_k with rerank_scores.

        Args:
            query: Query string.
            candidates: Sequence of RetrievalCandidate objects.
            top_k: Number of highest-scoring candidates to return.

        Returns:
            List[RetrievalCandidate]: Top-k candidates ordered descending by rerank_score.
        """
        if not candidates:
            return []

        texts = [c.text for c in candidates]
        scores = self.compute_scores(query, texts)

        # Attach scores to candidate copies
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            cand.rerank_score = score
            scored_candidates.append(cand)

        # Sort descending by rerank_score
        scored_candidates.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else -1.0, reverse=True)
        return scored_candidates[:top_k]

