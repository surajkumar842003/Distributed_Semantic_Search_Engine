"""Downstream Answer Generation Quality Metrics.

Evaluates generated answer fidelity, exact match (EM), token F1, citation precision,
and context insufficiency detection accuracy.

Separated completely from retrieval metrics to isolate retrieval ranking errors
from generation hallucinations.
"""

from typing import Sequence, Dict, Any, List, Tuple, Set
import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Standard SQuAD / NQ text normalization for open-domain QA evaluation.

    Steps:
    1. Lowercase
    2. Strip punctuation
    3. Remove articles ('a', 'an', 'the')
    4. Collapse duplicate whitespace
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(text)))))


def compute_exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    """Computes Exact Match (EM) binary score between prediction and any ground truth.

    Args:
        prediction: Model-generated answer string.
        ground_truths: List of acceptable gold reference strings.

    Returns:
        float: 1.0 if normalized prediction matches any normalized gold answer, else 0.0.
    """
    if not ground_truths:
        return 0.0

    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if norm_pred == normalize_answer(gt):
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, ground_truths: Sequence[str]) -> Dict[str, float]:
    """Computes unigram Token Precision, Recall, and F1 score against gold references.

    Args:
        prediction: Model-generated answer string.
        ground_truths: List of acceptable gold reference strings.

    Returns:
        Dict[str, float]: Maximum F1, Precision, and Recall across all gold references.
    """
    if not ground_truths:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    max_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)

        if f1 > max_f1:
            max_f1 = f1
            best_prec = precision
            best_rec = recall

    return {
        "f1": round(max_f1, 4),
        "precision": round(best_prec, 4),
        "recall": round(best_rec, 4),
    }


def compute_citation_precision(
    cited_chunk_ids: Sequence[int],
    ground_truth_chunk_ids: Set[int],
) -> float:
    """Computes citation precision for grounded RAG generation.

    Citation Precision = (number of cited chunks that are true evidence) / (total citations)
    If no citations were generated:
      - returns 1.0 if there were no ground truth chunks (correct abstention)
      - returns 0.0 if ground truth chunks existed

    Args:
        cited_chunk_ids: List of chunk IDs cited in the answer.
        ground_truth_chunk_ids: Set of chunk IDs containing actual answer evidence.

    Returns:
        float: Precision score between 0.0 and 1.0.
    """
    if not cited_chunk_ids:
        return 1.0 if not ground_truth_chunk_ids else 0.0

    num_valid = sum(1 for cid in cited_chunk_ids if cid in ground_truth_chunk_ids)
    return round(float(num_valid) / float(len(cited_chunk_ids)), 4)


def evaluate_generation_quality(
    prediction: str,
    gold_answers: Sequence[str],
    cited_chunk_ids: Sequence[int],
    ground_truth_chunk_ids: Set[int],
    insufficient_context_flag: bool,
) -> Dict[str, Any]:
    """Computes end-to-end generation quality metrics for a single RAG response."""
    em = compute_exact_match(prediction, gold_answers)
    token_metrics = compute_token_f1(prediction, gold_answers)
    citation_prec = compute_citation_precision(cited_chunk_ids, ground_truth_chunk_ids)

    # Correctness of context insufficiency detection
    has_evidence = len(ground_truth_chunk_ids) > 0
    correct_abstention = (insufficient_context_flag and not has_evidence) or \
                         (not insufficient_context_flag and has_evidence)

    return {
        "exact_match": em,
        "token_f1": token_metrics["f1"],
        "token_precision": token_metrics["precision"],
        "token_recall": token_metrics["recall"],
        "citation_precision": citation_prec,
        "correct_abstention": 1.0 if correct_abstention else 0.0,
        "insufficient_context_flag": insufficient_context_flag,
    }
