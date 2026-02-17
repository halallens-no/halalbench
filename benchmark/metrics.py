"""
HalalBench — Metric computation for OCR evaluation.

Provides exact-match F1, fuzzy-match F1 (Levenshtein), catastrophic failure
rate, and per-language breakdowns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    raise ImportError(
        "python-Levenshtein is required. Install with: pip install Levenshtein"
    )


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace / commas / parentheses."""
    text = text.lower().strip()
    for ch in "(),;:[]{}":
        text = text.replace(ch, " ")
    return [t for t in text.split() if t]


def _best_match(token: str, candidates: list[str], threshold: int) -> bool:
    """Return True if *token* fuzzy-matches any candidate within edit distance."""
    for c in candidates:
        if levenshtein_distance(token, c) <= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def exact_match_f1(predicted: str, ground_truth: str) -> dict[str, float]:
    """
    Compute token-level precision, recall, and F1 using exact string match.

    Parameters
    ----------
    predicted : str
        OCR-predicted ingredient text.
    ground_truth : str
        Ground-truth ingredient text.

    Returns
    -------
    dict with keys "precision", "recall", "f1".
    """
    pred_tokens = _tokenize(predicted)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def fuzzy_match_f1(
    predicted: str,
    ground_truth: str,
    threshold: int = 2,
) -> dict[str, float]:
    """
    Compute token-level precision, recall, and F1 using Levenshtein fuzzy match.

    A predicted token counts as a true positive if its edit distance to any
    ground-truth token is <= *threshold*.

    Parameters
    ----------
    predicted : str
        OCR-predicted ingredient text.
    ground_truth : str
        Ground-truth ingredient text.
    threshold : int
        Maximum edit distance for a fuzzy match (default 2).

    Returns
    -------
    dict with keys "precision", "recall", "f1".
    """
    pred_tokens = _tokenize(predicted)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Precision: fraction of predicted tokens that fuzzy-match a GT token
    tp_pred = sum(1 for t in pred_tokens if _best_match(t, gt_tokens, threshold))
    precision = tp_pred / len(pred_tokens)

    # Recall: fraction of GT tokens that fuzzy-match a predicted token
    tp_gt = sum(1 for t in gt_tokens if _best_match(t, pred_tokens, threshold))
    recall = tp_gt / len(gt_tokens)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def catastrophic_rate(
    f1_scores: Sequence[float],
    threshold: float = 0.05,
) -> float:
    """
    Fraction of images where F1 falls below *threshold* (catastrophic failure).

    Parameters
    ----------
    f1_scores : sequence of float
        Per-image F1 scores.
    threshold : float
        F1 below this value counts as catastrophic (default 0.05).

    Returns
    -------
    float
        Catastrophic failure rate in [0, 1].
    """
    if not f1_scores:
        return 0.0
    scores = np.asarray(f1_scores, dtype=float)
    return float(np.mean(scores < threshold))


def per_language_f1(
    results: list[dict],
    languages: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Aggregate F1 scores per language.

    Parameters
    ----------
    results : list of dict
        Each dict must have keys "language", "exact_f1", "fuzzy_f1".
    languages : list of str, optional
        If provided, only report these languages. Otherwise report all.

    Returns
    -------
    dict mapping language code to {"exact_f1": float, "fuzzy_f1": float,
    "count": int, "catastrophic_rate": float}.
    """
    by_lang: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"exact": [], "fuzzy": []}
    )

    for r in results:
        lang = r["language"]
        if languages and lang not in languages:
            continue
        by_lang[lang]["exact"].append(r["exact_f1"])
        by_lang[lang]["fuzzy"].append(r["fuzzy_f1"])

    summary: dict[str, dict[str, float]] = {}
    for lang in sorted(by_lang):
        exact_scores = by_lang[lang]["exact"]
        fuzzy_scores = by_lang[lang]["fuzzy"]
        summary[lang] = {
            "exact_f1": float(np.mean(exact_scores)),
            "fuzzy_f1": float(np.mean(fuzzy_scores)),
            "count": len(exact_scores),
            "catastrophic_rate": catastrophic_rate(exact_scores),
        }

    return summary
