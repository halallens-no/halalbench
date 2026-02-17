#!/usr/bin/env python3
"""
HalalBench — Evaluation script.

Loads COCO-format ground-truth annotations and OCR prediction files, then
computes exact-match F1, fuzzy-match F1, and catastrophic failure rate per
engine and per language.

Usage
-----
    python evaluate.py \
        --annotations data/annotations.json \
        --predictions predictions/mlkit.json \
        --output results/

    python evaluate.py \
        --annotations data/annotations.json \
        --predictions predictions/mlkit.json \
        --metrics-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import exact_match_f1, fuzzy_match_f1, catastrophic_rate, per_language_f1


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

def load_coco_annotations(path: str) -> dict:
    """Load a COCO-format JSON and return {image_id: {filename, language, text}}."""
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    # Build ground-truth text per image by joining all annotation texts
    gt: dict[int, dict] = {}
    for img_id, img_info in images.items():
        gt[img_id] = {
            "filename": img_info.get("file_name", ""),
            "language": img_info.get("language", "unknown"),
            "text": "",
        }

    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        text = ann.get("attributes", {}).get("text", ann.get("text", ""))
        if img_id in gt and text:
            gt[img_id]["text"] += (" " + text) if gt[img_id]["text"] else text

    return gt


def load_predictions(path: str) -> dict[int, str]:
    """
    Load OCR predictions. Expected format:

        {
          "predictions": [
            {"image_id": 1, "text": "..."},
            ...
          ]
        }

    Returns {image_id: predicted_text}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds: dict[int, str] = {}
    for entry in data.get("predictions", []):
        preds[entry["image_id"]] = entry.get("text", "")

    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    gt: dict[int, dict],
    predictions: dict[int, str],
    fuzzy_threshold: int = 2,
) -> list[dict]:
    """Run per-image evaluation and return a list of result dicts."""
    results = []
    for img_id, info in gt.items():
        gt_text = info["text"]
        pred_text = predictions.get(img_id, "")

        exact = exact_match_f1(pred_text, gt_text)
        fuzzy = fuzzy_match_f1(pred_text, gt_text, threshold=fuzzy_threshold)

        results.append({
            "image_id": img_id,
            "filename": info["filename"],
            "language": info["language"],
            "exact_precision": exact["precision"],
            "exact_recall": exact["recall"],
            "exact_f1": exact["f1"],
            "fuzzy_precision": fuzzy["precision"],
            "fuzzy_recall": fuzzy["recall"],
            "fuzzy_f1": fuzzy["f1"],
        })

    return results


def print_summary(results: list[dict], engine_name: str = "OCR Engine") -> None:
    """Print aggregate and per-language summary tables."""
    if not results:
        print("No results to summarize.")
        return

    df = pd.DataFrame(results)

    # -- Aggregate --
    exact_f1s = df["exact_f1"].values
    fuzzy_f1s = df["fuzzy_f1"].values

    print("=" * 70)
    print(f"  HalalBench Evaluation — {engine_name}")
    print("=" * 70)
    print(f"  Images evaluated:    {len(df)}")
    print(f"  Exact-match F1:      {np.mean(exact_f1s):.3f}  (std {np.std(exact_f1s):.3f})")
    print(f"  Fuzzy-match F1:      {np.mean(fuzzy_f1s):.3f}  (std {np.std(fuzzy_f1s):.3f})")
    print(f"  Catastrophic rate:   {catastrophic_rate(exact_f1s):.3f}")
    print("=" * 70)

    # -- Per-language --
    lang_results = per_language_f1(results)
    if lang_results:
        print()
        print(f"  {'Language':<12} {'Count':>6} {'Exact F1':>10} {'Fuzzy F1':>10} {'Catast.':>10}")
        print("  " + "-" * 52)
        for lang in sorted(lang_results):
            s = lang_results[lang]
            print(
                f"  {lang:<12} {s['count']:>6} "
                f"{s['exact_f1']:>10.3f} {s['fuzzy_f1']:>10.3f} "
                f"{s['catastrophic_rate']:>10.3f}"
            )
        print()


def save_results(results: list[dict], output_dir: str, engine_name: str) -> None:
    """Save detailed results to CSV and summary to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{engine_name}_detailed.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Detailed results saved to: {csv_path}")

    summary = {
        "engine": engine_name,
        "images": len(df),
        "exact_f1_mean": float(np.mean(df["exact_f1"])),
        "exact_f1_std": float(np.std(df["exact_f1"])),
        "fuzzy_f1_mean": float(np.mean(df["fuzzy_f1"])),
        "fuzzy_f1_std": float(np.std(df["fuzzy_f1"])),
        "catastrophic_rate": float(catastrophic_rate(df["exact_f1"].tolist())),
        "per_language": per_language_f1(results),
    }

    json_path = os.path.join(output_dir, f"{engine_name}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to:          {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HalalBench OCR Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations", "-a",
        required=True,
        help="Path to COCO-format ground-truth annotations JSON.",
    )
    parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Path to OCR predictions JSON.",
    )
    parser.add_argument(
        "--engine-name", "-e",
        default=None,
        help="Engine name for labeling results (default: inferred from filename).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for detailed CSV and summary JSON.",
    )
    parser.add_argument(
        "--fuzzy-threshold", "-t",
        type=int,
        default=2,
        help="Max edit distance for fuzzy matching (default: 2).",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print metrics to stdout without saving files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Infer engine name from prediction filename if not provided
    engine_name = args.engine_name
    if engine_name is None:
        engine_name = Path(args.predictions).stem.replace("_predictions", "")

    # Load data
    print(f"\nLoading annotations from: {args.annotations}")
    gt = load_coco_annotations(args.annotations)
    print(f"  Ground-truth images: {len(gt)}")

    print(f"Loading predictions from: {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"  Prediction entries:  {len(predictions)}")

    # Evaluate
    results = evaluate(gt, predictions, fuzzy_threshold=args.fuzzy_threshold)

    # Report
    print_summary(results, engine_name=engine_name)

    # Save
    if not args.metrics_only and args.output:
        save_results(results, args.output, engine_name)
    elif not args.metrics_only and not args.output:
        print("  (Use --output DIR to save detailed results, or --metrics-only to skip.)\n")


if __name__ == "__main__":
    main()
