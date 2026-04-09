"""
Stage 11: External baseline comparisons.

Two external baselines for comparison (different prediction strategies,
not classifier corrections):

    1. Keyword matching (top log-odds words → binary per class)
    2. Single-label top-k (classify main only, predict top-2 classes)

Note: "No correction" is Condition 1 in the ablation table, not a separate
baseline. See ablation.py for the 4-condition component ablation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oversight._imports import ABUSE_ORDER
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult
from oversight.models.classifier import ClassifierResult, FIVE_CLASSES
from oversight.models.prediction import PredictionResult, _enforce_gt_anchor
from oversight.stats.logodds import LogoddsResult
from oversight.evaluation.metrics import evaluate_predictions


def _baseline_keyword_matching(
    corpus: CorpusResult,
    logodds_result: LogoddsResult,
    prediction_result: PredictionResult,
    top_k: int = 20,
) -> dict[str, Any]:
    """Baseline 2: Keyword matching using top log-odds words."""
    train_df = corpus.train_df
    if train_df.empty or logodds_result.logodds_train_5class.empty:
        return {}

    logodds_df = logodds_result.logodds_train_5class

    # Get top-k words per class
    top_words: dict[str, set[str]] = {}
    for cls in ABUSE_ORDER:
        cls_df = logodds_df[logodds_df["group"] == cls].copy()
        cls_df = cls_df.sort_values("log_odds", ascending=False, kind="mergesort")
        top_words[cls] = set(cls_df.head(top_k)["word"])

    # Predict: class = 1 if child has any top-k word for that class
    n = len(train_df)
    y_pred = np.zeros((n, len(FIVE_CLASSES)), dtype=int)

    for i, text in enumerate(train_df["text"]):
        if not isinstance(text, str):
            continue
        tokens = set(text.split())
        for j, cls in enumerate(FIVE_CLASSES):
            if cls in top_words and tokens & top_words[cls]:
                y_pred[i, j] = 1

    y_pred = _enforce_gt_anchor(
        y_pred, prediction_result.gt_main_indices
    )

    baseline_pred = PredictionResult(
        y_pred_a=y_pred,
        y_true=prediction_result.y_true,
        gt_main_indices=prediction_result.gt_main_indices,
    )
    return evaluate_predictions(baseline_pred, method="method_a")


def _baseline_single_label_top2(
    classifier_result: ClassifierResult,
    prediction_result: PredictionResult,
) -> dict[str, Any]:
    """Baseline 3: Single-label top-2 — take top 2 classes by probability."""
    p1 = classifier_result.oof_probs
    if p1.size == 0:
        return {}

    n = p1.shape[0]
    y_pred = np.zeros_like(p1, dtype=int)

    for i in range(n):
        top2 = np.argsort(-p1[i])[:2]
        for j in top2:
            y_pred[i, j] = 1

    y_pred = _enforce_gt_anchor(y_pred, classifier_result.gt_main_indices)

    baseline_pred = PredictionResult(
        y_pred_a=y_pred,
        y_true=prediction_result.y_true,
        gt_main_indices=prediction_result.gt_main_indices,
    )
    return evaluate_predictions(baseline_pred, method="method_a")


def run_baselines(
    corpus: CorpusResult,
    classifier_result: ClassifierResult,
    prediction_result: PredictionResult,
    logodds_result: LogoddsResult,
    config: OversightConfig,
) -> dict[str, dict[str, Any]]:
    """Run external baselines and return results.

    Note: "No correction" is Condition 1 in the ablation table (ablation.py),
    not a separate baseline here.
    """
    results = {}

    results["keyword_matching"] = _baseline_keyword_matching(
        corpus, logodds_result, prediction_result
    )
    results["single_label_top2"] = _baseline_single_label_top2(
        classifier_result, prediction_result
    )

    return results


def save_baselines(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save baseline comparison results."""
    out = output_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, metrics in results.items():
        row = {"baseline": name}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                row[k] = v
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(
            out / "baselines_comparison.csv",
            index=False, encoding="utf-8-sig"
        )
