"""
Stage 11: Evaluation metrics.

Computes multi-label classification metrics:
    - micro-F1, macro-F1, per-class P/R/F1
    - Hamming loss, Jaccard similarity
    - Information recovery rate (IRR)
    - Boundary recall for all 6 abuse-type pairs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)

from oversight._imports import ABUSE_ORDER, compute_recovery_metrics
from oversight.models.classifier import FIVE_CLASSES
from oversight.models.prediction import PredictionResult


def evaluate_predictions(
    prediction_result: PredictionResult,
    method: str = "method_a",
) -> dict[str, Any]:
    """Evaluate predictions using multi-label metrics.

    Parameters
    ----------
    prediction_result : PredictionResult
        Contains y_true and y_pred for both methods.
    method : str
        "method_a" or "method_c".

    Returns
    -------
    dict with all computed metrics.
    """
    y_true = prediction_result.y_true
    y_pred = (
        prediction_result.y_pred_a if method == "method_a"
        else prediction_result.y_pred_c
    )
    gt_main_indices = prediction_result.gt_main_indices

    if y_true.size == 0 or y_pred.size == 0:
        return {}

    n_classes = min(y_true.shape[1], y_pred.shape[1])
    y_true = y_true[:, :n_classes]
    y_pred = y_pred[:, :n_classes]

    results: dict[str, Any] = {"method": method}

    # Global metrics
    results["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    results["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    results["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    # Jaccard similarity (sample-averaged)
    jaccard_scores = []
    for i in range(len(y_true)):
        intersection = np.logical_and(y_true[i], y_pred[i]).sum()
        union = np.logical_or(y_true[i], y_pred[i]).sum()
        if union > 0:
            jaccard_scores.append(intersection / union)
        else:
            jaccard_scores.append(1.0)
    results["jaccard_mean"] = float(np.mean(jaccard_scores))

    # Per-class metrics
    classes = FIVE_CLASSES[:n_classes]
    per_class = []
    for j, cls in enumerate(classes):
        p = float(precision_score(y_true[:, j], y_pred[:, j], zero_division=0))
        r = float(recall_score(y_true[:, j], y_pred[:, j], zero_division=0))
        f = float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0))
        per_class.append({"class": cls, "precision": p, "recall": r, "f1": f})
    results["per_class"] = per_class

    # Information recovery metrics (using legacy function)
    if gt_main_indices.size > 0 and n_classes >= 4:
        try:
            irr_metrics = compute_recovery_metrics(
                f"oversight_{method}",
                y_true[:, :4],  # Abuse types only
                y_pred[:, :4],
                gt_main_indices[gt_main_indices < 4],  # Only abuse children
            )
            results["irr"] = irr_metrics
        except Exception:
            pass

    return results


def save_evaluation(
    metrics: dict[str, Any],
    output_dir: Path,
    method: str = "method_a",
) -> None:
    """Save evaluation metrics to CSV."""
    out = output_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    # Global metrics
    global_metrics = {
        k: v for k, v in metrics.items()
        if k not in ("per_class", "irr")
    }
    pd.DataFrame([global_metrics]).to_csv(
        out / f"metrics_global_{method}.csv",
        index=False, encoding="utf-8-sig"
    )

    # Per-class metrics
    if "per_class" in metrics:
        pd.DataFrame(metrics["per_class"]).to_csv(
            out / f"metrics_per_class_{method}.csv",
            index=False, encoding="utf-8-sig"
        )

    # IRR metrics
    if "irr" in metrics and isinstance(metrics["irr"], dict):
        irr_flat = {}
        for k, v in metrics["irr"].items():
            if isinstance(v, (int, float, str, bool)):
                irr_flat[k] = v
        if irr_flat:
            pd.DataFrame([irr_flat]).to_csv(
                out / f"metrics_irr_{method}.csv",
                index=False, encoding="utf-8-sig"
            )
