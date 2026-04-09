"""
Stage 9: Final prediction assembly.

Combines original and corrected probabilities:
    - Ambiguous cases → use corrected probabilities (Method A or C)
    - Clear cases → use original p1 probabilities

Produces binary predictions via thresholding and enforces GT anchor:
    if a child has a known gt_main, that class is always set to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from oversight._imports import ABUSE_ORDER
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult
from oversight.models.classifier import ClassifierResult, FIVE_CLASSES
from oversight.models.ambiguous import AmbiguousResult
from oversight.models.correction import CorrectionResult


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """Container for final predictions."""

    # Final probabilities: shape (N, 5)
    final_probs_a: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    final_probs_c: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))

    # Binary predictions: shape (N, 5)
    y_pred_a: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    y_pred_c: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))

    # Ground truth: shape (N, 5)
    y_true: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))

    # GT main indices: shape (N,)
    gt_main_indices: np.ndarray = field(default_factory=lambda: np.array([]))


# ═══════════════════════════════════════════════════════════════════
#  Prediction assembly
# ═══════════════════════════════════════════════════════════════════

def _assemble_probs(
    p1: np.ndarray,
    p_corrected: np.ndarray,
    ambiguous_indices: set[int],
) -> np.ndarray:
    """Merge p1 (clear cases) with corrected probs (ambiguous cases)."""
    if p_corrected.size == 0:
        return p1.copy()

    final = p1.copy()
    for idx in ambiguous_indices:
        if idx < len(final):
            final[idx] = p_corrected[idx]
    return final


def _threshold_predictions(
    probs: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert probabilities to binary predictions."""
    return (probs >= threshold).astype(int)


def _enforce_gt_anchor(
    y_pred: np.ndarray,
    gt_main_indices: np.ndarray,
) -> np.ndarray:
    """Enforce GT anchor: if child has gt_main, that class = 1.

    gt_main_indices: index into FIVE_CLASSES (0-3 for abuse, 4 for 해당없음).
    """
    y_pred = y_pred.copy()
    for i, gt_idx in enumerate(gt_main_indices):
        gt_idx = int(gt_idx)
        if 0 <= gt_idx < y_pred.shape[1]:
            y_pred[i, gt_idx] = 1
    return y_pred


def assemble_final_predictions(
    corpus: CorpusResult,
    classifier_result: ClassifierResult,
    ambiguous_result: AmbiguousResult,
    correction_result: CorrectionResult,
    config: OversightConfig,
) -> PredictionResult:
    """Stage 9: Assemble final predictions from corrected probabilities.

    For each correction method (A and C):
    1. Use corrected probs for ambiguous, original for clear cases
    2. Threshold to binary predictions
    3. Enforce GT anchor
    """
    train_df = corpus.train_df
    if train_df.empty or classifier_result.oof_probs.size == 0:
        return PredictionResult()

    p1 = classifier_result.oof_probs
    gt_main_indices = classifier_result.gt_main_indices
    ambiguous = ambiguous_result.ambiguous_indices

    # Method A
    final_probs_a = _assemble_probs(p1, correction_result.probs_method_a, ambiguous)
    y_pred_a = _threshold_predictions(final_probs_a)
    y_pred_a = _enforce_gt_anchor(y_pred_a, gt_main_indices)

    # Method C
    final_probs_c = _assemble_probs(p1, correction_result.probs_method_c, ambiguous)
    y_pred_c = _threshold_predictions(final_probs_c)
    y_pred_c = _enforce_gt_anchor(y_pred_c, gt_main_indices)

    # Ground truth
    y_cols = [f"y_{c}" for c in FIVE_CLASSES]
    existing = [c for c in y_cols if c in train_df.columns]
    y_true = train_df[existing].values.astype(int)

    return PredictionResult(
        final_probs_a=final_probs_a,
        final_probs_c=final_probs_c,
        y_pred_a=y_pred_a,
        y_pred_c=y_pred_c,
        y_true=y_true,
        gt_main_indices=gt_main_indices,
    )


def save_predictions(
    result: PredictionResult,
    train_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save final predictions."""
    out = output_dir / "predictions"
    out.mkdir(parents=True, exist_ok=True)

    for method, y_pred, probs in [
        ("method_a", result.y_pred_a, result.final_probs_a),
        ("method_c", result.y_pred_c, result.final_probs_c),
    ]:
        if y_pred.size == 0:
            continue

        pred_df = pd.DataFrame(
            y_pred,
            columns=[f"pred_{c}" for c in FIVE_CLASSES],
        )
        prob_df = pd.DataFrame(
            probs,
            columns=[f"prob_{c}" for c in FIVE_CLASSES],
        )

        combined = pd.concat([pred_df, prob_df], axis=1)
        if "doc_id" in train_df.columns:
            combined.insert(0, "doc_id", train_df["doc_id"].values)
        if "gt_main" in train_df.columns:
            combined.insert(1, "gt_main", train_df["gt_main"].values)

        combined.to_csv(
            out / f"predictions_{method}.csv",
            index=False, encoding="utf-8-sig"
        )
