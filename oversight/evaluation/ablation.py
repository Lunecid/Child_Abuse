"""
Stage 11: Ablation experiments.

Grid search over configurable dimensions:
    - Correction method (A vs C)
    - Correction strength (λ, β_B, β_L)
    - Ambiguous case definition (condition 1 only / 2 only / OR)
    - 해당없음 composition (POS only / NEU only / POS+NEU)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from oversight._imports import ABUSE_ORDER
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult
from oversight.models.classifier import ClassifierResult, FIVE_CLASSES
from oversight.models.ambiguous import AmbiguousResult
from oversight.models.correction import (
    _compute_child_logodds_features,
    _compute_child_bridge_features,
    _apply_method_a,
    _apply_method_c,
)
from oversight.models.prediction import _enforce_gt_anchor
from oversight.stats.logodds import LogoddsResult
from oversight.stats.bridge import BridgeResult


@dataclass
class AblationResult:
    """Container for ablation experiment results."""
    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_ablation(
    corpus: CorpusResult,
    classifier_result: ClassifierResult,
    ambiguous_result: AmbiguousResult,
    logodds_result: LogoddsResult,
    bridge_result: BridgeResult,
    config: OversightConfig,
) -> AblationResult:
    """Run ablation grid search over correction parameters.

    Sweeps over λ (Method A) and β_B, β_L (Method C) values.
    """
    train_df = corpus.train_df
    if train_df.empty or classifier_result.oof_probs.size == 0:
        return AblationResult()

    p1 = classifier_result.oof_probs
    texts = train_df["text"].tolist()
    gt_main_indices = classifier_result.gt_main_indices

    # Ground truth
    y_cols = [f"y_{c}" for c in FIVE_CLASSES]
    existing = [c for c in y_cols if c in train_df.columns]
    y_true = train_df[existing].values.astype(int)

    # Ambiguous mask
    n = len(train_df)
    ambiguous_mask = np.zeros(n, dtype=bool)
    for idx in ambiguous_result.ambiguous_indices:
        if idx < n:
            ambiguous_mask[idx] = True

    # Precompute features
    abuse_classes = list(ABUSE_ORDER)
    logodds_df = logodds_result.logodds_train_5class
    doc_counts = logodds_result.doc_counts_train

    logodds_feats_4 = _compute_child_logodds_features(texts, logodds_df, abuse_classes)
    logodds_feats = np.zeros((n, len(FIVE_CLASSES)), dtype=float)
    logodds_feats[:, :4] = logodds_feats_4

    bridge_feats_4 = _compute_child_bridge_features(
        texts, bridge_result.bridge_correction, doc_counts, abuse_classes
    )
    bridge_feats = np.zeros((n, len(FIVE_CLASSES)), dtype=float)
    bridge_feats[:, :4] = bridge_feats_4

    # Grid
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_values = [0.1, 0.3, 0.5, 1.0, 2.0]

    rows: list[dict[str, Any]] = []

    # Method A sweep
    for lam in lambda_values:
        probs = _apply_method_a(p1, logodds_feats, ambiguous_mask, lam)
        y_pred = (probs >= 0.5).astype(int)
        y_pred = _enforce_gt_anchor(y_pred, gt_main_indices)

        n_classes = min(y_true.shape[1], y_pred.shape[1])
        micro = float(f1_score(
            y_true[:, :n_classes], y_pred[:, :n_classes],
            average="micro", zero_division=0
        ))
        macro = float(f1_score(
            y_true[:, :n_classes], y_pred[:, :n_classes],
            average="macro", zero_division=0
        ))

        rows.append({
            "method": "A",
            "lambda": lam,
            "beta_bridge": None,
            "beta_logodds": None,
            "micro_f1": micro,
            "macro_f1": macro,
        })

    # Method C sweep
    for beta_b, beta_l in product(beta_values, beta_values):
        probs = _apply_method_c(
            p1, bridge_feats, logodds_feats, ambiguous_mask, beta_b, beta_l
        )
        y_pred = (probs >= 0.5).astype(int)
        y_pred = _enforce_gt_anchor(y_pred, gt_main_indices)

        n_classes = min(y_true.shape[1], y_pred.shape[1])
        micro = float(f1_score(
            y_true[:, :n_classes], y_pred[:, :n_classes],
            average="micro", zero_division=0
        ))
        macro = float(f1_score(
            y_true[:, :n_classes], y_pred[:, :n_classes],
            average="macro", zero_division=0
        ))

        rows.append({
            "method": "C",
            "lambda": None,
            "beta_bridge": beta_b,
            "beta_logodds": beta_l,
            "micro_f1": micro,
            "macro_f1": macro,
        })

    return AblationResult(
        results_df=pd.DataFrame(rows)
    )


def save_ablation(result: AblationResult, output_dir: Path) -> None:
    """Save ablation results."""
    out = output_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    if not result.results_df.empty:
        result.results_df.to_csv(
            out / "ablation_results.csv",
            index=False, encoding="utf-8-sig"
        )
