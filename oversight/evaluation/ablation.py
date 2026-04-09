"""
Stage 11: Ablation experiments — structured contribution proof.

Two ablation axes:

Primary axis — Correction component ablation (Paper Table):
    Uses Method C only, which separates bridge and log-odds terms.
    Condition 1: No correction (β_B=0, β_L=0)     — floor
    Condition 2: Log-odds only (β_B=0, β_L=β_L*)   — log-odds contribution
    Condition 3: Bridge only  (β_B=β_B*, β_L=0)    — bridge contribution
    Condition 4: Full correction (β_B=β_B*, β_L=β_L*) — proposed method

Secondary axis — Method comparison:
    At full correction, compare Method A (merged) vs Method C (separated).

Hyperparameter sweep — Supplementary:
    Grid search to find optimal β_B*, β_L*, λ*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
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


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AblationResult:
    """Container for structured ablation results."""

    # Primary: 4-condition component ablation (Method C)
    component_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Secondary: Method A vs C comparison at full correction
    method_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Supplementary: hyperparameter sweep
    sweep_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Optimal hyperparameters found by sweep
    beta_b_star: float = 0.0
    beta_l_star: float = 0.0
    lambda_star: float = 0.0


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _precompute_features(
    corpus: CorpusResult,
    logodds_result: LogoddsResult,
    bridge_result: BridgeResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute log-odds and bridge features for all children.

    Returns (logodds_feats, bridge_feats), both shape (N, 5).
    """
    train_df = corpus.train_df
    n = len(train_df)
    texts = train_df["text"].tolist()
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

    return logodds_feats, bridge_feats


def _build_ambiguous_mask(
    ambiguous_result: AmbiguousResult,
    n: int,
) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for idx in ambiguous_result.ambiguous_indices:
        if idx < n:
            mask[idx] = True
    return mask


def _evaluate_condition(
    p1: np.ndarray,
    bridge_feats: np.ndarray,
    logodds_feats: np.ndarray,
    ambiguous_mask: np.ndarray,
    gt_main_indices: np.ndarray,
    y_true: np.ndarray,
    beta_bridge: float,
    beta_logodds: float,
    method: str = "C",
    lam: float = 0.0,
) -> dict[str, Any]:
    """Evaluate a single correction condition and return metrics."""
    if method == "A":
        probs = _apply_method_a(p1, logodds_feats, ambiguous_mask, lam)
    else:
        probs = _apply_method_c(
            p1, bridge_feats, logodds_feats, ambiguous_mask,
            beta_bridge, beta_logodds,
        )

    y_pred = (probs >= 0.5).astype(int)
    y_pred = _enforce_gt_anchor(y_pred, gt_main_indices)

    n_classes = min(y_true.shape[1], y_pred.shape[1])
    yt = y_true[:, :n_classes]
    yp = y_pred[:, :n_classes]

    metrics: dict[str, Any] = {
        "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "hamming": float(hamming_loss(yt, yp)),
    }

    # Jaccard (sample-averaged)
    jaccard_scores = []
    for i in range(len(yt)):
        inter = np.logical_and(yt[i], yp[i]).sum()
        union = np.logical_or(yt[i], yp[i]).sum()
        jaccard_scores.append(inter / union if union > 0 else 1.0)
    metrics["jaccard"] = float(np.mean(jaccard_scores))

    # Per-class F1
    classes = FIVE_CLASSES[:n_classes]
    for j, cls in enumerate(classes):
        metrics[f"f1_{cls}"] = float(
            f1_score(yt[:, j], yp[:, j], zero_division=0)
        )

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  Hyperparameter sweep (supplementary)
# ═══════════════════════════════════════════════════════════════════

def _sweep_hyperparams(
    p1: np.ndarray,
    bridge_feats: np.ndarray,
    logodds_feats: np.ndarray,
    ambiguous_mask: np.ndarray,
    gt_main_indices: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float, float, pd.DataFrame]:
    """Find optimal β_B*, β_L*, λ* via grid search.

    Returns (beta_b_star, beta_l_star, lambda_star, sweep_df).
    """
    beta_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    rows: list[dict[str, Any]] = []

    # Method C sweep
    best_c_score = -1.0
    best_bb, best_bl = 0.5, 0.5

    for bb, bl in product(beta_values, beta_values):
        m = _evaluate_condition(
            p1, bridge_feats, logodds_feats, ambiguous_mask,
            gt_main_indices, y_true, bb, bl,
        )
        rows.append({
            "method": "C", "lambda": None,
            "beta_bridge": bb, "beta_logodds": bl,
            **m,
        })
        if m["macro_f1"] > best_c_score:
            best_c_score = m["macro_f1"]
            best_bb, best_bl = bb, bl

    # Method A sweep
    best_a_score = -1.0
    best_lam = 1.0

    for lam in lambda_values:
        m = _evaluate_condition(
            p1, bridge_feats, logodds_feats, ambiguous_mask,
            gt_main_indices, y_true, 0.0, 0.0,
            method="A", lam=lam,
        )
        rows.append({
            "method": "A", "lambda": lam,
            "beta_bridge": None, "beta_logodds": None,
            **m,
        })
        if m["macro_f1"] > best_a_score:
            best_a_score = m["macro_f1"]
            best_lam = lam

    return best_bb, best_bl, best_lam, pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Primary axis: 4-condition component ablation (Method C)
# ═══════════════════════════════════════════════════════════════════

def _run_component_ablation(
    p1: np.ndarray,
    bridge_feats: np.ndarray,
    logodds_feats: np.ndarray,
    ambiguous_mask: np.ndarray,
    gt_main_indices: np.ndarray,
    y_true: np.ndarray,
    beta_b_star: float,
    beta_l_star: float,
) -> pd.DataFrame:
    """Run the 4-condition ablation using Method C.

    All conditions use the same code path (_apply_method_c) with different
    β values. Condition 1 (β_B=0, β_L=0) returns p1 unchanged.
    """
    conditions = [
        ("cond1_no_correction",  0.0,         0.0),
        ("cond2_logodds_only",   0.0,         beta_l_star),
        ("cond3_bridge_only",    beta_b_star,  0.0),
        ("cond4_full_correction", beta_b_star,  beta_l_star),
    ]

    rows: list[dict[str, Any]] = []
    cond1_metrics: dict[str, Any] | None = None

    for cond_name, bb, bl in conditions:
        m = _evaluate_condition(
            p1, bridge_feats, logodds_feats, ambiguous_mask,
            gt_main_indices, y_true, bb, bl,
        )

        row: dict[str, Any] = {
            "condition": cond_name,
            "beta_bridge": bb,
            "beta_logodds": bl,
            **m,
        }

        if cond1_metrics is None:
            cond1_metrics = m
            row["delta_micro_f1_pp"] = 0.0
            row["delta_macro_f1_pp"] = 0.0
        else:
            row["delta_micro_f1_pp"] = round(
                (m["micro_f1"] - cond1_metrics["micro_f1"]) * 100, 2
            )
            row["delta_macro_f1_pp"] = round(
                (m["macro_f1"] - cond1_metrics["macro_f1"]) * 100, 2
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Secondary axis: Method A vs C comparison
# ═══════════════════════════════════════════════════════════════════

def _run_method_comparison(
    p1: np.ndarray,
    bridge_feats: np.ndarray,
    logodds_feats: np.ndarray,
    ambiguous_mask: np.ndarray,
    gt_main_indices: np.ndarray,
    y_true: np.ndarray,
    beta_b_star: float,
    beta_l_star: float,
    lambda_star: float,
) -> pd.DataFrame:
    """Compare Method A vs Method C at full correction settings."""
    rows: list[dict[str, Any]] = []

    # Method A
    m_a = _evaluate_condition(
        p1, bridge_feats, logodds_feats, ambiguous_mask,
        gt_main_indices, y_true, 0.0, 0.0,
        method="A", lam=lambda_star,
    )
    rows.append({
        "method": "A",
        "params": f"λ={lambda_star}",
        **m_a,
    })

    # Method C
    m_c = _evaluate_condition(
        p1, bridge_feats, logodds_feats, ambiguous_mask,
        gt_main_indices, y_true, beta_b_star, beta_l_star,
    )
    rows.append({
        "method": "C",
        "params": f"β_B={beta_b_star}, β_L={beta_l_star}",
        **m_c,
    })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════

def run_ablation(
    corpus: CorpusResult,
    classifier_result: ClassifierResult,
    ambiguous_result: AmbiguousResult,
    logodds_result: LogoddsResult,
    bridge_result: BridgeResult,
    config: OversightConfig,
) -> AblationResult:
    """Run the full structured ablation: sweep → component → method comparison."""
    train_df = corpus.train_df
    if train_df.empty or classifier_result.oof_probs.size == 0:
        return AblationResult()

    p1 = classifier_result.oof_probs
    gt_main_indices = classifier_result.gt_main_indices
    n = len(train_df)

    # Ground truth
    y_cols = [f"y_{c}" for c in FIVE_CLASSES]
    existing = [c for c in y_cols if c in train_df.columns]
    y_true = train_df[existing].values.astype(int)

    # Ambiguous mask
    ambiguous_mask = _build_ambiguous_mask(ambiguous_result, n)

    # Precompute features
    logodds_feats, bridge_feats = _precompute_features(
        corpus, logodds_result, bridge_result
    )

    # Step 1: Hyperparameter sweep
    beta_b_star, beta_l_star, lambda_star, sweep_table = _sweep_hyperparams(
        p1, bridge_feats, logodds_feats, ambiguous_mask,
        gt_main_indices, y_true,
    )

    # Step 2: Component ablation (primary axis)
    component_table = _run_component_ablation(
        p1, bridge_feats, logodds_feats, ambiguous_mask,
        gt_main_indices, y_true,
        beta_b_star, beta_l_star,
    )

    # Step 3: Method comparison (secondary axis)
    method_table = _run_method_comparison(
        p1, bridge_feats, logodds_feats, ambiguous_mask,
        gt_main_indices, y_true,
        beta_b_star, beta_l_star, lambda_star,
    )

    return AblationResult(
        component_table=component_table,
        method_table=method_table,
        sweep_table=sweep_table,
        beta_b_star=beta_b_star,
        beta_l_star=beta_l_star,
        lambda_star=lambda_star,
    )


def save_ablation(result: AblationResult, output_dir: Path) -> None:
    """Save ablation results to separate CSV files."""
    out = output_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    if not result.component_table.empty:
        result.component_table.to_csv(
            out / "ablation_component.csv",
            index=False, encoding="utf-8-sig"
        )

    if not result.method_table.empty:
        result.method_table.to_csv(
            out / "ablation_method_comparison.csv",
            index=False, encoding="utf-8-sig"
        )

    if not result.sweep_table.empty:
        result.sweep_table.to_csv(
            out / "ablation_hyperparam_sweep.csv",
            index=False, encoding="utf-8-sig"
        )
