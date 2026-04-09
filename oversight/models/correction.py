"""
Stage 8: Correction layers (Method A + Method C).

Applied ONLY to ambiguous cases. Clear cases retain original p1 probabilities.

Method A (log-linear):
    p2(c|child) ∝ p1(c|child) · exp(λ · mean_logodds(c, child))
    "해당없음" class: boost = 0 (no correction)
    Normalize to sum = 1.

Method C (separated):
    p2(c) = normalize(p1(c) + β_B · bridge_boost(c) + β_L · logodds_boost(c))
    bridge_boost(c) = mean P(c|w) for bridge words in child's tokens
    logodds_boost(c) = mean log-odds of child's tokens for class c
    "해당없음": both boosts = 0
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
from oversight.stats.logodds import LogoddsResult
from oversight.stats.bridge import BridgeResult


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CorrectionResult:
    """Container for corrected probability matrices."""

    # Corrected probabilities: shape (N, 5)
    probs_method_a: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    probs_method_c: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))


# ═══════════════════════════════════════════════════════════════════
#  Helper: per-child log-odds feature
# ═══════════════════════════════════════════════════════════════════

def _compute_child_logodds_features(
    texts: list[str],
    logodds_df: pd.DataFrame,
    classes: list[str],
) -> np.ndarray:
    """Compute mean log-odds per class for each child's tokens.

    Returns: (N, len(classes)) array.
    """
    n = len(texts)
    n_classes = len(classes)
    features = np.zeros((n, n_classes), dtype=float)

    if logodds_df.empty:
        return features

    # Build word → {class: log_odds} lookup
    word_logodds: dict[str, dict[str, float]] = {}
    for _, row in logodds_df.iterrows():
        w = row["word"]
        g = row["group"]
        lo = row["log_odds"]
        if g in classes:
            if w not in word_logodds:
                word_logodds[w] = {}
            word_logodds[w][g] = lo

    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = text.split()
        if not tokens:
            continue

        sums = np.zeros(n_classes, dtype=float)
        counts = np.zeros(n_classes, dtype=float)
        for token in tokens:
            if token in word_logodds:
                for j, cls in enumerate(classes):
                    if cls in word_logodds[token]:
                        sums[j] += word_logodds[token][cls]
                        counts[j] += 1

        for j in range(n_classes):
            if counts[j] > 0:
                features[i, j] = sums[j] / counts[j]

    return features


def _compute_child_bridge_features(
    texts: list[str],
    bridge_df: pd.DataFrame,
    doc_counts: pd.DataFrame,
    classes: list[str],
) -> np.ndarray:
    """Compute bridge word boost per class for each child.

    bridge_boost(c) = mean P(c|w) for bridge words in child's tokens.

    Returns: (N, len(classes)) array.
    """
    n = len(texts)
    n_classes = len(classes)
    features = np.zeros((n, n_classes), dtype=float)

    if bridge_df.empty or doc_counts.empty:
        return features

    bridge_words = set(bridge_df["word"])

    # Precompute P(c|w) for bridge words
    bridge_probs: dict[str, np.ndarray] = {}
    for w in bridge_words:
        if w not in doc_counts.index:
            continue
        row = doc_counts.loc[w]
        vals = np.array([row.get(c, 0) for c in classes], dtype=float)
        total = vals.sum()
        if total > 0:
            bridge_probs[w] = vals / total

    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = set(text.split())
        matched = tokens & set(bridge_probs.keys())
        if matched:
            prob_sum = np.zeros(n_classes, dtype=float)
            for w in matched:
                prob_sum += bridge_probs[w]
            features[i] = prob_sum / len(matched)

    return features


# ═══════════════════════════════════════════════════════════════════
#  Method A: Log-linear correction
# ═══════════════════════════════════════════════════════════════════

def _apply_method_a(
    p1: np.ndarray,
    logodds_features: np.ndarray,
    ambiguous_mask: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Apply log-linear correction to ambiguous cases.

    p2(c) ∝ p1(c) · exp(λ · logodds_feature(c))
    "해당없음" (index 4): boost = 0
    """
    p2 = p1.copy()

    for i in range(len(p1)):
        if not ambiguous_mask[i]:
            continue

        boost = np.zeros(p1.shape[1], dtype=float)
        # Only apply to abuse classes (index 0-3)
        for j in range(min(4, p1.shape[1])):
            boost[j] = lam * logodds_features[i, j]
        # "해당없음" (index 4): no boost

        log_p = np.log(np.clip(p1[i], 1e-10, None)) + boost
        log_p -= log_p.max()  # numerical stability
        p2[i] = np.exp(log_p)
        total = p2[i].sum()
        if total > 0:
            p2[i] /= total

    return p2


# ═══════════════════════════════════════════════════════════════════
#  Method C: Separated correction
# ═══════════════════════════════════════════════════════════════════

def _apply_method_c(
    p1: np.ndarray,
    bridge_features: np.ndarray,
    logodds_features: np.ndarray,
    ambiguous_mask: np.ndarray,
    beta_bridge: float,
    beta_logodds: float,
) -> np.ndarray:
    """Apply separated correction to ambiguous cases.

    p2(c) = normalize(p1(c) + β_B · bridge_boost(c) + β_L · logodds_boost(c))
    "해당없음" (index 4): both boosts = 0
    """
    p2 = p1.copy()

    for i in range(len(p1)):
        if not ambiguous_mask[i]:
            continue

        corrected = p1[i].copy()
        for j in range(min(4, p1.shape[1])):
            corrected[j] += (
                beta_bridge * bridge_features[i, j]
                + beta_logodds * logodds_features[i, j]
            )
        # "해당없음" (index 4): no correction

        corrected = np.clip(corrected, 0, None)
        total = corrected.sum()
        if total > 0:
            corrected /= total
        p2[i] = corrected

    return p2


# ═══════════════════════════════════════════════════════════════════
#  Stage 8: Main entry point
# ═══════════════════════════════════════════════════════════════════

def apply_corrections(
    corpus: CorpusResult,
    classifier_result: ClassifierResult,
    ambiguous_result: AmbiguousResult,
    logodds_result: LogoddsResult,
    bridge_result: BridgeResult,
    config: OversightConfig,
) -> CorrectionResult:
    """Stage 8: Apply Method A and Method C corrections to ambiguous cases."""
    train_df = corpus.train_df
    if train_df.empty or classifier_result.oof_probs.size == 0:
        return CorrectionResult()

    p1 = classifier_result.oof_probs
    texts = train_df["text"].tolist()

    # Ambiguous mask
    n = len(train_df)
    ambiguous_mask = np.zeros(n, dtype=bool)
    for idx in ambiguous_result.ambiguous_indices:
        if idx < n:
            ambiguous_mask[idx] = True

    # Use training-data log-odds (5-class) for correction features
    # But restrict features to the 4 abuse classes only
    abuse_classes = list(ABUSE_ORDER)
    logodds_df = logodds_result.logodds_train_5class
    doc_counts = logodds_result.doc_counts_train

    # Compute per-child features (for abuse classes only, pad with 0 for 해당없음)
    logodds_feats_4 = _compute_child_logodds_features(texts, logodds_df, abuse_classes)
    logodds_feats = np.zeros((n, len(FIVE_CLASSES)), dtype=float)
    logodds_feats[:, :4] = logodds_feats_4

    bridge_feats_4 = _compute_child_bridge_features(
        texts, bridge_result.bridge_correction, doc_counts, abuse_classes
    )
    bridge_feats = np.zeros((n, len(FIVE_CLASSES)), dtype=float)
    bridge_feats[:, :4] = bridge_feats_4

    # Method A
    lam = config.correction_lambda if config.correction_lambda is not None else 1.0
    probs_a = _apply_method_a(p1, logodds_feats, ambiguous_mask, lam)

    # Method C
    beta_b = config.correction_beta_bridge if config.correction_beta_bridge is not None else 0.5
    beta_l = config.correction_beta_logodds if config.correction_beta_logodds is not None else 0.5
    probs_c = _apply_method_c(p1, bridge_feats, logodds_feats, ambiguous_mask, beta_b, beta_l)

    return CorrectionResult(
        probs_method_a=probs_a,
        probs_method_c=probs_c,
    )


def save_correction_results(
    result: CorrectionResult,
    train_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save corrected probabilities."""
    out = output_dir / "correction"
    out.mkdir(parents=True, exist_ok=True)

    for name, probs in [
        ("corrected_probs_method_a", result.probs_method_a),
        ("corrected_probs_method_c", result.probs_method_c),
    ]:
        if probs.size == 0:
            continue
        df = pd.DataFrame(probs, columns=[f"p_{c}" for c in FIVE_CLASSES])
        if "doc_id" in train_df.columns:
            df.insert(0, "doc_id", train_df["doc_id"].values)
        if "gt_main" in train_df.columns:
            df.insert(1, "gt_main", train_df["gt_main"].values)
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
