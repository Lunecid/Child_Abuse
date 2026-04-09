"""
Stage 7: Ambiguous case identification.

A child is flagged as "ambiguous" if:
    Condition 1 (bridge frequency): correction bridge word count exceeds threshold
    OR
    Condition 2 (prediction instability):
        2a: K-fold instability — re-run CV with different seeds, measure variance
        OR
        2b: Bootstrap instability — bootstrap resample, predict OOB, measure variance

Final ambiguous = Condition 1 OR Condition 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

from oversight._imports import ABUSE_ORDER, TFIDF_PARAMS
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult
from oversight.models.classifier import (
    ClassifierResult,
    FIVE_CLASSES,
    Y_COLS,
    _safe_fit_vectorizer,
    _build_stratification_labels,
    _build_y_matrix,
)
from oversight.stats.bridge import BridgeResult


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AmbiguousResult:
    """Container for ambiguous case identification."""

    # Set of ambiguous doc indices (row indices into train_df)
    ambiguous_indices: set[int] = field(default_factory=set)

    # Per-child flags DataFrame
    flags_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════
#  Condition 1: Bridge word frequency
# ═══════════════════════════════════════════════════════════════════

def _check_bridge_frequency(
    corpus: CorpusResult,
    bridge_result: BridgeResult,
    config: OversightConfig,
) -> np.ndarray:
    """Count correction bridge words per child. Return boolean mask."""
    train_df = corpus.train_df
    if train_df.empty or bridge_result.bridge_correction.empty:
        return np.zeros(len(train_df), dtype=bool)

    bridge_words = set(bridge_result.bridge_correction["word"])
    threshold = config.ambiguous_bridge_threshold
    if threshold is None:
        # Default: flag if child has >= 3 bridge words
        threshold = 3.0

    counts = np.zeros(len(train_df), dtype=float)
    for i, text in enumerate(train_df["text"]):
        if isinstance(text, str):
            tokens = set(text.split())
            counts[i] = len(tokens & bridge_words)

    return counts >= threshold


# ═══════════════════════════════════════════════════════════════════
#  Condition 2a: K-fold instability
# ═══════════════════════════════════════════════════════════════════

def _check_kfold_instability(
    corpus: CorpusResult,
    config: OversightConfig,
) -> np.ndarray:
    """Re-run CV with different seeds, measure prediction variance."""
    train_df = corpus.train_df
    if train_df.empty:
        return np.zeros(len(train_df), dtype=bool)

    seeds = config.kfold_repeat_seeds
    if seeds is None:
        seeds = [42, 123, 456]

    texts = train_df["text"].tolist()
    y_matrix = _build_y_matrix(train_df)
    strat_labels = _build_stratification_labels(train_df)
    n_samples = len(train_df)

    # Collect OOF predictions across seeds
    all_preds = np.zeros((len(seeds), n_samples, len(FIVE_CLASSES)), dtype=float)

    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(
            n_splits=config.n_splits, shuffle=True, random_state=seed
        )
        oof = np.zeros((n_samples, len(FIVE_CLASSES)), dtype=float)

        for train_idx, test_idx in skf.split(texts, strat_labels):
            train_texts = [texts[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            y_train = y_matrix[train_idx]

            vec, X_train = _safe_fit_vectorizer(train_texts)
            X_test = vec.transform(test_texts)

            ovr = OneVsRestClassifier(
                LogisticRegression(
                    solver="lbfgs", max_iter=300, random_state=seed
                )
            )
            ovr.fit(X_train, y_train)
            probs = ovr.predict_proba(X_test)

            if probs.shape[1] < len(FIVE_CLASSES):
                full_probs = np.zeros((len(test_idx), len(FIVE_CLASSES)))
                for j, cls in enumerate(ovr.classes_):
                    full_probs[:, cls] = probs[:, j]
                probs = full_probs

            oof[test_idx] = probs

        all_preds[s_idx] = oof

    # Variance across seeds per child (mean across classes)
    variance = all_preds.var(axis=0).mean(axis=1)

    threshold = config.instability_threshold
    if threshold is None:
        # Default: flag if variance > median + 1 * IQR
        q75 = np.percentile(variance, 75)
        q25 = np.percentile(variance, 25)
        iqr = q75 - q25
        threshold = np.median(variance) + iqr

    return variance > threshold


# ═══════════════════════════════════════════════════════════════════
#  Condition 2b: Bootstrap instability
# ═══════════════════════════════════════════════════════════════════

def _check_bootstrap_instability(
    corpus: CorpusResult,
    config: OversightConfig,
) -> np.ndarray:
    """Bootstrap resample training data, predict OOB, measure variance."""
    train_df = corpus.train_df
    if train_df.empty:
        return np.zeros(len(train_df), dtype=bool)

    n_iter = config.bootstrap_n_iter
    if n_iter is None:
        n_iter = 50

    texts = train_df["text"].tolist()
    y_matrix = _build_y_matrix(train_df)
    n_samples = len(train_df)

    # Accumulate OOB predictions
    oob_sum = np.zeros((n_samples, len(FIVE_CLASSES)), dtype=float)
    oob_sq_sum = np.zeros((n_samples, len(FIVE_CLASSES)), dtype=float)
    oob_count = np.zeros(n_samples, dtype=int)

    rng = np.random.RandomState(config.random_state)

    for _ in range(n_iter):
        # Bootstrap sample
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) == 0:
            continue

        train_texts = [texts[i] for i in boot_idx]
        oob_texts = [texts[i] for i in oob_idx]
        y_train = y_matrix[boot_idx]

        vec, X_train = _safe_fit_vectorizer(train_texts)
        X_oob = vec.transform(oob_texts)

        ovr = OneVsRestClassifier(
            LogisticRegression(
                solver="lbfgs", max_iter=300, random_state=config.random_state
            )
        )
        ovr.fit(X_train, y_train)
        probs = ovr.predict_proba(X_oob)

        if probs.shape[1] < len(FIVE_CLASSES):
            full_probs = np.zeros((len(oob_idx), len(FIVE_CLASSES)))
            for j, cls in enumerate(ovr.classes_):
                full_probs[:, cls] = probs[:, j]
            probs = full_probs

        oob_sum[oob_idx] += probs
        oob_sq_sum[oob_idx] += probs ** 2
        oob_count[oob_idx] += 1

    # Variance per child
    valid = oob_count >= 2
    variance = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        if valid[i]:
            mean = oob_sum[i] / oob_count[i]
            var = oob_sq_sum[i] / oob_count[i] - mean ** 2
            variance[i] = var.mean()  # Mean across classes

    threshold = config.instability_threshold
    if threshold is None:
        valid_vars = variance[valid]
        if len(valid_vars) > 0:
            q75 = np.percentile(valid_vars, 75)
            q25 = np.percentile(valid_vars, 25)
            iqr = q75 - q25
            threshold = np.median(valid_vars) + iqr
        else:
            threshold = 0.0

    return variance > threshold


# ═══════════════════════════════════════════════════════════════════
#  Stage 7: Main entry point
# ═══════════════════════════════════════════════════════════════════

def identify_ambiguous_cases(
    corpus: CorpusResult,
    classifier_result: ClassifierResult,
    bridge_result: BridgeResult,
    config: OversightConfig,
) -> AmbiguousResult:
    """Stage 7: Identify ambiguous cases.

    Condition 1: bridge word frequency above threshold
    Condition 2: k-fold instability (2a) OR bootstrap instability (2b)
    Final: Condition 1 OR Condition 2
    """
    train_df = corpus.train_df
    n = len(train_df)

    # Condition 1
    cond1 = _check_bridge_frequency(corpus, bridge_result, config)

    # Condition 2a
    cond2a = _check_kfold_instability(corpus, config)

    # Condition 2b
    cond2b = _check_bootstrap_instability(corpus, config)

    # Condition 2 = 2a OR 2b
    cond2 = cond2a | cond2b

    # Final = Condition 1 OR Condition 2
    ambiguous = cond1 | cond2

    # Build flags DataFrame
    flags_df = pd.DataFrame({
        "doc_id": train_df["doc_id"].values if "doc_id" in train_df.columns else range(n),
        "gt_main": train_df["gt_main"].values if "gt_main" in train_df.columns else "",
        "cond1_bridge": cond1,
        "cond2a_kfold": cond2a,
        "cond2b_bootstrap": cond2b,
        "cond2_instability": cond2,
        "ambiguous": ambiguous,
    })

    return AmbiguousResult(
        ambiguous_indices=set(np.where(ambiguous)[0]),
        flags_df=flags_df,
    )


def save_ambiguous_results(result: AmbiguousResult, output_dir: Path) -> None:
    """Save ambiguous case identification results."""
    out = output_dir / "ambiguous"
    out.mkdir(parents=True, exist_ok=True)

    if not result.flags_df.empty:
        result.flags_df.to_csv(
            out / "ambiguous_flags.csv", index=False, encoding="utf-8-sig"
        )

    summary = {
        "total": len(result.flags_df),
        "ambiguous_count": int(result.flags_df["ambiguous"].sum()) if not result.flags_df.empty else 0,
        "cond1_count": int(result.flags_df["cond1_bridge"].sum()) if not result.flags_df.empty else 0,
        "cond2a_count": int(result.flags_df["cond2a_kfold"].sum()) if not result.flags_df.empty else 0,
        "cond2b_count": int(result.flags_df["cond2b_bootstrap"].sum()) if not result.flags_df.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(
        out / "ambiguous_summary.csv", index=False, encoding="utf-8-sig"
    )
