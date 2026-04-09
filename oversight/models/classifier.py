"""
Stage 6: Primary multi-label classifier.

TF-IDF + OneVsRestClassifier(LogisticRegression) for 5-class multi-label
classification (4 abuse types + "해당없음").

Stratified 5-fold CV produces out-of-fold probability estimates for every
child in the training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

from oversight._imports import ABUSE_ORDER, TFIDF_PARAMS
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult


FIVE_CLASSES = list(ABUSE_ORDER) + ["해당없음"]
Y_COLS = [f"y_{c}" for c in FIVE_CLASSES]


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ClassifierResult:
    """Container for the primary classifier outputs."""

    # Out-of-fold probabilities: shape (N, 5), aligned with train_df rows
    oof_probs: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))

    # Per-fold trained models (list of OneVsRestClassifier)
    fold_models: list[Any] = field(default_factory=list)

    # Per-fold TF-IDF vectorizers
    fold_vectorizers: list[Any] = field(default_factory=list)

    # Fold assignments for each sample (which fold was it in test?)
    fold_assignments: np.ndarray = field(default_factory=lambda: np.array([]))

    # Class names
    classes: list[str] = field(default_factory=lambda: list(FIVE_CLASSES))

    # GT main index per child (index into ABUSE_ORDER, -1 for 해당없음)
    gt_main_indices: np.ndarray = field(default_factory=lambda: np.array([]))


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF vectorizer with auto-relaxation
# ═══════════════════════════════════════════════════════════════════

def _safe_fit_vectorizer(train_texts: list[str]) -> tuple[TfidfVectorizer, Any]:
    """Fit TF-IDF vectorizer with automatic min_df relaxation."""
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    try:
        X = vec.fit_transform(train_texts)
    except ValueError as e:
        if "no terms remain" not in str(e):
            raise
        relaxed = dict(TFIDF_PARAMS)
        relaxed["min_df"] = 1
        vec = TfidfVectorizer(**relaxed)
        X = vec.fit_transform(train_texts)
    return vec, X


# ═══════════════════════════════════════════════════════════════════
#  Stage 6: Train primary classifier
# ═══════════════════════════════════════════════════════════════════

def _build_stratification_labels(train_df: pd.DataFrame) -> np.ndarray:
    """Build single-label stratification column for StratifiedKFold.

    Uses gt_main for stratification (not the multi-label vector).
    """
    return train_df["gt_main"].values


def _build_y_matrix(train_df: pd.DataFrame) -> np.ndarray:
    """Build the multi-label binary matrix (N, 5)."""
    y_cols = [c for c in Y_COLS if c in train_df.columns]
    return train_df[y_cols].values.astype(int)


def _gt_main_to_index(gt_main_series: pd.Series) -> np.ndarray:
    """Map gt_main string to index (0-3 for abuse types, 4 for 해당없음)."""
    type_to_idx = {a: i for i, a in enumerate(FIVE_CLASSES)}
    return np.array([type_to_idx.get(v, 4) for v in gt_main_series])


def train_primary_classifier(
    corpus: CorpusResult,
    config: OversightConfig,
) -> ClassifierResult:
    """Stage 6: Train 5-class multi-label classifier with stratified 5-fold CV.

    Produces out-of-fold probability estimates for every child.
    """
    train_df = corpus.train_df
    if train_df.empty:
        return ClassifierResult()

    texts = train_df["text"].tolist()
    y_matrix = _build_y_matrix(train_df)
    strat_labels = _build_stratification_labels(train_df)
    n_samples = len(train_df)

    oof_probs = np.zeros((n_samples, len(FIVE_CLASSES)), dtype=float)
    fold_assignments = np.full(n_samples, -1, dtype=int)
    fold_models = []
    fold_vectorizers = []

    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, strat_labels)):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = y_matrix[train_idx]

        # TF-IDF
        vec, X_train = _safe_fit_vectorizer(train_texts)
        X_test = vec.transform(test_texts)

        # OneVsRest(LogisticRegression) — uses predict_proba
        if config.classifier_type == "lr":
            base_clf = LogisticRegression(
                solver="lbfgs", max_iter=300, random_state=config.random_state
            )
        else:
            # SVM fallback — use LR anyway since we need predict_proba
            base_clf = LogisticRegression(
                solver="lbfgs", max_iter=300, random_state=config.random_state
            )

        ovr = OneVsRestClassifier(base_clf)
        ovr.fit(X_train, y_train)

        # Out-of-fold predictions
        probs = ovr.predict_proba(X_test)

        # Handle case where some classes might be missing
        if probs.shape[1] < len(FIVE_CLASSES):
            full_probs = np.zeros((len(test_idx), len(FIVE_CLASSES)))
            for j, cls in enumerate(ovr.classes_):
                full_probs[:, cls] = probs[:, j]
            probs = full_probs

        oof_probs[test_idx] = probs
        fold_assignments[test_idx] = fold_idx
        fold_models.append(ovr)
        fold_vectorizers.append(vec)

    gt_main_indices = _gt_main_to_index(train_df["gt_main"])

    return ClassifierResult(
        oof_probs=oof_probs,
        fold_models=fold_models,
        fold_vectorizers=fold_vectorizers,
        fold_assignments=fold_assignments,
        classes=list(FIVE_CLASSES),
        gt_main_indices=gt_main_indices,
    )


def save_classifier_results(
    result: ClassifierResult,
    train_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save classifier out-of-fold probabilities."""
    out = output_dir / "classifier"
    out.mkdir(parents=True, exist_ok=True)

    if result.oof_probs.size == 0:
        return

    # OOF probabilities
    probs_df = pd.DataFrame(
        result.oof_probs,
        columns=[f"p_{c}" for c in result.classes],
    )
    if "doc_id" in train_df.columns:
        probs_df.insert(0, "doc_id", train_df["doc_id"].values)
    if "gt_main" in train_df.columns:
        probs_df.insert(1, "gt_main", train_df["gt_main"].values)

    probs_df.to_csv(
        out / "oof_probabilities.csv", index=False, encoding="utf-8-sig"
    )
