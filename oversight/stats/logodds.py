"""
Stage 3: Log-odds computation for 4 sub-corpora.

Computes stabilized log-odds ratios (Dirichlet smoothing, alpha=0.01) on
document-level frequency tables for each of the following sub-corpora:

    1. ALL corpus, 5 classes (4 abuse + 해당없음)
    2. NEG corpus, 4 abuse classes
    3. ABUSE_NEG corpus, 4 abuse classes
    4. Training data, 5 classes (4 abuse + 해당없음)

All heavy lifting is delegated to abuse_pipeline.stats.stats.compute_log_odds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from oversight._imports import (
    ABUSE_ORDER,
    MIN_DOC_COUNT,
    compute_log_odds,
)
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult


FIVE_CLASSES = list(ABUSE_ORDER) + ["해당없음"]


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LogoddsResult:
    """Container for the 4 log-odds DataFrames."""

    # Each DataFrame has columns: word, group, count, log_odds, se_log_odds, z_log_odds
    logodds_all_5class: pd.DataFrame = field(default_factory=pd.DataFrame)
    logodds_neg_4class: pd.DataFrame = field(default_factory=pd.DataFrame)
    logodds_abuse_neg_4class: pd.DataFrame = field(default_factory=pd.DataFrame)
    logodds_train_5class: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Document-level frequency tables (word × class pivot tables)
    doc_counts_all: pd.DataFrame = field(default_factory=pd.DataFrame)
    doc_counts_neg: pd.DataFrame = field(default_factory=pd.DataFrame)
    doc_counts_abuse_neg: pd.DataFrame = field(default_factory=pd.DataFrame)
    doc_counts_train: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════
#  Doc-level frequency table builders
# ═══════════════════════════════════════════════════════════════════

def _build_doc_word_counts(
    df: pd.DataFrame,
    class_col: str,
    classes: list[str],
    min_doc_count: int = MIN_DOC_COUNT,
) -> pd.DataFrame:
    """Build a word × class document-level frequency table from a DataFrame.

    Each row in ``df`` represents one child. The 'text' column contains
    space-separated tokens. A word is counted once per child (set-based).

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'text' and ``class_col`` columns.
    class_col : str
        Column name for the class label (e.g. 'gt_main').
    classes : list[str]
        Expected class labels (ensures all columns exist).
    min_doc_count : int
        Drop words appearing in fewer than this many documents.

    Returns
    -------
    pd.DataFrame
        Index = word, columns = classes. Values = document counts.
    """
    rows = []
    for _, record in df.iterrows():
        label = record[class_col]
        text = record.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        # Set-based: count each word once per document
        tokens = set(text.split())
        for w in tokens:
            rows.append({"word": w, "group": label})

    if not rows:
        return pd.DataFrame(columns=classes)

    doc_df = pd.DataFrame(rows)
    counts = (
        doc_df
        .groupby(["word", "group"])
        .size()
        .unstack("group")
        .fillna(0)
    )

    # Ensure all classes are present as columns
    for c in classes:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[classes]

    # Filter by minimum document count
    counts["_total"] = counts.sum(axis=1)
    counts = counts[counts["_total"] >= min_doc_count]
    counts = counts.drop(columns=["_total"])

    return counts


# ═══════════════════════════════════════════════════════════════════
#  Stage 3: Compute all log-odds
# ═══════════════════════════════════════════════════════════════════

def compute_all_logodds(
    corpus: CorpusResult,
    config: OversightConfig,
) -> LogoddsResult:
    """Stage 3: Compute stabilized log-odds for 4 sub-corpora.

    Sub-corpus 1 (ALL, 5-class): All children in train_df
        → 4 abuse types + "해당없음"

    Sub-corpus 2 (NEG, 4-class): Only 부정 (negative) children from abuse_neg_df
        → 4 abuse types (by gt_main)

    Sub-corpus 3 (ABUSE_NEG, 4-class): Same as NEG but filtered to children
        with gt_main in ABUSE_ORDER (should be identical in practice since
        abuse_neg_df already filters for this)

    Sub-corpus 4 (Training, 5-class): Same as sub-corpus 1, but kept as a
        separate computation for clarity and potential future differences.
    """
    alpha = config.logodds_alpha
    min_doc = config.min_doc_count

    result = LogoddsResult()

    # Sub-corpus 1: ALL, 5-class (training data = ABUSE_NEG + POS/NEU)
    if not corpus.train_df.empty:
        result.doc_counts_all = _build_doc_word_counts(
            corpus.train_df, "gt_main", FIVE_CLASSES, min_doc
        )
        if not result.doc_counts_all.empty:
            result.logodds_all_5class = compute_log_odds(
                result.doc_counts_all, FIVE_CLASSES, alpha=alpha
            )

    # Sub-corpus 2: NEG, 4-class (abuse_neg_df only)
    if not corpus.abuse_neg_df.empty:
        result.doc_counts_neg = _build_doc_word_counts(
            corpus.abuse_neg_df, "gt_main", list(ABUSE_ORDER), min_doc
        )
        if not result.doc_counts_neg.empty:
            result.logodds_neg_4class = compute_log_odds(
                result.doc_counts_neg, list(ABUSE_ORDER), alpha=alpha
            )

    # Sub-corpus 3: ABUSE_NEG, 4-class (same data, explicit filter)
    if not corpus.abuse_neg_df.empty:
        abuse_neg_filtered = corpus.abuse_neg_df[
            corpus.abuse_neg_df["gt_main"].isin(ABUSE_ORDER)
        ]
        result.doc_counts_abuse_neg = _build_doc_word_counts(
            abuse_neg_filtered, "gt_main", list(ABUSE_ORDER), min_doc
        )
        if not result.doc_counts_abuse_neg.empty:
            result.logodds_abuse_neg_4class = compute_log_odds(
                result.doc_counts_abuse_neg, list(ABUSE_ORDER), alpha=alpha
            )

    # Sub-corpus 4: Training, 5-class
    if not corpus.train_df.empty:
        result.doc_counts_train = _build_doc_word_counts(
            corpus.train_df, "gt_main", FIVE_CLASSES, min_doc
        )
        if not result.doc_counts_train.empty:
            result.logodds_train_5class = compute_log_odds(
                result.doc_counts_train, FIVE_CLASSES, alpha=alpha
            )

    return result


def save_logodds(result: LogoddsResult, output_dir: Path) -> None:
    """Save all log-odds DataFrames to CSV."""
    out = output_dir / "logodds"
    out.mkdir(parents=True, exist_ok=True)

    for name, df in [
        ("logodds_all_5class", result.logodds_all_5class),
        ("logodds_neg_4class", result.logodds_neg_4class),
        ("logodds_abuse_neg_4class", result.logodds_abuse_neg_4class),
        ("logodds_train_5class", result.logodds_train_5class),
    ]:
        if not df.empty:
            df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    for name, df in [
        ("doc_counts_all", result.doc_counts_all),
        ("doc_counts_neg", result.doc_counts_neg),
        ("doc_counts_abuse_neg", result.doc_counts_abuse_neg),
        ("doc_counts_train", result.doc_counts_train),
    ]:
        if not df.empty:
            df.to_csv(out / f"{name}.csv", encoding="utf-8-sig")
