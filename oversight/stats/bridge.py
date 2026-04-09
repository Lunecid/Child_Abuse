"""
Stage 4: Bridge word selection (diagnostic + correction sets).

Bridge words are tokens that span multiple abuse types with similar conditional
probability P(k|w). Two sets are produced:

    Diagnostic set:  From ABUSE_NEG doc-counts (used in Part 1 diagnosis)
    Correction set:  From training-data doc-counts (used in Part 2 correction)

The overlap between the two sets is also recorded.

All heavy lifting is delegated to:
    - abuse_pipeline.stats.stats.compute_chi_square
    - abuse_pipeline.stats.stats.compute_prob_bridge_for_words
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from oversight._imports import (
    ABUSE_ORDER,
    BRIDGE_MIN_P1,
    BRIDGE_MIN_P2,
    BRIDGE_MAX_GAP,
    BRIDGE_MIN_COUNT,
    compute_chi_square,
    compute_prob_bridge_for_words,
)
from oversight.config import OversightConfig
from oversight.stats.logodds import LogoddsResult


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BridgeResult:
    """Container for bridge word selection results."""

    # Bridge word DataFrames
    # Columns: word, primary_abuse, secondary_abuse, p1, p2, gap, source
    bridge_diagnostic: pd.DataFrame = field(default_factory=pd.DataFrame)
    bridge_correction: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Chi-square DataFrames (index=word, columns: chi2, p_value)
    chi_diagnostic: pd.DataFrame = field(default_factory=pd.DataFrame)
    chi_correction: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Overlap statistics
    overlap_count: int = 0
    overlap_ratio: float = 0.0
    overlap_words: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  Stage 4: Bridge word selection
# ═══════════════════════════════════════════════════════════════════

def _select_bridge_from_counts(
    doc_counts: pd.DataFrame,
    logodds_df: pd.DataFrame,
    config: OversightConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run chi-square → top-k → bridge identification on a doc-count table.

    Returns (chi_df, bridge_df).
    """
    cols = list(ABUSE_ORDER)

    # Only keep abuse-type columns that exist
    available = [c for c in cols if c in doc_counts.columns]
    if not available or doc_counts.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Chi-square test
    chi_df = compute_chi_square(doc_counts, available)
    if chi_df.empty:
        return chi_df, pd.DataFrame()

    # Top-k words by chi-square
    top_k = config.chi_top_k
    chi_sorted = chi_df.sort_values("chi2", ascending=False, kind="mergesort")
    top_words = chi_sorted.head(top_k).index.tolist()

    # Bridge word identification
    min_p1 = config.bridge_min_p1 if config.bridge_min_p1 is not None else BRIDGE_MIN_P1
    min_p2 = config.bridge_min_p2 if config.bridge_min_p2 is not None else BRIDGE_MIN_P2
    max_gap = config.bridge_max_gap if config.bridge_max_gap is not None else BRIDGE_MAX_GAP
    count_min = config.bridge_count_min

    bridge_df = compute_prob_bridge_for_words(
        doc_counts,
        top_words,
        logodds_df=logodds_df,
        min_p1=min_p1,
        min_p2=min_p2,
        max_gap=max_gap,
        count_min=count_min,
    )

    return chi_df, bridge_df


def select_bridge_words(
    logodds_result: LogoddsResult,
    config: OversightConfig,
) -> BridgeResult:
    """Stage 4: Select diagnostic and correction bridge word sets.

    Diagnostic set: from ABUSE_NEG doc-counts + logodds_abuse_neg_4class
    Correction set: from training-data doc-counts + logodds_train_5class
    """
    result = BridgeResult()

    # Diagnostic bridge words (from ABUSE_NEG corpus)
    if (not logodds_result.doc_counts_abuse_neg.empty
            and not logodds_result.logodds_abuse_neg_4class.empty):
        result.chi_diagnostic, result.bridge_diagnostic = _select_bridge_from_counts(
            logodds_result.doc_counts_abuse_neg,
            logodds_result.logodds_abuse_neg_4class,
            config,
        )

    # Correction bridge words (from training data)
    # Use only the abuse-type columns from training doc-counts
    if (not logodds_result.doc_counts_train.empty
            and not logodds_result.logodds_train_5class.empty):
        # Filter training doc-counts to abuse types only for bridge identification
        train_counts = logodds_result.doc_counts_train.copy()
        abuse_cols = [c for c in ABUSE_ORDER if c in train_counts.columns]
        if abuse_cols:
            train_counts_abuse = train_counts[abuse_cols].copy()
            # Filter to words that appear in at least some abuse-type docs
            train_counts_abuse = train_counts_abuse[
                train_counts_abuse.sum(axis=1) > 0
            ]
            result.chi_correction, result.bridge_correction = _select_bridge_from_counts(
                train_counts_abuse,
                logodds_result.logodds_train_5class,
                config,
            )

    # Compute overlap
    if not result.bridge_diagnostic.empty and not result.bridge_correction.empty:
        diag_words = set(result.bridge_diagnostic["word"])
        corr_words = set(result.bridge_correction["word"])
        overlap = diag_words & corr_words
        union = diag_words | corr_words
        result.overlap_words = sorted(overlap)
        result.overlap_count = len(overlap)
        result.overlap_ratio = len(overlap) / len(union) if union else 0.0

    return result


def save_bridge(result: BridgeResult, output_dir: Path) -> None:
    """Save bridge word results to CSV."""
    out = output_dir / "bridge"
    out.mkdir(parents=True, exist_ok=True)

    if not result.bridge_diagnostic.empty:
        result.bridge_diagnostic.to_csv(
            out / "bridge_diagnostic.csv", index=False, encoding="utf-8-sig"
        )
    if not result.bridge_correction.empty:
        result.bridge_correction.to_csv(
            out / "bridge_correction.csv", index=False, encoding="utf-8-sig"
        )
    if not result.chi_diagnostic.empty:
        result.chi_diagnostic.to_csv(
            out / "chi_diagnostic.csv", encoding="utf-8-sig"
        )
    if not result.chi_correction.empty:
        result.chi_correction.to_csv(
            out / "chi_correction.csv", encoding="utf-8-sig"
        )

    # Overlap summary
    overlap_summary = {
        "diagnostic_count": len(result.bridge_diagnostic),
        "correction_count": len(result.bridge_correction),
        "overlap_count": result.overlap_count,
        "overlap_ratio": result.overlap_ratio,
        "overlap_words": "|".join(result.overlap_words),
    }
    pd.DataFrame([overlap_summary]).to_csv(
        out / "bridge_overlap.csv", index=False, encoding="utf-8-sig"
    )
