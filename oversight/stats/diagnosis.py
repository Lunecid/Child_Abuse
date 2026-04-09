"""
Stage 5: Part 1 diagnostics.

Produces the diagnostic analyses for Part 1 of the paper:
    1. Information loss quantification (how many labels are lost by single-label)
    2. Label composition (Group A / B / C breakdown)
    3. Correspondence analysis (abuse-type × word biplot) — optional (prince)
    4. Bridge word stability (bootstrap resample)

Heavy lifting delegated to:
    - abuse_pipeline.experiments.information_recovery.quantify_information_loss
    - abuse_pipeline.experiments.information_recovery.compute_label_composition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oversight._imports import (
    ABUSE_ORDER,
    compute_label_composition,
    quantify_information_loss,
    compute_prob_bridge_for_words,
    compute_chi_square,
    BRIDGE_MIN_P1,
    BRIDGE_MIN_P2,
    BRIDGE_MAX_GAP,
    BRIDGE_MIN_COUNT,
)
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult
from oversight.stats.logodds import LogoddsResult
from oversight.stats.bridge import BridgeResult


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DiagnosisResult:
    """Container for all Part 1 diagnostic outputs."""

    # Information loss
    info_loss: dict[str, Any] = field(default_factory=dict)

    # Label composition (Group A / B / C)
    label_composition: dict[str, Any] = field(default_factory=dict)

    # Correspondence analysis (optional)
    ca_coords: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Bridge stability (bootstrap)
    bridge_stability: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════
#  Stage 5.1: Information loss
# ═══════════════════════════════════════════════════════════════════

def _run_information_loss(corpus: CorpusResult) -> dict[str, Any]:
    """Quantify information lost by single-label recording."""
    if corpus.abuse_neg_df.empty:
        return {}
    return quantify_information_loss(corpus.abuse_neg_df)


# ═══════════════════════════════════════════════════════════════════
#  Stage 5.2: Label composition
# ═══════════════════════════════════════════════════════════════════

def _run_label_composition(corpus: CorpusResult) -> dict[str, Any]:
    """Compute Group A/B/C composition."""
    if corpus.abuse_neg_df.empty:
        return {}
    return compute_label_composition(corpus.abuse_neg_df)


# ═══════════════════════════════════════════════════════════════════
#  Stage 5.3: Correspondence analysis (optional)
# ═══════════════════════════════════════════════════════════════════

def _run_correspondence_analysis(
    logodds_result: LogoddsResult,
    bridge_result: BridgeResult,
) -> pd.DataFrame:
    """Run CA on word × abuse-type frequency matrix.

    Uses prince.CA if available; returns empty DataFrame otherwise.
    """
    try:
        import prince
    except ImportError:
        return pd.DataFrame()

    doc_counts = logodds_result.doc_counts_abuse_neg
    if doc_counts.empty:
        return pd.DataFrame()

    abuse_cols = [c for c in ABUSE_ORDER if c in doc_counts.columns]
    if len(abuse_cols) < 2:
        return pd.DataFrame()

    # Use top chi-square words + bridge words as rows
    candidate_words = set()
    if not bridge_result.chi_diagnostic.empty:
        top_chi = bridge_result.chi_diagnostic.sort_values(
            "chi2", ascending=False, kind="mergesort"
        ).head(100)
        candidate_words |= set(top_chi.index)
    if not bridge_result.bridge_diagnostic.empty:
        candidate_words |= set(bridge_result.bridge_diagnostic["word"])

    if not candidate_words:
        return pd.DataFrame()

    # Filter doc_counts to candidate words
    valid_words = [w for w in candidate_words if w in doc_counts.index]
    if len(valid_words) < 3:
        return pd.DataFrame()

    matrix = doc_counts.loc[valid_words, abuse_cols]
    matrix = matrix[matrix.sum(axis=1) > 0]

    if matrix.shape[0] < 3:
        return pd.DataFrame()

    ca = prince.CA(n_components=2)
    ca.fit(matrix)

    row_coords = ca.row_coordinates(matrix)
    row_coords.columns = ["Dim1", "Dim2"]
    row_coords["type"] = "word"

    col_coords = ca.column_coordinates(matrix)
    col_coords.columns = ["Dim1", "Dim2"]
    col_coords["type"] = "abuse_type"

    coords = pd.concat([row_coords, col_coords])
    return coords


# ═══════════════════════════════════════════════════════════════════
#  Stage 5.4: Bridge stability (bootstrap)
# ═══════════════════════════════════════════════════════════════════

def _run_bridge_stability(
    logodds_result: LogoddsResult,
    config: OversightConfig,
    n_bootstrap: int = 100,
) -> pd.DataFrame:
    """Bootstrap resample the ABUSE_NEG doc-counts and re-identify bridge words.

    For each bootstrap sample:
    1. Resample rows (children) with replacement
    2. Rebuild doc-count table
    3. Run chi-square → top-k → bridge identification
    4. Record which words were identified as bridge

    Returns a DataFrame with word, frequency (how many bootstrap samples
    identified it as bridge), and stability ratio.
    """
    doc_counts = logodds_result.doc_counts_abuse_neg
    if doc_counts.empty:
        return pd.DataFrame()

    abuse_cols = [c for c in ABUSE_ORDER if c in doc_counts.columns]
    if not abuse_cols:
        return pd.DataFrame()

    min_p1 = config.bridge_min_p1 if config.bridge_min_p1 is not None else BRIDGE_MIN_P1
    min_p2 = config.bridge_min_p2 if config.bridge_min_p2 is not None else BRIDGE_MIN_P2
    max_gap = config.bridge_max_gap if config.bridge_max_gap is not None else BRIDGE_MAX_GAP
    count_min = config.bridge_count_min
    top_k = config.chi_top_k

    rng = np.random.RandomState(config.random_state)
    word_list = doc_counts.index.tolist()
    n_words = len(word_list)
    bridge_counts: dict[str, int] = {}

    for _ in range(n_bootstrap):
        # Resample words (rows) with replacement
        sample_idx = rng.choice(n_words, size=n_words, replace=True)
        sample_counts = doc_counts.iloc[sample_idx].copy()
        # Re-index to avoid duplicates
        sample_counts = sample_counts.groupby(sample_counts.index).sum()

        # Chi-square → top-k
        chi_df = compute_chi_square(sample_counts, abuse_cols)
        if chi_df.empty:
            continue
        top_words = chi_df.sort_values(
            "chi2", ascending=False, kind="mergesort"
        ).head(top_k).index.tolist()

        # Bridge identification
        bridge_df = compute_prob_bridge_for_words(
            sample_counts,
            top_words,
            min_p1=min_p1,
            min_p2=min_p2,
            max_gap=max_gap,
            count_min=count_min,
        )
        if bridge_df.empty:
            continue

        for w in bridge_df["word"]:
            bridge_counts[w] = bridge_counts.get(w, 0) + 1

    if not bridge_counts:
        return pd.DataFrame()

    stability_df = pd.DataFrame([
        {"word": w, "bootstrap_freq": c, "stability_ratio": c / n_bootstrap}
        for w, c in bridge_counts.items()
    ]).sort_values("stability_ratio", ascending=False)

    return stability_df


# ═══════════════════════════════════════════════════════════════════
#  Stage 5: Main entry point
# ═══════════════════════════════════════════════════════════════════

def run_part1_diagnosis(
    corpus: CorpusResult,
    logodds_result: LogoddsResult,
    bridge_result: BridgeResult,
    config: OversightConfig,
) -> DiagnosisResult:
    """Stage 5: Run all Part 1 diagnostic analyses."""
    result = DiagnosisResult()

    result.info_loss = _run_information_loss(corpus)
    result.label_composition = _run_label_composition(corpus)
    result.ca_coords = _run_correspondence_analysis(logodds_result, bridge_result)
    result.bridge_stability = _run_bridge_stability(logodds_result, config)

    return result


def save_diagnosis(result: DiagnosisResult, output_dir: Path) -> None:
    """Save diagnostic results to CSV."""
    out = output_dir / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    # Information loss
    if result.info_loss:
        if "loss_detail_df" in result.info_loss:
            result.info_loss["loss_detail_df"].to_csv(
                out / "info_loss_detail.csv", index=False, encoding="utf-8-sig"
            )
        if "loss_pair_matrix" in result.info_loss:
            result.info_loss["loss_pair_matrix"].to_csv(
                out / "info_loss_pair_matrix.csv", encoding="utf-8-sig"
            )
        # Summary
        summary = {
            k: v for k, v in result.info_loss.items()
            if k not in ("loss_detail_df", "loss_pair_matrix", "lost_by_type")
        }
        if "lost_by_type" in result.info_loss:
            summary.update({
                f"lost_{k}": v
                for k, v in result.info_loss["lost_by_type"].items()
            })
        pd.DataFrame([summary]).to_csv(
            out / "info_loss_summary.csv", index=False, encoding="utf-8-sig"
        )

    # Label composition
    if result.label_composition:
        comp = result.label_composition
        if "detail_df" in comp:
            comp["detail_df"].to_csv(
                out / "label_composition_detail.csv",
                index=False, encoding="utf-8-sig"
            )
        summary = {
            k: v for k, v in comp.items() if k != "detail_df"
        }
        pd.DataFrame([summary]).to_csv(
            out / "label_composition_summary.csv",
            index=False, encoding="utf-8-sig"
        )

    # CA coordinates
    if not result.ca_coords.empty:
        result.ca_coords.to_csv(
            out / "ca_coordinates.csv", encoding="utf-8-sig"
        )

    # Bridge stability
    if not result.bridge_stability.empty:
        result.bridge_stability.to_csv(
            out / "bridge_stability.csv", index=False, encoding="utf-8-sig"
        )
