"""
Import adapter for legacy abuse_pipeline modules.

This module is the single entry point for all legacy code used by the new
oversight pipeline. If a legacy module's path or function name changes, only
this file needs to be updated.

Usage (from other oversight modules):
    from oversight._imports import tokenize_korean, compute_log_odds
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so that `abuse_pipeline` is
# importable when oversight scripts are run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# -- Text processing --------------------------------------------------------
try:
    from abuse_pipeline.core.text import tokenize_korean
except ImportError as e:
    raise ImportError(
        f"Failed to import tokenize_korean from abuse_pipeline.core.text. "
        f"Original error: {e}"
    ) from e

try:
    from abuse_pipeline.core.text import extract_child_speech
except ImportError as e:
    raise ImportError(
        f"Failed to import extract_child_speech from abuse_pipeline.core.text. "
        f"Original error: {e}"
    ) from e


# -- Statistical functions ---------------------------------------------------
try:
    from abuse_pipeline.stats.stats import compute_log_odds
except ImportError as e:
    raise ImportError(
        f"Failed to import compute_log_odds from abuse_pipeline.stats.stats. "
        f"Original error: {e}"
    ) from e

try:
    from abuse_pipeline.stats.stats import compute_chi_square
except ImportError as e:
    raise ImportError(
        f"Failed to import compute_chi_square from abuse_pipeline.stats.stats. "
        f"Original error: {e}"
    ) from e

try:
    from abuse_pipeline.stats.stats import compute_prob_bridge_for_words
except ImportError as e:
    raise ImportError(
        f"Failed to import compute_prob_bridge_for_words from "
        f"abuse_pipeline.stats.stats. Original error: {e}"
    ) from e


# -- Labels / constants ------------------------------------------------------
try:
    from abuse_pipeline.core.common import (
        ABUSE_ORDER,
        SEVERITY_RANK,
        VALENCE_ORDER,
        TFIDF_PARAMS,
        BRIDGE_MIN_P1,
        BRIDGE_MIN_P2,
        BRIDGE_MAX_GAP,
        BRIDGE_MIN_COUNT,
        MIN_DOC_COUNT,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import constants from abuse_pipeline.core.common. "
        f"Original error: {e}"
    ) from e

# Alias for PROJECT_PLAN.md terminology compatibility.
ABUSE_TYPES: tuple[str, ...] = tuple(ABUSE_ORDER)

try:
    from abuse_pipeline.core.labels import (
        classify_child_group,
        classify_abuse_main_sub,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import classification functions from "
        f"abuse_pipeline.core.labels. Original error: {e}"
    ) from e


# -- Data processing --------------------------------------------------------
try:
    from abuse_pipeline.data.doc_level import build_abuse_doc_word_table
except ImportError as e:
    raise ImportError(
        f"Failed to import build_abuse_doc_word_table from "
        f"abuse_pipeline.data.doc_level. Original error: {e}"
    ) from e


# -- GT-anchored dataset builder + diagnostics -------------------------------
try:
    from abuse_pipeline.experiments.information_recovery import (
        build_gt_anchored_dataset,
        compute_label_composition,
        quantify_information_loss,
        compute_recovery_metrics,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import functions from "
        f"abuse_pipeline.experiments.information_recovery. Original error: {e}"
    ) from e


# -- Cross-validation split -------------------------------------------------
try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    StratifiedKFold = None  # type: ignore[assignment,misc]


# -- Sanity check -----------------------------------------------------------
def _verify_imports() -> dict[str, bool]:
    """Return a dict of {name: is_importable} for all legacy dependencies.

    Called by the Stage 0 checkpoint.
    """
    return {
        "tokenize_korean": tokenize_korean is not None,
        "extract_child_speech": extract_child_speech is not None,
        "compute_log_odds": compute_log_odds is not None,
        "compute_chi_square": compute_chi_square is not None,
        "compute_prob_bridge_for_words": compute_prob_bridge_for_words is not None,
        "ABUSE_ORDER": len(ABUSE_ORDER) == 4,
        "SEVERITY_RANK": len(SEVERITY_RANK) == 4,
        "VALENCE_ORDER": len(VALENCE_ORDER) == 3,
        "TFIDF_PARAMS": isinstance(TFIDF_PARAMS, dict),
        "classify_child_group": classify_child_group is not None,
        "classify_abuse_main_sub": classify_abuse_main_sub is not None,
        "build_abuse_doc_word_table": build_abuse_doc_word_table is not None,
        "build_gt_anchored_dataset": build_gt_anchored_dataset is not None,
        "compute_label_composition": compute_label_composition is not None,
        "quantify_information_loss": quantify_information_loss is not None,
        "compute_recovery_metrics": compute_recovery_metrics is not None,
        "StratifiedKFold": StratifiedKFold is not None,
    }


__all__ = [
    # Text processing
    "tokenize_korean",
    "extract_child_speech",
    # Statistics
    "compute_log_odds",
    "compute_chi_square",
    "compute_prob_bridge_for_words",
    # Constants
    "ABUSE_ORDER",
    "ABUSE_TYPES",
    "SEVERITY_RANK",
    "VALENCE_ORDER",
    "TFIDF_PARAMS",
    "BRIDGE_MIN_P1",
    "BRIDGE_MIN_P2",
    "BRIDGE_MAX_GAP",
    "BRIDGE_MIN_COUNT",
    "MIN_DOC_COUNT",
    # Labels
    "classify_child_group",
    "classify_abuse_main_sub",
    # Data processing
    "build_abuse_doc_word_table",
    # GT-anchored dataset + diagnostics
    "build_gt_anchored_dataset",
    "compute_label_composition",
    "quantify_information_loss",
    "compute_recovery_metrics",
    # Cross-validation
    "StratifiedKFold",
    # Verification
    "_verify_imports",
]
