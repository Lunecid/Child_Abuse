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
# The function is named compute_log_odds (not stabilized_log_odds).
# It implements stabilized log-odds with Dirichlet smoothing (alpha=0.01).
try:
    from abuse_pipeline.stats.stats import compute_log_odds
except ImportError as e:
    raise ImportError(
        f"Failed to import compute_log_odds from abuse_pipeline.stats.stats. "
        f"Original error: {e}"
    ) from e


# -- Labels / constants ------------------------------------------------------
# The canonical list is ABUSE_ORDER (not ABUSE_TYPES).
try:
    from abuse_pipeline.core.common import ABUSE_ORDER, SEVERITY_RANK
except ImportError as e:
    raise ImportError(
        f"Failed to import ABUSE_ORDER/SEVERITY_RANK from "
        f"abuse_pipeline.core.common. Original error: {e}"
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


# -- GT-anchored dataset builder --------------------------------------------
try:
    from abuse_pipeline.experiments.information_recovery import (
        build_gt_anchored_dataset,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import build_gt_anchored_dataset from "
        f"abuse_pipeline.experiments.information_recovery. Original error: {e}"
    ) from e


# -- Cross-validation split -------------------------------------------------
# No dedicated build_stratified_folds function exists in the legacy code.
# Each module uses sklearn StratifiedKFold directly.
# We re-export StratifiedKFold here for convenience.
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
        "ABUSE_ORDER": len(ABUSE_ORDER) == 4,
        "SEVERITY_RANK": len(SEVERITY_RANK) == 4,
        "classify_child_group": classify_child_group is not None,
        "classify_abuse_main_sub": classify_abuse_main_sub is not None,
        "build_gt_anchored_dataset": build_gt_anchored_dataset is not None,
        "StratifiedKFold": StratifiedKFold is not None,
    }


__all__ = [
    "tokenize_korean",
    "extract_child_speech",
    "compute_log_odds",
    "ABUSE_ORDER",
    "ABUSE_TYPES",
    "SEVERITY_RANK",
    "classify_child_group",
    "classify_abuse_main_sub",
    "build_gt_anchored_dataset",
    "StratifiedKFold",
    "_verify_imports",
]
