"""
Configuration for the oversight prevention pipeline.

All paths are resolved relative to the repository root (the parent of the
`oversight/` folder), so that scripts can be run from any working directory.

The config supports both the Korean child maltreatment data (Stage 1-10) and
the English counsel-chat data (Stage 11, methodological generalization check
only).

IMPORTANT: Stage 11 is a methodological generalization check ONLY. It does NOT
assert clinical utility on counsel-chat data. See PROJECT_PLAN.md Section 1.4
and Stage 11 for the full rationale. The `active_f_beta` property and the
assertion in `validate()` enforce this boundary at the config level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# Repository root (one level above this file's parent folder).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return _REPO_ROOT.joinpath(*parts)


@dataclass
class OversightConfig:
    # ---- Dataset selection ------------------------------------------------
    # "korean_abuse"         = Stage 1-10 main experiments
    #                          (clinical utility IS claimed)
    # "english_counselchat"  = Stage 11 MGC only
    #                          (clinical utility NOT claimed)
    dataset_name: Literal["korean_abuse", "english_counselchat"] = "korean_abuse"

    # ---- Data paths (relative to repo root) -------------------------------
    # Korean paths (Stage 1-10)
    korean_dataset_path: Path = field(
        default_factory=lambda: _repo_path("data", "dataset_gt_anchored.csv")
    )
    korean_bridge_words_path: Path = field(
        default_factory=lambda: _repo_path("data", "stage0_bridge_words_doclevel.csv")
    )
    korean_logodds_path: Path = field(
        default_factory=lambda: _repo_path(
            "data", "abuse_word_stats_logodds_mainOnly_childtokens.csv"
        )
    )

    # English paths (Stage 11 only)
    english_dataset_path: Path = field(
        default_factory=lambda: _repo_path("english_data", "20220401_counsel_chat.csv")
    )
    english_bridge_words_path: Path | None = None  # recomputed in Stage 11.3
    english_logodds_path: Path | None = None  # recomputed in Stage 11.3

    # Output (always under oversight/outputs/)
    output_dir: Path = field(default_factory=lambda: _repo_path("oversight", "outputs"))

    # ---- Utterance splitting ----------------------------------------------
    min_utterance_tokens: int = 3
    max_utterance_tokens: int = 100

    # ---- Layer 1 channels -------------------------------------------------
    channel_a_enabled: bool = True
    channel_b_enabled: bool = True
    channel_c_enabled: bool = True

    # Channel A: bridge words
    bridge_stability_filter: Literal["strict", "baseline", "loose"] = "loose"

    # Channel B: log-odds
    logodds_top_k_per_type: int = 20

    # Channel C: uncertainty
    uncertainty_top_k_per_child: int = 10
    uncertainty_metric: Literal["entropy", "margin"] = "entropy"

    # ---- Layer 2 classifier -----------------------------------------------
    classifier_type: Literal["tfidf_lr", "tfidf_svm"] = "tfidf_lr"
    classifier_ngram: tuple[int, int] = (1, 2)
    classifier_min_df: int = 3
    classifier_max_features: int = 5000

    # ---- Layer 3 aggregation ----------------------------------------------
    aggregation_method: Literal["max", "mean", "max_plus_count"] = "max_plus_count"
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.2, "medium": 0.4, "high": 0.6}
    )

    # ---- Evaluation -------------------------------------------------------
    n_splits: int = 5
    random_state: int = 42
    ground_truth_threshold: int = 4  # A_k >= 4 defines multi-label (Korean)
    f_beta: float = 2.0  # recall-weighted, Korean data only

    # For English counsel-chat (Stage 11), F1 is used.
    # See PROJECT_PLAN.md Stage 11.5.
    english_f_beta: float = 1.0

    # ---- Resolved properties ----------------------------------------------
    @property
    def dataset_path(self) -> Path:
        """Resolve the active dataset path based on dataset_name."""
        if self.dataset_name == "korean_abuse":
            return self.korean_dataset_path
        elif self.dataset_name == "english_counselchat":
            return self.english_dataset_path
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

    @property
    def active_f_beta(self) -> float:
        """Return the F-beta value appropriate for the active dataset.

        Korean data: beta=2.0 (recall-weighted, clinical context)
        English counsel-chat: beta=1.0 (standard F1, MGC only)
        """
        if self.dataset_name == "korean_abuse":
            return self.f_beta
        else:
            return self.english_f_beta

    @property
    def repo_root(self) -> Path:
        """Return the repository root path."""
        return _REPO_ROOT

    def validate(self) -> None:
        """Raise if the config is internally inconsistent or dataset missing."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path} "
                f"(dataset_name={self.dataset_name}). "
                f"Expected location relative to repo root: "
                f"{self.dataset_path.relative_to(_REPO_ROOT)}"
            )
        if self.dataset_name == "english_counselchat":
            assert self.active_f_beta == 1.0, (
                "English counsel-chat runs must use F1 (beta=1.0), not F2. "
                "This is enforced to prevent accidentally claiming clinical "
                "utility on MGC data. See PROJECT_PLAN.md Stage 11.5."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
