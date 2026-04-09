"""
Configuration for the 11-stage oversight pipeline.

All paths are resolved relative to the repository root (the parent of the
``oversight/`` folder), so that scripts can be run from any working directory.

Stage overview
--------------
Stages 1-5 : Thin wrappers around existing ``abuse_pipeline/`` functions
              (corpus, tokenization, log-odds, bridge words, Part 1 diagnosis).
Stages 6-11: New code (classifier, ambiguous detection, correction,
              prediction, evidence, evaluation).

The config supports both the Korean child maltreatment data (Stages 1-10) and
the English counsel-chat data (Stage 11, methodological generalization check
only).

IMPORTANT: Stage 11 is a methodological generalization check ONLY. It does NOT
assert clinical utility on counsel-chat data.
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
    # "english_counselchat"  = Stage 11 MGC only
    dataset_name: Literal["korean_abuse", "english_counselchat"] = "korean_abuse"

    # ---- Data paths -------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: _repo_path("data")
    )
    output_dir: Path = field(
        default_factory=lambda: _repo_path("oversight", "outputs")
    )

    # Korean paths (Stage 1-10)
    korean_dataset_path: Path = field(
        default_factory=lambda: _repo_path("data", "dataset_gt_anchored.csv")
    )

    # English paths (Stage 11 only)
    english_dataset_path: Path = field(
        default_factory=lambda: _repo_path("english_data", "20220401_counsel_chat.csv")
    )

    # ---- Stage 1: Corpus stratification -----------------------------------
    only_negative_for_abuse: bool = True
    include_pos_in_none: bool = True
    include_neu_in_none: bool = True
    sub_threshold: int = 4

    # ---- Stage 3: Log-odds ------------------------------------------------
    logodds_alpha: float = 0.01
    min_doc_count: int = 5

    # ---- Stage 4: Bridge words --------------------------------------------
    # None = use legacy defaults from abuse_pipeline.core.common
    bridge_min_p1: float | None = None
    bridge_min_p2: float | None = None
    bridge_max_gap: float | None = None
    bridge_count_min: int = 5
    chi_top_k: int = 200

    # ---- Stage 6: Classifier ----------------------------------------------
    classifier_type: Literal["lr", "svm"] = "lr"
    # TF-IDF params inherited from TFIDF_PARAMS at runtime

    # ---- Stage 7: Ambiguous case detection --------------------------------
    ambiguous_bridge_threshold: float | None = None
    kfold_repeat_seeds: list[int] | None = None
    bootstrap_n_iter: int | None = None
    instability_threshold: float | None = None

    # ---- Stage 8: Correction layers ---------------------------------------
    # Method A (log-linear)
    correction_lambda: float | None = None
    # Method C (separated)
    correction_beta_bridge: float | None = None
    correction_beta_logodds: float | None = None

    # ---- Stage 11: Evaluation ---------------------------------------------
    n_splits: int = 5
    random_state: int = 42
    ground_truth_threshold: int = 4
    f_beta: float = 2.0
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
                "utility on MGC data."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
