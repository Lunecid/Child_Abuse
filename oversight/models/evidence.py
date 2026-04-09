"""
Stage 10: Evidence utterance extraction.

For each child with a positive class prediction, find sentences in their
raw speech that contain evidence for that prediction:
    (a) Top log-odds words for the predicted class
    (b) Bridge words connecting that class
    (c) Overall token-level salience

Each evidence sentence is annotated with trigger type and trigger words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oversight._imports import ABUSE_ORDER
from oversight.config import OversightConfig
from oversight.core.corpus import CorpusResult
from oversight.models.classifier import FIVE_CLASSES
from oversight.models.prediction import PredictionResult
from oversight.stats.logodds import LogoddsResult
from oversight.stats.bridge import BridgeResult


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvidenceResult:
    """Container for evidence utterance extraction."""

    # Per-child evidence records
    # Each row: doc_id, predicted_class, sentence, trigger_type, trigger_words
    evidence_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════
#  Sentence splitting
# ═══════════════════════════════════════════════════════════════════

def _split_sentences(raw_text: str) -> list[str]:
    """Split raw text into sentences using Korean punctuation boundaries."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    # Split on common Korean sentence boundaries
    sentences = re.split(r'[.?!。]\s*', raw_text)
    # Also split on double spaces (audio segment boundaries)
    expanded = []
    for s in sentences:
        expanded.extend(s.split("  "))
    return [s.strip() for s in expanded if s.strip()]


# ═══════════════════════════════════════════════════════════════════
#  Top log-odds words per class
# ═══════════════════════════════════════════════════════════════════

def _get_top_logodds_words(
    logodds_df: pd.DataFrame,
    top_k: int = 20,
) -> dict[str, set[str]]:
    """Get top-k log-odds words per class."""
    result: dict[str, set[str]] = {}
    if logodds_df.empty:
        return result

    for cls in ABUSE_ORDER:
        cls_df = logodds_df[logodds_df["group"] == cls].copy()
        if cls_df.empty:
            continue
        cls_df = cls_df.sort_values("log_odds", ascending=False, kind="mergesort")
        result[cls] = set(cls_df.head(top_k)["word"])

    return result


# ═══════════════════════════════════════════════════════════════════
#  Stage 10: Extract evidence
# ═══════════════════════════════════════════════════════════════════

def extract_evidence_utterances(
    corpus: CorpusResult,
    prediction_result: PredictionResult,
    logodds_result: LogoddsResult,
    bridge_result: BridgeResult,
    config: OversightConfig,
    method: str = "method_a",
) -> EvidenceResult:
    """Stage 10: Extract evidence sentences for each positive prediction.

    Parameters
    ----------
    method : str
        "method_a" or "method_c" — which prediction set to use.
    """
    train_df = corpus.train_df
    if train_df.empty:
        return EvidenceResult()

    y_pred = (
        prediction_result.y_pred_a if method == "method_a"
        else prediction_result.y_pred_c
    )
    if y_pred.size == 0:
        return EvidenceResult()

    # Get top log-odds words per abuse class
    top_words = _get_top_logodds_words(
        logodds_result.logodds_train_5class, top_k=20
    )

    # Bridge words per class pair
    bridge_words: set[str] = set()
    if not bridge_result.bridge_correction.empty:
        bridge_words = set(bridge_result.bridge_correction["word"])

    # Bridge words by primary abuse type
    bridge_by_class: dict[str, set[str]] = {}
    if not bridge_result.bridge_correction.empty:
        for _, row in bridge_result.bridge_correction.iterrows():
            cls = row.get("primary_abuse", "")
            w = row.get("word", "")
            if cls and w:
                bridge_by_class.setdefault(cls, set()).add(w)
            cls2 = row.get("secondary_abuse", "")
            if cls2 and w:
                bridge_by_class.setdefault(cls2, set()).add(w)

    evidence_rows: list[dict[str, Any]] = []

    for i, row in train_df.iterrows():
        doc_id = row.get("doc_id", i)
        raw_text = row.get("raw_text", "")
        idx = train_df.index.get_loc(i) if isinstance(i, (int, np.integer)) else i

        if idx >= len(y_pred):
            continue

        sentences = _split_sentences(raw_text)
        if not sentences:
            continue

        # For each predicted positive class
        for j, cls in enumerate(FIVE_CLASSES):
            if j >= y_pred.shape[1] or y_pred[idx, j] != 1:
                continue
            if cls == "해당없음":
                continue  # No evidence needed for non-abuse

            # Find evidence sentences
            cls_top_words = top_words.get(cls, set())
            cls_bridge_words = bridge_by_class.get(cls, set())

            for sent in sentences:
                sent_tokens = set(sent.split())
                triggers = []
                trigger_words = []

                # Check log-odds words
                lo_match = sent_tokens & cls_top_words
                if lo_match:
                    triggers.append("logodds")
                    trigger_words.extend(sorted(lo_match))

                # Check bridge words
                br_match = sent_tokens & cls_bridge_words
                if br_match:
                    triggers.append("bridge")
                    trigger_words.extend(sorted(br_match))

                if triggers:
                    evidence_rows.append({
                        "doc_id": doc_id,
                        "predicted_class": cls,
                        "sentence": sent,
                        "trigger_type": "+".join(triggers),
                        "trigger_words": "|".join(trigger_words),
                    })

    return EvidenceResult(
        evidence_df=pd.DataFrame(evidence_rows) if evidence_rows else pd.DataFrame()
    )


def save_evidence(result: EvidenceResult, output_dir: Path, method: str = "method_a") -> None:
    """Save evidence extraction results."""
    out = output_dir / "evidence"
    out.mkdir(parents=True, exist_ok=True)

    if not result.evidence_df.empty:
        result.evidence_df.to_csv(
            out / f"evidence_{method}.csv",
            index=False, encoding="utf-8-sig"
        )
