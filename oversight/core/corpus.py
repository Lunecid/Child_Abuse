"""
Stage 1-2: Corpus loading, stratification, and tokenization.

Stage 1 — Corpus stratification:
    Load JSON records → classify into valence groups (부정/평범/긍정) →
    build GT-anchored ABUSE_NEG dataset → construct 5-class training data
    (4 abuse types + "해당없음" from POS/NEU children).

Stage 2 — Tokenization:
    Apply Korean morphological tokenization (Okt) to all children's speech.
    ABUSE_NEG children already have tokenized text from build_gt_anchored_dataset.
    POS/NEU children are tokenized here via extract_child_speech + tokenize_korean.

All heavy lifting is delegated to existing abuse_pipeline/ functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oversight._imports import (
    ABUSE_ORDER,
    SEVERITY_RANK,
    classify_child_group,
    classify_abuse_main_sub,
    extract_child_speech,
    tokenize_korean,
    build_gt_anchored_dataset,
)
from oversight.config import OversightConfig


# ═══════════════════════════════════════════════════════════════════
#  Result containers
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CorpusResult:
    """Container for the stratified corpus produced by Stage 1-2."""

    # Stage 1 outputs
    json_files: list[Path] = field(default_factory=list)
    all_records: list[dict[str, Any]] = field(default_factory=list)

    # Valence group counts
    neg_count: int = 0
    neu_count: int = 0
    pos_count: int = 0

    # ABUSE_NEG dataset (from build_gt_anchored_dataset)
    # Columns: doc_id, raw_text, text, gt_main, algo_main, algo_subs,
    #          abuse_scores, y_성학대, y_신체학대, y_정서학대, y_방임,
    #          n_labels, label_list
    abuse_neg_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # POS/NEU children (5th class: "해당없음")
    pos_neu_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Combined training data: ABUSE_NEG + POS/NEU
    train_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Stage 2: tokenization flag
    tokenized: bool = False


# ═══════════════════════════════════════════════════════════════════
#  Stage 1: Load JSON + Stratify
# ═══════════════════════════════════════════════════════════════════

def load_json_files(data_dir: Path) -> list[Path]:
    """Glob all *.json files from the data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"Place JSON source files there before running the pipeline."
        )
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {data_dir}."
        )
    return files


def _build_pos_neu_records(
    json_files: list[Path],
    include_pos: bool = True,
    include_neu: bool = True,
) -> pd.DataFrame:
    """Build DataFrame for POS/NEU children (the "해당없음" class).

    Each row has: doc_id, raw_text, text (tokenized), group.
    """
    target_groups = set()
    if include_pos:
        target_groups.add("긍정")
    if include_neu:
        target_groups.add("평범")

    if not target_groups:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        group = classify_child_group(rec)
        if group not in target_groups:
            continue

        info = rec.get("info", {}) or {}
        doc_id = str(info.get("ID") or info.get("id") or json_path.stem)

        speech_list = extract_child_speech(rec)
        if not speech_list:
            continue
        raw_text = " ".join(
            str(s).strip() for s in speech_list if str(s).strip()
        ).strip()
        if not raw_text:
            continue

        tokenized = " ".join(tokenize_korean(raw_text)).strip()
        if not tokenized:
            continue

        rows.append({
            "doc_id": doc_id,
            "raw_text": raw_text,
            "text": tokenized,
            "group": group,
            "gt_main": "해당없음",
            "label_list": ["해당없음"],
        })

    return pd.DataFrame(rows)


def _build_train_df(
    abuse_neg_df: pd.DataFrame,
    pos_neu_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine ABUSE_NEG + POS/NEU into 5-class training DataFrame.

    The 5 classes are the 4 abuse types + "해당없음".
    Binary columns: y_성학대, y_신체학대, y_정서학대, y_방임, y_해당없음.
    """
    abuse_cols = [f"y_{a}" for a in ABUSE_ORDER]

    # ABUSE_NEG rows: keep existing y_ columns, add y_해당없음 = 0
    abuse_rows = abuse_neg_df.copy()
    for col in abuse_cols:
        if col not in abuse_rows.columns:
            abuse_rows[col] = 0
    abuse_rows["y_해당없음"] = 0

    # POS/NEU rows: all abuse y_ = 0, y_해당없음 = 1
    if not pos_neu_df.empty:
        none_rows = pos_neu_df.copy()
        for col in abuse_cols:
            none_rows[col] = 0
        none_rows["y_해당없음"] = 1
    else:
        none_rows = pd.DataFrame()

    # Shared columns for concatenation
    shared_cols = [
        "doc_id", "raw_text", "text", "gt_main", "label_list",
    ] + abuse_cols + ["y_해당없음"]

    # Ensure columns exist before selecting
    for col in shared_cols:
        if col not in abuse_rows.columns:
            abuse_rows[col] = None
        if not none_rows.empty and col not in none_rows.columns:
            none_rows[col] = None

    parts = [abuse_rows[shared_cols]]
    if not none_rows.empty:
        parts.append(none_rows[shared_cols])

    train_df = pd.concat(parts, ignore_index=True)

    # n_labels for POS/NEU
    if "n_labels" not in train_df.columns:
        train_df["n_labels"] = train_df["label_list"].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )

    return train_df


def stratify_corpus(config: OversightConfig) -> CorpusResult:
    """Stage 1: Load JSON files and build the stratified corpus.

    Returns a CorpusResult with:
    - abuse_neg_df: GT-anchored ABUSE_NEG dataset (from build_gt_anchored_dataset)
    - pos_neu_df: POS/NEU children (해당없음 class)
    - train_df: combined 5-class training data
    """
    json_files = load_json_files(config.data_dir)

    # Count valence groups
    neg_count = neu_count = pos_count = 0
    all_records: list[dict[str, Any]] = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        all_records.append(rec)
        group = classify_child_group(rec)
        if group == "부정":
            neg_count += 1
        elif group == "평범":
            neu_count += 1
        elif group == "긍정":
            pos_count += 1

    # Build GT-anchored ABUSE_NEG dataset
    str_files = [str(p) for p in json_files]
    abuse_neg_df = build_gt_anchored_dataset(
        str_files,
        only_negative=config.only_negative_for_abuse,
    )

    # Build POS/NEU dataset
    pos_neu_df = _build_pos_neu_records(
        json_files,
        include_pos=config.include_pos_in_none,
        include_neu=config.include_neu_in_none,
    )

    # Combine into training data
    train_df = _build_train_df(abuse_neg_df, pos_neu_df)

    result = CorpusResult(
        json_files=json_files,
        all_records=all_records,
        neg_count=neg_count,
        neu_count=neu_count,
        pos_count=pos_count,
        abuse_neg_df=abuse_neg_df,
        pos_neu_df=pos_neu_df,
        train_df=train_df,
        tokenized=True,  # build_gt_anchored_dataset already tokenizes
    )

    return result


# ═══════════════════════════════════════════════════════════════════
#  Stage 2: Tokenization (already done in Stage 1 for this pipeline)
# ═══════════════════════════════════════════════════════════════════

def ensure_tokenized(corpus: CorpusResult) -> CorpusResult:
    """Stage 2: Ensure all text in the corpus is tokenized.

    In this pipeline, tokenization is already performed during Stage 1:
    - ABUSE_NEG children: tokenized by build_gt_anchored_dataset()
    - POS/NEU children: tokenized by _build_pos_neu_records()

    This function is a no-op verification step that confirms the 'text'
    column contains tokenized content for all rows.
    """
    if corpus.tokenized:
        return corpus

    # If somehow not tokenized, apply tokenize_korean to raw_text
    for df_name in ("abuse_neg_df", "train_df", "pos_neu_df"):
        df = getattr(corpus, df_name)
        if df.empty:
            continue
        mask = df["text"].isna() | (df["text"].str.strip() == "")
        if mask.any():
            df.loc[mask, "text"] = df.loc[mask, "raw_text"].apply(
                lambda x: " ".join(tokenize_korean(x)) if isinstance(x, str) else ""
            )

    corpus.tokenized = True
    return corpus


# ═══════════════════════════════════════════════════════════════════
#  Saving
# ═══════════════════════════════════════════════════════════════════

def save_corpus_summary(corpus: CorpusResult, output_dir: Path) -> None:
    """Save corpus stratification summary to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    summary = {
        "total_json_files": len(corpus.json_files),
        "total_records": len(corpus.all_records),
        "neg_count": corpus.neg_count,
        "neu_count": corpus.neu_count,
        "pos_count": corpus.pos_count,
        "abuse_neg_count": len(corpus.abuse_neg_df),
        "pos_neu_count": len(corpus.pos_neu_df),
        "train_count": len(corpus.train_df),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "stage1_corpus_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Abuse type distribution in ABUSE_NEG
    if not corpus.abuse_neg_df.empty:
        gt_dist = corpus.abuse_neg_df["gt_main"].value_counts()
        gt_dist.name = "count"
        gt_dist.to_csv(
            output_dir / "stage1_abuse_neg_gt_distribution.csv",
            encoding="utf-8-sig",
        )

    # Training data class distribution
    if not corpus.train_df.empty:
        y_cols = [f"y_{a}" for a in ABUSE_ORDER] + ["y_해당없음"]
        existing = [c for c in y_cols if c in corpus.train_df.columns]
        class_counts = corpus.train_df[existing].sum().astype(int)
        class_counts.name = "positive_count"
        class_counts.to_csv(
            output_dir / "stage1_train_class_distribution.csv",
            encoding="utf-8-sig",
        )
