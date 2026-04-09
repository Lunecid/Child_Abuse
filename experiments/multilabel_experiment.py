from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "child_abuse_mpl"))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_abuse_main_sub, classify_child_group
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean
from abuse_pipeline.stats.stats import (
    compute_chi_square,
    compute_log_odds,
    compute_prob_bridge_for_words,
)

ABUSE_TYPES = list(C.ABUSE_ORDER)
TYPE_TO_IDX = {label: idx for idx, label in enumerate(ABUSE_TYPES)}
IDX_TO_TYPE = {idx: label for label, idx in TYPE_TO_IDX.items()}
CHAIN_ORDER = [TYPE_TO_IDX[label] for label in ABUSE_TYPES]
DEFAULT_SOURCE_ROOT = Path("/Users/lunecid/project/Child_Abuse")


@dataclass(slots=True)
class ExperimentConfig:
    source_root: Path
    output_dir: Path
    n_splits: int = 5
    inner_splits: int = 3
    random_state: int = 42
    only_negative: bool = True
    chi_top_k: int = 200
    bridge_min_total_docs: int = 5
    bridge_count_min: int = 5
    bridge_min_p1: float = float(C.BRIDGE_MIN_P1)
    bridge_min_p2: float = float(C.BRIDGE_MIN_P2)
    bridge_max_gap: float = float(C.BRIDGE_MAX_GAP)
    min_main_preserved_for_tuning: float = 0.80
    lambda_candidates: tuple[float, ...] = field(
        default_factory=lambda: (0.1, 0.5, 1.0, 2.0, 5.0)
    )
    tau_candidates: tuple[float, ...] = field(
        default_factory=lambda: (-0.5, -0.3, 0.0, 0.3, 0.5)
    )


@dataclass(slots=True)
class BinaryStageResult:
    y_pred: np.ndarray
    scores: np.ndarray
    main_pred_indices: np.ndarray


def normalize_doc_id(value: Any, fallback: str | None = None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = fallback
    text = "" if value is None else str(value).strip()
    if not text:
        text = "" if fallback is None else str(fallback).strip()
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    try:
        as_float = float(text)
    except ValueError:
        return text
    if np.isfinite(as_float) and as_float.is_integer():
        return str(int(as_float))
    return text


def pair_key(label_a: str, label_b: str) -> tuple[str, str]:
    return tuple(sorted((label_a, label_b), key=lambda item: TYPE_TO_IDX[item]))


def resolve_source_root(source_root: str | None = None) -> Path:
    candidates: list[Path] = []
    if source_root:
        candidates.append(Path(source_root).expanduser())
    env_root = os.environ.get("CHILD_ABUSE_SOURCE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([PROJECT_ROOT, DEFAULT_SOURCE_ROOT])
    for candidate in candidates:
        if candidate and (candidate / "data").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate a source root with a data directory. "
        "Pass --source-root or set CHILD_ABUSE_SOURCE_ROOT."
    )


def build_config(source_root: str | None, output_dir: str | None) -> ExperimentConfig:
    resolved_source = resolve_source_root(source_root)
    resolved_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (PROJECT_ROOT / "results")
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    (resolved_output / "latex_tables").mkdir(parents=True, exist_ok=True)
    return ExperimentConfig(source_root=resolved_source, output_dir=resolved_output)


def default_stage_paths(source_root: Path) -> dict[str, Path]:
    neg_root = source_root / "ver28_negOnly"
    main_sub_root = neg_root / "main_sub_analysis"
    abuse_stats_root = neg_root / "abuse" / "stats"
    return {
        "data_dir": source_root / "data",
        "sub_scores": main_sub_root / "stage1_sub_scores.csv",
        "main_sub_matrix": main_sub_root / "stage1_main_sub_matrix.csv",
        "entropy": main_sub_root / "stage4_entropy.csv",
        "confusion": main_sub_root / "stage6_confusion_matrix.csv",
        "bridge_words": abuse_stats_root / "stage0_bridge_words.csv",
        "bridge_words_doclevel": abuse_stats_root / "stage0_bridge_words_doclevel.csv",
    }


def _sort_docs(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["_doc_sort"] = pd.to_numeric(ordered["doc_id"], errors="coerce")
    ordered = ordered.sort_values(["_doc_sort", "doc_id", "source_file"]).drop(columns="_doc_sort")
    return ordered.reset_index(drop=True)


def build_base_documents(data_dir: Path, only_negative: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for json_path in sorted(data_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue

        if only_negative and classify_child_group(record) != "부정":
            continue

        main_label, sub_labels = classify_abuse_main_sub(record)
        if main_label not in ABUSE_TYPES:
            continue

        speech_list = extract_child_speech(record)
        if not speech_list:
            continue

        raw_text = " ".join(str(item).strip() for item in speech_list if str(item).strip()).strip()
        if not raw_text:
            continue

        tokenized_text = " ".join(tokenize_korean(raw_text)).strip()
        if not tokenized_text:
            continue

        info = record.get("info", {}) or {}
        doc_id = normalize_doc_id(
            info.get("ID") or info.get("id") or info.get("Id"),
            fallback=json_path.stem,
        )
        label_set = [main_label, *[label for label in sub_labels if label in ABUSE_TYPES]]
        rows.append(
            {
                "doc_id": doc_id,
                "source_file": str(json_path),
                "raw_text": raw_text,
                "text": tokenized_text,
                "main": main_label,
                "direct_label_list": list(dict.fromkeys(label_set)),
                "direct_sub_list": [label for label in sub_labels if label in ABUSE_TYPES],
            }
        )

    if not rows:
        raise RuntimeError(f"No usable documents found in {data_dir}")
    return _sort_docs(pd.DataFrame(rows))


def load_stage_sub_scores(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["doc_id", "main", "sub", "score", "source"])
    df = pd.read_csv(csv_path)
    if "doc_id" in df.columns:
        df["doc_id"] = df["doc_id"].map(normalize_doc_id)
    return df


def build_multilabel_dataset(base_df: pd.DataFrame, sub_scores_df: pd.DataFrame) -> pd.DataFrame:
    label_sets = {row.doc_id: {row.main} for row in base_df.itertuples(index=False)}

    if sub_scores_df is not None and not sub_scores_df.empty:
        for row in sub_scores_df.itertuples(index=False):
            doc_id = normalize_doc_id(getattr(row, "doc_id", None))
            sub_label = getattr(row, "sub", None)
            if doc_id in label_sets and sub_label in ABUSE_TYPES:
                label_sets[doc_id].add(sub_label)
    else:
        for row in base_df.itertuples(index=False):
            label_sets[row.doc_id] = set(row.direct_label_list)

    mlb = MultiLabelBinarizer(classes=ABUSE_TYPES)
    y_multi = mlb.fit_transform([sorted(label_sets[row.doc_id], key=lambda item: TYPE_TO_IDX[item]) for row in base_df.itertuples(index=False)])

    dataset_df = base_df[["doc_id", "source_file", "raw_text", "text", "main"]].copy()
    dataset_df["label_list"] = [
        sorted(label_sets[row.doc_id], key=lambda item: TYPE_TO_IDX[item])
        for row in base_df.itertuples(index=False)
    ]
    dataset_df["label_set"] = dataset_df["label_list"].apply(lambda items: "|".join(items))
    dataset_df["n_labels"] = dataset_df["label_list"].apply(len)
    for label in ABUSE_TYPES:
        dataset_df[f"y_{label}"] = y_multi[:, TYPE_TO_IDX[label]]

    direct_sets = {
        row.doc_id: set(row.direct_label_list)
        for row in base_df.itertuples(index=False)
    }
    dataset_df["direct_match_stage1"] = [
        int(set(labels) == direct_sets[row.doc_id])
        for row, labels in zip(base_df.itertuples(index=False), dataset_df["label_list"])
    ]

    if len(dataset_df) != 1503:
        raise AssertionError(f"Expected 1503 labeled ABUSE_NEG cases, found {len(dataset_df)}")
    observed_dist = dataset_df["n_labels"].value_counts().sort_index().to_dict()
    print(f"[INFO] n_labels distribution (n={len(dataset_df)}): {observed_dist}")
    if not (dataset_df.apply(lambda row: row["main"] in row["label_list"], axis=1).all()):
        raise AssertionError("Main labels must always be included in label_set")
    if not (dataset_df[[f"y_{label}" for label in ABUSE_TYPES]].sum(axis=1) >= 1).all():
        raise AssertionError("Every multilabel row must contain at least one label")

    return dataset_df.reset_index(drop=True)


def save_dataset_reports(
    dataset_df: pd.DataFrame,
    output_dir: Path,
    pair_reference_path: Path,
    entropy_reference_path: Path,
) -> None:
    single_counts = dataset_df["main"].value_counts().reindex(ABUSE_TYPES, fill_value=0)
    multi_counts = {
        label: int(dataset_df[f"y_{label}"].sum())
        for label in ABUSE_TYPES
    }
    frequency_rows = []
    for label in ABUSE_TYPES:
        frequency_rows.append(
            {
                "label": label,
                "single_label_count": int(single_counts[label]),
                "multilabel_count": int(multi_counts[label]),
                "delta": int(multi_counts[label] - single_counts[label]),
            }
        )
    pd.DataFrame(frequency_rows).to_csv(
        output_dir / "label_frequency_report.csv",
        encoding="utf-8-sig",
        index=False,
    )

    pair_counter: Counter[tuple[str, str]] = Counter()
    for labels in dataset_df["label_list"]:
        if len(labels) < 2:
            continue
        for label_a, label_b in combinations(labels, 2):
            pair_counter[pair_key(label_a, label_b)] += 1

    reference_pairs: dict[tuple[str, str], int] = {}
    if pair_reference_path.exists():
        reference_df = pd.read_csv(pair_reference_path, index_col=0)
        for label_a, label_b in combinations(ABUSE_TYPES, 2):
            ref_count = int(reference_df.loc[label_a, label_b]) + int(reference_df.loc[label_b, label_a])
            reference_pairs[pair_key(label_a, label_b)] = ref_count

    pair_rows = []
    for label_a, label_b in combinations(ABUSE_TYPES, 2):
        pair = pair_key(label_a, label_b)
        pair_rows.append(
            {
                "pair": f"{pair[0]}|{pair[1]}",
                "count": int(pair_counter.get(pair, 0)),
                "stage1_directed_total": int(reference_pairs.get(pair, 0)),
            }
        )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "label_pair_frequency.csv",
        encoding="utf-8-sig",
        index=False,
    )

    cardinality_df = (
        dataset_df["n_labels"]
        .value_counts()
        .sort_index()
        .rename_axis("n_labels")
        .reset_index(name="count")
    )
    cardinality_df["pct"] = (cardinality_df["count"] / len(dataset_df)).round(6)
    pd.DataFrame(cardinality_df).to_csv(
        output_dir / "label_cardinality_distribution.csv",
        encoding="utf-8-sig",
        index=False,
    )

    entropy_rows: list[dict[str, Any]] = []
    if entropy_reference_path.exists():
        entropy_df = pd.read_csv(entropy_reference_path)
        entropy_counts = entropy_df["n_labels"].value_counts().sort_index().to_dict()
        for label_count in range(1, 5):
            entropy_rows.append(
                {
                    "metric": f"stage4_entropy_n_labels_{label_count}",
                    "value": int(entropy_counts.get(label_count, 0)),
                }
            )
    entropy_rows.extend(
        [
            {"metric": "dataset_size", "value": int(len(dataset_df))},
            {"metric": "stage1_direct_match_rate", "value": float(dataset_df["direct_match_stage1"].mean())},
        ]
    )
    pd.DataFrame(entropy_rows).to_csv(
        output_dir / "dataset_checks.csv",
        encoding="utf-8-sig",
        index=False,
    )


def make_targets(dataset_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_true = dataset_df[[f"y_{label}" for label in ABUSE_TYPES]].to_numpy(dtype=int)
    gt_main_indices = dataset_df["main"].map(TYPE_TO_IDX).to_numpy(dtype=int)
    return y_true, gt_main_indices


def safe_fit_vectorizer(train_texts: list[str]) -> tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer(**C.TFIDF_PARAMS)
    try:
        train_matrix = vectorizer.fit_transform(train_texts)
    except ValueError as exc:
        if "no terms remain" not in str(exc):
            raise
        relaxed = dict(C.TFIDF_PARAMS)
        relaxed["min_df"] = 1
        vectorizer = TfidfVectorizer(**relaxed)
        train_matrix = vectorizer.fit_transform(train_texts)
    return vectorizer, train_matrix


def fit_single_stage(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    random_state: int,
) -> BinaryStageResult:
    classifier = LinearSVC(max_iter=1000, dual="auto", random_state=random_state)
    classifier.fit(X_train, y_train)
    pred_labels = classifier.predict(X_test)
    raw_scores = classifier.decision_function(X_test)
    if raw_scores.ndim == 1:
        raw_scores = raw_scores[:, np.newaxis]
    aligned_scores = np.full((X_test.shape[0], len(ABUSE_TYPES)), -np.inf, dtype=float)
    for column_idx, label in enumerate(classifier.classes_):
        aligned_scores[:, TYPE_TO_IDX[str(label)]] = raw_scores[:, column_idx]

    main_pred_indices = np.array([TYPE_TO_IDX[str(label)] for label in pred_labels], dtype=int)
    y_pred = np.zeros((X_test.shape[0], len(ABUSE_TYPES)), dtype=int)
    y_pred[np.arange(len(main_pred_indices)), main_pred_indices] = 1
    return BinaryStageResult(y_pred=y_pred, scores=aligned_scores, main_pred_indices=main_pred_indices)


def fit_br_stage(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    random_state: int,
) -> BinaryStageResult:
    n_labels = y_train.shape[1]
    scores = np.zeros((X_test.shape[0], n_labels), dtype=float)
    y_pred = np.zeros((X_test.shape[0], n_labels), dtype=int)

    for label_idx in range(n_labels):
        y_col = y_train[:, label_idx]
        if np.unique(y_col).size < 2:
            constant = int(y_col[0])
            y_pred[:, label_idx] = constant
            scores[:, label_idx] = 1.0 if constant == 1 else -1.0
            continue

        classifier = LinearSVC(max_iter=1000, dual="auto", random_state=random_state)
        classifier.fit(X_train, y_col)
        score_column = np.asarray(classifier.decision_function(X_test), dtype=float).reshape(-1)
        scores[:, label_idx] = score_column
        y_pred[:, label_idx] = (score_column > 0.0).astype(int)

    main_pred_indices = np.argmax(scores, axis=1)
    y_pred[np.arange(len(main_pred_indices)), main_pred_indices] = 1
    return BinaryStageResult(y_pred=y_pred, scores=scores, main_pred_indices=main_pred_indices)


def fit_cc_stage(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    random_state: int,
) -> BinaryStageResult:
    X_train_csr = X_train.tocsr()
    X_test_csr = X_test.tocsr()
    n_test = X_test_csr.shape[0]
    n_labels = y_train.shape[1]

    scores = np.zeros((n_test, n_labels), dtype=float)
    y_pred = np.zeros((n_test, n_labels), dtype=int)

    for position, label_idx in enumerate(CHAIN_ORDER):
        if position == 0:
            X_train_aug = X_train_csr
            X_test_aug = X_test_csr
        else:
            prev_train = csr_matrix(y_train[:, CHAIN_ORDER[:position]])
            prev_test = csr_matrix(y_pred[:, CHAIN_ORDER[:position]])
            X_train_aug = hstack([X_train_csr, prev_train], format="csr")
            X_test_aug = hstack([X_test_csr, prev_test], format="csr")

        y_col = y_train[:, label_idx]
        if np.unique(y_col).size < 2:
            constant = int(y_col[0])
            y_pred[:, label_idx] = constant
            scores[:, label_idx] = 1.0 if constant == 1 else -1.0
            continue

        classifier = LinearSVC(max_iter=1000, dual="auto", random_state=random_state)
        classifier.fit(X_train_aug, y_col)
        score_column = np.asarray(classifier.decision_function(X_test_aug), dtype=float).reshape(-1)
        scores[:, label_idx] = score_column
        y_pred[:, label_idx] = (score_column > 0.0).astype(int)

    main_pred_indices = np.argmax(scores, axis=1)
    y_pred[np.arange(len(main_pred_indices)), main_pred_indices] = 1
    return BinaryStageResult(y_pred=y_pred, scores=scores, main_pred_indices=main_pred_indices)


def build_doc_count_table(train_df: pd.DataFrame, min_total_docs: int) -> pd.DataFrame:
    rows = []
    for row in train_df.itertuples(index=False):
        for word in set(str(row.text).split()):
            rows.append({"word": word, "abuse": row.main})
    if not rows:
        return pd.DataFrame()

    count_df = (
        pd.DataFrame(rows)
        .groupby(["word", "abuse"])
        .size()
        .unstack("abuse")
        .fillna(0)
        .astype(int)
    )
    for label in ABUSE_TYPES:
        if label not in count_df.columns:
            count_df[label] = 0
    count_df = count_df[ABUSE_TYPES]
    count_df["total_docs"] = count_df.sum(axis=1)
    count_df = count_df[count_df["total_docs"] >= min_total_docs].drop(columns="total_docs")
    return count_df


def bridge_df_to_dict(bridge_df: pd.DataFrame, weighting_mode: str) -> dict[tuple[str, str], dict[str, float]]:
    bridge_dict: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    if bridge_df is None or bridge_df.empty:
        return {}
    for row in bridge_df.itertuples(index=False):
        pair = pair_key(str(row.primary_abuse), str(row.secondary_abuse))
        weight = float(row.p2) if weighting_mode == "weighted" else 1.0
        prev = bridge_dict[pair].get(str(row.word), 0.0)
        bridge_dict[pair][str(row.word)] = max(prev, weight)
    return dict(bridge_dict)


def extract_bridge_words_from_fold(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
    full_bridge_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    count_df = build_doc_count_table(train_df, min_total_docs=config.bridge_min_total_docs)
    if count_df.empty:
        return {
            "count_df": count_df,
            "chi_df": pd.DataFrame(),
            "logodds_df": pd.DataFrame(),
            "bridge_df": pd.DataFrame(),
            "candidate_words": [],
            "bridge_dict_weighted": {},
            "bridge_dict_uniform": {},
            "stats": {"n_words": 0, "overlap_with_full": 0, "jaccard_with_full": 0.0, "pairs_found": ""},
        }

    chi_df = compute_chi_square(count_df, ABUSE_TYPES).sort_values("chi2", ascending=False)
    candidate_words = chi_df.head(min(config.chi_top_k, len(chi_df))).index.tolist()
    logodds_df = compute_log_odds(count_df, ABUSE_TYPES)
    bridge_df = compute_prob_bridge_for_words(
        df_counts=count_df,
        words=candidate_words,
        logodds_df=logodds_df,
        min_p1=config.bridge_min_p1,
        min_p2=config.bridge_min_p2,
        max_gap=config.bridge_max_gap,
        logodds_min=None,
        count_min=config.bridge_count_min,
        z_min=None,
    )

    full_words = set()
    if full_bridge_df is not None and not full_bridge_df.empty and "word" in full_bridge_df.columns:
        full_words = set(full_bridge_df["word"].astype(str))
    fold_words = set(bridge_df["word"].astype(str)) if not bridge_df.empty else set()
    pairs_found = sorted(
        {
            f"{pair_key(str(row.primary_abuse), str(row.secondary_abuse))[0]}|"
            f"{pair_key(str(row.primary_abuse), str(row.secondary_abuse))[1]}"
            for row in bridge_df.itertuples(index=False)
        }
    )
    overlap = len(full_words & fold_words)
    union = len(full_words | fold_words)
    stats = {
        "n_words": int(len(fold_words)),
        "overlap_with_full": int(overlap),
        "jaccard_with_full": float(overlap / union) if union else 0.0,
        "pairs_found": ";".join(pairs_found),
    }

    return {
        "count_df": count_df,
        "chi_df": chi_df,
        "logodds_df": logodds_df,
        "bridge_df": bridge_df,
        "candidate_words": candidate_words,
        "bridge_dict_weighted": bridge_df_to_dict(bridge_df, "weighted"),
        "bridge_dict_uniform": bridge_df_to_dict(bridge_df, "uniform"),
        "stats": stats,
    }


def build_random_bridge_dict(
    bridge_df: pd.DataFrame,
    candidate_words: list[str],
    rng: np.random.Generator,
) -> dict[tuple[str, str], dict[str, float]]:
    if bridge_df is None or bridge_df.empty:
        return {}

    actual_words = set(bridge_df["word"].astype(str))
    base_pool = [word for word in candidate_words if word not in actual_words]
    if not base_pool:
        base_pool = list(dict.fromkeys(candidate_words))
    if not base_pool:
        return {}

    pair_counts = Counter(
        pair_key(str(row.primary_abuse), str(row.secondary_abuse))
        for row in bridge_df.itertuples(index=False)
    )
    random_bridge: dict[tuple[str, str], dict[str, float]] = {}
    used_words: set[str] = set()

    for pair, count in pair_counts.items():
        available = [word for word in base_pool if word not in used_words]
        if len(available) < count:
            available = list(base_pool)
        if not available:
            continue
        sample_size = min(count, len(available))
        chosen = rng.choice(np.array(available, dtype=object), size=sample_size, replace=False)
        random_bridge[pair] = {str(word): 1.0 for word in np.atleast_1d(chosen)}
        used_words.update(random_bridge[pair])

    return random_bridge


def compute_bridge_score_matrix(
    X: Any,
    main_pred_indices: np.ndarray,
    bridge_dict: dict[tuple[str, str], dict[str, float]],
    feature_names: np.ndarray,
) -> np.ndarray:
    score_matrix = np.zeros((X.shape[0], len(ABUSE_TYPES)), dtype=float)
    if not bridge_dict:
        return score_matrix

    for row_idx in range(X.shape[0]):
        sparse_row = X[row_idx]
        if sparse_row.nnz == 0:
            continue
        row_map = {
            str(feature_names[col_idx]): float(value)
            for col_idx, value in zip(sparse_row.indices, sparse_row.data)
        }
        main_label = IDX_TO_TYPE[int(main_pred_indices[row_idx])]
        for candidate_idx, candidate_label in enumerate(ABUSE_TYPES):
            if candidate_idx == int(main_pred_indices[row_idx]):
                continue
            weights = bridge_dict.get(pair_key(main_label, candidate_label))
            if not weights:
                continue
            score_matrix[row_idx, candidate_idx] = float(
                sum(row_map.get(word, 0.0) * weight for word, weight in weights.items())
            )
    return score_matrix


def apply_bridge_reranking(
    scores: np.ndarray,
    main_pred_indices: np.ndarray,
    bridge_score_matrix: np.ndarray,
    lambda_value: float,
    tau: float,
) -> np.ndarray:
    adjusted_scores = scores + (lambda_value * bridge_score_matrix)
    y_pred = (adjusted_scores > tau).astype(int)
    y_pred[np.arange(len(main_pred_indices)), main_pred_indices] = 1
    return y_pred


def companion_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gt_main_indices: np.ndarray,
    pred_main_indices: np.ndarray,
) -> tuple[float, int]:
    recalls = []
    for row_idx in range(len(y_true)):
        true_subs = set(np.where(y_true[row_idx] == 1)[0]) - {int(gt_main_indices[row_idx])}
        if not true_subs:
            continue
        pred_subs = set(np.where(y_pred[row_idx] == 1)[0]) - {int(pred_main_indices[row_idx])}
        recalls.append(len(true_subs & pred_subs) / len(true_subs))
    return (float(np.mean(recalls)) if recalls else 0.0, len(recalls))


def main_preserved(y_pred: np.ndarray, gt_main_indices: np.ndarray) -> float:
    return float(np.mean(y_pred[np.arange(len(gt_main_indices)), gt_main_indices] == 1))


def boundary_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gt_main_indices: np.ndarray,
    pred_main_indices: np.ndarray,
    type_a_idx: int,
    type_b_idx: int,
) -> tuple[float, int]:
    recalls = []
    for row_idx in range(len(y_true)):
        pred_subs = set(np.where(y_pred[row_idx] == 1)[0]) - {int(pred_main_indices[row_idx])}
        main_idx = int(gt_main_indices[row_idx])
        if main_idx == type_a_idx and int(y_true[row_idx, type_b_idx]) == 1:
            recalls.append(float(type_b_idx in pred_subs))
        elif main_idx == type_b_idx and int(y_true[row_idx, type_a_idx]) == 1:
            recalls.append(float(type_a_idx in pred_subs))
    return (float(np.mean(recalls)) if recalls else 0.0, len(recalls))


def compute_stage_metrics(
    stage_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gt_main_indices: np.ndarray,
    pred_main_indices: np.ndarray,
) -> dict[str, Any]:
    companion_value, companion_cases = companion_recall(
        y_true=y_true,
        y_pred=y_pred,
        gt_main_indices=gt_main_indices,
        pred_main_indices=pred_main_indices,
    )
    boundary_phy_emo, boundary_phy_emo_n = boundary_recall(
        y_true=y_true,
        y_pred=y_pred,
        gt_main_indices=gt_main_indices,
        pred_main_indices=pred_main_indices,
        type_a_idx=TYPE_TO_IDX["신체학대"],
        type_b_idx=TYPE_TO_IDX["정서학대"],
    )
    boundary_neg_emo, boundary_neg_emo_n = boundary_recall(
        y_true=y_true,
        y_pred=y_pred,
        gt_main_indices=gt_main_indices,
        pred_main_indices=pred_main_indices,
        type_a_idx=TYPE_TO_IDX["방임"],
        type_b_idx=TYPE_TO_IDX["정서학대"],
    )
    return {
        "stage": stage_name,
        "main_preserved": main_preserved(y_pred, gt_main_indices),
        "companion_recall": companion_value,
        "companion_cases": int(companion_cases),
        "hidden_risk_miss": float(1.0 - companion_value),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "main_accuracy": float(accuracy_score(gt_main_indices, pred_main_indices)),
        "main_macro_f1": float(
            f1_score(gt_main_indices, pred_main_indices, average="macro", labels=list(range(len(ABUSE_TYPES))), zero_division=0)
        ),
        "boundary_신체_정서": boundary_phy_emo,
        "boundary_신체_정서_n": int(boundary_phy_emo_n),
        "boundary_방임_정서": boundary_neg_emo,
        "boundary_방임_정서_n": int(boundary_neg_emo_n),
    }


def flatten_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df.columns = [f"{metric}_{agg}" for metric, agg in summary_df.columns]
    return summary_df.reset_index()


def summarise_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "main_accuracy",
        "main_macro_f1",
        "main_preserved",
        "companion_recall",
        "hidden_risk_miss",
        "micro_f1",
        "macro_f1",
        "hamming_loss",
        "subset_accuracy",
        "boundary_신체_정서",
        "boundary_방임_정서",
    ]
    summary = results_df.groupby("stage")[metric_columns].agg(["mean", "std"])
    return flatten_summary(summary.round(4))


def save_latex_tables(summary_df: pd.DataFrame, output_dir: Path) -> None:
    stage_labels = {
        "A_single": "A Single",
        "B1_BR": "B1 BR",
        "B2_CC": "B2 CC",
        "C_bridge": "C Bridge",
    }
    lower_is_better = {"hamming_loss", "hidden_risk_miss"}

    def make_table(metric_specs: list[tuple[str, str]]) -> str:
        means = {
            metric: summary_df.set_index("stage")[f"{metric}_mean"].to_dict()
            for metric, _ in metric_specs
        }
        best_stage_by_metric: dict[str, str] = {}
        for metric, _ in metric_specs:
            values = means[metric]
            if metric in lower_is_better:
                best_stage_by_metric[metric] = min(values, key=values.get)
            else:
                best_stage_by_metric[metric] = max(values, key=values.get)

        lines = [
            "\\begin{tabular}{l" + "c" * len(metric_specs) + "}",
            "\\toprule",
            "Stage & " + " & ".join(label for _, label in metric_specs) + " \\\\",
            "\\midrule",
        ]
        for stage in summary_df["stage"]:
            cells = [stage_labels.get(stage, stage)]
            for metric, _ in metric_specs:
                mean_val = summary_df.loc[summary_df["stage"] == stage, f"{metric}_mean"].iloc[0]
                std_val = summary_df.loc[summary_df["stage"] == stage, f"{metric}_std"].iloc[0]
                cell = f"{mean_val:.4f} $\\pm$ {std_val:.4f}"
                if best_stage_by_metric[metric] == stage:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            lines.append(" & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        return "\n".join(lines) + "\n"

    main_table = make_table(
        [
            ("main_accuracy", "Main Acc."),
            ("main_preserved", "Main Preserved"),
            ("companion_recall", "Companion Recall"),
            ("macro_f1", "Macro F1"),
            ("hamming_loss", "Hamming Loss"),
        ]
    )
    boundary_table = make_table(
        [
            ("companion_recall", "Companion Recall"),
            ("boundary_신체_정서", "Physical-Emotional"),
            ("boundary_방임_정서", "Neglect-Emotional"),
            ("hidden_risk_miss", "Hidden-Risk Miss"),
        ]
    )
    (output_dir / "latex_tables" / "table_main_comparison.tex").write_text(main_table, encoding="utf-8")
    (output_dir / "latex_tables" / "table_boundary_recall.tex").write_text(boundary_table, encoding="utf-8")


def load_full_bridge_reference(paths: dict[str, Path]) -> pd.DataFrame:
    for key in ("bridge_words", "bridge_words_doclevel"):
        candidate = paths[key]
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame(columns=["word", "primary_abuse", "secondary_abuse", "p1", "p2", "gap", "source"])


def get_actual_splits(labels: pd.Series, requested_splits: int) -> int:
    min_count = int(labels.value_counts().min())
    return min(int(requested_splits), min_count)


def tune_bridge_parameters(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
    full_bridge_df: pd.DataFrame,
    weighting_mode: str,
    random_words: bool = False,
) -> tuple[float, float, pd.DataFrame]:
    actual_splits = get_actual_splits(train_df["main"], config.inner_splits)
    if actual_splits < 2:
        default_lambda = 1.0 if 1.0 in config.lambda_candidates else config.lambda_candidates[0]
        default_tau = 0.0 if 0.0 in config.tau_candidates else config.tau_candidates[0]
        fallback_df = pd.DataFrame(
            [
                {
                    "lambda": default_lambda,
                    "tau": default_tau,
                    "mean_companion_recall": np.nan,
                    "mean_main_preserved": np.nan,
                    "mean_micro_f1": np.nan,
                    "feasible": np.nan,
                }
            ]
        )
        return float(default_lambda), float(default_tau), fallback_df

    skf = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    aggregate_rows: list[dict[str, Any]] = []
    search_stats: dict[tuple[float, float], dict[str, list[float]]] = {
        (lambda_value, tau): defaultdict(list)
        for lambda_value in config.lambda_candidates
        for tau in config.tau_candidates
    }

    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
        skf.split(train_df["text"], train_df["main"]),
        start=1,
    ):
        inner_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)

        vectorizer, X_inner_train = safe_fit_vectorizer(inner_train_df["text"].tolist())
        X_inner_val = vectorizer.transform(inner_val_df["text"].tolist())
        y_inner_train = inner_train_df[[f"y_{label}" for label in ABUSE_TYPES]].to_numpy(dtype=int)
        y_inner_val = inner_val_df[[f"y_{label}" for label in ABUSE_TYPES]].to_numpy(dtype=int)
        gt_main_inner_val = inner_val_df["main"].map(TYPE_TO_IDX).to_numpy(dtype=int)

        single_stage = fit_single_stage(
            X_train=X_inner_train,
            y_train=inner_train_df["main"].to_numpy(),
            X_test=X_inner_val,
            random_state=config.random_state,
        )
        br_stage = fit_br_stage(
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_test=X_inner_val,
            random_state=config.random_state,
        )

        bridge_info = extract_bridge_words_from_fold(
            train_df=inner_train_df,
            config=config,
            full_bridge_df=full_bridge_df,
        )
        if random_words:
            rng = np.random.default_rng(config.random_state + inner_fold)
            bridge_dict = build_random_bridge_dict(
                bridge_df=bridge_info["bridge_df"],
                candidate_words=bridge_info["candidate_words"],
                rng=rng,
            )
        else:
            bridge_dict = bridge_info[f"bridge_dict_{weighting_mode}"]

        bridge_scores = compute_bridge_score_matrix(
            X=X_inner_val,
            main_pred_indices=single_stage.main_pred_indices,
            bridge_dict=bridge_dict,
            feature_names=vectorizer.get_feature_names_out(),
        )

        for lambda_value in config.lambda_candidates:
            for tau in config.tau_candidates:
                reranked = apply_bridge_reranking(
                    scores=br_stage.scores,
                    main_pred_indices=single_stage.main_pred_indices,
                    bridge_score_matrix=bridge_scores,
                    lambda_value=lambda_value,
                    tau=tau,
                )
                stage_metrics = compute_stage_metrics(
                    stage_name="C_bridge",
                    y_true=y_inner_val,
                    y_pred=reranked,
                    gt_main_indices=gt_main_inner_val,
                    pred_main_indices=single_stage.main_pred_indices,
                )
                key = (float(lambda_value), float(tau))
                search_stats[key]["companion_recall"].append(stage_metrics["companion_recall"])
                search_stats[key]["main_preserved"].append(stage_metrics["main_preserved"])
                search_stats[key]["micro_f1"].append(stage_metrics["micro_f1"])

    for (lambda_value, tau), metric_lists in search_stats.items():
        mean_companion = float(np.mean(metric_lists["companion_recall"]))
        mean_preserved = float(np.mean(metric_lists["main_preserved"]))
        mean_micro = float(np.mean(metric_lists["micro_f1"]))
        feasible = mean_preserved >= config.min_main_preserved_for_tuning
        aggregate_rows.append(
            {
                "lambda": lambda_value,
                "tau": tau,
                "mean_companion_recall": mean_companion,
                "mean_main_preserved": mean_preserved,
                "mean_micro_f1": mean_micro,
                "feasible": int(feasible),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["feasible", "mean_companion_recall", "mean_main_preserved", "mean_micro_f1", "lambda", "tau"],
        ascending=[False, False, False, False, True, True],
    )
    best_row = aggregate_df.iloc[0]
    return float(best_row["lambda"]), float(best_row["tau"]), aggregate_df.reset_index(drop=True)


def _run_single_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    full_bridge_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    vectorizer, X_train = safe_fit_vectorizer(train_df["text"].tolist())
    X_test = vectorizer.transform(test_df["text"].tolist())

    y_train_multi = train_df[[f"y_{label}" for label in ABUSE_TYPES]].to_numpy(dtype=int)
    y_test_multi = test_df[[f"y_{label}" for label in ABUSE_TYPES]].to_numpy(dtype=int)
    gt_main_test = test_df["main"].map(TYPE_TO_IDX).to_numpy(dtype=int)

    single_stage = fit_single_stage(
        X_train=X_train,
        y_train=train_df["main"].to_numpy(),
        X_test=X_test,
        random_state=config.random_state,
    )
    br_stage = fit_br_stage(
        X_train=X_train,
        y_train=y_train_multi,
        X_test=X_test,
        random_state=config.random_state,
    )
    cc_stage = fit_cc_stage(
        X_train=X_train,
        y_train=y_train_multi,
        X_test=X_test,
        random_state=config.random_state,
    )

    best_lambda, best_tau, tuning_df = tune_bridge_parameters(
        train_df=train_df,
        config=config,
        full_bridge_df=full_bridge_df,
        weighting_mode="weighted",
        random_words=False,
    )
    tuning_df["fold"] = fold_idx

    bridge_info = extract_bridge_words_from_fold(
        train_df=train_df,
        config=config,
        full_bridge_df=full_bridge_df,
    )
    bridge_scores = compute_bridge_score_matrix(
        X=X_test,
        main_pred_indices=single_stage.main_pred_indices,
        bridge_dict=bridge_info["bridge_dict_weighted"],
        feature_names=vectorizer.get_feature_names_out(),
    )
    bridge_stage_pred = apply_bridge_reranking(
        scores=br_stage.scores,
        main_pred_indices=single_stage.main_pred_indices,
        bridge_score_matrix=bridge_scores,
        lambda_value=best_lambda,
        tau=best_tau,
    )

    fold_rows = []
    for stage_name, stage_pred, stage_main in [
        ("A_single", single_stage.y_pred, single_stage.main_pred_indices),
        ("B1_BR", br_stage.y_pred, br_stage.main_pred_indices),
        ("B2_CC", cc_stage.y_pred, cc_stage.main_pred_indices),
        ("C_bridge", bridge_stage_pred, single_stage.main_pred_indices),
    ]:
        row = compute_stage_metrics(
            stage_name=stage_name,
            y_true=y_test_multi,
            y_pred=stage_pred,
            gt_main_indices=gt_main_test,
            pred_main_indices=stage_main,
        )
        row["fold"] = fold_idx
        fold_rows.append(row)

    if abs(fold_rows[0]["companion_recall"]) > 1e-12:
        raise AssertionError("Stage A companion recall should be exactly 0.0")
    if not np.all(br_stage.y_pred[np.arange(len(gt_main_test)), br_stage.main_pred_indices] == 1):
        raise AssertionError("Stage B1 predictions must include their predicted main label")
    if not np.all(cc_stage.y_pred[np.arange(len(gt_main_test)), cc_stage.main_pred_indices] == 1):
        raise AssertionError("Stage B2 predictions must include their predicted main label")
    if not np.all(bridge_stage_pred[np.arange(len(gt_main_test)), single_stage.main_pred_indices] == 1):
        raise AssertionError("Stage C predictions must include the anchored main label")

    bridge_stats = dict(bridge_info["stats"])
    bridge_stats["fold"] = fold_idx
    bridge_stats["lambda"] = best_lambda
    bridge_stats["tau"] = best_tau

    return fold_rows, bridge_stats, tuning_df


def run_multilabel_experiment(config: ExperimentConfig) -> dict[str, str]:
    stage_paths = default_stage_paths(config.source_root)
    dataset_df = build_multilabel_dataset(
        base_df=build_base_documents(stage_paths["data_dir"], only_negative=config.only_negative),
        sub_scores_df=load_stage_sub_scores(stage_paths["sub_scores"]),
    )
    dataset_df.to_csv(
        config.output_dir / "dataset_multilabel.csv",
        encoding="utf-8-sig",
        index=False,
    )
    save_dataset_reports(
        dataset_df=dataset_df,
        output_dir=config.output_dir,
        pair_reference_path=stage_paths["main_sub_matrix"],
        entropy_reference_path=stage_paths["entropy"],
    )

    full_bridge_df = load_full_bridge_reference(stage_paths)
    outer_splits = get_actual_splits(dataset_df["main"], config.n_splits)
    if outer_splits < 2:
        raise RuntimeError("Not enough data to run cross-validation.")

    results_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    tuning_rows: list[pd.DataFrame] = []

    skf = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(dataset_df["text"], dataset_df["main"]),
        start=1,
    ):
        train_df = dataset_df.iloc[train_idx].reset_index(drop=True)
        test_df = dataset_df.iloc[test_idx].reset_index(drop=True)
        fold_rows, bridge_stats, tuning_df = _run_single_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            test_df=test_df,
            config=config,
            full_bridge_df=full_bridge_df,
        )
        results_rows.extend(fold_rows)
        bridge_rows.append(bridge_stats)
        tuning_rows.append(tuning_df)

    results_df = pd.DataFrame(results_rows).sort_values(["fold", "stage"]).reset_index(drop=True)
    results_df.to_csv(
        config.output_dir / "multilabel_results_by_fold.csv",
        encoding="utf-8-sig",
        index=False,
    )

    summary_df = summarise_results(results_df)
    summary_df.to_csv(
        config.output_dir / "multilabel_summary.csv",
        encoding="utf-8-sig",
        index=False,
    )

    boundary_df = summary_df[
        [
            "stage",
            "companion_recall_mean",
            "companion_recall_std",
            "boundary_신체_정서_mean",
            "boundary_신체_정서_std",
            "boundary_방임_정서_mean",
            "boundary_방임_정서_std",
            "hidden_risk_miss_mean",
            "hidden_risk_miss_std",
        ]
    ].copy()
    boundary_df.to_csv(
        config.output_dir / "boundary_analysis.csv",
        encoding="utf-8-sig",
        index=False,
    )

    pd.DataFrame(bridge_rows).to_csv(
        config.output_dir / "bridge_fold_stability.csv",
        encoding="utf-8-sig",
        index=False,
    )
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        config.output_dir / "bridge_grid_search_by_fold.csv",
        encoding="utf-8-sig",
        index=False,
    )

    save_latex_tables(summary_df=summary_df, output_dir=config.output_dir)

    return {
        "dataset": str(config.output_dir / "dataset_multilabel.csv"),
        "results_by_fold": str(config.output_dir / "multilabel_results_by_fold.csv"),
        "summary": str(config.output_dir / "multilabel_summary.csv"),
        "bridge_stability": str(config.output_dir / "bridge_fold_stability.csv"),
        "boundary_analysis": str(config.output_dir / "boundary_analysis.csv"),
    }


def run_bridge_ablation_study(
    config: ExperimentConfig,
    variants: tuple[str, ...] = ("uniform", "weighted", "random"),
) -> pd.DataFrame:
    stage_paths = default_stage_paths(config.source_root)
    dataset_df = build_multilabel_dataset(
        base_df=build_base_documents(stage_paths["data_dir"], only_negative=config.only_negative),
        sub_scores_df=load_stage_sub_scores(stage_paths["sub_scores"]),
    )
    full_bridge_df = load_full_bridge_reference(stage_paths)
    ablation_dir = config.output_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    outer_splits = get_actual_splits(dataset_df["main"], config.n_splits)
    skf = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(dataset_df["text"], dataset_df["main"]),
        start=1,
    ):
        train_df = dataset_df.iloc[train_idx].reset_index(drop=True)
        test_df = dataset_df.iloc[test_idx].reset_index(drop=True)

        vectorizer, X_train = safe_fit_vectorizer(train_df["text"].tolist())
        X_test = vectorizer.transform(test_df["text"].tolist())
        y_train_multi = train_df[[f"y_{label}" for label in ABUSE_TYPES]].to_numpy(dtype=int)
        y_test_multi = test_df[[f"y_{label}" for label in ABUSE_TYPES]].to_numpy(dtype=int)
        gt_main_test = test_df["main"].map(TYPE_TO_IDX).to_numpy(dtype=int)

        single_stage = fit_single_stage(
            X_train=X_train,
            y_train=train_df["main"].to_numpy(),
            X_test=X_test,
            random_state=config.random_state,
        )
        br_stage = fit_br_stage(
            X_train=X_train,
            y_train=y_train_multi,
            X_test=X_test,
            random_state=config.random_state,
        )
        bridge_info = extract_bridge_words_from_fold(
            train_df=train_df,
            config=config,
            full_bridge_df=full_bridge_df,
        )

        for variant in variants:
            if variant == "weighted":
                bridge_dict = bridge_info["bridge_dict_weighted"]
                best_lambda, best_tau, _ = tune_bridge_parameters(
                    train_df=train_df,
                    config=config,
                    full_bridge_df=full_bridge_df,
                    weighting_mode="weighted",
                    random_words=False,
                )
            elif variant == "uniform":
                bridge_dict = bridge_info["bridge_dict_uniform"]
                best_lambda, best_tau, _ = tune_bridge_parameters(
                    train_df=train_df,
                    config=config,
                    full_bridge_df=full_bridge_df,
                    weighting_mode="uniform",
                    random_words=False,
                )
            else:
                rng = np.random.default_rng(config.random_state + fold_idx)
                bridge_dict = build_random_bridge_dict(
                    bridge_df=bridge_info["bridge_df"],
                    candidate_words=bridge_info["candidate_words"],
                    rng=rng,
                )
                best_lambda, best_tau, _ = tune_bridge_parameters(
                    train_df=train_df,
                    config=config,
                    full_bridge_df=full_bridge_df,
                    weighting_mode="uniform",
                    random_words=True,
                )

            bridge_scores = compute_bridge_score_matrix(
                X=X_test,
                main_pred_indices=single_stage.main_pred_indices,
                bridge_dict=bridge_dict,
                feature_names=vectorizer.get_feature_names_out(),
            )
            y_pred = apply_bridge_reranking(
                scores=br_stage.scores,
                main_pred_indices=single_stage.main_pred_indices,
                bridge_score_matrix=bridge_scores,
                lambda_value=best_lambda,
                tau=best_tau,
            )
            metrics = compute_stage_metrics(
                stage_name=f"C_bridge_{variant}",
                y_true=y_test_multi,
                y_pred=y_pred,
                gt_main_indices=gt_main_test,
                pred_main_indices=single_stage.main_pred_indices,
            )
            metrics["fold"] = fold_idx
            metrics["lambda"] = best_lambda
            metrics["tau"] = best_tau
            metrics["variant"] = variant
            rows.append(metrics)

    ablation_df = pd.DataFrame(rows).sort_values(["variant", "fold"]).reset_index(drop=True)
    ablation_df.to_csv(
        ablation_dir / "ablation_results_by_fold.csv",
        encoding="utf-8-sig",
        index=False,
    )
    summarise_results(ablation_df).to_csv(
        ablation_dir / "ablation_summary.csv",
        encoding="utf-8-sig",
        index=False,
    )
    return ablation_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multilabel child-abuse experiment.")
    parser.add_argument("--source-root", type=str, default=None, help="Root of the original Child_Abuse repo.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for experiment outputs.")
    parser.add_argument("--ablation", action="store_true", help="Run bridge ablations instead of the main experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(source_root=args.source_root, output_dir=args.output_dir)
    if args.ablation:
        run_bridge_ablation_study(config)
    else:
        run_multilabel_experiment(config)


if __name__ == "__main__":
    main()
