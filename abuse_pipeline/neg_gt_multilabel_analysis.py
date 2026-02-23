from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from . import common as C
from .compare_abuse_labels import extract_gt_abuse_types_from_info
from .labels import classify_child_group
from .text import extract_child_speech, tokenize_korean

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        cohen_kappa_score,
        jaccard_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MultiLabelBinarizer

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


_SEVERITY_RANK = {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}


def _labels_to_str(labels: list[str] | set[str] | tuple[str, ...]) -> str:
    if not labels:
        return ""
    return "|".join(sorted(set(labels), key=lambda x: _SEVERITY_RANK.get(x, 999)))


def _error_type(true_set: set[str], pred_set: set[str]) -> str:
    if true_set == pred_set:
        return "exact"
    miss = true_set - pred_set
    extra = pred_set - true_set
    if miss and not extra:
        return "under_predict"
    if extra and not miss:
        return "over_predict"
    return "mixed"


def _extract_neg_gt_dataset(
    json_files: list[str],
    label_order: list[str],
    gt_field: str = "학대의심",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        if classify_child_group(rec) != "부정":
            continue

        info = rec.get("info", {}) or {}
        child_id = info.get("ID") or info.get("id") or os.path.basename(path)

        gt_set = extract_gt_abuse_types_from_info(info, field=gt_field)
        gt_set = {x for x in gt_set if x in label_order}
        if not gt_set:
            continue

        speech = extract_child_speech(rec)
        if not speech:
            continue
        raw_text = " ".join(speech)
        tokens = tokenize_korean(raw_text)
        if not tokens:
            continue

        gt_main = sorted(gt_set, key=lambda x: _SEVERITY_RANK.get(x, 999))[0]
        rows.append(
            {
                "doc_id": child_id,
                "source_file": str(path),
                "tfidf_text": " ".join(tokens),
                "raw_text": raw_text,
                "gt_labels": sorted(gt_set, key=lambda x: _SEVERITY_RANK.get(x, 999)),
                "gt_main": gt_main,
                "gt_n_labels": len(gt_set),
                "has_sub": int(len(gt_set) >= 2),
            }
        )

    return pd.DataFrame(rows)


def _fit_multilabel_fold(
    x_train: list[str],
    y_train_bin: np.ndarray,
    x_test: list[str],
    random_state: int,
) -> np.ndarray:
    vec = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )
    x_train_vec = vec.fit_transform(x_train)
    x_test_vec = vec.transform(x_test)

    n_test = x_test_vec.shape[0]
    n_labels = y_train_bin.shape[1]
    probs = np.zeros((n_test, n_labels), dtype=float)

    for j in range(n_labels):
        y_col = y_train_bin[:, j]
        if np.unique(y_col).size < 2:
            probs[:, j] = float(y_col[0])
            continue

        clf = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )
        clf.fit(x_train_vec, y_col)
        probs[:, j] = clf.predict_proba(x_test_vec)[:, 1]

    return probs


def _fit_singlelabel_fold(
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    random_state: int,
) -> np.ndarray:
    vec = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )
    x_train_vec = vec.fit_transform(x_train)
    x_test_vec = vec.transform(x_test)

    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(x_train_vec, y_train)
    return clf.predict(x_test_vec)


def run_neg_gt_multilabel_study(
    json_files: list[str],
    out_dir: str,
    gt_field: str = "학대의심",
    n_splits: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    ABUSE_NEG + GT 존재 샘플에서:
      1) main+sub(다중라벨) 예측 모델
      2) main(단일라벨) 예측 모델
    을 동일 CV 기준으로 비교하고 논문용 산출물을 저장한다.
    """
    os.makedirs(out_dir, exist_ok=True)
    label_order = list(C.ABUSE_ORDER)

    df = _extract_neg_gt_dataset(
        json_files=[str(x) for x in json_files],
        label_order=label_order,
        gt_field=gt_field,
    )

    if df.empty:
        msg = "[NEG-GT-MULTI] ABUSE_NEG + GT 존재 + 텍스트 유효 샘플이 없어 분석을 건너뜁니다."
        print(msg)
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"),
            encoding="utf-8-sig",
            index=False,
        )
        return {"status": "skipped", "reason": "empty_dataset"}

    df["gt_labels_str"] = df["gt_labels"].apply(_labels_to_str)
    df.to_csv(os.path.join(out_dir, "dataset_neg_gt.csv"), encoding="utf-8-sig", index=False)

    overview = pd.DataFrame(
        [
            {"item": "n_samples", "value": int(len(df))},
            {"item": "n_main_classes", "value": int(df["gt_main"].nunique())},
            {"item": "n_with_sub", "value": int(df["has_sub"].sum())},
            {"item": "pct_with_sub", "value": float(df["has_sub"].mean() * 100.0)},
        ]
    )
    overview.to_csv(os.path.join(out_dir, "dataset_overview.csv"), encoding="utf-8-sig", index=False)

    if not HAS_SKLEARN:
        msg = "[NEG-GT-MULTI] scikit-learn 미설치로 모델 학습을 건너뜁니다."
        print(msg)
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"),
            encoding="utf-8-sig",
            index=False,
        )
        return {"status": "skipped", "reason": "sklearn_missing"}

    class_counts = df["gt_main"].value_counts()
    min_count = int(class_counts.min())
    actual_splits = min(int(n_splits), min_count)
    if actual_splits < 2 or df["gt_main"].nunique() < 2:
        msg = (
            "[NEG-GT-MULTI] 클래스 수가 부족해 CV 학습이 불가능합니다. "
            f"(n={len(df)}, classes={df['gt_main'].nunique()}, min_class_count={min_count})"
        )
        print(msg)
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"),
            encoding="utf-8-sig",
            index=False,
        )
        return {"status": "skipped", "reason": "insufficient_class_count"}

    mlb = MultiLabelBinarizer(classes=label_order)
    mlb.fit([[]])

    x_all = df["tfidf_text"].astype(str).tolist()
    y_main_all = df["gt_main"].astype(str).values
    y_bin_all = mlb.transform(df["gt_labels"].tolist())

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    pred_bin_all = np.zeros_like(y_bin_all)
    pred_main_single_all = np.array([""] * len(df), dtype=object)
    pred_main_from_multi_all = np.array([""] * len(df), dtype=object)

    for train_idx, test_idx in skf.split(np.zeros(len(y_main_all)), y_main_all):
        x_train = [x_all[i] for i in train_idx]
        x_test = [x_all[i] for i in test_idx]

        y_train_bin = y_bin_all[train_idx]
        y_train_main = y_main_all[train_idx]

        probs = _fit_multilabel_fold(
            x_train=x_train,
            y_train_bin=y_train_bin,
            x_test=x_test,
            random_state=random_state,
        )
        pred_bin = (probs >= float(threshold)).astype(int)
        zero_rows = np.where(pred_bin.sum(axis=1) == 0)[0]
        for ridx in zero_rows:
            pred_bin[ridx, int(np.argmax(probs[ridx]))] = 1

        pred_single = _fit_singlelabel_fold(
            x_train=x_train,
            y_train=y_train_main,
            x_test=x_test,
            random_state=random_state,
        )

        pred_bin_all[test_idx] = pred_bin
        pred_main_single_all[test_idx] = pred_single
        pred_main_from_multi_all[test_idx] = [label_order[int(np.argmax(r))] for r in probs]

    true_sets = [set(x) for x in df["gt_labels"].tolist()]
    pred_sets = [
        {label_order[j] for j, v in enumerate(row) if int(v) == 1}
        for row in pred_bin_all
    ]

    per_sample_rows: list[dict[str, Any]] = []
    for i, row in df.reset_index(drop=True).iterrows():
        true_set = true_sets[i]
        pred_set = pred_sets[i]
        single_correct = row["gt_main"] == pred_main_single_all[i]
        multi_exact = true_set == pred_set
        multi_main_hit = row["gt_main"] in pred_set

        miss = sorted(true_set - pred_set, key=lambda x: _SEVERITY_RANK.get(x, 999))
        extra = sorted(pred_set - true_set, key=lambda x: _SEVERITY_RANK.get(x, 999))

        per_sample_rows.append(
            {
                "doc_id": row["doc_id"],
                "source_file": row["source_file"],
                "gt_main": row["gt_main"],
                "gt_labels": _labels_to_str(true_set),
                "gt_n_labels": int(row["gt_n_labels"]),
                "has_sub": int(row["has_sub"]),
                "pred_main_single": pred_main_single_all[i],
                "pred_main_from_multi": pred_main_from_multi_all[i],
                "pred_labels_multi": _labels_to_str(pred_set),
                "single_correct": int(single_correct),
                "multi_exact_match": int(multi_exact),
                "multi_main_hit": int(multi_main_hit),
                "multi_error_type": _error_type(true_set, pred_set),
                "missing_labels_multi": _labels_to_str(miss),
                "extra_labels_multi": _labels_to_str(extra),
            }
        )

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_df.to_csv(
        os.path.join(out_dir, "per_sample_predictions.csv"),
        encoding="utf-8-sig",
        index=False,
    )

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_bin_all, pred_bin_all, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_bin_all, pred_bin_all, average="macro", zero_division=0
    )
    jaccard_samples = jaccard_score(y_bin_all, pred_bin_all, average="samples", zero_division=0)
    subset_acc = float(np.mean(np.all(y_bin_all == pred_bin_all, axis=1)))
    multi_main_hit_rate = float(per_sample_df["multi_main_hit"].mean())

    sp, sr, sf_macro, _ = precision_recall_fscore_support(
        y_main_all, pred_main_single_all, labels=label_order, average="macro", zero_division=0
    )
    _, _, sf_weighted, _ = precision_recall_fscore_support(
        y_main_all, pred_main_single_all, labels=label_order, average="weighted", zero_division=0
    )
    single_acc = float(np.mean(y_main_all == pred_main_single_all))
    single_kappa = float(cohen_kappa_score(y_main_all, pred_main_single_all, labels=label_order))

    metrics_df = pd.DataFrame(
        [
            {"model": "multilabel_main+sub", "metric": "precision_micro", "value": float(p_micro)},
            {"model": "multilabel_main+sub", "metric": "recall_micro", "value": float(r_micro)},
            {"model": "multilabel_main+sub", "metric": "f1_micro", "value": float(f_micro)},
            {"model": "multilabel_main+sub", "metric": "precision_macro", "value": float(p_macro)},
            {"model": "multilabel_main+sub", "metric": "recall_macro", "value": float(r_macro)},
            {"model": "multilabel_main+sub", "metric": "f1_macro", "value": float(f_macro)},
            {"model": "multilabel_main+sub", "metric": "subset_accuracy_exact_match", "value": subset_acc},
            {"model": "multilabel_main+sub", "metric": "jaccard_samples", "value": float(jaccard_samples)},
            {"model": "multilabel_main+sub", "metric": "main_hit_rate", "value": multi_main_hit_rate},
            {"model": "singlelabel_main", "metric": "accuracy", "value": single_acc},
            {"model": "singlelabel_main", "metric": "macro_precision", "value": float(sp)},
            {"model": "singlelabel_main", "metric": "macro_recall", "value": float(sr)},
            {"model": "singlelabel_main", "metric": "macro_f1", "value": float(sf_macro)},
            {"model": "singlelabel_main", "metric": "weighted_f1", "value": float(sf_weighted)},
            {"model": "singlelabel_main", "metric": "cohen_kappa", "value": single_kappa},
        ]
    )
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), encoding="utf-8-sig", index=False)

    p_ml, r_ml, f_ml, s_ml = precision_recall_fscore_support(
        y_bin_all, pred_bin_all, average=None, zero_division=0
    )
    per_label_multi = pd.DataFrame(
        {
            "model": "multilabel_main+sub",
            "label": label_order,
            "precision": p_ml,
            "recall": r_ml,
            "f1": f_ml,
            "support": s_ml,
        }
    )

    p_sl, r_sl, f_sl, s_sl = precision_recall_fscore_support(
        y_main_all, pred_main_single_all, labels=label_order, average=None, zero_division=0
    )
    per_label_single = pd.DataFrame(
        {
            "model": "singlelabel_main",
            "label": label_order,
            "precision": p_sl,
            "recall": r_sl,
            "f1": f_sl,
            "support": s_sl,
        }
    )
    per_label_df = pd.concat([per_label_multi, per_label_single], ignore_index=True)
    per_label_df.to_csv(os.path.join(out_dir, "per_label_metrics.csv"), encoding="utf-8-sig", index=False)

    cm = pd.crosstab(
        pd.Series(y_main_all, name="gt_main"),
        pd.Series(pred_main_single_all, name="pred_main_single"),
        dropna=False,
    ).reindex(index=label_order, columns=label_order, fill_value=0)
    cm.to_csv(os.path.join(out_dir, "confusion_matrix_singlelabel.csv"), encoding="utf-8-sig")

    top_conf_rows = []
    for t in label_order:
        row_total = int(cm.loc[t].sum()) if t in cm.index else 0
        for p in label_order:
            if t == p:
                continue
            c = int(cm.loc[t, p]) if (t in cm.index and p in cm.columns) else 0
            if c <= 0:
                continue
            top_conf_rows.append(
                {
                    "gt_main": t,
                    "pred_main": p,
                    "count": c,
                    "rate_within_gt": float(c / row_total) if row_total > 0 else np.nan,
                }
            )
    if top_conf_rows:
        top_conf_df = pd.DataFrame(top_conf_rows).sort_values(
            ["count", "rate_within_gt"], ascending=[False, False]
        )
    else:
        top_conf_df = pd.DataFrame(
            columns=["gt_main", "pred_main", "count", "rate_within_gt"]
        )
    top_conf_df.to_csv(os.path.join(out_dir, "top_confusions_singlelabel.csv"), encoding="utf-8-sig", index=False)

    region_main_sub = (
        per_sample_df.groupby(["gt_main", "has_sub"], as_index=False)
        .agg(
            n_cases=("doc_id", "count"),
            single_error_rate=("single_correct", lambda s: float(1.0 - np.mean(s))),
            multi_exact_error_rate=("multi_exact_match", lambda s: float(1.0 - np.mean(s))),
            multi_main_miss_rate=("multi_main_hit", lambda s: float(1.0 - np.mean(s))),
        )
        .sort_values(["multi_exact_error_rate", "n_cases"], ascending=[False, False])
    )
    region_main_sub.to_csv(os.path.join(out_dir, "failure_region_main_sub.csv"), encoding="utf-8-sig", index=False)

    region_combo = (
        per_sample_df.groupby(["gt_labels", "gt_n_labels"], as_index=False)
        .agg(
            n_cases=("doc_id", "count"),
            single_error_rate=("single_correct", lambda s: float(1.0 - np.mean(s))),
            multi_exact_error_rate=("multi_exact_match", lambda s: float(1.0 - np.mean(s))),
            multi_main_miss_rate=("multi_main_hit", lambda s: float(1.0 - np.mean(s))),
        )
        .sort_values(["multi_exact_error_rate", "n_cases"], ascending=[False, False])
    )
    region_combo.to_csv(os.path.join(out_dir, "failure_region_by_gt_combo.csv"), encoding="utf-8-sig", index=False)

    eligible = region_combo[region_combo["n_cases"] >= 5]
    hardest = eligible.iloc[0] if not eligible.empty else region_combo.iloc[0]
    hardest_conf = top_conf_df.iloc[0] if not top_conf_df.empty else None

    report_lines = [
        "NEG-GT Multi-label (main+sub) vs Single-label (main) comparison",
        f"- Samples (ABUSE_NEG & GT available): {len(df)}",
        f"- CV folds: {actual_splits}",
        f"- Single-label accuracy: {single_acc:.4f}",
        f"- Single-label macro F1: {float(sf_macro):.4f}",
        f"- Multi-label F1 (micro): {float(f_micro):.4f}",
        f"- Multi-label F1 (macro): {float(f_macro):.4f}",
        f"- Multi-label exact match: {subset_acc:.4f}",
        f"- Multi-label main-hit rate: {multi_main_hit_rate:.4f}",
        "",
        "[Hardest failure region]",
        f"- GT combo: {hardest['gt_labels']}",
        f"- Cases: {int(hardest['n_cases'])}",
        f"- Single error rate: {float(hardest['single_error_rate']):.4f}",
        f"- Multi exact error rate: {float(hardest['multi_exact_error_rate']):.4f}",
        f"- Multi main miss rate: {float(hardest['multi_main_miss_rate']):.4f}",
    ]
    if hardest_conf is not None:
        report_lines.extend(
            [
                "",
                "[Top single-label confusion]",
                f"- {hardest_conf['gt_main']} -> {hardest_conf['pred_main']}: {int(hardest_conf['count'])} cases",
                f"- Rate within GT={hardest_conf['gt_main']}: {float(hardest_conf['rate_within_gt']):.4f}",
            ]
        )

    with open(os.path.join(out_dir, "paper_ready_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_order": label_order,
                "gt_field": gt_field,
                "n_splits": int(actual_splits),
                "random_state": int(random_state),
                "multilabel_threshold": float(threshold),
                "vectorizer": {
                    "ngram_range": [1, 2],
                    "min_df": 1,
                    "max_features": 30000,
                    "tokenizer": "str.split",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[NEG-GT-MULTI] 분석 완료. 산출물 디렉토리: {out_dir}")
    return {
        "status": "ok",
        "n_samples": int(len(df)),
        "n_splits": int(actual_splits),
        "single_accuracy": single_acc,
        "single_macro_f1": float(sf_macro),
        "multi_f1_micro": float(f_micro),
        "multi_f1_macro": float(f_macro),
        "multi_exact_match": subset_acc,
        "multi_main_hit_rate": multi_main_hit_rate,
        "out_dir": out_dir,
    }
