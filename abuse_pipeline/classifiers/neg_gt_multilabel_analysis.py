"""
neg_gt_multilabel_analysis.py
=============================
ABUSE_NEG + GT 존재 케이스에서 main+sub(다중라벨) vs main(단일라벨) 비교.

분류기 4종:
  - TF-IDF + LogisticRegression
  - TF-IDF + RandomForest
  - TF-IDF + LinearSVM
  - KLUE-BERT (klue/bert-base)

tfidf_vs_bert_comparision.py 와 동일한 모델 설정을 사용하며,
다중라벨(Binary Relevance) / 단일라벨 양쪽 모두에 대해 Stratified K-Fold CV 를 수행한다.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from abuse_pipeline.core import common as C
from abuse_pipeline.classifiers.classifier_utils import (
    fit_multilabel_fold_tfidf,
    fit_singlelabel_fold_tfidf,
)
if C.HAS_TRANSFORMERS:
    from abuse_pipeline.classifiers.classifier_utils import fit_multilabel_fold_bert, fit_singlelabel_fold_bert
from abuse_pipeline.analysis.compare_abuse_labels import extract_gt_abuse_types_from_info
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean

# 분류기 축약명 (TF-IDF 기반)
_CLF_NAMES = ["LR", "RF", "SVM"]


# ═══════════════════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════════════════

def _labels_to_str(labels) -> str:
    if not labels:
        return ""
    return "|".join(sorted(set(labels), key=lambda x: C.SEVERITY_RANK.get(x, 999)))


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


# ═══════════════════════════════════════════════════════════════════
#  1. 데이터 추출
# ═══════════════════════════════════════════════════════════════════

def _extract_neg_gt_dataset(
    json_files: list[str],
    label_order: list[str],
    gt_field: str = "학대의심",
    compute_algo_main: bool = True,
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

        gt_main = sorted(gt_set, key=lambda x: C.SEVERITY_RANK.get(x, 999))[0]
        algo_main = ""
        has_algo_main = 0
        if compute_algo_main:
            try:
                algo_main, _ = classify_abuse_main_sub(rec)
            except Exception:
                algo_main = None
            has_algo_main = int(algo_main in label_order)

        rows.append(
            {
                "doc_id": child_id,
                "source_file": str(path),
                "tfidf_text": " ".join(tokens),
                "raw_text": raw_text,
                "gt_labels": sorted(gt_set, key=lambda x: C.SEVERITY_RANK.get(x, 999)),
                "gt_main": gt_main,
                "gt_n_labels": len(gt_set),
                "has_sub": int(len(gt_set) >= 2),
                "algo_main": algo_main if algo_main else "",
                "has_algo_main": has_algo_main,
            }
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  2. 메인 스터디 함수
# ═══════════════════════════════════════════════════════════════════

def run_neg_gt_multilabel_study(
    json_files: list[str],
    out_dir: str | None = None,
    gt_field: str = "학대의심",
    require_algo_main_for_corpus: bool = False,
    dedupe_by_doc_id: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
    multilabel_min_k: int = 0,
    bert_model_name: str = "klue/bert-base",
    bert_max_length: int = 256,
    bert_batch_size: int = 16,
    bert_epochs: int = 10,
    bert_lr: float = 2e-5,
    skip_bert: bool = False,
) -> dict[str, Any]:
    """
    ABUSE_NEG + GT 존재 샘플에서 4종 분류기(LR, RF, SVM, KLUE-BERT)를 사용하여:
      1) main+sub(다중라벨) 예측 모델
      2) main(단일라벨) 예측 모델
    을 동일 CV 기준으로 비교하고 논문용 산출물을 저장한다.
    """
    if out_dir is None:
        out_dir = C.NEG_GT_MULTILABEL_DIR or os.path.join(C.BASE_DIR, "neg_gt_multilabel")
    os.makedirs(out_dir, exist_ok=True)
    label_order = list(C.ABUSE_ORDER)

    # ── 데이터 추출 ─────────────────────────────────────────────
    df = _extract_neg_gt_dataset(
        json_files=[str(x) for x in json_files],
        label_order=label_order,
        gt_field=gt_field,
        compute_algo_main=True,
    )

    if df.empty:
        msg = "[NEG-GT-MULTI] ABUSE_NEG + GT 존재 + 텍스트 유효 샘플이 없어 분석을 건너뜁니다."
        print(msg)
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False,
        )
        return {"status": "skipped", "reason": "empty_dataset"}

    n_total_raw = int(len(df))
    n_has_algo = int(df["has_algo_main"].sum()) if "has_algo_main" in df.columns else 0
    n_no_algo = int(n_total_raw - n_has_algo)
    n_unique_doc_raw = int(df["doc_id"].nunique())
    n_dup_rows_raw = int(n_total_raw - n_unique_doc_raw)

    if require_algo_main_for_corpus:
        before = len(df)
        df = df[df["has_algo_main"] == 1].copy()
        print(f"[NEG-GT-MULTI] require_algo_main_for_corpus=True → {before} -> {len(df)}")

    if dedupe_by_doc_id:
        before = len(df)
        df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
        print(f"[NEG-GT-MULTI] dedupe_by_doc_id=True -> {before} -> {len(df)}")

    diag_rows = [
        {"item": "n_neg_gt_text_tokens_raw", "value": n_total_raw},
        {"item": "n_neg_gt_text_tokens_with_algo_main", "value": n_has_algo},
        {"item": "n_neg_gt_text_tokens_without_algo_main", "value": n_no_algo},
        {"item": "n_neg_gt_text_tokens_unique_doc_id_raw", "value": n_unique_doc_raw},
        {"item": "n_neg_gt_text_tokens_duplicate_rows_raw", "value": n_dup_rows_raw},
        {"item": "require_algo_main_for_corpus", "value": int(require_algo_main_for_corpus)},
        {"item": "dedupe_by_doc_id", "value": int(dedupe_by_doc_id)},
        {"item": "n_modeling_samples_final", "value": int(len(df))},
        {"item": "n_modeling_unique_doc_id_final", "value": int(df["doc_id"].nunique())},
    ]
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(out_dir, "corpus_alignment_diagnosis.csv"), encoding="utf-8-sig", index=False
    )

    if df.empty:
        msg = "[NEG-GT-MULTI] 필터 적용 후 학습 대상 샘플이 없어 분석을 건너뜁니다."
        print(msg)
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False,
        )
        return {"status": "skipped", "reason": "empty_after_alignment_filters"}

    df["gt_labels_str"] = df["gt_labels"].apply(_labels_to_str)
    df.to_csv(os.path.join(out_dir, "dataset_neg_gt.csv"), encoding="utf-8-sig", index=False)

    overview = pd.DataFrame([
        {"item": "n_samples", "value": int(len(df))},
        {"item": "n_unique_doc_id", "value": int(df["doc_id"].nunique())},
        {"item": "n_main_classes", "value": int(df["gt_main"].nunique())},
        {"item": "n_with_sub", "value": int(df["has_sub"].sum())},
        {"item": "pct_with_sub", "value": float(df["has_sub"].mean() * 100.0)},
        {"item": "n_with_algo_main", "value": int(df["has_algo_main"].sum())},
        {"item": "n_without_algo_main", "value": int((df["has_algo_main"] == 0).sum())},
    ])
    overview.to_csv(os.path.join(out_dir, "dataset_overview.csv"), encoding="utf-8-sig", index=False)

    if not C.HAS_SKLEARN:
        msg = "[NEG-GT-MULTI] scikit-learn 미설치로 모델 학습을 건너뜁니다."
        print(msg)
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False,
        )
        return {"status": "skipped", "reason": "sklearn_missing"}

    # ── CV 설정 ─────────────────────────────────────────────────
    class_counts = df["gt_main"].value_counts()
    min_count = int(class_counts.min())
    actual_splits = min(int(n_splits), min_count)
    if actual_splits < 2 or df["gt_main"].nunique() < 2:
        msg = (
            f"[NEG-GT-MULTI] 클래스 수 부족 (n={len(df)}, "
            f"classes={df['gt_main'].nunique()}, min_class={min_count})"
        )
        print(msg)
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False,
        )
        return {"status": "skipped", "reason": "insufficient_class_count"}

    mlb = C.MultiLabelBinarizer(classes=label_order)
    mlb.fit([[]])

    x_tfidf = df["tfidf_text"].astype(str).tolist()
    x_raw = df["raw_text"].astype(str).tolist()
    y_main_all = df["gt_main"].astype(str).values
    y_bin_all = mlb.transform(df["gt_labels"].tolist())

    label2id = {l: i for i, l in enumerate(label_order)}
    id2label = {i: l for l, i in label2id.items()}
    y_main_int = np.array([label2id[l] for l in y_main_all])

    skf = C.StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    # ── 분류기 목록 ─────────────────────────────────────────────
    use_bert = C.HAS_TRANSFORMERS and not skip_bert
    clf_list = list(_CLF_NAMES)
    if use_bert:
        clf_list.append("BERT")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [BERT] device = {device}")
    else:
        device = None

    # ── 예측 저장 구조 ──────────────────────────────────────────
    pred_bin_by_clf = {c: np.zeros_like(y_bin_all) for c in clf_list}
    pred_single_by_clf = {c: np.array([""] * len(df), dtype=object) for c in clf_list}
    pred_main_from_multi_by_clf = {c: np.array([""] * len(df), dtype=object) for c in clf_list}

    # ── CV 루프 ─────────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(y_main_all)), y_main_all), start=1
    ):
        print(f"\n  [Fold {fold_idx}/{actual_splits}] Train={len(train_idx)}, Test={len(test_idx)}")

        x_tr_tfidf = [x_tfidf[i] for i in train_idx]
        x_te_tfidf = [x_tfidf[i] for i in test_idx]
        x_tr_raw = [x_raw[i] for i in train_idx]
        x_te_raw = [x_raw[i] for i in test_idx]
        y_tr_bin = y_bin_all[train_idx]
        y_tr_main = y_main_all[train_idx]
        y_tr_main_int = y_main_int[train_idx]
        train_cardinality = float(np.mean(y_tr_bin.sum(axis=1)))
        fallback_k = int(multilabel_min_k) if int(multilabel_min_k) > 0 else int(np.ceil(train_cardinality))
        fallback_k = max(1, min(fallback_k, len(label_order)))

        # TF-IDF 분류기 3종
        for cname in _CLF_NAMES:
            print(f"    [{cname}] multilabel + singlelabel ...")

            probs = None
            try:
                probs = fit_multilabel_fold_tfidf(x_tr_tfidf, y_tr_bin, x_te_tfidf, cname, random_state)
                _store_multilabel_preds(
                    probs=probs,
                    threshold=threshold,
                    label_order=label_order,
                    test_idx=test_idx,
                    pred_bin_arr=pred_bin_by_clf[cname],
                    pred_main_arr=pred_main_from_multi_by_clf[cname],
                    fallback_k=fallback_k,
                )
            except Exception as e:
                print(f"      [WARN][{cname}] multilabel fold 실패: {e}")
                label_prior = np.mean(y_tr_bin, axis=0)
                top_idx = np.argsort(label_prior)[::-1][:fallback_k]
                fallback_bin = np.zeros((len(test_idx), len(label_order)), dtype=int)
                fallback_bin[:, top_idx] = 1
                pred_bin_by_clf[cname][test_idx] = fallback_bin
                pred_main_from_multi_by_clf[cname][test_idx] = [label_order[int(top_idx[0])]] * len(test_idx)

            try:
                pred_single_by_clf[cname][test_idx] = fit_singlelabel_fold_tfidf(
                    x_tr_tfidf, y_tr_main, x_te_tfidf, cname, random_state
                )
            except Exception as e:
                print(f"      [WARN][{cname}] singlelabel fold 실패: {e}")
                if probs is not None:
                    pred_single_by_clf[cname][test_idx] = [label_order[int(np.argmax(r))] for r in probs]
                else:
                    pred_single_by_clf[cname][test_idx] = [str(y_tr_main[0])] * len(test_idx)

        # KLUE-BERT
        if use_bert:
            cname = "BERT"
            print(f"    [{cname}] multilabel + singlelabel ...")
            probs_bert = fit_multilabel_fold_bert(
                x_tr_raw, y_tr_bin, x_te_raw, n_labels=len(label_order),
                model_name=bert_model_name, max_length=bert_max_length,
                batch_size=bert_batch_size, epochs=bert_epochs,
                learning_rate=bert_lr, device=device,
            )
            if probs_bert is not None:
                _store_multilabel_preds(
                    probs=probs_bert,
                    threshold=threshold,
                    label_order=label_order,
                    test_idx=test_idx,
                    pred_bin_arr=pred_bin_by_clf[cname],
                    pred_main_arr=pred_main_from_multi_by_clf[cname],
                    fallback_k=fallback_k,
                )
            else:
                label_prior = np.mean(y_tr_bin, axis=0)
                top_idx = np.argsort(label_prior)[::-1][:fallback_k]
                fallback_bin = np.zeros((len(test_idx), len(label_order)), dtype=int)
                fallback_bin[:, top_idx] = 1
                pred_bin_by_clf[cname][test_idx] = fallback_bin
                pred_main_from_multi_by_clf[cname][test_idx] = [label_order[int(top_idx[0])]] * len(test_idx)

            pred_single_bert = fit_singlelabel_fold_bert(
                x_tr_raw, y_tr_main_int, x_te_raw,
                label2id=label2id, id2label=id2label,
                model_name=bert_model_name, max_length=bert_max_length,
                batch_size=bert_batch_size, epochs=bert_epochs,
                learning_rate=bert_lr, device=device,
            )
            if pred_single_bert is not None:
                pred_single_by_clf[cname][test_idx] = pred_single_bert

    # ── 메트릭 산출 & 저장 ──────────────────────────────────────
    true_sets = [set(x) for x in df["gt_labels"].tolist()]
    _save_all_outputs(
        out_dir, df, label_order, clf_list, y_bin_all, y_main_all, true_sets,
        pred_bin_by_clf, pred_single_by_clf, pred_main_from_multi_by_clf,
        actual_splits, gt_field, random_state, threshold,
        use_bert, bert_model_name, bert_max_length, bert_batch_size, bert_epochs, bert_lr,
        multilabel_min_k, require_algo_main_for_corpus, dedupe_by_doc_id,
    )

    print(f"\n[NEG-GT-MULTI] 분석 완료. 산출물: {out_dir}")
    return {
        "status": "ok",
        "n_samples": int(len(df)),
        "n_unique_doc_id": int(df["doc_id"].nunique()),
        "n_splits": int(actual_splits),
        "multilabel_min_k": int(multilabel_min_k),
        "require_algo_main_for_corpus": bool(require_algo_main_for_corpus),
        "dedupe_by_doc_id": bool(dedupe_by_doc_id),
        "classifiers": clf_list,
        "out_dir": out_dir,
    }


# ═══════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ═══════════════════════════════════════════════════════════════════

def _store_multilabel_preds(
    probs,
    threshold,
    label_order,
    test_idx,
    pred_bin_arr,
    pred_main_arr,
    fallback_k: int = 1,
):
    """확률 → 이진 예측 변환 후 저장.

    fallback_k:
        threshold 기준으로 선택된 라벨 수가 부족한 경우 최소한 확보할 라벨 수.
        (train fold 평균 label cardinality 기반으로 호출부에서 설정)
    """
    pred_bin = (probs >= float(threshold)).astype(int)
    fallback_k = max(1, min(int(fallback_k), pred_bin.shape[1]))
    low_rows = np.where(pred_bin.sum(axis=1) < fallback_k)[0]
    for ridx in low_rows:
        keep = set(np.where(pred_bin[ridx] == 1)[0].tolist())
        for j in np.argsort(probs[ridx])[::-1]:
            keep.add(int(j))
            if len(keep) >= fallback_k:
                break
        pred_bin[ridx] = 0
        pred_bin[ridx, list(keep)] = 1
    pred_bin_arr[test_idx] = pred_bin
    pred_main_arr[test_idx] = [label_order[int(np.argmax(r))] for r in probs]


def _save_all_outputs(
    out_dir, df, label_order, clf_list, y_bin_all, y_main_all, true_sets,
    pred_bin_by_clf, pred_single_by_clf, pred_main_from_multi_by_clf,
    actual_splits, gt_field, random_state, threshold,
    use_bert, bert_model_name, bert_max_length, bert_batch_size, bert_epochs, bert_lr,
    multilabel_min_k, require_algo_main_for_corpus, dedupe_by_doc_id,
):
    """모든 산출물을 CSV/JSON/TXT 로 저장."""
    metrics_rows = []
    per_label_rows = []
    per_sample_rows = []
    cardinality_rows = []

    for cname in clf_list:
        pred_bin_c = pred_bin_by_clf[cname]
        pred_single_c = np.array(
            [
                pred_single_by_clf[cname][i]
                if isinstance(pred_single_by_clf[cname][i], str) and pred_single_by_clf[cname][i]
                else pred_main_from_multi_by_clf[cname][i]
                for i in range(len(df))
            ],
            dtype=object,
        )
        pred_single_by_clf[cname] = pred_single_c
        pred_main_multi_c = pred_main_from_multi_by_clf[cname]
        pred_sets = [{label_order[j] for j, v in enumerate(row) if int(v) == 1} for row in pred_bin_c]

        # ── 다중라벨 메트릭 ──
        p_mi, r_mi, f_mi, _ = C.precision_recall_fscore_support(y_bin_all, pred_bin_c, average="micro", zero_division=0)
        p_ma, r_ma, f_ma, _ = C.precision_recall_fscore_support(y_bin_all, pred_bin_c, average="macro", zero_division=0)
        jac = C.jaccard_score(y_bin_all, pred_bin_c, average="samples", zero_division=0)
        subset_acc = float(np.mean(np.all(y_bin_all == pred_bin_c, axis=1)))
        multi_main_hits = [int(df.iloc[i]["gt_main"] in pred_sets[i]) for i in range(len(df))]
        multi_main_hit_rate = float(np.mean(multi_main_hits))

        for m, v in [("precision_micro", p_mi), ("recall_micro", r_mi), ("f1_micro", f_mi),
                      ("precision_macro", p_ma), ("recall_macro", r_ma), ("f1_macro", f_ma),
                      ("subset_accuracy", subset_acc), ("jaccard_samples", jac), ("main_hit_rate", multi_main_hit_rate)]:
            metrics_rows.append({"scenario": "multilabel_main+sub", "classifier": cname, "metric": m, "value": float(v)})

        p_ml, r_ml, f_ml, s_ml = C.precision_recall_fscore_support(y_bin_all, pred_bin_c, average=None, zero_division=0)
        for i, lbl in enumerate(label_order):
            per_label_rows.append({"scenario": "multilabel_main+sub", "classifier": cname, "label": lbl,
                                   "precision": float(p_ml[i]), "recall": float(r_ml[i]), "f1": float(f_ml[i]), "support": int(s_ml[i])})
        pred_cardinality = pred_bin_c.sum(axis=1).astype(int)
        true_cardinality = y_bin_all.sum(axis=1).astype(int)
        cardinality_rows.append(
            {
                "classifier": cname,
                "n_samples": int(len(df)),
                "true_label_cardinality_mean": float(np.mean(true_cardinality)),
                "pred_label_cardinality_mean": float(np.mean(pred_cardinality)),
                "pct_true_ge2": float(np.mean(true_cardinality >= 2) * 100.0),
                "pct_pred_ge2": float(np.mean(pred_cardinality >= 2) * 100.0),
                "pct_pred_exact_cardinality": float(np.mean(pred_cardinality == true_cardinality) * 100.0),
            }
        )

        # ── 단일라벨 메트릭 ──
        sp, sr, sf_ma, _ = C.precision_recall_fscore_support(y_main_all, pred_single_c, labels=label_order, average="macro", zero_division=0)
        _, _, sf_w, _ = C.precision_recall_fscore_support(y_main_all, pred_single_c, labels=label_order, average="weighted", zero_division=0)
        single_acc = float(np.mean(y_main_all == pred_single_c))
        single_kappa = float(C.cohen_kappa_score(y_main_all, pred_single_c, labels=label_order))

        for m, v in [("accuracy", single_acc), ("macro_precision", sp), ("macro_recall", sr),
                      ("macro_f1", sf_ma), ("weighted_f1", sf_w), ("cohen_kappa", single_kappa)]:
            metrics_rows.append({"scenario": "singlelabel_main", "classifier": cname, "metric": m, "value": float(v)})

        p_sl, r_sl, f_sl, s_sl = C.precision_recall_fscore_support(y_main_all, pred_single_c, labels=label_order, average=None, zero_division=0)
        for i, lbl in enumerate(label_order):
            per_label_rows.append({"scenario": "singlelabel_main", "classifier": cname, "label": lbl,
                                   "precision": float(p_sl[i]), "recall": float(r_sl[i]), "f1": float(f_sl[i]), "support": int(s_sl[i])})

        # ── 혼동행렬 (단일라벨) ──
        cm = pd.crosstab(
            pd.Series(y_main_all, name="gt_main"),
            pd.Series(pred_single_c, name="pred_main"),
            dropna=False,
        ).reindex(index=label_order, columns=label_order, fill_value=0)
        cm.to_csv(os.path.join(out_dir, f"confusion_matrix_singlelabel_{cname}.csv"), encoding="utf-8-sig")
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
                        "classifier": cname,
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
                columns=["classifier", "gt_main", "pred_main", "count", "rate_within_gt"]
            )
        top_conf_df.to_csv(
            os.path.join(out_dir, f"top_confusions_singlelabel_{cname}.csv"),
            encoding="utf-8-sig",
            index=False,
        )

        # ── 샘플별 예측 ──
        for i, row in df.reset_index(drop=True).iterrows():
            ts = true_sets[i]
            ps = pred_sets[i]
            miss = sorted(ts - ps, key=lambda x: C.SEVERITY_RANK.get(x, 999))
            extra = sorted(ps - ts, key=lambda x: C.SEVERITY_RANK.get(x, 999))
            per_sample_rows.append({
                "classifier": cname,
                "doc_id": row["doc_id"],
                "source_file": row["source_file"],
                "gt_main": row["gt_main"],
                "gt_labels": _labels_to_str(ts),
                "gt_n_labels": int(row["gt_n_labels"]),
                "has_sub": int(row["has_sub"]),
                "pred_main_single": pred_single_c[i],
                "pred_main_from_multi": pred_main_multi_c[i],
                "pred_labels_multi": _labels_to_str(ps),
                "pred_n_labels_multi": int(len(ps)),
                "single_correct": int(row["gt_main"] == pred_single_c[i]),
                "multi_exact_match": int(ts == ps),
                "multi_main_hit": int(row["gt_main"] in ps),
                "multi_error_type": _error_type(ts, ps),
                "missing_labels_multi": _labels_to_str(miss),
                "extra_labels_multi": _labels_to_str(extra),
            })

    # 저장
    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "metrics_summary.csv"), encoding="utf-8-sig", index=False)
    pd.DataFrame(per_label_rows).to_csv(os.path.join(out_dir, "per_label_metrics.csv"), encoding="utf-8-sig", index=False)
    pd.DataFrame(cardinality_rows).to_csv(
        os.path.join(out_dir, "multilabel_cardinality_report.csv"), encoding="utf-8-sig", index=False
    )
    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_df.to_csv(os.path.join(out_dir, "per_sample_predictions.csv"), encoding="utf-8-sig", index=False)

    # 실패 지역
    region_main_sub = (
        per_sample_df.groupby(["classifier", "gt_main", "has_sub"], as_index=False)
        .agg(
            n_cases=("doc_id", "count"),
            single_error_rate=("single_correct", lambda s: float(1.0 - np.mean(s))),
            multi_exact_error_rate=("multi_exact_match", lambda s: float(1.0 - np.mean(s))),
            multi_main_miss_rate=("multi_main_hit", lambda s: float(1.0 - np.mean(s))),
        )
        .sort_values(["classifier", "multi_exact_error_rate", "n_cases"], ascending=[True, False, False])
    )
    region_main_sub.to_csv(os.path.join(out_dir, "failure_region_main_sub.csv"), encoding="utf-8-sig", index=False)

    region_combo = (
        per_sample_df.groupby(["classifier", "gt_labels", "gt_n_labels"], as_index=False)
        .agg(
            n_cases=("doc_id", "count"),
            single_error_rate=("single_correct", lambda s: float(1.0 - np.mean(s))),
            multi_exact_error_rate=("multi_exact_match", lambda s: float(1.0 - np.mean(s))),
            multi_main_miss_rate=("multi_main_hit", lambda s: float(1.0 - np.mean(s))),
        )
        .sort_values(["classifier", "multi_exact_error_rate", "n_cases"], ascending=[True, False, False])
    )
    region_combo.to_csv(os.path.join(out_dir, "failure_region_by_gt_combo.csv"), encoding="utf-8-sig", index=False)

    # 모호성 지대 분석
    if len(clf_list) >= 2:
        _save_ambiguity_analysis(df, y_main_all, pred_single_by_clf, clf_list, out_dir)

    # model_config.json
    config: dict[str, Any] = {
        "label_order": label_order, "gt_field": gt_field,
        "n_splits": int(actual_splits), "random_state": int(random_state),
        "multilabel_threshold": float(threshold), "classifiers": clf_list,
        "multilabel_min_k": int(multilabel_min_k),
        "multilabel_min_k_note": "0이면 train fold 평균 GT cardinality의 ceil 값을 사용",
        "require_algo_main_for_corpus": bool(require_algo_main_for_corpus),
        "dedupe_by_doc_id": bool(dedupe_by_doc_id),
        "tfidf_params": {"ngram_range": list(C.TFIDF_PARAMS["ngram_range"]),
                         "min_df": C.TFIDF_PARAMS["min_df"], "max_features": C.TFIDF_PARAMS["max_features"]},
    }
    if use_bert:
        config["bert"] = {"model_name": bert_model_name, "max_length": bert_max_length,
                          "batch_size": bert_batch_size, "epochs": bert_epochs, "learning_rate": bert_lr}
    import json as _json
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        _json.dump(config, f, ensure_ascii=False, indent=2)

    # 요약 보고서
    _save_summary_report(out_dir, df, actual_splits, pd.DataFrame(metrics_rows), clf_list)


def _save_ambiguity_analysis(df, y_main_all, pred_single_by_clf, clf_list, out_dir):
    """모호성 지대 분석: 모든 분류기가 공통으로 오분류하는 패턴."""
    amb_rows = []
    for idx in range(len(df)):
        true_label = y_main_all[idx]
        preds = {c: pred_single_by_clf[c][idx] for c in clf_list}
        n_wrong = sum(1 for p in preds.values() if p != true_label)
        wrong_labels = [p for p in preds.values() if p != true_label]
        wrong_consensus = len(set(wrong_labels)) == 1 if wrong_labels else False

        row: dict[str, Any] = {
            "doc_id": df.iloc[idx]["doc_id"], "true_label": true_label,
            "n_correct": len(clf_list) - n_wrong, "n_wrong": n_wrong,
            "all_wrong": int(n_wrong == len(clf_list)),
            "wrong_consensus": int(wrong_consensus),
            "wrong_consensus_label": wrong_labels[0] if wrong_consensus and wrong_labels else None,
        }
        for c in clf_list:
            row[f"pred_{c}"] = preds[c]
        amb_rows.append(row)

    df_amb = pd.DataFrame(amb_rows)
    df_amb.to_csv(os.path.join(out_dir, "ambiguity_zone_singlelabel.csv"), encoding="utf-8-sig", index=False)

    n_total = len(df_amb)
    n_all_wrong = int(df_amb["all_wrong"].sum())
    n_consensus = int(df_amb["wrong_consensus"].sum())

    if n_consensus > 0:
        consensus_sub = df_amb[df_amb["wrong_consensus"] == 1]
        (consensus_sub.groupby(["true_label", "wrong_consensus_label"]).size()
         .reset_index(name="count").sort_values("count", ascending=False)
         .to_csv(os.path.join(out_dir, "model_invariant_misclass_patterns.csv"), encoding="utf-8-sig", index=False))

    pd.DataFrame([{
        "n_total": n_total, "n_all_wrong": n_all_wrong, "n_model_invariant": n_consensus,
        "pct_all_wrong": n_all_wrong / n_total * 100 if n_total else 0,
        "pct_invariant_of_all_wrong": n_consensus / n_all_wrong * 100 if n_all_wrong else 0,
        "interpretation": (
            "HIGH" if n_all_wrong and n_consensus / n_all_wrong > 0.7
            else "MODERATE" if n_all_wrong and n_consensus / n_all_wrong > 0.4
            else "LOW"
        ),
    }]).to_csv(os.path.join(out_dir, "ambiguity_zone_summary.csv"), encoding="utf-8-sig", index=False)


def _save_summary_report(out_dir, df, actual_splits, metrics_df, clf_list):
    lines = [
        "NEG-GT Multi-label (main+sub) vs Single-label (main) comparison",
        f"  Samples: {len(df)}, CV folds: {actual_splits}, Classifiers: {', '.join(clf_list)}",
        "",
    ]
    for cname in clf_list:
        lines.append(f"=== {cname} ===")
        for _, r in metrics_df[metrics_df["classifier"] == cname].iterrows():
            lines.append(f"  [{r['scenario']}] {r['metric']}: {r['value']:.4f}")
        lines.append("")

    with open(os.path.join(out_dir, "paper_ready_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
