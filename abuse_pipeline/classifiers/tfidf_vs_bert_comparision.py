"""
tfidf_vs_bert_comparison.py
============================
TF-IDF 기반 분류기 vs KLUE-BERT 트랜스포머 모델 비교 분석

4종 분류기:
  (1) TF-IDF + LogisticRegression
  (2) TF-IDF + RandomForest
  (3) TF-IDF + LinearSVM
  (4) KLUE-BERT (klue/bert-base) 파인튜닝

classifier_utils.py 의 공용 코드를 사용하여 중복을 제거한 리팩토링 버전.
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from abuse_pipeline.core import common as C
from abuse_pipeline.classifiers.classifier_utils import (
    make_singlelabel_clf,
    run_tfidf_classifiers_cv,
)
if C.HAS_TRANSFORMERS:
    from abuse_pipeline.classifiers.classifier_utils import fit_singlelabel_fold_bert
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean


# ═══════════════════════════════════════════════════════════════════
#  1. 데이터 준비: JSON → DataFrame
# ═══════════════════════════════════════════════════════════════════

def prepare_classification_data(
    json_files: List[str],
    abuse_order: List[str],
    only_negative: bool = True,
) -> pd.DataFrame:
    """
    JSON 파일들에서 아동별 발화 텍스트와 학대유형 라벨을 추출한다.

    Returns
    -------
    pd.DataFrame
        columns = ["ID", "main_abuse", "tfidf_text", "raw_text"]
    """
    rows = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        info = rec.get("info", {})
        child_id = info.get("ID") or info.get("id")

        group = classify_child_group(rec)
        if only_negative and group != "부정":
            continue

        main_abuse, _ = classify_abuse_main_sub(rec)
        if main_abuse not in abuse_order:
            continue

        speech_list = extract_child_speech(rec)
        if not speech_list:
            continue

        raw_text = " ".join(speech_list)
        tokens = tokenize_korean(raw_text)
        tfidf_text = " ".join(tokens)

        if not tfidf_text.strip():
            continue

        rows.append({
            "ID": child_id,
            "main_abuse": main_abuse,
            "tfidf_text": tfidf_text,
            "raw_text": raw_text,
        })

    df = pd.DataFrame(rows)
    print(f"[DATA] 총 {len(df)}명 아동 데이터 준비 완료")
    print(f"[DATA] 라벨 분포:")
    for a in abuse_order:
        n = (df["main_abuse"] == a).sum()
        print(f"  {a}: {n}명 ({n / len(df) * 100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════
#  2. TF-IDF 기반 분류기 3종
# ═══════════════════════════════════════════════════════════════════

def run_tfidf_classifiers(
    df: pd.DataFrame,
    label_col: str,
    label_order: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Optional[Dict]:
    """
    TF-IDF + {LR, RF, SVM} 3종 분류기를 Stratified K-Fold CV 로 평가.

    Returns
    -------
    dict: per_label_metrics, overall_metrics, all_true, all_pred, per_sample
    """
    if not C.HAS_SKLEARN:
        print("[SKIP] scikit-learn 미설치")
        return None

    texts = df["tfidf_text"].astype(str).tolist()
    y = df[label_col].astype(str).values

    cv_results = run_tfidf_classifiers_cv(
        texts=texts, y=y, label_order=label_order,
        clf_names=["LR", "RF", "SVM"],
        n_splits=n_splits, random_state=random_state,
    )
    if not cv_results:
        return None

    # 분류기 이름 매핑 (출력용)
    _name_map = {
        "LR": "TF-IDF + LogisticRegression",
        "RF": "TF-IDF + RandomForest",
        "SVM": "TF-IDF + LinearSVM",
    }

    per_label_rows = []
    overall_rows = []
    all_true_dict = {}
    all_pred_dict = {}
    per_sample_dict = {}

    for cname, res in cv_results.items():
        display_name = _name_map.get(cname, cname)
        true = res["all_true"]
        pred = res["all_pred"]

        all_true_dict[display_name] = true
        all_pred_dict[display_name] = pred
        per_sample_dict[display_name] = res["per_sample"]

        prec, rec, f1, sup = C.precision_recall_fscore_support(
            true, pred, labels=label_order, zero_division=0,
        )
        for i, label in enumerate(label_order):
            per_label_rows.append({
                "classifier": display_name, "label": label,
                "precision": prec[i], "recall": rec[i],
                "f1_score": f1[i], "support": int(sup[i]),
            })

        report = C.classification_report(
            true, pred, labels=label_order, output_dict=True, zero_division=0,
        )
        overall_rows.append({
            "classifier": display_name,
            "accuracy": report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "cohen_kappa": C.cohen_kappa_score(true, pred, labels=label_order),
        })

        acc = np.mean([t == p for t, p in zip(true, pred)])
        kappa = C.cohen_kappa_score(true, pred, labels=label_order)
        print(f"    {display_name} -> Acc: {acc:.4f}, kappa: {kappa:.4f}")

    return {
        "per_label_metrics": pd.DataFrame(per_label_rows),
        "overall_metrics": pd.DataFrame(overall_rows),
        "all_true": all_true_dict,
        "all_pred": all_pred_dict,
        "per_sample": per_sample_dict,
    }


# ═══════════════════════════════════════════════════════════════════
#  3. KLUE-BERT 파인튜닝 분류기
# ═══════════════════════════════════════════════════════════════════

def run_bert_classifier(
    df: pd.DataFrame,
    label_col: str,
    label_order: List[str],
    model_name: str = "klue/bert-base",
    n_splits: int = 5,
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    random_state: int = 42,
) -> Optional[Dict]:
    """KLUE-BERT 를 Stratified K-Fold CV 로 평가."""
    if not C.HAS_TRANSFORMERS:
        print("[SKIP] transformers/torch 미설치")
        return None

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [BERT] 모델: {model_name}, 디바이스: {device}")

    texts = df["raw_text"].astype(str).tolist()
    y_str = df[label_col].astype(str).values

    label2id = {label: i for i, label in enumerate(label_order)}
    id2label = {i: label for label, i in label2id.items()}
    y = np.array([label2id[l] for l in y_str])

    min_count = int(df[label_col].value_counts().min())
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        print(f"[SKIP] 최소 클래스 샘플 수 = {min_count}")
        return None

    skf = C.StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    clf_name = f"KLUE-BERT ({model_name})"
    all_true, all_pred = [], []
    sample_preds = {}

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(y)), y), start=1
    ):
        print(f"    [Fold {fold_idx}/{actual_splits}] Train: {len(train_idx)}, Test: {len(test_idx)}")

        X_train = [texts[i] for i in train_idx]
        y_train = y[train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = y[test_idx]

        preds_str = fit_singlelabel_fold_bert(
            x_train_raw=X_train, y_train=y_train, x_test_raw=X_test,
            label2id=label2id, id2label=id2label,
            model_name=model_name, max_length=max_length,
            batch_size=batch_size, epochs=epochs,
            learning_rate=learning_rate, device=device,
        )

        y_test_str = [id2label[int(i)] for i in y_test]
        all_true.extend(y_test_str)
        all_pred.extend(list(preds_str))

        for i, idx in enumerate(test_idx):
            sample_preds[int(idx)] = preds_str[i]

        fold_acc = np.mean([t == p for t, p in zip(y_test_str, preds_str)])
        print(f"      Fold {fold_idx} Acc: {fold_acc:.4f}")

    acc = np.mean([t == p for t, p in zip(all_true, all_pred)])
    kappa = C.cohen_kappa_score(all_true, all_pred, labels=label_order)
    print(f"\n  [BERT] Acc: {acc:.4f}, kappa: {kappa:.4f}")

    prec, rec, f1, sup = C.precision_recall_fscore_support(
        all_true, all_pred, labels=label_order, zero_division=0,
    )
    per_label_rows = [
        {"classifier": clf_name, "label": label_order[i],
         "precision": prec[i], "recall": rec[i], "f1_score": f1[i], "support": int(sup[i])}
        for i in range(len(label_order))
    ]

    report = C.classification_report(
        all_true, all_pred, labels=label_order, output_dict=True, zero_division=0,
    )
    overall_rows = [{
        "classifier": clf_name,
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "cohen_kappa": kappa,
    }]

    return {
        "per_label_metrics": pd.DataFrame(per_label_rows),
        "overall_metrics": pd.DataFrame(overall_rows),
        "all_true": {clf_name: all_true},
        "all_pred": {clf_name: all_pred},
        "per_sample": {clf_name: sample_preds},
    }


# ═══════════════════════════════════════════════════════════════════
#  4. 통합 비교: TF-IDF 3종 + KLUE-BERT
# ═══════════════════════════════════════════════════════════════════

def run_full_model_comparison(
    json_files: List[str],
    abuse_order: List[str] = None,
    out_dir: str | None = None,
    only_negative: bool = True,
    n_splits: int = 5,
    bert_model_name: str = "klue/bert-base",
    bert_max_length: int = 256,
    bert_batch_size: int = 16,
    bert_epochs: int = 10,
    bert_lr: float = 2e-5,
    random_state: int = 42,
) -> Optional[Dict]:
    """TF-IDF 기반 분류기 3종 + KLUE-BERT 분류기의 성능을 통합 비교."""
    if abuse_order is None:
        abuse_order = list(C.ABUSE_ORDER)

    if out_dir is None:
        out_dir = C.MODEL_COMPARISON_DIR or os.path.join(C.BASE_DIR, "model_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 데이터 준비
    print("=" * 72)
    print("  [STEP 1] 데이터 준비")
    print("=" * 72)
    df = prepare_classification_data(json_files, abuse_order, only_negative)

    if df.empty or df["main_abuse"].nunique() < 2:
        print("[ERROR] 유효한 데이터가 부족합니다.")
        return None

    df.to_csv(os.path.join(out_dir, "classification_data.csv"), encoding="utf-8-sig", index=False)

    # 2. TF-IDF 분류기
    print("\n" + "=" * 72)
    print("  [STEP 2] TF-IDF 기반 분류기 3종")
    print("=" * 72)
    tfidf_results = run_tfidf_classifiers(
        df=df, label_col="main_abuse", label_order=abuse_order,
        n_splits=n_splits, random_state=random_state,
    )

    # 3. KLUE-BERT
    bert_results = None
    print("\n" + "=" * 72)
    print("  [STEP 3] KLUE-BERT 파인튜닝 분류기")
    print("=" * 72)
    if C.HAS_TRANSFORMERS:
        bert_results = run_bert_classifier(
            df=df, label_col="main_abuse", label_order=abuse_order,
            model_name=bert_model_name, n_splits=n_splits,
            max_length=bert_max_length, batch_size=bert_batch_size,
            epochs=bert_epochs, learning_rate=bert_lr, random_state=random_state,
        )
    else:
        print("  [SKIP] transformers/torch 미설치")

    # 4. 결과 통합
    print("\n" + "=" * 72)
    print("  [STEP 4] 결과 통합")
    print("=" * 72)

    all_per_label, all_overall = [], []
    all_true_combined, all_pred_combined, all_sample_combined = {}, {}, {}

    for res in [tfidf_results, bert_results]:
        if res:
            all_per_label.append(res["per_label_metrics"])
            all_overall.append(res["overall_metrics"])
            all_true_combined.update(res["all_true"])
            all_pred_combined.update(res["all_pred"])
            all_sample_combined.update(res["per_sample"])

    df_per_label = pd.concat(all_per_label, ignore_index=True) if all_per_label else pd.DataFrame()
    df_overall = pd.concat(all_overall, ignore_index=True) if all_overall else pd.DataFrame()

    # 5. 혼동행렬
    cm_dfs = {}
    for clf_name in all_true_combined:
        cm = C.confusion_matrix(all_true_combined[clf_name], all_pred_combined[clf_name], labels=abuse_order)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in abuse_order], columns=[f"pred_{l}" for l in abuse_order])
        cm_dfs[clf_name] = cm_df
        safe_name = clf_name.replace(" ", "_").replace("/", "_").replace("+", "_")
        cm_df.to_csv(os.path.join(out_dir, f"confusion_matrix_{safe_name}.csv"), encoding="utf-8-sig")

    # 6. 모호성 지대 분석
    df_ambiguity = pd.DataFrame()
    if len(all_sample_combined) >= 2:
        clf_names = list(all_sample_combined.keys())
        common_indices = sorted(set.intersection(*[set(all_sample_combined[c].keys()) for c in clf_names]))
        y_full = df["main_abuse"].astype(str).values
        amb_rows = []
        for idx in common_indices:
            true_label = y_full[idx]
            preds = {c: all_sample_combined[c][idx] for c in clf_names}
            n_correct = sum(1 for p in preds.values() if p == true_label)
            n_wrong = len(clf_names) - n_correct
            wrong_labels = [p for p in preds.values() if p != true_label]
            wrong_consensus = len(set(wrong_labels)) == 1 if wrong_labels else False
            row = {
                "sample_idx": idx, "true_label": true_label,
                "n_correct": n_correct, "n_wrong": n_wrong,
                "all_wrong": int(n_wrong == len(clf_names)),
                "wrong_consensus": int(wrong_consensus),
            }
            for c in clf_names:
                safe_c = c.replace(" ", "_").replace("/", "_").replace("+", "_")
                row[f"pred_{safe_c}"] = preds[c]
            amb_rows.append(row)
        df_ambiguity = pd.DataFrame(amb_rows)

    # 7. 저장
    if not df_per_label.empty:
        df_per_label.to_csv(os.path.join(out_dir, "per_label_metrics_all_models.csv"), encoding="utf-8-sig", index=False)
    if not df_overall.empty:
        df_overall.to_csv(os.path.join(out_dir, "overall_metrics_all_models.csv"), encoding="utf-8-sig", index=False)
    if not df_ambiguity.empty:
        df_ambiguity.to_csv(os.path.join(out_dir, "ambiguity_zone_analysis.csv"), encoding="utf-8-sig", index=False)

    # 8. 결과 출력
    _print_results(df_per_label, df_overall, abuse_order, df_ambiguity)

    return {
        "per_label_all": df_per_label,
        "overall_all": df_overall,
        "confusion_matrices": cm_dfs,
        "ambiguity_analysis": df_ambiguity,
    }


def _print_results(df_per_label, df_overall, abuse_order, df_ambiguity):
    """결과 출력."""
    if not df_overall.empty:
        print("\n" + "=" * 80)
        print("  모델별 전체 성능 비교")
        print("=" * 80)
        for _, row in df_overall.iterrows():
            print(f"\n  {row['classifier']}")
            print(f"     Accuracy: {row['accuracy']:.4f}  Macro F1: {row['macro_f1']:.4f}  "
                  f"Weighted F1: {row['weighted_f1']:.4f}  kappa: {row['cohen_kappa']:.4f}")

    if not df_per_label.empty:
        print("\n" + "=" * 80)
        print("  학대유형별 Precision / Recall / F1-Score")
        print("=" * 80)
        for label in abuse_order:
            sub = df_per_label[df_per_label["label"] == label]
            if sub.empty:
                continue
            en = C.ABUSE_LABEL_EN.get(label, label)
            print(f"\n  [{label} ({en})]")
            for _, r in sub.iterrows():
                print(f"    {r['classifier']:<40s} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1_score']:.4f}")

    if not df_ambiguity.empty:
        n_total = len(df_ambiguity)
        n_all_wrong = int(df_ambiguity["all_wrong"].sum())
        print(f"\n  모호성 지대: 전체={n_total}, 모든 모델 오분류={n_all_wrong} ({n_all_wrong / n_total * 100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
#  __main__
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob
    from pathlib import Path

    print("=" * 72)
    print("  TF-IDF vs KLUE-BERT 학대유형 분류 비교 분석")
    print("=" * 72)

    _this = Path(__file__).resolve()
    _project_root = _this.parent
    for _ in range(5):
        if (_project_root / "data").exists():
            break
        _project_root = _project_root.parent

    _data_dir = _project_root / "data"
    if not _data_dir.exists():
        print(f"  data 폴더 없음: {_data_dir}")
        sys.exit(1)

    _jsons = sorted(glob.glob(str(_data_dir / "*.json")))
    if not _jsons:
        print("  JSON 없음")
        sys.exit(1)

    print(f"  프로젝트: {_project_root}, JSON: {len(_jsons)}개")

    C.configure_output_dirs(subset_name="NEG_ONLY", base_dir=str(_project_root))
    _out = C.MODEL_COMPARISON_DIR

    results = run_full_model_comparison(
        json_files=_jsons,
        abuse_order=list(C.ABUSE_ORDER),
        out_dir=_out,
        only_negative=True,
        n_splits=5,
        bert_model_name="klue/bert-base",
        bert_max_length=256,
        bert_batch_size=16,
        bert_epochs=10,
        bert_lr=2e-5,
        random_state=42,
    )
    print(f"\n  분석 완료! 결과: {_out}")
