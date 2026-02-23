"""
tfidf_vs_bert_comparison.py
============================
TF-IDF 기반 분류기 vs KLUE-BERT 트랜스포머 모델 비교 분석

목적
----
기존 파이프라인의 TF-IDF + {LogisticRegression, RandomForest, LinearSVM} 분류기에
KLUE-BERT (klue/bert-base) 파인튜닝 분류기를 추가하여,
4가지 학대유형(성학대, 신체학대, 정서학대, 방임) 각각에 대한
Precision, Recall, F1-score를 비교한다.

구조
----
  (1) 데이터 준비: JSON → 아동별 발화 텍스트 + 학대유형 라벨
  (2) TF-IDF 기반 분류기 3종: Stratified K-Fold CV
  (3) KLUE-BERT 파인튜닝 분류기: Stratified K-Fold CV
  (4) 라벨별 Precision, Recall, F1 비교표 생성
  (5) 모호성 지대(Ambiguity Zone) 분석: 모든 모델이 공통으로 틀리는 패턴

실행 환경 요구사항
-----------------
  pip install torch transformers scikit-learn pandas numpy matplotlib

사용법
------
  이 파일을 프로젝트 루트 또는 abuse_pipeline/ 디렉토리에 놓고:
    python tfidf_vs_bert_comparison.py

  또는 파이프라인 내부에서:
    from tfidf_vs_bert_comparison import run_full_model_comparison
    results = run_full_model_comparison(json_files, abuse_order, out_dir)
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from copy import deepcopy
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── sklearn ──
try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
        precision_recall_fscore_support,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn 미설치. TF-IDF 분류기를 사용할 수 없습니다.")

# ── transformers + torch ──
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARN] transformers/torch 미설치. KLUE-BERT 분류기를 사용할 수 없습니다.")

# ── 프로젝트 모듈 import (3단계 폴백) ──
_IMPORT_OK = False
try:
    from . import common as C
    from .labels import classify_child_group, classify_abuse_main_sub
    from .text import extract_child_speech, tokenize_korean
    _IMPORT_OK = True
except ImportError:
    pass

if not _IMPORT_OK:
    try:
        from abuse_pipeline import common as C
        from abuse_pipeline.labels import (
            classify_child_group, classify_abuse_main_sub,
        )
        from abuse_pipeline.text import (
            extract_child_speech, tokenize_korean,
        )
        _IMPORT_OK = True
    except ImportError:
        pass

if not _IMPORT_OK:
    from pathlib import Path as _Path
    _this_dir = _Path(__file__).resolve().parent
    _proj_root = _this_dir.parent
    _p = str(_proj_root)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    try:
        from abuse_pipeline import common as C
        from abuse_pipeline.labels import (
            classify_child_group, classify_abuse_main_sub,
        )
        from abuse_pipeline.text import (
            extract_child_speech, tokenize_korean,
        )
        _IMPORT_OK = True
    except ImportError:
        print("[ERROR] 프로젝트 모듈을 찾을 수 없습니다.")


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

    Parameters
    ----------
    json_files : List[str]
        data/*.json 파일 경로 리스트
    abuse_order : List[str]
        학대유형 순서 (예: ["성학대","신체학대","정서학대","방임"])
    only_negative : bool
        True이면 정서군이 '부정'인 아동만 포함

    Returns
    -------
    pd.DataFrame
        columns = ["ID", "main_abuse", "tfidf_text", "raw_text"]
        - tfidf_text: 형태소 분석 후 공백으로 결합한 텍스트 (TF-IDF용)
        - raw_text: 원본 발화를 공백으로 결합한 텍스트 (BERT용)
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
        print(f"  {a}: {n}명 ({n/len(df)*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════
#  2. TF-IDF 기반 분류기 3종 (기존 코드 확장)
# ═══════════════════════════════════════════════════════════════════

def run_tfidf_classifiers(
    df: pd.DataFrame,
    label_col: str,
    label_order: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    TF-IDF + {LogisticRegression, RandomForest, LinearSVM} 3종 분류기를
    동일한 Stratified K-Fold CV로 평가한다.

    ★ 기존 revision_extensions.py의 run_multi_classifier_comparison()과 동일한 구조이나,
       라벨별 Precision/Recall/F1을 상세히 기록하는 부분을 강화하였다.

    Parameters
    ----------
    df : pd.DataFrame
        columns = ["ID", label_col, "tfidf_text"]
    label_col : str
        라벨 칼럼명 (예: "main_abuse")
    label_order : List[str]
        라벨 순서 (예: ["성학대","신체학대","정서학대","방임"])
    n_splits : int
        Stratified K-Fold 분할 수
    random_state : int
        재현성을 위한 랜덤 시드

    Returns
    -------
    dict:
        "per_label_metrics": pd.DataFrame  (classifier, label, precision, recall, f1, support)
        "overall_metrics": pd.DataFrame    (classifier, accuracy, macro_f1, weighted_f1, kappa)
        "all_true": dict[str, list]        분류기별 전체 정답
        "all_pred": dict[str, list]        분류기별 전체 예측
        "per_sample": dict[str, dict]      분류기별 샘플별 예측
    """
    if not HAS_SKLEARN:
        print("[SKIP] scikit-learn 미설치")
        return None

    texts = df["tfidf_text"].astype(str).tolist()
    y = df[label_col].astype(str).values

    # n_splits 안전 조정
    min_count = int(df[label_col].value_counts().min())
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        print(f"[SKIP] 최소 클래스 샘플 수 = {min_count}, CV 불가")
        return None

    skf = StratifiedKFold(
        n_splits=actual_splits, shuffle=True, random_state=random_state
    )

    # ── 분류기 정의 ──
    classifiers = {
        "TF-IDF + LogisticRegression": LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=300, n_jobs=-1, random_state=random_state,
        ),
        "TF-IDF + RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=random_state,
        ),
        "TF-IDF + LinearSVM": LinearSVC(
            max_iter=1000, random_state=random_state,
            dual="auto",
        ),
    }

    tfidf_params = dict(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    all_true_dict = {}
    all_pred_dict = {}
    per_sample_dict = {}

    for clf_name, clf in classifiers.items():
        print(f"\n  [TF-IDF] {clf_name} 학습 중... (n_splits={actual_splits})")
        all_true, all_pred = [], []
        sample_preds = {}

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(y)), y), start=1
        ):
            X_train = [texts[i] for i in train_idx]
            y_train = y[train_idx]
            X_test = [texts[i] for i in test_idx]
            y_test = y[test_idx]

            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf", deepcopy(clf)),
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            all_true.extend(list(y_test))
            all_pred.extend(list(pred))

            for i, idx in enumerate(test_idx):
                sample_preds[int(idx)] = pred[i]

        all_true_dict[clf_name] = all_true
        all_pred_dict[clf_name] = all_pred
        per_sample_dict[clf_name] = sample_preds

        # 즉시 결과 출력
        acc = np.mean([t == p for t, p in zip(all_true, all_pred)])
        kappa = cohen_kappa_score(all_true, all_pred, labels=label_order)
        print(f"    → Accuracy: {acc:.4f}, Cohen's κ: {kappa:.4f}")

    # ── 라벨별 메트릭 통합 ──
    per_label_rows = []
    overall_rows = []

    for clf_name in classifiers:
        true = all_true_dict[clf_name]
        pred = all_pred_dict[clf_name]

        # 라벨별 precision, recall, f1, support
        prec, rec, f1, sup = precision_recall_fscore_support(
            true, pred, labels=label_order, zero_division=0,
        )
        for i, label in enumerate(label_order):
            per_label_rows.append({
                "classifier": clf_name,
                "label": label,
                "precision": prec[i],
                "recall": rec[i],
                "f1_score": f1[i],
                "support": int(sup[i]),
            })

        # macro / weighted averages
        report = classification_report(
            true, pred, labels=label_order, output_dict=True, zero_division=0,
        )
        overall_rows.append({
            "classifier": clf_name,
            "accuracy": report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "cohen_kappa": cohen_kappa_score(true, pred, labels=label_order),
        })

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

class AbuseTextDataset(Dataset):
    """KLUE-BERT용 PyTorch Dataset."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_bert_one_fold(
    X_train_texts, y_train, X_test_texts, y_test,
    label2id, id2label,
    model_name="klue/bert-base",
    max_length=256,
    batch_size=16,
    epochs=4,
    learning_rate=2e-5,
    device=None,
):
    """
    KLUE-BERT를 한 Fold에 대해 파인튜닝하고 예측값을 반환한다.

    Parameters
    ----------
    X_train_texts : List[str]
        학습용 원본 텍스트 (형태소 분석 전)
    y_train : np.ndarray
        학습용 라벨 (정수 인코딩)
    X_test_texts : List[str]
        검증용 원본 텍스트
    y_test : np.ndarray
        검증용 라벨
    label2id : dict
        라벨→정수 매핑
    id2label : dict
        정수→라벨 매핑
    model_name : str
        HuggingFace 모델 이름
    max_length : int
        최대 시퀀스 길이
    batch_size : int
        배치 크기
    epochs : int
        학습 에포크 수
    learning_rate : float
        학습률
    device : torch.device or None

    Returns
    -------
    predictions : np.ndarray
        테스트 세트에 대한 예측 라벨 (정수)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).to(device)

    train_dataset = AbuseTextDataset(X_train_texts, y_train, tokenizer, max_length)
    test_dataset = AbuseTextDataset(X_test_texts, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )

    # ── 학습 ──
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"      Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    # ── 예측 ──
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    # 메모리 해제
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(all_preds)


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
) -> Dict:
    """
    KLUE-BERT를 Stratified K-Fold CV로 평가한다.

    Parameters
    ----------
    df : pd.DataFrame
        columns = [..., label_col, "raw_text"]
    label_col : str
        라벨 칼럼명
    label_order : List[str]
        라벨 순서
    model_name : str
        HuggingFace 모델명 (예: "klue/bert-base", "klue/roberta-base")
    n_splits, max_length, batch_size, epochs, learning_rate, random_state
        하이퍼파라미터

    Returns
    -------
    dict:
        "per_label_metrics": pd.DataFrame
        "overall_metrics": pd.DataFrame
        "all_true": list
        "all_pred": list
        "per_sample": dict
    """
    if not HAS_TRANSFORMERS:
        print("[SKIP] transformers/torch 미설치")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [BERT] 모델: {model_name}")
    print(f"  [BERT] 디바이스: {device}")
    print(f"  [BERT] max_length={max_length}, batch={batch_size}, epochs={epochs}, lr={learning_rate}")

    texts = df["raw_text"].astype(str).tolist()
    y_str = df[label_col].astype(str).values

    # 라벨 인코딩
    label2id = {label: i for i, label in enumerate(label_order)}
    id2label = {i: label for label, i in label2id.items()}
    y = np.array([label2id[l] for l in y_str])

    # n_splits 안전 조정
    min_count = int(df[label_col].value_counts().min())
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        print(f"[SKIP] 최소 클래스 샘플 수 = {min_count}, CV 불가")
        return None

    skf = StratifiedKFold(
        n_splits=actual_splits, shuffle=True, random_state=random_state,
    )

    clf_name = f"KLUE-BERT ({model_name})"
    all_true, all_pred = [], []
    sample_preds = {}

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(y)), y), start=1
    ):
        print(f"\n    [Fold {fold_idx}/{actual_splits}] "
              f"Train: {len(train_idx)}, Test: {len(test_idx)}")

        X_train = [texts[i] for i in train_idx]
        y_train = y[train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = y[test_idx]

        preds = train_bert_one_fold(
            X_train_texts=X_train,
            y_train=y_train,
            X_test_texts=X_test,
            y_test=y_test,
            label2id=label2id,
            id2label=id2label,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )

        # 정수 → 문자열 라벨 변환
        y_test_str = [id2label[int(i)] for i in y_test]
        preds_str = [id2label[int(i)] for i in preds]

        all_true.extend(y_test_str)
        all_pred.extend(preds_str)

        for i, idx in enumerate(test_idx):
            sample_preds[int(idx)] = preds_str[i]

        # Fold별 결과 출력
        fold_acc = np.mean([t == p for t, p in zip(y_test_str, preds_str)])
        print(f"    → Fold {fold_idx} Accuracy: {fold_acc:.4f}")

    # 전체 결과
    acc = np.mean([t == p for t, p in zip(all_true, all_pred)])
    kappa = cohen_kappa_score(all_true, all_pred, labels=label_order)
    print(f"\n  [BERT] 전체 Accuracy: {acc:.4f}, Cohen's κ: {kappa:.4f}")

    # ── 라벨별 메트릭 ──
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_true, all_pred, labels=label_order, zero_division=0,
    )
    per_label_rows = []
    for i, label in enumerate(label_order):
        per_label_rows.append({
            "classifier": clf_name,
            "label": label,
            "precision": prec[i],
            "recall": rec[i],
            "f1_score": f1[i],
            "support": int(sup[i]),
        })

    report = classification_report(
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
    out_dir: str = "model_comparison_output",
    only_negative: bool = True,
    n_splits: int = 5,
    bert_model_name: str = "klue/bert-base",
    bert_max_length: int = 256,
    bert_batch_size: int = 16,
    bert_epochs: int = 10,
    bert_lr: float = 2e-5,
    random_state: int = 42,
) -> Dict:
    """
    TF-IDF 기반 분류기 3종 + KLUE-BERT 분류기의 성능을 통합 비교한다.

    Returns
    -------
    dict:
        "per_label_all": pd.DataFrame     — 모든 모델의 라벨별 P/R/F1
        "overall_all": pd.DataFrame        — 모든 모델의 전체 성능
        "confusion_matrices": dict          — 모델별 혼동행렬
        "ambiguity_analysis": pd.DataFrame  — 모호성 지대 분석
    """
    if abuse_order is None:
        abuse_order = ["성학대", "신체학대", "정서학대", "방임"]

    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 데이터 준비 ──
    print("=" * 72)
    print("  [STEP 1] 데이터 준비")
    print("=" * 72)
    df = prepare_classification_data(json_files, abuse_order, only_negative)

    if df.empty or df["main_abuse"].nunique() < 2:
        print("[ERROR] 유효한 데이터가 부족합니다.")
        return None

    df.to_csv(os.path.join(out_dir, "classification_data.csv"),
              encoding="utf-8-sig", index=False)

    # ── 2. TF-IDF 분류기 ──
    print("\n" + "=" * 72)
    print("  [STEP 2] TF-IDF 기반 분류기 3종")
    print("=" * 72)
    tfidf_results = run_tfidf_classifiers(
        df=df, label_col="main_abuse", label_order=abuse_order,
        n_splits=n_splits, random_state=random_state,
    )

    # ── 3. KLUE-BERT ──
    bert_results = None
    print("\n" + "=" * 72)
    print("  [STEP 3] KLUE-BERT 파인튜닝 분류기")
    print("=" * 72)
    if HAS_TRANSFORMERS:
        bert_results = run_bert_classifier(
            df=df, label_col="main_abuse", label_order=abuse_order,
            model_name=bert_model_name,
            n_splits=n_splits,
            max_length=bert_max_length,
            batch_size=bert_batch_size,
            epochs=bert_epochs,
            learning_rate=bert_lr,
            random_state=random_state,
        )
    else:
        print("  [SKIP] transformers/torch 미설치 → KLUE-BERT 생략")

    # ── 4. 결과 통합 ──
    print("\n" + "=" * 72)
    print("  [STEP 4] 결과 통합 및 비교표 생성")
    print("=" * 72)

    all_per_label = []
    all_overall = []
    all_true_combined = {}
    all_pred_combined = {}
    all_sample_combined = {}

    if tfidf_results:
        all_per_label.append(tfidf_results["per_label_metrics"])
        all_overall.append(tfidf_results["overall_metrics"])
        all_true_combined.update(tfidf_results["all_true"])
        all_pred_combined.update(tfidf_results["all_pred"])
        all_sample_combined.update(tfidf_results["per_sample"])

    if bert_results:
        all_per_label.append(bert_results["per_label_metrics"])
        all_overall.append(bert_results["overall_metrics"])
        all_true_combined.update(bert_results["all_true"])
        all_pred_combined.update(bert_results["all_pred"])
        all_sample_combined.update(bert_results["per_sample"])

    # 통합 DataFrame
    df_per_label = pd.concat(all_per_label, ignore_index=True) if all_per_label else pd.DataFrame()
    df_overall = pd.concat(all_overall, ignore_index=True) if all_overall else pd.DataFrame()

    # ── 5. 혼동행렬 생성 ──
    cm_dfs = {}
    for clf_name in all_true_combined:
        cm = confusion_matrix(
            all_true_combined[clf_name],
            all_pred_combined[clf_name],
            labels=abuse_order,
        )
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in abuse_order],
            columns=[f"pred_{l}" for l in abuse_order],
        )
        cm_dfs[clf_name] = cm_df

        # 저장
        safe_name = clf_name.replace(" ", "_").replace("/", "_").replace("+", "_")
        cm_path = os.path.join(out_dir, f"confusion_matrix_{safe_name}.csv")
        cm_df.to_csv(cm_path, encoding="utf-8-sig")

    # ── 6. 모호성 지대 분석 ──
    df_ambiguity = pd.DataFrame()
    if len(all_sample_combined) >= 2:
        clf_names = list(all_sample_combined.keys())
        common_indices = sorted(
            set.intersection(*[set(all_sample_combined[c].keys()) for c in clf_names])
        )

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
                "sample_idx": idx,
                "true_label": true_label,
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "all_wrong": int(n_wrong == len(clf_names)),
                "wrong_consensus": int(wrong_consensus),
            }
            for c in clf_names:
                safe_c = c.replace(" ", "_").replace("/", "_").replace("+", "_")
                row[f"pred_{safe_c}"] = preds[c]
            amb_rows.append(row)

        df_ambiguity = pd.DataFrame(amb_rows)

    # ── 7. 저장 ──
    if not df_per_label.empty:
        path = os.path.join(out_dir, "per_label_metrics_all_models.csv")
        df_per_label.to_csv(path, encoding="utf-8-sig", index=False)
        print(f"\n  [저장] 라벨별 메트릭 → {path}")

    if not df_overall.empty:
        path = os.path.join(out_dir, "overall_metrics_all_models.csv")
        df_overall.to_csv(path, encoding="utf-8-sig", index=False)
        print(f"  [저장] 전체 성능 비교 → {path}")

    if not df_ambiguity.empty:
        path = os.path.join(out_dir, "ambiguity_zone_analysis.csv")
        df_ambiguity.to_csv(path, encoding="utf-8-sig", index=False)
        print(f"  [저장] 모호성 지대 분석 → {path}")

    # ── 8. 결과 출력 ──
    _print_results(df_per_label, df_overall, abuse_order, df_ambiguity)

    return {
        "per_label_all": df_per_label,
        "overall_all": df_overall,
        "confusion_matrices": cm_dfs,
        "ambiguity_analysis": df_ambiguity,
    }


def _print_results(df_per_label, df_overall, abuse_order, df_ambiguity):
    """결과를 보기 좋게 출력한다."""

    print("\n" + "═" * 80)
    print("  📊 모델별 전체 성능 비교")
    print("═" * 80)
    if not df_overall.empty:
        for _, row in df_overall.iterrows():
            print(f"\n  📌 {row['classifier']}")
            print(f"     Accuracy:         {row['accuracy']:.4f}")
            print(f"     Macro F1:         {row['macro_f1']:.4f}")
            print(f"     Weighted F1:      {row['weighted_f1']:.4f}")
            print(f"     Cohen's κ:        {row['cohen_kappa']:.4f}")

    print("\n" + "═" * 80)
    print("  📊 학대유형별 Precision / Recall / F1-Score")
    print("═" * 80)
    if not df_per_label.empty:
        for label in abuse_order:
            sub = df_per_label[df_per_label["label"] == label]
            if sub.empty:
                continue
            en_label = {
                "성학대": "Sexual abuse",
                "신체학대": "Physical abuse",
                "정서학대": "Emotional abuse",
                "방임": "Neglect",
            }.get(label, label)

            print(f"\n  ┌─── {label} ({en_label}) ───")
            print(f"  │ {'모델':<40s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
            print(f"  │ {'─'*80}")
            for _, r in sub.iterrows():
                print(f"  │ {r['classifier']:<40s} "
                      f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
                      f"{r['f1_score']:>10.4f} {r['support']:>10d}")
            print(f"  └{'─'*83}")

    # 모호성 지대
    if not df_ambiguity.empty:
        n_total = len(df_ambiguity)
        n_all_wrong = int(df_ambiguity["all_wrong"].sum())
        n_consensus = int(df_ambiguity["wrong_consensus"].sum())

        print(f"\n" + "═" * 80)
        print(f"  📊 모호성 지대 분석 (Ambiguity Zone)")
        print(f"═" * 80)
        print(f"  전체 샘플: {n_total}")
        print(f"  모든 모델이 틀린 샘플: {n_all_wrong} ({n_all_wrong/n_total*100:.1f}%)")
        print(f"  모든 모델이 같은 오답: {n_consensus} ({n_consensus/n_total*100:.1f}%)")

        if n_consensus > 0:
            consensus = df_ambiguity[df_ambiguity["wrong_consensus"] == 1]
            pattern = consensus.groupby("true_label").size().sort_values(ascending=False)
            print(f"\n  모델 불변 오분류의 정답 라벨 분포:")
            for label, cnt in pattern.items():
                print(f"    {label}: {cnt}건")


# ═══════════════════════════════════════════════════════════════════
#  __main__
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob
    from pathlib import Path

    print("=" * 72)
    print("  TF-IDF vs KLUE-BERT 학대유형 분류 비교 분석")
    print("=" * 72)

    # 경로 설정
    _this = Path(__file__).resolve()
    _project_root = _this.parent
    # 프로젝트 루트까지 올라감
    for _ in range(5):
        if (_project_root / "data").exists():
            break
        _project_root = _project_root.parent

    _data_dir = _project_root / "data"
    print(f"\n  프로젝트: {_project_root}")
    print(f"  데이터  : {_data_dir}")

    if not _data_dir.exists():
        print(f"\n  ❌ data 폴더 없음: {_data_dir}")
        sys.exit(1)

    _jsons = sorted(glob.glob(str(_data_dir / "*.json")))
    print(f"  JSON    : {len(_jsons)}개")

    if not _jsons:
        print("  ❌ JSON 없음")
        sys.exit(1)

    # 출력 디렉토리
    _out = str(_project_root / "output" / "model_comparison")
    os.makedirs(_out, exist_ok=True)

    # 분석 실행
    abuse_order = ["성학대", "신체학대", "정서학대", "방임"]

    results = run_full_model_comparison(
        json_files=_jsons,
        abuse_order=abuse_order,
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

    print("\n" + "=" * 72)
    print("  ✅ 분석 완료!")
    print(f"  결과: {_out}")
    print("=" * 72)
