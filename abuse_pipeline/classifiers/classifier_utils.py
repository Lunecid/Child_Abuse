"""
classifier_utils.py
===================
TF-IDF + sklearn 분류기 / KLUE-BERT 파인튜닝 공용 유틸리티.

모든 분류기 관련 모듈(tfidf_vs_bert_comparision, neg_gt_multilabel_analysis,
revision_extensions 등)에서 공유하는 코드를 하나로 통합한다.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from abuse_pipeline.core import common as C

# ── scipy (SVM 확률 변환) ────────────────────────────────────────
try:
    from scipy.special import expit as sigmoid
except Exception:
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ═══════════════════════════════════════════════════════════════════
#  1. TF-IDF 분류기 팩토리
# ═══════════════════════════════════════════════════════════════════

def make_singlelabel_clf(name: str, random_state: int = 42):
    """tfidf_vs_bert_comparision.py 와 동일한 다중분류 설정."""
    if not C.HAS_SKLEARN:
        raise ImportError("scikit-learn이 설치되지 않았습니다.")

    if name == "LR":
        return C.LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=300,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "RF":
        return C.RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "SVM":
        return C.LinearSVC(
            max_iter=1000,
            random_state=random_state,
            dual="auto",
        )
    raise ValueError(f"Unknown classifier: {name}")


def make_multilabel_clf(name: str, random_state: int = 42):
    """Binary Relevance 용 이진 분류기 (class_weight='balanced')."""
    if not C.HAS_SKLEARN:
        raise ImportError("scikit-learn이 설치되지 않았습니다.")

    if name == "LR":
        return C.LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )
    if name == "RF":
        return C.RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "SVM":
        return C.LinearSVC(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            dual="auto",
        )
    raise ValueError(f"Unknown classifier: {name}")


# ═══════════════════════════════════════════════════════════════════
#  2. TF-IDF fold 함수
# ═══════════════════════════════════════════════════════════════════

def fit_singlelabel_fold_tfidf(
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    clf_name: str,
    random_state: int = 42,
) -> np.ndarray:
    """단일라벨 다중분류 예측 → 예측 라벨 배열."""
    vec = C.TfidfVectorizer(**C.TFIDF_PARAMS)
    X_train_v = vec.fit_transform(x_train)
    X_test_v = vec.transform(x_test)

    clf = make_singlelabel_clf(clf_name, random_state)
    clf.fit(X_train_v, y_train)
    return clf.predict(X_test_v)


def fit_multilabel_fold_tfidf(
    x_train: list[str],
    y_train_bin: np.ndarray,
    x_test: list[str],
    clf_name: str,
    random_state: int = 42,
) -> np.ndarray:
    """Binary Relevance 방식 다중라벨 예측 → 확률 행렬 (n_test, n_labels)."""
    vec = C.TfidfVectorizer(**C.TFIDF_PARAMS)
    X_train_v = vec.fit_transform(x_train)
    X_test_v = vec.transform(x_test)

    n_test = X_test_v.shape[0]
    n_labels = y_train_bin.shape[1]
    probs = np.zeros((n_test, n_labels), dtype=float)

    for j in range(n_labels):
        y_col = y_train_bin[:, j]
        if np.unique(y_col).size < 2:
            probs[:, j] = float(y_col[0])
            continue

        clf = make_multilabel_clf(clf_name, random_state)
        clf.fit(X_train_v, y_col)

        if hasattr(clf, "predict_proba"):
            probs[:, j] = clf.predict_proba(X_test_v)[:, 1]
        elif hasattr(clf, "decision_function"):
            probs[:, j] = sigmoid(clf.decision_function(X_test_v))
        else:
            probs[:, j] = clf.predict(X_test_v).astype(float)

    return probs


def run_tfidf_classifiers_cv(
    texts: list[str],
    y: np.ndarray,
    label_order: list[str],
    clf_names: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    TF-IDF + 분류기 N종을 동일한 Stratified K-Fold CV 로 평가.

    Returns
    -------
    dict: {clf_name: {"all_true": list, "all_pred": list, "per_sample": dict}}
    """
    if clf_names is None:
        clf_names = ["LR", "RF", "SVM"]

    min_count = int(np.unique(y, return_counts=True)[1].min())
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        return {}

    skf = C.StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    results = {}

    for cname in clf_names:
        all_true, all_pred = [], []
        sample_preds = {}

        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            X_train = [texts[i] for i in train_idx]
            y_train = y[train_idx]
            X_test = [texts[i] for i in test_idx]
            y_test = y[test_idx]

            pipe = C.SklearnPipeline([
                ("tfidf", C.TfidfVectorizer(**C.TFIDF_PARAMS)),
                ("clf", deepcopy(make_singlelabel_clf(cname, random_state))),
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            all_true.extend(list(y_test))
            all_pred.extend(list(pred))
            for i, idx in enumerate(test_idx):
                sample_preds[int(idx)] = pred[i]

        results[cname] = {
            "all_true": all_true,
            "all_pred": all_pred,
            "per_sample": sample_preds,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
#  3. BERT helpers
# ═══════════════════════════════════════════════════════════════════

if C.HAS_TRANSFORMERS:
    import torch
    from torch.utils.data import Dataset as _TorchDataset, DataLoader

    class BertDataset(_TorchDataset):
        """KLUE-BERT 용 PyTorch Dataset (다중라벨 / 단일라벨 겸용)."""

        def __init__(self, texts, labels, tokenizer, max_length=256, multilabel=False):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.multilabel = multilabel

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
            if self.multilabel:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
            else:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item


def _bert_train_loop(model, train_loader, epochs, learning_rate, device, tag=""):
    """BERT 모델 공용 학습 루프."""
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = C.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

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

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"      {tag}Epoch {epoch + 1}/{epochs} -- Loss: {avg_loss:.4f}")


def _bert_cleanup():
    """BERT 학습 후 GPU 메모리 해제."""
    if C.HAS_TRANSFORMERS:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fit_singlelabel_fold_bert(
    x_train_raw: list[str],
    y_train: np.ndarray,
    x_test_raw: list[str],
    label2id: dict[str, int],
    id2label: dict[int, str],
    model_name: str = "klue/bert-base",
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    device=None,
) -> np.ndarray:
    """KLUE-BERT 단일라벨 파인튜닝 → 예측 라벨 문자열 배열."""
    if not C.HAS_TRANSFORMERS:
        return None

    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_labels = len(label2id)
    tokenizer = C.AutoTokenizer.from_pretrained(model_name)
    model = C.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).to(device)

    train_ds = BertDataset(x_train_raw, y_train, tokenizer, max_length, multilabel=False)
    dummy_labels = np.zeros(len(x_test_raw), dtype=int)
    test_ds = BertDataset(x_test_raw, dummy_labels, tokenizer, max_length, multilabel=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    _bert_train_loop(model, train_loader, epochs, learning_rate, device, tag="BERT-single ")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    del model, tokenizer
    _bert_cleanup()

    return np.array([id2label[int(p)] for p in all_preds])


def fit_multilabel_fold_bert(
    x_train_raw: list[str],
    y_train_bin: np.ndarray,
    x_test_raw: list[str],
    n_labels: int,
    model_name: str = "klue/bert-base",
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    device=None,
) -> np.ndarray:
    """KLUE-BERT 다중라벨 파인튜닝 → 확률 행렬 (n_test, n_labels)."""
    if not C.HAS_TRANSFORMERS:
        return None

    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = C.AutoTokenizer.from_pretrained(model_name)
    model = C.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_labels,
        problem_type="multi_label_classification",
    ).to(device)

    train_ds = BertDataset(x_train_raw, y_train_bin, tokenizer, max_length, multilabel=True)
    dummy_labels = np.zeros((len(x_test_raw), n_labels), dtype=float)
    test_ds = BertDataset(x_test_raw, dummy_labels, tokenizer, max_length, multilabel=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    _bert_train_loop(model, train_loader, epochs, learning_rate, device, tag="BERT-multi ")

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    del model, tokenizer
    _bert_cleanup()

    return np.vstack(all_probs)
