from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_child_group
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold
    from sklearn.multioutput import MultiOutputRegressor

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def _iter_records(json_obj: Any) -> list[dict[str, Any]]:
    if isinstance(json_obj, dict) and ("info" in json_obj or "list" in json_obj):
        return [json_obj]
    if isinstance(json_obj, list):
        return [x for x in json_obj if isinstance(x, dict)]
    return []


def _extract_abuse_scores(rec: dict[str, Any], label_order: list[str]) -> dict[str, int]:
    scores = {a: 0 for a in label_order}
    for q in rec.get("list", []) or []:
        if q.get("문항") != "학대여부":
            continue
        for it in q.get("list", []) or []:
            name = it.get("항목")
            try:
                sc = int(it.get("점수"))
            except (TypeError, ValueError):
                sc = 0
            if not isinstance(name, str):
                continue
            for a in label_order:
                if a in name:
                    scores[a] += sc
    return scores


def _scores_to_prob(scores: dict[str, int], label_order: list[str], alpha: float) -> np.ndarray | None:
    v = np.array([max(0, int(scores.get(a, 0))) for a in label_order], dtype=float)
    if float(v.sum()) <= 0.0:
        return None
    v = v + float(alpha)
    v = v / float(v.sum())
    return v


def _normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    m = np.clip(m, 0.0, None)
    s = m.sum(axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    m = m / s
    m = np.clip(m, eps, 1.0)
    m = m / m.sum(axis=1, keepdims=True)
    return m


def _safe_tfidf_fit_transform(x_train: list[str]) -> tuple[TfidfVectorizer, Any]:
    vec = TfidfVectorizer(**C.TFIDF_PARAMS)
    try:
        x_train_vec = vec.fit_transform(x_train)
    except ValueError as e:
        if "no terms remain" not in str(e):
            raise
        relaxed = dict(C.TFIDF_PARAMS)
        relaxed["min_df"] = 1
        vec = TfidfVectorizer(**relaxed)
        x_train_vec = vec.fit_transform(x_train)
    return vec, x_train_vec


def _kl_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pp = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    qq = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    pp = pp / pp.sum(axis=1, keepdims=True)
    qq = qq / qq.sum(axis=1, keepdims=True)
    return np.sum(pp * (np.log(pp) - np.log(qq)), axis=1)


def _cos_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pp = np.asarray(p, dtype=float)
    qq = np.asarray(q, dtype=float)
    pn = np.linalg.norm(pp, axis=1, keepdims=True)
    qn = np.linalg.norm(qq, axis=1, keepdims=True)
    pn = np.where(pn <= 0, eps, pn)
    qn = np.where(qn <= 0, eps, qn)
    return np.sum((pp / pn) * (qq / qn), axis=1)


def _brier_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.mean((np.asarray(p, dtype=float) - np.asarray(q, dtype=float)) ** 2, axis=1)


def _build_dataset(
    json_files: list[str],
    label_order: list[str],
    alpha: float,
    only_negative: bool,
    min_a_sum: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        recs = _iter_records(obj)
        for ridx, rec in enumerate(recs):
            try:
                valence = classify_child_group(rec)
            except Exception:
                valence = None

            if only_negative and valence != "부정":
                continue

            speech = extract_child_speech(rec)
            if not speech:
                continue
            raw_text = " ".join(speech)
            toks = tokenize_korean(raw_text)
            if not toks:
                continue

            scores = _extract_abuse_scores(rec, label_order)
            a_sum = int(sum(scores.values()))
            if a_sum < int(min_a_sum):
                continue

            prob = _scores_to_prob(scores, label_order, alpha=float(alpha))
            if prob is None:
                continue

            info = rec.get("info", {}) or {}
            doc_id = info.get("ID") or info.get("id") or info.get("Id") or f"{Path(path).stem}__{ridx}"
            main = label_order[int(np.argmax(prob))]

            row: dict[str, Any] = {
                "doc_id": str(doc_id),
                "source_file": str(path),
                "valence_group": valence if valence else "",
                "raw_text": raw_text,
                "tfidf_text": " ".join(toks),
                "a_sum": a_sum,
                "a_main_label": main,
            }
            for i, a in enumerate(label_order):
                row[f"a_score_{a}"] = int(scores[a])
                row[f"y_true_{a}"] = float(prob[i])
            rows.append(row)

    return pd.DataFrame(rows)


def run_softlabel_vs_singlelabel_study(
    json_files: list[str],
    out_dir: str,
    only_negative: bool = True,
    dedupe_by_doc_id: bool = True,
    alpha: float = 0.5,
    min_a_sum: int = 1,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    label_order = list(C.ABUSE_ORDER)

    if not HAS_SKLEARN:
        msg = "[SOFT-vs-SINGLE] scikit-learn 미설치로 분석을 건너뜁니다."
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False
        )
        return {"status": "skipped", "reason": "sklearn_missing"}

    df = _build_dataset(
        json_files=[str(x) for x in json_files],
        label_order=label_order,
        alpha=float(alpha),
        only_negative=bool(only_negative),
        min_a_sum=int(min_a_sum),
    )

    if df.empty:
        msg = "[SOFT-vs-SINGLE] 유효 샘플이 없어 분석을 건너뜁니다."
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False
        )
        return {"status": "skipped", "reason": "empty_dataset"}

    if dedupe_by_doc_id:
        df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)

    df.to_csv(os.path.join(out_dir, "dataset_softlabel.csv"), encoding="utf-8-sig", index=False)

    class_counts = df["a_main_label"].value_counts()
    min_count = int(class_counts.min())
    actual_splits = min(int(n_splits), min_count)
    if actual_splits < 2 or df["a_main_label"].nunique() < 2:
        msg = (
            f"[SOFT-vs-SINGLE] 클래스 수 부족으로 CV 불가 "
            f"(n={len(df)}, classes={df['a_main_label'].nunique()}, min_class={min_count})"
        )
        pd.DataFrame([{"message": msg}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False
        )
        return {"status": "skipped", "reason": "insufficient_class_count"}

    y_cols = [f"y_true_{a}" for a in label_order]
    x_text = df["tfidf_text"].astype(str).tolist()
    y_soft = df[y_cols].values.astype(float)
    y_main = df["a_main_label"].astype(str).values

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    oof_soft = np.zeros_like(y_soft, dtype=float)
    oof_soft_main = np.array([""] * len(df), dtype=object)
    oof_single_main = np.array([""] * len(df), dtype=object)

    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_main)), y_main), start=1):
        x_tr = [x_text[i] for i in train_idx]
        x_te = [x_text[i] for i in test_idx]
        y_tr_soft = y_soft[train_idx]
        y_te_soft = y_soft[test_idx]
        y_tr_main = y_main[train_idx]
        y_te_main = y_main[test_idx]

        vec, x_tr_vec = _safe_tfidf_fit_transform(x_tr)
        x_te_vec = vec.transform(x_te)

        soft_model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=random_state))
        try:
            soft_model.fit(x_tr_vec, y_tr_soft)
            pred_soft = _normalize_rows(soft_model.predict(x_te_vec))
        except Exception:
            prior = np.mean(y_tr_soft, axis=0, keepdims=True)
            pred_soft = np.repeat(_normalize_rows(prior), len(test_idx), axis=0)

        pred_soft_main = np.array([label_order[int(np.argmax(r))] for r in pred_soft], dtype=object)

        single_model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        try:
            single_model.fit(x_tr_vec, y_tr_main)
            pred_single = single_model.predict(x_te_vec)
        except Exception:
            pred_single = pred_soft_main.copy()

        oof_soft[test_idx] = pred_soft
        oof_soft_main[test_idx] = pred_soft_main
        oof_single_main[test_idx] = pred_single

        kl = _kl_rows(y_te_soft, pred_soft)
        cs = _cos_rows(y_te_soft, pred_soft)
        br = _brier_rows(y_te_soft, pred_soft)
        top1_soft = float(np.mean(pred_soft_main == y_te_main))
        top1_single = float(np.mean(pred_single == y_te_main))
        sp, sr, sf_macro, _ = precision_recall_fscore_support(
            y_te_main, pred_single, labels=label_order, average="macro", zero_division=0
        )
        _, _, sf_weighted, _ = precision_recall_fscore_support(
            y_te_main, pred_single, labels=label_order, average="weighted", zero_division=0
        )
        kappa = float(cohen_kappa_score(y_te_main, pred_single, labels=label_order))

        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "soft_kl_mean": float(np.mean(kl)),
                "soft_cosine_mean": float(np.mean(cs)),
                "soft_brier_mean": float(np.mean(br)),
                "soft_top1_acc": top1_soft,
                "single_acc": top1_single,
                "single_macro_precision": float(sp),
                "single_macro_recall": float(sr),
                "single_macro_f1": float(sf_macro),
                "single_weighted_f1": float(sf_weighted),
                "single_kappa": kappa,
                "soft_single_agreement": float(np.mean(pred_soft_main == pred_single)),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(out_dir, "fold_metrics.csv"), encoding="utf-8-sig", index=False)

    kl_all = _kl_rows(y_soft, oof_soft)
    cs_all = _cos_rows(y_soft, oof_soft)
    br_all = _brier_rows(y_soft, oof_soft)
    soft_top1_all = float(np.mean(oof_soft_main == y_main))
    single_acc_all = float(np.mean(oof_single_main == y_main))
    s_p, s_r, s_fm, _ = precision_recall_fscore_support(
        y_main, oof_single_main, labels=label_order, average="macro", zero_division=0
    )
    _, _, s_fw, _ = precision_recall_fscore_support(
        y_main, oof_single_main, labels=label_order, average="weighted", zero_division=0
    )
    single_kappa_all = float(cohen_kappa_score(y_main, oof_single_main, labels=label_order))

    metrics_rows = [
        {"model": "softlabel_vector", "metric": "kl_mean", "value": float(np.mean(kl_all))},
        {"model": "softlabel_vector", "metric": "kl_median", "value": float(np.median(kl_all))},
        {"model": "softlabel_vector", "metric": "cosine_mean", "value": float(np.mean(cs_all))},
        {"model": "softlabel_vector", "metric": "brier_mean", "value": float(np.mean(br_all))},
        {"model": "softlabel_vector", "metric": "top1_accuracy", "value": soft_top1_all},
        {"model": "singlelabel_main", "metric": "accuracy", "value": single_acc_all},
        {"model": "singlelabel_main", "metric": "macro_precision", "value": float(s_p)},
        {"model": "singlelabel_main", "metric": "macro_recall", "value": float(s_r)},
        {"model": "singlelabel_main", "metric": "macro_f1", "value": float(s_fm)},
        {"model": "singlelabel_main", "metric": "weighted_f1", "value": float(s_fw)},
        {"model": "singlelabel_main", "metric": "cohen_kappa", "value": single_kappa_all},
        {"model": "soft_vs_single", "metric": "top1_prediction_agreement", "value": float(np.mean(oof_soft_main == oof_single_main))},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), encoding="utf-8-sig", index=False)

    cm_soft = pd.crosstab(
        pd.Series(y_main, name="true_main"),
        pd.Series(oof_soft_main, name="pred_main_from_soft"),
        dropna=False,
    ).reindex(index=label_order, columns=label_order, fill_value=0)
    cm_single = pd.crosstab(
        pd.Series(y_main, name="true_main"),
        pd.Series(oof_single_main, name="pred_main_single"),
        dropna=False,
    ).reindex(index=label_order, columns=label_order, fill_value=0)
    cm_soft.to_csv(os.path.join(out_dir, "confusion_matrix_soft_top1.csv"), encoding="utf-8-sig")
    cm_single.to_csv(os.path.join(out_dir, "confusion_matrix_singlelabel.csv"), encoding="utf-8-sig")

    out_rows = []
    for i, row in df.iterrows():
        r = {
            "doc_id": row["doc_id"],
            "source_file": row["source_file"],
            "valence_group": row["valence_group"],
            "a_main_label_true": row["a_main_label"],
            "pred_main_from_soft": oof_soft_main[i],
            "pred_main_single": oof_single_main[i],
            "soft_top1_correct": int(oof_soft_main[i] == row["a_main_label"]),
            "single_correct": int(oof_single_main[i] == row["a_main_label"]),
            "soft_single_agree": int(oof_soft_main[i] == oof_single_main[i]),
            "a_sum": int(row["a_sum"]),
        }
        for j, a in enumerate(label_order):
            r[f"y_true_{a}"] = float(y_soft[i, j])
            r[f"y_pred_soft_{a}"] = float(oof_soft[i, j])
            r[f"a_score_{a}"] = int(row[f"a_score_{a}"])
        out_rows.append(r)

    per_sample_df = pd.DataFrame(out_rows)
    per_sample_df.to_csv(os.path.join(out_dir, "per_sample_predictions.csv"), encoding="utf-8-sig", index=False)

    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_order": label_order,
                "only_negative": bool(only_negative),
                "dedupe_by_doc_id": bool(dedupe_by_doc_id),
                "alpha": float(alpha),
                "min_a_sum": int(min_a_sum),
                "n_splits": int(actual_splits),
                "random_state": int(random_state),
                "soft_model": "TF-IDF + MultiOutputRegressor(Ridge)",
                "single_model": "TF-IDF + LogisticRegression(multiclass)",
                "tfidf_params": {
                    "ngram_range": list(C.TFIDF_PARAMS["ngram_range"]),
                    "min_df": int(C.TFIDF_PARAMS["min_df"]),
                    "max_features": int(C.TFIDF_PARAMS["max_features"]),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(out_dir, "paper_ready_summary.txt"), "w", encoding="utf-8") as f:
        lines = [
            "A-score soft label vector prediction vs single-label prediction",
            f"- Samples: {len(df)}",
            f"- Unique docs: {df['doc_id'].nunique()}",
            f"- CV folds: {actual_splits}",
            f"- Soft model KL mean: {float(np.mean(kl_all)):.6f}",
            f"- Soft model Cosine mean: {float(np.mean(cs_all)):.6f}",
            f"- Soft model Brier mean: {float(np.mean(br_all)):.6f}",
            f"- Soft model Top-1 acc: {soft_top1_all:.6f}",
            f"- Single model Accuracy: {single_acc_all:.6f}",
            f"- Single model Macro F1: {float(s_fm):.6f}",
            f"- Single model Kappa: {single_kappa_all:.6f}",
            f"- Soft vs Single Top-1 agreement: {float(np.mean(oof_soft_main == oof_single_main)):.6f}",
        ]
        f.write("\n".join(lines))

    print(f"[SOFT-vs-SINGLE] 분석 완료. 산출물: {out_dir}")
    return {
        "status": "ok",
        "n_samples": int(len(df)),
        "n_unique_doc_id": int(df["doc_id"].nunique()),
        "n_splits": int(actual_splits),
        "soft_top1_acc": soft_top1_all,
        "single_acc": single_acc_all,
        "single_macro_f1": float(s_fm),
        "out_dir": out_dir,
    }
