"""
recovery_ablation.py
====================
Ablation study: weighted vs uniform vs random bridge words.

GT anchor 사용. 기존 bridge ablation 구조를 information recovery 프레이밍으로 재구현.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from abuse_pipeline.core import common as C
from abuse_pipeline.experiments.information_recovery import (
    RecoveryConfig,
    TYPE_TO_IDX,
    IDX_TO_TYPE,
    _safe_fit_vectorizer,
    _fit_br_stage,
    _build_fold_doc_counts,
    _extract_bridge_words,
    _compute_bridge_score_matrix,
    _apply_bridge_reranking,
    compute_recovery_metrics,
)


def _build_uniform_bridge_dict(bridge_df: pd.DataFrame) -> pd.DataFrame:
    """브릿지 워드의 p2를 모두 1.0으로 교체."""
    if bridge_df.empty:
        return bridge_df
    uniform = bridge_df.copy()
    uniform["p2"] = 1.0
    return uniform


def _build_random_bridge_dict(
    bridge_df: pd.DataFrame,
    all_words: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """비브릿지 단어를 같은 개수만큼 무작위 대체."""
    if bridge_df.empty:
        return bridge_df

    bridge_words = set(bridge_df["word"].tolist())
    non_bridge = [w for w in all_words if w not in bridge_words]

    n = min(len(bridge_df), len(non_bridge))
    if n == 0:
        return bridge_df

    sampled = rng.choice(non_bridge, size=n, replace=False)
    random_df = bridge_df.head(n).copy()
    random_df["word"] = sampled
    random_df["p2"] = 1.0
    return random_df


def run_recovery_ablation(
    dataset_df: pd.DataFrame,
    config: RecoveryConfig,
    lambda_val: float = 1.0,
    tau_val: float = 0.0,
    n_random_repeats: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    3가지 브릿지 변형 비교:
      - weighted: 실제 브릿지 워드 + p2 가중치
      - uniform:  실제 브릿지 워드 + 균일 가중치 (1.0)
      - random:   비브릿지 단어를 같은 개수만큼 무작위 대체

    anchor = GT.
    """
    output_dir = Path(config.output_dir) / "ablation"
    os.makedirs(output_dir, exist_ok=True)

    label_order = list(C.ABUSE_ORDER)

    mlb = C.MultiLabelBinarizer(classes=label_order)
    mlb.fit([[]])
    y_all = mlb.transform(dataset_df["label_list"].tolist())
    gt_main_all = np.array([TYPE_TO_IDX[m] for m in dataset_df["gt_main"]])
    texts = dataset_df["text"].tolist()
    y_main_str = dataset_df["gt_main"].values

    actual_splits = min(config.n_splits, int(dataset_df["gt_main"].value_counts().min()))
    if actual_splits < 2:
        print("[ABLATION] 클래스 수 부족")
        return {"status": "skipped"}

    skf = C.StratifiedKFold(
        n_splits=actual_splits, shuffle=True, random_state=config.random_state
    )

    rng = np.random.default_rng(config.random_state)
    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(texts)), y_main_str), start=1
    ):
        print(f"\n[ABLATION] Fold {fold_idx}/{actual_splits}")

        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        gt_main_test = gt_main_all[test_idx]
        train_df = dataset_df.iloc[train_idx].reset_index(drop=True)

        vec, X_train = _safe_fit_vectorizer(train_texts)
        X_test = vec.transform(test_texts)
        feature_names = vec.get_feature_names_out().tolist()

        # BR baseline
        br_scores = _fit_br_stage(X_train, y_train, X_test, config.random_state)

        # Bridge words
        doc_counts = _build_fold_doc_counts(train_df, config.bridge_min_total_docs)
        bridge_df = _extract_bridge_words(doc_counts, config)

        # --- weighted ---
        bridge_matrix_w = _compute_bridge_score_matrix(
            X_test, gt_main_test, bridge_df, feature_names
        )
        y_pred_w = _apply_bridge_reranking(br_scores, gt_main_test, bridge_matrix_w, lambda_val, tau_val)
        m_w = compute_recovery_metrics("weighted", y_test, y_pred_w, gt_main_test)
        m_w["fold"] = fold_idx
        all_results.append(m_w)

        # --- uniform ---
        uniform_df = _build_uniform_bridge_dict(bridge_df)
        bridge_matrix_u = _compute_bridge_score_matrix(
            X_test, gt_main_test, uniform_df, feature_names
        )
        y_pred_u = _apply_bridge_reranking(br_scores, gt_main_test, bridge_matrix_u, lambda_val, tau_val)
        m_u = compute_recovery_metrics("uniform", y_test, y_pred_u, gt_main_test)
        m_u["fold"] = fold_idx
        all_results.append(m_u)

        # --- random (n_random_repeats 회 반복 평균) ---
        all_words = list(set(feature_names))
        random_metrics = []
        for rep in range(n_random_repeats):
            random_df = _build_random_bridge_dict(bridge_df, all_words, rng)
            bridge_matrix_r = _compute_bridge_score_matrix(
                X_test, gt_main_test, random_df, feature_names
            )
            y_pred_r = _apply_bridge_reranking(br_scores, gt_main_test, bridge_matrix_r, lambda_val, tau_val)
            m_r = compute_recovery_metrics(f"random_rep{rep}", y_test, y_pred_r, gt_main_test)
            random_metrics.append(m_r)

        # Average random repeats
        avg_random = {"stage": "random", "fold": fold_idx}
        numeric_keys = [k for k in random_metrics[0] if isinstance(random_metrics[0][k], (int, float)) and k != "fold"]
        for k in numeric_keys:
            vals = [m[k] for m in random_metrics if k in m]
            avg_random[k] = float(np.mean(vals))
        all_results.append(avg_random)

        print(f"  weighted IRR={m_w['information_recovery_rate']:.3f}  "
              f"uniform IRR={m_u['information_recovery_rate']:.3f}  "
              f"random IRR={avg_random['information_recovery_rate']:.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "ablation_results_by_fold.csv", encoding="utf-8-sig", index=False)

    # Summary
    summary_rows = []
    for stage in ["weighted", "uniform", "random"]:
        sub = results_df[results_df["stage"] == stage]
        row = {"stage": stage}
        for col in ["information_recovery_rate", "micro_f1", "macro_f1", "hamming_loss"]:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"] = float(sub[col].std())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ablation_summary.csv", encoding="utf-8-sig", index=False)

    print(f"\n[ABLATION] 완료: {output_dir}")
    return {"status": "done", "results_df": results_df, "summary_df": summary_df}
