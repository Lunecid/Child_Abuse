"""
information_recovery.py
=======================
기록 시스템의 단일 레이블 제약으로 소실된 동반 학대 정보를,
아동 발화 텍스트로부터 복원하는 실험 파이프라인.

핵심 원칙: 임상가의 판단(GT)이 제1원칙이다.
  main_label 출처 = info["학대의심"] (GT), classify_abuse_main_sub()가 아님.

Steps:
  1. build_gt_anchored_dataset     — GT 기반 데이터셋 구축
  3. compute_label_composition     — main-only vs main+sub 비율
  4. quantify_information_loss     — 정보 손실 정량화
  5. run_recovery_experiment       — 4-Stage 다중 레이블 분류 실험
  6. compute_recovery_metrics      — 복원율 측정
  7. analyze_recovery_failures     — 복원 실패 분석
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean
from abuse_pipeline.analysis.compare_abuse_labels import (
    extract_gt_abuse_types_from_info,
    normalize_abuse_label,
)

try:
    from scipy.special import expit as sigmoid
except ImportError:
    def sigmoid(x):
        return 1.0 / (1.0 + np.asarray(x, dtype=float).__neg__().__rpow__(np.e).__radd__(1.0).__rtruediv__(1.0))

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════
TYPE_TO_IDX = {a: i for i, a in enumerate(C.ABUSE_ORDER)}
IDX_TO_TYPE = {i: a for a, i in TYPE_TO_IDX.items()}

# ═══════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RecoveryConfig:
    output_dir: Path = Path("results/recovery")
    n_splits: int = 5
    inner_splits: int = 3
    random_state: int = 42
    chi_top_k: int = 200
    bridge_min_total_docs: int = C.MIN_DOC_COUNT   # 5
    bridge_count_min: int = C.BRIDGE_MIN_COUNT      # 5
    bridge_min_p1: float = C.BRIDGE_MIN_P1          # 0.40
    bridge_min_p2: float = C.BRIDGE_MIN_P2          # 0.25
    bridge_max_gap: float = C.BRIDGE_MAX_GAP        # 0.20
    lambda_candidates: tuple = (0.1, 0.5, 1.0, 2.0, 5.0)
    tau_candidates: tuple = (-0.1, 0.0, 0.1, 0.2, 0.3)


# ═══════════════════════════════════════════════════════════════════
#  Step 1: GT 기반 데이터셋 구축
# ═══════════════════════════════════════════════════════════════════

def _normalize_doc_id(info: dict) -> str:
    return str(info.get("ID") or info.get("id") or "unknown")


def build_gt_anchored_dataset(
    json_files: list[str | Path],
    only_negative: bool = True,
) -> pd.DataFrame:
    """
    GT(임상가 판단) 기반 데이터셋 구축.

    핵심 변경: main_label ← extract_gt_abuse_types_from_info(info)
    기존:      main_label ← classify_abuse_main_sub(rec)[0]
    """
    if not C.HAS_SKLEARN:
        raise ImportError("scikit-learn 필요")

    rows: list[dict[str, Any]] = []

    for json_path in sorted(str(p) for p in json_files):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        # 1) 정서가 필터
        if only_negative and classify_child_group(rec) != "부정":
            continue

        # 2) GT 레이블 추출 (★ 핵심)
        info = rec.get("info", {}) or {}
        gt_label_set = extract_gt_abuse_types_from_info(info, field="학대의심")
        gt_label_set = {x for x in gt_label_set if x in C.ABUSE_ORDER}
        if not gt_label_set:
            continue

        # GT가 여러 개면 SEVERITY_RANK 기준으로 대표 1개 선택
        gt_main = sorted(gt_label_set, key=lambda x: C.SEVERITY_RANK.get(x, 999))[0]

        # 3) 알고리즘 라벨 (sub 탐지용)
        try:
            algo_main, algo_subs = classify_abuse_main_sub(rec)
        except Exception:
            algo_main, algo_subs = None, []
        algo_subs = algo_subs or []
        algo_set = set()
        if algo_main and algo_main in C.ABUSE_ORDER:
            algo_set.add(algo_main)
        algo_set |= {s for s in algo_subs if s in C.ABUSE_ORDER}

        # 4) 문항 점수 추출
        abuse_scores = {a: 0 for a in C.ABUSE_ORDER}
        for q in rec.get("list", []):
            if q.get("문항") == "학대여부":
                for it in q.get("list", []):
                    name = it.get("항목")
                    try:
                        sc = int(it.get("점수"))
                    except (TypeError, ValueError):
                        sc = 0
                    if not isinstance(name, str):
                        continue
                    for a in C.ABUSE_ORDER:
                        if a in name:
                            abuse_scores[a] += sc

        # 5) 아동 발화 추출 + 토큰화
        speech_list = extract_child_speech(rec)
        if not speech_list:
            continue
        raw_text = " ".join(str(s).strip() for s in speech_list if str(s).strip()).strip()
        if not raw_text:
            continue
        tokenized = " ".join(tokenize_korean(raw_text)).strip()
        if not tokenized:
            continue

        # 6) 다중 레이블 GT 구성: gt_main 무조건 포함 + algo_subs 추가
        label_set = {gt_main}
        for sub in algo_subs:
            if sub in C.ABUSE_ORDER and sub != gt_main:
                label_set.add(sub)

        rows.append({
            "doc_id": _normalize_doc_id(info),
            "source_file": str(json_path),
            "raw_text": raw_text,
            "text": tokenized,
            "gt_main": gt_main,
            "algo_main": algo_main or "",
            "algo_subs": algo_subs,
            "algo_set": sorted(algo_set, key=lambda x: C.SEVERITY_RANK.get(x, 999)),
            "abuse_scores": abuse_scores,
            "label_list": sorted(label_set, key=lambda x: C.ABUSE_ORDER.index(x)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 이진 벡터 생성
    mlb = C.MultiLabelBinarizer(classes=list(C.ABUSE_ORDER))
    mlb.fit([[]])
    y_multi = mlb.transform(df["label_list"])
    for idx, label in enumerate(C.ABUSE_ORDER):
        df[f"y_{label}"] = y_multi[:, idx]
    df["n_labels"] = df["label_list"].apply(len)

    return df


# ═══════════════════════════════════════════════════════════════════
#  Step 3: main-only vs main+sub 비율 집계
# ═══════════════════════════════════════════════════════════════════

def compute_label_composition(dataset_df: pd.DataFrame) -> dict[str, Any]:
    """
    Group A (main only): n_labels == 1 & algo_main 존재
    Group B (main + sub): n_labels >= 2
    Group C (algo blind):  algo_main 없음 (GT만 존재)
    """
    n = len(dataset_df)

    has_algo = dataset_df["algo_main"].astype(str).replace("", pd.NA).notna()
    group_c = ~has_algo
    group_a = has_algo & (dataset_df["n_labels"] == 1)
    group_b = has_algo & (dataset_df["n_labels"] >= 2)
    # n_labels == 1이면서 algo_main 없는 경우도 Group C에 포함
    group_c = group_c | (~has_algo & (dataset_df["n_labels"] == 1))

    # 유형별 세부 분포
    detail_rows = []
    for gt in C.ABUSE_ORDER:
        mask = dataset_df["gt_main"] == gt
        detail_rows.append({
            "gt_main": gt,
            "total": int(mask.sum()),
            "group_a": int((mask & group_a).sum()),
            "group_b": int((mask & group_b).sum()),
            "group_c": int((mask & group_c).sum()),
        })
    detail_df = pd.DataFrame(detail_rows)

    return {
        "total": n,
        "group_a_count": int(group_a.sum()),
        "group_a_pct": float(group_a.mean() * 100),
        "group_b_count": int(group_b.sum()),
        "group_b_pct": float(group_b.mean() * 100),
        "group_c_count": int(group_c.sum()),
        "group_c_pct": float(group_c.mean() * 100),
        "detail_df": detail_df,
    }


# ═══════════════════════════════════════════════════════════════════
#  Step 4: 정보 손실 정량화
# ═══════════════════════════════════════════════════════════════════

def quantify_information_loss(dataset_df: pd.DataFrame) -> dict[str, Any]:
    """기록 시스템에 gt_main 1개만 남을 때 소실되는 정보를 유형별로 집계."""
    lost_by_type = {a: 0 for a in C.ABUSE_ORDER}
    loss_pairs = {(a, b): 0 for a in C.ABUSE_ORDER for b in C.ABUSE_ORDER if a != b}
    detail_rows = []

    for _, row in dataset_df.iterrows():
        recorded = {row["gt_main"]}
        full_set = set(row["label_list"])
        lost = full_set - recorded
        for lbl in lost:
            lost_by_type[lbl] += 1
            loss_pairs[(row["gt_main"], lbl)] += 1
        detail_rows.append({
            "doc_id": row["doc_id"],
            "gt_main": row["gt_main"],
            "full_labels": "|".join(row["label_list"]),
            "lost_labels": "|".join(sorted(lost, key=lambda x: C.SEVERITY_RANK.get(x, 999))),
            "n_lost": len(lost),
        })

    total_lost = sum(lost_by_type.values())
    n_with_loss = sum(1 for r in detail_rows if r["n_lost"] > 0)

    # Pair matrix
    pair_df = pd.DataFrame(0, index=C.ABUSE_ORDER, columns=C.ABUSE_ORDER, dtype=int)
    for (main, sub), cnt in loss_pairs.items():
        pair_df.loc[main, sub] = cnt

    return {
        "lost_by_type": lost_by_type,
        "total_lost_labels": total_lost,
        "n_children_with_loss": n_with_loss,
        "avg_lost_per_child": total_lost / len(dataset_df) if len(dataset_df) else 0.0,
        "loss_pair_matrix": pair_df,
        "loss_detail_df": pd.DataFrame(detail_rows),
    }


# ═══════════════════════════════════════════════════════════════════
#  Step 6: 복원율 측정
# ═══════════════════════════════════════════════════════════════════

def compute_recovery_metrics(
    stage_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gt_main_indices: np.ndarray,
) -> dict[str, Any]:
    """
    복원 프레이밍 메트릭.

    삭제: main_preserved, main_accuracy, main_macro_f1
    변경: companion_recall → information_recovery_rate
    """
    from sklearn.metrics import f1_score, hamming_loss as sklearn_hamming

    N = len(y_true)

    # information_recovery_rate
    recovery_rates = []
    for i in range(N):
        true_labels = set(np.where(y_true[i] == 1)[0])
        main_idx = int(gt_main_indices[i])
        lost = true_labels - {main_idx}
        if not lost:
            continue
        pred_labels = set(np.where(y_pred[i] == 1)[0])
        recovered = pred_labels & lost
        recovery_rates.append(len(recovered) / len(lost))

    info_recovery = float(np.mean(recovery_rates)) if recovery_rates else 0.0

    # boundary recovery (6 pairs)
    boundary = {}
    for a, b in combinations(range(len(C.ABUSE_ORDER)), 2):
        key = f"boundary_{C.ABUSE_ORDER[a]}_{C.ABUSE_ORDER[b]}"
        pair_cases = []
        for i in range(N):
            true_set = set(np.where(y_true[i] == 1)[0])
            if a in true_set and b in true_set:
                pred_set = set(np.where(y_pred[i] == 1)[0])
                pair_cases.append(int(a in pred_set and b in pred_set))
        boundary[key] = float(np.mean(pair_cases)) if pair_cases else float("nan")

    return {
        "stage": stage_name,
        "information_recovery_rate": info_recovery,
        "recovery_failure_rate": 1.0 - info_recovery,
        "recovery_cases": len(recovery_rates),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(sklearn_hamming(y_true, y_pred)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        **boundary,
    }


# ═══════════════════════════════════════════════════════════════════
#  Step 5: 4-Stage 복원 실험
# ═══════════════════════════════════════════════════════════════════

def _safe_fit_vectorizer(train_texts: list[str]):
    """TF-IDF 벡터화. min_df 자동 완화."""
    vec = C.TfidfVectorizer(**C.TFIDF_PARAMS)
    try:
        X = vec.fit_transform(train_texts)
    except ValueError as e:
        if "no terms remain" not in str(e):
            raise
        relaxed = dict(C.TFIDF_PARAMS)
        relaxed["min_df"] = 1
        vec = C.TfidfVectorizer(**relaxed)
        X = vec.fit_transform(train_texts)
    return vec, X


def _fit_br_stage(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    random_state: int = 42,
) -> np.ndarray:
    """Binary Relevance: 유형별 독립 이진 분류기."""
    try:
        from scipy.special import expit as _sigmoid
    except ImportError:
        _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

    n_test = X_test.shape[0]
    n_labels = y_train.shape[1]
    scores = np.zeros((n_test, n_labels), dtype=float)

    for j in range(n_labels):
        y_col = y_train[:, j]
        if np.unique(y_col).size < 2:
            scores[:, j] = float(y_col[0])
            continue

        clf = C.LinearSVC(max_iter=1000, dual="auto", random_state=random_state)
        clf.fit(X_train, y_col)
        scores[:, j] = _sigmoid(clf.decision_function(X_test))

    return scores


def _fit_cc_stage(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    random_state: int = 42,
) -> np.ndarray:
    """Classifier Chain: ABUSE_ORDER 순서로 체이닝."""
    try:
        from scipy.special import expit as _sigmoid
    except ImportError:
        _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
    from scipy.sparse import issparse, hstack as sp_hstack

    n_test = X_test.shape[0]
    n_labels = y_train.shape[1]
    scores = np.zeros((n_test, n_labels), dtype=float)

    # chain augmented features
    train_aug = X_train
    test_aug = X_test

    for j in range(n_labels):
        y_col = y_train[:, j]
        if np.unique(y_col).size < 2:
            scores[:, j] = float(y_col[0])
        else:
            clf = C.LinearSVC(max_iter=1000, dual="auto", random_state=random_state)
            clf.fit(train_aug, y_col)
            scores[:, j] = _sigmoid(clf.decision_function(test_aug))

        # augment with predictions for next classifier in chain
        if j < n_labels - 1:
            pred_train = y_train[:, j].reshape(-1, 1).astype(float)
            pred_test = (scores[:, j] > 0.5).astype(float).reshape(-1, 1)
            if issparse(train_aug):
                from scipy.sparse import csr_matrix
                train_aug = sp_hstack([train_aug, csr_matrix(pred_train)])
                test_aug = sp_hstack([test_aug, csr_matrix(pred_test)])
            else:
                train_aug = np.hstack([train_aug, pred_train])
                test_aug = np.hstack([test_aug, pred_test])

    return scores


def build_doc_count_table_unified(
    train_df: pd.DataFrame,
    min_total_docs: int = C.MIN_DOC_COUNT,
    main_col: str = "gt_main",
) -> pd.DataFrame:
    """
    메인 파이프라인의 build_abuse_doc_word_table() + pivot과 동일한 결과를 생성.

    메인 파이프라인과의 일관성:
      - doc_level.py의 build_abuse_doc_word_table()은 JSON을 직접 읽지만,
        여기서는 이미 구축된 train_df에서 동일한 정보를 추출한다.
      - train_df["text"]는 tokenize_korean() 결과의 join이므로,
        split()하면 원래 토큰 목록이 복원된다.
      - 문서당 고유 단어만 카운트 (set 사용) — doc_level.py와 동일.

    Parameters
    ----------
    train_df : pd.DataFrame
        columns: doc_id, text (토큰화 완료), gt_main, ...
    min_total_docs : int
        전체 문서 빈도가 이 값 미만인 단어는 제외 (= MIN_DOC_COUNT)
    main_col : str
        main label 컬럼명. "gt_main"이면 GT 사용.

    Returns
    -------
    pd.DataFrame
        index = word, columns = ABUSE_ORDER (= [성학대, 신체학대, 정서학대, 방임])
        값 = 해당 학대유형 문서에서 해당 단어가 출현한 문서 수

        이 형식은 메인 파이프라인에서 ca.py가 받는 df_abuse_counts와 동일하며,
        compute_chi_square(), compute_log_odds(), compute_prob_bridge_for_words()에
        직접 전달할 수 있다.
    """
    rows = []
    for row in train_df.itertuples(index=False):
        main_label = getattr(row, main_col)
        if main_label not in C.ABUSE_ORDER:
            continue
        # set() → 문서당 1회 카운트 (doc_level.py와 동일)
        tokens = set(str(getattr(row, "text", "")).split())
        if not tokens:
            continue
        for word in tokens:
            rows.append({"word": word, "abuse": main_label})

    if not rows:
        return pd.DataFrame(columns=list(C.ABUSE_ORDER))

    # pivot: long format → word × abuse 빈도 테이블
    count_df = (
        pd.DataFrame(rows)
        .groupby(["word", "abuse"])
        .size()
        .unstack("abuse")
        .fillna(0)
        .astype(int)
    )

    # ABUSE_ORDER에 없는 컬럼 보충
    for label in C.ABUSE_ORDER:
        if label not in count_df.columns:
            count_df[label] = 0
    count_df = count_df[list(C.ABUSE_ORDER)]

    # 최소 문서 빈도 필터 (= doc_level.py의 min_doc_count)
    count_df["total_docs"] = count_df.sum(axis=1)
    count_df = count_df[count_df["total_docs"] >= min_total_docs].drop(columns="total_docs")

    return count_df


def _extract_bridge_words(
    doc_counts: pd.DataFrame,
    config: RecoveryConfig,
) -> pd.DataFrame:
    """
    fold 학습 데이터에서 브릿지 워드 추출.
    pipeline.py lines 682-707과 동일한 로직.

    호출 체인:
      1) compute_log_odds  → 안정성 필터용 (pipeline line 682)
      2) compute_chi_square + add_bh_fdr → 후보 선정 (pipeline lines 683-684)
      3) 상위 chi_top_k개 안정 정렬 (pipeline line 691, kind="mergesort")
      4) compute_prob_bridge_for_words → 최종 브릿지 필터 (pipeline lines 697-707)

    경로 1(메인 파이프라인)과의 일관성 검증:
      - count_df의 형식: index=word, columns=ABUSE_ORDER → 동일
      - chi_top_k=200 → ca.py의 top_chi_for_ca=200과 동일
      - min_p1/min_p2/max_gap: C.BRIDGE_MIN_P1/P2/MAX_GAP과 동일 값 사용
      - count_min: C.BRIDGE_MIN_COUNT (=5) 와 동일
    """
    from abuse_pipeline.stats.stats import (
        compute_chi_square,
        compute_log_odds,
        compute_prob_bridge_for_words,
        add_bh_fdr,
    )

    _empty = pd.DataFrame(columns=["word", "primary_abuse", "secondary_abuse", "p1", "p2", "gap"])

    if doc_counts.empty:
        return _empty

    # ── 1단계: 로그오즈비 (전체 빈도표) ── pipeline line 682
    logodds_df = compute_log_odds(doc_counts, C.ABUSE_ORDER)

    # ── 2단계: 카이제곱 (전체 빈도표) ── pipeline line 683
    chi_df = compute_chi_square(doc_counts, C.ABUSE_ORDER)
    if chi_df.empty:
        return _empty

    # ── BH-FDR 보정 ── pipeline line 684
    chi_df = add_bh_fdr(chi_df, p_col="p_value", out_col="p_fdr_bh")

    # ── 3단계: 상위 chi_top_k 후보 (안정 정렬) ── pipeline line 691
    chi_sorted = chi_df.sort_values("chi2", ascending=False, kind="mergesort")
    candidate_words = chi_sorted.head(
        min(config.chi_top_k, len(chi_sorted))
    ).index.tolist()

    # ── 디버그 로그 (경로 일관성 확인용) ──
    print(f"    [BRIDGE] count_df: {len(doc_counts)} words")
    print(f"    [BRIDGE] chi_top_k={config.chi_top_k}, candidates={len(candidate_words)}")

    # ── 4단계: 브릿지 필터 ── pipeline lines 697-707
    #   ca.py의 호출과 동일한 파라미터:
    #     min_p1=C.BRIDGE_MIN_P1 (0.40)
    #     min_p2=C.BRIDGE_MIN_P2 (0.25)
    #     max_gap=C.BRIDGE_MAX_GAP (0.20)
    #     count_min=C.BRIDGE_MIN_COUNT (5)
    #     logodds_min=None, z_min=None
    bridge_df = compute_prob_bridge_for_words(
        df_counts=doc_counts,
        words=candidate_words,
        logodds_df=logodds_df,
        min_p1=config.bridge_min_p1,
        min_p2=config.bridge_min_p2,
        max_gap=config.bridge_max_gap,
        logodds_min=None,
        count_min=config.bridge_count_min,
        z_min=None,
    )

    # ── 디버그 로그 ──
    n_bridge = bridge_df["word"].nunique() if not bridge_df.empty else 0
    print(f"    [BRIDGE] min_p1={config.bridge_min_p1}, min_p2={config.bridge_min_p2}, "
          f"max_gap={config.bridge_max_gap}, count_min={config.bridge_count_min}")
    print(f"    [BRIDGE] result: {n_bridge} unique bridge words from {len(candidate_words)} candidates")

    # ── 기대값 검증 ──
    if n_bridge > 50:
        print(f"    [BRIDGE] ⚠ WARNING: {n_bridge} bridge words is unexpectedly high. "
              f"Expected 20-35. Check filter parameters.")

    return bridge_df


def _compute_bridge_score_matrix(
    X_tfidf,
    gt_main_indices: np.ndarray,
    bridge_df: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    브릿지 점수 행렬 계산.

    bridge_score(d, c) = Σ_w tfidf(d,w) × weight(w, gt_main(d), c)
    여기서 weight는 브릿지 워드의 p2 값 (gt_main → c 방향일 때만).
    """
    from scipy.sparse import issparse

    N = X_tfidf.shape[0]
    n_labels = len(C.ABUSE_ORDER)
    scores = np.zeros((N, n_labels), dtype=float)

    if bridge_df.empty:
        return scores

    # Build bridge weight lookup: {word: {(primary, secondary): p2}}
    feat_idx = {w: i for i, w in enumerate(feature_names)}
    bridge_lookup: dict[str, list[tuple[str, str, float]]] = {}
    for _, brow in bridge_df.iterrows():
        w = brow["word"]
        if w not in feat_idx:
            continue
        pa = brow.get("primary_abuse", "")
        sa = brow.get("secondary_abuse", "")
        p2 = float(brow.get("p2", 0.0))
        if pa in TYPE_TO_IDX and sa in TYPE_TO_IDX:
            bridge_lookup.setdefault(w, []).append((pa, sa, p2))

    if not bridge_lookup:
        return scores

    X_dense = X_tfidf.toarray() if issparse(X_tfidf) else np.asarray(X_tfidf)

    for i in range(N):
        main_type = IDX_TO_TYPE.get(int(gt_main_indices[i]), "")
        for w, entries in bridge_lookup.items():
            fidx = feat_idx[w]
            tfidf_val = X_dense[i, fidx]
            if tfidf_val == 0:
                continue
            for pa, sa, p2 in entries:
                # 이 아동의 gt_main이 primary이면, secondary 방향에 가산
                if pa == main_type:
                    scores[i, TYPE_TO_IDX[sa]] += tfidf_val * p2
                # 반대 방향도 가산 (양방향)
                elif sa == main_type:
                    scores[i, TYPE_TO_IDX[pa]] += tfidf_val * p2

    return scores


def _apply_bridge_reranking(
    br_scores: np.ndarray,
    gt_main_indices: np.ndarray,
    bridge_score_matrix: np.ndarray,
    lambda_val: float,
    tau: float,
) -> np.ndarray:
    """브릿지 리랭킹 적용. GT anchor 강제."""
    N = br_scores.shape[0]
    adjusted = br_scores + lambda_val * bridge_score_matrix
    y_pred = (adjusted > (0.5 + tau)).astype(int)

    # GT anchor 강제: gt_main 위치는 항상 1
    for i in range(N):
        y_pred[i, int(gt_main_indices[i])] = 1

    return y_pred


def _tune_bridge_params(
    train_df: pd.DataFrame,
    y_train_bin: np.ndarray,
    config: RecoveryConfig,
) -> tuple[float, float]:
    """내부 CV로 λ, τ 최적화."""
    gt_main_indices = np.array([
        TYPE_TO_IDX[m] for m in train_df["gt_main"]
    ])

    best_score = -1.0
    best_lam, best_tau = 1.0, 0.0

    n_inner = min(config.inner_splits, int(train_df["gt_main"].value_counts().min()))
    if n_inner < 2:
        return best_lam, best_tau

    skf_inner = C.StratifiedKFold(
        n_splits=n_inner, shuffle=True, random_state=config.random_state
    )
    texts = train_df["text"].tolist()
    y_main_str = train_df["gt_main"].values

    for lam in config.lambda_candidates:
        for tau in config.tau_candidates:
            fold_scores = []
            for inner_train, inner_val in skf_inner.split(np.zeros(len(texts)), y_main_str):
                inner_train_texts = [texts[i] for i in inner_train]
                inner_val_texts = [texts[i] for i in inner_val]
                inner_y_train = y_train_bin[inner_train]
                inner_y_val = y_train_bin[inner_val]
                inner_gt_idx = gt_main_indices[inner_val]

                vec, X_tr = _safe_fit_vectorizer(inner_train_texts)
                X_va = vec.transform(inner_val_texts)

                br_scores = _fit_br_stage(X_tr, inner_y_train, X_va, config.random_state)

                # Build bridge words from inner train
                inner_train_df = train_df.iloc[inner_train]
                doc_counts = build_doc_count_table_unified(inner_train_df, config.bridge_min_total_docs)
                bridge_df = _extract_bridge_words(doc_counts, config)
                bridge_matrix = _compute_bridge_score_matrix(
                    X_va, inner_gt_idx, bridge_df, vec.get_feature_names_out().tolist()
                )
                y_pred = _apply_bridge_reranking(br_scores, inner_gt_idx, bridge_matrix, lam, tau)

                # macro F1 as optimization target (balances precision and recall)
                from sklearn.metrics import f1_score as _f1
                fold_scores.append(_f1(inner_y_val, y_pred, average="macro", zero_division=0))

            avg = np.mean(fold_scores)
            if avg > best_score:
                best_score = avg
                best_lam, best_tau = lam, tau

    return best_lam, best_tau


def run_recovery_experiment(
    dataset_df: pd.DataFrame,
    config: RecoveryConfig,
) -> dict[str, Any]:
    """
    4-Stage 비교 실험.

    Stage A: 기록 시스템 시뮬레이션 (gt_main만 1, 나머지 0)
    Stage B1: Binary Relevance (TF-IDF + LinearSVC)
    Stage B2: Classifier Chain
    Stage C: Bridge Reranking (GT anchor)
    """
    from sklearn.metrics import f1_score

    output_dir = Path(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "latex_tables", exist_ok=True)

    label_order = list(C.ABUSE_ORDER)
    n_labels = len(label_order)

    # Prepare arrays
    mlb = C.MultiLabelBinarizer(classes=label_order)
    mlb.fit([[]])
    y_all = mlb.transform(dataset_df["label_list"].tolist())
    gt_main_all = np.array([TYPE_TO_IDX[m] for m in dataset_df["gt_main"]])

    texts = dataset_df["text"].tolist()
    y_main_str = dataset_df["gt_main"].values

    # CV
    class_counts = dataset_df["gt_main"].value_counts()
    actual_splits = min(config.n_splits, int(class_counts.min()))
    if actual_splits < 2:
        print("[RECOVERY] 클래스 수 부족으로 실험 건너뜀")
        return {"status": "skipped"}

    skf = C.StratifiedKFold(
        n_splits=actual_splits, shuffle=True, random_state=config.random_state
    )

    all_fold_results = []
    bridge_stability = []
    grid_search_log = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(texts)), y_main_str), start=1
    ):
        print(f"\n[RECOVERY] Fold {fold_idx}/{actual_splits}  "
              f"Train={len(train_idx)} Test={len(test_idx)}")

        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        gt_main_test = gt_main_all[test_idx]
        train_df = dataset_df.iloc[train_idx].reset_index(drop=True)

        # TF-IDF
        vec, X_train = _safe_fit_vectorizer(train_texts)
        X_test = vec.transform(test_texts)
        feature_names = vec.get_feature_names_out().tolist()

        # ── Stage A: 기록 시스템 시뮬레이션 ──
        y_pred_a = np.zeros_like(y_test)
        for i in range(len(test_idx)):
            y_pred_a[i, int(gt_main_test[i])] = 1
        metrics_a = compute_recovery_metrics("A_baseline", y_test, y_pred_a, gt_main_test)

        # ── Stage B1: Binary Relevance ──
        br_scores = _fit_br_stage(X_train, y_train, X_test, config.random_state)
        y_pred_b1 = (br_scores > 0.5).astype(int)
        # GT anchor 강제
        for i in range(len(test_idx)):
            y_pred_b1[i, int(gt_main_test[i])] = 1
        metrics_b1 = compute_recovery_metrics("B1_BR", y_test, y_pred_b1, gt_main_test)

        # ── Stage B2: Classifier Chain ──
        cc_scores = _fit_cc_stage(X_train, y_train, X_test, config.random_state)
        y_pred_b2 = (cc_scores > 0.5).astype(int)
        for i in range(len(test_idx)):
            y_pred_b2[i, int(gt_main_test[i])] = 1
        metrics_b2 = compute_recovery_metrics("B2_CC", y_test, y_pred_b2, gt_main_test)

        # ── Stage C: Bridge Reranking ──
        # Tune λ, τ on train set
        best_lam, best_tau = _tune_bridge_params(train_df, y_train, config)
        grid_search_log.append({
            "fold": fold_idx, "best_lambda": best_lam, "best_tau": best_tau
        })
        print(f"  Bridge params: λ={best_lam}, τ={best_tau}")

        # Bridge words from full train set
        doc_counts = build_doc_count_table_unified(train_df, config.bridge_min_total_docs)
        bridge_df = _extract_bridge_words(doc_counts, config)
        bridge_matrix = _compute_bridge_score_matrix(
            X_test, gt_main_test, bridge_df, feature_names
        )
        y_pred_c = _apply_bridge_reranking(br_scores, gt_main_test, bridge_matrix, best_lam, best_tau)
        metrics_c = compute_recovery_metrics("C_Bridge", y_test, y_pred_c, gt_main_test)

        # Bridge stability tracking
        bridge_words_set = set(bridge_df["word"].tolist()) if not bridge_df.empty else set()
        bridge_stability.append({
            "fold": fold_idx,
            "n_bridge_words": len(bridge_words_set),
            "bridge_words": "|".join(sorted(bridge_words_set)),
        })

        # Fold results
        for m in [metrics_a, metrics_b1, metrics_b2, metrics_c]:
            m["fold"] = fold_idx
        all_fold_results.extend([metrics_a, metrics_b1, metrics_b2, metrics_c])

        print(f"  A_baseline IRR={metrics_a['information_recovery_rate']:.3f}  "
              f"B1_BR IRR={metrics_b1['information_recovery_rate']:.3f}  "
              f"B2_CC IRR={metrics_b2['information_recovery_rate']:.3f}  "
              f"C_Bridge IRR={metrics_c['information_recovery_rate']:.3f}")

    # Aggregate
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(output_dir / "recovery_results_by_fold.csv", encoding="utf-8-sig", index=False)

    summary_rows = []
    for stage in ["A_baseline", "B1_BR", "B2_CC", "C_Bridge"]:
        sub = results_df[results_df["stage"] == stage]
        row = {"stage": stage}
        for col in ["information_recovery_rate", "recovery_failure_rate",
                     "micro_f1", "macro_f1", "hamming_loss", "subset_accuracy"]:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"] = float(sub[col].std())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "recovery_summary.csv", encoding="utf-8-sig", index=False)

    # Bridge stability
    stability_df = pd.DataFrame(bridge_stability)
    stability_df.to_csv(output_dir / "bridge_fold_stability.csv", encoding="utf-8-sig", index=False)

    # Jaccard similarity between folds
    if len(bridge_stability) >= 2:
        bridge_sets = [set(bs["bridge_words"].split("|")) - {""} for bs in bridge_stability]
        jaccard_vals = []
        for i, j in combinations(range(len(bridge_sets)), 2):
            inter = len(bridge_sets[i] & bridge_sets[j])
            union = len(bridge_sets[i] | bridge_sets[j])
            jaccard_vals.append(inter / union if union else 0.0)
        avg_jaccard = np.mean(jaccard_vals) if jaccard_vals else 0.0
        print(f"\n[RECOVERY] Bridge word fold Jaccard: {avg_jaccard:.3f}")

    # Grid search log
    grid_df = pd.DataFrame(grid_search_log)
    grid_df.to_csv(output_dir / "bridge_grid_search_by_fold.csv", encoding="utf-8-sig", index=False)

    # LaTeX tables
    _write_latex_tables(summary_df, results_df, output_dir / "latex_tables")

    print(f"\n[RECOVERY] 완료. 결과: {output_dir}")
    return {
        "status": "done",
        "results_df": results_df,
        "summary_df": summary_df,
        "bridge_stability_df": stability_df,
    }


def _write_latex_tables(summary_df: pd.DataFrame, results_df: pd.DataFrame, latex_dir: Path):
    """논문용 LaTeX 테이블 생성."""
    os.makedirs(latex_dir, exist_ok=True)

    # Table: Recovery Comparison
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Information Recovery Comparison Across Stages}",
        r"\label{tab:recovery_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Stage & IRR $\uparrow$ & Micro-F1 $\uparrow$ & Macro-F1 $\uparrow$ & Hamming $\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        stage = row["stage"]
        irr = row.get("information_recovery_rate_mean", 0)
        irr_s = row.get("information_recovery_rate_std", 0)
        mif = row.get("micro_f1_mean", 0)
        maf = row.get("macro_f1_mean", 0)
        ham = row.get("hamming_loss_mean", 0)
        lines.append(
            f"  {stage} & {irr:.3f}$\\pm${irr_s:.3f} & {mif:.3f} & {maf:.3f} & {ham:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (latex_dir / "table_recovery_comparison.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table: Boundary Recovery
    boundary_cols = [c for c in results_df.columns if c.startswith("boundary_")]
    if boundary_cols:
        lines2 = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Pairwise Boundary Recovery Rates}",
            r"\label{tab:boundary_recovery}",
            r"\begin{tabular}{l" + "c" * len(boundary_cols) + "}",
            r"\toprule",
        ]
        header = "Stage & " + " & ".join(c.replace("boundary_", "").replace("_", "/") for c in boundary_cols) + r" \\"
        lines2.append(header)
        lines2.append(r"\midrule")
        for stage in ["A_baseline", "B1_BR", "B2_CC", "C_Bridge"]:
            sub = results_df[results_df["stage"] == stage]
            vals = " & ".join(f"{sub[c].mean():.3f}" for c in boundary_cols)
            lines2.append(f"  {stage} & {vals} \\\\")
        lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (latex_dir / "table_boundary_recovery.tex").write_text("\n".join(lines2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
#  Step 7: 복원 실패 분석
# ═══════════════════════════════════════════════════════════════════

def analyze_recovery_failures(
    experiment_results: dict[str, Any],
    dataset_df: pd.DataFrame,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """복원 실패 케이스 분석."""
    if output_dir is None:
        output_dir = Path("results/recovery")
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results_df = experiment_results.get("results_df")
    if results_df is None:
        return {"status": "no_results"}

    # 유형 쌍별 복원율 (전체 fold 평균)
    boundary_cols = [c for c in results_df.columns if c.startswith("boundary_")]
    pair_rows = []
    for stage in ["A_baseline", "B1_BR", "B2_CC", "C_Bridge"]:
        sub = results_df[results_df["stage"] == stage]
        row = {"stage": stage}
        for bc in boundary_cols:
            vals = sub[bc].dropna()
            row[bc + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
        pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(output_dir / "boundary_recovery_all_pairs.csv", encoding="utf-8-sig", index=False)

    # 텍스트 길이별 분석
    if "text" in dataset_df.columns:
        dataset_df = dataset_df.copy()
        dataset_df["text_len"] = dataset_df["text"].apply(lambda x: len(str(x).split()))
        length_bins = pd.qcut(dataset_df["text_len"], q=4, duplicates="drop")
        length_analysis = dataset_df.groupby(length_bins, observed=False).agg(
            n=("doc_id", "count"),
            mean_n_labels=("n_labels", "mean"),
            pct_multilabel=("n_labels", lambda x: (x > 1).mean() * 100),
        ).reset_index()
        length_analysis.to_csv(
            output_dir / "recovery_failure_by_text_length.csv", encoding="utf-8-sig", index=False
        )

    # 유형 조합별 분석
    combo_counts = dataset_df["label_list"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else str(x)
    ).value_counts().reset_index()
    combo_counts.columns = ["label_combo", "count"]
    combo_counts.to_csv(
        output_dir / "recovery_failure_analysis.csv", encoding="utf-8-sig", index=False
    )

    print(f"[RECOVERY] 실패 분석 완료: {output_dir}")
    return {"status": "done", "pair_df": pair_df}
