"""
bridge_threshold_justification.py
=================================
교량 단어(bridge word) 임계값의 객관적 근거를 확보하는 통합 검증 모듈.

세 가지 전략을 결합한다:
  [전략 1] 우연 수준(chance level) 기반 해석적 근거
  [전략 2] 순열 귀무분포(permutation null) → 단어별 경험적 p-value
  [전략 3] 부트스트랩 안정성 + 다중 설정 감도 분석

기존 doc_level.py의 run_bridge_bootstrap_and_shuffle_doc_level()은
"선정 여부"만 기록한다. 이 모듈은 단어별 p1·p2·gap의 **귀무분포 전체**를
기록하여, 관찰값이 귀무에서 얼마나 극단적인지 정량화한다.

최종 산출물:
  1) bridge_null_distributions.csv — 단어별 p1/p2/gap 귀무 분위수
  2) bridge_empirical_pvalues.csv  — 단어별 경험적 p-value
  3) bridge_confirmed_words.csv    — 3중 검증 통과 확정 교량 단어
  4) bridge_threshold_report.txt   — 논문용 보고서 텍스트

수정 로그 (Fix Log)
-------------------
[FIX-1] 경험적 p-value +1 보정 누락 (Phipson & Smyth, 2010)
  위치: compute_null_summaries_and_pvalues() 라인 302-307
  문제: p = count / n → 관찰값이 가장 극단적이면 p = 0.0 발생
  수정: p = (count + 1) / (n + 1) — doc_level.py 라인 279와 일관성 확보

[FIX-2] 3조건 동시충족 검정의 구조적 불가능성
  위치: compute_null_summaries_and_pvalues() 라인 310-314
  문제: p1_pval < 0.05 AND p2_pval < 0.05 AND gap_pval < 0.05를 동시에
        만족하는 것은 수학적으로 불가능 (K=4에서 p1_null 중앙값 ≈ 0.30-0.35)
  수정: 복합 통계량 bridge_score = p2 / (gap + ε) 추가.
        원래 3조건은 is_perm_significant_triple로 유지 (구조적 불가능성 기록용),
        실질 판정은 is_perm_significant (복합 통계량 기준)으로 전환.

[FIX-3] 빈 DataFrame 가드 누락
  위치: compute_null_summaries_and_pvalues(), compute_multiconfig_stability()
  문제: df_null.empty 시 groupby에서 KeyError/IndexError 발생
  수정: 함수 시작부에 빈 DataFrame 조기 반환 + 경고 메시지

[FIX-4] count_min 적용 불일치 (부트스트랩)
  위치: run_bootstrap_with_stats() 라인 411-414
  문제: total >= count_min만 검사하고 k1/k2 개별 카운트는 미검사
  수정: counts[k1] >= count_min AND counts[k2] >= count_min 추가.
        stats.py compute_prob_bridge_for_words() 라인 279-283과 일관성 확보.

[FIX-5] 데이터 소스 불일치 (valid_words 필터링)
  위치: run_bridge_threshold_justification() 라인 866-869
  문제: build_doc_level_abuse_counts(allowed_groups=None)은 VALENCE_ORDER
        필터 없이 모든 아동 포함, 반면 build_abuse_doc_word_table()은
        항상 VALENCE_ORDER 필터 적용 → df_abuse_counts_doc에는 있지만
        doc_to_words에는 없는 단어 발생 가능
  수정: valid_words를 df_abuse_counts_doc.index ∩ doc_to_words 어휘의
        교집합으로 필터링.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set

try:
    # 패키지로 임포트될 때 (pipeline.py 등에서 호출)
    from . import common as C
    from .common import (
        ABUSE_ORDER,
        ABUSE_LABEL_EN,
        BRIDGE_MIN_P1,
        BRIDGE_MIN_P2,
        BRIDGE_MAX_GAP,
        BRIDGE_MIN_COUNT,
        BRIDGE_P_CONFIGS,
    )
except ImportError:
    # 단독 실행 또는 PyCharm에서 직접 열 때
    import common as C
    from common import (
        ABUSE_ORDER,
        ABUSE_LABEL_EN,
        BRIDGE_MIN_P1,
        BRIDGE_MIN_P2,
        BRIDGE_MAX_GAP,
        BRIDGE_MIN_COUNT,
        BRIDGE_P_CONFIGS,
    )


# ═══════════════════════════════════════════════════════════════
#  상수
# ═══════════════════════════════════════════════════════════════
K = len(ABUSE_ORDER)                       # 학대유형 수 = 4
P_CHANCE = 1.0 / K                         # 균등 확률 = 0.25
BOOT_STABILITY_THRESHOLD = 0.80            # 부트스트랩 선정 확률 최소값
NULL_ALPHA = 0.05                          # 순열 귀무 유의수준
MAJORITY_RATIO = 0.5                       # 다중 설정 다수결 비율
DEFAULT_N_PERM = 500                       # 순열 반복 수
DEFAULT_N_BOOT = 500                       # 부트스트랩 반복 수


# ═══════════════════════════════════════════════════════════════
#  [전략 1] 우연 수준 해석적 근거
# ═══════════════════════════════════════════════════════════════

def chance_level_justification(
    min_p1: float = BRIDGE_MIN_P1,
    min_p2: float = BRIDGE_MIN_P2,
    max_gap: float = BRIDGE_MAX_GAP,
) -> dict:
    """
    K=4 학대유형의 균등 확률 P_chance = 1/K = 0.25를 기준으로
    각 임계값이 어떤 의미를 갖는지 해석적으로 정리한다.

    Returns
    -------
    dict with keys: K, P_chance, interpretations (list of str)
    """
    interp = []
    interp.append(
        f"K = {K} abuse types → P_chance = 1/K = {P_CHANCE:.4f}"
    )
    interp.append(
        f"min_p1 = {min_p1:.2f} = P_chance + {min_p1 - P_CHANCE:.2f}"
        f"  → primary type probability exceeds chance by {min_p1 - P_CHANCE:.2f}"
    )
    interp.append(
        f"min_p2 = {min_p2:.2f} = P_chance + {min_p2 - P_CHANCE:.2f}"
        f"  → secondary type probability at "
        f"{'chance level' if abs(min_p2 - P_CHANCE) < 0.01 else 'above chance'}"
    )
    interp.append(
        f"max_gap = {max_gap:.2f}"
        f"  → difference between top-2 types is at most {max_gap:.2f}"
        f" (if uniform among top-2: gap = 0)"
    )

    # "교량"의 이론적 범위
    # p1 ∈ [min_p1, min_p1 + max_gap] 에서 p2 = p1 - gap 이므로
    # 가장 "교량적"인 상태: p1 ≈ p2 ≈ 0.40 (gap → 0)
    # 가장 "경계적"인 상태: p1 = 0.40, p2 = 0.25, gap = 0.15
    interp.append(
        f"Theoretical bridge range: "
        f"most balanced (gap→0): p1≈p2≈{min_p1:.2f}, "
        f"least balanced: p1={min_p1:.2f}, p2={min_p2:.2f}, gap={min_p1 - min_p2:.2f}"
    )

    return {
        "K": K,
        "P_chance": P_CHANCE,
        "min_p1": min_p1,
        "min_p2": min_p2,
        "max_gap": max_gap,
        "interpretations": interp,
    }


# ═══════════════════════════════════════════════════════════════
#  [전략 2] 순열 귀무분포 — 단어별 p1·p2·gap 귀무분포 + p-value
# ═══════════════════════════════════════════════════════════════

def _compute_word_stats_from_counts(
    df_counts: pd.DataFrame,
    group_cols: list = None,
) -> pd.DataFrame:
    """
    df_counts (word × abuse 빈도표)에서 각 단어의 p1, p2, gap을 계산.

    Parameters
    ----------
    df_counts : pd.DataFrame
        index = word, columns ⊇ ABUSE_ORDER
    group_cols : list, optional
        사용할 학대유형 컬럼. None이면 ABUSE_ORDER 사용.

    Returns
    -------
    DataFrame: word, g1, g2, p1, p2, gap
    """
    if group_cols is None:
        group_cols = ABUSE_ORDER

    records = []
    for w in df_counts.index:
        row = df_counts.loc[w, group_cols].astype(float).values
        total = row.sum()
        if total <= 0:
            continue
        probs = row / total
        idx_sorted = np.argsort(-probs)
        if len(idx_sorted) < 2:
            continue
        k1, k2 = idx_sorted[0], idx_sorted[1]
        p1, p2 = probs[k1], probs[k2]
        records.append({
            "word": w,
            "g1": group_cols[k1],
            "g2": group_cols[k2],
            "p1": p1,
            "p2": p2,
            "gap": p1 - p2,
        })
    return pd.DataFrame(records)


def run_permutation_null_distributions(
    doc_to_words: Dict[int, set],
    labels_int: np.ndarray,
    target_words: List[str],
    n_perm: int = DEFAULT_N_PERM,
    count_min: int = BRIDGE_MIN_COUNT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    라벨 순열을 통해 각 단어의 p1·p2·gap 귀무분포를 생성한다.

    Parameters
    ----------
    doc_to_words : {doc_idx: set of words}
    labels_int   : 1-D array, labels_int[doc_idx] = abuse type index (0..K-1)
    target_words : 귀무분포를 구할 단어 목록 (보통 chi² top 200)
    n_perm       : 순열 반복 수
    count_min    : 최소 문서 수 (이 이하인 단어는 해당 반복에서 제외)
    seed         : 난수 시드

    Returns
    -------
    DataFrame: word, perm_idx, p1_null, p2_null, gap_null, g1_null, g2_null
    """
    rng = np.random.default_rng(seed)
    target_set = set(target_words)

    # 미리 word → doc_indices 매핑 (역색인)
    word_to_docs: Dict[str, List[int]] = {}
    for idx, words in doc_to_words.items():
        for w in words:
            if w in target_set:
                word_to_docs.setdefault(w, []).append(idx)

    # 결과 버퍼 (큰 리스트 → 마지막에 DataFrame)
    all_records = []

    for b in range(n_perm):
        perm_labels = rng.permutation(labels_int)

        # 단어별 카운트 계산 (target_words만)
        for w, doc_indices in word_to_docs.items():
            counts = np.zeros(K, dtype=int)
            for idx in doc_indices:
                counts[perm_labels[idx]] += 1

            total = counts.sum()
            if total < count_min:
                continue

            probs = counts / total
            idx_sorted = np.argsort(-probs)
            k1, k2 = idx_sorted[0], idx_sorted[1]

            all_records.append({
                "word": w,
                "perm_idx": b,
                "p1_null": probs[k1],
                "p2_null": probs[k2],
                "gap_null": probs[k1] - probs[k2],
                "g1_null": ABUSE_ORDER[k1],
                "g2_null": ABUSE_ORDER[k2],
            })

        if (b + 1) % 50 == 0:
            print(f"[PERM-NULL] {b+1}/{n_perm} 완료 "
                  f"(누적 레코드: {len(all_records):,})")

    df_null = pd.DataFrame(all_records)
    print(f"[PERM-NULL] 완료: {len(df_null):,} 레코드 "
          f"({len(target_words)} 단어 × ~{n_perm} 순열)")
    return df_null


def compute_null_summaries_and_pvalues(
    df_null: pd.DataFrame,
    df_observed: pd.DataFrame,
    n_perm: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    순열 귀무분포에서 분위수 요약 + 관찰값의 경험적 p-value를 계산한다.

    Parameters
    ----------
    df_null    : run_permutation_null_distributions() 결과
    df_observed: _compute_word_stats_from_counts() 결과 (관찰값)
    n_perm     : 순열 반복 수 (정규화용)

    Returns
    -------
    df_null_summary : word, p1_null_{5,25,50,75,95}, p2_null_..., gap_null_...
    df_pvalues      : word, p1_obs, p2_obs, gap_obs,
                      p1_pval (p1_obs > null의 비율),
                      p2_pval (p2_obs > null의 비율),
                      gap_pval (gap_obs < null의 비율, 작을수록 교량적),
                      is_bridge_significant (3조건 모두 충족)
    """
    # --- FIX-3: 빈 DataFrame 가드 ---
    if df_null.empty:
        import warnings
        warnings.warn("[PERM-PVAL] df_null이 비어있습니다. 순열에서 유효한 결과가 없습니다.")
        return pd.DataFrame(), pd.DataFrame()

    # --- 귀무분포 분위수 ---
    percentiles = [5, 25, 50, 75, 95]
    summary_records = []

    for w, grp in df_null.groupby("word"):
        rec = {"word": w, "n_perm_valid": len(grp)}
        for col_base in ["p1_null", "p2_null", "gap_null"]:
            vals = grp[col_base].values
            for pct in percentiles:
                rec[f"{col_base}_{pct}pct"] = np.percentile(vals, pct)
            rec[f"{col_base}_mean"] = vals.mean()
            rec[f"{col_base}_std"] = vals.std()
        summary_records.append(rec)

    df_null_summary = pd.DataFrame(summary_records)

    # --- 경험적 p-value ---
    obs_dict = df_observed.set_index("word")
    pval_records = []

    for w, grp in df_null.groupby("word"):
        if w not in obs_dict.index:
            continue

        obs = obs_dict.loc[w]
        p1_obs = obs["p1"]
        p2_obs = obs["p2"]
        gap_obs = obs["gap"]

        p1_nulls = grp["p1_null"].values
        p2_nulls = grp["p2_null"].values
        gap_nulls = grp["gap_null"].values
        n = len(grp)

        # ── FIX-1: +1 보정 (Phipson & Smyth, 2010) ──
        # doc_level.py line 279: p_perm = (count_ge + 1) / (n_perm + 1)
        # 관찰값이 귀무에서 가장 극단적이어도 p-value가 정확히 0이 되지 않도록 보정.
        # p1_pval: 귀무에서 관찰된 p1 이상이 나올 확률 (작을수록 p1이 극단적으로 높음)
        p1_pval = ((p1_nulls >= p1_obs).sum() + 1) / (n + 1)
        # p2_pval: 귀무에서 관찰된 p2 이상이 나올 확률 (작을수록 p2가 극단적으로 높음)
        p2_pval = ((p2_nulls >= p2_obs).sum() + 1) / (n + 1)
        # gap_pval: 귀무에서 관찰된 gap 이하가 나올 확률 (작을수록 gap이 극단적으로 작음)
        # → 교량 단어는 gap이 작아야 하므로, gap_pval이 작으면 유의미하게 교량적
        gap_pval = ((gap_nulls <= gap_obs).sum() + 1) / (n + 1)

        # ── FIX-2: 복합 통계량(bridge_score) 기반 검정 추가 ──
        # 문제: 3조건(p1, p2, gap) 동시 충족은 구조적으로 불가능하다.
        #   - K=4일 때, 귀무(균등)분포에서 p1_null 중앙값 ≈ 0.30~0.35
        #   - 교량 단어의 p1_obs ≈ 0.40은 귀무 대비 충분히 극단적이지 않음
        #   - 교량 단어는 gap이 작고 p2가 높은 것이 핵심 → 이를 하나의 통계량으로 통합
        # bridge_score = p2 / (gap + ε): 높을수록 교량적 (p2↑, gap↓)
        BRIDGE_SCORE_EPS = 0.01
        bridge_score_obs = p2_obs / (gap_obs + BRIDGE_SCORE_EPS)

        bs_nulls = p2_nulls / (gap_nulls + BRIDGE_SCORE_EPS)
        bs_pval = ((bs_nulls >= bridge_score_obs).sum() + 1) / (n + 1)

        # 원래 3조건 (구조적 불가능성 기록용으로 유지)
        is_sig_triple = (
            (p1_pval < NULL_ALPHA)
            and (p2_pval < NULL_ALPHA)
            and (gap_pval < NULL_ALPHA)
        )
        # 실질적 검정: 복합 통계량 기반
        is_sig_composite = (bs_pval < NULL_ALPHA)

        pval_records.append({
            "word": w,
            "p1_obs": p1_obs,
            "p2_obs": p2_obs,
            "gap_obs": gap_obs,
            "g1_obs": obs["g1"],
            "g2_obs": obs["g2"],
            "p1_empirical_pval": p1_pval,
            "p2_empirical_pval": p2_pval,
            "gap_empirical_pval": gap_pval,
            "bridge_score_obs": bridge_score_obs,
            "bridge_score_pval": bs_pval,
            "is_perm_significant_triple": is_sig_triple,   # 3조건 (기록용)
            "is_perm_significant": is_sig_composite,        # 복합 통계량 (실질 판정)
            "n_perm_valid": n,
        })

    df_pvalues = pd.DataFrame(pval_records)
    if not df_pvalues.empty:
        n_sig_triple = df_pvalues["is_perm_significant_triple"].sum()
        n_sig_composite = df_pvalues["is_perm_significant"].sum()
        n_total = len(df_pvalues)
        print(f"[PERM-PVAL] 순열 검정 유의미 단어 (α={NULL_ALPHA}):")
        print(f"  3조건 동시충족 (구조적 불가능 기록용): {n_sig_triple}/{n_total}")
        print(f"  복합 통계량 bridge_score 기준: {n_sig_composite}/{n_total}")

    return df_null_summary, df_pvalues


# ═══════════════════════════════════════════════════════════════
#  [전략 3] 부트스트랩 안정성 + 다중 설정 감도 분석
# ═══════════════════════════════════════════════════════════════

def run_bootstrap_with_stats(
    doc_to_words: Dict[int, set],
    labels_int: np.ndarray,
    target_words: List[str],
    p_configs: list = None,
    n_boot: int = DEFAULT_N_BOOT,
    count_min: int = BRIDGE_MIN_COUNT,
    seed: int = 123,
) -> pd.DataFrame:
    """
    문서 부트스트랩에서 각 설정별로 교량 단어 선정 확률을 계산한다.

    기존 doc_level.py의 함수와 동일한 로직이지만,
    여러 설정을 한 번에 처리하고 효율적으로 구현한다.

    Parameters
    ----------
    doc_to_words : {doc_idx: set of words}
    labels_int   : 1-D array of abuse type indices (0..K-1)
    target_words : 교량 후보 단어 목록
    p_configs    : 교량 판정 파라미터 설정 목록. None이면 BRIDGE_P_CONFIGS 사용.
    n_boot       : 부트스트랩 반복 수
    count_min    : 최소 문서 수
    seed         : 난수 시드

    Returns
    -------
    DataFrame: word, config, n_boot, sel_count, sel_prob (n_boot 중 선정된 비율)
    """
    if p_configs is None:
        p_configs = BRIDGE_P_CONFIGS

    N_doc = len(labels_int)
    rng = np.random.default_rng(seed)
    target_set = set(target_words)

    # word → doc_indices 역색인
    word_to_docs: Dict[str, List[int]] = {}
    for idx, words in doc_to_words.items():
        for w in words:
            if w in target_set:
                word_to_docs.setdefault(w, []).append(idx)

    # 카운터 초기화: {config_name: {word: count}}
    sel_counts = {cfg["name"]: {} for cfg in p_configs}

    for b in range(n_boot):
        sample_indices = rng.integers(0, N_doc, size=N_doc)

        # 단어별 카운트
        word_counts: Dict[str, np.ndarray] = {}
        for idx in sample_indices:
            abuse_idx = labels_int[idx]
            for w in doc_to_words.get(idx, set()):
                if w not in target_set:
                    continue
                if w not in word_counts:
                    word_counts[w] = np.zeros(K, dtype=int)
                word_counts[w][abuse_idx] += 1

        # 각 config에서 bridge 여부 판정
        for cfg in p_configs:
            cfg_name = cfg["name"]
            min_p1 = cfg["min_p1"]
            min_p2 = cfg["min_p2"]
            max_gap = cfg["max_gap"]

            for w, counts in word_counts.items():
                total = counts.sum()
                if total < count_min:
                    continue
                probs = counts / total
                idx_sorted = np.argsort(-probs)
                k1_idx = idx_sorted[0]
                k2_idx = idx_sorted[1] if len(idx_sorted) > 1 else None
                p1 = probs[k1_idx]
                p2 = probs[k2_idx] if k2_idx is not None else 0.0

                # FIX-4: k1, k2 각각의 문서 수가 count_min 이상인지 검증
                # stats.py compute_prob_bridge_for_words() 라인 279-283과 일관성 유지
                if k2_idx is None:
                    continue
                if counts[k1_idx] < count_min or counts[k2_idx] < count_min:
                    continue

                if p1 >= min_p1 and p2 >= min_p2 and (p1 - p2) <= max_gap:
                    sel_counts[cfg_name][w] = (
                        sel_counts[cfg_name].get(w, 0) + 1
                    )

        if (b + 1) % 50 == 0:
            print(f"[BOOT] {b+1}/{n_boot} 완료")

    # 결과 테이블
    rows = []
    for cfg in p_configs:
        cfg_name = cfg["name"]
        # target_words 전체에 대해 기록 (선정 안 된 단어도 0으로)
        for w in target_words:
            c = sel_counts[cfg_name].get(w, 0)
            rows.append({
                "word": w,
                "config": cfg_name,
                "n_boot": n_boot,
                "sel_count": c,
                "sel_prob": c / n_boot,
            })

    df_boot = pd.DataFrame(rows)
    print(f"[BOOT] 완료: {len(target_words)} 단어 × {len(p_configs)} configs "
          f"× {n_boot} 부트스트랩")
    return df_boot


def compute_multiconfig_stability(
    df_boot: pd.DataFrame,
    stability_threshold: float = BOOT_STABILITY_THRESHOLD,
    majority_ratio: float = MAJORITY_RATIO,
) -> pd.DataFrame:
    """
    다중 설정 감도 분석: 각 단어가 몇 개 설정에서 안정적 교량인지 계산.

    Parameters
    ----------
    df_boot              : run_bootstrap_with_stats() 결과
    stability_threshold  : 이 이상이면 해당 설정에서 "안정적"
    majority_ratio       : 전체 설정 중 이 비율 이상에서 안정적이면 "다수결 통과"

    Returns
    -------
    DataFrame: word, n_configs_stable, ratio_configs_stable, is_majority_stable,
               per-config sel_prob columns
    """
    # --- FIX-3: 빈 DataFrame 가드 ---
    if df_boot.empty:
        import warnings
        warnings.warn("[STABILITY] df_boot가 비어있습니다.")
        return pd.DataFrame()

    configs = df_boot["config"].unique().tolist()
    n_cfg = len(configs)
    majority_count = max(2, int(np.ceil(n_cfg * majority_ratio)))

    pivot = df_boot.pivot(
        index="word", columns="config", values="sel_prob"
    ).fillna(0)

    records = []
    for w in pivot.index:
        row = {"word": w}
        n_stable = 0
        for cfg in configs:
            sp = pivot.loc[w, cfg] if cfg in pivot.columns else 0.0
            row[f"sel_prob_{cfg}"] = sp
            if sp >= stability_threshold:
                n_stable += 1

        row["n_configs_stable"] = n_stable
        row["ratio_configs_stable"] = n_stable / n_cfg if n_cfg > 0 else 0
        row["is_majority_stable"] = n_stable >= majority_count
        records.append(row)

    df_stability = pd.DataFrame(records)
    n_majority = df_stability["is_majority_stable"].sum()
    print(f"[STABILITY] 다수결 안정 단어: {n_majority}/{len(df_stability)} "
          f"(≥{majority_count}/{n_cfg} configs, "
          f"sel_prob≥{stability_threshold})")
    return df_stability


# ═══════════════════════════════════════════════════════════════
#  통합: 3중 검증으로 확정 교량 단어 도출
# ═══════════════════════════════════════════════════════════════

def combine_three_criteria(
    df_pvalues: pd.DataFrame,
    df_stability: pd.DataFrame,
    df_observed: pd.DataFrame,
    baseline_config: str = "B0_baseline",
) -> pd.DataFrame:
    """
    3중 검증:
      (1) 순열 귀무 유의성: is_perm_significant = True
      (2) 부트스트랩 안정성: baseline config에서 sel_prob ≥ 0.80
      (3) 다중 설정 다수결: is_majority_stable = True

    Parameters
    ----------
    df_pvalues   : compute_null_summaries_and_pvalues() 결과 [2]
    df_stability : compute_multiconfig_stability() 결과
    df_observed  : 관찰값 (p1, p2, gap, g1, g2)
    baseline_config : 기준 설정 이름

    Returns
    -------
    DataFrame with all words and 3중 검증 결과 컬럼
    """
    sel_prob_col = f"sel_prob_{baseline_config}"
    boot_cols = ["word", "is_majority_stable", "n_configs_stable"]
    if sel_prob_col in df_stability.columns:
        boot_cols.append(sel_prob_col)

    df = df_observed[["word", "p1", "p2", "gap", "g1", "g2"]].rename(
        columns={
            "p1": "p1_obs", "p2": "p2_obs", "gap": "gap_obs",
            "g1": "g1_obs", "g2": "g2_obs",
        }
    ).copy()

    # 순열 p-value merge
    pval_cols = [
        "word", "p1_empirical_pval", "p2_empirical_pval",
        "gap_empirical_pval", "bridge_score_obs", "bridge_score_pval",
        "is_perm_significant_triple", "is_perm_significant",
    ]
    pval_cols_exist = [c for c in pval_cols if c in df_pvalues.columns]
    df = df.merge(df_pvalues[pval_cols_exist], on="word", how="left")

    # 부트스트랩 merge
    stability_cols_exist = [
        c for c in boot_cols if c in df_stability.columns
    ]
    df = df.merge(df_stability[stability_cols_exist], on="word", how="left")

    # baseline 설정에서의 선정 확률
    if sel_prob_col in df.columns:
        df["boot_sel_prob_baseline"] = df[sel_prob_col]
        df["is_boot_stable"] = (
            df["boot_sel_prob_baseline"] >= BOOT_STABILITY_THRESHOLD
        )
    else:
        df["boot_sel_prob_baseline"] = np.nan
        df["is_boot_stable"] = False

    # NaN 처리
    df["is_perm_significant"] = (
        df["is_perm_significant"].fillna(False).astype(bool)
    )
    df["is_boot_stable"] = df["is_boot_stable"].fillna(False).astype(bool)
    df["is_majority_stable"] = (
        df["is_majority_stable"].fillna(False).astype(bool)
    )

    # 충족 기준 수
    df["n_criteria_met"] = (
        df["is_perm_significant"].astype(int)
        + df["is_boot_stable"].astype(int)
        + df["is_majority_stable"].astype(int)
    )
    df["is_confirmed_bridge"] = df["n_criteria_met"] == 3

    n_confirmed = df["is_confirmed_bridge"].sum()
    n_total = len(df)
    print(f"[CONFIRMED] 3중 검증 통과 확정 교량 단어: {n_confirmed}/{n_total}")
    for nc in [3, 2, 1, 0]:
        cnt = (df["n_criteria_met"] == nc).sum()
        print(f"  {nc}/3 기준 충족: {cnt}개")

    return df.sort_values(
        ["is_confirmed_bridge", "n_criteria_met", "gap_obs"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  보고서 생성
# ═══════════════════════════════════════════════════════════════

def generate_threshold_report(
    chance_info: dict,
    df_confirmed: pd.DataFrame,
    df_null_summary: pd.DataFrame,
    df_boot: pd.DataFrame,
    n_perm: int,
    n_boot: int,
    out_dir: str,
) -> str:
    """논문용 보고서 텍스트 생성 및 저장."""

    lines = []
    lines.append("=" * 72)
    lines.append("  교량 단어(Bridge Word) 임계값 객관적 근거 보고서")
    lines.append("  Bridge Word Threshold Justification Report")
    lines.append("=" * 72)

    # --- 전략 1 ---
    lines.append("\n[전략 1] 우연 수준(Chance Level) 기반 해석적 근거")
    lines.append("-" * 60)
    for interp in chance_info["interpretations"]:
        lines.append(f"  {interp}")

    # --- 전략 2 ---
    lines.append(
        f"\n[전략 2] 순열 귀무분포 기반 유의성 검정 (B = {n_perm})"
    )
    lines.append("-" * 60)

    n_tested = len(df_confirmed)
    n_perm_sig = df_confirmed["is_perm_significant"].sum()
    n_perm_sig_triple = (
        df_confirmed["is_perm_significant_triple"].sum()
        if "is_perm_significant_triple" in df_confirmed.columns else 0
    )
    lines.append(f"  검정 대상 단어 수: {n_tested}")
    lines.append(
        f"  3조건 동시충족 (p1 AND p2 AND gap, α = {NULL_ALPHA}): "
        f"{n_perm_sig_triple}  ← 구조적 불가능성 확인"
    )
    lines.append(
        f"  복합 통계량 bridge_score 기준 (α = {NULL_ALPHA}): {n_perm_sig}"
    )

    if not df_null_summary.empty:
        lines.append(f"\n  귀무분포 요약 (전체 단어 평균):")
        for col in ["p1_null_mean", "p2_null_mean", "gap_null_mean",
                     "p1_null_std", "p2_null_std", "gap_null_std"]:
            if col in df_null_summary.columns:
                val = df_null_summary[col].mean()
                lines.append(f"    {col}: {val:.4f}")

    # --- 전략 3 ---
    lines.append(
        f"\n[전략 3] 부트스트랩 안정성 + 다중 설정 감도 분석 (B = {n_boot})"
    )
    lines.append("-" * 60)

    n_boot_stable = df_confirmed["is_boot_stable"].sum()
    n_majority = df_confirmed["is_majority_stable"].sum()
    lines.append(
        f"  부트스트랩 안정 (sel_prob ≥ {BOOT_STABILITY_THRESHOLD}): "
        f"{n_boot_stable}"
    )
    lines.append(f"  다수결 안정 (≥50% configs): {n_majority}")

    # --- 통합 결과 ---
    lines.append(f"\n[통합] 3중 검증 결과")
    lines.append("=" * 60)
    for nc in [3, 2, 1, 0]:
        cnt = (df_confirmed["n_criteria_met"] == nc).sum()
        label = "★ 확정 교량" if nc == 3 else f"  {nc}/3 충족"
        lines.append(f"  {label}: {cnt}개")

    confirmed = df_confirmed[df_confirmed["is_confirmed_bridge"]].copy()
    if not confirmed.empty:
        lines.append(f"\n  확정 교량 단어 목록 ({len(confirmed)}개):")
        for _, row in confirmed.iterrows():
            g1_en = ABUSE_LABEL_EN.get(row["g1_obs"], row["g1_obs"])
            g2_en = ABUSE_LABEL_EN.get(row["g2_obs"], row["g2_obs"])
            lines.append(
                f"    {row['word']:12s}  "
                f"p1={row['p1_obs']:.3f} ({g1_en}) / "
                f"p2={row['p2_obs']:.3f} ({g2_en}) / "
                f"gap={row['gap_obs']:.3f}  "
                f"boot={row['boot_sel_prob_baseline']:.2f}"
            )

    # --- Jaccard (B0 기준) ---
    if not confirmed.empty:
        lines.append(f"\n  [참고] 설정 간 Jaccard 유사도:")
        configs = [
            c for c in df_confirmed.columns
            if c.startswith("sel_prob_B")
        ]
        for cfg_col in configs:
            if cfg_col == "sel_prob_B0_baseline":
                continue
            cfg_name = cfg_col.replace("sel_prob_", "")
            set_base = set(
                df_confirmed[
                    df_confirmed.get(
                        "sel_prob_B0_baseline",
                        pd.Series(dtype=float),
                    ) >= BOOT_STABILITY_THRESHOLD
                ]["word"]
            )
            set_comp = set(
                df_confirmed[
                    df_confirmed.get(
                        cfg_col, pd.Series(dtype=float)
                    ) >= BOOT_STABILITY_THRESHOLD
                ]["word"]
            )
            if set_base or set_comp:
                jacc = (
                    len(set_base & set_comp) / len(set_base | set_comp)
                    if (set_base | set_comp) else 1.0
                )
                lines.append(
                    f"    B0_baseline vs {cfg_name}: "
                    f"Jaccard = {jacc:.3f} "
                    f"(교집합 {len(set_base & set_comp)}, "
                    f"합집합 {len(set_base | set_comp)})"
                )

    # --- 논문용 요약 문장 ---
    lines.append(f"\n[논문용 요약 문장]")
    lines.append("-" * 60)
    lines.append(
        f"교량 단어의 최소 확률 임계값(p_min,1 = {chance_info['min_p1']:.2f}, "
        f"p_min,2 = {chance_info['min_p2']:.2f})은 "
        f"K = {K} 유형의 균등 확률(1/K = {P_CHANCE:.2f})을 기준점으로 설정하였다. "
        f"순열 검정(B = {n_perm})에서 귀무분포 상위 "
        f"{int(NULL_ALPHA * 100)}%를 초과하는 "
        f"단어만을 후보로 선별하였으며, "
        f"부트스트랩 리샘플링(B = {n_boot})에서 선정 확률이 "
        f"{BOOT_STABILITY_THRESHOLD:.2f} 이상인 단어를 "
        f"안정적 교량 단어로 확정하였다. "
        f"{len(BRIDGE_P_CONFIGS)}개 파라미터 설정에 걸친 감도 분석에서 "
        f"3중 검증을 모두 통과한 확정 교량 단어는 {len(confirmed)}개이다."
    )

    report_text = "\n".join(lines)

    # 저장
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "bridge_threshold_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[저장] 임계값 근거 보고서 -> {report_path}")

    return report_text


# ═══════════════════════════════════════════════════════════════
#  헬퍼: doc_word_df + doc_meta → doc_to_words + labels_int 변환
# ═══════════════════════════════════════════════════════════════

def prepare_doc_structures(
    doc_word_df: pd.DataFrame,
    doc_meta: pd.DataFrame,
) -> Tuple[Dict[int, set], np.ndarray, List[str]]:
    """
    doc_word_df / doc_meta를 순열·부트스트랩 함수가 사용하는
    내부 자료구조로 변환한다.

    Parameters
    ----------
    doc_word_df : columns [doc_id, word, ...]
    doc_meta    : columns [doc_id, main_abuse, ...]

    Returns
    -------
    doc_to_words : {doc_idx: set of words}
    labels_int   : np.ndarray of shape (N_doc,), abuse type index (0..K-1)
    doc_ids      : list of doc_id (순서 = doc_idx)
    """
    meta = doc_meta[doc_meta["main_abuse"].isin(ABUSE_ORDER)].copy()
    meta = meta.sort_values("doc_id").reset_index(drop=True)

    doc_ids = meta["doc_id"].tolist()
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    abuse2idx = {a: i for i, a in enumerate(ABUSE_ORDER)}

    labels_int = np.array([abuse2idx[ab] for ab in meta["main_abuse"]])
    N_doc = len(doc_ids)

    # doc_word_df에서 유효 문서만
    dw = doc_word_df[doc_word_df["doc_id"].isin(doc_ids)][["doc_id", "word"]]
    dw = dw.drop_duplicates()

    doc_to_words: Dict[int, set] = {i: set() for i in range(N_doc)}
    for doc_id, word in dw.itertuples(index=False):
        idx = doc_index.get(doc_id)
        if idx is not None:
            doc_to_words[idx].add(word)

    return doc_to_words, labels_int, doc_ids


def get_chi_top_words(
    chi_df: pd.DataFrame,
    top_k: int = 200,
) -> List[str]:
    """chi² 통계량 상위 top_k 단어를 반환한다."""
    if chi_df is None or chi_df.empty:
        return []
    sorted_chi = chi_df.sort_values("chi2", ascending=False)
    return sorted_chi.head(top_k).index.tolist()


# ═══════════════════════════════════════════════════════════════
#  메인 실행 함수
# ═══════════════════════════════════════════════════════════════

def run_bridge_threshold_justification(
    df_abuse_counts_doc: pd.DataFrame,
    doc_to_words: Dict[int, set],
    labels_int: np.ndarray,
    target_words: List[str],
    p_configs: list = None,
    n_perm: int = DEFAULT_N_PERM,
    n_boot: int = DEFAULT_N_BOOT,
    count_min: int = BRIDGE_MIN_COUNT,
    out_dir: str = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    교량 단어 임계값 객관적 근거 확보 — 통합 실행 함수.

    Parameters
    ----------
    df_abuse_counts_doc : 문서 수 기반 빈도표 (word × 학대유형)
    doc_to_words        : {doc_idx: set of words}
    labels_int          : 문서별 학대유형 인덱스 배열
    target_words        : 검증 대상 단어 (chi² top 200 등)
    p_configs           : 교량 판정 파라미터 설정 목록. None이면 BRIDGE_P_CONFIGS 사용.
    n_perm              : 순열 반복 수
    n_boot              : 부트스트랩 반복 수
    count_min           : 최소 문서 수
    out_dir             : 결과 저장 디렉토리
    seed                : 난수 시드

    Returns
    -------
    dict: {
        "chance_info": dict,
        "df_observed": DataFrame,
        "df_null": DataFrame,
        "df_null_summary": DataFrame,
        "df_pvalues": DataFrame,
        "df_boot": DataFrame,
        "df_stability": DataFrame,
        "df_confirmed": DataFrame,
        "report": str,
    }
    """
    if p_configs is None:
        p_configs = BRIDGE_P_CONFIGS

    if out_dir is None:
        # C.BRIDGE_PROB_ABLATION_DIR은 configure_output_dirs() 호출 후 설정됨
        _base = C.BRIDGE_PROB_ABLATION_DIR or "output/pbridge_ablation"
        out_dir = os.path.join(_base, "threshold_justification")
    os.makedirs(out_dir, exist_ok=True)

    print("╔" + "═" * 68 + "╗")
    print("║  교량 단어 임계값 객관적 근거 확보                                ║")
    print("║  Bridge Word Threshold Justification                             ║")
    print("╚" + "═" * 68 + "╝")

    # ── [전략 1] 우연 수준 ──
    print("\n▶ [전략 1] 우연 수준 기반 해석적 근거")
    chance_info = chance_level_justification()
    for interp in chance_info["interpretations"]:
        print(f"  {interp}")

    # ── 관찰값 계산 ──
    # FIX-5: 데이터 소스 불일치 방지
    # build_doc_level_abuse_counts()는 allowed_groups=None일 때 VALENCE_ORDER 필터가 없지만,
    # build_abuse_doc_word_table()은 항상 VALENCE_ORDER를 필터링한다.
    # → df_abuse_counts_doc에는 있지만 doc_to_words에는 없는 단어가 존재할 수 있음.
    # → 두 소스의 교집합만 사용해야 순열/부트스트랩에서 빈 결과를 방지할 수 있다.
    vocab_in_docs = set()
    for ws in doc_to_words.values():
        vocab_in_docs.update(ws)

    valid_words = [
        w for w in target_words
        if w in df_abuse_counts_doc.index and w in vocab_in_docs
    ]
    n_only_counts = sum(1 for w in target_words
                        if w in df_abuse_counts_doc.index and w not in vocab_in_docs)
    if n_only_counts > 0:
        print(f"  ⚠ {n_only_counts}개 단어가 빈도표에는 있지만 "
              f"doc_to_words에 없어 제외됨 (데이터 소스 불일치)")
    print(
        f"\n▶ 검증 대상 단어: {len(valid_words)}/{len(target_words)} "
        f"(빈도표 ∩ doc_to_words에 존재)"
    )

    df_observed = _compute_word_stats_from_counts(
        df_abuse_counts_doc.loc[
            df_abuse_counts_doc.index.intersection(valid_words)
        ]
    )
    print(f"  관찰값 계산 완료: {len(df_observed)} 단어")

    # ── [전략 2] 순열 귀무분포 ──
    print(f"\n▶ [전략 2] 순열 귀무분포 생성 (B = {n_perm})")
    df_null = run_permutation_null_distributions(
        doc_to_words=doc_to_words,
        labels_int=labels_int,
        target_words=valid_words,
        n_perm=n_perm,
        count_min=count_min,
        seed=seed,
    )

    df_null_summary, df_pvalues = compute_null_summaries_and_pvalues(
        df_null=df_null,
        df_observed=df_observed,
        n_perm=n_perm,
    )

    # 저장
    df_null_summary.to_csv(
        os.path.join(out_dir, "bridge_null_distributions.csv"),
        encoding="utf-8-sig", index=False,
    )
    df_pvalues.to_csv(
        os.path.join(out_dir, "bridge_empirical_pvalues.csv"),
        encoding="utf-8-sig", index=False,
    )

    # ── [전략 3] 부트스트랩 + 다중 설정 ──
    print(f"\n▶ [전략 3] 부트스트랩 안정성 (B = {n_boot})")
    df_boot = run_bootstrap_with_stats(
        doc_to_words=doc_to_words,
        labels_int=labels_int,
        target_words=valid_words,
        p_configs=p_configs,
        n_boot=n_boot,
        count_min=count_min,
        seed=seed + 1,  # 순열과 다른 시드
    )

    df_stability = compute_multiconfig_stability(df_boot)

    df_boot.to_csv(
        os.path.join(out_dir, "bridge_bootstrap_stability.csv"),
        encoding="utf-8-sig", index=False,
    )
    df_stability.to_csv(
        os.path.join(out_dir, "bridge_multiconfig_stability.csv"),
        encoding="utf-8-sig", index=False,
    )

    # ── 통합: 3중 검증 ──
    print(f"\n▶ 3중 검증 통합")
    df_confirmed = combine_three_criteria(
        df_pvalues=df_pvalues,
        df_stability=df_stability,
        df_observed=df_observed,
    )

    df_confirmed.to_csv(
        os.path.join(out_dir, "bridge_confirmed_words.csv"),
        encoding="utf-8-sig", index=False,
    )

    # ── 보고서 ──
    report = generate_threshold_report(
        chance_info=chance_info,
        df_confirmed=df_confirmed,
        df_null_summary=df_null_summary,
        df_boot=df_boot,
        n_perm=n_perm,
        n_boot=n_boot,
        out_dir=out_dir,
    )

    print("\n" + report)

    return {
        "chance_info": chance_info,
        "df_observed": df_observed,
        "df_null": df_null,
        "df_null_summary": df_null_summary,
        "df_pvalues": df_pvalues,
        "df_boot": df_boot,
        "df_stability": df_stability,
        "df_confirmed": df_confirmed,
        "report": report,
    }


# ═══════════════════════════════════════════════════════════════
#  단독 실행용 엔트리포인트
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    PyCharm 또는 터미널에서 단독 실행 시,
    pipeline.py 라인 770–797과 동일한 경로로
    교량 단어 임계값 근거 확보를 수행한다.

    프로젝트 구조:
        Childeren/                          ← ROOT_DIR
        ├── data/                           ← JSON 파일들
        └── v28_refactor/                   ← 패키지 부모
            └── abuse_pipeline/             ← 파이썬 패키지
                ├── __init__.py
                ├── bridge_threshold_justification.py  ← 이 파일
                ├── common.py
                ├── doc_level.py            ← from . import common (상대 import)
                └── stats.py               ← from .common import * (상대 import)

    실행 방법:
        PyCharm에서 이 파일을 직접 Run (우클릭 → Run)
    """
    import sys
    import glob
    from pathlib import Path

    # ──────────────────────────────────────────────────────
    # 패키지 경로 설정
    # ──────────────────────────────────────────────────────
    # doc_level.py, stats.py 는 상대 import(from . import ...)만 사용한다.
    # 단독 실행 시 이들을 import하려면, 패키지 부모(v28_refactor/)가
    # sys.path에 있어야 Python이 abuse_pipeline을 패키지로 인식한다.
    #
    #   이 파일 위치: .../v28_refactor/abuse_pipeline/bridge_threshold_justification.py
    #   패키지 부모:  .../v28_refactor/
    #
    _this_file = Path(__file__).resolve()
    _pkg_parent = str(_this_file.parent.parent)   # v28_refactor/
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)

    from abuse_pipeline.doc_level import (
        build_abuse_doc_word_table,
        build_doc_level_abuse_counts,
    )
    from abuse_pipeline.stats import compute_chi_square, add_bh_fdr

    # C 와 ABUSE_ORDER 등은 이미 파일 상단 try/except에서
    # standalone fallback으로 import 완료된 상태:
    #   import common as C
    #   from common import ABUSE_ORDER, BRIDGE_P_CONFIGS, ...

    # ──────────────────────────────────────────────────────
    # 출력 디렉토리 설정
    # ──────────────────────────────────────────────────────
    # common.py의 _find_project_root()가 Childeren/ (data/ 폴더 존재)을 자동 탐지
    C.configure_output_dirs(
        subset_name="NEG",
        base_dir=C.BASE_DIR,
        version_tag="ver28",
    )

    print("=" * 72)
    print("  bridge_threshold_justification.py — 단독 실행 모드")
    print("=" * 72)
    print(f"  ROOT_DIR       = {C.BASE_DIR}")
    print(f"  DATA_JSON_DIR  = {C.DATA_JSON_DIR}")
    print(f"  OUTPUT_DIR     = {C.OUTPUT_DIR}")
    print()

    # ──────────────────────────────────────────────────────
    # [1] JSON 파일 로드
    # ──────────────────────────────────────────────────────
    json_files = sorted(glob.glob(os.path.join(C.DATA_JSON_DIR, "*.json")))
    if not json_files:
        print(f"[ERROR] JSON 파일을 찾을 수 없습니다: {C.DATA_JSON_DIR}")
        sys.exit(1)
    print(f"[1/5] JSON 파일 {len(json_files)}개 로드 완료")

    # ──────────────────────────────────────────────────────
    # [2] doc-level abuse × 단어 테이블 생성
    # ──────────────────────────────────────────────────────
    allowed_groups = {"부정"}   # NEG subset (pipeline.py와 동일)

    print(f"[2/5] build_abuse_doc_word_table (allowed_groups={allowed_groups})")
    doc_word_df, doc_meta = build_abuse_doc_word_table(
        json_files=json_files,
        allowed_groups=allowed_groups,
    )
    print(f"       doc_word_df: {doc_word_df.shape}  |  doc_meta: {doc_meta.shape}")

    # ──────────────────────────────────────────────────────
    # [3] doc-level abuse 빈도표 생성
    # ──────────────────────────────────────────────────────
    print(f"[3/5] build_doc_level_abuse_counts")
    df_abuse_counts_doc = build_doc_level_abuse_counts(
        json_files=json_files,
        allowed_groups=allowed_groups,
    )
    print(f"       df_abuse_counts_doc: {df_abuse_counts_doc.shape}")

    # ──────────────────────────────────────────────────────
    # [4] χ² 통계량 + BH-FDR 보정
    # ──────────────────────────────────────────────────────
    print(f"[4/5] compute_chi_square + BH-FDR")
    abuse_chi_doc = compute_chi_square(df_abuse_counts_doc, ABUSE_ORDER)
    abuse_chi_doc = add_bh_fdr(abuse_chi_doc, p_col="p_value", out_col="p_fdr_bh")
    print(f"       abuse_chi_doc: {abuse_chi_doc.shape}")

    # ──────────────────────────────────────────────────────
    # [5] 자료구조 변환 → 임계값 근거 확보 실행
    # ──────────────────────────────────────────────────────
    print(f"[5/5] 자료구조 변환 → run_bridge_threshold_justification")
    _doc_to_words, _labels_int, _doc_ids = prepare_doc_structures(
        doc_word_df=doc_word_df,
        doc_meta=doc_meta,
    )
    print(f"       문서 수: {len(_doc_ids)}")
    print(f"       라벨 분포: {dict(zip(ABUSE_ORDER, np.bincount(_labels_int)))}")

    _target_words = get_chi_top_words(abuse_chi_doc, top_k=200)
    print(f"       χ² top-200 후보 단어: {len(_target_words)}개")

    # ── 통합 실행 ──
    out_dir = os.path.join(
        C.BRIDGE_PROB_ABLATION_DIR, "threshold_justification"
    )

    result = run_bridge_threshold_justification(
        df_abuse_counts_doc=df_abuse_counts_doc,
        doc_to_words=_doc_to_words,
        labels_int=_labels_int,
        target_words=_target_words,
        p_configs=C.BRIDGE_P_CONFIGS,
        n_perm=1000,    # pipeline.py와 동일
        n_boot=500,     # pipeline.py와 동일
        count_min=5,
        out_dir=out_dir,
        seed=42,
    )

    # ──────────────────────────────────────────────────────
    # 완료
    # ──────────────────────────────────────────────────────
    n_confirmed = result["df_confirmed"]["is_confirmed_bridge"].sum()
    print("\n" + "=" * 72)
    print(f"  [완료] 결과 저장: {out_dir}")
    print(f"  확정 교량 단어: {n_confirmed}개")
    print("=" * 72)