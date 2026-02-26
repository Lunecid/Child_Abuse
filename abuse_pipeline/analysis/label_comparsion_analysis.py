"""
label_comparison_analysis.py
=============================
정답(Ground Truth) 단일 라벨 vs. 알고리즘 주+부 학대유형 라벨의 체계적 비교 분석 모듈.

본 모듈은 다음 분석을 수행한다:
  Phase 1: 기존 라벨 비교 (Main Hit Rate, Jaccard, Confusion Matrix)
  Phase 2: 중복학대(Poly-victimization) 분석
  Phase 3: 동시발생 행렬 및 전이 패턴
  Phase 4: 정보 손실 정량화 (엔트로피)
  Phase 5: Hidden Companion 조건부 확률
  Phase 6: 가중 빈도표 구축 (CA 확장용)
  Phase 7: 시각화

────────────────────────────────────────────────────────────────
⚠️ 독립 실행 불가 — 반드시 파이프라인 내에서 실행해야 합니다.
────────────────────────────────────────────────────────────────
알고리즘이 부여하는 주/부 학대유형 라벨(classify_abuse_main_sub)은
JSON 파일에 사전 기록되어 있지 않고, 파이프라인 실행 시점에 동적으로
생성됩니다. 따라서 본 모듈은 반드시 패키지 내부 모듈로서 파이프라인
안에서 호출되어야 하며, 독립 실행(standalone)은 지원하지 않습니다.

사용법 (파이프라인 통합):
    from .label_comparison_analysis import run_label_comparison_from_pipeline
    results = run_label_comparison_from_pipeline(
        json_files=json_files,
        out_dir=os.path.join(C.OUTPUT_DIR, "label_comparison"),
        only_negative=only_negative,
    )

코드 구조 참조:
    - labels.py: classify_abuse_main_sub() → 주+부 학대유형 동적 할당
    - compare_abuse_labels.py: extract_gt_abuse_types_from_info() → 정답 라벨 추출
    - text.py: extract_child_speech(), tokenize_korean() → 텍스트 추출/토큰화
    - doc_level.py: build_abuse_doc_word_table() → 문서 단위 빈도표
"""

from __future__ import annotations

import os
import json
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── matplotlib 백엔드 설정 ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  0. 프로젝트 모듈 Import — 패키지 내부 상대 import
# ═══════════════════════════════════════════════════════════════
# 이 파일은 abuse_pipeline/ 패키지 내부에 위치하므로
# 반드시 상대 import(relative import)를 사용합니다.
from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean
from abuse_pipeline.analysis.compare_abuse_labels import (
    extract_gt_abuse_types_from_info,
    normalize_abuse_label,
    DEFAULT_ABUSE_ORDER,
)

# ── 상수 정의 (common.py와 일관성 유지) ──
ABUSE_ORDER = C.ABUSE_ORDER  # ["성학대", "신체학대", "정서학대", "방임"]
ABUSE_COLORS = {
    "방임": "#1f77b4",
    "정서학대": "#ff7f0e",
    "신체학대": "#2ca02c",
    "성학대": "#d62728",
}
ABUSE_LABEL_EN = C.ABUSE_LABEL_EN  # common.py에서 이미 정의된 영문 라벨 사용

_SEVERITY_RANK = C.SEVERITY_RANK


# ═══════════════════════════════════════════════════════════════
#  1. 데이터 로딩 및 전처리 — 한 아동의 모든 라벨 정보 추출
# ═══════════════════════════════════════════════════════════════

def _extract_all_labels_from_record(
    rec: Dict[str, Any],
    abuse_order: List[str] = None,
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
) -> Dict[str, Any]:
    """
    한 아동의 JSON record에서 모든 라벨 정보를 추출한다.

    이 함수가 파이프라인 내부에서만 동작하는 핵심 이유:
    ─────────────────────────────────────────────────────────────
    · 정답 라벨(Ground Truth): JSON의 info["학대의심"] 필드에 기록되어
      있으므로 파일만 읽으면 추출 가능. 이 라벨은 데이터에 이미 존재하는
      학대유형 분류 결과로, 본 분석에서 정답(Ground Truth)으로 간주한다.
    · 알고리즘 라벨: classify_abuse_main_sub(rec)를 호출해야만
      주/부 학대유형이 동적으로 계산됨. JSON에는 저장되지 않음.
    ─────────────────────────────────────────────────────────────

    Returns
    -------
    dict with keys:
        "doc_id": 문서 ID
        "gt_label": 정답(Ground Truth) 단일 라벨 (str or None)
        "gt_label_set": 정답 라벨 집합 ({str})
        "algo_main": 알고리즘 주 학대유형 (str or None)
        "algo_subs": 알고리즘 부 학대유형 리스트 ([str])
        "algo_set": 알고리즘 전체 라벨 집합 ({str})
        "abuse_scores": {학대유형: 점수} 딕셔너리
        "valence_group": 정서군 (str or None)
        "speech_tokens": 토큰 리스트 ([str])
        "speech_raw": 원본 발화 리스트 ([str])
    """
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    info = rec.get("info", {}) or {}
    doc_id = info.get("ID") or info.get("id") or info.get("Id")

    # ── 정답(Ground Truth) 라벨 추출 ──
    # JSON에 기록된 info["학대의심"] 필드에서 추출
    # 이 라벨은 데이터에 이미 존재하는 학대유형으로, 정답으로 간주한다.
    gt_label_set = extract_gt_abuse_types_from_info(info, field="학대의심")

    # 정답 라벨이 여러 개 추출될 수 있으나, 원본은 단일 라벨 → 하나만 대표로
    gt_label = (
        sorted(gt_label_set, key=lambda x: _SEVERITY_RANK.get(x, 999))[0]
        if gt_label_set else None
    )

    # ── 알고리즘 라벨 추출 (파이프라인 실행 시 동적 생성) ──
    # ⚠️ 이 호출이 바로 독립 실행이 불가능한 핵심 이유:
    #    classify_abuse_main_sub()는 rec 전체를 읽고
    #    문항 점수 + 임상 텍스트를 조합하여 라벨을 실시간 결정한다.
    algo_main, algo_subs = classify_abuse_main_sub(
        rec,
        abuse_order=abuse_order,
        sub_threshold=sub_threshold,
        use_clinical_text=use_clinical_text,
    )
    algo_subs = algo_subs or []
    algo_set = set()
    if algo_main:
        algo_set.add(algo_main)
    algo_set |= set(algo_subs)

    # ── 학대여부 문항별 점수 집계 ──
    abuse_scores = {a: 0 for a in abuse_order}
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
                for a in abuse_order:
                    if a in name:
                        abuse_scores[a] += sc

    # ── 정서군 ──
    valence_group = classify_child_group(rec)

    # ── 발화 토큰 ──
    speech_raw = extract_child_speech(rec)
    speech_tokens = []
    if speech_raw:
        joined = " ".join(speech_raw)
        speech_tokens = tokenize_korean(joined)

    return {
        "doc_id": doc_id,
        "gt_label": gt_label,
        "gt_label_set": gt_label_set,
        "algo_main": algo_main,
        "algo_subs": algo_subs,
        "algo_set": algo_set,
        "abuse_scores": abuse_scores,
        "valence_group": valence_group,
        "speech_tokens": speech_tokens,
        "speech_raw": speech_raw,
    }


def _load_all_records(
    json_files: List[str],
    abuse_order: List[str] = None,
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
    only_negative: bool = False,
) -> List[Dict[str, Any]]:
    """
    모든 JSON 파일에서 아동별 라벨 정보를 추출한다.

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    only_negative : bool
        True이면 정서군이 '부정'인 아동만 포함

    Returns
    -------
    list[dict]: 아동별 라벨 정보 딕셔너리 리스트
    """
    records = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        result = _extract_all_labels_from_record(
            rec, abuse_order=abuse_order,
            sub_threshold=sub_threshold,
            use_clinical_text=use_clinical_text,
        )
        result["source_file"] = str(path)

        # 정서군 필터
        if only_negative and result["valence_group"] != "부정":
            continue

        records.append(result)

    print(f"[LABEL-COMP][LOAD] 총 {len(records)}명의 아동 레코드 로드 완료")
    return records


# ═══════════════════════════════════════════════════════════════
#  Phase 1: 기본 라벨 비교 지표
# ═══════════════════════════════════════════════════════════════

def compute_basic_comparison_metrics(records: List[Dict]) -> pd.DataFrame:
    """
    각 아동별 정답(GT) 라벨 vs. 알고리즘 라벨 비교 지표를 계산한다.

    계산되는 지표:
    ─────────────────────────────────────────────────────────────
    main_hit:
        정답 라벨(단일)이 알고리즘 주 학대유형과 일치하면 1, 아니면 0.

    jaccard:
        정답 라벨 집합과 알고리즘 라벨 집합의 Jaccard 유사도.
        수식: J_i = |GT ∩ Algo| / |GT ∪ Algo|

    exact_set_match:
        두 집합이 완전히 동일하면 1, 아니면 0.

    n_algo_labels:
        알고리즘이 부여한 총 라벨 수 (주 + 부).

    extra_in_algo:
        알고리즘에는 있지만 정답에는 없는 학대유형들.

    missing_in_algo:
        정답에는 있지만 알고리즘에는 없는 학대유형들.
    ─────────────────────────────────────────────────────────────
    """
    rows = []
    for r in records:
        gt_set = r["gt_label_set"]
        algo_set = r["algo_set"]
        algo_main = r["algo_main"]

        if not gt_set and not algo_set:
            continue

        union = gt_set | algo_set
        inter = gt_set & algo_set
        jaccard = len(inter) / len(union) if union else None

        main_hit = (r["gt_label"] == algo_main) if (r["gt_label"] and algo_main) else False
        exact_match = (gt_set == algo_set) if (gt_set or algo_set) else False

        extra = sorted(algo_set - gt_set)
        missing = sorted(gt_set - algo_set)

        rows.append({
            "doc_id": r["doc_id"],
            "gt_label": r["gt_label"] or "",
            "algo_main": algo_main or "",
            "algo_subs": "|".join(sorted(r["algo_subs"])),
            "algo_set_str": "|".join(sorted(algo_set)),
            "n_algo_labels": len(algo_set),
            "n_algo_subs": len(r["algo_subs"]),
            "main_hit": int(main_hit),
            "jaccard": jaccard if jaccard is not None else np.nan,
            "exact_set_match": int(exact_match),
            "extra_in_algo": "|".join(extra),
            "missing_in_algo": "|".join(missing),
            "has_gt_label": int(bool(gt_set)),
            "has_algo_label": int(bool(algo_set)),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df_labeled = df[(df["has_gt_label"] == 1) & (df["has_algo_label"] == 1)]
        n = len(df_labeled)
        if n > 0:
            print(f"\n{'='*60}")
            print(f"  Phase 1: 기본 라벨 비교 (정답 라벨 있는 {n}명)")
            print(f"{'='*60}")
            print(f"  주 학대유형 일치율 (Main Hit Rate)  : {df_labeled['main_hit'].mean():.4f}")
            print(f"  평균 Jaccard 유사도                  : {df_labeled['jaccard'].mean():.4f}")
            print(f"  정확 집합 일치율 (Exact Set Match)  : {df_labeled['exact_set_match'].mean():.4f}")
            print(f"  알고리즘 평균 라벨 수               : {df_labeled['n_algo_labels'].mean():.2f}")
            print(f"{'='*60}")

    return df


# ═══════════════════════════════════════════════════════════════
#  Phase 2: 중복학대 (Poly-victimization) 분석
# ═══════════════════════════════════════════════════════════════

def compute_poly_victimization(records: List[Dict]) -> Dict[str, Any]:
    """
    중복학대(poly-victimization) 빈도를 계산한다.

    핵심 지표:
    ─────────────────────────────────────────────────────────────
    poly_rate:
        부 학대유형이 1개 이상 있는 사례의 비율.
        수식: Poly Rate = |{i : |S_i| ≥ 1}| / N_labeled

    n_labels_distribution:
        알고리즘이 부여한 총 라벨 수(주+부)의 분포.
    ─────────────────────────────────────────────────────────────
    """
    labeled = [r for r in records if r["algo_main"] is not None]
    n_labeled = len(labeled)

    if n_labeled == 0:
        print("[LABEL-COMP][POLY] 알고리즘 라벨이 부여된 사례가 없습니다.")
        return {}

    n_subs_list = [len(r["algo_subs"]) for r in labeled]
    n_total_labels_list = [len(r["algo_set"]) for r in labeled]

    poly_count = sum(1 for ns in n_subs_list if ns >= 1)
    poly_rate = poly_count / n_labeled

    label_count_dist = Counter(n_total_labels_list)

    main_to_n_subs = defaultdict(list)
    for r in labeled:
        main_to_n_subs[r["algo_main"]].append(len(r["algo_subs"]))

    avg_subs_by_main = {
        m: np.mean(counts) for m, counts in main_to_n_subs.items()
    }

    print(f"\n{'='*60}")
    print(f"  Phase 2: 중복학대(Poly-victimization) 분석")
    print(f"{'='*60}")
    print(f"  전체 라벨 부여 아동 수: {n_labeled}")
    print(f"  중복학대 사례 수      : {poly_count}")
    print(f"  중복학대 비율          : {poly_rate:.4f} ({poly_rate*100:.1f}%)")
    print(f"")
    print(f"  총 라벨 수 분포:")
    for n_labels in sorted(label_count_dist.keys()):
        cnt = label_count_dist[n_labels]
        pct = cnt / n_labeled * 100
        print(f"    {n_labels}개 라벨: {cnt}명 ({pct:.1f}%)")
    print(f"")
    print(f"  주 학대유형별 평균 부 학대유형 수:")
    for m in ABUSE_ORDER:
        if m in avg_subs_by_main:
            print(f"    {m}: {avg_subs_by_main[m]:.2f}개 (n={len(main_to_n_subs[m])})")
    print(f"{'='*60}")

    rows = []
    for r in labeled:
        rows.append({
            "doc_id": r["doc_id"],
            "algo_main": r["algo_main"],
            "n_subs": len(r["algo_subs"]),
            "n_total_labels": len(r["algo_set"]),
            "is_poly": int(len(r["algo_subs"]) >= 1),
            "algo_subs_str": "|".join(sorted(r["algo_subs"])),
        })

    df_poly = pd.DataFrame(rows)

    return {
        "poly_rate": poly_rate,
        "poly_count": poly_count,
        "n_labeled": n_labeled,
        "label_count_dist": dict(label_count_dist),
        "avg_subs_by_main": avg_subs_by_main,
        "df_poly": df_poly,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 3: 동시발생 행렬 + 전이 패턴
# ═══════════════════════════════════════════════════════════════

def compute_cooccurrence_matrix(records: List[Dict]) -> Dict[str, Any]:
    """
    학대유형 동시발생 행렬(Co-occurrence Matrix)과 전이 패턴을 계산한다.

    ─── 동시발생 행렬 C ───
    수식: C_{jk} = Σ_i 1[A_j ∈ L^Algo-set_i  and  A_k ∈ L^Algo-set_i]  (j ≠ k)

    ─── 조건부 확률 ───
    수식: P(A_k | A_j) = C_{jk} / N_j

    ─── 주→부 전이 패턴 ───
    수식: P_trans(A_k | Main=A_j) = |{i : M_i=A_j and A_k ∈ S_i}| / |{i : M_i=A_j}|
    """
    labeled = [r for r in records if r["algo_main"] is not None]

    cooc_matrix = pd.DataFrame(0, index=ABUSE_ORDER, columns=ABUSE_ORDER, dtype=int)
    type_counts = Counter()

    for r in labeled:
        algo_set = r["algo_set"]
        for a in algo_set:
            if a in ABUSE_ORDER:
                type_counts[a] += 1
        for a1, a2 in combinations(algo_set, 2):
            if a1 in ABUSE_ORDER and a2 in ABUSE_ORDER:
                cooc_matrix.loc[a1, a2] += 1
                cooc_matrix.loc[a2, a1] += 1

    for a in ABUSE_ORDER:
        cooc_matrix.loc[a, a] = type_counts[a]

    cond_prob = pd.DataFrame(0.0, index=ABUSE_ORDER, columns=ABUSE_ORDER)
    for a_j in ABUSE_ORDER:
        n_j = type_counts[a_j]
        if n_j > 0:
            for a_k in ABUSE_ORDER:
                if a_j == a_k:
                    cond_prob.loc[a_j, a_k] = 1.0
                else:
                    cond_prob.loc[a_j, a_k] = cooc_matrix.loc[a_j, a_k] / n_j

    transition_counts = pd.DataFrame(0, index=ABUSE_ORDER, columns=ABUSE_ORDER, dtype=int)
    main_counts = Counter()

    for r in labeled:
        main = r["algo_main"]
        if main not in ABUSE_ORDER:
            continue
        main_counts[main] += 1
        for sub in r["algo_subs"]:
            if sub in ABUSE_ORDER:
                transition_counts.loc[main, sub] += 1

    transition_prob = pd.DataFrame(0.0, index=ABUSE_ORDER, columns=ABUSE_ORDER)
    for main_type in ABUSE_ORDER:
        n_main = main_counts[main_type]
        if n_main > 0:
            for sub_type in ABUSE_ORDER:
                transition_prob.loc[main_type, sub_type] = (
                    transition_counts.loc[main_type, sub_type] / n_main
                )

    print(f"\n{'='*60}")
    print(f"  Phase 3: 동시발생 행렬 & 전이 패턴")
    print(f"{'='*60}")
    print(f"\n  [3-A] 학대유형별 아동 수 (주+부 포함):")
    for a in ABUSE_ORDER:
        print(f"    {a}: {type_counts[a]}명")

    print(f"\n  [3-B] 동시발생 행렬 (절대 빈도):")
    print(cooc_matrix.to_string())

    print(f"\n  [3-C] 조건부 확률 P(열 | 행):")
    print(cond_prob.round(4).to_string())

    print(f"\n  [3-D] 주→부 전이 확률 P(부=열 | 주=행):")
    print(transition_prob.round(4).to_string())
    print(f"{'='*60}")

    return {
        "cooc_matrix": cooc_matrix,
        "cond_prob": cond_prob,
        "transition_counts": transition_counts,
        "transition_prob": transition_prob,
        "type_counts": dict(type_counts),
        "main_counts": dict(main_counts),
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 4: 정보 손실 정량화 (엔트로피)
# ═══════════════════════════════════════════════════════════════

def compute_information_loss(records: List[Dict]) -> Dict[str, Any]:
    """
    정답(GT) 단일 라벨 대비 알고리즘 다중 라벨의 정보량 차이를 엔트로피로 측정한다.

    ─── 정보 손실 정의 ───
    수식: ΔH_i = H_i^{multi} - H_i^{single}

    여기서:
      H_i^{multi}  = -Σ_{A_k ∈ L^Algo-set_i} p_k · log₂(p_k)   (알고리즘 다중 라벨 엔트로피)
      H_i^{single} = 0 bits                                      (GT 단일 라벨: 확정적 할당)
      p_k = Score(i, A_k) / Σ_{A_j ∈ L^Algo-set_i} Score(i, A_j)

    따라서 ΔH_i = H_i^{multi} - 0 = H_i^{multi}

    ─── 하위집단 분리 보고 ───
    Main only (n_labels=1): ΔH = 0 (단일 라벨 → 정보 손실 없음)
    Main+Sub  (n_labels>1): ΔH > 0 (다중 라벨 → 정보 손실 발생)
    전체 평균이 중앙값보다 작을 수 있음: Main only(ΔH=0)가 평균을 끌어내리기 때문
    """
    labeled = [r for r in records if r["algo_main"] is not None and len(r["algo_set"]) > 0]

    entropies = []
    details = []

    for r in labeled:
        algo_set = r["algo_set"]
        scores = r["abuse_scores"]

        relevant_scores = {a: scores.get(a, 0) for a in algo_set}
        total_score = sum(relevant_scores.values())

        if total_score <= 0:
            n = len(algo_set)
            if n > 1:
                h_multi = np.log2(n)
            else:
                h_multi = 0.0
        else:
            probs = {a: s / total_score for a, s in relevant_scores.items() if s > 0}
            h_multi = 0.0
            for a, p in probs.items():
                if p > 0:
                    h_multi -= p * np.log2(p)

        # GT 단일 라벨의 엔트로피: 확정적 할당이므로 H_single = 0
        h_single = 0.0
        # 정보 손실: ΔH = H_multi - H_single
        delta_h = h_multi - h_single

        is_multi = len(algo_set) > 1
        subgroup = "Main+Sub" if is_multi else "Main only"

        entropies.append(delta_h)
        details.append({
            "doc_id": r["doc_id"],
            "algo_main": r["algo_main"],
            "n_labels": len(algo_set),
            "subgroup": subgroup,
            "H_multi": h_multi,
            "H_single": h_single,
            "entropy_algo": h_multi,
            "entropy_gt": h_single,
            "info_loss": delta_h,
        })

    df_entropy = pd.DataFrame(details)

    # ── 하위집단별 기술통계 ──
    subgroup_stats = {}
    if not df_entropy.empty:
        N = len(df_entropy)
        mean_entropy = df_entropy["info_loss"].mean()
        median_entropy = df_entropy["info_loss"].median()

        df_main_only = df_entropy[df_entropy["n_labels"] == 1]
        df_main_sub = df_entropy[df_entropy["n_labels"] > 1]
        n_main_only = len(df_main_only)
        n_main_sub = len(df_main_sub)

        subgroup_stats["total"] = {
            "n": N,
            "mean": float(mean_entropy),
            "median": float(median_entropy),
        }
        subgroup_stats["main_only"] = {
            "n": n_main_only,
            "pct": n_main_only / N * 100 if N > 0 else 0,
            "mean": float(df_main_only["info_loss"].mean()) if n_main_only > 0 else 0.0,
            "median": float(df_main_only["info_loss"].median()) if n_main_only > 0 else 0.0,
        }
        subgroup_stats["main_sub"] = {
            "n": n_main_sub,
            "pct": n_main_sub / N * 100 if N > 0 else 0,
            "mean": float(df_main_sub["info_loss"].mean()) if n_main_sub > 0 else 0.0,
            "median": float(df_main_sub["info_loss"].median()) if n_main_sub > 0 else 0.0,
        }

        entropy_by_n = df_entropy.groupby("n_labels")["info_loss"].agg(["mean", "count"])

        print(f"\n{'='*70}")
        print(f"  Phase 4: 정보 손실 정량화 (ΔH = H_multi - H_single)")
        print(f"{'='*70}")
        print(f"  정의: ΔH_i = H_i^{{multi}} - H_i^{{single}}")
        print(f"        H_single = 0 bits (GT 단일 라벨: 확정적 할당)")
        print(f"        따라서 ΔH_i = H_i^{{multi}}")
        print(f"{'─'*70}")
        print(f"  [전체] 분석 대상 아동 수   : {N}")
        print(f"  [전체] 평균 ΔH             : {mean_entropy:.4f} bits")
        print(f"  [전체] 중앙값 ΔH           : {median_entropy:.4f} bits")
        print(f"{'─'*70}")
        print(f"  [Main only] n={n_main_only} ({subgroup_stats['main_only']['pct']:.1f}%)")
        print(f"    ΔH = 0 bits (단일 라벨 → 정보 손실 없음)")
        print(f"  [Main+Sub]  n={n_main_sub} ({subgroup_stats['main_sub']['pct']:.1f}%)")
        if n_main_sub > 0:
            print(f"    평균 ΔH  = {subgroup_stats['main_sub']['mean']:.4f} bits")
            print(f"    중앙값 ΔH = {subgroup_stats['main_sub']['median']:.4f} bits")
        print(f"{'─'*70}")
        if mean_entropy < median_entropy:
            print(f"  ※ 평균({mean_entropy:.4f}) < 중앙값({median_entropy:.4f}): "
                  f"Main only({n_main_only}명, ΔH=0)가 평균을 끌어내림")
        print(f"{'─'*70}")
        print(f"  라벨 수별 평균 ΔH:")
        for n_labels, row in entropy_by_n.iterrows():
            print(f"    {n_labels}개 라벨: 평균 ΔH = {row['mean']:.4f} bits (n={int(row['count'])})")
        print(f"{'='*70}")

    return {
        "df_entropy": df_entropy,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_info_loss": float(np.mean(entropies)) if entropies else 0.0,
        "subgroup_stats": subgroup_stats,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 5: Hidden Companion 분석
# ═══════════════════════════════════════════════════════════════

def compute_hidden_companions(records: List[Dict]) -> Dict[str, Any]:
    """
    정답(GT)이 단일 라벨로 분류한 아동들 속에 "숨겨진" 부 학대유형을 찾아낸다.

    ─── Hidden Companion 조건부 확률 ───
    수식: P(Sub=A_k | GT=A_j) =
             |{i : L^GT_i = A_j  and  A_k ∈ S_i}|
             ─────────────────────────────────────────
             |{i : L^GT_i = A_j}|
    """
    labeled = [r for r in records if r["gt_label"] and r["algo_main"]]

    gt_groups = defaultdict(list)
    for r in labeled:
        gt_groups[r["gt_label"]].append(r)

    hc_rows = []
    for gt_type in ABUSE_ORDER:
        group = gt_groups.get(gt_type, [])
        n_group = len(group)
        if n_group == 0:
            continue

        for sub_type in ABUSE_ORDER:
            if sub_type == gt_type:
                continue

            n_with_sub = sum(1 for r in group if sub_type in r["algo_subs"])
            n_with_any = sum(1 for r in group if sub_type in r["algo_set"])

            p_sub = n_with_sub / n_group
            p_any = n_with_any / n_group

            hc_rows.append({
                "gt_label": gt_type,
                "hidden_type": sub_type,
                "n_gt_group": n_group,
                "n_with_sub": n_with_sub,
                "n_with_any": n_with_any,
                "p_hidden_sub": p_sub,
                "p_hidden_any": p_any,
            })

    df_hc = pd.DataFrame(hc_rows)

    if not df_hc.empty:
        pivot_sub = df_hc.pivot(
            index="gt_label", columns="hidden_type", values="p_hidden_sub"
        ).reindex(index=ABUSE_ORDER, columns=ABUSE_ORDER).fillna(0)

        print(f"\n{'='*60}")
        print(f"  Phase 5: Hidden Companion 분석")
        print(f"{'='*60}")
        print(f"\n  정답(GT) 라벨별 아동 수:")
        for gt in ABUSE_ORDER:
            n = len(gt_groups.get(gt, []))
            if n > 0:
                print(f"    GT={gt}: {n}명")

        print(f"\n  Hidden Companion 확률 P(부=열 | GT=행):")
        print(pivot_sub.round(4).to_string())

        print(f"\n  주요 Hidden Companion 발견:")
        for _, row in df_hc.sort_values("p_hidden_sub", ascending=False).head(6).iterrows():
            if row["p_hidden_sub"] > 0:
                print(f"    GT={row['gt_label']} → 숨겨진 {row['hidden_type']}: "
                      f"{row['p_hidden_sub']:.1%} ({row['n_with_sub']}/{row['n_gt_group']}명)")
        print(f"{'='*60}")
    else:
        pivot_sub = pd.DataFrame()

    return {
        "df_hidden_companions": df_hc,
        "pivot_sub": pivot_sub,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 6: 가중 빈도표 구축 (CA 확장용)
# ═══════════════════════════════════════════════════════════════

def build_weighted_frequency_table(
    records: List[Dict],
    alpha: float = 0.5,
    min_total_count: int = 8,
) -> pd.DataFrame:
    """
    주+부 학대유형을 모두 반영하는 가중 단어-학대유형 빈도표를 구축한다.

    ─── 가중치 정의 ───
    수식: w(A_k, i) =
            1.0    if A_k = M_i   (주 학대유형)
            α      if A_k ∈ S_i   (부 학대유형, 0 < α ≤ 1)
            0      otherwise

    Parameters
    ----------
    records : list[dict]
        _load_all_records()의 출력
    alpha : float
        부 학대유형 가중치 (0 < α ≤ 1). 기본값 0.5
    min_total_count : int
        빈도표에 포함되려면 총 가중 빈도가 이 값 이상이어야 함

    Returns
    -------
    pd.DataFrame
        index=word, columns=ABUSE_ORDER, values=가중 빈도
    """
    labeled = [r for r in records if r["algo_main"] in ABUSE_ORDER and r["speech_tokens"]]

    word_counts = defaultdict(lambda: {a: 0.0 for a in ABUSE_ORDER})

    for r in labeled:
        main = r["algo_main"]
        subs = set(r["algo_subs"]) & set(ABUSE_ORDER)
        tokens = r["speech_tokens"]

        unique_tokens = set(tokens)

        for w in unique_tokens:
            word_counts[w][main] += 1.0
            for sub in subs:
                word_counts[w][sub] += alpha

    rows = []
    for word, counts in word_counts.items():
        row = {"word": word}
        row.update(counts)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("word")
    df = df[ABUSE_ORDER]

    df["total_weighted"] = df.sum(axis=1)
    df = df[df["total_weighted"] >= min_total_count]
    df = df.drop(columns=["total_weighted"])

    print(f"\n{'='*60}")
    print(f"  Phase 6: 가중 빈도표 (α={alpha})")
    print(f"{'='*60}")
    print(f"  입력 아동 수 (라벨+발화 있음): {len(labeled)}")
    print(f"  빈도표 단어 수 (min≥{min_total_count}): {len(df)}")
    print(f"  학대유형별 가중 빈도 합계:")
    for a in ABUSE_ORDER:
        print(f"    {a}: {df[a].sum():.1f}")
    print(f"{'='*60}")

    return df


def _run_alpha_sensitivity(
    records: List[Dict],
    alphas: List[float] = None,
    min_total_count: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    부 학대유형 가중치 α의 민감도 분석을 수행한다.
    """
    if alphas is None:
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    results = {}
    for alpha in alphas:
        df = build_weighted_frequency_table(
            records, alpha=alpha, min_total_count=min_total_count,
        )
        results[alpha] = df

    print(f"\n{'='*60}")
    print(f"  α 민감도 비교 요약")
    print(f"{'='*60}")
    print(f"  {'α':>6s}  {'단어수':>8s}  ", end="")
    for a in ABUSE_ORDER:
        print(f"  {ABUSE_LABEL_EN.get(a, a):>14s}", end="")
    print()
    print(f"  {'─'*6}  {'─'*8}  " + "  ".join(["─"*14]*len(ABUSE_ORDER)))

    for alpha in alphas:
        df = results[alpha]
        print(f"  {alpha:6.1f}  {len(df):8d}  ", end="")
        for a in ABUSE_ORDER:
            print(f"  {df[a].sum():14.1f}", end="")
        print()
    print(f"{'='*60}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 7: 시각화
# ═══════════════════════════════════════════════════════════════

def _plot_cooccurrence_heatmap(cooc_result: Dict, out_dir: str, fig_prefix: str = "phase3") -> None:
    """동시발생 행렬과 조건부 확률을 히트맵으로 시각화."""
    os.makedirs(out_dir, exist_ok=True)

    cooc = cooc_result["cooc_matrix"]
    labels_en = [ABUSE_LABEL_EN.get(a, a) for a in ABUSE_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    im = ax.imshow(cooc.values.astype(float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(ABUSE_ORDER)))
    ax.set_yticks(range(len(ABUSE_ORDER)))
    ax.set_xticklabels(labels_en, fontsize=10)
    ax.set_yticklabels(labels_en, fontsize=10)
    for i in range(len(ABUSE_ORDER)):
        for j in range(len(ABUSE_ORDER)):
            val = cooc.values[i, j]
            ax.text(j, i, f"{val}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val > cooc.values.max()*0.6 else "black")
    ax.set_title("(A) Co-occurrence Matrix\n(absolute frequency)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    cond = cooc_result["cond_prob"]
    ax = axes[1]
    cond_display = cond.copy()
    for a in ABUSE_ORDER:
        cond_display.loc[a, a] = np.nan

    im2 = ax.imshow(cond_display.values.astype(float), cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ABUSE_ORDER)))
    ax.set_yticks(range(len(ABUSE_ORDER)))
    ax.set_xticklabels(labels_en, fontsize=10)
    ax.set_yticklabels(labels_en, fontsize=10)
    for i in range(len(ABUSE_ORDER)):
        for j in range(len(ABUSE_ORDER)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="gray")
            else:
                val = cond.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > 0.5 else "black")
    ax.set_title("(B) Conditional Probability\nP(column | row)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Given this type (column)...", fontsize=10)
    ax.set_ylabel("...if a child has this type (row)", fontsize=10)
    fig.colorbar(im2, ax=ax, shrink=0.8)

    fig.suptitle("Maltreatment Type Co-occurrence Analysis\n"
                 "(Algorithm main + sub labels)",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{fig_prefix}_cooccurrence_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {path}")


def _plot_transition_diagram(cooc_result: Dict, out_dir: str, fig_prefix: str = "phase3") -> None:
    """주→부 전이 패턴을 방향성 화살표 다이어그램으로 시각화."""
    os.makedirs(out_dir, exist_ok=True)
    trans_prob = cooc_result["transition_prob"]
    main_counts = cooc_result["main_counts"]

    fig, ax = plt.subplots(figsize=(10, 8))

    n = len(ABUSE_ORDER)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 2.0
    positions = {a: (radius*np.cos(ang), radius*np.sin(ang))
                 for a, ang in zip(ABUSE_ORDER, angles)}

    for a in ABUSE_ORDER:
        x, y = positions[a]
        n_main = main_counts.get(a, 0)
        ax.scatter(x, y, s=800, c=ABUSE_COLORS[a], zorder=5, edgecolors="black", linewidths=1.5)
        ax.text(x, y + 0.35, f"{ABUSE_LABEL_EN[a]}\n(n={n_main})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    for main_type in ABUSE_ORDER:
        for sub_type in ABUSE_ORDER:
            if main_type == sub_type:
                continue
            prob = trans_prob.loc[main_type, sub_type]
            if prob < 0.05:
                continue

            x1, y1 = positions[main_type]
            x2, y2 = positions[sub_type]

            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            shrink = 0.3 / dist if dist > 0 else 0
            sx, sy = x1 + dx*shrink, y1 + dy*shrink
            ex, ey = x2 - dx*shrink, y2 - dy*shrink

            linewidth = max(0.5, prob * 8)
            alpha = min(0.9, 0.3 + prob)

            ax.annotate("",
                xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ABUSE_COLORS[main_type],
                    lw=linewidth,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.15",
                ),
            )
            mid_x = (sx + ex) / 2 + 0.15 * (dy / dist if dist > 0 else 0)
            mid_y = (sy + ey) / 2 - 0.15 * (dx / dist if dist > 0 else 0)
            ax.text(mid_x, mid_y, f"{prob:.0%}",
                    fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Main → Sub Maltreatment Transition Pattern\n"
                 "Arrow width ∝ P(Sub=target | Main=source)",
                 fontsize=13, fontweight="bold")

    path = os.path.join(out_dir, f"{fig_prefix}_transition_diagram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {path}")


def _plot_hidden_companion_heatmap(hc_result: Dict, out_dir: str, fig_prefix: str = "phase5") -> None:
    """Hidden Companion 확률을 히트맵으로 시각화."""
    os.makedirs(out_dir, exist_ok=True)
    pivot = hc_result.get("pivot_sub")
    if pivot is None or pivot.empty:
        print("[PLOT] Hidden Companion 데이터가 없어 시각화를 건너뜁니다.")
        return

    labels_en = [ABUSE_LABEL_EN.get(a, a) for a in ABUSE_ORDER]

    fig, ax = plt.subplots(figsize=(8, 6))
    data = pivot.reindex(index=ABUSE_ORDER, columns=ABUSE_ORDER).fillna(0).values

    mask = np.eye(len(ABUSE_ORDER), dtype=bool)
    data_masked = np.ma.array(data, mask=mask)

    im = ax.imshow(data_masked, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ABUSE_ORDER)))
    ax.set_yticks(range(len(ABUSE_ORDER)))
    ax.set_xticklabels(labels_en, fontsize=10)
    ax.set_yticklabels(labels_en, fontsize=10)

    for i in range(len(ABUSE_ORDER)):
        for j in range(len(ABUSE_ORDER)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="gray")
            else:
                val = data[i, j]
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > 0.4 else "black")

    ax.set_xlabel("Hidden sub-type (detected by algorithm)", fontsize=11)
    ax.set_ylabel("Ground Truth single label (original)", fontsize=11)
    ax.set_title("Hidden Companion Analysis\n"
                 "P(Algorithm detects column as sub | GT labeled as row)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Probability")
    fig.tight_layout()

    path = os.path.join(out_dir, f"{fig_prefix}_hidden_companion_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {path}")


def _plot_poly_victimization_summary(poly_result: Dict, out_dir: str, fig_prefix: str = "phase2") -> None:
    """중복학대 분석 요약 시각화."""
    os.makedirs(out_dir, exist_ok=True)
    df_poly = poly_result.get("df_poly")
    if df_poly is None or df_poly.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    label_dist = poly_result["label_count_dist"]
    x_vals = sorted(label_dist.keys())
    y_vals = [label_dist[x] for x in x_vals]
    total = sum(y_vals)
    bars = ax.bar(x_vals, y_vals, color="#4C72B0", edgecolor="black", alpha=0.8)
    for bar, yv in zip(bars, y_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{yv}\n({yv/total:.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Number of maltreatment types assigned", fontsize=11)
    ax.set_ylabel("Number of children", fontsize=11)
    ax.set_title("(A) Distribution of Label Count per Child\n"
                 f"(Poly-victimization rate: {poly_result['poly_rate']:.1%})",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x_vals)

    ax = axes[1]
    main_types = [a for a in ABUSE_ORDER if a in poly_result["avg_subs_by_main"]]
    avg_subs = [poly_result["avg_subs_by_main"][a] for a in main_types]
    colors = [ABUSE_COLORS[a] for a in main_types]
    labels_en_list = [ABUSE_LABEL_EN.get(a, a) for a in main_types]

    bars = ax.barh(labels_en_list, avg_subs, color=colors, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, avg_subs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean number of sub-types", fontsize=11)
    ax.set_title("(B) Average Sub-type Count by Main Type",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(avg_subs) * 1.3 if avg_subs else 1)

    fig.suptitle("Poly-victimization Analysis: Algorithm Multi-Label Results",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{fig_prefix}_poly_victimization_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {path}")


def _plot_information_loss(entropy_result: Dict, out_dir: str, fig_prefix: str = "phase4") -> None:
    """정보 손실 분포 시각화 (하위집단 분리 표시)."""
    os.makedirs(out_dir, exist_ok=True)
    df = entropy_result.get("df_entropy")
    if df is None or df.empty:
        return

    sg = entropy_result.get("subgroup_stats", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (A) 전체 분포 히스토그램 + 하위집단 표시
    ax = axes[0]
    ax.hist(df["info_loss"], bins=30, color="#4C72B0", edgecolor="black", alpha=0.8)
    overall_mean = df["info_loss"].mean()
    overall_median = df["info_loss"].median()
    ax.axvline(overall_mean, color="red", linestyle="--", linewidth=2,
               label=f"Mean ΔH = {overall_mean:.3f} bits")
    ax.axvline(overall_median, color="orange", linestyle=":", linewidth=2,
               label=f"Median ΔH = {overall_median:.3f} bits")
    ax.set_xlabel("ΔH = H_multi − H_single (bits)", fontsize=11)
    ax.set_ylabel("Number of children", fontsize=11)

    # 하위집단 정보 표시
    subtitle_parts = ["(H_single = 0: deterministic GT label)"]
    if sg.get("main_only") and sg.get("main_sub"):
        subtitle_parts.append(
            f"Main only: n={sg['main_only']['n']} ({sg['main_only']['pct']:.0f}%, ΔH=0)  |  "
            f"Main+Sub: n={sg['main_sub']['n']} ({sg['main_sub']['pct']:.0f}%, "
            f"mean={sg['main_sub']['mean']:.3f})")
    ax.set_title("(A) Distribution of Information Loss ΔH\n"
                 + "\n".join(subtitle_parts),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # (B) 라벨 수별 박스플롯
    ax = axes[1]
    groups_for_box = []
    labels_for_box = []
    for n_labels in sorted(df["n_labels"].unique()):
        subset = df[df["n_labels"] == n_labels]["info_loss"].values
        if len(subset) > 0:
            groups_for_box.append(subset)
            labels_for_box.append(f"{n_labels} types\n(n={len(subset)})")

    if groups_for_box:
        bp = ax.boxplot(groups_for_box, labels=labels_for_box, patch_artist=True,
                        showmeans=True, meanline=True)
        colors_box = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f"]
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors_box[i % len(colors_box)])
            patch.set_alpha(0.7)

    ax.set_xlabel("Number of maltreatment types assigned", fontsize=11)
    ax.set_ylabel("ΔH = H_multi − H_single (bits)", fontsize=11)
    ms_mean = sg.get("main_sub", {}).get("mean", entropy_result["mean_info_loss"])
    ax.set_title("(B) ΔH by Label Count\n"
                 f"(Overall mean ΔH = {entropy_result['mean_info_loss']:.3f} bits, "
                 f"Main+Sub mean = {ms_mean:.3f} bits)",
                 fontsize=11, fontweight="bold")

    fig.suptitle("Information Loss: ΔH = H_multi − H_single\n"
                 "(GT single-label assignment → H_single = 0 bits)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{fig_prefix}_information_loss.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {path}")


def _plot_confusion_matrix(records: List[Dict], out_dir: str, fig_prefix: str = "phase1") -> None:
    """정답(GT) 라벨(행) × 알고리즘 주 학대유형(열)의 혼동행렬 시각화."""
    os.makedirs(out_dir, exist_ok=True)

    labeled = [r for r in records
               if r["gt_label"] in ABUSE_ORDER and r["algo_main"] in ABUSE_ORDER]

    if not labeled:
        print("[PLOT] 혼동행렬을 그릴 사례가 없습니다.")
        return

    gt_labels = [r["gt_label"] for r in labeled]
    algo_labels = [r["algo_main"] for r in labeled]

    cm = pd.crosstab(
        pd.Categorical(gt_labels, categories=ABUSE_ORDER),
        pd.Categorical(algo_labels, categories=ABUSE_ORDER),
    )

    labels_en = [ABUSE_LABEL_EN.get(a, a) for a in ABUSE_ORDER]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto")

    for i in range(len(ABUSE_ORDER)):
        for j in range(len(ABUSE_ORDER)):
            val = cm.values[i, j]
            ax.text(j, i, f"{val}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val > cm.values.max()*0.5 else "black")

    ax.set_xticks(range(len(ABUSE_ORDER)))
    ax.set_yticks(range(len(ABUSE_ORDER)))
    ax.set_xticklabels(labels_en, fontsize=10)
    ax.set_yticklabels(labels_en, fontsize=10)
    ax.set_xlabel("Algorithm Main Label (predicted)", fontsize=11)
    ax.set_ylabel("Ground Truth Label (original)", fontsize=11)
    ax.set_title(f"Confusion Matrix: GT Label vs. Algorithm Main Label\n"
                 f"(n={len(labeled)} children with both labels)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{fig_prefix}_confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {path}")


# ═══════════════════════════════════════════════════════════════
#  파이프라인 진입점 — pipeline.py에서 호출하는 함수
# ═══════════════════════════════════════════════════════════════

def run_label_comparison_from_pipeline(
    json_files: List[str],
    out_dir: str = None,
    abuse_order: List[str] = None,
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
    only_negative: bool = False,
    alpha_values: List[float] = None,
) -> Dict[str, Any]:
    """
    파이프라인 내부에서 호출되는 라벨 비교 분석 통합 함수.

    이 함수는 pipeline.py의 run_pipeline() 안에서 호출됩니다.
    독립 실행은 지원하지 않습니다 — classify_abuse_main_sub()가
    파이프라인 실행 시에만 올바른 라벨을 동적 생성하기 때문입니다.

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    out_dir : str
        결과 저장 디렉토리. None이면 C.OUTPUT_DIR/label_comparison 사용.
    abuse_order : list[str]
        학대유형 순서. 기본값은 C.ABUSE_ORDER.
    sub_threshold : int
        부 학대유형 인정 임계값. 기본값 2.
    use_clinical_text : bool
        임상 텍스트 기반 라벨 추가 여부.
    only_negative : bool
        True이면 정서군이 '부정'인 아동만 분석.
    alpha_values : list[float]
        CA 가중 빈도표의 α 민감도 분석 값. 기본값 [0.0, 0.3, 0.5, 0.7, 1.0]

    Returns
    -------
    dict: 모든 Phase의 결과를 담은 딕셔너리
    """
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    if out_dir is None:
        out_dir = C.LABEL_COMPARISON_DIR or os.path.join(C.OUTPUT_DIR, "label_comparison")

    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "╔" + "═"*68 + "╗")
    print("║  [LABEL-COMP] 정답(GT) 단일 라벨 vs. 알고리즘 주+부 학대유형 비교  ║")
    print("╚" + "═"*68 + "╝")

    # ── 데이터 로드 (핵심: classify_abuse_main_sub 동적 호출) ──
    records = _load_all_records(
        json_files, abuse_order=abuse_order,
        sub_threshold=sub_threshold,
        use_clinical_text=use_clinical_text,
        only_negative=only_negative,
    )

    if not records:
        print("[LABEL-COMP][ERROR] 유효한 레코드가 없습니다.")
        return {}

    results = {"records": records}

    # ── Phase 1: 기본 비교 ──
    print("\n" + "▓"*60)
    print("  [LABEL-COMP] Phase 1: 기본 라벨 비교")
    print("▓"*60)
    df_comparison = compute_basic_comparison_metrics(records)
    results["phase1_comparison"] = df_comparison
    if not df_comparison.empty:
        df_comparison.to_csv(
            os.path.join(out_dir, "phase1_label_comparison.csv"),
            encoding="utf-8-sig", index=False,
        )
        _plot_confusion_matrix(records, out_dir=out_dir)

    # ── Phase 2: 중복학대 ──
    print("\n" + "▓"*60)
    print("  [LABEL-COMP] Phase 2: 중복학대 분석")
    print("▓"*60)
    poly_result = compute_poly_victimization(records)
    results["phase2_poly"] = poly_result
    if poly_result:
        poly_result["df_poly"].to_csv(
            os.path.join(out_dir, "phase2_poly_victimization.csv"),
            encoding="utf-8-sig", index=False,
        )
        _plot_poly_victimization_summary(poly_result, out_dir=out_dir)

    # ── Phase 3: 동시발생 + 전이 ──
    print("\n" + "▓"*60)
    print("  [LABEL-COMP] Phase 3: 동시발생 & 전이 패턴")
    print("▓"*60)
    cooc_result = compute_cooccurrence_matrix(records)
    results["phase3_cooccurrence"] = cooc_result
    if cooc_result:
        cooc_result["cooc_matrix"].to_csv(
            os.path.join(out_dir, "phase3_cooccurrence_matrix.csv"),
            encoding="utf-8-sig",
        )
        cooc_result["cond_prob"].to_csv(
            os.path.join(out_dir, "phase3_conditional_probability.csv"),
            encoding="utf-8-sig",
        )
        cooc_result["transition_prob"].to_csv(
            os.path.join(out_dir, "phase3_transition_probability.csv"),
            encoding="utf-8-sig",
        )
        _plot_cooccurrence_heatmap(cooc_result, out_dir=out_dir)
        _plot_transition_diagram(cooc_result, out_dir=out_dir)

    # ── Phase 4: 정보 손실 ──
    print("\n" + "▓"*60)
    print("  [LABEL-COMP] Phase 4: 정보 손실 정량화")
    print("▓"*60)
    entropy_result = compute_information_loss(records)
    results["phase4_entropy"] = entropy_result
    if entropy_result.get("df_entropy") is not None and not entropy_result["df_entropy"].empty:
        entropy_result["df_entropy"].to_csv(
            os.path.join(out_dir, "phase4_information_loss.csv"),
            encoding="utf-8-sig", index=False,
        )
        _plot_information_loss(entropy_result, out_dir=out_dir)

    # ── Phase 5: Hidden Companion ──
    print("\n" + "▓"*60)
    print("  [LABEL-COMP] Phase 5: Hidden Companion")
    print("▓"*60)
    hc_result = compute_hidden_companions(records)
    results["phase5_hidden_companion"] = hc_result
    if hc_result.get("df_hidden_companions") is not None:
        hc_result["df_hidden_companions"].to_csv(
            os.path.join(out_dir, "phase5_hidden_companions.csv"),
            encoding="utf-8-sig", index=False,
        )
        _plot_hidden_companion_heatmap(hc_result, out_dir=out_dir)

    # ── Phase 6: 가중 빈도표 ──
    print("\n" + "▓"*60)
    print("  [LABEL-COMP] Phase 6: 가중 빈도표 (CA 확장)")
    print("▓"*60)
    alpha_results = _run_alpha_sensitivity(records, alphas=alpha_values)
    results["phase6_weighted_tables"] = alpha_results
    for alpha_val, df_wt in alpha_results.items():
        df_wt.to_csv(
            os.path.join(out_dir, f"phase6_weighted_freq_alpha{alpha_val:.1f}.csv"),
            encoding="utf-8-sig",
        )

    # ── 최종 요약 보고서 ──
    summary_rows = []
    if not df_comparison.empty:
        df_both = df_comparison[(df_comparison["has_gt_label"]==1) & (df_comparison["has_algo_label"]==1)]
        if not df_both.empty:
            summary_rows.append({"metric": "Main Hit Rate", "value": f"{df_both['main_hit'].mean():.4f}"})
            summary_rows.append({"metric": "Mean Jaccard", "value": f"{df_both['jaccard'].mean():.4f}"})
            summary_rows.append({"metric": "Exact Set Match", "value": f"{df_both['exact_set_match'].mean():.4f}"})
    if poly_result:
        summary_rows.append({"metric": "Poly-victimization Rate", "value": f"{poly_result['poly_rate']:.4f}"})
        summary_rows.append({"metric": "N Labeled Children", "value": str(poly_result['n_labeled'])})
    if entropy_result.get("mean_info_loss"):
        summary_rows.append({"metric": "Mean Info Loss (ΔH)", "value": f"{entropy_result['mean_info_loss']:.4f} bits"})
    sg = entropy_result.get("subgroup_stats", {})
    if sg.get("main_only"):
        summary_rows.append({"metric": "Main only (ΔH=0) n", "value": str(sg["main_only"]["n"])})
        summary_rows.append({"metric": "Main only (ΔH=0) %", "value": f"{sg['main_only']['pct']:.1f}%"})
    if sg.get("main_sub"):
        summary_rows.append({"metric": "Main+Sub mean ΔH", "value": f"{sg['main_sub']['mean']:.4f} bits"})
        summary_rows.append({"metric": "Main+Sub median ΔH", "value": f"{sg['main_sub']['median']:.4f} bits"})

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(out_dir, "SUMMARY_all_phases.csv"),
        encoding="utf-8-sig", index=False,
    )

    print("\n" + "╔" + "═"*68 + "╗")
    print("║  [LABEL-COMP] 전체 분석 완료!                                   ║")
    print("╚" + "═"*68 + "╝")
    print(f"\n  결과 저장 디렉토리: {out_dir}")
    print(f"  생성된 파일 목록:")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size = os.path.getsize(fpath)
        print(f"    {f} ({size:,} bytes)")

    return results