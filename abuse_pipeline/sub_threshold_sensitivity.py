#!/usr/bin/env python3
"""
sub_threshold_sensitivity.py
============================
부(sub) 학대유형 할당 임계값(sub_threshold) 민감도 분석

[목적]
  classify_abuse_main_sub()에서 sub_threshold 값을 2, 3, 4, 5로 변화시킬 때
  부 학대유형 할당이 어떻게 변하는지를 체계적으로 분석한다.

  ┌─────────────────────────────────────────────────────────┐
  │  핵심 질문: "sub_threshold를 몇으로 설정해야               │
  │   CA 플롯에서 가장 의미 있는 sub abuse 할당이 되는가?"    │
  └─────────────────────────────────────────────────────────┘

[분석 항목]
  A. 기초 통계: 임계값별 sub abuse 보유 아동 수/비율
  B. 학대유형별 점수 분포: sub abuse 점수의 히스토그램
  C. 유형별 할당 변화: 어떤 학대유형이 sub로 가장 많이 붙는가
  D. 동시발생(Co-occurrence) 패턴: main × sub 조합 빈도
  E. 임계값 간 Jaccard 안정성: threshold 변화에 따른 라벨 세트 유사도
  F. 토큰 풍부도(Token Richness): sub abuse 포함 시 CA 입력 토큰 수 변화
  G. 권장 임계값 도출: 종합 지표 기반 자동 추천

[사용법]
  python sub_threshold_sensitivity.py --data_dir ./data --out_dir ./sub_threshold_output

[출력]
  - CSV: 모든 분석 결과 테이블
  - PNG: 시각화 6종
  - TXT: 최종 권장 임계값 보고서
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
# 한글 폰트 설정 시도
try:
    matplotlib.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']
except:
    pass

import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  0. 상수 정의
# ═══════════════════════════════════════════════════════════════
ABUSE_ORDER = ["성학대", "신체학대", "정서학대", "방임"]
ABUSE_LABEL_EN = {
    "방임": "Neglect",
    "정서학대": "Emotional",
    "신체학대": "Physical",
    "성학대": "Sexual",
}
ABUSE_COLORS = {
    "방임": "#1f77b4",
    "정서학대": "#ff7f0e",
    "신체학대": "#2ca02c",
    "성학대": "#d62728",
}
_SEVERITY_RANK = {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}

THRESHOLDS = [2, 3, 4, 5]

# 정서군 분류에 필요한 상수
VALENCE_ORDER = ["부정", "평범", "긍정"]


# ═══════════════════════════════════════════════════════════════
#  1. 핵심 함수: 학대유형 분류 (sub_threshold 파라미터화)
# ═══════════════════════════════════════════════════════════════

def extract_abuse_scores(rec):
    """
    한 아동의 JSON record에서 학대유형별 점수를 추출한다.

    [수식]
    각 학대유형 a ∈ {성학대, 신체학대, 정서학대, 방임}에 대해:

        S_a = Σ_{i ∈ items(a)} score_i

    예시: 정서학대 관련 문항이 3개이고 각각 점수가 (2, 3, 1)이면
        S_정서학대 = 2 + 3 + 1 = 6

    Returns
    -------
    dict : {학대유형: 합산점수}
    """
    abuse_scores = {a: 0 for a in ABUSE_ORDER}
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
                for a in ABUSE_ORDER:
                    if a in name:
                        abuse_scores[a] += sc
    return abuse_scores


def classify_abuse_main_sub(rec, sub_threshold=2, use_clinical_text=True):
    """
    한 아동의 주(main) 학대유형과 부(sub) 학대유형을 분류한다.

    [알고리즘]
    1단계: 점수 추출
        각 학대유형 a에 대해 S_a = Σ score_i 를 계산

    2단계: 주(main) 학대유형 결정
        main = argmax_{a: S_a > 6} S_a
        (동점 시 심각도 위계: 성학대 > 신체학대 > 정서학대 > 방임)

    3단계: 부(sub) 학대유형 결정  ← ★ 이 단계가 sub_threshold의 영향을 받음
        sub = {a : a ≠ main  AND  S_a ≥ sub_threshold}

    [예시] sub_threshold = 2인 경우:
        S = {성학대: 0, 신체학대: 8, 정서학대: 3, 방임: 1}
        → main = 신체학대 (8 > 6, 최고점)
        → sub = {정서학대}  (3 ≥ 2 이고, 방임은 1 < 2이므로 제외)

    [예시] sub_threshold = 4인 경우:
        같은 점수에서:
        → main = 신체학대
        → sub = {}  (정서학대 3 < 4이므로 제외)

    Parameters
    ----------
    rec : dict
        아동 JSON record
    sub_threshold : int
        부 학대유형 인정 최소 점수 (2, 3, 4, 5 중 택1)
    use_clinical_text : bool
        임상진단/종합소견 텍스트에서 학대유형 키워드 추출 여부

    Returns
    -------
    (main_abuse, sub_abuses, abuse_scores)
    """
    abuse_scores = extract_abuse_scores(rec)

    # 2단계: main 결정 (S_a > 6)
    nonzero = {a: s for a, s in abuse_scores.items() if s > 6}
    main = None
    if nonzero:
        main = max(nonzero,
                   key=lambda a: (nonzero[a], -_SEVERITY_RANK.get(a, 999)))

    # 3단계: sub 결정 (S_a ≥ sub_threshold)
    subs = set()
    for a, s in abuse_scores.items():
        if a == main:
            continue
        if s >= sub_threshold:
            subs.add(a)

    # 4단계: 임상 텍스트 보완
    if use_clinical_text:
        info = rec.get("info", {})
        clin_text = " ".join(
            str(info.get(k, "")) for k in ["임상진단", "임상가 종합소견"] if k in info
        )
        for a in ABUSE_ORDER:
            if a in clin_text:
                if main is None:
                    main = a
                elif a != main:
                    subs.add(a)

    # 5단계: main이 없지만 sub가 있으면 sub 중 최고를 main으로 승격
    if main is None and subs:
        main = sorted(subs,
                      key=lambda x: (abuse_scores.get(x, 0),
                                     -_SEVERITY_RANK.get(x, 999)),
                      reverse=True)[0]
        subs.remove(main)

    if main is None:
        return None, [], abuse_scores

    return main, sorted(subs, key=lambda x: _SEVERITY_RANK.get(x, 999)), abuse_scores


def classify_child_group(rec):
    """정서군 분류 (부정/평범/긍정) - 간소화 버전"""
    info = rec.get("info", {})
    crisis = info.get("위기단계")

    try:
        total_score = int(info.get("합계점수"))
    except (TypeError, ValueError):
        total_score = None

    q_sum = {}
    item_scores = {}
    for q in rec.get("list", []):
        qname = q.get("문항")
        try:
            q_total = int(q.get("문항합계"))
        except (TypeError, ValueError):
            q_total = None
        if qname is not None:
            q_sum[qname] = q_total
        for it in q.get("list", []):
            iname = it.get("항목")
            try:
                sc = int(it.get("점수"))
            except (TypeError, ValueError):
                continue
            if iname:
                item_scores.setdefault(iname, []).append(sc)

    def get_item_max(name, default=0):
        vals = item_scores.get(name)
        return default if not vals else max(vals)

    def get_q(name, default=0):
        v = q_sum.get(name, default)
        return default if v is None else v

    if get_item_max("자해/자살", 0) > 0:
        return "부정"

    happy = get_item_max("행복", 0)
    worry = get_item_max("걱정", 0)
    abuse_total = get_q("학대여부", 0)

    if happy >= 7:
        return "부정"
    if happy == 0 and worry == 0 and abuse_total == 0 and crisis in {None, "정상군", "관찰필요"}:
        return "긍정"

    negative_crisis = {"응급", "위기아동", "학대의심", "상담필요"}
    if crisis in negative_crisis:
        return "부정"
    elif crisis == "정상군":
        return "긍정"
    elif crisis == "관찰필요":
        return "평범"

    if total_score is not None:
        if total_score >= 45:
            return "부정"
        elif total_score <= 10:
            return "긍정"

    risk_qs = ["기분문제", "기본생활", "학대여부", "응급"]
    risk_score = sum(get_q(q, 0) for q in risk_qs)

    if risk_score >= 25:
        return "부정"
    elif risk_score <= 10:
        return "긍정"

    return "평범"


# ═══════════════════════════════════════════════════════════════
#  2. 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_all_records(data_dir, only_negative=True):
    """
    data_dir에서 모든 JSON 파일을 로드하고 아동별 정보를 추출한다.

    Parameters
    ----------
    data_dir : str
        JSON 파일이 들어있는 디렉토리
    only_negative : bool
        True이면 정서군='부정' 아동만 반환 (학대 분석 대상)

    Returns
    -------
    list[dict] : 아동별 정보 리스트
    """
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not json_files:
        print(f"[ERROR] {data_dir}에 JSON 파일이 없습니다.")
        sys.exit(1)

    print(f"[LOAD] JSON 파일 {len(json_files)}개 발견")

    records = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue

        # JSON 구조가 리스트인 경우 처리
        if isinstance(raw, list):
            recs = [x for x in raw if isinstance(x, dict)]
        elif isinstance(raw, dict):
            recs = [raw]
        else:
            continue

        for rec in recs:
            info = rec.get("info", {})
            doc_id = info.get("ID") or info.get("id") or os.path.basename(path)

            # 정서군 분류
            valence = classify_child_group(rec)
            if only_negative and valence != "부정":
                continue

            # 학대 점수 추출 (threshold 독립적)
            abuse_scores = extract_abuse_scores(rec)

            records.append({
                "doc_id": doc_id,
                "rec": rec,
                "valence": valence,
                "abuse_scores": abuse_scores,
                "source_file": path,
            })

    print(f"[LOAD] 유효 아동 수: {len(records)}명"
          + (f" (정서군='부정'만)" if only_negative else ""))
    return records


# ═══════════════════════════════════════════════════════════════
#  3. 분석 함수들
# ═══════════════════════════════════════════════════════════════

def analyze_threshold(records, threshold):
    """
    특정 sub_threshold에서의 분류 결과를 반환한다.

    Returns
    -------
    list[dict] : 아동별 분류 결과
    """
    results = []
    for r in records:
        main, subs, scores = classify_abuse_main_sub(
            r["rec"], sub_threshold=threshold
        )
        results.append({
            "doc_id": r["doc_id"],
            "main": main,
            "subs": subs,
            "n_subs": len(subs),
            "abuse_scores": scores,
            "total_types": 1 + len(subs) if main else 0,
            "label_set": frozenset([main] + subs) if main else frozenset(),
        })
    return results


# ─── A. 기초 통계 ────────────────────────────────────────────

def compute_basic_stats(all_results):
    """
    [분석 A] 각 임계값별 기초 통계

    측정 항목:
    - 전체 아동 수 (N)
    - main abuse 보유 아동 수
    - sub abuse 보유 아동 수/비율
    - 평균 sub abuse 개수
    - 중복학대(poly-victimization) 아동 수: total_types ≥ 2
    """
    rows = []
    for thr, results in all_results.items():
        n_total = len(results)
        n_with_main = sum(1 for r in results if r["main"] is not None)
        n_with_sub = sum(1 for r in results if r["n_subs"] > 0)
        n_poly = sum(1 for r in results if r["total_types"] >= 2)
        avg_subs = np.mean([r["n_subs"] for r in results]) if results else 0
        avg_types = np.mean([r["total_types"] for r in results]) if results else 0

        # sub 개수 분포
        sub_counts = Counter(r["n_subs"] for r in results)

        rows.append({
            "threshold": thr,
            "N_total": n_total,
            "N_with_main": n_with_main,
            "N_with_sub": n_with_sub,
            "pct_with_sub": round(n_with_sub / n_total * 100, 1) if n_total else 0,
            "N_poly_victim": n_poly,
            "pct_poly": round(n_poly / n_total * 100, 1) if n_total else 0,
            "avg_n_subs": round(avg_subs, 3),
            "avg_total_types": round(avg_types, 3),
            "n_sub_0": sub_counts.get(0, 0),
            "n_sub_1": sub_counts.get(1, 0),
            "n_sub_2": sub_counts.get(2, 0),
            "n_sub_3": sub_counts.get(3, 0),
        })

    return pd.DataFrame(rows)


# ─── B. 학대유형별 점수 분포 ─────────────────────────────────

def compute_score_distribution(records):
    """
    [분석 B] 학대유형별 점수 분포 (threshold 독립적)

    main이 아닌 학대유형의 점수만 모아서,
    "이 점수가 sub_threshold 이상/미만인 비율"을 계산한다.

    [수식]
    각 임계값 τ에 대해:
        포함률(τ) = |{아동 i : S_{a,i} ≥ τ, a ≠ main_i}| / |{아동 i : S_{a,i} > 0, a ≠ main_i}|

    즉, "점수가 1 이상인 비-주학대유형 중 τ 이상인 비율"
    """
    non_main_scores = {a: [] for a in ABUSE_ORDER}

    for r in records:
        scores = r["abuse_scores"]
        # main 결정 (threshold 독립적으로 > 6 기준)
        nonzero = {a: s for a, s in scores.items() if s > 6}
        main = max(nonzero, key=lambda a: (nonzero[a], -_SEVERITY_RANK.get(a, 999))) if nonzero else None

        for a in ABUSE_ORDER:
            if a == main:
                continue
            if scores[a] > 0:  # 점수가 0인 것은 아예 해당 없음
                non_main_scores[a].append(scores[a])

    # 각 threshold에서의 포함률
    inclusion_rows = []
    for thr in THRESHOLDS:
        for a in ABUSE_ORDER:
            vals = non_main_scores[a]
            n_total = len(vals)
            n_included = sum(1 for v in vals if v >= thr)
            n_excluded = n_total - n_included
            inclusion_rows.append({
                "threshold": thr,
                "abuse_type": a,
                "abuse_en": ABUSE_LABEL_EN[a],
                "n_nonzero_nonmain": n_total,
                "n_included": n_included,
                "n_excluded": n_excluded,
                "inclusion_rate": round(n_included / n_total * 100, 1) if n_total else 0,
            })

    return pd.DataFrame(inclusion_rows), non_main_scores


# ─── C. 유형별 할당 변화 ─────────────────────────────────────

def compute_type_allocation(all_results):
    """
    [분석 C] 각 임계값에서 어떤 학대유형이 sub로 할당되는 빈도

    예시 결과:
        threshold=2: {정서학대: 45, 방임: 30, 신체학대: 20, 성학대: 5}
        threshold=5: {정서학대: 12, 방임:  8, 신체학대:  5, 성학대: 2}
    """
    rows = []
    for thr, results in all_results.items():
        sub_counter = Counter()
        for r in results:
            for s in r["subs"]:
                sub_counter[s] += 1

        for a in ABUSE_ORDER:
            rows.append({
                "threshold": thr,
                "abuse_type": a,
                "abuse_en": ABUSE_LABEL_EN[a],
                "n_as_sub": sub_counter.get(a, 0),
            })

    return pd.DataFrame(rows)


# ─── D. 동시발생 패턴 ────────────────────────────────────────

def compute_cooccurrence(all_results):
    """
    [분석 D] main × sub 동시발생 행렬

    각 threshold에서 main=X, sub에 Y가 포함된 아동 수를 집계.
    → "어떤 조합이 가장 빈번한가?"
    """
    matrices = {}
    for thr, results in all_results.items():
        mat = pd.DataFrame(0, index=ABUSE_ORDER, columns=ABUSE_ORDER)
        for r in results:
            if r["main"] is None:
                continue
            for s in r["subs"]:
                mat.loc[r["main"], s] += 1
        matrices[thr] = mat
    return matrices


# ─── E. Jaccard 안정성 ───────────────────────────────────────

def compute_jaccard_stability(all_results):
    """
    [분석 E] 인접 임계값 간 Jaccard 유사도

    [수식]
    두 임계값 τ₁, τ₂에서 아동 i의 학대유형 라벨 세트를 L_i(τ₁), L_i(τ₂)라 하면:

        J_i(τ₁, τ₂) = |L_i(τ₁) ∩ L_i(τ₂)| / |L_i(τ₁) ∪ L_i(τ₂)|

    전체 평균:
        J̄(τ₁, τ₂) = (1/N) Σᵢ J_i(τ₁, τ₂)

    [예시]
    아동 A: threshold=2 → {신체학대, 정서학대}, threshold=3 → {신체학대}
        J_A(2, 3) = |{신체학대}| / |{신체학대, 정서학대}| = 1/2 = 0.5
    """
    rows = []
    for t1, t2 in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
        r1 = all_results[t1]
        r2 = all_results[t2]

        jaccards = []
        n_changed = 0
        for i in range(len(r1)):
            s1 = r1[i]["label_set"]
            s2 = r2[i]["label_set"]

            if not s1 and not s2:
                j = 1.0
            elif not s1 or not s2:
                j = 0.0
            else:
                inter = len(s1 & s2)
                union = len(s1 | s2)
                j = inter / union if union > 0 else 1.0

            jaccards.append(j)
            if s1 != s2:
                n_changed += 1

        rows.append({
            "pair": f"τ={t1} → τ={t2}",
            "threshold_from": t1,
            "threshold_to": t2,
            "mean_jaccard": round(np.mean(jaccards), 4),
            "median_jaccard": round(np.median(jaccards), 4),
            "std_jaccard": round(np.std(jaccards), 4),
            "n_changed": n_changed,
            "pct_changed": round(n_changed / len(r1) * 100, 1) if r1 else 0,
        })

    return pd.DataFrame(rows)


# ─── F. 토큰 풍부도 (CA 입력 관련) ──────────────────────────

def compute_token_richness_proxy(all_results):
    """
    [분석 F] sub abuse 포함에 따른 "CA에 입력될 수 있는 학대유형-아동 연결 수" 변화

    CA의 빈도표는 (학대유형 × 토큰) 행렬이므로,
    sub abuse를 인정하면 한 아동의 토큰이 여러 학대유형에 중복 기여할 수 있다.

    여기서는 "총 학대유형-아동 할당 쌍 수"를 proxy로 계산한다.

    [수식]
    총 할당 쌍 수:
        Pairs(τ) = Σᵢ |L_i(τ)|

    예시: 아동 100명, threshold=2에서 30명이 sub 1개씩 추가로 가지면
        Pairs(2) = 100 (main) + 30 (sub) = 130
    """
    rows = []
    for thr, results in all_results.items():
        total_pairs = sum(r["total_types"] for r in results)
        only_main = sum(1 for r in results if r["total_types"] == 1)
        with_sub = sum(1 for r in results if r["total_types"] >= 2)

        rows.append({
            "threshold": thr,
            "total_assignment_pairs": total_pairs,
            "n_main_only": only_main,
            "n_with_sub": with_sub,
            "avg_types_per_child": round(total_pairs / len(results), 3) if results else 0,
        })

    return pd.DataFrame(rows)


# ─── G. 종합 권장 임계값 도출 ────────────────────────────────

def recommend_threshold(basic_stats, jaccard_df, type_alloc_df):
    """
    [분석 G] 종합 지표 기반 최적 임계값 추천

    [기준]
    1. 과소할당 방지: sub abuse 비율이 너무 낮으면 중복학대를 놓침
    2. 과다할당 방지: sub abuse 비율이 너무 높으면 노이즈 유입
    3. 안정성: 인접 threshold와의 Jaccard가 높을수록 robust
    4. 학대유형 균형: 특정 유형만 sub로 몰리지 않는 것이 바람직

    [점수 산출]
    각 기준을 0~1로 정규화한 뒤 가중합:
        Score(τ) = w₁·(적정 sub 비율) + w₂·(안정성) + w₃·(유형 균형)

    w₁ = 0.4, w₂ = 0.3, w₃ = 0.3
    """
    scores = {}

    for thr in THRESHOLDS:
        row = basic_stats[basic_stats["threshold"] == thr].iloc[0]

        # 1. 적정 sub 비율 (10~40%가 이상적이라 가정)
        pct = row["pct_with_sub"]
        if pct <= 5:
            s1 = 0.2  # 너무 적음
        elif pct <= 10:
            s1 = 0.5
        elif pct <= 40:
            s1 = 1.0  # 적정 범위
        elif pct <= 60:
            s1 = 0.7
        else:
            s1 = 0.3  # 너무 많음

        # 2. 안정성 (Jaccard)
        jacc_rows = jaccard_df[
            (jaccard_df["threshold_from"] == thr) | (jaccard_df["threshold_to"] == thr)
            ]
        s2 = jacc_rows["mean_jaccard"].mean() if not jacc_rows.empty else 0.5

        # 3. 유형 균형 (Gini coefficient의 역수 개념)
        type_row = type_alloc_df[type_alloc_df["threshold"] == thr]
        sub_counts = type_row["n_as_sub"].values
        if sub_counts.sum() > 0:
            # 균등 분포일수록 좋음
            p = sub_counts / sub_counts.sum()
            # Shannon entropy를 정규화
            entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
            max_entropy = np.log2(len(ABUSE_ORDER))
            s3 = entropy / max_entropy if max_entropy > 0 else 0
        else:
            s3 = 0

        total_score = 0.4 * s1 + 0.3 * s2 + 0.3 * s3
        scores[thr] = {
            "threshold": thr,
            "s1_adequate_ratio": round(s1, 3),
            "s2_stability": round(s2, 3),
            "s3_type_balance": round(s3, 3),
            "total_score": round(total_score, 3),
        }

    scores_df = pd.DataFrame(scores.values())
    best = scores_df.loc[scores_df["total_score"].idxmax()]

    return scores_df, int(best["threshold"])


# ═══════════════════════════════════════════════════════════════
#  4. 시각화 함수들
# ═══════════════════════════════════════════════════════════════

def plot_basic_stats(basic_stats, out_dir):
    """[그림 1] 임계값별 기초 통계 막대 그래프"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    thrs = basic_stats["threshold"].values
    x = np.arange(len(thrs))

    # (a) sub abuse 보유 비율
    ax = axes[0]
    bars = ax.bar(x, basic_stats["pct_with_sub"], color="#4ECDC4", edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={t}" for t in thrs])
    ax.set_ylabel("% of children with sub abuse")
    ax.set_title("(a) Sub-abuse prevalence")
    for bar, val in zip(bars, basic_stats["pct_with_sub"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=10)

    # (b) 평균 sub 개수
    ax = axes[1]
    bars = ax.bar(x, basic_stats["avg_n_subs"], color="#FF6B6B", edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={t}" for t in thrs])
    ax.set_ylabel("Avg. number of sub-abuse types")
    ax.set_title("(b) Average sub-abuse count")
    for bar, val in zip(bars, basic_stats["avg_n_subs"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=10)

    # (c) 중복학대 비율
    ax = axes[2]
    bars = ax.bar(x, basic_stats["pct_poly"], color="#FFE66D", edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={t}" for t in thrs])
    ax.set_ylabel("% of poly-victimized children")
    ax.set_title("(c) Poly-victimization rate")
    for bar, val in zip(bars, basic_stats["pct_poly"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=10)

    plt.suptitle("Analysis A: Basic Statistics by Sub-threshold (τ)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_basic_stats.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig1_basic_stats.png 저장 완료")


def plot_score_distribution(non_main_scores, out_dir):
    """[그림 2] 비-주학대유형의 점수 히스토그램 + 임계값 선"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, a in enumerate(ABUSE_ORDER):
        ax = axes[idx]
        vals = non_main_scores[a]

        if vals:
            max_val = max(vals) + 1
            bins = np.arange(0.5, max_val + 1, 1)
            ax.hist(vals, bins=bins, color=ABUSE_COLORS[a], alpha=0.7, edgecolor='black')

            # 임계값 선 표시
            colors_line = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
            for i, thr in enumerate(THRESHOLDS):
                ax.axvline(thr - 0.5, color=colors_line[i], linestyle='--', linewidth=2,
                           label=f'τ={thr}')

        ax.set_xlabel("Score (non-main abuse)")
        ax.set_ylabel("Count")
        ax.set_title(f"{ABUSE_LABEL_EN[a]} ({a})")
        ax.legend(fontsize=8)

    plt.suptitle("Analysis B: Non-main Abuse Score Distribution\n"
                 "(Dashed lines = sub_threshold candidates)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_score_distribution.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig2_score_distribution.png 저장 완료")


def plot_type_allocation(type_alloc_df, out_dir):
    """[그림 3] 학대유형별 sub 할당 빈도 (Grouped Bar)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_types = len(ABUSE_ORDER)
    n_thresholds = len(THRESHOLDS)
    width = 0.18
    x = np.arange(n_types)

    for i, thr in enumerate(THRESHOLDS):
        vals = type_alloc_df[type_alloc_df["threshold"] == thr]["n_as_sub"].values
        offset = (i - n_thresholds / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=f"τ={thr}",
                      edgecolor='black', alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(v), ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ABUSE_LABEL_EN[a]}\n({a})" for a in ABUSE_ORDER])
    ax.set_ylabel("Number of children assigned as sub-abuse")
    ax.set_title("Analysis C: Sub-abuse Type Allocation by Threshold", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_type_allocation.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig3_type_allocation.png 저장 완료")


def plot_cooccurrence(matrices, out_dir):
    """[그림 4] Main × Sub 동시발생 히트맵 (4개 패널)"""
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(20, 5))

    en_labels = [ABUSE_LABEL_EN[a] for a in ABUSE_ORDER]

    for idx, thr in enumerate(THRESHOLDS):
        ax = axes[idx]
        mat = matrices[thr]

        im = ax.imshow(mat.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(ABUSE_ORDER)))
        ax.set_yticks(range(len(ABUSE_ORDER)))
        ax.set_xticklabels(en_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(en_labels, fontsize=8)
        ax.set_xlabel("Sub abuse")
        ax.set_ylabel("Main abuse")
        ax.set_title(f"τ = {thr}", fontweight='bold')

        # 숫자 표시
        for i in range(len(ABUSE_ORDER)):
            for j in range(len(ABUSE_ORDER)):
                val = int(mat.values[i, j])
                if val > 0:
                    ax.text(j, i, str(val), ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='white' if val > mat.values.max() * 0.6 else 'black')

    plt.suptitle("Analysis D: Main × Sub Co-occurrence Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_cooccurrence.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig4_cooccurrence.png 저장 완료")


def plot_jaccard_stability(jaccard_df, out_dir):
    """[그림 5] Jaccard 안정성 그래프"""
    fig, ax = plt.subplots(figsize=(8, 5))

    pairs = jaccard_df["pair"].values
    means = jaccard_df["mean_jaccard"].values
    stds = jaccard_df["std_jaccard"].values
    pct_changed = jaccard_df["pct_changed"].values

    x = np.arange(len(pairs))

    # Jaccard bar
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color='#A8E6CF', edgecolor='black', width=0.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=10)
    ax.set_ylabel("Mean Jaccard Similarity", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='High stability (0.8)')

    # 숫자 표시
    for bar, m, pc in zip(bars, means, pct_changed):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[0] + 0.02,
                f"J={m:.3f}\n({pc:.1f}% changed)",
                ha='center', va='bottom', fontsize=9)

    ax.set_title("Analysis E: Label-set Stability (Jaccard Similarity)\n"
                 "Between Adjacent Thresholds", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_jaccard_stability.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig5_jaccard_stability.png 저장 완료")


def plot_recommendation(scores_df, best_thr, out_dir):
    """[그림 6] 종합 점수 + 추천 임계값"""
    fig, ax = plt.subplots(figsize=(10, 6))

    thrs = scores_df["threshold"].values
    x = np.arange(len(thrs))
    width = 0.2

    # 세부 점수
    ax.bar(x - width, scores_df["s1_adequate_ratio"] * 0.4, width=width,
           label="Adequate ratio (×0.4)", color="#FF6B6B", edgecolor='black', alpha=0.8)
    ax.bar(x, scores_df["s2_stability"] * 0.3, width=width,
           label="Stability (×0.3)", color="#4ECDC4", edgecolor='black', alpha=0.8)
    ax.bar(x + width, scores_df["s3_type_balance"] * 0.3, width=width,
           label="Type balance (×0.3)", color="#FFE66D", edgecolor='black', alpha=0.8)

    # 총점 선
    ax.plot(x, scores_df["total_score"], 'ko-', markersize=10, linewidth=2,
            label="Total score", zorder=5)

    # 최적 표시
    best_idx = list(thrs).index(best_thr)
    best_score = scores_df.loc[scores_df["threshold"] == best_thr, "total_score"].values[0]
    ax.annotate(f"★ RECOMMENDED\nτ={best_thr} (score={best_score:.3f})",
                xy=(best_idx, best_score),
                xytext=(best_idx + 0.3, best_score + 0.08),
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={t}" for t in thrs])
    ax.set_ylabel("Score")
    ax.set_title("Analysis G: Composite Score for Threshold Recommendation",
                 fontweight='bold', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(scores_df["total_score"]) + 0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_recommendation.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig6_recommendation.png 저장 완료")


def plot_inclusion_rate(inclusion_df, out_dir):
    """[그림 7] 임계값별 유형별 포함률 라인 차트"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for a in ABUSE_ORDER:
        sub = inclusion_df[inclusion_df["abuse_type"] == a]
        ax.plot(sub["threshold"], sub["inclusion_rate"], 'o-',
                color=ABUSE_COLORS[a], linewidth=2, markersize=8,
                label=f"{ABUSE_LABEL_EN[a]} ({a})")

    ax.set_xlabel("Sub-threshold (τ)", fontsize=12)
    ax.set_ylabel("Inclusion Rate (%)", fontsize=12)
    ax.set_xticks(THRESHOLDS)
    ax.set_title("Analysis B': Inclusion Rate by Abuse Type and Threshold\n"
                 "(% of non-main, non-zero scores that pass the threshold)",
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_inclusion_rate.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig7_inclusion_rate.png 저장 완료")


# ═══════════════════════════════════════════════════════════════
#  5. 보고서 생성
# ═══════════════════════════════════════════════════════════════

def generate_report(basic_stats, inclusion_df, jaccard_df, scores_df,
                    best_thr, out_dir):
    """최종 분석 보고서를 텍스트 파일로 저장"""

    lines = []
    lines.append("=" * 72)
    lines.append("  SUB-THRESHOLD (τ) 민감도 분석 보고서")
    lines.append("  Sub-abuse Assignment Threshold Sensitivity Report")
    lines.append("=" * 72)
    lines.append("")

    # A. 기초 통계
    lines.append("━" * 60)
    lines.append("  [A] 기초 통계 (Basic Statistics)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  sub_threshold(τ)란?")
    lines.append("    - 주(main) 학대유형이 아닌 다른 학대유형의 점수가")
    lines.append("      τ 이상일 때 부(sub) 학대유형으로 인정하는 기준값")
    lines.append("")
    lines.append("  수식: sub = {a : a ≠ main  AND  S_a ≥ τ}")
    lines.append("")

    for _, row in basic_stats.iterrows():
        thr = int(row["threshold"])
        lines.append(f"  ┌── τ = {thr} ──────────────────────────────")
        lines.append(f"  │ sub abuse 보유 아동: {int(row['N_with_sub'])}명 ({row['pct_with_sub']:.1f}%)")
        lines.append(f"  │ 평균 sub 개수:       {row['avg_n_subs']:.3f}")
        lines.append(f"  │ 중복학대 아동:       {int(row['N_poly_victim'])}명 ({row['pct_poly']:.1f}%)")
        lines.append(f"  │ sub 개수 분포: 0개={int(row['n_sub_0'])}, 1개={int(row['n_sub_1'])}, "
                     f"2개={int(row['n_sub_2'])}, 3개={int(row['n_sub_3'])}")
        lines.append(f"  └────────────────────────────────────────")
        lines.append("")

    # B. 포함률
    lines.append("━" * 60)
    lines.append("  [B] 학대유형별 포함률 (Inclusion Rate)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  포함률 = (점수 ≥ τ인 비-주학대 유형 수) / (점수 > 0인 비-주학대 유형 수)")
    lines.append("")

    pivot = inclusion_df.pivot_table(
        index="abuse_en", columns="threshold", values="inclusion_rate"
    )
    lines.append(pivot.to_string())
    lines.append("")

    # E. Jaccard 안정성
    lines.append("━" * 60)
    lines.append("  [E] Jaccard 안정성 (Label-set Stability)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  J(τ₁,τ₂) = |L(τ₁) ∩ L(τ₂)| / |L(τ₁) ∪ L(τ₂)|")
    lines.append("  (1에 가까울수록 안정적)")
    lines.append("")

    for _, row in jaccard_df.iterrows():
        lines.append(f"  {row['pair']}: J = {row['mean_jaccard']:.4f} ± {row['std_jaccard']:.4f}, "
                     f"변화 아동 {int(row['n_changed'])}명 ({row['pct_changed']:.1f}%)")
    lines.append("")

    # G. 종합 추천
    lines.append("━" * 60)
    lines.append("  [G] 종합 추천 (Composite Recommendation)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  Score(τ) = 0.4·(적정비율) + 0.3·(안정성) + 0.3·(유형균형)")
    lines.append("")

    for _, row in scores_df.iterrows():
        thr = int(row["threshold"])
        marker = " ★ RECOMMENDED" if thr == best_thr else ""
        lines.append(f"  τ={thr}: "
                     f"적정비율={row['s1_adequate_ratio']:.3f}, "
                     f"안정성={row['s2_stability']:.3f}, "
                     f"유형균형={row['s3_type_balance']:.3f} "
                     f"→ 총점={row['total_score']:.3f}{marker}")

    lines.append("")
    lines.append("━" * 60)
    lines.append(f"  ★★★ 권장 임계값: τ = {best_thr} ★★★")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  근거:")
    lines.append(f"  - 종합 점수가 가장 높은 임계값: τ = {best_thr}")

    best_row = scores_df[scores_df["threshold"] == best_thr].iloc[0]
    lines.append(f"  - 적정 sub abuse 비율: {best_row['s1_adequate_ratio']:.3f}")
    lines.append(f"  - 인접 임계값과의 안정성: {best_row['s2_stability']:.3f}")
    lines.append(f"  - 학대유형 간 균형: {best_row['s3_type_balance']:.3f}")

    report_text = "\n".join(lines)

    report_path = os.path.join(out_dir, "sub_threshold_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[REPORT] {report_path} 저장 완료")
    print(report_text)

    return report_text


# ═══════════════════════════════════════════════════════════════
#  6. 메인 실행
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sub-threshold sensitivity analysis for sub-abuse assignment"
    )
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="JSON 데이터 디렉토리 (default: ./data)")
    parser.add_argument("--out_dir", type=str, default="./sub_threshold_output",
                        help="결과 저장 디렉토리 (default: ./sub_threshold_output)")
    parser.add_argument("--only_negative", action="store_true", default=True,
                        help="정서군='부정' 아동만 분석 (default: True)")
    parser.add_argument("--all_children", action="store_true",
                        help="모든 아동 포함 (--only_negative 무시)")

    args = parser.parse_args()

    if args.all_children:
        only_negative = False
    else:
        only_negative = args.only_negative

    os.makedirs(args.out_dir, exist_ok=True)

    print("╔" + "═" * 68 + "╗")
    print("║  SUB-THRESHOLD (τ) 민감도 분석                                    ║")
    print("║  Sub-abuse Assignment Threshold Sensitivity Analysis              ║")
    print("║  탐색 범위: τ ∈ {2, 3, 4, 5}                                     ║")
    print("╚" + "═" * 68 + "╝")

    # ── 데이터 로드 ──
    records = load_all_records(args.data_dir, only_negative=only_negative)

    if not records:
        print("[ERROR] 유효한 레코드가 없습니다. --data_dir를 확인하세요.")
        sys.exit(1)

    # ── 각 threshold에서 분류 수행 ──
    print("\n" + "─" * 60)
    print("  각 임계값(τ)에서 분류 수행...")
    print("─" * 60)

    all_results = {}
    for thr in THRESHOLDS:
        all_results[thr] = analyze_threshold(records, thr)
        n_sub = sum(1 for r in all_results[thr] if r["n_subs"] > 0)
        print(f"  τ={thr}: sub abuse 보유 {n_sub}명 / {len(records)}명")

    # ── 분석 A: 기초 통계 ──
    print("\n" + "▓" * 60)
    print("  [A] 기초 통계 계산")
    print("▓" * 60)
    basic_stats = compute_basic_stats(all_results)
    basic_stats.to_csv(os.path.join(args.out_dir, "A_basic_stats.csv"),
                       encoding="utf-8-sig", index=False)
    print(basic_stats.to_string(index=False))

    # ── 분석 B: 점수 분포 ──
    print("\n" + "▓" * 60)
    print("  [B] 학대유형별 점수 분포 계산")
    print("▓" * 60)
    inclusion_df, non_main_scores = compute_score_distribution(records)
    inclusion_df.to_csv(os.path.join(args.out_dir, "B_inclusion_rate.csv"),
                        encoding="utf-8-sig", index=False)
    print(inclusion_df.to_string(index=False))

    # ── 분석 C: 유형별 할당 ──
    print("\n" + "▓" * 60)
    print("  [C] 유형별 sub 할당 빈도")
    print("▓" * 60)
    type_alloc_df = compute_type_allocation(all_results)
    type_alloc_df.to_csv(os.path.join(args.out_dir, "C_type_allocation.csv"),
                         encoding="utf-8-sig", index=False)
    print(type_alloc_df.to_string(index=False))

    # ── 분석 D: 동시발생 ──
    print("\n" + "▓" * 60)
    print("  [D] Main × Sub 동시발생 행렬")
    print("▓" * 60)
    matrices = compute_cooccurrence(all_results)
    for thr, mat in matrices.items():
        mat.to_csv(os.path.join(args.out_dir, f"D_cooccurrence_thr{thr}.csv"),
                   encoding="utf-8-sig")
        print(f"\n  τ={thr}:")
        print(mat.to_string())

    # ── 분석 E: Jaccard 안정성 ──
    print("\n" + "▓" * 60)
    print("  [E] Jaccard 안정성")
    print("▓" * 60)
    jaccard_df = compute_jaccard_stability(all_results)
    jaccard_df.to_csv(os.path.join(args.out_dir, "E_jaccard_stability.csv"),
                      encoding="utf-8-sig", index=False)
    print(jaccard_df.to_string(index=False))

    # ── 분석 F: 토큰 풍부도 ──
    print("\n" + "▓" * 60)
    print("  [F] 토큰 풍부도 (CA 입력 proxy)")
    print("▓" * 60)
    richness_df = compute_token_richness_proxy(all_results)
    richness_df.to_csv(os.path.join(args.out_dir, "F_token_richness.csv"),
                       encoding="utf-8-sig", index=False)
    print(richness_df.to_string(index=False))

    # ── 분석 G: 종합 추천 ──
    print("\n" + "▓" * 60)
    print("  [G] 종합 추천")
    print("▓" * 60)
    scores_df, best_thr = recommend_threshold(basic_stats, jaccard_df, type_alloc_df)
    scores_df.to_csv(os.path.join(args.out_dir, "G_recommendation_scores.csv"),
                     encoding="utf-8-sig", index=False)
    print(scores_df.to_string(index=False))
    print(f"\n  ★ 권장 임계값: τ = {best_thr}")

    # ── 시각화 ──
    print("\n" + "▓" * 60)
    print("  시각화 생성")
    print("▓" * 60)
    plot_basic_stats(basic_stats, args.out_dir)
    plot_score_distribution(non_main_scores, args.out_dir)
    plot_type_allocation(type_alloc_df, args.out_dir)
    plot_cooccurrence(matrices, args.out_dir)
    plot_jaccard_stability(jaccard_df, args.out_dir)
    plot_recommendation(scores_df, best_thr, args.out_dir)
    plot_inclusion_rate(inclusion_df, args.out_dir)

    # ── 보고서 ──
    generate_report(basic_stats, inclusion_df, jaccard_df, scores_df,
                    best_thr, args.out_dir)

    print("\n" + "═" * 60)
    print(f"  ✅ 모든 분석 완료! 결과: {args.out_dir}/")
    print("═" * 60)


if __name__ == "__main__":
    main()