#!/usr/bin/env python3
"""
main_threshold_sensitivity.py
=============================
주(main) 학대유형 할당 임계값(main_threshold, θ) 민감도 분석

[목적]
  classify_abuse_main_sub()에서 주(main) 학대유형을 결정하는 규칙:

      nonzero = {a : S_a > θ}        ← 현재 θ = 6 하드코딩
      main = argmax_{a ∈ nonzero} S_a

  θ 값을 4, 5, 6, 7, 8로 변화시킬 때
  주 학대유형 할당이 어떻게 변하는지를 체계적으로 분석한다.

  ┌────────────────────────────────────────────────────────────┐
  │  핵심 질문: "main_threshold(θ)를 몇으로 설정해야             │
  │   CA 플롯에서 가장 의미 있는 main abuse 할당이 되는가?"     │
  └────────────────────────────────────────────────────────────┘

[분석 항목]
  A. 기초 통계: 임계값별 main abuse 보유 아동 수/비율
  B. 학대유형별 점수 분포: 전체 점수의 히스토그램 + θ 선
  C. 유형별 할당 변화: 어떤 학대유형이 main으로 가장 많이 선택되는가
  D. 유형 전환 행렬: θ 변화 시 main이 바뀌는 패턴 (X→Y 전이)
  E. 임계값 간 Cohen's κ 일치도: threshold 변화에 따른 라벨 일관성
  F. 임상 텍스트 보완율: θ를 높이면 임상 텍스트 fallback이 얼마나 증가하는가
  G. 코퍼스 크기 변화: θ 변화에 따른 ABUSE_NEG 코퍼스 아동 수 변화
  H. 권장 임계값 도출: 종합 지표 기반 자동 추천

[사용법]
  python main_threshold_sensitivity.py --data_dir ./data --out_dir ./main_threshold_output

[출력]
  - CSV: 모든 분석 결과 테이블
  - PNG: 시각화 8종
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
try:
    matplotlib.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']
except:
    pass

import warnings

warnings.filterwarnings("ignore")

# ── labels.py에서 정서군 분류 및 학대유형 분류 함수 import (3-tier fallback) ──
from abuse_pipeline.core.labels import classify_child_group
from abuse_pipeline.core.labels import classify_abuse_main_sub as _labels_classify_abuse_main_sub

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
from abuse_pipeline.core import common as _C

_SEVERITY_RANK = getattr(_C, "SEVERITY_RANK", None) or {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}

# ★ 탐색할 main threshold 후보
THRESHOLDS = [4, 5, 6, 7, 8]

VALENCE_ORDER = ["부정", "평범", "긍정"]


# ═══════════════════════════════════════════════════════════════
#  1. 핵심 함수: 학대유형 분류 (main_threshold 파라미터화)
# ═══════════════════════════════════════════════════════════════

def extract_abuse_scores(rec):
    """
    한 아동의 JSON record에서 학대유형별 점수를 추출한다.

    [수식]
    각 학대유형 a ∈ {성학대, 신체학대, 정서학대, 방임}에 대해:

        S_a = Σ_{i ∈ items(a)} score_i

    예시: 신체학대 관련 문항이 4개이고 각각 점수가 (3, 2, 0, 4)이면
        S_신체학대 = 3 + 2 + 0 + 4 = 9

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


def classify_abuse_with_main_threshold(rec, main_threshold=6, sub_threshold=4,
                                       use_clinical_text=True):
    """
    한 아동의 주(main) 학대유형과 부(sub) 학대유형을 분류한다.
    main_threshold(θ)를 파라미터로 받아 민감도 분석에 활용.

    ★ sub_threshold 기본값은 labels.py와 동일한 4를 사용.
    ★ θ=6일 때 labels.py의 classify_abuse_main_sub()과 동일한 결과를 보장.

    [알고리즘]
    ─────────────────────────────────────────────────────────
    1단계: 점수 추출
        각 학대유형 a에 대해 S_a = Σ score_i 를 계산

    2단계: 주(main) 학대유형 결정  ← ★ 이 단계가 main_threshold(θ)의 영향을 받음
        nonzero = {a : S_a > θ}
        main = argmax_{a ∈ nonzero} S_a
        (동점 시 심각도 위계: 성학대 > 신체학대 > 정서학대 > 방임)

    3단계: 부(sub) 학대유형 결정
        sub = {a : a ≠ main  AND  S_a ≥ sub_threshold}

    4단계: 임상 텍스트 보완 (use_clinical_text=True일 때)
        - 임상진단, 임상가 종합소견 텍스트에서 학대유형 키워드 추출
        - main이 None이면 → 텍스트에서 찾은 유형을 main으로
        - main이 이미 있으면 → 텍스트에서 찾은 다른 유형을 sub로
    ─────────────────────────────────────────────────────────

    [예시] θ = 6 (현재 기본값):
        S = {성학대: 0, 신체학대: 8, 정서학대: 5, 방임: 12}
        nonzero = {신체학대: 8, 방임: 12}  (> 6인 것만)
        → main = 방임 (12 > 8)

    [예시] θ = 4:
        같은 점수에서:
        nonzero = {신체학대: 8, 정서학대: 5, 방임: 12}  (> 4인 것만)
        → main = 방임 (여전히 12가 최대)
        → 하지만 정서학대도 nonzero에 포함 → 다른 아동에서 main 결정에 영향

    [예시] θ = 8:
        같은 점수에서:
        nonzero = {방임: 12}  (> 8인 것만, 신체학대 8은 > 8 아님)
        → main = 방임
        → 하지만 신체학대가 8인 다른 아동은 main을 잃을 수 있음!

    Parameters
    ----------
    rec : dict
        아동 JSON record
    main_threshold : int
        주 학대유형 인정 기준: S_a > θ 여야 main 후보
    sub_threshold : int
        부 학대유형 인정 최소 점수: S_a ≥ sub_threshold
    use_clinical_text : bool
        임상진단/종합소견 텍스트에서 학대유형 키워드 추출 여부

    Returns
    -------
    (main_abuse, sub_abuses, abuse_scores, clinical_used)
        clinical_used: 임상 텍스트가 main 결정에 사용되었는지 여부
    """
    abuse_scores = extract_abuse_scores(rec)

    # 2단계: main 결정 (S_a > θ)
    nonzero = {a: s for a, s in abuse_scores.items() if s > main_threshold}
    main = None
    clinical_used = False  # 임상 텍스트 보완 여부 추적

    if nonzero:
        main = max(nonzero,
                   key=lambda a: (nonzero[a], -_SEVERITY_RANK.get(a, 999)))

    # 3단계: sub 결정
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
                    clinical_used = True  # ★ 점수로는 결정 못하고 텍스트로 보완
                elif a != main:
                    subs.add(a)

    # 5단계: main이 없지만 sub가 있으면 sub 중 최고를 main으로 승격
    if main is None and subs:
        main = sorted(subs,
                      key=lambda x: (abuse_scores.get(x, 0),
                                     -_SEVERITY_RANK.get(x, 999)),
                      reverse=True)[0]
        subs.remove(main)
        clinical_used = True  # sub 승격도 fallback 메커니즘

    if main is None:
        return None, [], abuse_scores, clinical_used

    return main, sorted(subs, key=lambda x: _SEVERITY_RANK.get(x, 999)), abuse_scores, clinical_used


# classify_child_group()은 labels.py에서 import하여 사용
# → 파이프라인(pipeline.py)과 동일한 정서군 분류 로직 적용
# → 보호요인(protective index) 미세조정 포함 정교한 버전
#
# classify_abuse_main_sub()도 labels.py에서 import (_labels_classify_abuse_main_sub)
# → θ=6 기본값에서 labels.py와 동일한 ABUSE_NEG 산출 보장
# → classify_abuse_with_main_threshold()는 θ를 파라미터화한 확장 버전
#    (sub_threshold=4는 labels.py와 동일)


# ═══════════════════════════════════════════════════════════════
#  2. 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_all_records(data_dir, only_negative=True):
    """
    data_dir에서 모든 JSON 파일을 로드하고 아동별 정보를 추출한다.
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

        if isinstance(raw, list):
            recs = [x for x in raw if isinstance(x, dict)]
        elif isinstance(raw, dict):
            recs = [raw]
        else:
            continue

        for rec in recs:
            info = rec.get("info", {})
            doc_id = info.get("ID") or info.get("id") or os.path.basename(path)

            valence = classify_child_group(rec)
            if only_negative and valence != "부정":
                continue

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
    """특정 main_threshold(θ)에서의 분류 결과를 반환"""
    results = []
    for r in records:
        main, subs, scores, clin_used = classify_abuse_with_main_threshold(
            r["rec"], main_threshold=threshold
        )
        results.append({
            "doc_id": r["doc_id"],
            "main": main,
            "subs": subs,
            "n_subs": len(subs),
            "abuse_scores": scores,
            "clinical_used": clin_used,
            "has_main": main is not None,
        })
    return results


# ─── A. 기초 통계 ────────────────────────────────────────────

def compute_basic_stats(all_results, n_total_neg):
    """
    [분석 A] 각 임계값별 기초 통계

    측정 항목:
    - N_total: 정서군='부정' 전체 아동 수
    - N_with_main: main abuse가 할당된 아동 수 (= ABUSE_NEG 코퍼스 크기)
    - N_no_main: main abuse가 없는 아동 수 (학대 증거 부족)
    - 유형별 main 분포
    """
    rows = []
    for thr, results in all_results.items():
        n_with_main = sum(1 for r in results if r["has_main"])
        n_no_main = sum(1 for r in results if not r["has_main"])
        n_clinical = sum(1 for r in results if r["clinical_used"])

        # 유형별 main 빈도
        main_counter = Counter(r["main"] for r in results if r["main"])

        row = {
            "threshold_θ": thr,
            "N_total_neg": n_total_neg,
            "N_with_main": n_with_main,
            "pct_with_main": round(n_with_main / n_total_neg * 100, 1) if n_total_neg else 0,
            "N_no_main": n_no_main,
            "pct_no_main": round(n_no_main / n_total_neg * 100, 1) if n_total_neg else 0,
            "N_clinical_fallback": n_clinical,
            "pct_clinical": round(n_clinical / n_total_neg * 100, 1) if n_total_neg else 0,
        }
        for a in ABUSE_ORDER:
            row[f"main_{ABUSE_LABEL_EN[a]}"] = main_counter.get(a, 0)

        rows.append(row)

    return pd.DataFrame(rows)


# ─── B. 학대유형별 점수 분포 ─────────────────────────────────

def compute_score_distribution(records):
    """
    [분석 B] 학대유형별 합산 점수의 전체 분포

    각 아동 × 학대유형의 S_a 값을 모아서 히스토그램을 그린다.
    θ 선을 함께 표시하여 "이 θ에서 몇 명이 main 후보에서 탈락하는가"를 시각화.
    """
    all_scores = {a: [] for a in ABUSE_ORDER}

    for r in records:
        scores = r["abuse_scores"]
        for a in ABUSE_ORDER:
            if scores[a] > 0:  # 0점은 해당 학대 아예 없음
                all_scores[a].append(scores[a])

    return all_scores


# ─── C. 유형별 할당 변화 ─────────────────────────────────────

def compute_type_allocation(all_results):
    """
    [분석 C] 각 임계값에서 어떤 학대유형이 main으로 할당되는 빈도

    [수식]
    N_main(a, θ) = |{아동 i : main_i(θ) = a}|

    → θ를 올리면 낮은 점수 아동들이 main을 잃고,
      유형별 비율이 달라질 수 있다.
    """
    rows = []
    for thr, results in all_results.items():
        main_counter = Counter(r["main"] for r in results if r["main"])
        total_with_main = sum(main_counter.values())

        for a in ABUSE_ORDER:
            n = main_counter.get(a, 0)
            rows.append({
                "threshold_θ": thr,
                "abuse_type": a,
                "abuse_en": ABUSE_LABEL_EN[a],
                "N_as_main": n,
                "pct_of_main": round(n / total_with_main * 100, 1) if total_with_main else 0,
            })

    return pd.DataFrame(rows)


# ─── D. 유형 전환 행렬 ───────────────────────────────────────

def compute_transition_matrices(all_results):
    """
    [분석 D] 인접 θ 변화 시 main 라벨이 어떻게 전이되는가

    전이 행렬 T(θ₁ → θ₂):
        T[X][Y] = |{아동 i : main_i(θ₁) = X  AND  main_i(θ₂) = Y}|

    예시:
        T[정서학대][None] = 15  → θ를 올리니 정서학대였던 15명이 main을 잃음
        T[정서학대][방임] = 3   → θ를 올리니 3명의 main이 정서학대→방임으로 전환
    """
    transitions = {}
    labels = ABUSE_ORDER + [None]
    label_names = ABUSE_ORDER + ["None(no main)"]

    for t1, t2 in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
        r1 = all_results[t1]
        r2 = all_results[t2]

        mat = pd.DataFrame(0, index=label_names, columns=label_names)
        for i in range(len(r1)):
            m1 = r1[i]["main"]
            m2 = r2[i]["main"]
            row_name = m1 if m1 else "None(no main)"
            col_name = m2 if m2 else "None(no main)"
            mat.loc[row_name, col_name] += 1

        transitions[(t1, t2)] = mat

    return transitions


# ─── E. Cohen's κ 일치도 ─────────────────────────────────────

def compute_kappa_agreement(all_results):
    """
    [분석 E] 모든 θ 쌍 간 Cohen's κ (가중 일치도)

    [수식]
    κ = (p₀ - pₑ) / (1 - pₑ)

    여기서:
        p₀ = 실제 일치율 = |{i : main_i(θ₁) = main_i(θ₂)}| / N
        pₑ = 우연 일치율 (기대값)

    κ ≈ 1: 거의 완벽한 일치
    κ ≈ 0.8: 상당한 일치 (substantial)
    κ ≈ 0.6: 보통 수준 일치 (moderate)
    κ < 0.4: 낮은 일치

    [예시]
    θ=6 vs θ=7: 1970명 중 1950명이 같은 main → p₀ = 1950/1970 ≈ 0.99
    """
    rows = []
    for t1, t2 in combinations(THRESHOLDS, 2):
        r1 = all_results[t1]
        r2 = all_results[t2]

        labels1 = [r["main"] if r["main"] else "None" for r in r1]
        labels2 = [r["main"] if r["main"] else "None" for r in r2]

        # 직접 계산 (sklearn 없이)
        n = len(labels1)
        agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
        p0 = agree / n

        # 우연 일치율
        all_labels = sorted(set(labels1) | set(labels2))
        pe = 0
        for lab in all_labels:
            p1 = sum(1 for x in labels1 if x == lab) / n
            p2 = sum(1 for x in labels2 if x == lab) / n
            pe += p1 * p2

        kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 1.0

        rows.append({
            "pair": f"θ={t1} vs θ={t2}",
            "θ_1": t1,
            "θ_2": t2,
            "N": n,
            "N_agree": agree,
            "p0_observed": round(p0, 4),
            "pe_expected": round(pe, 4),
            "cohen_kappa": round(kappa, 4),
        })

    return pd.DataFrame(rows)


# ─── F. 임상 텍스트 보완율 ───────────────────────────────────

def compute_clinical_fallback_rate(all_results):
    """
    [분석 F] θ를 높이면 점수 기반 main 결정 실패율이 올라가고,
    임상 텍스트 보완(fallback) 의존도가 증가하는지 측정.

    [수식]
    임상보완율(θ) = |{i : clinical_used_i(θ) = True}| / N

    θ를 올릴수록 이 비율이 올라가면, "점수만으로는 학대유형을 결정할 수 없는
    아동이 늘어난다"는 것을 의미 → 그 θ는 너무 높다는 신호.
    """
    rows = []
    for thr, results in all_results.items():
        n_total = len(results)
        n_clinical = sum(1 for r in results if r["clinical_used"])
        n_score_only = sum(1 for r in results if r["has_main"] and not r["clinical_used"])
        n_no_main = sum(1 for r in results if not r["has_main"])

        rows.append({
            "threshold_θ": thr,
            "N_total": n_total,
            "N_score_based": n_score_only,
            "pct_score_based": round(n_score_only / n_total * 100, 1),
            "N_clinical_fallback": n_clinical,
            "pct_clinical": round(n_clinical / n_total * 100, 1),
            "N_no_main": n_no_main,
            "pct_no_main": round(n_no_main / n_total * 100, 1),
        })

    return pd.DataFrame(rows)


# ─── G. 코퍼스 크기 변화 ─────────────────────────────────────

def compute_corpus_size(all_results):
    """
    [분석 G] ABUSE_NEG 코퍼스 크기 변화

    ABUSE_NEG = {아동 i : 정서군='부정' AND main_abuse ≠ None}

    θ를 올리면 main이 None이 되는 아동이 늘어나 코퍼스가 줄어든다.
    CA 분석의 통계적 검정력(power)에 직접 영향.
    """
    rows = []
    for thr, results in all_results.items():
        n_abuse_neg = sum(1 for r in results if r["has_main"])
        type_counts = Counter(r["main"] for r in results if r["main"])

        row = {
            "threshold_θ": thr,
            "ABUSE_NEG_size": n_abuse_neg,
        }
        for a in ABUSE_ORDER:
            row[f"n_{ABUSE_LABEL_EN[a]}"] = type_counts.get(a, 0)
        rows.append(row)

    return pd.DataFrame(rows)


# ─── H. 종합 권장 임계값 도출 ────────────────────────────────

def recommend_threshold(basic_stats, kappa_df, type_alloc_df, clinical_df):
    """
    [분석 H] 종합 지표 기반 최적 θ 추천

    [기준]
    1. 코퍼스 보존율: ABUSE_NEG 크기가 클수록 좋음 (통계적 검정력)
    2. 라벨 일관성: 인접 θ와의 Cohen's κ가 높을수록 좋음
    3. 유형 균형: 4개 학대유형이 골고루 분포할수록 좋음
    4. 낮은 임상 의존도: 점수만으로 결정되는 비율이 높을수록 좋음

    [점수 산출]
    Score(θ) = w₁·(코퍼스 보존) + w₂·(라벨 일관성) + w₃·(유형 균형) + w₄·(점수 기반율)

    w₁ = 0.30, w₂ = 0.25, w₃ = 0.20, w₄ = 0.25
    """
    scores = {}

    # 정규화를 위한 min/max
    corpus_sizes = basic_stats["N_with_main"].values
    max_corpus = max(corpus_sizes) if len(corpus_sizes) > 0 else 1

    for thr in THRESHOLDS:
        row = basic_stats[basic_stats["threshold_θ"] == thr].iloc[0]
        clin_row = clinical_df[clinical_df["threshold_θ"] == thr].iloc[0]

        # 1. 코퍼스 보존율 (0~1)
        s1 = row["N_with_main"] / max_corpus if max_corpus > 0 else 0

        # 2. 라벨 일관성 (인접 θ와의 κ 평균)
        kappa_rows = kappa_df[
            (kappa_df["θ_1"] == thr) | (kappa_df["θ_2"] == thr)
            ]
        s2 = kappa_rows["cohen_kappa"].mean() if not kappa_rows.empty else 0.5

        # 3. 유형 균형 (Shannon entropy 정규화)
        type_row = type_alloc_df[type_alloc_df["threshold_θ"] == thr]
        counts = type_row["N_as_main"].values
        total = counts.sum()
        if total > 0:
            p = counts / total
            entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
            max_entropy = np.log2(len(ABUSE_ORDER))
            s3 = entropy / max_entropy if max_entropy > 0 else 0
        else:
            s3 = 0

        # 4. 점수 기반율 (임상 텍스트 fallback 낮을수록 좋음)
        s4 = clin_row["pct_score_based"] / 100.0

        total_score = 0.30 * s1 + 0.25 * s2 + 0.20 * s3 + 0.25 * s4
        scores[thr] = {
            "threshold_θ": thr,
            "s1_corpus_preserve": round(s1, 4),
            "s2_label_consistency": round(s2, 4),
            "s3_type_balance": round(s3, 4),
            "s4_score_based_rate": round(s4, 4),
            "total_score": round(total_score, 4),
        }

    scores_df = pd.DataFrame(scores.values())
    best = scores_df.loc[scores_df["total_score"].idxmax()]

    return scores_df, int(best["threshold_θ"])


# ═══════════════════════════════════════════════════════════════
#  4. 시각화 함수들
# ═══════════════════════════════════════════════════════════════

def plot_basic_stats(basic_stats, out_dir):
    """[그림 1] 임계값별 코퍼스 크기 및 main 보유율"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    thrs = basic_stats["threshold_θ"].values
    x = np.arange(len(thrs))

    # (a) ABUSE_NEG 코퍼스 크기
    ax = axes[0]
    bars = ax.bar(x, basic_stats["N_with_main"], color="#4ECDC4", edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in thrs])
    ax.set_ylabel("Number of children (ABUSE_NEG)")
    ax.set_title("(a) ABUSE_NEG Corpus Size")
    for bar, val in zip(bars, basic_stats["N_with_main"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{int(val)}", ha='center', va='bottom', fontsize=10)

    # (b) main 보유율
    ax = axes[1]
    bars = ax.bar(x, basic_stats["pct_with_main"], color="#FF6B6B", edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in thrs])
    ax.set_ylabel("% of NEG children with main abuse")
    ax.set_title("(b) Main Abuse Assignment Rate")
    for bar, val in zip(bars, basic_stats["pct_with_main"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=10)

    # (c) 임상 텍스트 보완율
    ax = axes[2]
    bars = ax.bar(x, basic_stats["pct_clinical"], color="#FFE66D", edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in thrs])
    ax.set_ylabel("% clinical text fallback")
    ax.set_title("(c) Clinical Text Fallback Rate")
    for bar, val in zip(bars, basic_stats["pct_clinical"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=10)

    plt.suptitle("Analysis A: Basic Statistics by Main-threshold (θ)\n"
                 "main = argmax{a : Sₐ > θ}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_basic_stats_main.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig1_basic_stats_main.png 저장 완료")


def plot_score_distribution(all_scores, out_dir):
    """[그림 2] 학대유형별 점수 히스토그램 + θ 선"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors_line = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for idx, a in enumerate(ABUSE_ORDER):
        ax = axes[idx]
        vals = all_scores[a]

        if vals:
            max_val = max(vals) + 1
            bins = np.arange(0.5, max_val + 1, 1)
            ax.hist(vals, bins=bins, color=ABUSE_COLORS[a], alpha=0.7, edgecolor='black')

            for i, thr in enumerate(THRESHOLDS):
                ax.axvline(thr + 0.5, color=colors_line[i], linestyle='--', linewidth=2,
                           label=f'θ={thr} (> {thr})')

        ax.set_xlabel("Abuse Score (Sₐ)")
        ax.set_ylabel("Count")
        ax.set_title(f"{ABUSE_LABEL_EN[a]} ({a})")
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle("Analysis B: Abuse Score Distribution by Type\n"
                 "(Dashed lines = main_threshold candidates; main requires Sₐ > θ)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_score_distribution_main.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig2_score_distribution_main.png 저장 완료")


def plot_type_allocation(type_alloc_df, out_dir):
    """[그림 3] 학대유형별 main 할당 빈도 (Grouped Bar)"""
    fig, ax = plt.subplots(figsize=(11, 6))

    n_types = len(ABUSE_ORDER)
    n_thresholds = len(THRESHOLDS)
    width = 0.15
    x = np.arange(n_types)

    for i, thr in enumerate(THRESHOLDS):
        vals = type_alloc_df[type_alloc_df["threshold_θ"] == thr]["N_as_main"].values
        offset = (i - n_thresholds / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=f"θ={thr}",
                      edgecolor='black', alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(v), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ABUSE_LABEL_EN[a]}\n({a})" for a in ABUSE_ORDER])
    ax.set_ylabel("Number of children assigned as main abuse")
    ax.set_title("Analysis C: Main Abuse Type Allocation by Threshold (θ)", fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_type_allocation_main.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig3_type_allocation_main.png 저장 완료")


def plot_transitions(transitions, out_dir):
    """[그림 4] Main 라벨 전이 히트맵"""
    n_pairs = len(transitions)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5.5))
    if n_pairs == 1:
        axes = [axes]

    for idx, ((t1, t2), mat) in enumerate(transitions.items()):
        ax = axes[idx]
        # 비대각선 요소만 색칠 (전이를 강조)
        data = mat.values.astype(float)
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        labels_display = [ABUSE_LABEL_EN.get(a, a) for a in mat.index]
        ax.set_xticks(range(len(labels_display)))
        ax.set_yticks(range(len(labels_display)))
        ax.set_xticklabels(labels_display, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels_display, fontsize=8)
        ax.set_xlabel(f"main at θ={t2}")
        ax.set_ylabel(f"main at θ={t1}")
        ax.set_title(f"θ={t1} → θ={t2}", fontweight='bold')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = int(data[i, j])
                if val > 0:
                    color = 'white' if val > data.max() * 0.6 else 'black'
                    weight = 'bold' if i != j else 'normal'
                    ax.text(j, i, str(val), ha='center', va='center',
                            fontsize=9, fontweight=weight, color=color)

    plt.suptitle("Analysis D: Main Label Transition Matrix\n"
                 "(Off-diagonal = label changes when θ increases)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_transition_main.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig4_transition_main.png 저장 완료")


def plot_kappa(kappa_df, out_dir):
    """[그림 5] Cohen's κ 일치도 히트맵"""
    fig, ax = plt.subplots(figsize=(8, 6))

    n = len(THRESHOLDS)
    mat = pd.DataFrame(1.0, index=THRESHOLDS, columns=THRESHOLDS)
    for _, row in kappa_df.iterrows():
        mat.loc[row["θ_1"], row["θ_2"]] = row["cohen_kappa"]
        mat.loc[row["θ_2"], row["θ_1"]] = row["cohen_kappa"]

    im = ax.imshow(mat.values, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"θ={t}" for t in THRESHOLDS])
    ax.set_yticklabels([f"θ={t}" for t in THRESHOLDS])

    for i in range(n):
        for j in range(n):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if val < 0.7 else 'black')

    plt.colorbar(im, ax=ax, label="Cohen's κ")
    ax.set_title("Analysis E: Pairwise Cohen's κ Agreement\n"
                 "Between Main-threshold Settings", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_kappa_main.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig5_kappa_main.png 저장 완료")


def plot_clinical_fallback(clinical_df, out_dir):
    """[그림 6] 점수 기반 vs 임상 보완 vs 미할당 스택 바"""
    fig, ax = plt.subplots(figsize=(10, 6))

    thrs = clinical_df["threshold_θ"].values
    x = np.arange(len(thrs))
    width = 0.5

    score = clinical_df["pct_score_based"].values
    clin = clinical_df["pct_clinical"].values
    none = clinical_df["pct_no_main"].values

    ax.bar(x, score, width, label="Score-based main", color="#2ca02c", edgecolor='black')
    ax.bar(x, clin, width, bottom=score, label="Clinical text fallback", color="#ff7f0e", edgecolor='black')
    ax.bar(x, none, width, bottom=score + clin, label="No main (excluded)", color="#d62728", edgecolor='black',
           alpha=0.6)

    for i in range(len(thrs)):
        # score-based %
        ax.text(x[i], score[i] / 2, f"{score[i]:.1f}%",
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        # clinical %
        if clin[i] > 2:
            ax.text(x[i], score[i] + clin[i] / 2, f"{clin[i]:.1f}%",
                    ha='center', va='center', fontsize=9, fontweight='bold')
        # none %
        if none[i] > 2:
            ax.text(x[i], score[i] + clin[i] + none[i] / 2, f"{none[i]:.1f}%",
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in thrs])
    ax.set_ylabel("% of children")
    ax.set_title("Analysis F: Main Abuse Assignment Method by Threshold\n"
                 "(Score-based vs Clinical fallback vs Excluded)", fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_clinical_fallback.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig6_clinical_fallback.png 저장 완료")


def plot_corpus_size(corpus_df, out_dir):
    """[그림 7] θ별 ABUSE_NEG 코퍼스 유형 구성 스택 바"""
    fig, ax = plt.subplots(figsize=(10, 6))

    thrs = corpus_df["threshold_θ"].values
    x = np.arange(len(thrs))
    width = 0.5

    bottom = np.zeros(len(thrs))
    for a in ABUSE_ORDER:
        vals = corpus_df[f"n_{ABUSE_LABEL_EN[a]}"].values
        ax.bar(x, vals, width, bottom=bottom, label=f"{ABUSE_LABEL_EN[a]}",
               color=ABUSE_COLORS[a], edgecolor='black')
        for i, v in enumerate(vals):
            if v > 20:
                ax.text(x[i], bottom[i] + v / 2, str(int(v)),
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        bottom += vals

    for i, total in enumerate(corpus_df["ABUSE_NEG_size"].values):
        ax.text(x[i], total + 10, f"N={int(total)}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in thrs])
    ax.set_ylabel("Number of children")
    ax.set_title("Analysis G: ABUSE_NEG Corpus Composition by Threshold",
                 fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_corpus_composition.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig7_corpus_composition.png 저장 완료")


def plot_recommendation(scores_df, best_thr, out_dir):
    """[그림 8] 종합 점수 + 추천 임계값"""
    fig, ax = plt.subplots(figsize=(11, 6))

    thrs = scores_df["threshold_θ"].values
    x = np.arange(len(thrs))
    width = 0.17

    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF"]
    labels = ["Corpus preserve (×0.30)", "Label consistency (×0.25)",
              "Type balance (×0.20)", "Score-based rate (×0.25)"]
    fields = ["s1_corpus_preserve", "s2_label_consistency",
              "s3_type_balance", "s4_score_based_rate"]
    weights = [0.30, 0.25, 0.20, 0.25]

    for i, (field, label, color, w) in enumerate(zip(fields, labels, colors, weights)):
        offset = (i - 2 + 0.5) * width
        vals = scores_df[field].values * w
        ax.bar(x + offset, vals, width=width, label=label,
               color=color, edgecolor='black', alpha=0.85)

    ax.plot(x, scores_df["total_score"], 'ko-', markersize=10, linewidth=2,
            label="Total score", zorder=5)

    best_idx = list(thrs).index(best_thr)
    best_score = scores_df.loc[scores_df["threshold_θ"] == best_thr, "total_score"].values[0]
    ax.annotate(f"★ RECOMMENDED\nθ={best_thr} (score={best_score:.4f})",
                xy=(best_idx, best_score),
                xytext=(best_idx + 0.5, best_score + 0.05),
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in thrs])
    ax.set_ylabel("Score")
    ax.set_title("Analysis H: Composite Score for Main-threshold Recommendation",
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0, max(scores_df["total_score"]) + 0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_recommendation_main.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[PLOT] fig8_recommendation_main.png 저장 완료")


# ═══════════════════════════════════════════════════════════════
#  5. 보고서 생성
# ═══════════════════════════════════════════════════════════════

def generate_report(basic_stats, kappa_df, clinical_df, corpus_df,
                    scores_df, best_thr, transitions, out_dir):
    """최종 분석 보고서를 텍스트 파일로 저장"""

    lines = []
    lines.append("=" * 72)
    lines.append("  MAIN-THRESHOLD (θ) 민감도 분석 보고서")
    lines.append("  Main Abuse Assignment Threshold Sensitivity Report")
    lines.append("=" * 72)
    lines.append("")

    # ── A. 기초 통계 ──
    lines.append("━" * 60)
    lines.append("  [A] 기초 통계 (Basic Statistics)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  main_threshold(θ)란?")
    lines.append("    - 학대유형별 합산 점수 S_a가 θ를 초과(>)해야")
    lines.append("      주(main) 학대유형 후보로 인정하는 기준값")
    lines.append("    - 현재 코드: nonzero = {a : S_a > 6}  (θ=6 하드코딩)")
    lines.append("")
    lines.append("  수식: main = argmax_{a : S_a > θ} S_a")
    lines.append("")

    for _, row in basic_stats.iterrows():
        thr = int(row["threshold_θ"])
        lines.append(f"  ┌── θ = {thr} {'(현재 기본값)' if thr == 6 else ''} ──────────────────────")
        lines.append(f"  │ main abuse 보유 아동: {int(row['N_with_main'])}명 ({row['pct_with_main']:.1f}%)")
        lines.append(f"  │ main 미할당 아동:    {int(row['N_no_main'])}명 ({row['pct_no_main']:.1f}%)")
        lines.append(f"  │ 임상 텍스트 보완:    {int(row['N_clinical_fallback'])}명 ({row['pct_clinical']:.1f}%)")
        for a in ABUSE_ORDER:
            n = int(row[f"main_{ABUSE_LABEL_EN[a]}"])
            lines.append(f"  │   {ABUSE_LABEL_EN[a]:>10s} ({a}): {n}명")
        lines.append(f"  └──────────────────────────────────────────────")
        lines.append("")

    # ── D. 전이 행렬 ──
    lines.append("━" * 60)
    lines.append("  [D] 라벨 전이 행렬 (Label Transition)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  θ를 1 올릴 때 main 라벨이 어떻게 변하는지 보여준다.")
    lines.append("  대각선: 유지, 비대각선: 전환 (특히 →None 은 main 상실)")
    lines.append("")

    for (t1, t2), mat in transitions.items():
        lines.append(f"  ── θ={t1} → θ={t2} ──")
        # 비대각선 합계 (변화한 아동 수)
        off_diag = mat.values.sum() - np.trace(mat.values)
        lines.append(f"  전이한 아동: {int(off_diag)}명")
        # 주요 전이만 표시
        for i, row_name in enumerate(mat.index):
            for j, col_name in enumerate(mat.columns):
                v = int(mat.values[i, j])
                if v > 0 and i != j:
                    lines.append(f"    {row_name} → {col_name}: {v}명")
        lines.append("")

    # ── E. Cohen's κ ──
    lines.append("━" * 60)
    lines.append("  [E] Cohen's κ 일치도")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  κ = (p₀ - pₑ) / (1 - pₑ)")
    lines.append("  ≈1: 완벽 일치, ≈0.8: 상당한 일치, ≈0.6: 보통")
    lines.append("")

    for _, row in kappa_df.iterrows():
        lines.append(f"  {row['pair']}: κ = {row['cohen_kappa']:.4f} "
                     f"(일치 {int(row['N_agree'])}/{int(row['N'])}명, p₀={row['p0_observed']:.4f})")
    lines.append("")

    # ── F. 임상 텍스트 보완율 ──
    lines.append("━" * 60)
    lines.append("  [F] 임상 텍스트 보완율 (Clinical Fallback)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  θ를 높이면 점수만으로 main을 결정할 수 없는 아동이 늘어나")
    lines.append("  임상진단/종합소견 텍스트에 의존하는 비율이 증가한다.")
    lines.append("")

    for _, row in clinical_df.iterrows():
        thr = int(row["threshold_θ"])
        lines.append(f"  θ={thr}: 점수기반 {row['pct_score_based']:.1f}% | "
                     f"임상보완 {row['pct_clinical']:.1f}% | "
                     f"미할당 {row['pct_no_main']:.1f}%")
    lines.append("")

    # ── G. 코퍼스 크기 ──
    lines.append("━" * 60)
    lines.append("  [G] ABUSE_NEG 코퍼스 크기 변화")
    lines.append("━" * 60)
    lines.append("")

    for _, row in corpus_df.iterrows():
        thr = int(row["threshold_θ"])
        n = int(row["ABUSE_NEG_size"])
        parts = [f"{ABUSE_LABEL_EN[a]}={int(row[f'n_{ABUSE_LABEL_EN[a]}'])}" for a in ABUSE_ORDER]
        lines.append(f"  θ={thr}: N={n} ({', '.join(parts)})")
    lines.append("")

    # ── H. 종합 추천 ──
    lines.append("━" * 60)
    lines.append("  [H] 종합 추천 (Composite Recommendation)")
    lines.append("━" * 60)
    lines.append("")
    lines.append("  Score(θ) = 0.30·(코퍼스보존) + 0.25·(라벨일관성)")
    lines.append("           + 0.20·(유형균형) + 0.25·(점수기반율)")
    lines.append("")

    for _, row in scores_df.iterrows():
        thr = int(row["threshold_θ"])
        marker = " ★ RECOMMENDED" if thr == best_thr else ""
        if thr == 6:
            marker += " (현재 기본값)"
        lines.append(f"  θ={thr}: "
                     f"보존={row['s1_corpus_preserve']:.4f}, "
                     f"일관성={row['s2_label_consistency']:.4f}, "
                     f"균형={row['s3_type_balance']:.4f}, "
                     f"점수율={row['s4_score_based_rate']:.4f} "
                     f"→ 총점={row['total_score']:.4f}{marker}")

    lines.append("")
    lines.append("━" * 60)
    lines.append(f"  ★★★ 권장 임계값: θ = {best_thr} ★★★")
    lines.append("━" * 60)
    lines.append("")

    best_row = scores_df[scores_df["threshold_θ"] == best_thr].iloc[0]
    lines.append("  근거:")
    lines.append(f"  - 종합 점수가 가장 높은 임계값: θ = {best_thr}")
    lines.append(f"  - 코퍼스 보존율: {best_row['s1_corpus_preserve']:.4f}")
    lines.append(f"  - 라벨 일관성(κ): {best_row['s2_label_consistency']:.4f}")
    lines.append(f"  - 유형 균형: {best_row['s3_type_balance']:.4f}")
    lines.append(f"  - 점수 기반율: {best_row['s4_score_based_rate']:.4f}")

    if best_thr == 6:
        lines.append("")
        lines.append("  → 현재 코드의 하드코딩 값(θ=6)이 최적으로 확인됨.")
        lines.append("    추가 변경 불필요.")

    report_text = "\n".join(lines)

    report_path = os.path.join(out_dir, "main_threshold_report.txt")
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
        description="Main-threshold sensitivity analysis for main abuse assignment"
    )
    parser.add_argument("--data_dir", type=str,
                        default=r"C:\Users\todtj\PycharmProjects\Childeren\data",
                        help="JSON 데이터 디렉토리")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="결과 저장 디렉토리 (default: configure_output_dirs 기반)")
    parser.add_argument("--only_negative", action="store_true", default=True,
                        help="정서군='부정' 아동만 분석 (default: True)")
    parser.add_argument("--all_children", action="store_true",
                        help="모든 아동 포함 (--only_negative 무시)")

    args = parser.parse_args()

    if args.all_children:
        only_negative = False
    else:
        only_negative = args.only_negative

    if args.out_dir is None:
        subset = "NEG_ONLY" if only_negative else "ALL"
        if _C is not None:
            _C.configure_output_dirs(subset_name=subset)
            args.out_dir = _C.MAIN_THRESHOLD_DIR
        else:
            args.out_dir = "./main_threshold_output"
    os.makedirs(args.out_dir, exist_ok=True)

    print("╔" + "═" * 68 + "╗")
    print("║  MAIN-THRESHOLD (θ) 민감도 분석                                   ║")
    print("║  Main Abuse Assignment Threshold Sensitivity Analysis             ║")
    print("║  탐색 범위: θ ∈ {4, 5, 6, 7, 8}                                  ║")
    print("║  현재 기본값: θ = 6  (코드: s > 6)                                ║")
    print("╚" + "═" * 68 + "╝")

    # ── 데이터 로드 ──
    records = load_all_records(args.data_dir, only_negative=only_negative)

    if not records:
        print("[ERROR] 유효한 레코드가 없습니다. --data_dir를 확인하세요.")
        sys.exit(1)

    n_total_neg = len(records)

    # ── 각 threshold에서 분류 수행 ──
    print("\n" + "─" * 60)
    print("  각 임계값(θ)에서 분류 수행...")
    print("  규칙: main = argmax{a : S_a > θ}")
    print("─" * 60)

    all_results = {}
    for thr in THRESHOLDS:
        all_results[thr] = analyze_threshold(records, thr)
        n_main = sum(1 for r in all_results[thr] if r["has_main"])
        n_clin = sum(1 for r in all_results[thr] if r["clinical_used"])
        marker = " ← 현재 기본값" if thr == 6 else ""
        print(f"  θ={thr}: main 보유 {n_main}명, 임상보완 {n_clin}명 / {n_total_neg}명{marker}")

    # ── 분석 A: 기초 통계 ──
    print("\n" + "▓" * 60)
    print("  [A] 기초 통계 계산")
    print("▓" * 60)
    basic_stats = compute_basic_stats(all_results, n_total_neg)
    basic_stats.to_csv(os.path.join(args.out_dir, "A_basic_stats_main.csv"),
                       encoding="utf-8-sig", index=False)
    print(basic_stats.to_string(index=False))

    # ── 분석 B: 점수 분포 ──
    print("\n" + "▓" * 60)
    print("  [B] 학대유형별 점수 분포")
    print("▓" * 60)
    all_scores = compute_score_distribution(records)
    for a in ABUSE_ORDER:
        vals = all_scores[a]
        if vals:
            print(f"  {ABUSE_LABEL_EN[a]}: n={len(vals)}, "
                  f"mean={np.mean(vals):.1f}, median={np.median(vals):.0f}, "
                  f"min={min(vals)}, max={max(vals)}")

    # ── 분석 C: 유형별 할당 ──
    print("\n" + "▓" * 60)
    print("  [C] 유형별 main 할당 빈도")
    print("▓" * 60)
    type_alloc_df = compute_type_allocation(all_results)
    type_alloc_df.to_csv(os.path.join(args.out_dir, "C_type_allocation_main.csv"),
                         encoding="utf-8-sig", index=False)
    print(type_alloc_df.to_string(index=False))

    # ── 분석 D: 전이 행렬 ──
    print("\n" + "▓" * 60)
    print("  [D] Main 라벨 전이 행렬")
    print("▓" * 60)
    transitions = compute_transition_matrices(all_results)
    for (t1, t2), mat in transitions.items():
        mat.to_csv(os.path.join(args.out_dir, f"D_transition_theta{t1}_to_{t2}.csv"),
                   encoding="utf-8-sig")
        off_diag = int(mat.values.sum() - np.trace(mat.values))
        print(f"\n  θ={t1} → θ={t2} (전이 {off_diag}명):")
        print(mat.to_string())

    # ── 분석 E: Cohen's κ ──
    print("\n" + "▓" * 60)
    print("  [E] Cohen's κ 일치도")
    print("▓" * 60)
    kappa_df = compute_kappa_agreement(all_results)
    kappa_df.to_csv(os.path.join(args.out_dir, "E_kappa_agreement.csv"),
                    encoding="utf-8-sig", index=False)
    print(kappa_df.to_string(index=False))

    # ── 분석 F: 임상 텍스트 보완율 ──
    print("\n" + "▓" * 60)
    print("  [F] 임상 텍스트 보완율")
    print("▓" * 60)
    clinical_df = compute_clinical_fallback_rate(all_results)
    clinical_df.to_csv(os.path.join(args.out_dir, "F_clinical_fallback.csv"),
                       encoding="utf-8-sig", index=False)
    print(clinical_df.to_string(index=False))

    # ── 분석 G: 코퍼스 크기 ──
    print("\n" + "▓" * 60)
    print("  [G] ABUSE_NEG 코퍼스 크기 변화")
    print("▓" * 60)
    corpus_df = compute_corpus_size(all_results)
    corpus_df.to_csv(os.path.join(args.out_dir, "G_corpus_size.csv"),
                     encoding="utf-8-sig", index=False)
    print(corpus_df.to_string(index=False))

    # ── 분석 H: 종합 추천 ──
    print("\n" + "▓" * 60)
    print("  [H] 종합 추천")
    print("▓" * 60)
    scores_df, best_thr = recommend_threshold(basic_stats, kappa_df, type_alloc_df, clinical_df)
    scores_df.to_csv(os.path.join(args.out_dir, "H_recommendation_scores.csv"),
                     encoding="utf-8-sig", index=False)
    print(scores_df.to_string(index=False))
    print(f"\n  ★ 권장 임계값: θ = {best_thr}")

    # ── 시각화 ──
    print("\n" + "▓" * 60)
    print("  시각화 생성 (8종)")
    print("▓" * 60)
    plot_basic_stats(basic_stats, args.out_dir)
    plot_score_distribution(all_scores, args.out_dir)
    plot_type_allocation(type_alloc_df, args.out_dir)
    plot_transitions(transitions, args.out_dir)
    plot_kappa(kappa_df, args.out_dir)
    plot_clinical_fallback(clinical_df, args.out_dir)
    plot_corpus_size(corpus_df, args.out_dir)
    plot_recommendation(scores_df, best_thr, args.out_dir)

    # ── 보고서 ──
    generate_report(basic_stats, kappa_df, clinical_df, corpus_df,
                    scores_df, best_thr, transitions, args.out_dir)

    print("\n" + "═" * 60)
    print(f"  ✅ 모든 분석 완료! 결과: {args.out_dir}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
