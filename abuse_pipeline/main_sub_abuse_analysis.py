"""
integrated_label_bridge_analysis.py
=====================================
7단계 통합 분석: BERT 단일 라벨 vs. 알고리즘 주+부 학대유형 비교
+ 교량 단어(Bridge Words) **직접 생성** + 연결 논증

★ 완전 자립형 — pipeline.py 의존 없이 JSON에서 교량 단어를 뽑고 전체 분석을 수행.

★★ 수정 사항 (논문 35개 교량 단어 방식 재현):
  [1] 빈도표: 토큰 빈도 → 문서 빈도(아동 1명 = 1회)
  [2] 후보 범위: 전체 어휘 → χ² 상위 200개 토큰
  [3] 필터: z_min=1.96 → count_min=5, z_min=None

  Stage 0: 데이터 로드 + 문서 빈도표 구축 + log-odds + χ² top200 + 교량 단어 추출
  Stage 1: Sub Abuse 할당 메커니즘 상세 통계
  Stage 2: 사례 유형 자동 분류 (Type A / B / C / D / E)
  Stage 3: 임상가 소견 다중 학대유형 언급 분석
  Stage 4: Hidden Companion + 정보 손실
  Stage 5: 교량 단어 ↔ Sub Abuse 연결 분석 (★ 핵심)
  Stage 6: BERT 구조적 약점 정량화
  Stage 7: 삼각검증(Triangulation) 수렴 보고서

실행:
  PyCharm에서 이 파일을 직접 우클릭 → Run 하면 됩니다.
  (파이프라인 내부에서 from .integrated_label_bridge_analysis import run_integrated_analysis 도 가능)
"""

from __future__ import annotations

import os
import glob
import json
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════
#  0. 프로젝트 모듈 Import — 3단계 폴백
# ═══════════════════════════════════════════════════════════════════════

_IMPORT_OK = False

# ── 폴백 1: 상대 import (pipeline.py 에서 호출될 때) ──
if not _IMPORT_OK:
    try:
        from . import common as C
        from .labels import classify_child_group, classify_abuse_main_sub
        from .text import extract_child_speech, tokenize_korean
        from .compare_abuse_labels import (
            extract_gt_abuse_types_from_info,
            normalize_abuse_label,
            DEFAULT_ABUSE_ORDER,
        )
        from .stats import (
            compute_log_odds, compute_chi_square, add_bh_fdr,
            compute_prob_bridge_for_words,
        )
        _IMPORT_OK = True
    except ImportError:
        pass

# ── 폴백 2: 절대 경로 import (PyCharm 직접 실행, 프로젝트 루트가 sys.path) ──
if not _IMPORT_OK:
    try:
        from v28_refactor.abuse_pipeline import common as C
        from v28_refactor.abuse_pipeline.labels import (
            classify_child_group, classify_abuse_main_sub,
        )
        from v28_refactor.abuse_pipeline.text import (
            extract_child_speech, tokenize_korean,
        )
        from v28_refactor.abuse_pipeline.compare_abuse_labels import (
            extract_gt_abuse_types_from_info,
            normalize_abuse_label,
            DEFAULT_ABUSE_ORDER,
        )
        from v28_refactor.abuse_pipeline.stats import (
            compute_log_odds, compute_chi_square, add_bh_fdr,
            compute_prob_bridge_for_words,
        )
        _IMPORT_OK = True
    except ImportError:
        pass

# ── 폴백 3: sys.path 직접 추가 후 재시도 ──
if not _IMPORT_OK:
    import sys
    from pathlib import Path as _Path
    _this_dir  = _Path(__file__).resolve().parent       # abuse_pipeline/
    _v28_dir   = _this_dir.parent                       # v28_refactor/
    _proj_root = _v28_dir.parent                        # Childeren/
    for _p in [str(_proj_root), str(_v28_dir), str(_this_dir)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from v28_refactor.abuse_pipeline import common as C
    from v28_refactor.abuse_pipeline.labels import (
        classify_child_group, classify_abuse_main_sub,
    )
    from v28_refactor.abuse_pipeline.text import (
        extract_child_speech, tokenize_korean,
    )
    from v28_refactor.abuse_pipeline.compare_abuse_labels import (
        extract_gt_abuse_types_from_info,
        normalize_abuse_label,
        DEFAULT_ABUSE_ORDER,
    )
    from v28_refactor.abuse_pipeline.stats import (
        compute_log_odds, compute_chi_square, add_bh_fdr,
        compute_prob_bridge_for_words,
    )
    _IMPORT_OK = True

# ── 상수 ──
ABUSE_ORDER    = C.ABUSE_ORDER            # ["성학대","신체학대","정서학대","방임"]
ABUSE_LABEL_EN = C.ABUSE_LABEL_EN
_SEVERITY_RANK = {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}


# ═══════════════════════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════════════════════

def _safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


def _cohens_kappa(cm: pd.DataFrame) -> float:
    arr = cm.values.astype(float)
    n = arr.sum()
    if n == 0:
        return 0.0
    p_o = np.trace(arr) / n
    p_e = (arr.sum(axis=1) * arr.sum(axis=0)).sum() / (n ** 2)
    return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 0.0


def _print_section(title: str, level: int = 1):
    if level == 1:
        print("\n" + "╔" + "═" * 68 + "╗")
        print(f"║  {title:<66s} ║")
        print("╚" + "═" * 68 + "╝")
    elif level == 2:
        print(f"\n{'▓' * 60}")
        print(f"  {title}")
        print(f"{'▓' * 60}")
    else:
        print(f"\n  ── {title} ──")


# ═══════════════════════════════════════════════════════════════════════
#  Stage 0  JSON → 문서 빈도표 → χ² top200 → log-odds → 교량 단어
#  ★★ 수정: 논문 35개 방식 (문서 빈도 + χ² top200 + count_min=5)
# ═══════════════════════════════════════════════════════════════════════

def build_bridge_words_from_json(
    json_files: List[str],
    only_negative: bool = True,
    min_total_count: int = None,
    min_p1: float = None,
    min_p2: float = None,
    max_gap: float = None,
    count_min: int = None,
    chi_top_n: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    JSON 파일 → 교량 단어 DataFrame 을 **직접** 생성한다.
    pipeline.py 를 실행하지 않아도 된다.

    ★ 논문 35개 방식을 충실히 재현:
      - 빈도표: 문서 빈도 (아동 1명 = 1회, 중복 토큰 무시)
      - 후보 범위: χ² 기여도 기준 상위 200개 토큰
      - 필터: count_min=5 (유형별 최소 문서 수), z_min=None

    흐름
    ────────────────────────────────────────
    JSON 파일들
      ① 아동별 정서군·주 학대유형 분류
      ② 아동 발화 추출 + 한국어 토큰화
      ③ 학대유형 × 단어 **문서 빈도표** (df_abuse_counts)
         → 아동 1명당 단어 w는 최대 1회 (출현 여부)
      ④ χ² 기여도 기준 상위 200개 토큰 선별
      ⑤ log-odds 통계 (abuse_stats_logodds)
      ⑥ P(abuse|word) + count_min=5 → 교량 단어 (bridge_df)
    ────────────────────────────────────────

    Parameters
    ----------
    json_files : list[str]
        JSON 파일 경로 목록
    only_negative : bool
        True이면 부정 정서군(NEG)만 사용
    min_total_count : int
        최소 전체 문서 빈도 (기본: MIN_TOTAL_COUNT_ABUSE = 8)
    min_p1, min_p2, max_gap : float
        교량 단어 조건부 확률 임계값 (기본: τ₁=0.40, τ₂=0.25, γ=0.20)
    count_min : int
        유형별 최소 문서 수 (기본: BRIDGE_MIN_COUNT = 5)
    chi_top_n : int
        χ² 기여도 기준 상위 N개 토큰 (기본: 200)

    Returns
    -------
    bridge_df          : 교량 단어 DataFrame
    df_abuse_counts    : 학대유형 × 단어 문서 빈도표
    abuse_stats_logodds: log-odds 통계 DataFrame
    """
    min_total_count = min_total_count or getattr(C, "MIN_TOTAL_COUNT_ABUSE", 8)
    min_p1    = min_p1    or getattr(C, "BRIDGE_MIN_P1", 0.40)
    min_p2    = min_p2    or getattr(C, "BRIDGE_MIN_P2", 0.25)
    max_gap   = max_gap   or getattr(C, "BRIDGE_MAX_GAP", 0.20)
    count_min = count_min or getattr(C, "BRIDGE_MIN_COUNT", 5)

    _print_section("Stage 0: 교량 단어 직접 생성 — 논문 35개 방식", 1)
    print(f"  ★ 빈도표: 문서 빈도 (아동 1명 = 1회)")
    print(f"  ★ 후보: χ² 상위 {chi_top_n}개 토큰")
    print(f"  ★ 필터: count_min={count_min}, z_min=None")

    # ── ① 아동별 발화 수집 ──
    print("\n  [0-1] 아동별 발화 수집 ...")
    # 각 아동의 (주 학대유형, 토큰 집합)을 모은다
    child_data: List[Tuple[str, set]] = []  # (abuse_type, {unique tokens})
    n_loaded = n_filtered = n_speech = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        n_loaded += 1

        group = classify_child_group(rec)
        if only_negative and group != "부정":
            n_filtered += 1
            continue

        main_abuse, _ = classify_abuse_main_sub(rec)
        speech_list = extract_child_speech(rec)

        if speech_list and main_abuse in ABUSE_ORDER:
            # ★ 핵심 변경: 한 아동의 모든 발화를 합친 뒤 고유 토큰만 추출
            all_tokens = set(tokenize_korean(" ".join(speech_list)))
            child_data.append((main_abuse, all_tokens))
            n_speech += 1

    print(f"        JSON 로드  : {n_loaded}")
    print(f"        부정 외 제외: {n_filtered}")
    print(f"        발화 보유   : {n_speech}명")

    abuse_child_counts = Counter(abuse for abuse, _ in child_data)
    for a in ABUSE_ORDER:
        print(f"          {a}: {abuse_child_counts.get(a, 0)}명")

    if not child_data:
        print("        ⚠ 데이터 없음 → 빈 DataFrame 반환")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── ② 문서 빈도표 (아동 1명 = 1회) ──
    # ★ 핵심 변경: 토큰 빈도가 아니라 문서 빈도
    print("\n  [0-2] 학대유형 × 단어 문서 빈도표 (아동 1명 = 1회) ...")
    rows_doc = []
    for abuse_type, unique_tokens in child_data:
        for w in unique_tokens:
            rows_doc.append({"abuse": abuse_type, "word": w})

    df_doc = pd.DataFrame(rows_doc)
    print(f"        (아동 × 고유단어) 행 수: {len(df_doc)}")

    if df_doc.empty:
        print("        ⚠ 토큰 없음 → 빈 DataFrame 반환")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # groupby로 문서 빈도 집계
    df_abuse_counts = (
        df_doc.groupby(["word", "abuse"]).size()
        .unstack("abuse").fillna(0).astype(int)
    )
    for a in ABUSE_ORDER:
        if a not in df_abuse_counts.columns:
            df_abuse_counts[a] = 0
    df_abuse_counts = df_abuse_counts[ABUSE_ORDER]

    # 최소 전체 문서 빈도 필터
    df_abuse_counts["total"] = df_abuse_counts.sum(axis=1)
    n_before = df_abuse_counts.shape[0]
    df_abuse_counts = (
        df_abuse_counts[df_abuse_counts["total"] >= min_total_count]
        .drop(columns=["total"])
    )
    print(f"        전체 고유 단어: {n_before}")
    print(f"        필터 후 단어 수 (문서 빈도 ≥ {min_total_count}): {df_abuse_counts.shape[0]}")

    if df_abuse_counts.empty:
        return pd.DataFrame(), df_abuse_counts, pd.DataFrame()

    # 유형별 총 문서 빈도 보고
    col_totals = df_abuse_counts.sum(axis=0)
    for a in ABUSE_ORDER:
        print(f"          {a}: 총 문서 빈도 {int(col_totals[a])}")

    # ── ③ χ² 계산 + 상위 200개 토큰 선별 ──
    # ★ 핵심 변경: CA에서 사용하는 χ² 상위 200개 범위로 후보 제한
    print(f"\n  [0-3] χ² 기여도 계산 + 상위 {chi_top_n}개 토큰 선별 ...")
    chi_df = compute_chi_square(df_abuse_counts, ABUSE_ORDER)
    chi_df = chi_df.sort_values("chi2", ascending=False)

    n_available = len(chi_df)
    actual_top_n = min(chi_top_n, n_available)
    top_words = chi_df.head(actual_top_n).index.tolist()

    # 상위 토큰의 최소 χ² 보고
    min_chi2 = chi_df.head(actual_top_n)["chi2"].min()
    print(f"        전체 단어: {n_available}")
    print(f"        상위 {actual_top_n}개 선별 (최소 χ² = {min_chi2:.2f})")

    # ── ④ log-odds (전체 빈도표 대상) ──
    print("\n  [0-4] log-odds 계산 ...")
    abuse_stats_logodds = compute_log_odds(df_abuse_counts, ABUSE_ORDER)
    print(f"        행 수: {len(abuse_stats_logodds)}")

    # ── ⑤ 교량 단어 추출 (χ² top200 범위 + count_min=5 + z_min=None) ──
    # ★ 핵심 변경: z_min 미적용, count_min만 적용
    print(f"\n  [0-5] 교량 단어 추출 — 논문 35개 방식")
    print(f"        범위: χ² 상위 {actual_top_n}개 토큰")
    print(f"        조건: P(k₁|w) ≥ {min_p1}, P(k₂|w) ≥ {min_p2}, "
          f"gap ≤ {max_gap}")
    print(f"        필터: count_min={count_min} (유형별 최소 문서 수)")
    print(f"        필터: z_min=None (미적용)")

    bridge_df = compute_prob_bridge_for_words(
        df_counts=df_abuse_counts,
        words=top_words,              # ★ χ² 상위 200개만
        logodds_df=abuse_stats_logodds,
        min_p1=min_p1,
        min_p2=min_p2,
        max_gap=max_gap,
        logodds_min=None,             # 미적용
        count_min=count_min,          # ★ 유형별 최소 문서 수 = 5
        z_min=None,                   # ★ 미적용 (논문 방식)
    )
    nb = len(bridge_df) if bridge_df is not None and not bridge_df.empty else 0
    print(f"\n        ✅ 교량 단어: {nb}개")

    if nb > 0:
        # 유형 쌍별 집계
        pairs = Counter(
            tuple(sorted([r["primary_abuse"], r["secondary_abuse"]]))
            for _, r in bridge_df.iterrows()
        )
        for pair, cnt in pairs.most_common():
            print(f"          {pair[0]}↔{pair[1]}: {cnt}개")

        # FDR 사후 검증 보고
        chi_with_fdr = add_bh_fdr(chi_df.copy(), p_col="p_value", out_col="q_fdr")
        bridge_words_set = set(bridge_df["word"])
        bridge_chi = chi_with_fdr.loc[
            chi_with_fdr.index.isin(bridge_words_set)
        ]
        if not bridge_chi.empty:
            max_q = bridge_chi["q_fdr"].max()
            min_chi2_bridge = bridge_chi["chi2"].min()
            all_sig = (bridge_chi["q_fdr"] < 0.05).all()
            print(f"\n        [사후 검증] FDR 보정:")
            print(f"          교량 단어 최소 χ² = {min_chi2_bridge:.2f}")
            print(f"          교량 단어 최대 q  = {max_q:.2e}")
            print(f"          전부 q < 0.05?   {'✅ 예' if all_sig else '❌ 아니오'}")

    return bridge_df, df_abuse_counts, abuse_stats_logodds


# ═══════════════════════════════════════════════════════════════════════
#  레코드 로딩 — Stage 1 ~ 7 공통
# ═══════════════════════════════════════════════════════════════════════

def _extract_full_record(
    rec: Dict[str, Any],
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
) -> Dict[str, Any]:
    """한 아동의 JSON 에서 라벨 + 임상 + 발화 정보를 모두 추출."""
    info = rec.get("info", {}) or {}
    doc_id = info.get("ID") or info.get("id") or info.get("Id")

    # BERT 라벨
    bert_set = extract_gt_abuse_types_from_info(info, field="학대의심")
    bert_label = (
        sorted(bert_set, key=lambda x: _SEVERITY_RANK.get(x, 99))[0]
        if bert_set else None
    )

    # 학대여부 문항 점수
    abuse_scores: Dict[str, int] = {a: 0 for a in ABUSE_ORDER}
    for q in rec.get("list", []):
        if q.get("문항") == "학대여부":
            for it in q.get("list", []):
                name = it.get("항목")
                try:
                    sc = int(it.get("점수"))
                except (TypeError, ValueError):
                    sc = 0
                if isinstance(name, str):
                    for a in ABUSE_ORDER:
                        if a in name:
                            abuse_scores[a] += sc

    # 알고리즘 라벨
    algo_main, algo_subs = classify_abuse_main_sub(
        rec, sub_threshold=sub_threshold, use_clinical_text=use_clinical_text,
    )
    algo_subs = algo_subs or []
    algo_set = set()
    if algo_main:
        algo_set.add(algo_main)
    algo_set |= set(algo_subs)

    # Sub 할당 소스 분석
    clin_text = " ".join(str(info.get(k, "")) for k in ["임상진단", "임상가 종합소견"] if k in info)
    sub_sources: Dict[str, str] = {}
    for sub in algo_subs:
        fs = abuse_scores.get(sub, 0) >= sub_threshold
        fc = sub in clin_text
        sub_sources[sub] = ("both" if fs and fc else "score" if fs
                            else "clinical" if fc else "unknown")

    # 임상가 소견
    clinical_text_raw = clin_text.strip()
    clinical_abuse_keywords = {a for a in ABUSE_ORDER if a in clinical_text_raw}

    # 정서군 · 발화
    valence_group = classify_child_group(rec)
    speech_raw = extract_child_speech(rec) or []
    speech_tokens = tokenize_korean(" ".join(speech_raw)) if speech_raw else []

    return dict(
        doc_id=doc_id,
        bert_label=bert_label, bert_label_set=bert_set,
        algo_main=algo_main, algo_subs=algo_subs, algo_set=algo_set,
        abuse_scores=abuse_scores, sub_sources=sub_sources,
        clinical_text_raw=clinical_text_raw,
        clinical_abuse_keywords=clinical_abuse_keywords,
        valence_group=valence_group,
        speech_raw=speech_raw, speech_tokens=speech_tokens,
    )


def _load_all_records(json_files, sub_threshold=4,
                      use_clinical_text=True, only_negative=False):
    records = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        r = _extract_full_record(rec, sub_threshold=sub_threshold,
                                 use_clinical_text=use_clinical_text)
        if only_negative and r["valence_group"] != "부정":
            continue
        records.append(r)
    print(f"  [LOAD] 아동 레코드: {len(records)}명")
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1  Sub Abuse 할당 메커니즘
# ═══════════════════════════════════════════════════════════════════════

def stage1_sub_abuse_mechanism(records, sub_threshold=4):
    _print_section("Stage 1: Sub Abuse 할당 메커니즘 상세 통계", 1)
    labeled = [r for r in records if r["algo_main"]]
    n = len(labeled)
    if n == 0:
        print("  ⚠ 라벨 없음"); return {}

    has_sub = [r for r in labeled if r["algo_subs"]]
    sub_rate = _safe_div(len(has_sub), n)
    n_subs_dist = Counter(len(r["algo_subs"]) for r in labeled)

    # 소스
    src_cnt = Counter()
    for r in has_sub:
        for _, src in r["sub_sources"].items():
            src_cnt[src] += 1
    total_src = sum(src_cnt.values())

    # 주→부 행렬
    mat = pd.DataFrame(0, index=ABUSE_ORDER, columns=ABUSE_ORDER, dtype=int)
    for r in labeled:
        if r["algo_main"] not in ABUSE_ORDER:
            continue
        for sub in r["algo_subs"]:
            if sub in ABUSE_ORDER:
                mat.loc[r["algo_main"], sub] += 1

    # 점수 상세
    rows_sc = []
    for r in has_sub:
        for sub in r["algo_subs"]:
            rows_sc.append(dict(doc_id=r["doc_id"], main=r["algo_main"], sub=sub,
                                score=r["abuse_scores"].get(sub, 0),
                                source=r["sub_sources"].get(sub, "")))
    df_sc = pd.DataFrame(rows_sc) if rows_sc else pd.DataFrame()

    # 출력
    print(f"\n  전체: {n}명  |  Sub 보유: {len(has_sub)}명 ({sub_rate:.1%})")
    print(f"  부 유형 수 분포:")
    for k in sorted(n_subs_dist):
        print(f"    {k}개: {n_subs_dist[k]}명 ({_safe_div(n_subs_dist[k], n):.1%})")
    print(f"\n  Sub 소스별 (threshold={sub_threshold}):")
    for s, lbl in [("score","설문"), ("clinical","임상"), ("both","설문+임상"), ("unknown","기타")]:
        c = src_cnt.get(s, 0)
        print(f"    {lbl}: {c}건 ({_safe_div(c, total_src):.1%})")
    print(f"\n  주→부 행렬:\n{mat.to_string()}")

    return dict(n_labeled=n, n_has_sub=len(has_sub), sub_rate=sub_rate,
                n_subs_dist=dict(n_subs_dist), source_counts=dict(src_cnt),
                main_sub_matrix=mat, df_sub_scores=df_sc)


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2  사례 유형 분류
# ═══════════════════════════════════════════════════════════════════════

def stage2_case_type_classification(records):
    _print_section("Stage 2: 사례 유형 자동 분류 (A / B / C / D / E)", 1)
    labeled = [r for r in records if r["bert_label"] and r["algo_main"]]
    cases: Dict[str, list] = {t: [] for t in "ABCDE"}

    for r in labeled:
        b, m = r["bert_label"], r["algo_main"]
        subs = set(r["algo_subs"])
        clin_sub = any(v in ("clinical", "both") for v in r["sub_sources"].values())

        if   b == m and subs:        cases["A"].append(r)
        elif b != m and b in subs:   cases["B"].append(r)
        elif clin_sub:               cases["C"].append(r)
        elif b == m and not subs:    cases["D"].append(r)
        else:                        cases["E"].append(r)

    n = len(labeled)
    desc = dict(A="주일치+부추가", B="BERT∈sub(우선순위)", C="임상 부 할당",
                D="완전 일치", E="진정한 불일치")
    for t in "ABDEC":
        c = len(cases[t])
        print(f"  Type {t}: {c:>4d}명 ({_safe_div(c,n):>6.1%})  {desc[t]}")

    # 예시
    rows = []
    for t in "ABC":
        for r in cases[t][:5]:
            rows.append(dict(type=f"Type_{t}", doc_id=r["doc_id"],
                             bert=r["bert_label"], algo_main=r["algo_main"],
                             algo_subs="|".join(sorted(r["algo_subs"])),
                             sub_sources=str(r["sub_sources"])))
    return dict(cases=cases, df_examples=pd.DataFrame(rows), n_total=n)


# ═══════════════════════════════════════════════════════════════════════
#  Stage 3  임상가 소견 다중 학대유형 언급
# ═══════════════════════════════════════════════════════════════════════

def stage3_clinical_text_analysis(records):
    _print_section("Stage 3: 임상가 소견 다중 학대유형 언급 분석", 1)
    labeled = [r for r in records if r["algo_main"]]
    has_clin = [r for r in labeled if r["clinical_text_raw"].strip()]
    nc = len(has_clin)
    print(f"\n  임상 소견 존재: {nc}/{len(labeled)}")

    multi = sum(1 for r in has_clin if len(r["clinical_abuse_keywords"]) >= 2)
    multi_rate = _safe_div(multi, nc)
    print(f"  2개+ 동시 언급: {multi}명 ({multi_rate:.1%})")

    # BERT별 추가 언급
    be = defaultdict(lambda: dict(total=0, extra=0, types=Counter()))
    for r in has_clin:
        if not r["bert_label"]:
            continue
        ext = r["clinical_abuse_keywords"] - {r["bert_label"]}
        be[r["bert_label"]]["total"] += 1
        if ext:
            be[r["bert_label"]]["extra"] += 1
            for e in ext:
                be[r["bert_label"]]["types"][e] += 1

    clin_rows = []
    print(f"\n  BERT별 추가 유형 언급:")
    for bt in ABUSE_ORDER:
        d = be.get(bt)
        if not d or d["total"] == 0:
            continue
        rate = _safe_div(d["extra"], d["total"])
        det = ", ".join(f"{k}({v})" for k, v in d["types"].most_common())
        print(f"    BERT={bt}: {d['extra']}/{d['total']} ({rate:.1%})  → {det}")
        clin_rows.append(dict(bert=bt, n=d["total"], extra=d["extra"],
                              rate=rate, detail=det))

    return dict(n_has_clinical=nc, multi_mention_rate=multi_rate,
                df_clinical_extra=pd.DataFrame(clin_rows))


# ═══════════════════════════════════════════════════════════════════════
#  Stage 4  Hidden Companion + 정보 손실
# ═══════════════════════════════════════════════════════════════════════

def stage4_hidden_companion_extended(records):
    _print_section("Stage 4: Hidden Companion + 정보 손실 (ΔH)", 1)
    labeled = [r for r in records if r["bert_label"] and r["algo_main"] and r["algo_set"]]

    # 엔트로피
    ent = []
    for r in labeled:
        scores = r["abuse_scores"]
        rel = {a: scores.get(a, 0) for a in r["algo_set"]}
        total = sum(rel.values())
        if total <= 0:
            h = np.log2(len(r["algo_set"])) if len(r["algo_set"]) > 1 else 0.0
        else:
            ps = [s / total for s in rel.values() if s > 0]
            h = sum(-p * np.log2(p) for p in ps)
        ent.append(dict(doc_id=r["doc_id"], n_labels=len(r["algo_set"]),
                        H_algo=h, delta_H=h))
    df_ent = pd.DataFrame(ent)
    mean_loss = df_ent["delta_H"].mean() if len(df_ent) else 0

    print(f"\n  대상: {len(df_ent)}명")
    print(f"  BERT H = 0 (단일 라벨)  |  Algo 평균 H = {df_ent['H_algo'].mean():.4f}")
    print(f"  평균 ΔH = {mean_loss:.4f} bits")

    # Hidden Companion P(Sub=열 | BERT=행)
    bg = defaultdict(list)
    for r in labeled:
        bg[r["bert_label"]].append(r)

    hc = []
    for bt in ABUSE_ORDER:
        grp = bg.get(bt, [])
        if not grp:
            continue
        for ht in ABUSE_ORDER:
            if ht == bt:
                continue
            nw = sum(1 for r in grp if ht in r["algo_subs"])
            ph = _safe_div(nw, len(grp))
            hc.append(dict(bert=bt, hidden=ht, n_group=len(grp),
                           n_with=nw, p_hidden=ph))
    df_hc = pd.DataFrame(hc)

    if not df_hc.empty:
        piv = (df_hc.pivot(index="bert", columns="hidden", values="p_hidden")
               .reindex(index=ABUSE_ORDER, columns=ABUSE_ORDER).fillna(0))
        print(f"\n  P(Sub=열 | BERT=행):\n{piv.round(4).to_string()}")

    pair_den = {}
    for _, row in df_hc.iterrows():
        p = tuple(sorted([row["bert"], row["hidden"]]))
        pair_den[p] = pair_den.get(p, 0) + row["p_hidden"]

    return dict(df_entropy=df_ent, mean_info_loss=mean_loss,
                df_hidden_companion=df_hc, pair_density=pair_den)


# ═══════════════════════════════════════════════════════════════════════
#  Stage 5  교량 단어 ↔ Sub Abuse 연결 (★ 핵심)
# ═══════════════════════════════════════════════════════════════════════

def stage5_bridge_sub_abuse_linkage(records, bridge_df=None, df_counts=None):
    _print_section("Stage 5: 교량 단어 ↔ Sub Abuse 연결 분석", 1)

    if bridge_df is None or bridge_df.empty:
        print("  ⚠ bridge_df 없음 → Stage 5 건너뜀")
        return {}

    labeled = [r for r in records if r["algo_main"] and r["speech_tokens"]]
    bridge_words = set(bridge_df["word"])
    binfo: Dict[str, dict] = {}
    for _, row in bridge_df.iterrows():
        binfo[row["word"]] = dict(
            k1=row["primary_abuse"], k2=row["secondary_abuse"],
            pair=tuple(sorted([row["primary_abuse"], row["secondary_abuse"]])),
            p1=row.get("p1", 0), p2=row.get("p2", 0),
        )

    print(f"\n  교량 단어 {len(bridge_words)}개  |  대상 아동 {len(labeled)}명")

    # ── 5-1 Sub 보유 vs 미보유 출현율 ──
    _print_section("5-1: Sub 보유 vs 미보유 교량 단어 출현율", 3)
    has_sub = [r for r in labeled if r["algo_subs"]]
    no_sub  = [r for r in labeled if not r["algo_subs"]]

    def _rate(grp):
        if not grp:
            return dict(n=0, n_with=0, rate=0, mean=0)
        cnts = [len(set(r["speech_tokens"]) & bridge_words) for r in grp]
        nw = sum(1 for c in cnts if c > 0)
        return dict(n=len(grp), n_with=nw, rate=_safe_div(nw, len(grp)),
                    mean=np.mean(cnts))

    ss, sn = _rate(has_sub), _rate(no_sub)
    print(f"  Sub 보유  (n={ss['n']}): {ss['rate']:.1%}  평균 {ss['mean']:.2f}개")
    print(f"  Sub 미보유(n={sn['n']}): {sn['rate']:.1%}  평균 {sn['mean']:.2f}개")

    # ── 5-2 유형 쌍별 교량 단어 수 ──
    _print_section("5-2: 유형 쌍별 교량 단어 분포", 3)
    bpc = Counter(bi["pair"] for bi in binfo.values())
    for pair, cnt in bpc.most_common():
        print(f"    {pair[0]}↔{pair[1]}: {cnt}개")

    # ── 5-3 주-부 매칭률 ──
    _print_section("5-3: Sub Abuse 유형별 교량 단어 매칭", 3)
    match_rows = []
    for r in has_sub:
        toks = set(r["speech_tokens"])
        for sub in r["algo_subs"]:
            if sub not in ABUSE_ORDER:
                continue
            tp = tuple(sorted([r["algo_main"], sub]))
            avail = [w for w, bi in binfo.items() if bi["pair"] == tp]
            found = toks & set(avail)
            match_rows.append(dict(
                doc_id=r["doc_id"], main=r["algo_main"], sub=sub,
                pair=f"{tp[0]}↔{tp[1]}",
                n_avail=len(avail), n_found=len(found),
                found_words="|".join(sorted(found)), hit=int(bool(found)),
            ))
    df_match = pd.DataFrame(match_rows) if match_rows else pd.DataFrame()

    if not df_match.empty:
        mr = df_match["hit"].mean()
        print(f"\n  전체 매칭률: {mr:.1%} ({df_match['hit'].sum()}/{len(df_match)})")
        for pair_str, g in df_match.groupby("pair"):
            print(f"    {pair_str}: {g['hit'].mean():.1%} ({g['hit'].sum()}/{len(g)})")

    # ── 5-4 발화 추출 + 불일치율 ──
    _print_section("5-4: 교량 단어 발화 & 라벨-불일치율", 3)
    utt_rows = []
    for r in labeled:
        if not r["speech_raw"]:
            continue
        found = set(r["speech_tokens"]) & bridge_words
        if not found:
            continue
        for raw in r["speech_raw"]:
            tks = tokenize_korean(raw)
            for bw in found:
                if bw in tks:
                    utt_rows.append(dict(
                        doc_id=r["doc_id"], bridge_word=bw,
                        algo_main=r["algo_main"],
                        algo_subs="|".join(sorted(r["algo_subs"])),
                        bert=r.get("bert_label", ""),
                        k1=binfo[bw]["k1"], k2=binfo[bw]["k2"],
                        utterance=raw[:200],
                        mismatch=int(r["algo_main"] != binfo[bw]["k1"]),
                    ))
    df_utt = pd.DataFrame(utt_rows) if utt_rows else pd.DataFrame()
    mis_rate = df_utt["mismatch"].mean() if not df_utt.empty else 0

    if not df_utt.empty:
        print(f"\n  발화 수: {len(df_utt)}  |  불일치율: {mis_rate:.1%}")
        by_w = (df_utt.groupby("bridge_word")
                .agg(n=("mismatch","count"), mis=("mismatch","sum"),
                     rate=("mismatch","mean"), k1=("k1","first"), k2=("k2","first"))
                .sort_values("n", ascending=False))
        print(f"\n  교량 단어별 불일치율 (상위 15):")
        for w, row in by_w.head(15).iterrows():
            print(f"    '{w}' ({row['k1']}↔{row['k2']}): "
                  f"{row['rate']:.1%} ({int(row['mis'])}/{int(row['n'])})")

    return dict(
        bridge_stats_sub=ss, bridge_stats_nosub=sn,
        bridge_pair_count=dict(bpc),
        df_bridge_match=df_match, df_bridge_utterances=df_utt,
        mismatch_rate=mis_rate,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Stage 6  BERT 구조적 약점
# ═══════════════════════════════════════════════════════════════════════

def stage6_bert_weakness(records, s4=None, s5=None):
    _print_section("Stage 6: BERT 구조적 약점 정량화", 1)
    labeled = [r for r in records if r["bert_label"] and r["algo_main"]]
    n = len(labeled)

    # 약점 1: 단일 범주 강제
    multi = [r for r in labeled if len(r["algo_set"]) >= 2]
    missed = sum(len(r["algo_set"] - r["bert_label_set"]) for r in multi)
    avg_miss = _safe_div(missed, len(multi))
    print(f"\n  [약점 1] 단일 범주 강제")
    print(f"    다중 라벨 아동: {len(multi)}명 ({_safe_div(len(multi), n):.1%})")
    print(f"    BERT 누락 유형: 총 {missed}, 평균 {avg_miss:.2f}개")

    # 약점 2: 모호성 지대
    mis = [r for r in labeled if r["bert_label"] != r["algo_main"]]
    non_sex = {"신체학대", "정서학대", "방임"}
    cont = [r for r in mis if r["bert_label"] in non_sex and r["algo_main"] in non_sex]
    cont_rate = _safe_div(len(cont), len(mis))
    emot = [r for r in mis if "정서학대" in {r["bert_label"], r["algo_main"]}]
    print(f"\n  [약점 2] 모호성 지대")
    print(f"    불일치: {len(mis)}건 ({_safe_div(len(mis), n):.1%})")
    print(f"    비성학대 내부: {len(cont)}건 ({cont_rate:.1%} of mismatch)")
    print(f"    정서학대 관여: {len(emot)}건 ({_safe_div(len(emot), len(mis)):.1%})")

    # BERT ∈ algo sub
    bis = [r for r in mis if r["bert_label"] in r["algo_subs"]]
    bis_rate = _safe_div(len(bis), len(mis))
    print(f"\n  [불일치의 성격]")
    print(f"    BERT ∈ algo sub: {len(bis)}건 ({bis_rate:.1%})  → 우선순위 차이")

    # 혼동행렬 + κ
    cm = pd.crosstab(
        pd.Categorical([r["bert_label"] for r in labeled], categories=ABUSE_ORDER),
        pd.Categorical([r["algo_main"]  for r in labeled], categories=ABUSE_ORDER),
    )
    kappa = _cohens_kappa(cm)
    acc = _safe_div(sum(cm.values[i, i] for i in range(len(ABUSE_ORDER))), n)
    print(f"\n  일치율: {acc:.1%}  |  Cohen's κ: {kappa:.4f}")
    print(f"\n  혼동행렬 (행=BERT, 열=Algo):\n{cm.to_string()}")

    return dict(
        multi_rate=_safe_div(len(multi), n), avg_missed=avg_miss,
        n_mismatch=len(mis), continuum_rate=cont_rate,
        bert_in_sub_rate=bis_rate, kappa=kappa, accuracy=acc, cm=cm,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Stage 7  삼각검증 수렴 보고서
# ═══════════════════════════════════════════════════════════════════════

def stage7_triangulation_report(s4, s5, s6):
    _print_section("Stage 7: 삼각검증(Triangulation) 수렴 보고서", 1)
    rows = []

    # 수치
    rows.append(dict(layer="수치적", indicator="평균 ΔH",
                     value=f"{s4.get('mean_info_loss',0):.4f} bits",
                     interp="BERT 단일 라벨이 소멸시키는 정보량"))
    if s4.get("pair_density"):
        tp = max(s4["pair_density"].items(), key=lambda x: x[1])
        rows.append(dict(layer="수치적", indicator="최대 HC 밀도",
                         value=f"{tp[0][0]}↔{tp[0][1]}: {tp[1]:.4f}",
                         interp="가장 빈번한 숨겨진 동반 학대"))

    # 언어
    if s5:
        rows.append(dict(layer="언어적", indicator="교량 불일치율",
                         value=f"{s5.get('mismatch_rate',0):.1%}",
                         interp="교량 단어가 경계에서 작동"))
        if s5.get("bridge_pair_count"):
            tbp = max(s5["bridge_pair_count"].items(), key=lambda x: x[1])
            rows.append(dict(layer="언어적", indicator="최다 교량 쌍",
                             value=f"{tbp[0][0]}↔{tbp[0][1]}: {tbp[1]}개",
                             interp="가장 활발한 교량 경계"))
        ss, sn = s5.get("bridge_stats_sub", {}), s5.get("bridge_stats_nosub", {})
        if ss and sn:
            rows.append(dict(layer="언어적", indicator="Sub 교량 출현율",
                             value=f"{ss.get('rate',0):.1%} vs {sn.get('rate',0):.1%}",
                             interp="Sub 보유 아동이 교량 단어를 더 자주 사용"))

    # 구조
    rows.append(dict(layer="구조적", indicator="비성학대 불일치 집중",
                     value=f"{s6.get('continuum_rate',0):.1%}",
                     interp="불일치가 비성학대 3유형에 구조적 집중"))
    rows.append(dict(layer="구조적", indicator="BERT∈Algo Sub",
                     value=f"{s6.get('bert_in_sub_rate',0):.1%}",
                     interp="오류가 아닌 우선순위 차이"))
    rows.append(dict(layer="구조적", indicator="Cohen's κ",
                     value=f"{s6.get('kappa',0):.4f}",
                     interp="거시적 일치 + 미시적 분기"))

    df = pd.DataFrame(rows)
    print(f"\n  {'층위':<8s} {'지표':<22s} {'값':<22s} {'해석'}")
    print(f"  {'─' * 72}")
    for _, r in df.iterrows():
        print(f"  {r['layer']:<8s} {r['indicator']:<22s} "
              f"{r['value']:<22s} {r['interp']}")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  run_integrated_analysis  — 통합 실행 진입점
# ═══════════════════════════════════════════════════════════════════════

def run_integrated_analysis(
    json_files,
    out_dir=None,
    bridge_df=None,
    df_counts=None,
    sub_threshold=4,
    use_clinical_text=True,
    only_negative=False,
):
    """7단계 통합 분석 실행."""
    if out_dir is None:
        out_dir = os.path.join(getattr(C, "OUTPUT_DIR", "."), "integrated_analysis")
    os.makedirs(out_dir, exist_ok=True)

    _print_section("7단계 통합 분석 실행", 1)
    print(f"  sub_threshold={sub_threshold}  clinical={use_clinical_text}  "
          f"neg_only={only_negative}")
    has_bridge = bridge_df is not None and not bridge_df.empty
    print(f"  bridge_df: {'있음 (' + str(len(bridge_df)) + '개)' if has_bridge else '없음'}")
    print(f"  출력: {out_dir}")

    records = _load_all_records(
        json_files, sub_threshold=sub_threshold,
        use_clinical_text=use_clinical_text, only_negative=only_negative,
    )
    if not records:
        print("  ❌ 유효한 레코드 없음"); return {}

    results = dict(records=records, out_dir=out_dir)

    # Stage 1
    s1 = stage1_sub_abuse_mechanism(records, sub_threshold)
    results["stage1"] = s1
    if s1:
        s1["main_sub_matrix"].to_csv(os.path.join(out_dir, "stage1_main_sub_matrix.csv"),
                                     encoding="utf-8-sig")
        if not s1["df_sub_scores"].empty:
            s1["df_sub_scores"].to_csv(os.path.join(out_dir, "stage1_sub_scores.csv"),
                                       encoding="utf-8-sig", index=False)

    # Stage 2
    s2 = stage2_case_type_classification(records)
    results["stage2"] = s2
    if s2 and not s2["df_examples"].empty:
        s2["df_examples"].to_csv(os.path.join(out_dir, "stage2_examples.csv"),
                                 encoding="utf-8-sig", index=False)

    # Stage 3
    s3 = stage3_clinical_text_analysis(records)
    results["stage3"] = s3
    if s3 and not s3["df_clinical_extra"].empty:
        s3["df_clinical_extra"].to_csv(os.path.join(out_dir, "stage3_clinical.csv"),
                                       encoding="utf-8-sig", index=False)

    # Stage 4
    s4 = stage4_hidden_companion_extended(records)
    results["stage4"] = s4
    if s4:
        if not s4["df_entropy"].empty:
            s4["df_entropy"].to_csv(os.path.join(out_dir, "stage4_entropy.csv"),
                                    encoding="utf-8-sig", index=False)
        if not s4["df_hidden_companion"].empty:
            s4["df_hidden_companion"].to_csv(
                os.path.join(out_dir, "stage4_hidden_companion.csv"),
                encoding="utf-8-sig", index=False)

    # Stage 5
    s5 = stage5_bridge_sub_abuse_linkage(records, bridge_df, df_counts)
    results["stage5"] = s5
    if s5:
        for key in ["df_bridge_match", "df_bridge_utterances"]:
            df = s5.get(key)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(os.path.join(out_dir, f"stage5_{key}.csv"),
                          encoding="utf-8-sig", index=False)

    # Stage 6
    s6 = stage6_bert_weakness(records, s4, s5)
    results["stage6"] = s6
    if s6:
        s6["cm"].to_csv(os.path.join(out_dir, "stage6_confusion_matrix.csv"),
                        encoding="utf-8-sig")

    # Stage 7
    s7 = stage7_triangulation_report(s4, s5 or {}, s6)
    results["stage7"] = s7
    s7.to_csv(os.path.join(out_dir, "stage7_triangulation.csv"),
              encoding="utf-8-sig", index=False)

    # ── 최종 요약 ──
    summary = [dict(metric="분석 아동 수", value=str(len(records)))]
    if s1:
        summary.append(dict(metric="Sub 보유율", value=f"{s1['sub_rate']:.1%}"))
    if s2:
        summary.append(dict(metric="Type A(주일치+부추가)", value=str(len(s2["cases"]["A"]))))
        summary.append(dict(metric="Type B(BERT∈sub)", value=str(len(s2["cases"]["B"]))))
    if s3:
        summary.append(dict(metric="임상 다중언급률", value=f"{s3['multi_mention_rate']:.1%}"))
    if s4:
        summary.append(dict(metric="평균 ΔH", value=f"{s4['mean_info_loss']:.4f} bits"))
    if s5 and s5.get("mismatch_rate"):
        summary.append(dict(metric="교량 불일치율", value=f"{s5['mismatch_rate']:.1%}"))
    if s5 and isinstance(s5.get("df_bridge_match"), pd.DataFrame) and not s5["df_bridge_match"].empty:
        summary.append(dict(metric="교량-Sub 매칭률", value=f"{s5['df_bridge_match']['hit'].mean():.1%}"))
    if s6:
        summary.append(dict(metric="Cohen's κ", value=f"{s6['kappa']:.4f}"))
        summary.append(dict(metric="비성학대 불일치 집중", value=f"{s6['continuum_rate']:.1%}"))

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(out_dir, "FINAL_SUMMARY.csv"),
                  encoding="utf-8-sig", index=False)

    _print_section("전체 분석 완료!", 1)
    w = 60
    print(f"\n  ╔{'═' * w}╗")
    print(f"  ║{'최종 요약':^{w}s}║")
    print(f"  ╠{'═' * w}╣")
    for _, r in df_sum.iterrows():
        line = f"  {r['metric']:<32s} {str(r['value']):<24s}"
        print(f"  ║{line:<{w}s}║")
    print(f"  ╚{'═' * w}╝")

    print(f"\n  생성 파일:")
    for fn in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, fn))
        print(f"    {fn}  ({sz:,} bytes)")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  __main__ — PyCharm에서 직접 실행 (완전 자립형)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path as _Path

    print("=" * 72)
    print("  7단계 통합 분석 — 완전 자립형")
    print("  (Stage 0: 논문 35개 방식 교량 단어 → Stage 1~7: 전체 분석)")
    print("  ★ 문서 빈도 + χ² top200 + count_min=5 + z_min=None")
    print("=" * 72)

    # ── 경로 ──
    _this = _Path(__file__).resolve()

    # 프로젝트 루트(= data 폴더가 있는 위치) 자동 탐색
    _project_root = None
    for p in [_this.parent] + list(_this.parents):
        if (p / "data").is_dir():
            _project_root = p
            break

    if _project_root is None:
        raise FileNotFoundError(f"data 폴더를 찾을 수 없습니다. 시작 위치: {_this}")

    _data_dir = _project_root / "data"

    print(f"\n  프로젝트: {_project_root}")
    print(f"  데이터  : {_data_dir}")

    C.configure_output_dirs(subset_name="NEG")
    _out = os.path.join(C.OUTPUT_DIR, "integrated_analysis")
    os.makedirs(_out, exist_ok=True)
    print(f"  OUTPUT  : {C.OUTPUT_DIR}")
    print(f"  결과    : {_out}")

    # ── JSON ──
    if not _data_dir.exists():
        print(f"\n  ❌ data 폴더 없음: {_data_dir}"); exit(1)
    _jsons = sorted(glob.glob(str(_data_dir / "*.json")))
    print(f"  JSON    : {len(_jsons)}개")
    if not _jsons:
        print("  ❌ JSON 없음"); exit(1)

    # ── Stage 0: 논문 35개 방식 교량 단어 생성 ──
    _bridge, _counts, _logodds = build_bridge_words_from_json(
        json_files=_jsons,
        only_negative=True,
        # ★ 논문 35개 방식 파라미터 (명시적으로 지정)
        min_total_count=8,      # ABUSE_NEG 최소 빈도 필터
        min_p1=0.40,            # τ₁
        min_p2=0.25,            # τ₂
        max_gap=0.20,           # γ
        count_min=5,            # 유형별 최소 문서 수
        chi_top_n=200,          # χ² 상위 200개 범위
    )

    # Stage 0 산출물 저장
    if _bridge is not None and not _bridge.empty:
        _bridge.to_csv(os.path.join(_out, "stage0_bridge_words.csv"),
                       encoding="utf-8-sig", index=False)
    if _counts is not None and not _counts.empty:
        _counts.to_csv(os.path.join(_out, "stage0_abuse_word_counts.csv"),
                       encoding="utf-8-sig")
    if _logodds is not None and not _logodds.empty:
        _logodds.to_csv(os.path.join(_out, "stage0_logodds.csv"),
                        encoding="utf-8-sig", index=False)

    # ── Stage 1~7: 통합 분석 ──
    _results = run_integrated_analysis(
        json_files=_jsons,
        out_dir=_out,
        bridge_df=_bridge,
        df_counts=_counts,
        sub_threshold=2,
        use_clinical_text=True,
        only_negative=True,
    )

    print("\n" + "=" * 72)
    print("  ✅ 완전 자립형 실행 완료!")
    print(f"  결과: {_out}")
    print("=" * 72)