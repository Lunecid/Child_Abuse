#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_gt_1350_vs_1343.py
════════════════════════════════════════════════════════════════
GT 사례 수 차이(1,350 vs 1,343) 진단 스크립트.

두 경로로 GT 사례를 뽑아서 차집합의 7건을 식별하고,
각 사례의 상세 정보를 출력한다.

경로 A (raw_score_distribution):
  - info["학대의심"]에서 GT 라벨을 추출
  - 위기단계 기반 간이 NEG 판정 → NEG 내 GT만 선택

경로 B (기존 파이프라인 = revision_v2.build_docs):
  - classify_child_group() 으로 정서군 판정 (Algorithm 1)
  - classify_abuse_main_sub() 으로 main abuse 할당 (Algorithm 2)
  - only_negative=True → 부정군만
  - gt_mapped가 비어있지 않은 사례

실행:
  python debug_gt_1350_vs_1343.py --data_dir ./data
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

# ── sys.path bootstrap ──
_this = Path(__file__).resolve().parent
for p in [_this.parent, _this.parent.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from abuse_pipeline.core.common import ABUSE_ORDER, DATA_JSON_DIR, SEVERITY_RANK
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.raw_score_distribution import (
    extract_raw_abuse_scores,
    filter_by_corpus,
    _extract_gt_labels,
    _extract_abuse_scores_from_rec,
    _read_json_any,
    _iter_records,
)


def _find_json_files(data_dir: str | Path) -> List[Path]:
    d = Path(data_dir)
    if not d.exists():
        return []
    return sorted(d.rglob("*.json"))


def main(data_dir: str | None = None):
    data_dir = data_dir or DATA_JSON_DIR
    json_files = _find_json_files(data_dir)
    print(f"JSON 파일 수: {len(json_files)}")
    if not json_files:
        print("[ERROR] JSON 파일 없음. --data_dir 확인 필요.")
        return

    # ══════════════════════════════════════════════════════════
    #  경로 A: raw_score_distribution (간이 NEG + GT)
    # ══════════════════════════════════════════════════════════
    scores_df = extract_raw_abuse_scores(json_files=json_files)
    gt_a_df = filter_by_corpus(scores_df, "GT")
    gt_a_ids = set(gt_a_df["case_id"].tolist())
    print(f"\n[경로 A] raw_score_distribution GT: {len(gt_a_ids)}명")

    # ══════════════════════════════════════════════════════════
    #  경로 B: 기존 파이프라인 (Algorithm 1 + 2 + GT)
    # ══════════════════════════════════════════════════════════
    gt_b_ids: Set[str] = set()
    all_rec_info: Dict[str, Dict[str, Any]] = {}  # case_id → 상세 정보

    for fp in json_files:
        try:
            obj = _read_json_any(fp)
        except Exception:
            continue

        for ridx, rec in enumerate(_iter_records(obj)):
            info = rec.get("info", {}) or {}
            case_id = info.get("ID") or info.get("id") or info.get("Id")
            if not case_id:
                case_id = f"{fp.stem}__{ridx}"
            case_id = str(case_id)

            # Algorithm 1: 정서군 판정
            try:
                valence = classify_child_group(rec)
            except Exception:
                valence = None

            # NEG가 아니면 기존 파이프라인에서 탈락
            if valence != "부정":
                # 하지만 상세 정보는 저장 (차집합 분석용)
                all_rec_info[case_id] = {
                    "valence": valence,
                    "crisis": info.get("위기단계"),
                    "total_score": info.get("합계점수"),
                    "main_abuse": None,
                    "sub_abuses": [],
                    "gt_raw": info.get("학대의심", ""),
                    "a_scores": _extract_abuse_scores_from_rec(rec),
                    "gt_labels": _extract_gt_labels(info),
                    "rec": rec,  # 원본 저장
                }
                continue

            # Algorithm 2: 학대유형 할당
            try:
                main_abuse, subs = classify_abuse_main_sub(rec)
            except Exception:
                main_abuse, subs = (None, [])

            # GT 라벨 추출 (revision_v2 방식)
            gt_set = _extract_gt_labels(info)

            a_scores = _extract_abuse_scores_from_rec(rec)

            all_rec_info[case_id] = {
                "valence": valence,
                "crisis": info.get("위기단계"),
                "total_score": info.get("합계점수"),
                "main_abuse": main_abuse,
                "sub_abuses": list(subs or []),
                "gt_raw": info.get("학대의심", ""),
                "a_scores": a_scores,
                "gt_labels": gt_set,
                "rec": rec,
            }

            # 기존 파이프라인에서 GT 보유 = gt_set이 비어있지 않음
            if gt_set:
                gt_b_ids.add(case_id)

    print(f"[경로 B] 기존 파이프라인 NEG+GT: {len(gt_b_ids)}명")

    # ══════════════════════════════════════════════════════════
    #  차집합 분석
    # ══════════════════════════════════════════════════════════
    only_in_a = gt_a_ids - gt_b_ids  # A에만 있음 (raw에서는 GT인데 기존에서 아님)
    only_in_b = gt_b_ids - gt_a_ids  # B에만 있음 (기존에서는 GT인데 raw에서 아님)
    intersection = gt_a_ids & gt_b_ids

    print(f"\n{'═' * 60}")
    print(f"교집합: {len(intersection)}명")
    print(f"A에만 (raw GT, 기존 탈락): {len(only_in_a)}명")
    print(f"B에만 (기존 GT, raw 탈락): {len(only_in_b)}명")
    print(f"{'═' * 60}")

    # ── A에만 있는 사례 상세 ──
    if only_in_a:
        print(f"\n{'─' * 60}")
        print(f"[A에만 존재] raw_score에서는 NEG+GT지만 기존 파이프라인에서 탈락한 사례")
        print(f"{'─' * 60}")
        for cid in sorted(only_in_a):
            _print_case_detail(cid, all_rec_info, gt_a_df)

    # ── B에만 있는 사례 상세 ──
    if only_in_b:
        print(f"\n{'─' * 60}")
        print(f"[B에만 존재] 기존 파이프라인에서는 NEG+GT지만 raw에서 탈락한 사례")
        print(f"{'─' * 60}")
        for cid in sorted(only_in_b):
            _print_case_detail(cid, all_rec_info, gt_a_df)

    # ── 수치 차이 없는 경우 ──
    if not only_in_a and not only_in_b:
        print("\n✓ 두 경로의 GT 사례 집합이 완전히 일치합니다.")

    # ══════════════════════════════════════════════════════════
    #  불일치 원인 진단 요약
    # ══════════════════════════════════════════════════════════
    if only_in_a or only_in_b:
        print(f"\n{'═' * 60}")
        print("[진단 요약]")
        _diagnose(only_in_a, only_in_b, all_rec_info)
        print(f"{'═' * 60}")

    # 결과 CSV 저장
    _save_diff_csv(only_in_a, only_in_b, all_rec_info, Path(data_dir).parent)


def _print_case_detail(
    case_id: str,
    info_map: Dict[str, Dict[str, Any]],
    gt_a_df: pd.DataFrame,
):
    """한 사례의 상세 정보를 출력한다."""
    print(f"\n--- {case_id} ---")
    detail = info_map.get(case_id)
    if detail is None:
        print("  (상세 정보 없음)")
        return

    crisis = detail.get("crisis", "?")
    total = detail.get("total_score", "?")
    valence = detail.get("valence", "?")
    main_abuse = detail.get("main_abuse", "?")
    subs = detail.get("sub_abuses", [])
    gt_raw = detail.get("gt_raw", "")
    gt_labels = detail.get("gt_labels", set())
    a_scores = detail.get("a_scores", {})

    a_neglect = a_scores.get("방임", 0)
    a_emotional = a_scores.get("정서학대", 0)
    a_physical = a_scores.get("신체학대", 0)
    a_sexual = a_scores.get("성학대", 0)

    print(f"  위기단계(L) = {crisis}")
    print(f"  합계점수(S) = {total}")
    print(f"  Algorithm 1 결과(정서군) = {valence}")
    print(f"  Algorithm 2 결과(main)  = {main_abuse}")
    print(f"  Algorithm 2 결과(subs)  = {subs}")
    print(f"  A_k = (방임={a_neglect}, 정서={a_emotional}, 신체={a_physical}, 성={a_sexual})")
    print(f"  info['학대의심'] (raw) = {gt_raw!r}")
    print(f"  GT 라벨 추출 결과 = {gt_labels}")

    # raw_score_distribution에서의 멤버십
    row = gt_a_df[gt_a_df["case_id"] == case_id]
    if not row.empty:
        membership = row.iloc[0].get("corpus_membership", set())
        print(f"  raw_score membership = {membership}")
    else:
        # scores_df 전체에서 찾기
        print(f"  raw_score membership = (GT 집합에 없음)")

    # 불일치 원인 힌트
    neg_crisis = {"응급", "위기아동", "학대의심", "상담필요"}
    raw_is_neg = (crisis in neg_crisis) or (
        isinstance(total, (int, float)) and total >= 45
    )
    algo1_is_neg = (valence == "부정")

    if raw_is_neg != algo1_is_neg:
        print(f"  ⚠ NEG 판정 불일치: raw 간이={raw_is_neg}, Algorithm 1={algo1_is_neg}")
        if algo1_is_neg and not raw_is_neg:
            print(f"    → raw 간이 판정이 Algorithm 1보다 좁음 (위기단계={crisis}, 점수={total})")
        elif raw_is_neg and not algo1_is_neg:
            print(f"    → Algorithm 1이 부정으로 분류하지 않음 (보호요인/risk 조정 가능성)")


def _diagnose(
    only_in_a: set,
    only_in_b: set,
    info_map: Dict[str, Dict[str, Any]],
):
    """불일치의 체계적 원인 분류."""
    reasons_a = {"neg_mismatch": [], "gt_mismatch": [], "unknown": []}
    reasons_b = {"neg_mismatch": [], "gt_mismatch": [], "unknown": []}

    neg_crisis = {"응급", "위기아동", "학대의심", "상담필요"}

    for cid in only_in_a:
        d = info_map.get(cid, {})
        valence = d.get("valence")
        crisis = d.get("crisis")
        total = d.get("total_score", 0)
        try:
            total = int(total) if total else 0
        except (TypeError, ValueError):
            total = 0

        raw_neg = (crisis in neg_crisis) or (total >= 45)
        algo_neg = (valence == "부정")

        if raw_neg and not algo_neg:
            reasons_a["neg_mismatch"].append(cid)
        else:
            reasons_a["unknown"].append(cid)

    for cid in only_in_b:
        d = info_map.get(cid, {})
        valence = d.get("valence")
        crisis = d.get("crisis")
        total = d.get("total_score", 0)
        try:
            total = int(total) if total else 0
        except (TypeError, ValueError):
            total = 0

        raw_neg = (crisis in neg_crisis) or (total >= 45)
        algo_neg = (valence == "부정")

        if algo_neg and not raw_neg:
            reasons_b["neg_mismatch"].append(cid)
        else:
            reasons_b["unknown"].append(cid)

    if only_in_a:
        print(f"\n[A에만 존재하는 {len(only_in_a)}건의 원인]")
        if reasons_a["neg_mismatch"]:
            print(f"  NEG 판정 불일치 (raw=NEG, Algo1≠부정): {len(reasons_a['neg_mismatch'])}건")
            print(f"    → raw 간이 판정은 위기단계/점수만 보지만, Algorithm 1은 보호요인 등으로 부정 판정을 번복할 수 있음")
        if reasons_a["unknown"]:
            print(f"  기타/미분류: {len(reasons_a['unknown'])}건")

    if only_in_b:
        print(f"\n[B에만 존재하는 {len(only_in_b)}건의 원인]")
        if reasons_b["neg_mismatch"]:
            print(f"  NEG 판정 불일치 (Algo1=부정, raw≠NEG): {len(reasons_b['neg_mismatch'])}건")
            print(f"    → Algorithm 1이 risk/보호요인으로 부정 판정했지만, 위기단계/점수 기준으로는 NEG 아님")
        if reasons_b["unknown"]:
            print(f"  기타/미분류: {len(reasons_b['unknown'])}건")


def _save_diff_csv(
    only_in_a: set,
    only_in_b: set,
    info_map: Dict[str, Dict[str, Any]],
    base_dir: Path,
):
    """차집합 사례를 CSV로 저장한다."""
    out_dir = base_dir / "outputs" / "revision" / "raw_score_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in sorted(only_in_a | only_in_b):
        d = info_map.get(cid, {})
        a_scores = d.get("a_scores", {})
        rows.append({
            "case_id": cid,
            "source": "A_only" if cid in only_in_a else "B_only",
            "crisis": d.get("crisis"),
            "total_score": d.get("total_score"),
            "valence_algo1": d.get("valence"),
            "main_abuse_algo2": d.get("main_abuse"),
            "gt_raw": str(d.get("gt_raw", "")),
            "gt_labels": str(d.get("gt_labels", set())),
            "A_neglect": a_scores.get("방임", 0),
            "A_emotional": a_scores.get("정서학대", 0),
            "A_physical": a_scores.get("신체학대", 0),
            "A_sexual": a_scores.get("성학대", 0),
        })

    if rows:
        df = pd.DataFrame(rows)
        path = out_dir / "gt_diff_diagnosis.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT 1350 vs 1343 진단")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir)
