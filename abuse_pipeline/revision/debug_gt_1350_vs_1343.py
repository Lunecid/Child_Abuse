#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_gt_1350_vs_1343.py
════════════════════════════════════════════════════════════════
GT 사례 수 차이 진단 스크립트.

비교 대상:
  집합 A: 전체 아동 중 GT 라벨이 있는 사례 (NEG 여부 무관)
  집합 B: NEG(부정군) 내에서 GT 라벨이 있는 사례

차집합 A - B = "GT 라벨이 있지만 NEG가 아닌 사례"
→ 이 사례들의 상세 정보를 출력하여, 왜 GT가 있는데 부정군이
  아닌지를 진단한다.

실행:
  python debug_gt_1350_vs_1343.py --data_dir ./data
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

# ── sys.path bootstrap ──
_this = Path(__file__).resolve().parent
for p in [_this.parent, _this.parent.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from abuse_pipeline.core.common import ABUSE_ORDER, DATA_JSON_DIR

# GT 추출용 로컬 헬퍼 (core.labels 비의존)
_GT_CANON_MAP = {
    "신체적학대": "신체학대", "정서적학대": "정서학대",
    "성적학대": "성학대", "성폭력": "성학대", "성폭행": "성학대",
    "유기": "방임",
}


def _to_text(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, list): return " ".join(_to_text(v) for v in x)
    if isinstance(x, dict):
        for k in ("val", "text", "value"):
            if k in x: return _to_text(x.get(k))
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _normalize_gt(label: str) -> str:
    s = re.sub(r"\s+", "", str(label))
    s = _GT_CANON_MAP.get(s, s)
    if s.endswith("적학대"):
        s = s.replace("적학대", "학대")
    return _GT_CANON_MAP.get(s, s)


def _extract_gt_labels(info: dict) -> Set[str]:
    raw = info.get("학대의심", "")
    text = _to_text(raw)
    text_nospace = re.sub(r"\s+", "", text)
    if not text_nospace:
        return set()
    found: Set[str] = set()
    for m in re.findall(r"([가-힣]+?)학대", text_nospace):
        cand = _normalize_gt(m + "학대")
        if cand in set(ABUSE_ORDER):
            found.add(cand)
    if "방임" in text_nospace:
        found.add("방임")
    if "유기" in text_nospace:
        found.add("방임")
    for a in ABUSE_ORDER:
        if re.sub(r"\s+", "", a) in text_nospace:
            found.add(a)
    return {a for a in found if a in set(ABUSE_ORDER)}


def _extract_abuse_scores(rec: dict) -> Dict[str, int]:
    scores = {a: 0 for a in ABUSE_ORDER}
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
            for a in ABUSE_ORDER:
                if a in name:
                    scores[a] += sc
    return scores


def _find_json_files(data_dir: str | Path) -> List[Path]:
    d = Path(data_dir)
    if not d.exists():
        return []
    return sorted(d.rglob("*.json"))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(obj: Any) -> list:
    if isinstance(obj, dict) and ("info" in obj or "list" in obj):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def main(data_dir: str | None = None):
    # Algorithm 1 import (NEG 판정용)
    from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub

    data_dir = data_dir or DATA_JSON_DIR
    json_files = _find_json_files(data_dir)
    print(f"JSON 파일 수: {len(json_files)}")
    if not json_files:
        print("[ERROR] JSON 파일 없음. --data_dir 확인 필요.")
        return

    # ── 모든 사례를 순회하며 두 집합을 동시에 구성 ──────────
    all_gt_ids: Set[str] = set()       # 집합 A: 전체 GT
    neg_gt_ids: Set[str] = set()       # 집합 B: NEG 내 GT
    case_details: Dict[str, dict] = {}

    n_total = 0
    n_neg = 0

    for fp in json_files:
        try:
            obj = _read_json(fp)
        except Exception:
            continue

        for ridx, rec in enumerate(_iter_records(obj)):
            info = rec.get("info", {}) or {}
            case_id = info.get("ID") or info.get("id") or info.get("Id")
            if not case_id:
                case_id = f"{fp.stem}__{ridx}"
            case_id = str(case_id)
            n_total += 1

            # GT 추출 (라벨 알고리즘 비의존)
            gt_set = _extract_gt_labels(info)
            has_gt = len(gt_set) > 0

            # Algorithm 1: 정서군 판정
            try:
                valence = classify_child_group(rec)
            except Exception:
                valence = None

            is_neg = (valence == "부정")
            if is_neg:
                n_neg += 1

            # Algorithm 2: 학대유형 (참고용)
            try:
                main_abuse, subs = classify_abuse_main_sub(rec)
            except Exception:
                main_abuse, subs = None, []

            a_scores = _extract_abuse_scores(rec)

            if has_gt:
                all_gt_ids.add(case_id)
                if is_neg:
                    neg_gt_ids.add(case_id)

            case_details[case_id] = {
                "valence": valence,
                "crisis": info.get("위기단계"),
                "total_score": info.get("합계점수"),
                "main_abuse": main_abuse,
                "sub_abuses": list(subs or []),
                "gt_raw": info.get("학대의심", ""),
                "gt_labels": gt_set,
                "a_scores": a_scores,
            }

    # ══════════════════════════════════════════════════════════
    #  결과 출력
    # ══════════════════════════════════════════════════════════
    outside_neg = all_gt_ids - neg_gt_ids  # GT는 있지만 NEG가 아닌 사례

    print(f"\n{'═' * 65}")
    print(f"전체 아동 수:                   {n_total}")
    print(f"NEG(부정군) 아동 수:            {n_neg}")
    print(f"[집합 A] 전체 GT 보유:          {len(all_gt_ids)}")
    print(f"[집합 B] NEG 내 GT 보유:        {len(neg_gt_ids)}")
    print(f"[A - B]  GT 있지만 NEG 밖:      {len(outside_neg)}")
    print(f"{'═' * 65}")

    if not outside_neg:
        print("\n✓ 모든 GT 사례가 NEG 내에 있습니다. 차이 없음.")
        return

    # ── 각 사례 상세 출력 ─────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"GT 라벨이 있지만 부정군(NEG)이 아닌 {len(outside_neg)}건의 상세:")
    print(f"{'─' * 65}")

    valence_counts: Dict[str, int] = {}

    for cid in sorted(outside_neg):
        d = case_details[cid]
        crisis = d["crisis"]
        total = d["total_score"]
        valence = d["valence"]
        main_abuse = d["main_abuse"]
        subs = d["sub_abuses"]
        gt_raw = d["gt_raw"]
        gt_labels = d["gt_labels"]
        a = d["a_scores"]

        valence_counts[valence] = valence_counts.get(valence, 0) + 1

        print(f"\n--- {cid} ---")
        print(f"  정서군(Algorithm 1) = {valence}")
        print(f"  위기단계            = {crisis}")
        print(f"  합계점수            = {total}")
        print(f"  학대유형(Algo 2)    = main={main_abuse}, subs={subs}")
        print(f"  A_k = 방임={a.get('방임',0)}, 정서={a.get('정서학대',0)}, "
              f"신체={a.get('신체학대',0)}, 성={a.get('성학대',0)}")
        print(f"  info['학대의심']    = {gt_raw!r}")
        print(f"  GT 추출 결과        = {gt_labels}")

        # 원인 힌트
        if valence == "긍정":
            print(f"  → 긍정군: Algorithm 1이 보호요인/낮은 점수로 긍정 판정")
        elif valence == "평범":
            print(f"  → 평범군: Algorithm 1이 경계 영역으로 판정")
        elif valence is None:
            print(f"  → 정서군 판정 실패 (None)")

    # ── 요약 통계 ─────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"[요약] GT 있지만 NEG 밖인 {len(outside_neg)}건의 정서군 분포:")
    for v, cnt in sorted(valence_counts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {cnt}건")

    print(f"\n[해석]")
    print(f"  이 사례들은 임상가가 학대의심 라벨(GT)을 부여했지만,")
    print(f"  Algorithm 1(정서군 분류)에서 부정군으로 분류되지 않은 아동이다.")
    print(f"  즉, 임상가 판단(학대 의심)과 알고리즘 판단(비부정군)이 불일치한다.")
    print(f"{'═' * 65}")

    # ── CSV 저장 ──────────────────────────────────────────────
    out_dir = Path(data_dir).parent / "outputs" / "revision" / "raw_score_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in sorted(outside_neg):
        d = case_details[cid]
        a = d["a_scores"]
        rows.append({
            "case_id": cid,
            "valence_algo1": d["valence"],
            "crisis": d["crisis"],
            "total_score": d["total_score"],
            "main_abuse_algo2": d["main_abuse"],
            "sub_abuses": str(d["sub_abuses"]),
            "gt_raw": str(d["gt_raw"]),
            "gt_labels": str(d["gt_labels"]),
            "A_neglect": a.get("방임", 0),
            "A_emotional": a.get("정서학대", 0),
            "A_physical": a.get("신체학대", 0),
            "A_sexual": a.get("성학대", 0),
        })

    df = pd.DataFrame(rows)
    path = out_dir / "gt_outside_neg_diagnosis.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 GT vs NEG 내 GT 차이 진단")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir)
