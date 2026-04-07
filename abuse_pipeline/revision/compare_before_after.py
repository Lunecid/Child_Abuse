#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_before_after.py
════════════════════════════════════════════════════════════════
알고리즘 1 수정(GT→NEG 규칙 추가) 전후의 핵심 지표를 비교한다.

기존 결과(old_ 값)는 본문에 이미 보고된 숫자를 하드코딩하고,
새 결과(new_ 값)는 수정된 코드로 재계산한다.

실행:
  python compare_before_after.py --data_dir ./data
  python compare_before_after.py --data_dir ./data --out_dir ./outputs/revision

산출물:
  before_after_comparison.csv   — 지표별 old/new/diff 표
  before_after_comparison.md    — 마크다운 형식 동일 표
  corpus_counts_new.json        — 새 코퍼스 사례 수 상세
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

# ── sys.path bootstrap ──
_this = Path(__file__).resolve().parent
for p in [_this.parent, _this.parent.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from abuse_pipeline.core.common import ABUSE_ORDER, DATA_JSON_DIR, BASE_DIR
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.raw_score_entropy import (
    shannon_entropy_from_raw,
    normalized_score_distribution,
)

# ── JSON I/O ──
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(obj: Any) -> list:
    if isinstance(obj, dict) and ("info" in obj or "list" in obj):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


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


# ═══════════════════════════════════════════════════════════════
#  기존 결과 (본문에 보고된 숫자)
# ═══════════════════════════════════════════════════════════════
OLD_VALUES = {
    "N_total": 3236,
    "N_NEG": 1970,
    "N_NEU": None,       # 본문 미보고 → 재계산 시 채움
    "N_POS": None,       # 본문 미보고 → 재계산 시 채움
    "N_ABUSE_NEG": 1496,
    "N_GT_ABUSE_NEG": 1343,
    "delta_h_pct_nonzero": 51.2,   # %
    "delta_h_mean_nonzero": 1.107,  # bits
    "n_dual_stable_bridge": 5,
    "CA_dist_emotional_physical": 0.402,
}


def main(data_dir: str | None = None, out_dir: str | None = None):
    data_dir = data_dir or DATA_JSON_DIR
    d = Path(data_dir)
    json_files = sorted(d.rglob("*.json")) if d.is_dir() else []
    print(f"JSON 파일 수: {len(json_files)}")
    if not json_files:
        print("[ERROR] JSON 파일 없음")
        return

    out_dir = Path(out_dir) if out_dir else Path(BASE_DIR) / "outputs" / "revision" / "raw_score_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 새 결과 계산 ──────────────────────────────────────────
    n_total = 0
    valence_counts: Counter = Counter()
    abuse_type_counts: Counter = Counter()
    n_abuse_neg = 0
    n_gt_abuse_neg = 0
    n_gt_all = 0
    delta_h_values: List[float] = []

    for fp in json_files:
        try:
            obj = _read_json(fp)
        except Exception:
            continue

        for ridx, rec in enumerate(_iter_records(obj)):
            info = rec.get("info", {}) or {}
            n_total += 1

            # Algorithm 1 (수정 버전: GT→NEG 포함)
            try:
                valence = classify_child_group(rec)
            except Exception:
                valence = None
            valence_counts[valence] += 1

            is_neg = (valence == "부정")

            # Algorithm 2
            try:
                main_abuse, subs = classify_abuse_main_sub(rec)
            except Exception:
                main_abuse, subs = None, []

            # GT 판정 (labels.py의 _extract_gt_main과 동일)
            from abuse_pipeline.core.labels import _extract_gt_main
            gt_main = _extract_gt_main(info, list(ABUSE_ORDER))
            has_gt = gt_main is not None

            if has_gt:
                n_gt_all += 1

            if is_neg and main_abuse is not None:
                n_abuse_neg += 1
                if main_abuse:
                    abuse_type_counts[main_abuse] += 1
                if has_gt:
                    n_gt_abuse_neg += 1

            # ΔH (원점수 기반)
            a_scores = _extract_abuse_scores(rec)
            a_vec = np.array([a_scores.get(a, 0) for a in
                              ["방임", "정서학대", "신체학대", "성학대"]], dtype=float)
            if a_vec.sum() > 0 and is_neg and main_abuse is not None:
                h = shannon_entropy_from_raw(a_vec, base=2)
                if not np.isnan(h) and h > 0:
                    delta_h_values.append(h)

    n_neg = valence_counts.get("부정", 0)
    n_neu = valence_counts.get("평범", 0)
    n_pos = valence_counts.get("긍정", 0)
    n_none = valence_counts.get(None, 0)

    dh_arr = np.array(delta_h_values) if delta_h_values else np.array([])
    dh_pct_nonzero = len(dh_arr) / n_abuse_neg * 100 if n_abuse_neg > 0 else 0.0
    dh_mean_nonzero = float(np.mean(dh_arr)) if len(dh_arr) > 0 else 0.0

    NEW_VALUES = {
        "N_total": n_total,
        "N_NEG": n_neg,
        "N_NEU": n_neu,
        "N_POS": n_pos,
        "N_valence_None": n_none,
        "N_GT_all": n_gt_all,
        "N_ABUSE_NEG": n_abuse_neg,
        "N_GT_ABUSE_NEG": n_gt_abuse_neg,
        "delta_h_pct_nonzero": round(dh_pct_nonzero, 1),
        "delta_h_mean_nonzero": round(dh_mean_nonzero, 3),
    }

    # ── 비교 표 작성 ─────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"{'지표':<30} {'이전':>10} {'이후':>10} {'차이':>10}")
    print(f"{'═' * 70}")

    rows = []
    for key in ["N_total", "N_NEG", "N_NEU", "N_POS", "N_ABUSE_NEG",
                 "N_GT_ABUSE_NEG", "delta_h_pct_nonzero", "delta_h_mean_nonzero"]:
        old = OLD_VALUES.get(key)
        new = NEW_VALUES.get(key)
        if old is not None and new is not None:
            diff = new - old
            diff_str = f"+{diff}" if diff > 0 else str(diff)
        elif old is None:
            diff_str = "(신규)"
        else:
            diff_str = "?"

        old_str = str(old) if old is not None else "—"
        new_str = str(new) if new is not None else "—"
        print(f"  {key:<28} {old_str:>10} {new_str:>10} {diff_str:>10}")
        rows.append({"metric": key, "old": old, "new": new, "diff": diff_str})

    # 추가: GT 전체 (NEG 무관)
    print(f"\n  {'N_GT_all (참고)':<28} {'—':>10} {NEW_VALUES['N_GT_all']:>10}")
    print(f"  {'N_valence_None':<28} {'—':>10} {NEW_VALUES.get('N_valence_None', 0):>10}")

    # 학대유형별 분포
    print(f"\n{'─' * 70}")
    print(f"학대유형별 분포 (ABUSE_NEG, N={n_abuse_neg}):")
    for a in ABUSE_ORDER:
        cnt = abuse_type_counts.get(a, 0)
        pct = cnt / n_abuse_neg * 100 if n_abuse_neg > 0 else 0
        print(f"  {a}: {cnt}명 ({pct:.1f}%)")
    print(f"{'═' * 70}")

    # ── 저장 ──────────────────────────────────────────────────
    # CSV
    df = pd.DataFrame(rows)
    csv_path = out_dir / "before_after_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {csv_path}")

    # Markdown
    md_path = out_dir / "before_after_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| 지표 | 이전 | 이후 | 차이 |\n")
        f.write("|------|------|------|------|\n")
        for r in rows:
            f.write(f"| {r['metric']} | {r['old']} | {r['new']} | {r['diff']} |\n")
        f.write(f"\n\n### 학대유형별 분포 (ABUSE_NEG, N={n_abuse_neg})\n\n")
        f.write("| 유형 | 수 | 비율 |\n|------|------|------|\n")
        for a in ABUSE_ORDER:
            cnt = abuse_type_counts.get(a, 0)
            pct = cnt / n_abuse_neg * 100 if n_abuse_neg > 0 else 0
            f.write(f"| {a} | {cnt} | {pct:.1f}% |\n")
    print(f"[SAVE] {md_path}")

    # JSON
    detail = {
        "new_values": NEW_VALUES,
        "old_values": {k: v for k, v in OLD_VALUES.items() if v is not None},
        "abuse_type_distribution": dict(abuse_type_counts),
        "valence_distribution": dict(valence_counts),
    }
    json_path = out_dir / "corpus_counts_new.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="알고리즘 1 수정 전후 비교")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir)
