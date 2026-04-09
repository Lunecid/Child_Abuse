"""
inspect_no_gt_cases.py
======================
GT 라벨 없음 + algo_main 존재하는 아동(ABUSE_NEG 1,503 − GT 1,350 = 153명 중 algo_main 보유 사례) 상세 조회

출력 내용:
  - 아동 ID
  - 알고리즘이 부여한 주/부 학대유형
  - 임상가 종합소견 전문
  - 임상진단 텍스트
  - 학대여부 설문 점수 (유형별)
  - 저장: no_gt_but_algo_cases.csv / no_gt_but_algo_clinical.txt
"""

from __future__ import annotations

import os
import glob
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ── 프로젝트 모듈 import (패키지/단독 실행 폴백) ──
import sys

_this = Path(__file__).resolve()
_project_root = _this.parent.parent.parent  # investigation/ → abuse_pipeline/ → project root

from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.analysis.compare_abuse_labels import extract_gt_abuse_types_from_info

ABUSE_ORDER = C.ABUSE_ORDER  # ["성학대","신체학대","정서학대","방임"]

# ═══════════════════════════════════════════════════════════════════════
#  설정
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR      = _project_root / "data"
SUB_THRESHOLD = 4
USE_CLINICAL  = True
ONLY_NEGATIVE = True   # NEG 군만 분석

# 출력 디렉토리: configure_output_dirs 기반, fallback 시 프로젝트 루트 아래
C.configure_output_dirs(subset_name="NEG_ONLY", base_dir=str(_project_root))
OUT_DIR = Path(C.NO_GT_DIR) if C.NO_GT_DIR else _project_root / "output_inspect"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════
def main():
    json_files = sorted(glob.glob(str(DATA_DIR / "*.json")))
    print(f"JSON 파일: {len(json_files)}개")

    rows       = []   # CSV 용 (GT 없음 + algo_main 있음)
    rows_all_no_gt = []  # GT 없는 전체 사례 (algo_main 유무 무관)
    txt_lines  = []   # 임상 소견 전문 텍스트 파일 용

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        info = rec.get("info", {}) or {}

        # ── 군 필터 ──
        if ONLY_NEGATIVE:
            group = classify_child_group(rec)
            if group != "부정":
                continue

        # ── GT 라벨 ──
        gt_set   = extract_gt_abuse_types_from_info(info, field="학대의심")
        gt_label = (sorted(gt_set)[0] if gt_set else None)

        # ── 알고리즘 라벨 ──
        algo_main, algo_subs = classify_abuse_main_sub(
            rec,
            sub_threshold=SUB_THRESHOLD,
            use_clinical_text=USE_CLINICAL,
        )
        algo_subs = algo_subs or []

        # ── GT 있으면 제외 ──
        if gt_label is not None:
            continue

        # ── 회수 집단 전체 기록 (algo_main 유무 무관) ──
        doc_id_all = (info.get("ID") or info.get("id") or info.get("Id")
                      or os.path.splitext(os.path.basename(path))[0])
        rows_all_no_gt.append({
            "doc_id": doc_id_all,
            "algo_main": algo_main or "",
            "has_algo_main": int(bool(algo_main)),
        })

        # ── 필터: algo_main 없으면 상세 조회 대상에서 제외 ──
        if not algo_main:
            continue

        # ── 아동 ID ──
        doc_id = (info.get("ID") or info.get("id") or info.get("Id")
                  or os.path.splitext(os.path.basename(path))[0])

        # ── 학대여부 설문 점수 수집 ──
        abuse_scores = {a: 0 for a in ABUSE_ORDER}
        for q in rec.get("list", []):
            if q.get("문항") == "학대여부":
                for it in q.get("list", []):
                    name = it.get("항목", "")
                    try:
                        sc = int(it.get("점수", 0))
                    except (TypeError, ValueError):
                        sc = 0
                    for a in ABUSE_ORDER:
                        if a in str(name):
                            abuse_scores[a] += sc

        # ── 임상 텍스트 ──
        clin_diag    = str(info.get("임상진단", "")).strip()
        clin_opinion = str(info.get("임상가 종합소견", "")).strip()
        clin_combined = " | ".join(filter(None, [clin_diag, clin_opinion]))

        # ── CSV 행 ──
        rows.append(dict(
            doc_id       = doc_id,
            algo_main    = algo_main,
            algo_subs    = "|".join(sorted(algo_subs)),
            score_성학대  = abuse_scores["성학대"],
            score_신체학대 = abuse_scores["신체학대"],
            score_정서학대 = abuse_scores["정서학대"],
            score_방임    = abuse_scores["방임"],
            clin_diag    = clin_diag[:200],        # CSV에는 앞 200자만
            clin_opinion = clin_opinion[:500],     # CSV에는 앞 500자만
        ))

        # ── 텍스트 파일 (전문) ──
        txt_lines.append("=" * 72)
        txt_lines.append(f"[{len(rows):>3d}] ID: {doc_id}")
        txt_lines.append(f"      algo_main : {algo_main}")
        txt_lines.append(f"      algo_subs : {', '.join(algo_subs) if algo_subs else '없음'}")
        txt_lines.append(f"      설문 점수 : " +
                         " | ".join(f"{a}={abuse_scores[a]}" for a in ABUSE_ORDER))
        txt_lines.append(f"      gt_label  : 없음 (제외 사유 확인 대상)")
        txt_lines.append("")
        txt_lines.append("  [임상진단]")
        txt_lines.append(f"  {clin_diag if clin_diag else '(없음)'}")
        txt_lines.append("")
        txt_lines.append("  [임상가 종합소견]")
        txt_lines.append(f"  {clin_opinion if clin_opinion else '(없음)'}")
        txt_lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    #  회수 집단 전체 요약 (ABUSE_NEG − GT = 153명)
    # ═══════════════════════════════════════════════════════════════════
    n_all_no_gt = len(rows_all_no_gt)
    print(f"\n  [회수 집단] ABUSE_NEG 중 GT 없는 전체: {n_all_no_gt}명")

    if rows_all_no_gt:
        df_all_no_gt = pd.DataFrame(rows_all_no_gt)
        n_with_algo = int(df_all_no_gt["has_algo_main"].sum())
        n_without_algo = n_all_no_gt - n_with_algo
        print(f"    algo_main 있음: {n_with_algo}명 / algo_main 없음: {n_without_algo}명")

        # 회수 집단 유형별 분포 (algo_main 기준)
        algo_dist = df_all_no_gt[df_all_no_gt["algo_main"] != ""]["algo_main"].value_counts()
        print(f"\n  [회수 집단 주 학대유형 분포 — algo_main 기준]")
        for atype in ABUSE_ORDER:
            cnt = algo_dist.get(atype, 0)
            pct = cnt / n_all_no_gt if n_all_no_gt else 0
            print(f"    {atype}: {cnt}명 ({pct:.1%})")

        # CSV 저장
        recovery_csv = OUT_DIR / "recovery_group_distribution.csv"
        df_all_no_gt.to_csv(recovery_csv, encoding="utf-8-sig", index=False)
        print(f"\n  회수 집단 CSV 저장: {recovery_csv}")

    # ═══════════════════════════════════════════════════════════════════
    #  상세 결과 (GT 없음 + algo_main 있음)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n  GT 없음 + algo_main 있음: {len(rows)}명 검출")

    if not rows:
        print("  ⚠ 해당 케이스 없음")
        return

    df = pd.DataFrame(rows)

    # algo_main 분포
    print("\n  [algo_main 분포]")
    for atype, cnt in df["algo_main"].value_counts().items():
        print(f"    {atype}: {cnt}명 ({cnt/len(df):.1%})")

    # algo_subs 보유 비율
    has_sub = (df["algo_subs"] != "").sum()
    print(f"\n  [Sub 보유 비율] {has_sub}/{len(df)} ({has_sub/len(df):.1%})")

    # 임상 소견 존재 비율
    has_opinion = (df["clin_opinion"].str.strip() != "").sum()
    print(f"  [임상 소견 존재] {has_opinion}/{len(df)} ({has_opinion/len(df):.1%})")

    # 임상 소견 없는 케이스 (GT가 없는 핵심 이유 후보)
    no_opinion = df[df["clin_opinion"].str.strip() == ""]
    print(f"  [임상 소견 없음] {len(no_opinion)}명 → GT 부재 원인 후보")

    # ── 저장 ──
    csv_path = OUT_DIR / "no_gt_but_algo_cases.csv"
    txt_path = OUT_DIR / "no_gt_but_algo_clinical.txt"

    df.to_csv(csv_path, encoding="utf-8-sig", index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print(f"\n  저장 완료:")
    print(f"    CSV (요약): {csv_path}")
    print(f"    TXT (전문): {txt_path}")

    # ── 콘솔 미리보기 (상위 5건) ──
    print(f"\n{'═'*72}")
    print("  콘솔 미리보기 (상위 5건 전문)")
    print(f"{'═'*72}")
    for line in txt_lines[:200]:   # 최대 200줄
        print(line)
        if line.startswith("=" * 72) and txt_lines.index(line) > 10:
            # 5번째 케이스 구분선 이후 중단
            shown = sum(1 for l in txt_lines[:txt_lines.index(line)+1]
                       if l.startswith("="))
            if shown >= 6:
                print("  ... (이하 생략, TXT 파일에서 전체 확인)")
                break


if __name__ == "__main__":
    main()
