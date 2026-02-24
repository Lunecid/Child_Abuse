"""
borderline_case_explorer.py
═══════════════════════════════════════════════════════════════════
"학대 점수는 있지만 높지 않은 사례"의 임상 텍스트·학대의심 텍스트 탐색

목적
────
논문의 ABUSE_NEG 코퍼스(n=1,496)는 main = argmax_{k: A_k > 6} A_k 규칙으로
주 학대유형이 부여된 사례만 포함한다. 그런데 학대 점수가 0은 아니면서 6 이하인
"경계선(borderline)" 사례들은 무엇일까?

이 스크립트는 다음 세 가지 계층의 사례를 추출·비교한다:
  (A) CLEAR:      max(A_k) > 6  → ABUSE_NEG에 포함된 확실한 사례
  (B) BORDERLINE:  0 < max(A_k) ≤ 6  → 점수는 있지만 임계값 미달
  (C) ZERO:        모든 A_k = 0  → 학대 점수가 전혀 없는 사례

각 계층별로 다음을 추출·출력한다:
  1) 학대 점수 분포 (유형별)
  2) 임상 텍스트 (임상진단, 임상가 종합소견, 학대의심 유형)
  3) 아동 발화 (상담 녹취록에서 아동 응답만)
  4) 메타 정보 (위기단계, 합계점수 등)

사용법
──────
  python borderline_case_explorer.py --data_dir /path/to/data

  --data_dir : JSON 파일들이 있는 폴더 (기본: ./data)
  --out_dir  : 결과 저장 폴더 (기본: ./borderline_output)
  --max_examples : 각 계층별 출력할 상세 사례 수 (기본: 20)
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ─── common 참조 ───
from abuse_pipeline.core import common as _C

ABUSE_ORDER = getattr(_C, "ABUSE_ORDER", None) or ["성학대", "신체학대", "정서학대", "방임"]
ABUSE_EN = {"성학대": "Sexual", "신체학대": "Physical", "정서학대": "Emotional", "방임": "Neglect"}
SEVERITY_RANK = getattr(_C, "SEVERITY_RANK", None) or {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}


# ═══════════════════════════════════════════════════════════════
#  1. 데이터 추출 함수들
# ═══════════════════════════════════════════════════════════════

def extract_abuse_scores(rec: dict) -> dict:
    """
    JSON record에서 학대여부 문항의 유형별 점수를 추출한다.

    JSON 구조:
      rec["list"] → 7개 대문항 리스트
        └ 각 문항["문항"] == "학대여부" 인 항목을 찾아
          └ 하위 항목["list"]에서 "방임", "정서학대", "신체학대", "성학대" 포함 항목의 점수 합산

    Returns: {"성학대": int, "신체학대": int, "정서학대": int, "방임": int}
    """
    scores = {a: 0 for a in ABUSE_ORDER}
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
                        scores[a] += sc
    return scores


def extract_clinical_text(rec: dict) -> dict:
    """
    JSON record에서 임상 관련 텍스트를 추출한다.

    추출 필드:
      - info["임상진단"]       : 임상 진단명
      - info["임상가 종합소견"]  : 임상가가 작성한 종합 소견 (자유 텍스트)
      - info["학대의심"]       : 데이터셋 원본 BERT 라벨 (학대 유형명)
      - info["위기단계"]       : 위기 수준 (정상군/관찰필요/상담필요/학대의심/위기아동)
      - info["합계점수"]       : 전체 설문 합계 점수
    """
    info = rec.get("info", {}) or {}

    clinical_diagnosis = str(info.get("임상진단", "")) if info.get("임상진단") else ""
    clinical_opinion = str(info.get("임상가 종합소견", "")) if info.get("임상가 종합소견") else ""

    # 학대의심 필드는 str, list, dict 등 다양한 형태로 올 수 있음
    abuse_suspect_raw = info.get("학대의심", "")
    if isinstance(abuse_suspect_raw, str):
        abuse_suspect = abuse_suspect_raw
    elif isinstance(abuse_suspect_raw, list):
        abuse_suspect = " ".join(str(x) for x in abuse_suspect_raw)
    elif isinstance(abuse_suspect_raw, dict):
        abuse_suspect = json.dumps(abuse_suspect_raw, ensure_ascii=False)
    else:
        abuse_suspect = str(abuse_suspect_raw) if abuse_suspect_raw else ""

    return {
        "임상진단": clinical_diagnosis,
        "임상가_종합소견": clinical_opinion,
        "학대의심_유형": abuse_suspect,
        "위기단계": str(info.get("위기단계", "")),
        "합계점수": info.get("합계점수", None),
        "나이": info.get("나이", ""),
        "성별": info.get("성별", ""),
        "가정환경": str(info.get("가정환경", "")),
    }


def extract_child_speech(rec: dict) -> List[str]:
    """
    JSON record에서 아동 발화(응답)만 추출한다.

    JSON 구조:
      rec["list"][i]["list"][j]["audio"] → 오디오 세그먼트 리스트
        └ seg["type"] == "A" 인 것만 (A=아동 응답, Q=상담사 질문)
        └ seg["text"] : 발화 텍스트
    """
    texts = []
    for q in rec.get("list", []):
        for it in q.get("list", []):
            for seg in it.get("audio", []):
                if seg.get("type") == "A":
                    t = seg.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
    return texts


def extract_qa_pairs(rec: dict) -> List[dict]:
    """
    JSON record에서 Q-A 쌍을 시간순으로 추출한다.
    상담사 질문(Q)과 아동 응답(A)을 쌍으로 묶어 대화 맥락을 파악한다.
    """
    pairs = []
    for q_block in rec.get("list", []):
        qname = q_block.get("문항", "")
        for it in q_block.get("list", []):
            iname = it.get("항목", "")
            pending_q = None
            for seg in it.get("audio", []):
                seg_type = seg.get("type")
                text = seg.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                text = text.strip()

                if seg_type == "Q":
                    pending_q = text
                elif seg_type == "A":
                    pairs.append({
                        "문항": qname,
                        "항목": iname,
                        "Q": pending_q or "(질문 없음)",
                        "A": text,
                    })
                    pending_q = None
    return pairs


def classify_case_tier(scores: dict) -> str:
    """
    학대 점수를 기반으로 사례 계층을 분류한다.

    - CLEAR (확실):     max(A_k) > 6  → ABUSE_NEG에 포함
    - BORDERLINE (경계):  0 < max(A_k) ≤ 6  → 점수는 있지만 임계값 미달
    - ZERO (없음):        모든 A_k = 0  → 학대 점수 없음
    """
    max_score = max(scores.values())
    total = sum(scores.values())

    if max_score > 6:
        return "CLEAR"
    elif total > 0:
        return "BORDERLINE"
    else:
        return "ZERO"


# ═══════════════════════════════════════════════════════════════
#  2. 메인 분석 함수
# ═══════════════════════════════════════════════════════════════

def analyze_all_cases(data_dir: str) -> List[dict]:
    """모든 JSON 파일을 읽고 사례별 정보를 추출한다."""
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print(f"\n{'=' * 70}")
    print(f"  JSON 파일 수: {len(json_files)}")
    print(f"  데이터 경로: {data_dir}")
    print(f"{'=' * 70}\n")

    all_cases = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  [WARN] 로드 실패: {os.path.basename(path)} ({e})")
            continue

        info = rec.get("info", {}) or {}
        child_id = info.get("ID", os.path.basename(path))

        # 학대 점수 추출
        scores = extract_abuse_scores(rec)

        # 임상 텍스트 추출
        clinical = extract_clinical_text(rec)

        # 아동 발화 추출
        speech = extract_child_speech(rec)

        # Q-A 쌍 추출
        qa_pairs = extract_qa_pairs(rec)

        # 계층 분류
        tier = classify_case_tier(scores)

        # 학대 유형별 임상소견 내 키워드 존재 여부
        clin_text = f"{clinical['임상진단']} {clinical['임상가_종합소견']}"
        abuse_in_clinical = {a: (a in clin_text) for a in ABUSE_ORDER}

        all_cases.append({
            "child_id": child_id,
            "file": os.path.basename(path),
            "tier": tier,
            "scores": scores,
            "max_score": max(scores.values()),
            "total_score": sum(scores.values()),
            "clinical": clinical,
            "abuse_in_clinical": abuse_in_clinical,
            "speech": speech,
            "speech_joined": " ".join(speech),
            "n_utterances": len(speech),
            "n_qa_pairs": len(qa_pairs),
            "qa_pairs": qa_pairs,
        })

    return all_cases


def generate_summary_report(all_cases: List[dict], out_dir: str, max_examples: int = 20):
    """종합 분석 리포트를 생성한다."""
    os.makedirs(out_dir, exist_ok=True)

    # ── 전체 분포 ──
    tier_counts = Counter(c["tier"] for c in all_cases)

    print(f"\n{'═' * 70}")
    print(f"  전체 사례 분류 결과")
    print(f"{'═' * 70}")
    print(f"  총 사례 수: {len(all_cases)}")
    print(f"  ┌─────────────┬──────┬────────┐")
    print(f"  │ 계층        │ 사례수│ 비율   │")
    print(f"  ├─────────────┼──────┼────────┤")
    for tier in ["CLEAR", "BORDERLINE", "ZERO"]:
        n = tier_counts.get(tier, 0)
        pct = n / len(all_cases) * 100 if all_cases else 0
        desc = {"CLEAR": "확실(>6)", "BORDERLINE": "경계(1~6)", "ZERO": "없음(0)"}[tier]
        print(f"  │ {desc:<10s}│ {n:>4d} │ {pct:>5.1f}% │")
    print(f"  └─────────────┴──────┴────────┘")

    # ── 계층별 상세 분석 ──
    tiers = {"CLEAR": [], "BORDERLINE": [], "ZERO": []}
    for c in all_cases:
        tiers[c["tier"]].append(c)

    # ═══════════════════════════════════════════
    # TABLE 1: 전체 요약 CSV
    # ═══════════════════════════════════════════
    summary_rows = []
    for c in all_cases:
        row = {
            "child_id": c["child_id"],
            "tier": c["tier"],
            "max_abuse_score": c["max_score"],
            "total_abuse_score": c["total_score"],
        }
        for a in ABUSE_ORDER:
            row[f"score_{ABUSE_EN[a]}"] = c["scores"][a]
        row.update({
            "위기단계": c["clinical"]["위기단계"],
            "합계점수": c["clinical"]["합계점수"],
            "학대의심_유형": c["clinical"]["학대의심_유형"],
            "임상소견_학대키워드": "|".join(
                a for a in ABUSE_ORDER if c["abuse_in_clinical"][a]
            ),
            "n_utterances": c["n_utterances"],
            "임상가_종합소견": c["clinical"]["임상가_종합소견"][:200],
        })
        summary_rows.append(row)

    df_all = pd.DataFrame(summary_rows)
    df_all.to_csv(os.path.join(out_dir, "01_all_cases_summary.csv"),
                  encoding="utf-8-sig", index=False)

    # ═══════════════════════════════════════════
    # TABLE 2: 계층별 학대 점수 기술통계
    # ═══════════════════════════════════════════
    stat_rows = []
    for tier_name, cases in tiers.items():
        if not cases:
            continue
        for a in ABUSE_ORDER:
            vals = [c["scores"][a] for c in cases]
            nonzero = [v for v in vals if v > 0]
            stat_rows.append({
                "Tier": tier_name,
                "학대유형": a,
                "학대유형_EN": ABUSE_EN[a],
                "N": len(cases),
                "점수>0_수": len(nonzero),
                "점수>0_비율%": round(len(nonzero) / len(cases) * 100, 1) if cases else 0,
                "평균": round(np.mean(vals), 2),
                "SD": round(np.std(vals), 2),
                "중위수": np.median(vals),
                "최솟값": min(vals),
                "최댓값": max(vals),
                "Q1": np.percentile(vals, 25),
                "Q3": np.percentile(vals, 75),
            })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(os.path.join(out_dir, "02_score_stats_by_tier.csv"),
                    encoding="utf-8-sig", index=False)

    print(f"\n{'═' * 70}")
    print(f"  BORDERLINE 사례 학대 점수 분포")
    print(f"{'═' * 70}")
    bl = df_stats[df_stats["Tier"] == "BORDERLINE"]
    if not bl.empty:
        print(bl.to_string(index=False))

    # ═══════════════════════════════════════════
    # TABLE 3: BORDERLINE 사례 점수 조합 패턴
    # ═══════════════════════════════════════════
    borderline = tiers["BORDERLINE"]

    if borderline:
        pattern_rows = []
        for c in borderline:
            # 점수가 0이 아닌 유형들을 정렬
            nonzero_types = sorted(
                [(a, c["scores"][a]) for a in ABUSE_ORDER if c["scores"][a] > 0],
                key=lambda x: (-x[1], SEVERITY_RANK[x[0]])
            )
            pattern = " + ".join(f"{a}({s})" for a, s in nonzero_types)
            types_only = "+".join(a for a, s in nonzero_types)

            pattern_rows.append({
                "child_id": c["child_id"],
                "점수패턴": pattern,
                "유형조합": types_only,
                "max_score": c["max_score"],
                "total_score": c["total_score"],
                "위기단계": c["clinical"]["위기단계"],
                "학대의심_유형": c["clinical"]["학대의심_유형"],
                "임상소견_내_학대키워드": "|".join(
                    a for a in ABUSE_ORDER if c["abuse_in_clinical"][a]
                ),
            })

        df_patterns = pd.DataFrame(pattern_rows)
        df_patterns.to_csv(os.path.join(out_dir, "03_borderline_score_patterns.csv"),
                           encoding="utf-8-sig", index=False)

        # 패턴 빈도
        combo_counts = df_patterns["유형조합"].value_counts()
        print(f"\n  BORDERLINE 점수 조합 패턴 (상위 15개):")
        print(f"  {'─' * 50}")
        for combo, cnt in combo_counts.head(15).items():
            pct = cnt / len(borderline) * 100
            print(f"    {combo:<30s} : {cnt:>4d} ({pct:>5.1f}%)")

    # ═══════════════════════════════════════════
    # TABLE 4: BORDERLINE 사례의 위기단계 분포
    # ═══════════════════════════════════════════
    if borderline:
        crisis_by_tier = {}
        for tier_name, cases in tiers.items():
            if cases:
                crisis_by_tier[tier_name] = Counter(
                    c["clinical"]["위기단계"] for c in cases
                )

        print(f"\n  계층별 위기단계 분포:")
        print(f"  {'─' * 60}")
        all_crisis = sorted(set(
            c["clinical"]["위기단계"] for c in all_cases if c["clinical"]["위기단계"]
        ))
        header = f"  {'위기단계':<15s}" + "".join(f"{t:>12s}" for t in ["CLEAR", "BORDERLINE", "ZERO"])
        print(header)
        for cr in all_crisis:
            vals = []
            for t in ["CLEAR", "BORDERLINE", "ZERO"]:
                cnt = crisis_by_tier.get(t, {}).get(cr, 0)
                total = len(tiers.get(t, []))
                pct = cnt / total * 100 if total else 0
                vals.append(f"{cnt}({pct:.0f}%)")
            print(f"  {cr:<15s}" + "".join(f"{v:>12s}" for v in vals))

    # ═══════════════════════════════════════════
    # TABLE 5: BORDERLINE 임상 텍스트 상세 (상위 N건)
    # ═══════════════════════════════════════════
    if borderline:
        # max_score 내림차순으로 정렬 (임계값에 가장 가까운 사례부터)
        bl_sorted = sorted(borderline, key=lambda x: (-x["max_score"], -x["total_score"]))

        detail_rows = []
        for c in bl_sorted[:max_examples * 5]:  # 넉넉히 추출
            # 가장 높은 점수의 유형
            top_type = max(ABUSE_ORDER, key=lambda a: (c["scores"][a], -SEVERITY_RANK[a]))

            # 대표 발화 (최대 5개)
            sample_speech = c["speech"][:5]

            # 대표 Q-A (최대 3쌍)
            sample_qa = c["qa_pairs"][:3]
            qa_text = " | ".join(
                f"[Q]{qa['Q']} → [A]{qa['A']}" for qa in sample_qa
            )

            detail_rows.append({
                "child_id": c["child_id"],
                "max_score": c["max_score"],
                "total_score": c["total_score"],
                "최고점_유형": top_type,
                **{f"A_{ABUSE_EN[a]}": c["scores"][a] for a in ABUSE_ORDER},
                "위기단계": c["clinical"]["위기단계"],
                "합계점수": c["clinical"]["합계점수"],
                "학대의심_유형(BERT)": c["clinical"]["학대의심_유형"],
                "임상진단": c["clinical"]["임상진단"][:150],
                "임상가_종합소견": c["clinical"]["임상가_종합소견"][:300],
                "발화수": c["n_utterances"],
                "대표_발화": " / ".join(sample_speech)[:500],
                "대표_QA": qa_text[:500],
            })

        df_detail = pd.DataFrame(detail_rows)
        df_detail.to_csv(os.path.join(out_dir, "04_borderline_clinical_detail.csv"),
                         encoding="utf-8-sig", index=False)

    # ═══════════════════════════════════════════
    # TABLE 6: BORDERLINE 임상소견 키워드 빈도
    # ═══════════════════════════════════════════
    if borderline:
        print(f"\n{'═' * 70}")
        print(f"  BORDERLINE 사례 임상소견 내 학대 키워드 출현 빈도")
        print(f"{'═' * 70}")

        keyword_counts = {a: 0 for a in ABUSE_ORDER}
        any_keyword = 0

        for c in borderline:
            has_any = False
            for a in ABUSE_ORDER:
                if c["abuse_in_clinical"][a]:
                    keyword_counts[a] += 1
                    has_any = True
            if has_any:
                any_keyword += 1

        for a in ABUSE_ORDER:
            cnt = keyword_counts[a]
            pct = cnt / len(borderline) * 100
            print(f"  {a} ({ABUSE_EN[a]}): {cnt}건 ({pct:.1f}%)")
        print(f"  {'─' * 40}")
        print(f"  임상소견에 학대 키워드 1개 이상: {any_keyword}건 ({any_keyword / len(borderline) * 100:.1f}%)")
        print(f"  임상소견에 학대 키워드 없음:     {len(borderline) - any_keyword}건")

    # ═══════════════════════════════════════════
    # TABLE 7: BORDERLINE → CLEAR 비교 (임상 불일치 사례)
    # ═══════════════════════════════════════════
    # 임상소견에는 학대 키워드가 있지만 점수로는 BORDERLINE인 "임상-점수 불일치" 사례
    if borderline:
        mismatch_rows = []
        for c in borderline:
            if any(c["abuse_in_clinical"].values()):
                clinical_types = [a for a in ABUSE_ORDER if c["abuse_in_clinical"][a]]
                mismatch_rows.append({
                    "child_id": c["child_id"],
                    "max_score": c["max_score"],
                    "임상소견_학대유형": "|".join(clinical_types),
                    **{f"A_{ABUSE_EN[a]}": c["scores"][a] for a in ABUSE_ORDER},
                    "위기단계": c["clinical"]["위기단계"],
                    "학대의심(BERT)": c["clinical"]["학대의심_유형"],
                    "임상가_종합소견": c["clinical"]["임상가_종합소견"][:300],
                })

        if mismatch_rows:
            df_mismatch = pd.DataFrame(mismatch_rows)
            df_mismatch.to_csv(
                os.path.join(out_dir, "05_borderline_clinical_mismatch.csv"),
                encoding="utf-8-sig", index=False
            )
            print(f"\n  ★ 임상-점수 불일치 사례: {len(mismatch_rows)}건")
            print(f"     (임상소견에 학대 키워드가 있으나 max(A_k) ≤ 6)")

    # ═══════════════════════════════════════════
    # TABLE 8: BORDERLINE max_score별 세부 분포
    # ═══════════════════════════════════════════
    if borderline:
        print(f"\n{'═' * 70}")
        print(f"  BORDERLINE max_score별 분포")
        print(f"{'═' * 70}")

        max_score_dist = Counter(c["max_score"] for c in borderline)
        for score in sorted(max_score_dist.keys(), reverse=True):
            cnt = max_score_dist[score]
            pct = cnt / len(borderline) * 100
            bar = "█" * int(pct / 2)
            print(f"  max_score={score}: {cnt:>4d}건 ({pct:>5.1f}%) {bar}")

    # ═══════════════════════════════════════════
    # 상세 사례 출력 (콘솔)
    # ═══════════════════════════════════════════
    if borderline:
        # max_score가 5~6인 사례 (임계값에 가장 가까운)
        near_threshold = [c for c in borderline if c["max_score"] >= 5]
        near_threshold.sort(key=lambda x: (-x["max_score"], -x["total_score"]))

        print(f"\n{'═' * 70}")
        print(f"  BORDERLINE 중 max_score ≥ 5 상세 사례 (상위 {min(max_examples, len(near_threshold))}건)")
        print(f"  → 이 사례들은 임계값(>6)에 가장 가까워, 분류 경계를 이해하는 데 핵심적")
        print(f"{'═' * 70}")

        for i, c in enumerate(near_threshold[:max_examples]):
            print(f"\n  ┌{'─' * 66}┐")
            print(f"  │ 사례 #{i + 1}: {c['child_id']}")
            print(f"  ├{'─' * 66}┤")

            scores_str = " | ".join(
                f"{ABUSE_EN[a]}={c['scores'][a]}" for a in ABUSE_ORDER
            )
            print(f"  │ 점수: {scores_str}")
            meta_str = f"max={c['max_score']}, total={c['total_score']}, 위기단계={c['clinical']['위기단계']}, 합계점수={c['clinical']['합계점수']}"
            print(f"  │ {meta_str}│")

            # 학대의심 유형 (BERT 라벨)
            bert = c["clinical"]["학대의심_유형"]
            if bert:
                print(f"  │ 학대의심(BERT): {bert}")

            # 임상소견
            opinion = c["clinical"]["임상가_종합소견"]
            if opinion:
                lines = [opinion[j:j + 60] for j in range(0, min(len(opinion), 240), 60)]
                print(f"  │ 임상소견:")
                for line in lines:
                    print(f"  │   {line}")

            # 대표 발화
            if c["speech"]:
                print(f"  │ 아동 발화 ({c['n_utterances']}건):")
                for utt in c["speech"][:3]:
                    utt_short = utt[:60]
                    print(f"  │   \"{utt_short}\"│")

            print(f"  └{'─' * 66}┘")

    # ═══════════════════════════════════════════
    # 최종 요약 출력
    # ═══════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"  출력 파일 목록 ({out_dir}/)")
    print(f"{'═' * 70}")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  📄 {f} ({size_kb:.1f} KB)")

    return tiers


# ═══════════════════════════════════════════════════════════════
#  3. 엔트리 포인트
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="학대 점수 경계선 사례 탐색 (Borderline Case Explorer)"
    )
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="JSON 파일들이 있는 폴더 경로")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="결과 저장 폴더 (default: configure_output_dirs 기반)")
    parser.add_argument("--max_examples", type=int, default=20,
                        help="콘솔에 출력할 상세 사례 수")
    args = parser.parse_args()

    if args.out_dir is None:
        if _C is not None:
            _C.configure_output_dirs(subset_name="ALL")
            args.out_dir = _C.BORDERLINE_DIR
        else:
            args.out_dir = "./borderline_output"

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] 데이터 디렉토리를 찾을 수 없습니다: {args.data_dir}")
        print(f"  사용법: python borderline_case_explorer.py --data_dir /path/to/data")
        sys.exit(1)

    all_cases = analyze_all_cases(args.data_dir)

    if not all_cases:
        print("[ERROR] 로드된 사례가 없습니다.")
        sys.exit(1)

    tiers = generate_summary_report(all_cases, args.out_dir, args.max_examples)

    print(f"\n✅ 분석 완료!")


if __name__ == "__main__":
    main()