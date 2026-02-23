"""
count_gt_abuse_types.py
════════════════════════════════════════════════════════════════
원본(Ground Truth) 데이터 기준 학대유형별 아동 수 카운팅

목적
────
JSON 데이터의 info["학대의심"] 필드에 기록된
임상가 원본 라벨(Ground Truth)을 읽어서
학대유형별 아동 수를 집계한다.

  ※ 알고리즘(classify_abuse_main_sub)을 전혀 사용하지 않음
  ※ 데이터에 기록된 값 그대로를 읽는 것이므로
     "원본 데이터에 존재하는 학대유형"의 n을 보여준다.

집계 기준
──────────
  · info["학대의심"] 필드에 해당 학대유형 키워드가 포함되면 n = 1
  · 한 아동이 "신체학대, 정서학대" 처럼 복수 라벨을 가질 수 있음
  · 학대의심 필드가 비어있거나 없으면 '학대유형 없음'으로 분류

출력 예시
─────────
  ══════════════════════════════════════════════════
    [전체 (ALL)] 분석 대상 아동 수 : 3596명
  ══════════════════════════════════════════════════
    학대유형       count      pct
    성학대           412    11.5%
    신체학대         852    23.7%
    정서학대         934    26.0%
    방임            1021    28.4%
    ──────────────────────────────────────
    학대유형 ≥ 1개 : 1896명 (52.7%)
    학대유형 없음   : 1700명 (47.3%)
    ──────────────────────────────────────
    ── 아동별 보유 학대유형 수 분포 ──
      0개 : 1700명
      1개 :  945명
      2개 :  712명
      ...

실행 방법
──────────
  python count_gt_abuse_types.py
  python count_gt_abuse_types.py --data_dir /path/to/data
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import glob
import json
import re
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ════════════════════════════════════════════════════════════════
#  0. 상수
# ════════════════════════════════════════════════════════════════
ABUSE_ORDER = ["성학대", "신체학대", "정서학대", "방임"]
ABUSE_LABEL_EN = {
    "성학대":   "Sexual",
    "신체학대": "Physical",
    "정서학대": "Emotional",
    "방임":     "Neglect",
}
_SEVERITY_RANK = {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}

# 표기 변형 → 표준형 정규화 맵
_CANON_MAP = {
    "신체적학대": "신체학대",
    "정서적학대": "정서학대",
    "성적학대":   "성학대",
    "성폭력":     "성학대",
    "성폭행":     "성학대",
    "유기":       "방임",
}


# ════════════════════════════════════════════════════════════════
#  1. 프로젝트 모듈 Import — 3단계 폴백
#     compare_abuse_labels.py의 extract_gt_abuse_types_from_info 사용
# ════════════════════════════════════════════════════════════════
_IMPORT_OK = False

# 폴백 1: 상대 import (패키지 내부 실행 시)
if not _IMPORT_OK:
    try:
        from .compare_abuse_labels import extract_gt_abuse_types_from_info
        from .labels import classify_child_group
        _IMPORT_OK = True
    except ImportError:
        pass

# 폴백 2: 절대 import (프로젝트 루트가 sys.path인 경우)
if not _IMPORT_OK:
    try:
        from abuse_pipeline.compare_abuse_labels import (
            extract_gt_abuse_types_from_info,
        )
        from abuse_pipeline.labels import classify_child_group
        _IMPORT_OK = True
    except ImportError:
        pass

# 폴백 3: sys.path 수동 추가
if not _IMPORT_OK:
    _this = Path(__file__).resolve()
    _proj_root = _this.parent.parent
    s = str(_proj_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    try:
        from abuse_pipeline.compare_abuse_labels import (
            extract_gt_abuse_types_from_info,
        )
        from abuse_pipeline.labels import classify_child_group
        _IMPORT_OK = True
    except ImportError:
        pass

# 폴백 4: 프로젝트 모듈 없이 로컬 함수로 대체
if not _IMPORT_OK:
    print("[IMPORT] 프로젝트 모듈 없음 → 로컬 함수 사용")

    def _to_text(x: Any) -> str:
        """학대의심 필드 값이 str / list / dict 어떤 형태든 문자열로 변환."""
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return " ".join(_to_text(v) for v in x)
        if isinstance(x, dict):
            for k in ("val", "text", "value"):
                if k in x:
                    return _to_text(x[k])
            return json.dumps(x, ensure_ascii=False)
        return str(x)

    def normalize_abuse_label(label: str) -> str:
        """표기 변형을 표준형으로 정규화한다."""
        s = re.sub(r"\s+", "", str(label))
        s = _CANON_MAP.get(s, s)
        if s.endswith("적학대"):
            s = s.replace("적학대", "학대")
        return _CANON_MAP.get(s, s)

    def extract_gt_abuse_types_from_info(
        info: Dict[str, Any],
        field: str = "학대의심",
    ) -> Set[str]:
        """
        info[field] 텍스트에서 학대유형 키워드를 추출한다.

        추출 규칙
        ---------
        1) "[가-힣]+학대" 패턴 → 신체학대, 정서학대, 성학대 등
        2) "방임" / "유기" → 방임으로 정규화
        3) 이미 표준형 키워드(ABUSE_ORDER)가 직접 등장하면 포함

        예시
        -----
        "신체학대, 정서학대" → {"신체학대", "정서학대"}
        "성적학대"           → {"성학대"}
        "방임 및 정서적학대" → {"방임", "정서학대"}
        """
        raw  = info.get(field, "")
        text = _to_text(raw)
        text_ns = re.sub(r"\s+", "", text)  # 공백 제거 버전

        found: Set[str] = set()

        # 규칙 1: "[가-힣]+학대" 패턴 (신체학대, 정서학대, 성학대 등)
        for m in re.findall(r"([가-힣]+?)학대", text_ns):
            cand = normalize_abuse_label(m + "학대")
            if cand in ABUSE_ORDER:
                found.add(cand)

        # 규칙 2: 방임 / 유기
        if "방임" in text_ns:
            found.add("방임")
        if "유기" in text_ns:
            found.add("방임")

        # 규칙 3: 표준형 키워드 직접 매칭 (혹시 위에서 못 잡은 경우 보완)
        for a in ABUSE_ORDER:
            if re.sub(r"\s+", "", a) in text_ns:
                found.add(a)

        return found

    def classify_child_group(rec: dict) -> Optional[str]:
        """정서군 분류 (부정 / 평범 / 긍정)."""
        info  = rec.get("info", {})
        crisis = info.get("위기단계")
        try:
            total_score = int(info.get("합계점수"))
        except (TypeError, ValueError):
            total_score = None

        q_sum: Dict[str, int] = {}
        item_scores: Dict[str, List[int]] = {}
        for q in rec.get("list", []):
            qname = q.get("문항")
            try:
                qt = int(q.get("문항합계"))
            except (TypeError, ValueError):
                qt = None
            if qname:
                q_sum[qname] = qt
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


# ════════════════════════════════════════════════════════════════
#  2. 데이터 로드
# ════════════════════════════════════════════════════════════════

def load_records(data_dir: str) -> List[dict]:
    """
    data_dir 안의 모든 JSON 파일을 로드한다.

    Parameters
    ----------
    data_dir : str
        JSON 파일이 위치한 디렉토리

    Returns
    -------
    list[dict] : 각 아동의 JSON record 리스트
    """
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not json_files:
        print(f"[ERROR] '{data_dir}' 에 JSON 파일이 없습니다.")
        sys.exit(1)
    print(f"[LOAD] JSON 파일 {len(json_files)}개 발견")

    records = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"  [WARN] 로드 실패: {path} ({e})")
            continue

        if isinstance(raw, list):
            recs = [x for x in raw if isinstance(x, dict)]
        elif isinstance(raw, dict):
            recs = [raw]
        else:
            continue

        records.extend(recs)

    print(f"[LOAD] 총 레코드 수: {len(records)}명\n")
    return records


# ════════════════════════════════════════════════════════════════
#  3. 핵심 카운팅 함수
# ════════════════════════════════════════════════════════════════

def count_gt_abuse_types(
    records: List[dict],
    only_negative: bool = False,
    gt_field: str = "학대의심",
) -> dict:
    """
    원본(GT) 학대유형 필드를 기준으로 아동 수를 집계한다.

    집계 항목
    ---------
    · n_total          : 분석 대상 총 아동 수
    · abuse_count      : 학대유형별 아동 수 (한 아동이 복수 유형 보유 가능)
    · n_with_any_abuse : 학대유형이 1개 이상 기록된 아동 수
    · n_no_abuse       : 학대의심 필드가 비어있거나 인식 불가한 아동 수
    · n_multi_abuse    : 학대유형이 2개 이상 기록된 아동 수 (복수 라벨)
    · type_count_dist  : 아동별 학대유형 수 분포 {0: n, 1: n, 2: n, ...}
    · raw_field_dist   : 학대의심 필드 원본 값 빈도 (상위 20개)
    · skipped          : 정서군 필터로 제외된 아동 수

    Parameters
    ----------
    records : list[dict]
        load_records() 반환값
    only_negative : bool
        True이면 정서군='부정' 아동만 포함 (ABUSE_NEG 코퍼스)
    gt_field : str
        학대유형이 기록된 info 필드명 (기본: "학대의심")

    Returns
    -------
    dict : 집계 결과
    """
    abuse_count    = Counter()   # 학대유형별 아동 수
    type_count_dist = Counter()  # 아동별 보유 학대유형 수 분포
    raw_field_dist  = Counter()  # 원본 필드 값 빈도
    n_total        = 0
    n_with_any     = 0
    n_no_abuse     = 0
    n_multi        = 0
    skipped        = 0

    for rec in records:
        # ── 정서군 필터 ──
        valence = classify_child_group(rec)
        if only_negative and valence != "부정":
            skipped += 1
            continue

        n_total += 1
        info = rec.get("info", {}) or {}

        # ── 원본 필드 값 기록 (디버깅용) ──
        raw_val = info.get(gt_field, "")
        if isinstance(raw_val, (list, dict)):
            raw_key = json.dumps(raw_val, ensure_ascii=False)[:80]
        else:
            raw_key = str(raw_val).strip()[:80] if raw_val else "(비어있음)"
        raw_field_dist[raw_key] += 1

        # ── GT 학대유형 추출 ──
        #    info["학대의심"] 필드에서 학대유형 키워드를 파싱한다
        gt_types: Set[str] = extract_gt_abuse_types_from_info(info, field=gt_field)

        # ── 집계 ──
        n_types = len(gt_types)
        type_count_dist[n_types] += 1

        if n_types >= 1:
            n_with_any += 1
            for a in gt_types:
                abuse_count[a] += 1
        else:
            n_no_abuse += 1

        if n_types >= 2:
            n_multi += 1

    return {
        "n_total":          n_total,
        "n_with_any_abuse": n_with_any,
        "n_no_abuse":       n_no_abuse,
        "n_multi_abuse":    n_multi,
        "skipped":          skipped,
        "abuse_count":      dict(abuse_count),
        "type_count_dist":  dict(type_count_dist),
        "raw_field_dist":   raw_field_dist.most_common(20),
    }


# ════════════════════════════════════════════════════════════════
#  4. 결과 출력
# ════════════════════════════════════════════════════════════════

def print_results(result: dict, label: str = "전체") -> None:
    """
    카운팅 결과를 보기 좋게 출력한다.

    Parameters
    ----------
    result : dict
        count_gt_abuse_types() 반환값
    label : str
        출력 헤더 그룹 이름
    """
    n   = result["n_total"]
    sep = "═" * 62

    print(f"\n{sep}")
    print(f"  [{label}] 분석 대상 아동 수 : {n:,}명")
    if result["skipped"] > 0:
        print(f"  (정서군 필터로 제외 : {result['skipped']:,}명)")
    print(sep)

    if n == 0:
        print("  분석 대상 아동이 없습니다.\n")
        return

    # ── 학대유형별 카운트 ──
    print(f"\n  {'학대유형':<12}  {'n':>7}  {'비율(%)':>8}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*8}")
    for a in ABUSE_ORDER:
        cnt = result["abuse_count"].get(a, 0)
        pct = cnt / n * 100
        en  = ABUSE_LABEL_EN[a]
        print(f"  {a:<6}({en:<9})  {cnt:>7,}  {pct:>7.1f}%")

    # ── 요약 ──
    w = result["n_with_any_abuse"]
    z = result["n_no_abuse"]
    m = result["n_multi_abuse"]
    print(f"\n  {'─'*50}")
    print(f"  학대유형 ≥ 1개 (학대 기록 있음)  : {w:>6,}명  ({w/n*100:.1f}%)")
    print(f"  학대유형 없음  (학대 기록 없음)  : {z:>6,}명  ({z/n*100:.1f}%)")
    print(f"  학대유형 ≥ 2개 (복수 라벨)       : {m:>6,}명  ({m/n*100:.1f}%)")

    # ── 유형 수 분포 ──
    print(f"\n  ── 아동별 보유 학대유형 수 분포 (원본 기준) ──")
    dist = result["type_count_dist"]
    for k in sorted(dist.keys()):
        bar = "█" * min(int(dist[k] / max(n / 40, 1)), 40)
        print(f"    {k}개 : {dist[k]:>5,}명  {bar}")

    # ── 원본 필드 값 상위 10개 ──
    print(f"\n  ── info['학대의심'] 필드 원본 값 상위 10개 ──")
    for val, cnt in result["raw_field_dist"][:10]:
        print(f"    [{cnt:>5}명]  {val}")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════
#  5. 메인 실행
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="원본(GT) 학대유형별 아동 수 카운팅 (전체 / 부정군)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="JSON 파일이 있는 디렉토리 (기본: ./data)"
    )
    parser.add_argument(
        "--gt_field", type=str, default="학대의심",
        help="학대유형이 기록된 info 필드명 (기본: 학대의심)"
    )
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 60 + "╗")
    print("║   원본(GT) 학대유형별 아동 수 카운팅                   ║")
    print("║   기준: info['" + args.gt_field + "'] 필드 (알고리즘 미사용)   ║")
    print("╚" + "═" * 60 + "╝")
    print(f"  data_dir : {args.data_dir}")
    print(f"  gt_field : {args.gt_field}")

    # ── 데이터 로드 ──
    records = load_records(args.data_dir)

    # ── [A] 전체 집계 ──
    result_all = count_gt_abuse_types(
        records,
        only_negative=False,
        gt_field=args.gt_field,
    )
    print_results(result_all, label="전체 (ALL)")

    # ── [B] 부정군만 집계 ──
    result_neg = count_gt_abuse_types(
        records,
        only_negative=True,
        gt_field=args.gt_field,
    )
    print_results(result_neg, label="부정군 (ABUSE_NEG)")

    # ── [C] 전체 vs 부정군 나란히 비교 ──
    n_all = result_all["n_total"]
    n_neg = result_neg["n_total"]
    print("=" * 62)
    print("  [비교] 원본 GT 기준 — 전체 vs 부정군")
    print("=" * 62)
    print(f"\n  {'학대유형':<10}  {'전체 n':>8}  {'전체%':>7}  │  {'부정 n':>8}  {'부정%':>7}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*7}  │  {'─'*8}  {'─'*7}")
    for a in ABUSE_ORDER:
        ca = result_all["abuse_count"].get(a, 0)
        cn = result_neg["abuse_count"].get(a, 0)
        pa = ca / n_all * 100 if n_all > 0 else 0
        pn = cn / n_neg * 100 if n_neg > 0 else 0
        print(f"  {a:<10}  {ca:>8,}  {pa:>6.1f}%  │  {cn:>8,}  {pn:>6.1f}%")

    print(f"\n  {'총 아동':<10}  {n_all:>8,}  {'100.0%':>7}  │  {n_neg:>8,}  {'100.0%':>7}")
    print()


if __name__ == "__main__":
    main()
