from __future__ import annotations

from .common import *


def classify_child_group(
    rec,
    high_total_thresh=45,
    low_total_thresh=10,
    risk_strong=25,
    risk_weak=10,
):
    """
    한 아동의 JSON record(rec)를 받아 정서군 라벨('부정','평범','긍정')을 반환.
    - 위기단계 '응급'을 명시적으로 부정으로 처리
    - 행복/걱정/학대여부 all-zero → 긍정 규칙은 정상군/관찰필요/미기재에서만 적용
    - 나머지 로직은 기존 Algorithm 1과 동일한 흐름 유지
    """
    info = rec.get("info", {})
    crisis = info.get("위기단계")

    try:
        total_score = int(info.get("합계점수"))
    except (TypeError, ValueError):
        total_score = np.nan

    # 문항별 합계 & 항목별 점수 저장
    q_sum = {}
    item_scores = {}

    for q in rec.get("list", []):
        qname = q.get("문항")
        try:
            q_total = int(q.get("문항합계"))
        except (TypeError, ValueError):
            q_total = np.nan
        if qname is not None:
            q_sum[qname] = q_total

        for it in q.get("list", []):
            iname = it.get("항목")
            try:
                sc = int(it.get("점수"))
            except (TypeError, ValueError):
                continue
            if iname is None:
                continue
            item_scores.setdefault(iname, []).append(sc)

    def get_q(name, default=0):
        v = q_sum.get(name, default)
        return default if pd.isna(v) else v

    def get_item_max(name, default=0):
        vals = item_scores.get(name)
        if not vals:
            return default
        return max(vals)

    # --------------------------------------------------
    # 1) Critical triggers
    # --------------------------------------------------
    # 1-1. 자해/자살이 1점이라도 있으면 무조건 부정
    if get_item_max("자해/자살", 0) > 0:
        return "부정"

    happy_score = get_item_max("행복", default=0)
    worry_score = get_item_max("걱정", default=0)
    abuse_total = get_q("학대여부", 0)

    # 1-2. 행복 문항: 점수가 클수록 문제 심각(역채점)이라는 가정
    #     → 행복 점수가 매우 높으면 부정으로 강제
    if happy_score >= 7:
        return "부정"

    # 1-3. 행복=0, 걱정=0, 학대여부=0이면 긍정
    #     단, 위기단계가 최소한 '정상군/관찰필요/미기재'일 때만 적용하도록 안전장치 추가
    safe_crisis_for_auto_positive = {None, "정상군", "관찰필요"}
    if (
        (happy_score == 0)
        and (worry_score == 0)
        and (abuse_total == 0)
        and (crisis in safe_crisis_for_auto_positive)
    ):
        return "긍정"

    # --------------------------------------------------
    # 2) Base rule: 위기단계 기반 초기 라벨
    # --------------------------------------------------
    label = None

    # '응급'을 명시적으로 부정에 포함 (이전 코드에서는 누락되어 있었음)
    negative_crisis = {"응급", "위기아동", "학대의심", "상담필요"}

    if crisis in negative_crisis:
        label = "부정"
    elif crisis == "정상군":
        label = "긍정"
    elif crisis == "관찰필요":
        label = "평범"
    # 그 외 위기단계 값(또는 None)은 일단 label=None으로 두고, 이후 risk/보호요인으로 결정

    # --------------------------------------------------
    # 3) 합계점수 기반 조정
    # --------------------------------------------------
    if not pd.isna(total_score):
        # 고위험 컷: 점수가 충분히 높으면 무조건 부정 쪽으로 밀어줌
        if total_score >= high_total_thresh:
            label = "부정"
        # 저위험 컷: 점수가 아주 낮고, 아직 label이 정해지지 않은 경우에만 긍정으로 설정
        elif total_score <= low_total_thresh and label is None:
            label = "긍정"

    # --------------------------------------------------
    # 4) Risk vs Protective balance
    # --------------------------------------------------
    # 위험 문항 점수 합산
    risk_questions = ["기분문제", "기본생활", "학대여부", "응급"]
    risk_score = sum(get_q(qname, 0) for qname in risk_questions)

    # 보호요인: 대인관계 합계, 미래/진로 문항 (점수가 낮을수록 보호적이라고 가정)
    relation_sum = get_q("대인관계", 0)
    future_score = get_item_max("미래/진로", default=0)

    protective_index = 0
    if relation_sum <= 2:
        protective_index += 1
    if future_score == 0:
        protective_index += 1

    # 4-1. 아직 label이 정해지지 않은 경우: risk/protective로 1차 결정
    if label is None:
        if risk_score >= risk_strong and protective_index == 0:
            label = "부정"
        elif risk_score <= risk_weak and protective_index >= 1:
            label = "긍정"
        else:
            label = "평범"
    else:
        # 4-2. 기존 label을 risk/protective로 미세 조정
        if label == "긍정":
            # 긍정인데 risk가 높으면 평범 또는 부정으로 내려감
            if risk_score >= risk_strong:
                label = "평범"
            if risk_score >= risk_strong + 10 and protective_index == 0:
                label = "부정"
        elif label == "평범":
            # 평범인데 risk가 높고 보호요인이 없으면 부정으로
            if risk_score >= risk_strong and protective_index == 0:
                label = "부정"
            # 평범인데 risk가 낮고 보호요인이 충분(2개 이상)하면 긍정으로
            elif risk_score <= risk_weak and protective_index >= 2:
                label = "긍정"

    # --------------------------------------------------
    # 5) 최종 체크
    # --------------------------------------------------
    if label not in VALENCE_ORDER:
        return None
    return label


# ─────────────────────────────────────────────────────────────
# Child-Safety-First Severity Hierarchy (아동 안전 우선 위계)
# ─────────────────────────────────────────────────────────────
# 숫자가 작을수록 심각한 학대유형 → 동점(tie) 시 우선 선택
# 이 위계는 classify_abuse_main_sub()의 재현성을 보장합니다.
_SEVERITY_RANK = SEVERITY_RANK  # common.py 에서 정의


def classify_abuse_main_sub(rec, abuse_order=ABUSE_ORDER,
                            sub_threshold=4,
                            use_clinical_text=True):
    """
    임상가의 판단(GT)을 최우선으로 main 학대유형을 결정하고,
    점수 기반으로 sub 학대유형을 할당한다.

    Main 할당 우선순위:
      1순위: info["학대의심"] (임상가 GT label)
      2순위: 문항 점수 > 6인 유형 중 최고점 (기존 로직)
      3순위: 임상 텍스트에서 발견된 학대유형 (기존 로직)

    Sub 할당 (변경 없음):
      main이 아닌 유형 중 점수 >= sub_threshold인 유형
      + 임상 텍스트에서 발견된 추가 유형

    동점(tie) 시 _SEVERITY_RANK 위계를 적용하여 결정적(deterministic) 결과를 보장.
    위계: 성학대(0) > 신체학대(1) > 정서학대(2) > 방임(3)
    """
    info = rec.get("info", {}) or {}
    main = None
    subs = set()

    # ── 1순위: GT label 확인 ──
    gt_main = _extract_gt_main(info, abuse_order)

    # ── 점수 추출 (기존 로직 유지) ──
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

    # ── Main 할당: GT 우선, 없으면 점수 기반 ──
    if gt_main is not None:
        # 1순위: GT label
        main = gt_main
    else:
        # 2순위: 점수 > 6인 유형 중 최고점 (기존 로직)
        nonzero = {a: s for a, s in abuse_scores.items() if s > 6}
        if nonzero:
            # [FIX] deterministic tie-breaking: (점수, -심각도)
            # 점수가 같으면 _SEVERITY_RANK가 작은(더 심각한) 유형 우선
            main = max(nonzero,
                       key=lambda a: (nonzero[a], -_SEVERITY_RANK.get(a, 999)))

    # ── Sub 할당 (기존 로직 유지) ──
    for a, s in abuse_scores.items():
        if a == main:
            continue
        if s >= sub_threshold:
            subs.add(a)

    # ── 임상 텍스트 보완 (기존 로직 유지) ──
    if use_clinical_text:
        clin_text = " ".join(
            str(info.get(k, "")) for k in ["임상진단", "임상가 종합소견"] if k in info
        )
        for a in abuse_order:
            if a in clin_text:
                if main is None:
                    # 3순위: 임상 텍스트에서 발견
                    main = a
                elif a != main:
                    subs.add(a)

    # ── Fallback (기존 로직 유지) ──
    if main is None and subs:
        # [FIX] deterministic tie-breaking: (점수, -심각도)
        main = sorted(subs,
                      key=lambda x: (abuse_scores.get(x, 0),
                                     -_SEVERITY_RANK.get(x, 999)),
                      reverse=True)[0]
        subs.remove(main)

    if main is None:
        return None, []

    # [FIX] subs도 심각도 순으로 정렬 (재현성 보장)
    return main, sorted(subs, key=lambda x: _SEVERITY_RANK.get(x, 999))


# ── GT 추출 헬퍼 (labels.py 내부) ──

# 표기 변형 정규화 맵 (compare_abuse_labels.py와 동일)
_GT_CANON_MAP = {
    "신체적학대": "신체학대",
    "정서적학대": "정서학대",
    "성적학대": "성학대",
    "성폭력": "성학대",
    "성폭행": "성학대",
    "유기": "방임",
}


def _normalize_gt_label(label: str) -> str:
    """GT 라벨 표기 변형을 표준형으로 정규화."""
    import re
    s = re.sub(r"\s+", "", str(label))
    s = _GT_CANON_MAP.get(s, s)
    if s.endswith("적학대"):
        s = s.replace("적학대", "학대")
    return _GT_CANON_MAP.get(s, s)


def _gt_field_to_text(x) -> str:
    """info["학대의심"] 값을 문자열로 변환 (다양한 타입 대응)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_gt_field_to_text(v) for v in x)
    if isinstance(x, dict):
        for k in ("val", "text", "value"):
            if k in x:
                return _gt_field_to_text(x.get(k))
        import json
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _extract_gt_main(info: dict, abuse_order: list) -> str | None:
    """
    info["학대의심"] 필드에서 GT 학대유형을 추출한다.

    compare_abuse_labels.py의 extract_gt_abuse_types_from_info()와
    동일한 로직이지만, 순환 import 방지를 위해 인라인으로 구현.

    여러 유형이 추출되면 SEVERITY_RANK 기준으로 가장 심각한 것을 main으로 선택.
    """
    import re

    raw = info.get("학대의심", "")
    text = _gt_field_to_text(raw)
    text_nospace = re.sub(r"\s+", "", text)

    if not text_nospace:
        return None

    found = set()

    # 규칙 1: "[가-힣]+학대" 패턴
    for m in re.findall(r"([가-힣]+?)학대", text_nospace):
        cand = _normalize_gt_label(m + "학대")
        if cand in abuse_order:
            found.add(cand)

    # 규칙 2: "방임", "유기"
    if "방임" in text_nospace:
        found.add("방임")
    if "유기" in text_nospace:
        found.add("방임")

    # 규칙 3: 표준형 직접 매칭
    for a in abuse_order:
        if re.sub(r"\s+", "", a) in text_nospace:
            found.add(a)

    # 유효한 유형만 필터
    found = {a for a in found if a in abuse_order}

    if not found:
        return None

    # 여러 개면 SEVERITY_RANK 기준 가장 심각한 것 선택
    return sorted(found, key=lambda x: _SEVERITY_RANK.get(x, 999))[0]