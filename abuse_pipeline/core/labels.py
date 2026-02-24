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
    상담사의 임상 정보 + 학대 관련 문항 점수를 이용해
    - main_abuse : 핵심(주) 학대유형
    - sub_abuses : 부 학대유형 리스트
    를 추정.

    동점(tie) 시 _SEVERITY_RANK 위계를 적용하여 결정적(deterministic) 결과를 보장.
    위계: 성학대(0) > 신체학대(1) > 정서학대(2) > 방임(3)
    """
    info = rec.get("info", {})
    main = None
    subs = set()

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

    nonzero = {a: s for a, s in abuse_scores.items() if s > 6}
    if nonzero:
        # [FIX] deterministic tie-breaking: (점수, -심각도)
        # 점수가 같으면 _SEVERITY_RANK가 작은(더 심각한) 유형 우선
        main = max(nonzero,
                   key=lambda a: (nonzero[a], -_SEVERITY_RANK.get(a, 999)))

    for a, s in abuse_scores.items():
        if a == main:
            continue
        if s >= sub_threshold:
            subs.add(a)

    if use_clinical_text:
        clin_text = " ".join(
            str(info.get(k, "")) for k in ["임상진단", "임상가 종합소견"] if k in info
        )
        for a in abuse_order:
            if a in clin_text:
                if main is None:
                    main = a
                elif a != main:
                    subs.add(a)

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