#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================
부록용 임계값 근거 통계 추출 스크립트
================================================================
목적: 본 프로젝트에서 사용된 모든 임계값(threshold)에 대해,
      해당 임계값이 왜 적절한지를 보여주는 기술통계량을 추출한다.

사용법:
    python extract_threshold_statistics.py

출력:
    threshold_justification_statistics.csv  (전체 통계 요약표)
    콘솔에도 카테고리별 상세 통계가 출력됨
================================================================
"""

from __future__ import annotations
import os, sys, json, re, glob, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ================================================================
# 0. 경로 설정
# ================================================================
# 프로젝트 루트 자동 탐색 (data/ 폴더가 있는 곳)
def find_project_root(start=None):
    if start is None:
        start = Path(__file__).resolve().parent
    p = start
    for _ in range(10):
        if (p / "data").exists():
            return p
        p = p.parent
    return start

ROOT = find_project_root()
DATA_DIR = ROOT / "data"
JSON_FILES = sorted(glob.glob(str(DATA_DIR / "*.json")))

print(f"프로젝트 루트: {ROOT}")
print(f"데이터 디렉토리: {DATA_DIR}")
print(f"JSON 파일 수: {len(JSON_FILES)}")

if not JSON_FILES:
    print("[ERROR] data/ 폴더에 JSON 파일이 없습니다.")
    print("  이 스크립트를 프로젝트 루트에서 실행하거나,")
    print("  data/ 폴더 경로를 수정해주세요.")
    sys.exit(1)

# ================================================================
# 1. 프로젝트 모듈 import (가능하면 사용, 불가하면 인라인 구현)
# ================================================================
try:
    sys.path.insert(0, str(ROOT))
    from abuse_pipeline.labels import classify_child_group, classify_abuse_main_sub
    from abuse_pipeline.text import extract_child_speech, tokenize_korean
    from abuse_pipeline.common import (
        VALENCE_ORDER, ABUSE_ORDER, STOPWORDS_BASE,
        MIN_TOTAL_COUNT_VALENCE, MIN_TOTAL_COUNT_ABUSE, MIN_DOC_COUNT,
        BRIDGE_MIN_P1, BRIDGE_MIN_P2, BRIDGE_MAX_GAP,
        BRIDGE_MIN_COUNT, BRIDGE_MIN_LOGODDS, DEFAULT_DELTA, DEFAULT_ZMIN,
    )
    USE_PROJECT_MODULE = True
    print("[OK] 프로젝트 모듈 import 성공\n")
except Exception as e:
    print(f"[WARN] 프로젝트 모듈 import 실패: {e}")
    print("  인라인 구현을 사용합니다.\n")
    USE_PROJECT_MODULE = False

    VALENCE_ORDER = ["부정", "평범", "긍정"]
    ABUSE_ORDER = ["방임", "정서학대", "신체학대", "성학대"]
    STOPWORDS_BASE = {
        "자다","모르다","아니다","않다","그렇다","싶다","나다",
        "이다","하다","되다","같다","있다","없다",
        "좀","조금","많이","그냥","막","또","근데",
        "이제","그래서","그러면","그리고","때문","때문에","거","것",
        "사람","애들","애기","어른","공부","시험","시간",
        "거기","여기","저기","이거","저거",
        "하루","맨날","항상","가끔","자주","매일",
        "정도","수준","상태","상황","요즘",
        "0점","1점","2점","3점","4점","5점","6점","7점","8점","9점",
        "정상군","위기","위기아동","상담필요","관찰필요",
    }
    MIN_TOTAL_COUNT_VALENCE = 10
    MIN_TOTAL_COUNT_ABUSE = 8
    MIN_DOC_COUNT = 5
    BRIDGE_MIN_P1 = 0.40
    BRIDGE_MIN_P2 = 0.25
    BRIDGE_MAX_GAP = 0.20
    BRIDGE_MIN_COUNT = 5
    BRIDGE_MIN_LOGODDS = 1.0
    DEFAULT_DELTA = 1.0
    DEFAULT_ZMIN = 1.96

    # 인라인 함수 정의
    def extract_child_speech(rec):
        texts = []
        for q in rec.get("list", []):
            for it in q.get("list", []):
                for seg in it.get("audio", []):
                    if seg.get("type") == "A":
                        t = seg.get("text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
        return texts

    try:
        from konlpy.tag import Okt
        okt = Okt()
        HAS_OKT = True
    except:
        okt = None
        HAS_OKT = False

    def tokenize_korean(text):
        if not isinstance(text, str):
            return []
        text = re.sub(r"[^가-힣0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        if HAS_OKT and okt:
            pos_list = okt.pos(text, stem=True)
            tokens = []
            i = 0
            while i < len(pos_list):
                word, pos = pos_list[i]
                if word in {"안","못"} and (i+1) < len(pos_list):
                    nw, np_ = pos_list[i+1]
                    if np_ in ["Verb","Adjective"]:
                        if len(nw) > 1 and nw not in STOPWORDS_BASE:
                            tokens.append(f"{word}_{nw}")
                        i += 2; continue
                if word == "않다":
                    if tokens and not tokens[-1].startswith("안_"):
                        tokens[-1] = "안_" + tokens[-1]
                    i += 1; continue
                if pos in ["Noun","Verb","Adjective"]:
                    if len(word) > 1 and word not in STOPWORDS_BASE:
                        tokens.append(word)
                i += 1
            return tokens
        else:
            return [t for t in text.split() if len(t)>1 and t not in STOPWORDS_BASE]

    def classify_child_group(rec, high_total_thresh=45, low_total_thresh=10,
                              risk_strong=25, risk_weak=10):
        info = rec.get("info", {})
        crisis = info.get("위기단계")
        try: total_score = int(info.get("합계점수"))
        except: total_score = np.nan
        q_sum = {}; item_scores = {}
        for q in rec.get("list", []):
            qname = q.get("문항")
            try: qt = int(q.get("문항합계"))
            except: qt = np.nan
            if qname: q_sum[qname] = qt
            for it in q.get("list", []):
                iname = it.get("항목")
                try: sc = int(it.get("점수"))
                except: continue
                if iname: item_scores.setdefault(iname, []).append(sc)
        def get_q(n, d=0):
            v = q_sum.get(n, d); return d if pd.isna(v) else v
        def get_item_max(n, d=0):
            vals = item_scores.get(n); return d if not vals else max(vals)
        if get_item_max("자해/자살", 0) > 0: return "부정"
        hs = get_item_max("행복", 0); ws = get_item_max("걱정", 0); at = get_q("학대여부", 0)
        if hs >= 7: return "부정"
        if hs==0 and ws==0 and at==0 and crisis in {None,"정상군","관찰필요"}: return "긍정"
        label = None
        if crisis in {"응급","위기아동","학대의심","상담필요"}: label = "부정"
        elif crisis == "정상군": label = "긍정"
        elif crisis == "관찰필요": label = "평범"
        if not pd.isna(total_score):
            if total_score >= high_total_thresh: label = "부정"
            elif total_score <= low_total_thresh and label is None: label = "긍정"
        risk_qs = ["기분문제","기본생활","학대여부","응급"]
        risk_score = sum(get_q(q, 0) for q in risk_qs)
        rs = get_q("대인관계", 0); fs = get_item_max("미래/진로", 0)
        pi = 0
        if rs <= 2: pi += 1
        if fs == 0: pi += 1
        if label is None:
            if risk_score >= risk_strong and pi==0: label = "부정"
            elif risk_score <= risk_weak and pi>=1: label = "긍정"
            else: label = "평범"
        else:
            if label == "긍정":
                if risk_score >= risk_strong: label = "평범"
                if risk_score >= risk_strong+10 and pi==0: label = "부정"
            elif label == "평범":
                if risk_score >= risk_strong and pi==0: label = "부정"
                elif risk_score <= risk_weak and pi>=2: label = "긍정"
        return label if label in VALENCE_ORDER else None

    _SEVERITY_RANK = {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}


    def classify_abuse_main_sub(rec, abuse_order=ABUSE_ORDER, sub_threshold=2):
        info = rec.get("info", {})
        abuse_scores = {a: 0 for a in abuse_order}
        for q in rec.get("list", []):
            if q.get("문항") == "학대여부":
                for it in q.get("list", []):
                    name = it.get("항목")
                    try: sc = int(it.get("점수"))
                    except: sc = 0
                    if not isinstance(name, str): continue
                    for a in abuse_order:
                        if a in name: abuse_scores[a] += sc
        nonzero = {a:s for a,s in abuse_scores.items() if s > 6}
        main = max(nonzero, key=lambda a: (nonzero[a], -_SEVERITY_RANK.get(a, 999))) if nonzero else None
        subs = set()
        for a, s in abuse_scores.items():
            if a == main: continue
            if s >= sub_threshold: subs.add(a)
        clin_text = " ".join(str(info.get(k,"")) for k in ["임상진단","임상가 종합소견"] if k in info)
        for a in abuse_order:
            if a in clin_text:
                if main is None: main = a
                elif a != main: subs.add(a)
        if main is None and subs:
            main = sorted(subs, key=lambda x: (abuse_scores.get(x,0), -_SEVERITY_RANK.get(x,999)), reverse=True)[0]
            subs.remove(main)
        if main is None: return None, []
        return main, sorted(subs, key=lambda x: _SEVERITY_RANK.get(x, 999))


# ================================================================
# 2. 데이터 로드 및 기초 처리
# ================================================================
print("=" * 72)
print("데이터 로드 및 기초 처리 시작...")
print("=" * 72)

records = []
for path in JSON_FILES:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        rec["_filepath"] = path
        records.append(rec)
    except Exception as e:
        print(f"  [WARN] 로드 실패: {path} ({e})")

print(f"로드된 레코드 수: {len(records)}")

# ================================================================
# 통계 결과 저장용
# ================================================================
all_stats = []  # (카테고리, 임계값명, 설정값, 통계항목, 통계값)

def add_stat(category, threshold_name, threshold_value, stat_name, stat_value):
    all_stats.append({
        "카테고리": category,
        "임계값_이름": threshold_name,
        "설정값": str(threshold_value),
        "통계항목": stat_name,
        "통계값": stat_value,
    })

# ================================================================
# SECTION A: 정서군 분류 임계값 (labels.py)
# ================================================================
print("\n" + "=" * 72)
print("SECTION A: 정서군 분류(Valence Classification) 임계값 근거")
print("=" * 72)

total_scores = []
groups = []
crisis_stages = []
risk_scores_all = []
happy_scores_all = []
worry_scores_all = []
abuse_totals_all = []
abuse_score_dicts = []

for rec in records:
    info = rec.get("info", {})
    group = classify_child_group(rec)
    if group is None:
        continue
    groups.append(group)

    # 합계점수
    try:
        ts = int(info.get("합계점수"))
    except:
        ts = np.nan
    total_scores.append(ts)

    # 위기단계
    crisis_stages.append(info.get("위기단계"))

    # 문항/항목별 점수 추출
    q_sum = {}
    item_scores = {}
    for q in rec.get("list", []):
        qname = q.get("문항")
        try:
            qt = int(q.get("문항합계"))
        except:
            qt = np.nan
        if qname:
            q_sum[qname] = qt
        for it in q.get("list", []):
            iname = it.get("항목")
            try:
                sc = int(it.get("점수"))
            except:
                continue
            if iname:
                item_scores.setdefault(iname, []).append(sc)

    def get_q(n, d=0):
        v = q_sum.get(n, d)
        return d if pd.isna(v) else v
    def get_item_max(n, d=0):
        vals = item_scores.get(n)
        return d if not vals else max(vals)

    happy_scores_all.append(get_item_max("행복", 0))
    worry_scores_all.append(get_item_max("걱정", 0))
    abuse_totals_all.append(get_q("학대여부", 0))

    risk_qs = ["기분문제", "기본생활", "학대여부", "응급"]
    risk_score = sum(get_q(q, 0) for q in risk_qs)
    risk_scores_all.append(risk_score)

    # 학대점수 (classify_abuse_main_sub 에서 사용)
    a_scores = {a: 0 for a in ABUSE_ORDER}
    for q in rec.get("list", []):
        if q.get("문항") == "학대여부":
            for it in q.get("list", []):
                name = it.get("항목")
                try:
                    sc = int(it.get("점수"))
                except:
                    sc = 0
                if isinstance(name, str):
                    for a in ABUSE_ORDER:
                        if a in name:
                            a_scores[a] += sc
    abuse_score_dicts.append(a_scores)

df_main = pd.DataFrame({
    "group": groups,
    "total_score": total_scores,
    "crisis": crisis_stages,
    "risk_score": risk_scores_all,
    "happy_score": happy_scores_all,
    "worry_score": worry_scores_all,
    "abuse_total": abuse_totals_all,
})

N = len(df_main)
print(f"\n유효 아동 수 (정서군 분류 성공): {N}")
print(f"정서군 분포:\n{df_main['group'].value_counts().to_string()}\n")

add_stat("A.정서군분류", "전체", "-", "유효_아동수(N)", N)
for g in VALENCE_ORDER:
    n_g = (df_main["group"]==g).sum()
    add_stat("A.정서군분류", "전체", "-", f"정서군_{g}_수", n_g)
    add_stat("A.정서군분류", "전체", "-", f"정서군_{g}_비율(%)", round(n_g/N*100, 1))

# --- A1. 합계점수 분포 ---
print("--- A1. 합계점수(total_score) 분포 ---")
ts_valid = df_main["total_score"].dropna()
print(f"  유효 수: {len(ts_valid)}")
print(f"  평균: {ts_valid.mean():.2f}")
print(f"  표준편차: {ts_valid.std():.2f}")
print(f"  중앙값: {ts_valid.median():.1f}")
print(f"  최소~최대: {ts_valid.min():.0f} ~ {ts_valid.max():.0f}")
pcts = [5, 10, 25, 50, 75, 90, 95]
for p in pcts:
    val = np.nanpercentile(ts_valid, p)
    print(f"  {p}번째 백분위수: {val:.1f}")
    add_stat("A.정서군분류", "합계점수분포", "-", f"백분위수_{p}%", round(val, 1))

add_stat("A.정서군분류", "합계점수분포", "-", "평균", round(ts_valid.mean(), 2))
add_stat("A.정서군분류", "합계점수분포", "-", "표준편차", round(ts_valid.std(), 2))
add_stat("A.정서군분류", "합계점수분포", "-", "중앙값", round(ts_valid.median(), 1))

print(f"\n  정서군별 합계점수 평균:")
for g in VALENCE_ORDER:
    sub = df_main.loc[df_main["group"]==g, "total_score"].dropna()
    if len(sub) > 0:
        print(f"    {g}: n={len(sub)}, 평균={sub.mean():.2f}, SD={sub.std():.2f}, "
              f"범위=[{sub.min():.0f},{sub.max():.0f}]")
        add_stat("A.정서군분류", f"합계점수_정서군={g}", "-", "n", len(sub))
        add_stat("A.정서군분류", f"합계점수_정서군={g}", "-", "평균", round(sub.mean(), 2))
        add_stat("A.정서군분류", f"합계점수_정서군={g}", "-", "SD", round(sub.std(), 2))

# --- A2. high_total_thresh = 45 ---
print(f"\n--- A2. high_total_thresh = 45 근거 ---")
for thr in [35, 40, 45, 50, 55]:
    n_above = (ts_valid >= thr).sum()
    pct = n_above / len(ts_valid) * 100
    neg_above = ((df_main["total_score"]>=thr) & (df_main["group"]=="부정")).sum()
    precision = neg_above / n_above * 100 if n_above > 0 else 0
    print(f"  합계점수 >= {thr}: {n_above}명 ({pct:.1f}%), 이 중 부정군={neg_above}명 (정밀도={precision:.1f}%)")
    add_stat("A.정서군분류", "high_total_thresh", thr,
             f">={thr}_해당수", n_above)
    add_stat("A.정서군분류", "high_total_thresh", thr,
             f">={thr}_부정군_정밀도(%)", round(precision, 1))

# --- A3. low_total_thresh = 10 ---
print(f"\n--- A3. low_total_thresh = 10 근거 ---")
for thr in [5, 8, 10, 12, 15]:
    n_below = (ts_valid <= thr).sum()
    pct = n_below / len(ts_valid) * 100
    pos_below = ((df_main["total_score"]<=thr) & (df_main["group"]=="긍정")).sum()
    precision = pos_below / n_below * 100 if n_below > 0 else 0
    print(f"  합계점수 <= {thr}: {n_below}명 ({pct:.1f}%), 이 중 긍정군={pos_below}명 (정밀도={precision:.1f}%)")
    add_stat("A.정서군분류", "low_total_thresh", thr,
             f"<={thr}_해당수", n_below)
    add_stat("A.정서군분류", "low_total_thresh", thr,
             f"<={thr}_긍정군_정밀도(%)", round(precision, 1))

# --- A4. risk_strong = 25, risk_weak = 10 ---
print(f"\n--- A4. risk_score 분포 ---")
rs = pd.Series(risk_scores_all)
print(f"  risk_score 평균: {rs.mean():.2f}, SD: {rs.std():.2f}")
print(f"  중앙값: {rs.median():.1f}")
for p in pcts:
    val = np.nanpercentile(rs, p)
    print(f"  {p}번째 백분위수: {val:.1f}")
    add_stat("A.정서군분류", "risk_score분포", "-", f"백분위수_{p}%", round(val, 1))

add_stat("A.정서군분류", "risk_score분포", "-", "평균", round(rs.mean(), 2))
add_stat("A.정서군분류", "risk_score분포", "-", "SD", round(rs.std(), 2))

print(f"\n  risk_strong 후보별 임계값 효과 (부정군 recall):")
for thr in [15, 20, 25, 30, 35]:
    n_above = (rs >= thr).sum()
    neg_above = sum(1 for r, g in zip(risk_scores_all, groups) if r>=thr and g=="부정")
    recall = neg_above / groups.count("부정") * 100 if groups.count("부정") > 0 else 0
    print(f"    risk >= {thr}: {n_above}명, 부정군 recall={recall:.1f}%")
    add_stat("A.정서군분류", "risk_strong", thr, f">={thr}_해당수", n_above)
    add_stat("A.정서군분류", "risk_strong", thr, f">={thr}_부정군_recall(%)", round(recall, 1))

# --- A5. happy_score >= 7 (행복 문항 강제 부정) ---
print(f"\n--- A5. happy_score 분포 ---")
hs = pd.Series(happy_scores_all)
print(f"  happy_score 분포:")
hs_vc = hs.value_counts().sort_index()
for val, cnt in hs_vc.items():
    print(f"    점수={val}: {cnt}명 ({cnt/N*100:.1f}%)")
    add_stat("A.정서군분류", "happy_score분포", "-", f"점수={val}_수", cnt)

# --- A6. 학대점수 > 6 (main_abuse 결정 임계값) ---
print(f"\n--- A6. 학대 항목별 점수 분포 (main_abuse 결정: >6) ---")
for a in ABUSE_ORDER:
    scores = [d[a] for d in abuse_score_dicts]
    s = pd.Series(scores)
    nonzero = (s > 0).sum()
    above6 = (s > 6).sum()
    print(f"  {a}: 평균={s.mean():.2f}, SD={s.std():.2f}, >0인 수={nonzero}, >6인 수={above6}")
    add_stat("A.정서군분류", f"학대점수_{a}", ">6", "평균", round(s.mean(), 2))
    add_stat("A.정서군분류", f"학대점수_{a}", ">6", ">0인_수", nonzero)
    add_stat("A.정서군분류", f"학대점수_{a}", ">6", ">6인_수(main판정)", above6)

# ================================================================
# SECTION B: 텍스트 처리 및 빈도 필터 임계값
# ================================================================
print("\n" + "=" * 72)
print("SECTION B: 텍스트 처리 & 빈도 필터 임계값 근거")
print("=" * 72)

# 전체 텍스트 토큰화
all_tokens_per_child = []
all_token_lengths = []  # 토큰 글자 수
corpus_abuse = defaultdict(list)  # abuse -> [token list per child]
corpus_valence = defaultdict(list)  # valence -> [token list per child]

n_children_with_speech = 0
n_children_no_speech = 0
utterance_counts = []
token_counts_per_child = []

for rec in records:
    group = classify_child_group(rec)
    if group is None:
        continue
    main_abuse, _ = classify_abuse_main_sub(rec)
    speech = extract_child_speech(rec)
    if not speech:
        n_children_no_speech += 1
        continue
    n_children_with_speech += 1
    utterance_counts.append(len(speech))

    joined = " ".join(speech)
    tokens = tokenize_korean(joined)
    all_tokens_per_child.append(tokens)
    token_counts_per_child.append(len(tokens))
    for t in tokens:
        all_token_lengths.append(len(t))

    corpus_valence[group].extend(tokens)
    if main_abuse and main_abuse in ABUSE_ORDER:
        corpus_abuse[main_abuse].extend(tokens)

print(f"\n아동 발화 보유: {n_children_with_speech}명, 미보유: {n_children_no_speech}명")
add_stat("B.텍스트처리", "기본통계", "-", "발화보유_아동수", n_children_with_speech)
add_stat("B.텍스트처리", "기본통계", "-", "발화미보유_아동수", n_children_no_speech)

# --- B1. 발화 수 / 토큰 수 분포 ---
print(f"\n--- B1. 아동별 발화 수(utterance) 분포 ---")
uc = pd.Series(utterance_counts)
print(f"  평균: {uc.mean():.1f}, SD: {uc.std():.1f}, 중앙값: {uc.median():.0f}")
print(f"  범위: [{uc.min()}, {uc.max()}]")
add_stat("B.텍스트처리", "발화수분포", "-", "평균", round(uc.mean(), 1))
add_stat("B.텍스트처리", "발화수분포", "-", "SD", round(uc.std(), 1))
add_stat("B.텍스트처리", "발화수분포", "-", "중앙값", uc.median())

print(f"\n--- B2. 아동별 토큰 수 분포 ---")
tc = pd.Series(token_counts_per_child)
print(f"  평균: {tc.mean():.1f}, SD: {tc.std():.1f}, 중앙값: {tc.median():.0f}")
print(f"  범위: [{tc.min()}, {tc.max()}]")
add_stat("B.텍스트처리", "토큰수분포", "-", "평균", round(tc.mean(), 1))
add_stat("B.텍스트처리", "토큰수분포", "-", "SD", round(tc.std(), 1))

# --- B3. 토큰 길이(글자 수) 분포 → "한 글자 토큰 제거" 근거 ---
print(f"\n--- B3. 토큰 길이(글자 수) 분포 → 1글자 토큰 제거 근거 ---")
tl = pd.Series(all_token_lengths)
tl_vc = tl.value_counts().sort_index().head(10)
total_tokens = len(tl)
for length, cnt in tl_vc.items():
    pct = cnt / total_tokens * 100
    print(f"  글자수={length}: {cnt}개 ({pct:.1f}%)")
    add_stat("B.텍스트처리", "토큰길이분포", "1글자제거", f"글자수={length}_수", cnt)
    add_stat("B.텍스트처리", "토큰길이분포", "1글자제거", f"글자수={length}_비율(%)", round(pct, 1))

print(f"  전체 토큰 수: {total_tokens}")
add_stat("B.텍스트처리", "토큰길이분포", "1글자제거", "전체토큰수", total_tokens)

# ================================================================
# SECTION C: 빈도 필터 임계값 (MIN_TOTAL_COUNT_*)
# ================================================================
print("\n" + "=" * 72)
print("SECTION C: 빈도 필터 임계값 근거")
print("=" * 72)

# --- C1. 정서군(Valence) 단어 빈도 분포 ---
print(f"\n--- C1. 정서군 단어 빈도 분포 (MIN_TOTAL_COUNT_VALENCE = {MIN_TOTAL_COUNT_VALENCE}) ---")
valence_word_counts = defaultdict(lambda: defaultdict(int))
for g in VALENCE_ORDER:
    for tok in corpus_valence.get(g, []):
        valence_word_counts[tok][g] += 1

rows_val = []
for w, counts in valence_word_counts.items():
    row = {"word": w}
    total = 0
    for g in VALENCE_ORDER:
        row[g] = counts.get(g, 0)
        total += row[g]
    row["total"] = total
    rows_val.append(row)

df_val_freq = pd.DataFrame(rows_val)
if not df_val_freq.empty:
    total_words_before = len(df_val_freq)
    total_words_after = (df_val_freq["total"] >= MIN_TOTAL_COUNT_VALENCE).sum()
    print(f"  전체 고유 단어 수: {total_words_before}")
    print(f"  빈도 >= {MIN_TOTAL_COUNT_VALENCE} 단어 수: {total_words_after} "
          f"({total_words_after/total_words_before*100:.1f}%)")
    print(f"  제거된 단어 수: {total_words_before - total_words_after}")

    add_stat("C.빈도필터", "MIN_TOTAL_COUNT_VALENCE", MIN_TOTAL_COUNT_VALENCE,
             "필터전_고유단어수", total_words_before)
    add_stat("C.빈도필터", "MIN_TOTAL_COUNT_VALENCE", MIN_TOTAL_COUNT_VALENCE,
             "필터후_고유단어수", total_words_after)
    add_stat("C.빈도필터", "MIN_TOTAL_COUNT_VALENCE", MIN_TOTAL_COUNT_VALENCE,
             "제거비율(%)", round((1 - total_words_after/total_words_before)*100, 1))

    # 다양한 임계값별 잔존 단어 수
    print(f"\n  다양한 빈도 임계값별 잔존 단어 수:")
    for thr in [1, 3, 5, 8, 10, 15, 20, 30]:
        n_remain = (df_val_freq["total"] >= thr).sum()
        pct = n_remain / total_words_before * 100
        # 잔존 단어가 전체 빈도에서 차지하는 비중(토큰 coverage)
        total_freq_remain = df_val_freq.loc[df_val_freq["total"]>=thr, "total"].sum()
        total_freq_all = df_val_freq["total"].sum()
        coverage = total_freq_remain / total_freq_all * 100 if total_freq_all > 0 else 0
        print(f"    빈도 >= {thr:>3d}: {n_remain:>5d}개 ({pct:>5.1f}%), 토큰 커버리지={coverage:.1f}%")
        add_stat("C.빈도필터", "valence_빈도임계값별", thr, "잔존_단어수", n_remain)
        add_stat("C.빈도필터", "valence_빈도임계값별", thr, "토큰_커버리지(%)", round(coverage, 1))

# --- C2. 학대유형(Abuse) 단어 빈도 분포 ---
print(f"\n--- C2. 학대유형 단어 빈도 분포 (MIN_TOTAL_COUNT_ABUSE = {MIN_TOTAL_COUNT_ABUSE}) ---")
abuse_word_counts = defaultdict(lambda: defaultdict(int))
for a in ABUSE_ORDER:
    for tok in corpus_abuse.get(a, []):
        abuse_word_counts[tok][a] += 1

rows_ab = []
for w, counts in abuse_word_counts.items():
    row = {"word": w}
    total = 0
    for a in ABUSE_ORDER:
        row[a] = counts.get(a, 0)
        total += row[a]
    row["total"] = total
    rows_ab.append(row)

df_ab_freq = pd.DataFrame(rows_ab)
if not df_ab_freq.empty:
    total_words_before_ab = len(df_ab_freq)
    total_words_after_ab = (df_ab_freq["total"] >= MIN_TOTAL_COUNT_ABUSE).sum()
    print(f"  전체 고유 단어 수: {total_words_before_ab}")
    print(f"  빈도 >= {MIN_TOTAL_COUNT_ABUSE} 단어 수: {total_words_after_ab} "
          f"({total_words_after_ab/total_words_before_ab*100:.1f}%)")

    add_stat("C.빈도필터", "MIN_TOTAL_COUNT_ABUSE", MIN_TOTAL_COUNT_ABUSE,
             "필터전_고유단어수", total_words_before_ab)
    add_stat("C.빈도필터", "MIN_TOTAL_COUNT_ABUSE", MIN_TOTAL_COUNT_ABUSE,
             "필터후_고유단어수", total_words_after_ab)

    print(f"\n  다양한 빈도 임계값별 잔존 단어 수:")
    for thr in [1, 3, 5, 8, 10, 15, 20, 30]:
        n_remain = (df_ab_freq["total"] >= thr).sum()
        pct = n_remain / total_words_before_ab * 100
        total_freq_remain = df_ab_freq.loc[df_ab_freq["total"]>=thr, "total"].sum()
        total_freq_all = df_ab_freq["total"].sum()
        coverage = total_freq_remain / total_freq_all * 100 if total_freq_all > 0 else 0
        print(f"    빈도 >= {thr:>3d}: {n_remain:>5d}개 ({pct:>5.1f}%), 토큰 커버리지={coverage:.1f}%")
        add_stat("C.빈도필터", "abuse_빈도임계값별", thr, "잔존_단어수", n_remain)
        add_stat("C.빈도필터", "abuse_빈도임계값별", thr, "토큰_커버리지(%)", round(coverage, 1))

# --- C3. MIN_DOC_COUNT = 5 (문서 단위 빈도) ---
print(f"\n--- C3. 문서 단위 빈도 (MIN_DOC_COUNT = {MIN_DOC_COUNT}) ---")
doc_word_presence = defaultdict(lambda: defaultdict(int))  # word -> abuse -> doc count
for rec in records:
    group = classify_child_group(rec)
    if group is None:
        continue
    main_abuse, _ = classify_abuse_main_sub(rec)
    if main_abuse not in ABUSE_ORDER:
        continue
    speech = extract_child_speech(rec)
    if not speech:
        continue
    joined = " ".join(speech)
    tokens_unique = set(tokenize_korean(joined))
    for w in tokens_unique:
        doc_word_presence[w][main_abuse] += 1

doc_total = {w: sum(v.values()) for w, v in doc_word_presence.items()}
ds = pd.Series(doc_total)

if not ds.empty:
    total_words_doc = len(ds)
    for thr in [1, 2, 3, 5, 8, 10]:
        n_remain = (ds >= thr).sum()
        pct = n_remain / total_words_doc * 100
        print(f"  doc_count >= {thr}: {n_remain}개 ({pct:.1f}%)")
        add_stat("C.빈도필터", "MIN_DOC_COUNT", thr, "잔존_단어수", n_remain)
        add_stat("C.빈도필터", "MIN_DOC_COUNT", thr, "비율(%)", round(pct, 1))

# ================================================================
# SECTION D: Bridge 단어 임계값 근거
# ================================================================
print("\n" + "=" * 72)
print("SECTION D: Bridge 단어 정의 임계값 근거")
print("=" * 72)

# Bridge 분석을 위한 빈도표 구성
if not df_ab_freq.empty:
    df_counts = df_ab_freq.set_index("word")[ABUSE_ORDER].copy()
    df_counts = df_counts[df_counts.sum(axis=1) >= MIN_TOTAL_COUNT_ABUSE]

    # p(abuse|word) 분포 계산
    p_distributions = []
    for w, row in df_counts.iterrows():
        total = row.sum()
        if total <= 0:
            continue
        probs = (row / total).values
        idx_sorted = np.argsort(-probs)
        p1 = probs[idx_sorted[0]]
        p2 = probs[idx_sorted[1]] if len(idx_sorted) > 1 else 0
        gap = p1 - p2
        g1 = ABUSE_ORDER[idx_sorted[0]]
        g2 = ABUSE_ORDER[idx_sorted[1]] if len(idx_sorted) > 1 else None
        p_distributions.append({
            "word": w, "total": total,
            "p1": p1, "p2": p2, "gap": gap,
            "g1": g1, "g2": g2,
        })

    df_pdist = pd.DataFrame(p_distributions)

    if not df_pdist.empty:
        print(f"\n분석 대상 단어 수 (빈도 >= {MIN_TOTAL_COUNT_ABUSE}): {len(df_pdist)}")

        # --- D1. p1 분포 ---
        print(f"\n--- D1. p1 (최고 확률) 분포 → BRIDGE_MIN_P1 = {BRIDGE_MIN_P1} 근거 ---")
        print(f"  평균: {df_pdist['p1'].mean():.4f}")
        print(f"  SD: {df_pdist['p1'].std():.4f}")
        print(f"  중앙값: {df_pdist['p1'].median():.4f}")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(df_pdist["p1"], p)
            print(f"  {p}번째 백분위수: {val:.4f}")
            add_stat("D.Bridge", "p1분포", f"BRIDGE_MIN_P1={BRIDGE_MIN_P1}", f"백분위수_{p}%", round(val, 4))

        add_stat("D.Bridge", "p1분포", f"BRIDGE_MIN_P1={BRIDGE_MIN_P1}", "평균", round(df_pdist['p1'].mean(), 4))
        add_stat("D.Bridge", "p1분포", f"BRIDGE_MIN_P1={BRIDGE_MIN_P1}", "SD", round(df_pdist['p1'].std(), 4))

        for thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
            n_pass = (df_pdist["p1"] >= thr).sum()
            pct = n_pass / len(df_pdist) * 100
            print(f"  p1 >= {thr:.2f}: {n_pass}개 ({pct:.1f}%)")
            add_stat("D.Bridge", "p1_임계값별", thr, "통과_단어수", n_pass)
            add_stat("D.Bridge", "p1_임계값별", thr, "통과_비율(%)", round(pct, 1))

        # --- D2. p2 분포 ---
        print(f"\n--- D2. p2 (차순위 확률) 분포 → BRIDGE_MIN_P2 = {BRIDGE_MIN_P2} 근거 ---")
        print(f"  평균: {df_pdist['p2'].mean():.4f}")
        print(f"  SD: {df_pdist['p2'].std():.4f}")
        print(f"  중앙값: {df_pdist['p2'].median():.4f}")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(df_pdist["p2"], p)
            print(f"  {p}번째 백분위수: {val:.4f}")
            add_stat("D.Bridge", "p2분포", f"BRIDGE_MIN_P2={BRIDGE_MIN_P2}", f"백분위수_{p}%", round(val, 4))

        for thr in [0.15, 0.20, 0.25, 0.30, 0.35]:
            n_pass = (df_pdist["p2"] >= thr).sum()
            pct = n_pass / len(df_pdist) * 100
            print(f"  p2 >= {thr:.2f}: {n_pass}개 ({pct:.1f}%)")
            add_stat("D.Bridge", "p2_임계값별", thr, "통과_단어수", n_pass)
            add_stat("D.Bridge", "p2_임계값별", thr, "통과_비율(%)", round(pct, 1))

        # --- D3. gap (p1-p2) 분포 ---
        print(f"\n--- D3. gap (p1 - p2) 분포 → BRIDGE_MAX_GAP = {BRIDGE_MAX_GAP} 근거 ---")
        print(f"  평균: {df_pdist['gap'].mean():.4f}")
        print(f"  SD: {df_pdist['gap'].std():.4f}")
        print(f"  중앙값: {df_pdist['gap'].median():.4f}")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(df_pdist["gap"], p)
            print(f"  {p}번째 백분위수: {val:.4f}")
            add_stat("D.Bridge", "gap분포", f"BRIDGE_MAX_GAP={BRIDGE_MAX_GAP}", f"백분위수_{p}%", round(val, 4))

        for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
            n_pass = (df_pdist["gap"] <= thr).sum()
            pct = n_pass / len(df_pdist) * 100
            print(f"  gap <= {thr:.2f}: {n_pass}개 ({pct:.1f}%)")
            add_stat("D.Bridge", "gap_임계값별", thr, "통과_단어수", n_pass)
            add_stat("D.Bridge", "gap_임계값별", thr, "통과_비율(%)", round(pct, 1))

        # --- D4. Bridge 조건 조합별 선택 단어 수 ---
        print(f"\n--- D4. Bridge 조건 조합별 선택 단어 수 ---")
        configs = [
            ("B_loose",    0.30, 0.20, 0.25),
            ("B0_baseline",0.40, 0.25, 0.20),
            ("B_strict",   0.45, 0.30, 0.20),
        ]
        for name, mp1, mp2, mg in configs:
            mask = (df_pdist["p1"] >= mp1) & (df_pdist["p2"] >= mp2) & (df_pdist["gap"] <= mg)
            n_bridge = mask.sum()
            pct = n_bridge / len(df_pdist) * 100
            print(f"  {name:>12s} (p1>={mp1:.2f}, p2>={mp2:.2f}, gap<={mg:.2f}): "
                  f"{n_bridge}개 ({pct:.1f}%)")
            add_stat("D.Bridge", f"config_{name}", f"p1>={mp1},p2>={mp2},gap<={mg}",
                     "bridge_단어수", n_bridge)
            add_stat("D.Bridge", f"config_{name}", f"p1>={mp1},p2>={mp2},gap<={mg}",
                     "비율(%)", round(pct, 1))

            if n_bridge > 0:
                sub = df_pdist[mask]
                # 어떤 학대유형 쌍이 bridge를 형성하는지
                pair_counts = sub.apply(
                    lambda r: "-".join(sorted([str(r["g1"]), str(r["g2"])])), axis=1
                ).value_counts()
                print(f"    학대유형 쌍 분포:")
                for pair, cnt in pair_counts.items():
                    print(f"      {pair}: {cnt}개")
                    add_stat("D.Bridge", f"config_{name}_pair분포", pair, "수", cnt)

# --- D5. BRIDGE_MIN_COUNT = 5 근거 ---
print(f"\n--- D5. BRIDGE_MIN_COUNT = {BRIDGE_MIN_COUNT} 근거 ---")
if not df_ab_freq.empty:
    # 각 단어의 학대유형별 count의 최솟값
    per_abuse_mins = df_ab_freq[ABUSE_ORDER].min(axis=1)
    # 상위 2개 abuse의 count 분포
    for thr in [1, 3, 5, 8, 10]:
        # 최소 2개 학대유형에서 count >= thr인 단어 수
        n_pass = 0
        for _, row in df_ab_freq.iterrows():
            counts_sorted = sorted(row[ABUSE_ORDER].values, reverse=True)
            if len(counts_sorted) >= 2 and counts_sorted[0] >= thr and counts_sorted[1] >= thr:
                n_pass += 1
        print(f"  상위 2개 학대유형 모두 count >= {thr}: {n_pass}개")
        add_stat("D.Bridge", "BRIDGE_MIN_COUNT", thr, "상위2_모두통과_단어수", n_pass)

# ================================================================
# SECTION E: Log-odds 관련 임계값
# ================================================================
print("\n" + "=" * 72)
print("SECTION E: Log-odds 관련 임계값 근거")
print("=" * 72)

if not df_ab_freq.empty:
    # Log-odds 계산 (simplified)
    df_c = df_ab_freq.set_index("word")[ABUSE_ORDER].copy()
    df_c = df_c[df_c.sum(axis=1) >= MIN_TOTAL_COUNT_ABUSE]

    alpha = 0.01
    V = len(df_c)
    totals = df_c.sum(axis=0)
    N_total = totals.sum()

    log_odds_records = []
    for word, row in df_c.iterrows():
        total_word = row.sum()
        for g in ABUSE_ORDER:
            c_g = row[g]
            c_not = total_word - c_g
            N_g = totals[g]
            N_not = N_total - N_g
            c_g_s = c_g + alpha
            c_not_s = c_not + alpha
            p_g = c_g_s / (N_g + alpha * V) if (N_g + alpha * V) > 0 else 0
            p_not = c_not_s / (N_not + alpha * V) if (N_not + alpha * V) > 0 else 0
            if p_g > 0 and p_g < 1 and p_not > 0 and p_not < 1:
                lo = np.log(p_g / (1-p_g)) - np.log(p_not / (1-p_not))
                var_lo = 1/c_g_s + 1/c_not_s
                se = np.sqrt(var_lo)
                z = lo / se if se > 0 else 0
            else:
                lo = 0; se = np.inf; z = 0
            log_odds_records.append({
                "word": word, "group": g, "count": c_g,
                "log_odds": lo, "se": se, "z": z,
            })

    df_lo = pd.DataFrame(log_odds_records)

    # --- E1. BRIDGE_MIN_LOGODDS = 1.0 ---
    print(f"\n--- E1. log-odds 분포 → BRIDGE_MIN_LOGODDS = {BRIDGE_MIN_LOGODDS} 근거 ---")
    lo_positive = df_lo[df_lo["log_odds"] > 0]["log_odds"]
    print(f"  log-odds > 0인 (word, group) 쌍 수: {len(lo_positive)}")
    print(f"  평균: {lo_positive.mean():.4f}")
    print(f"  SD: {lo_positive.std():.4f}")
    print(f"  중앙값: {lo_positive.median():.4f}")
    for p in [25, 50, 75, 90, 95]:
        val = np.percentile(lo_positive, p)
        print(f"  {p}번째 백분위수: {val:.4f}")
        add_stat("E.LogOdds", "log_odds_양수분포", f"MIN_LOGODDS={BRIDGE_MIN_LOGODDS}", f"백분위수_{p}%", round(val, 4))

    for thr in [0.5, 0.8, 1.0, 1.5, 2.0]:
        n_pass = (lo_positive >= thr).sum()
        pct = n_pass / len(lo_positive) * 100
        print(f"  log-odds >= {thr:.1f}: {n_pass}개 ({pct:.1f}%)")
        add_stat("E.LogOdds", "log_odds_임계값별", thr, "통과수", n_pass)
        add_stat("E.LogOdds", "log_odds_임계값별", thr, "통과비율(%)", round(pct, 1))

    # --- E2. DEFAULT_ZMIN = 1.96 ---
    print(f"\n--- E2. z_log_odds 분포 → DEFAULT_ZMIN = {DEFAULT_ZMIN} 근거 ---")
    z_positive = df_lo[df_lo["z"] > 0]["z"]
    z_finite = z_positive[np.isfinite(z_positive)]
    print(f"  z > 0이고 유한한 쌍 수: {len(z_finite)}")
    print(f"  평균: {z_finite.mean():.4f}")
    print(f"  SD: {z_finite.std():.4f}")
    for thr in [1.28, 1.65, 1.96, 2.33, 2.58]:
        n_pass = (z_finite >= thr).sum()
        pct = n_pass / len(z_finite) * 100
        alpha_z = {"1.28": "0.10", "1.65": "0.05(단측)", "1.96": "0.05(양측)",
                    "2.33": "0.01(단측)", "2.58": "0.01(양측)"}.get(f"{thr}", "")
        print(f"  z >= {thr:.2f} (α≈{alpha_z}): {n_pass}개 ({pct:.1f}%)")
        add_stat("E.LogOdds", "z_임계값별", thr, "통과수", n_pass)
        add_stat("E.LogOdds", "z_임계값별", thr, f"통과비율(%)_α≈{alpha_z}", round(pct, 1))

    # --- E3. DEFAULT_DELTA = 1.0 ---
    print(f"\n--- E3. δ (log-odds 차이) 분포 → DEFAULT_DELTA = {DEFAULT_DELTA} 근거 ---")
    delta_records = []
    for w, sub in df_lo.groupby("word"):
        sub_sorted = sub.sort_values("log_odds", ascending=False)
        if len(sub_sorted) >= 2:
            l1 = sub_sorted.iloc[0]["log_odds"]
            l2 = sub_sorted.iloc[1]["log_odds"]
            if l1 > 0 and l2 > 0:
                delta = l1 - l2
                delta_records.append({"word": w, "delta": delta, "l1": l1, "l2": l2})

    if delta_records:
        df_delta = pd.DataFrame(delta_records)
        print(f"  l1>0, l2>0인 단어 수: {len(df_delta)}")
        print(f"  δ 평균: {df_delta['delta'].mean():.4f}")
        print(f"  δ SD: {df_delta['delta'].std():.4f}")
        print(f"  δ 중앙값: {df_delta['delta'].median():.4f}")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(df_delta["delta"], p)
            print(f"  {p}번째 백분위수: {val:.4f}")
            add_stat("E.LogOdds", "delta분포", f"DEFAULT_DELTA={DEFAULT_DELTA}", f"백분위수_{p}%", round(val, 4))

        for thr in [0.5, 0.8, 1.0, 1.2, 1.5]:
            n_pass = (df_delta["delta"] < thr).sum()
            pct = n_pass / len(df_delta) * 100
            print(f"  δ < {thr:.1f} (bridge 후보): {n_pass}개 ({pct:.1f}%)")
            add_stat("E.LogOdds", "delta_임계값별", thr, "bridge후보수(δ<thr)", n_pass)

# ================================================================
# SECTION F: 임베딩 / CA 관련 임계값
# ================================================================
print("\n" + "=" * 72)
print("SECTION F: 임베딩 & CA 관련 임계값 근거")
print("=" * 72)

# --- F1. min_count = 5 (임베딩 학습) ---
print(f"\n--- F1. 임베딩 min_count = {BRIDGE_MIN_COUNT} 근거 ---")
all_tokens_flat = []
for tl in all_tokens_per_child:
    all_tokens_flat.extend(tl)
word_freq = Counter(all_tokens_flat)
total_vocab = len(word_freq)
print(f"  전체 vocabulary 크기: {total_vocab}")

for thr in [1, 2, 3, 5, 8, 10]:
    n_remain = sum(1 for w, c in word_freq.items() if c >= thr)
    pct = n_remain / total_vocab * 100
    token_coverage = sum(c for w, c in word_freq.items() if c >= thr)
    total_toks = sum(word_freq.values())
    cov_pct = token_coverage / total_toks * 100 if total_toks > 0 else 0
    print(f"  빈도 >= {thr}: vocab={n_remain}개 ({pct:.1f}%), "
          f"토큰 커버리지={cov_pct:.1f}%")
    add_stat("F.임베딩CA", "embedding_min_count", thr, "vocab수", n_remain)
    add_stat("F.임베딩CA", "embedding_min_count", thr, "토큰커버리지(%)", round(cov_pct, 1))

# --- F2. top_chi_for_ca = 200 ---
print(f"\n--- F2. CA top_chi_for_ca = 200 근거 ---")
if not df_ab_freq.empty:
    # 간략 chi2 계산
    df_c2 = df_ab_freq.set_index("word")[ABUSE_ORDER].copy()
    df_c2 = df_c2[df_c2.sum(axis=1) >= MIN_TOTAL_COUNT_ABUSE]

    if not df_c2.empty:
        col_totals = df_c2.sum(axis=0).values
        row_totals = df_c2.sum(axis=1).values
        N_chi = col_totals.sum()
        expected = np.outer(row_totals, col_totals) / N_chi
        observed = df_c2.values
        with np.errstate(divide="ignore", invalid="ignore"):
            chi_sq = (observed - expected)**2 / expected
            chi_sq = np.nan_to_num(chi_sq, nan=0, posinf=0, neginf=0)
        chi_total = chi_sq.sum(axis=1)
        df_chi = pd.DataFrame({"word": df_c2.index, "chi2": chi_total}).sort_values("chi2", ascending=False)

        total_chi_words = len(df_chi)
        print(f"  chi² 계산 대상 단어 수: {total_chi_words}")

        for k in [50, 100, 150, 200, 300, 500]:
            if k > total_chi_words:
                continue
            top_k = df_chi.head(k)
            min_chi2 = top_k["chi2"].min()
            max_chi2 = top_k["chi2"].max()
            mean_chi2 = top_k["chi2"].mean()
            print(f"  top {k}: chi2 범위=[{min_chi2:.2f}, {max_chi2:.2f}], 평균={mean_chi2:.2f}")
            add_stat("F.임베딩CA", "top_chi_for_ca", k, "최소chi2", round(min_chi2, 2))
            add_stat("F.임베딩CA", "top_chi_for_ca", k, "최대chi2", round(max_chi2, 2))
            add_stat("F.임베딩CA", "top_chi_for_ca", k, "평균chi2", round(mean_chi2, 2))

# ================================================================
# SECTION G: Log-odds smoothing alpha = 0.01
# ================================================================
print("\n" + "=" * 72)
print("SECTION G: Log-odds 스무딩 파라미터 α = 0.01 근거")
print("=" * 72)
print(f"  총 단어 수 (V): {V if not df_ab_freq.empty else 'N/A'}")
print(f"  α × V = {alpha * V:.2f}" if not df_ab_freq.empty else "  N/A")
print(f"  학대유형별 총 토큰 수:")
if not df_ab_freq.empty:
    for a in ABUSE_ORDER:
        n_tokens = totals[a]
        ratio = (alpha * V) / n_tokens * 100 if n_tokens > 0 else 0
        print(f"    {a}: N={n_tokens:.0f}, αV/N = {ratio:.4f}%")
        add_stat("G.스무딩", "alpha=0.01", a, "총토큰수", int(n_tokens))
        add_stat("G.스무딩", "alpha=0.01", a, "αV/N(%)", round(ratio, 4))

    add_stat("G.스무딩", "alpha=0.01", "-", "V(고유단어수)", V)
    add_stat("G.스무딩", "alpha=0.01", "-", "αV", round(alpha * V, 2))


# ================================================================
# 최종 출력
# ================================================================
print("\n" + "=" * 72)
print("통계 요약표 저장")
print("=" * 72)

df_stats = pd.DataFrame(all_stats)

# 출력 경로 결정
output_dir = str(ROOT)
output_path = os.path.join(output_dir, "threshold_justification_statistics.csv")
df_stats.to_csv(output_path, encoding="utf-8-sig", index=False)
print(f"\n[저장 완료] {output_path}")
print(f"  총 {len(df_stats)}개의 통계 항목이 저장되었습니다.")

# 콘솔 요약
print("\n" + "=" * 72)
print("  ★ 임계값 요약 ★")
print("=" * 72)
summary = f"""
┌──────────────────────────────────────────────────────────────────┐
│  A. 정서군 분류 임계값 (labels.py)                               │
│    - high_total_thresh = 45  (합계점수 상위 고위험 컷)            │
│    - low_total_thresh  = 10  (합계점수 하위 저위험 컷)            │
│    - risk_strong       = 25  (위험 점수 강한 기준)                │
│    - risk_weak         = 10  (위험 점수 약한 기준)                │
│    - happy_score       >= 7  (행복 문항 강제 부정)                │
│    - abuse_scores      > 6   (main_abuse 결정)                   │
│    - sub_threshold     = 2   (sub_abuse 결정)                    │
├──────────────────────────────────────────────────────────────────┤
│  B. 텍스트 처리 임계값 (text.py)                                  │
│    - 토큰 길이 > 1글자       (한 글자 토큰 제거)                  │
│    - 품사 필터: Noun, Verb, Adjective만 사용                     │
├──────────────────────────────────────────────────────────────────┤
│  C. 빈도 필터 임계값 (common.py)                                  │
│    - MIN_TOTAL_COUNT_VALENCE = {MIN_TOTAL_COUNT_VALENCE:>3d}                                │
│    - MIN_TOTAL_COUNT_ABUSE   = {MIN_TOTAL_COUNT_ABUSE:>3d}                                │
│    - MIN_DOC_COUNT           = {MIN_DOC_COUNT:>3d}                                │
├──────────────────────────────────────────────────────────────────┤
│  D. Bridge 단어 정의 임계값 (common.py / stats.py)                │
│    - BRIDGE_MIN_P1   = {BRIDGE_MIN_P1:.2f}   (최고 확률 하한)                │
│    - BRIDGE_MIN_P2   = {BRIDGE_MIN_P2:.2f}   (차순위 확률 하한)              │
│    - BRIDGE_MAX_GAP  = {BRIDGE_MAX_GAP:.2f}   (p1-p2 상한)                  │
│    - BRIDGE_MIN_COUNT= {BRIDGE_MIN_COUNT:>3d}    (학대유형별 최소 빈도)          │
├──────────────────────────────────────────────────────────────────┤
│  E. Log-odds 관련 임계값                                          │
│    - BRIDGE_MIN_LOGODDS = {BRIDGE_MIN_LOGODDS:.1f}   (log-odds 하한)              │
│    - DEFAULT_DELTA      = {DEFAULT_DELTA:.1f}   (log-odds 차이 상한)           │
│    - DEFAULT_ZMIN       = {DEFAULT_ZMIN:.2f}  (z-값 하한, α≈0.05 양측)      │
│    - alpha (스무딩)     = 0.01  (Laplace 스무딩 계수)             │
├──────────────────────────────────────────────────────────────────┤
│  F. 임베딩 / CA 관련 임계값                                       │
│    - min_count (임베딩) = 5     (gensim min_count)                │
│    - top_chi_for_ca     = 200   (CA 분석용 상위 단어 수)           │
│    - vector_size        = 100   (임베딩 차원)                     │
│    - window             = 5     (문맥 윈도우)                     │
└──────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("[완료] 모든 임계값 통계가 추출되었습니다.")
print(f"       CSV 파일: {output_path}")
