import os
os.environ["MPLBACKEND"] = "Agg"
import glob
import json
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import font_manager
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

# =========================================================
# 0. 경로 & 기본 설정
# =========================================================
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DATA_JSON_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_DIR = None
META_DIR = None
VALENCE_DIR = None
VALENCE_STATS_DIR = None
VALENCE_FIG_DIR = None
ABUSE_DIR = None
ABUSE_STATS_DIR = None
ABUSE_FIG_DIR = None


def configure_output_dirs(subset_name: str = "ALL") -> None:
    global OUTPUT_DIR, META_DIR
    global VALENCE_DIR, VALENCE_STATS_DIR, VALENCE_FIG_DIR
    global ABUSE_DIR, ABUSE_STATS_DIR, ABUSE_FIG_DIR

    subset_name = (subset_name or "ALL").upper()

    if subset_name == "NEG_ONLY":
        base_tag = "ver30_negOnly"
    else:
        base_tag = "ver30_all"

    OUTPUT_DIR = os.path.join(BASE_DIR, base_tag)

    META_DIR = os.path.join(OUTPUT_DIR, "meta")

    VALENCE_DIR = os.path.join(OUTPUT_DIR, "valence")
    VALENCE_STATS_DIR = os.path.join(VALENCE_DIR, "stats")
    VALENCE_FIG_DIR = os.path.join(VALENCE_DIR, "fig")

    ABUSE_DIR = os.path.join(OUTPUT_DIR, "abuse")
    ABUSE_STATS_DIR = os.path.join(ABUSE_DIR, "stats")
    ABUSE_FIG_DIR = os.path.join(ABUSE_DIR, "fig")

    dirs_to_make = [
        OUTPUT_DIR, META_DIR,
        VALENCE_DIR, VALENCE_STATS_DIR, VALENCE_FIG_DIR,
        ABUSE_DIR, ABUSE_STATS_DIR, ABUSE_FIG_DIR,
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    print(f"[CONFIG] subset={subset_name}, OUTPUT_DIR={OUTPUT_DIR}")


# =========================================================
# 라벨/기준
# =========================================================
VALENCE_ORDER = ["부정", "평범", "긍정"]
ABUSE_ORDER = ["방임", "정서학대", "신체학대", "성학대"]

MIN_DOC_COUNT = 5

# 폰트 (Windows/macOS 우선)
font_candidates = [
    r"C:\Windows\Fonts\malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
]
font_path = None
for _fp in font_candidates:
    if os.path.exists(_fp):
        font_path = _fp
        break

if font_path:
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
else:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[WARN] Korean font not found. 기본 폰트로 진행합니다.")
plt.rcParams["axes.unicode_minus"] = False

# SciPy (점근 p 필요하면)
try:
    from scipy.stats import chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[INFO] SciPy 미설치: permutation p만 사용합니다.")

# Okt
try:
    from konlpy.tag import Okt
    try:
        okt = Okt()
        HAS_OKT = True
        print("[INFO] Okt 형태소 분석기를 사용합니다.")
    except Exception as e:
        print(f"[INFO] Okt 초기화 실패: {e}")
        okt = None
        HAS_OKT = False
except ImportError:
    okt = None
    HAS_OKT = False
    print("[INFO] konlpy 미설치: 공백 기반 토큰화만 사용합니다. (pip install konlpy)")

STOPWORDS_BASE = {
    "자다", "모르다", "아니다", "않다", "그렇다", "싶다", "나다",
    "이다", "하다", "되다", "같다", "있다", "없다",
    "좀", "조금", "많이", "그냥", "막", "또", "근데",
    "이제", "그래서", "그러면", "그리고",
    "때문", "때문에", "거", "것",
    "사람", "애들", "애기", "어른",
    "공부", "시험", "시간",
    "거기", "여기", "저기", "이거", "저거",
    "하루", "맨날", "항상", "가끔", "자주", "매일",
    "정도", "수준", "상태", "상황", "요즘",
    "0점", "1점", "2점", "3점", "4점", "5점", "6점", "7점", "8점", "9점",
    "정상군", "위기", "위기아동", "상담필요", "관찰필요",
}


# =========================================================
# 1. 정서군 분류
# =========================================================
def classify_child_group(rec,
                        high_total_thresh=45,
                        low_total_thresh=10,
                        risk_strong=25,
                        risk_weak=10):
    info = rec.get("info", {})
    crisis = info.get("위기단계")

    try:
        total_score = int(info.get("합계점수"))
    except (TypeError, ValueError):
        total_score = np.nan

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

    if get_item_max("자해/자살", 0) > 0:
        return "부정"

    happy_score = get_item_max("행복", default=0)
    worry_score = get_item_max("걱정", default=0)
    abuse_total = get_q("학대여부", 0)

    if happy_score >= 7:
        return "부정"

    safe_crisis_for_auto_positive = {None, "정상군", "관찰필요"}
    if (happy_score == 0) and (worry_score == 0) and (abuse_total == 0) and (crisis in safe_crisis_for_auto_positive):
        return "긍정"

    label = None
    negative_crisis = {"응급", "위기아동", "학대의심", "상담필요"}

    if crisis in negative_crisis:
        label = "부정"
    elif crisis == "정상군":
        label = "긍정"
    elif crisis == "관찰필요":
        label = "평범"

    if not pd.isna(total_score):
        if total_score >= high_total_thresh:
            label = "부정"
        elif total_score <= low_total_thresh and label is None:
            label = "긍정"

    risk_questions = ["기분문제", "기본생활", "학대여부", "응급"]
    risk_score = sum(get_q(qname, 0) for qname in risk_questions)

    relation_sum = get_q("대인관계", 0)
    future_score = get_item_max("미래/진로", default=0)

    protective_index = 0
    if relation_sum <= 2:
        protective_index += 1
    if future_score == 0:
        protective_index += 1

    if label is None:
        if risk_score >= risk_strong and protective_index == 0:
            label = "부정"
        elif risk_score <= risk_weak and protective_index >= 1:
            label = "긍정"
        else:
            label = "평범"
    else:
        if label == "긍정":
            if risk_score >= risk_strong:
                label = "평범"
            if risk_score >= risk_strong + 10 and protective_index == 0:
                label = "부정"
        elif label == "평범":
            if risk_score >= risk_strong and protective_index == 0:
                label = "부정"
            elif risk_score <= risk_weak and protective_index >= 2:
                label = "긍정"

    return label if label in VALENCE_ORDER else None


# =========================================================
# 2. 임상 기반 학대유형(main/sub)
# =========================================================

def classify_abuse_main_sub(rec, abuse_order=ABUSE_ORDER,
                            sub_threshold=2,
                            use_clinical_text=True):
    """
    상담사의 임상 정보 + 학대 관련 문항 점수를 이용해
    - main_abuse : 핵심(주) 학대유형
    - sub_abuses : 부 학대유형 리스트
    를 추정.

    ──────────────────────────────────────────
    [재현성 보장 수정]
    동점(tie) 발생 시 ABUSE_SEVERITY 위계를 타이브레이커로 사용하여
    실행 결과가 항상 동일하도록 합니다.

    위계: 성학대(0) > 신체학대(1) > 정서학대(2) > 방임(3)
    숫자가 작을수록 심각 → 동점 시 우선 선택
    ──────────────────────────────────────────
    """
    info = rec.get("info", {})
    main = None
    subs = set()

    # ── 헬퍼: 결정적 정렬 키 ─────────────────────────────
    def _severity_tiebreak(abuse_type, score):
        """
        (점수, -심각도순위) 튜플을 반환합니다.

        Python의 튜플 비교 규칙:
          1) 먼저 점수를 비교 → 점수가 높은 것이 우선
          2) 점수가 같으면 -심각도순위를 비교 → 심각도가 높은 것(순위 숫자가 작은 것)이 우선

        예시:
          신체학대(점수=8, 심각도=1) → (8, -1)
          정서학대(점수=8, 심각도=2) → (8, -2)
          (8, -1) > (8, -2) 이므로 신체학대 선택

          성학대(점수=7, 심각도=0)   → (7, 0)
          신체학대(점수=8, 심각도=1) → (8, -1)
          (8, -1) > (7, 0) 이므로 신체학대 선택 (점수가 더 높으므로)
        """
        sev = ABUSE_SEVERITY.get(abuse_type, 999)
        return (score, -sev)
    # ──────────────────────────────────────────────────────

    # 1단계: 학대여부 문항의 항목별 점수 집계
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

    # 2단계: 점수 > 6인 유형 중 main 선택
    nonzero = {a: s for a, s in abuse_scores.items() if s > 6}
    if nonzero:
        # ── [수정 ①] 결정적 타이브레이커 적용 ──────────
        # 기존: main = max(nonzero, key=nonzero.get)
        # 문제: 신체학대=8, 정서학대=8이면 실행마다 다른 결과
        # 수정: (점수, -심각도) 복합 키 → 항상 동일한 결과
        main = max(nonzero,
                   key=lambda a: _severity_tiebreak(a, nonzero[a]))
        # ──────────────────────────────────────────────────

    # 3단계: sub 유형 선택 (main 제외, 점수 ≥ sub_threshold)
    for a, s in abuse_scores.items():
        if a == main:
            continue
        if s >= sub_threshold:
            subs.add(a)

    # 4단계: 임상 텍스트에서 학대유형 키워드 탐색
    if use_clinical_text:
        clin_text = " ".join(
            str(info.get(k, "")) for k in ["임상진단", "임상가 종합소견"] if k in info
        )
        # ── [수정 ②] abuse_order가 이미 위계순이므로, ────
        # 가장 심각한 유형이 먼저 순회되어 main으로 선택됨.
        # abuse_order = ["성학대", "신체학대", "정서학대", "방임"]
        # → "성학대"가 텍스트에 있으면 항상 main이 됨
        for a in abuse_order:
            if a in clin_text:
                if main is None:
                    main = a
                elif a != main:
                    subs.add(a)

    # 5단계: main이 없는데 subs가 있으면 subs 중 최고를 main으로
    if main is None and subs:
        # ── [수정 ③] 결정적 타이브레이커 적용 ──────────
        # 기존: sorted(subs, key=lambda x: abuse_scores.get(x, 0), reverse=True)[0]
        # 문제: set의 순서가 비결정적이고, 동점 시 타이브레이커 없음
        # 수정: (점수, -심각도) 복합 키로 정렬
        main = sorted(subs,
                      key=lambda x: _severity_tiebreak(x, abuse_scores.get(x, 0)),
                      reverse=True)[0]
        # ──────────────────────────────────────────────────
        subs.remove(main)

    if main is None:
        return None, []

    # subs도 위계 순서로 정렬하여 반환
    return main, sorted(subs, key=lambda x: ABUSE_SEVERITY.get(x, 999))

# =========================================================
# 3. 아동 발화 추출/토큰화
# =========================================================
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


def tokenize_korean(text: str):
    if not isinstance(text, str):
        return []

    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    stopwords = STOPWORDS_BASE

    if HAS_OKT and okt is not None:
        pos_list = okt.pos(text, stem=True)
        tokens = []
        i = 0
        while i < len(pos_list):
            word, pos = pos_list[i]

            if word in {"안", "못"} and (i + 1) < len(pos_list):
                next_word, next_pos = pos_list[i + 1]
                if next_pos in ["Verb", "Adjective"]:
                    base = next_word
                    if len(base) > 1 and base not in stopwords:
                        tokens.append(f"{word}_{base}")
                    i += 2
                    continue

            if word == "않다":
                if tokens:
                    last = tokens[-1]
                    if not last.startswith("안_"):
                        tokens[-1] = "안_" + last
                i += 1
                continue

            if pos in ["Noun", "Verb", "Adjective"]:
                if len(word) > 1 and word not in stopwords:
                    tokens.append(word)

            i += 1
        return tokens

    toks = text.split()
    return [t for t in toks if len(t) > 1 and t not in stopwords]


# =========================================================
# Global stat = sum of word-wise doc-level χ²
# =========================================================
def _load_records(json_files: List[str]) -> List[dict]:
    recs = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                recs.append(json.load(f))
        except Exception as e:
            print(f"[WARN] JSON load fail: {fp} ({e})")
    return recs


def _build_doc_token_sets(json_files: List[str],
                          test_type: str = "valence",
                          only_negative: bool = False,
                          min_doc_freq: int = 5):
    assert test_type in {"valence", "abuse_main"}

    group_order = VALENCE_ORDER[:] if test_type == "valence" else ABUSE_ORDER[:]
    group_to_idx = {g: i for i, g in enumerate(group_order)}

    recs = _load_records(json_files)

    labels = []
    doc_tokens = []

    for rec in recs:
        val = classify_child_group(rec)
        if val is None:
            continue
        if only_negative and val != "부정":
            continue

        if test_type == "valence":
            lab = val
        else:
            main, _ = classify_abuse_main_sub(rec)
            lab = main

        if lab is None or lab not in group_to_idx:
            continue

        utts = extract_child_speech(rec)
        tokset = set()
        for t in utts:
            toks = tokenize_korean(t)
            if toks:
                tokset.update(toks)

        if not tokset:
            continue

        labels.append(group_to_idx[lab])
        doc_tokens.append(tokset)

    if len(labels) == 0:
        return np.array([], dtype=int), [], group_order, [], np.zeros(len(group_order), dtype=int)

    labels_int = np.asarray(labels, dtype=int)

    # doc frequency
    df_counter: Dict[str, int] = {}
    for tokset in doc_tokens:
        for w in tokset:
            df_counter[w] = df_counter.get(w, 0) + 1

    vocab = [w for w, df in df_counter.items() if df >= int(min_doc_freq)]
    vocab.sort()

    if len(vocab) == 0:
        return np.array([], dtype=int), [], group_order, [], np.zeros(len(group_order), dtype=int)

    w2i = {w: i for i, w in enumerate(vocab)}

    doc_word_idx_list = []
    keep_idx = []
    for i, tokset in enumerate(doc_tokens):
        idxs = [w2i[w] for w in tokset if w in w2i]
        if len(idxs) == 0:
            continue
        keep_idx.append(i)
        doc_word_idx_list.append(np.asarray(idxs, dtype=int))

    labels_int = labels_int[keep_idx]
    group_counts_docs = np.bincount(labels_int, minlength=len(group_order))

    return labels_int, doc_word_idx_list, group_order, vocab, group_counts_docs


def _aggregate_group_word_table(labels_int: np.ndarray,
                               doc_word_idx_list: List[np.ndarray],
                               n_groups: int,
                               n_words: int) -> np.ndarray:
    O = np.zeros((n_groups, n_words), dtype=np.int32)
    for lab, widx in zip(labels_int, doc_word_idx_list):
        O[lab, widx] += 1
    return O


def _global_sum_word_chi2(O: np.ndarray, group_doc_counts: np.ndarray) -> Tuple[float, np.ndarray]:
    O = O.astype(float)
    N_g = group_doc_counts.astype(float)

    keep_g = N_g > 0
    O = O[keep_g, :]
    N_g = N_g[keep_g]

    G, V = O.shape
    N = N_g.sum()
    if N <= 0 or V == 0 or G < 2:
        return np.nan, np.full(V, np.nan)

    P_w = O.sum(axis=0, keepdims=True)
    E = (N_g.reshape(-1, 1) @ P_w) / N

    diff = O - E
    num = diff ** 2

    denom1 = E
    denom0 = N_g.reshape(-1, 1) - E

    m1 = denom1 > 0
    m0 = denom0 > 0

    term1 = np.zeros_like(num)
    term0 = np.zeros_like(num)

    term1[m1] = num[m1] / denom1[m1]
    term0[m0] = num[m0] / denom0[m0]

    chi2_word = (term1 + term0).sum(axis=0)
    chi2_global = float(np.nansum(chi2_word))
    return chi2_global, chi2_word


def _plot_perm_hist(perm_stats: np.ndarray, obs: float, out_png: str, title: str) -> None:
    finite = perm_stats[np.isfinite(perm_stats)]
    if finite.size == 0 or (not np.isfinite(obs)):
        print("[PLOT] skip hist (no finite stats).")
        return

    plt.figure(figsize=(7, 4))
    plt.hist(finite, bins=40, alpha=0.85)
    plt.axvline(obs, linewidth=2)
    plt.title(title)
    plt.xlabel("Global statistic (sum of word-wise doc-level χ²)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def global_doclevel_chi2sum_permutation_null(
    json_files: List[str],
    subset_name: str = "ALL",
    test_type: str = "valence",
    only_negative: bool = False,
    B: int = 1000,
    seed: int = 7,
    min_doc_freq: int = 5,
    save_prefix: str = None,
) -> Dict[str, Any]:
    configure_output_dirs(subset_name=subset_name)

    labels_int, doc_word_idx_list, group_order, vocab, group_counts_docs = _build_doc_token_sets(
        json_files=json_files,
        test_type=test_type,
        only_negative=only_negative,
        min_doc_freq=min_doc_freq,
    )

    # ---- (A) 기본 필터링 실패 ----
    if labels_int.size == 0 or len(vocab) == 0:
        print("[GLOBAL-CHI2SUM] empty after filtering")
        return {"ok": False, "reason": "empty_after_filter"}

    # ---- (B) 유효 그룹 수 체크 ----
    n_groups = len(group_order)
    nonempty_groups = int(np.sum(np.asarray(group_counts_docs) > 0))
    if nonempty_groups < 2:
        if save_prefix is None:
            save_prefix = f"{subset_name}_{test_type}_DOCLEVEL".upper()
        out_dir = VALENCE_STATS_DIR if test_type == "valence" else ABUSE_STATS_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_summary = os.path.join(out_dir, f"{save_prefix}_global_chi2sum_perm_summary.json")

        summary = {
            "ok": False,
            "reason": "insufficient_groups",
            "subset_name": subset_name,
            "test_type": test_type,
            "only_negative": bool(only_negative),
            "min_doc_freq": int(min_doc_freq),
            "n_docs": int(labels_int.size),
            "n_words": int(len(vocab)),
            "group_order": group_order,
            "group_counts_docs_obs": group_counts_docs.tolist(),
            "nonempty_groups": nonempty_groups,
        }
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[GLOBAL-CHI2SUM] skip: only {nonempty_groups} non-empty group(s). saved: {out_summary}")
        return summary

    n_words = len(vocab)
    n_docs = int(labels_int.size)

    # observed
    O_obs = _aggregate_group_word_table(labels_int, doc_word_idx_list, n_groups, n_words)
    S_obs, _ = _global_sum_word_chi2(O_obs, group_counts_docs)

    if not np.isfinite(S_obs):
        print("[GLOBAL-CHI2SUM] S_obs is NaN/inf -> skip.")
        return {"ok": False, "reason": "nan_obs"}

    # permutation (라벨 permute는 그룹 카운트가 변하지 않음)
    rng = np.random.default_rng(seed)
    perm_stats = np.zeros(B, dtype=float)

    for b in range(B):
        perm_labels = rng.permutation(labels_int)
        O_b = _aggregate_group_word_table(perm_labels, doc_word_idx_list, n_groups, n_words)
        S_b, _ = _global_sum_word_chi2(O_b, group_counts_docs)  # ✅ group_counts_docs 그대로
        perm_stats[b] = S_b

    finite = perm_stats[np.isfinite(perm_stats)]
    p_perm = None
    if finite.size > 0:
        p_perm = float((np.sum(finite >= S_obs) + 1) / (finite.size + 1))

    # 저장
    if save_prefix is None:
        save_prefix = f"{subset_name}_{test_type}_DOCLEVEL".upper()

    out_dir = VALENCE_STATS_DIR if test_type == "valence" else ABUSE_STATS_DIR
    os.makedirs(out_dir, exist_ok=True)

    out_summary = os.path.join(out_dir, f"{save_prefix}_global_chi2sum_perm_summary.json")
    out_stats   = os.path.join(out_dir, f"{save_prefix}_global_chi2sum_perm_stats.csv")
    out_fig     = os.path.join(out_dir, f"{save_prefix}_global_chi2sum_perm_hist.png")

    summary = {
        "ok": True,
        "subset_name": subset_name,
        "test_type": test_type,
        "only_negative": bool(only_negative),
        "B": int(B),
        "seed": int(seed),
        "min_doc_freq": int(min_doc_freq),
        "n_docs": n_docs,
        "n_words": n_words,
        "group_order": group_order,
        "group_counts_docs_obs": group_counts_docs.tolist(),
        "global_stat_sum_word_chi2_obs": float(S_obs),
        "p_perm": p_perm,
        "perm_mean": (float(np.mean(finite)) if finite.size > 0 else None),
        "perm_std": (float(np.std(finite, ddof=1)) if finite.size > 1 else None),
        "finite_perm_n": int(finite.size),
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame({"global_stat_perm": perm_stats}).to_csv(out_stats, index=False, encoding="utf-8-sig")

    title = f"Global doc-level χ²-sum permutation null ({save_prefix})\nobs={S_obs:.2f}, p_perm={p_perm}, B={B}"
    _plot_perm_hist(perm_stats, S_obs, out_fig, title=title)

    print(f"[GLOBAL-CHI2SUM] obs={S_obs:.4f}, p_perm={p_perm}")
    print(f"[GLOBAL-CHI2SUM] saved: {out_summary}")
    return summary


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(json_files, subset_name="ALL", only_negative=False):
    subset_upper = (subset_name or "").upper()

    # 1) ALL: valence 전역검정 (NEG_ONLY는 single-group라 skip)
    if subset_upper != "NEG_ONLY":
        global_doclevel_chi2sum_permutation_null(
            json_files=json_files,
            subset_name=subset_name,
            test_type="valence",
            only_negative=False,
            B=1000,
            seed=7,
            min_doc_freq=MIN_DOC_COUNT,
            save_prefix="ALL_VALENCE_DOCLEVEL",
        )
    else:
        print("[PIPE] skip valence global test for NEG_ONLY (single-group).")

    # 2) ✅ 추가: ALL에서도 abuse_main 전역검정 (ABUSE_ALL)
    #    (only_negative=False, 즉 ALL 코퍼스에서 main_abuse 라벨을 가진 문서들로 구성됨)
    if subset_upper != "NEG_ONLY":
        global_doclevel_chi2sum_permutation_null(
            json_files=json_files,
            subset_name=subset_name,
            test_type="abuse_main",
            only_negative=False,
            B=1000,
            seed=7,
            min_doc_freq=MIN_DOC_COUNT,
            save_prefix="ABUSE_ALL_DOCLEVEL",
        )

    # 3) NEG_ONLY: abuse_main 전역검정 (ABUSE_NEG)
    if subset_upper == "NEG_ONLY":
        global_doclevel_chi2sum_permutation_null(
            json_files=json_files,
            subset_name=subset_name,
            test_type="abuse_main",
            only_negative=True,
            B=1000,
            seed=7,
            min_doc_freq=MIN_DOC_COUNT,
            save_prefix="ABUSE_NEG_DOCLEVEL",
        )


def main():
    json_files = sorted(glob.glob(os.path.join(DATA_JSON_DIR, "*.json")))
    print(f"[MAIN] #json_files = {len(json_files)}")

    # ALL: valence + abuse_all
    run_pipeline(json_files, subset_name="ALL", only_negative=False)

    # NEG_ONLY: abuse_neg
    run_pipeline(json_files, subset_name="NEG_ONLY", only_negative=True)


if __name__ == "__main__":
    main()
