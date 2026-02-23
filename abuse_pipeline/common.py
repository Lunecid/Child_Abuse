from __future__ import annotations

import os
import re
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =========================================================
# 0. Project paths (package lives under <ROOT>/abuse_pipeline)
# =========================================================
def _find_project_root(start: Path) -> Path:
    p = start
    for _ in range(10):  # 위로 10단계까지만 탐색
        if (p / "data").exists():
            return p
        p = p.parent
    return start.parents[2]  # fallback

ROOT_DIR = _find_project_root(Path(__file__).resolve())
BASE_DIR = str(ROOT_DIR)
DATA_JSON_DIR = str(ROOT_DIR / "data")



# Subset별 출력 디렉토리는 configure_output_dirs(...)에서 설정
OUTPUT_DIR: str | None = None
META_DIR: str | None = None
VALENCE_DIR: str | None = None
VALENCE_STATS_DIR: str | None = None
VALENCE_FIG_DIR: str | None = None
ABUSE_DIR: str | None = None
ABUSE_STATS_DIR: str | None = None
ABUSE_FIG_DIR: str | None = None
EMBED_DIR: str | None = None
EMBED_MODEL_DIR: str | None = None
EMBED_PROJ_DIR: str | None = None
CA_PROB_DIR: str | None = None
BRIDGE_PROB_ABLATION_DIR: str | None = None
BRIDGE_DELTA_DIR: str | None = None

# =========================================================
# 0-1. Orders & colors
# =========================================================
VALENCE_ORDER = ["부정", "평범", "긍정"]
VALENCE_COLORS = {
    "부정": "#d62728",
    "평범": "#1f77b4",
    "긍정": "#2ca02c",
}

ABUSE_ORDER = ["성학대", "신체학대", "정서학대", "방임"]
# =========================================================
# 0-1b. English labels for plots
# =========================================================
ABUSE_LABEL_EN = {
    "방임": "Neglect",
    "정서학대": "Emotional abuse",
    "신체학대": "Physical abuse",
    "성학대": "Sexual abuse",
}
# =========================================================
# 0-1c. Abuse clusters (for connectivity emphasis in CA plot)
# =========================================================
ABUSE_CLUSTERS = {
    "Continuum": ["방임", "정서학대", "신체학대"],
    "Distinct": ["성학대"],
}

def abuse_label(x: str, lang: str = "en") -> str:
    if lang == "en":
        return ABUSE_LABEL_EN.get(x, x)
    return x

ABUSE_COLORS = {
    "방임": "#1f77b4",
    "정서학대": "#ff7f0e",
    "신체학대": "#2ca02c",
    "성학대": "#d62728",
}

# =========================================================
# 0-2. Bridge configs / thresholds
# =========================================================
DEFAULT_DELTA = 1.0
DEFAULT_ZMIN = 1.96
BRIDGE_MIN_LOGODDS = 1.0
BRIDGE_MIN_COUNT = 5

BRIDGE_DELTA_CANDS = [0.5, 0.8, 1.0, 1.2, 1.5]
JACCARD_REF_DELTA = 1.0

BRIDGE_MIN_P1 = 0.40
BRIDGE_MIN_P2 = 0.25
BRIDGE_MAX_GAP = 0.20

BRIDGE_P_CONFIGS = [
    {"name": "B0_baseline", "min_p1": 0.40, "min_p2": 0.25, "max_gap": 0.20},
    {"name": "B_loose",     "min_p1": 0.30, "min_p2": 0.20, "max_gap": 0.25},
    {"name": "B_strict",    "min_p1": 0.45, "min_p2": 0.30, "max_gap": 0.20},
]

BRIDGE_FILTER_CONFIGS = [
    {
        "name": cfg["name"],
        "min_p1": cfg["min_p1"],
        "min_p2": cfg["min_p2"],
        "max_gap": cfg["max_gap"],
        "logodds_min": None,
        "count_min": BRIDGE_MIN_COUNT,
        "z_min": None,
    }
    for cfg in BRIDGE_P_CONFIGS
]

TOP_K_VALENCE_WC = 120
TOP_N_TABLE_VALENCE = 30
TOP_K_ABUSE_WC = 120
TOP_N_TABLE_ABUSE = 30

MIN_TOTAL_COUNT_VALENCE = 10
MIN_TOTAL_COUNT_ABUSE = 8
MIN_DOC_COUNT = 5

# =========================================================
# 0-3. Stopwords
#   ※ '엄마', '아빠', '선생님', '학교' 등은 제거하지 않음
# =========================================================
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

STOPWORDS_FOR_CA = STOPWORDS_BASE | {"자다", "모르다", "아니다", "않다"}

# 임베딩 학습용 전역 코퍼스 (각 발화 단위 토큰 리스트)
EMBEDDING_CORPUS: list[list[str]] = []

# =========================================================
# 0-4. Optional dependencies flags
# =========================================================
try:
    from scipy.stats import chi2, chi2_contingency, f_oneway, ttest_ind, spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except Exception:
    HAS_ADJUSTTEXT = False

# konlpy Okt (형태소)
try:
    from konlpy.tag import Okt
    try:
        okt = Okt()
        HAS_OKT = True
    except Exception:
        okt = None
        HAS_OKT = False
except Exception:
    okt = None
    HAS_OKT = False

# prince (CA)
try:
    import prince
    HAS_PRINCE = True
except Exception:
    HAS_PRINCE = False

# statsmodels
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# gensim
try:
    from gensim.models import Word2Vec, FastText
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

# =========================================================
# 0-5. Token translation (KO->EN) for figures
# =========================================================
TOKEN_TRANS_CACHE_BASENAME = "token_translation_ko2en.json"

# 1) 수동(도메인) 사전
MANUAL_KO2EN: dict[str, str] = {
    # "폭력": "violence",
    # "학대": "abuse",
    # "엄마": "mother",
    # "아빠": "father",
}


def _load_json(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path: str, obj: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class TokenTranslator:
    """KO->EN token translator with on-disk cache for reproducibility."""

    def __init__(self, cache_dir: str | None, use_auto: bool = True):
        self.cache_path = os.path.join(cache_dir, TOKEN_TRANS_CACHE_BASENAME) if cache_dir else None
        self.cache = _load_json(self.cache_path) if self.cache_path else {}
        self.use_auto = use_auto
        self.auto = None

        if use_auto:
            try:
                from deep_translator import GoogleTranslator

                self.auto = GoogleTranslator(source="ko", target="en")
            except Exception:
                self.auto = None

    def translate(self, token: str) -> str:
        if not token:
            return token

        if token in self.cache:
            return self.cache[token]

        if token in MANUAL_KO2EN:
            en = MANUAL_KO2EN[token]
            self.cache[token] = en
            return en

        en = token
        if self.auto is not None:
            try:
                en = self.auto.translate(token)
            except Exception:
                en = token

        self.cache[token] = en
        return en

    def save(self):
        if self.cache_path:
            _save_json(self.cache_path, self.cache)


def disp_token(token: str, lang: str, tt: TokenTranslator | None = None, mode: str = "en_only") -> str:
    """Translate token for plots.

    mode:
      - en_only: English only
      - en_ko: English + (Korean)
    """
    if lang != "en" or tt is None:
        return token

    en = tt.translate(token)
    return f"{en}\n({token})" if mode == "en_ko" else en


# =========================================================
# 0-6. Output paths (BUGFIX: robust separation for ALL vs NEG_ONLY)
# =========================================================

def _looks_like_path(s: str) -> bool:
    if not s:
        return False
    if os.path.sep in s or (os.path.altsep and os.path.altsep in s):
        return True
    # windows drive like C:\...
    if re.match(r"^[A-Za-z]:\\", s):
        return True
    return False


def configure_output_dirs(subset_name: str = "ALL", base_dir: str | None = None, version_tag: str = "ver28") -> None:
    """Configure global output directories.

    Parameters
    ----------
    subset_name:
        Either "ALL" / "NEG_ONLY" (preferred), OR an explicit output root path.
        Backward-compatible behavior: if a path is passed, it is used as OUTPUT_DIR.

    base_dir:
        Project root (defaults to BASE_DIR).

    version_tag:
        Prefix for auto-generated output directory names (e.g., ver28_all).

    Notes
    -----
    Fixes the bug where ALL and NEG_ONLY were written into the same directory when
    configure_output_dirs() was mistakenly called with a path-like string.
    """
    global OUTPUT_DIR, META_DIR
    global VALENCE_DIR, VALENCE_STATS_DIR, VALENCE_FIG_DIR
    global ABUSE_DIR, ABUSE_STATS_DIR, ABUSE_FIG_DIR
    global EMBED_DIR, EMBED_MODEL_DIR, EMBED_PROJ_DIR
    global CA_PROB_DIR, BRIDGE_PROB_ABLATION_DIR, BRIDGE_DELTA_DIR

    base_dir = base_dir or BASE_DIR
    print("[DEBUG] configure_output_dirs called with subset_name =", repr(subset_name))
    print("[DEBUG] looks_like_path =", _looks_like_path(subset_name))

    # Backward compatible: if caller passes an explicit path, respect it.
    if _looks_like_path(subset_name):
        out_root = subset_name
    else:
        sn = (subset_name or "ALL").upper()
        subset_key = "negOnly" if sn in {"NEG_ONLY", "NEGONLY", "NEG"} else "all"
        out_root = os.path.join(base_dir, f"{version_tag}_{subset_key}")

    OUTPUT_DIR = out_root
    META_DIR = os.path.join(OUTPUT_DIR, "meta")

    VALENCE_DIR = os.path.join(OUTPUT_DIR, "valence")
    VALENCE_STATS_DIR = os.path.join(VALENCE_DIR, "stats")
    VALENCE_FIG_DIR = os.path.join(VALENCE_DIR, "figures")

    ABUSE_DIR = os.path.join(OUTPUT_DIR, "abuse")
    ABUSE_STATS_DIR = os.path.join(ABUSE_DIR, "stats")
    ABUSE_FIG_DIR = os.path.join(ABUSE_DIR, "figures")

    EMBED_DIR = os.path.join(OUTPUT_DIR, "embeddings")
    EMBED_MODEL_DIR = os.path.join(EMBED_DIR, "models")
    EMBED_PROJ_DIR = os.path.join(EMBED_DIR, "projections")

    CA_PROB_DIR = os.path.join(OUTPUT_DIR, "ca_prob")
    BRIDGE_PROB_ABLATION_DIR = os.path.join(OUTPUT_DIR, "pbridge_ablation")
    BRIDGE_DELTA_DIR = os.path.join(OUTPUT_DIR, "delta_bridge")

    for d in [
        META_DIR,
        VALENCE_STATS_DIR,
        VALENCE_FIG_DIR,
        ABUSE_STATS_DIR,
        ABUSE_FIG_DIR,
        EMBED_MODEL_DIR,
        EMBED_PROJ_DIR,
        CA_PROB_DIR,
        BRIDGE_PROB_ABLATION_DIR,
        BRIDGE_DELTA_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
