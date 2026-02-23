#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""revision_appendix_runner.py

(논문 Appendix용) 1회 실행으로 아래 3가지를 모두 산출합니다.

A) 다중라벨(soft label) 보조 분석
   - 각 아동의 '학대여부' 문항에서 학대유형별 합산점수(A-score)를 추출
   - A-score를 확률분포(soft label)로 변환
   - 텍스트(TF-IDF) → soft label을 예측하는 간단 모델(다중출력 Ridge 회귀)을 학습/평가
   - out-of-fold 예측과 KL/Brier/Cosine 등 확률분포 기반 지표를 저장

B) 전처리 민감도(robustness)
   - 1글자 토큰 제거 on/off
   - 토크나이저 대안(whitespace)
   - 품사 포함 범위 변화(명사만 vs N/V/Adj vs N/V/Adj/Adv)
   - 각 설정별 doc-level(문서 존재) 빈도표 → χ² 상위 단어로 CA → 학대유형 row좌표 산출
   - baseline 대비 Procrustes R² + top-word Jaccard 등 구조 불변성 지표 저장

C) GT 부재/매칭 불가(excluded) 편향 점검 + 민감도(최선/최악) 분석
   - info['학대의심'] 등 GT 필드에서 학대유형 라벨을 추출/정규화
   - (i) GT missing, (ii) GT unmappable(4대 유형으로 매핑 불가) 사례 분포를 비교
   - 제외 사례가 어떤 특성을 가지는지(정서군, main_abuse, 점수/모호성 등) 요약
   - 평가 지표에 대한 best/worst-case bound를 산출

실행 예:
  python revision_appendix_runner.py --data_dir ./data --out_dir ./revision_appendix_out --only_negative

주의:
  - 스크립트를 abuse_pipeline 폴더에서 직접 실행하면 ./data 상대경로가 달라져
    JSON을 못 찾는 경우가 많습니다. 이 스크립트는 project-root/data를 자동 탐색하지만,
    그래도 실패하면 --data_dir에 절대경로를 주는 것을 권장합니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 0) Import bootstrap: allow running as a file (PyCharm) or as a module
# ---------------------------------------------------------------------

def _bootstrap_syspath() -> None:
    """Add probable project roots to sys.path so that imports work in 3 modes:

    1) python -m v28_refactor.abuse_pipeline.revision_appendix_runner
    2) python v28_refactor/abuse_pipeline/revision_appendix_runner.py
    3) PyCharm 'Run file' from within abuse_pipeline
    """
    this_file = Path(__file__).resolve()
    abuse_dir = this_file.parent
    v28_dir = abuse_dir.parent
    proj_root = v28_dir.parent

    for p in [proj_root, v28_dir, abuse_dir]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


_bootstrap_syspath()

_IMPORT_OK = False
try:
    from . import common as C  # type: ignore
    from .labels import classify_child_group, classify_abuse_main_sub  # type: ignore
    from .text import extract_child_speech, tokenize_korean  # type: ignore
    from .stats import compute_chi_square  # type: ignore
    _IMPORT_OK = True
except Exception:
    pass

if not _IMPORT_OK:
    try:
        from v28_refactor.abuse_pipeline import common as C  # type: ignore
        from v28_refactor.abuse_pipeline.labels import classify_child_group, classify_abuse_main_sub  # type: ignore
        from v28_refactor.abuse_pipeline.text import extract_child_speech, tokenize_korean  # type: ignore
        from v28_refactor.abuse_pipeline.stats import compute_chi_square  # type: ignore
        _IMPORT_OK = True
    except Exception as e:
        raise ImportError(
            "abuse_pipeline import 실패. 이 파일을 v28_refactor/abuse_pipeline 아래에 두고 실행하거나, "
            "python -m v28_refactor.abuse_pipeline.revision_appendix_runner 형태로 실행하세요. "
            f"원인: {e}"
        )


ABUSE_ORDER: List[str] = list(getattr(C, "ABUSE_ORDER", ["성학대", "신체학대", "정서학대", "방임"]))
VALENCE_ORDER: List[str] = list(getattr(C, "VALENCE_ORDER", ["부정", "평범", "긍정"]))


# ---------------------------------------------------------------------
# 1) I/O helpers
# ---------------------------------------------------------------------

def _find_candidate_project_roots(start: Path) -> List[Path]:
    """Heuristic: walk up and collect likely roots."""
    roots: List[Path] = []
    p = start.resolve()
    for _ in range(10):
        roots.append(p)
        p = p.parent
    # de-dup
    out: List[Path] = []
    seen = set()
    for r in roots:
        rs = str(r)
        if rs not in seen:
            out.append(r)
            seen.add(rs)
    return out


def resolve_data_dir(data_dir: str) -> Path:
    """Resolve data directory robustly.

    - If data_dir exists -> use it.
    - Else, search upward from cwd and from this script location for a folder named like data_dir.
    """
    p = Path(data_dir)
    if p.exists():
        return p.resolve()

    start_points = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    target_name = p.name

    for sp in start_points:
        for cand_root in _find_candidate_project_roots(sp):
            cand = (cand_root / target_name)
            if cand.exists() and cand.is_dir():
                return cand.resolve()

    for sp in start_points:
        for cand_root in _find_candidate_project_roots(sp):
            cand = cand_root / "data"
            if cand.exists() and cand.is_dir():
                return cand.resolve()

    return p.resolve()  # will not exist


def find_json_files(data_dir: Path) -> List[Path]:
    if data_dir.is_file() and data_dir.suffix.lower() == ".json":
        return [data_dir]
    if not data_dir.exists() or not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.json"))


def _read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_records(json_obj: Any) -> List[Dict[str, Any]]:
    """Normalize file structure: dict or list[dict]."""
    if isinstance(json_obj, dict) and ("info" in json_obj or "list" in json_obj):
        return [json_obj]
    if isinstance(json_obj, list):
        return [x for x in json_obj if isinstance(x, dict)]
    return []


# ---------------------------------------------------------------------
# 2) A-score → soft label
# ---------------------------------------------------------------------

def extract_abuse_scores(rec: Dict[str, Any], abuse_order: Sequence[str] = ABUSE_ORDER) -> Dict[str, int]:
    """A-score: 학대여부 문항에서 학대유형별 합산점수를 계산."""
    scores = {a: 0 for a in abuse_order}
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
            for a in abuse_order:
                if a in name:
                    scores[a] += sc
    return scores


def scores_to_prob(scores: Dict[str, int], abuse_order: Sequence[str] = ABUSE_ORDER, *, alpha: float = 0.5) -> Optional[np.ndarray]:
    """Convert nonnegative scores to a probability vector (add-α smoothing).
    Returns None if all scores are 0.
    """
    v = np.array([max(0, int(scores.get(a, 0))) for a in abuse_order], dtype=float)
    s = float(v.sum())
    if s <= 0:
        return None
    v = v + float(alpha)
    v = v / float(v.sum())
    return v


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------
# 3) GT parsing (missing / unmappable)
# ---------------------------------------------------------------------

_CANON_MAP = {
    "신체적학대": "신체학대",
    "정서적학대": "정서학대",
    "성적학대": "성학대",
    "성폭력": "성학대",
    "성폭행": "성학대",
    "유기": "방임",
}


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_to_text(v) for v in x)
    if isinstance(x, dict):
        for k in ("val", "text", "value"):
            if k in x:
                return _to_text(x.get(k))
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def normalize_abuse_label(label: str) -> str:
    s = re.sub(r"\s+", "", str(label))
    s = _CANON_MAP.get(s, s)
    if s.endswith("적학대"):
        s = s.replace("적학대", "학대")
    return _CANON_MAP.get(s, s)


def extract_gt_labels(
    info: Dict[str, Any],
    *,
    field: str = "학대의심",
    abuse_order: Sequence[str] = ABUSE_ORDER,
) -> Tuple[Set[str], Set[str], str]:
    """Extract GT labels from info[field].
    Returns (mapped_set, unmapped_set, raw_text)
    """
    raw = info.get(field, "")
    text = _to_text(raw)
    text_nospace = re.sub(r"\s+", "", text)

    found_all: Set[str] = set()

    for m in re.findall(r"([가-힣]+?)학대", text_nospace):
        cand = normalize_abuse_label(m + "학대")
        found_all.add(cand)

    if "방임" in text_nospace:
        found_all.add(normalize_abuse_label("방임"))
    if "유기" in text_nospace:
        found_all.add(normalize_abuse_label("유기"))

    for a in abuse_order:
        if re.sub(r"\s+", "", a) in text_nospace:
            found_all.add(normalize_abuse_label(a))

    found_all = {normalize_abuse_label(x) for x in found_all if x}
    mapped = {x for x in found_all if x in set(abuse_order)}
    unmapped = found_all - mapped

    return mapped, unmapped, text


# ---------------------------------------------------------------------
# 4) Dataset build
# ---------------------------------------------------------------------

@dataclass
class Doc:
    doc_id: str
    source_file: str
    valence: Optional[str]
    crisis: Optional[str]
    total_score: Optional[float]
    main_abuse: Optional[str]
    sub_abuses: List[str]
    a_scores: Dict[str, int]
    a_prob: Optional[np.ndarray]
    a_entropy: Optional[float]
    a_gap12: Optional[float]
    raw_text: str
    tfidf_text: str
    gt_mapped: Set[str]
    gt_unmapped: Set[str]
    gt_raw_text: str


def build_docs(
    json_files: Sequence[Path],
    *,
    only_negative: bool,
    gt_field: str = "학대의심",
    alpha_smooth: float = 0.5,
) -> List[Doc]:
    docs: List[Doc] = []

    for fp in json_files:
        try:
            obj = _read_json_any(fp)
        except Exception:
            continue

        recs = iter_records(obj)
        for ridx, rec in enumerate(recs):
            info = rec.get("info", {}) or {}
            doc_id = info.get("ID") or info.get("id") or info.get("Id")
            if not doc_id:
                doc_id = f"{fp.stem}__{ridx}"

            try:
                valence = classify_child_group(rec)
            except Exception:
                valence = None

            if only_negative and valence != "부정":
                continue

            crisis = info.get("위기단계")
            try:
                total_score = float(info.get("합계점수"))
            except Exception:
                total_score = np.nan

            try:
                main_abuse, subs = classify_abuse_main_sub(rec)
            except Exception:
                main_abuse, subs = (None, [])
            subs = subs or []

            try:
                speech_list = extract_child_speech(rec) or []
            except Exception:
                speech_list = []

            raw_text = " ".join([t for t in speech_list if isinstance(t, str)])

            toks = tokenize_korean(raw_text)
            tfidf_text = " ".join(toks)

            a_scores = extract_abuse_scores(rec)
            a_prob = scores_to_prob(a_scores, alpha=alpha_smooth)
            if a_prob is not None:
                a_ent = entropy(a_prob)
                srt = np.sort(a_prob)[::-1]
                a_gap12 = float(srt[0] - srt[1]) if len(srt) >= 2 else np.nan
            else:
                a_ent, a_gap12 = None, None

            gt_mapped, gt_unmapped, gt_raw_text = extract_gt_labels(info, field=gt_field)

            docs.append(
                Doc(
                    doc_id=str(doc_id),
                    source_file=str(fp),
                    valence=valence,
                    crisis=crisis,
                    total_score=None if pd.isna(total_score) else float(total_score),
                    main_abuse=main_abuse,
                    sub_abuses=list(subs),
                    a_scores=a_scores,
                    a_prob=a_prob,
                    a_entropy=a_ent,
                    a_gap12=a_gap12,
                    raw_text=raw_text,
                    tfidf_text=tfidf_text,
                    gt_mapped=set(gt_mapped),
                    gt_unmapped=set(gt_unmapped),
                    gt_raw_text=gt_raw_text,
                )
            )

    return docs


# ---------------------------------------------------------------------
# 5) Soft-label model (TF-IDF → prob vector)
# ---------------------------------------------------------------------

def _safe_normalize_rows(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    m = np.clip(m, 0.0, None)
    s = m.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    m = m / s
    m = np.clip(m, eps, 1.0)
    m = m / m.sum(axis=1, keepdims=True)
    return m


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def cosine_sim(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pn = np.linalg.norm(p, axis=1, keepdims=True)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    pn = np.where(pn <= 0, eps, pn)
    qn = np.where(qn <= 0, eps, qn)
    return np.sum((p / pn) * (q / qn), axis=1)


def brier_multiclass(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Multi-class Brier (mean squared error across classes)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.mean((p - q) ** 2, axis=1)


def run_softlabel_aux_analysis(
    docs: Sequence[Doc],
    *,
    out_dir: Path,
    k_folds: int = 5,
    max_features: int = 30000,
    min_df: int = 2,
    seed: int = 42,
    require_algo_main: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in docs:
        if require_algo_main and (d.main_abuse not in ABUSE_ORDER):
            continue
        if d.a_prob is None:
            continue
        if not d.tfidf_text.strip():
            continue
        rows.append(d)

    if len(rows) < 30:
        print(f"[SOFT] data too small: n={len(rows)} (skip)")
        return

    texts = [d.tfidf_text for d in rows]
    Y = np.vstack([d.a_prob for d in rows])
    dominant = np.argmax(Y, axis=1)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import StratifiedKFold
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import Ridge
    except Exception as e:
        print(f"[SOFT] scikit-learn import failed -> skip soft-label analysis: {e}")
        return

    vec = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
    )

    X = vec.fit_transform(texts)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros_like(Y)
    fold_rows = []

    for fold, (tr, te) in enumerate(skf.split(X, dominant), start=1):
        base = Ridge(alpha=1.0, random_state=seed)
        model = MultiOutputRegressor(base)
        model.fit(X[tr], Y[tr])

        pred = model.predict(X[te])
        pred = _safe_normalize_rows(pred)
        oof_pred[te] = pred

        kls = kl_div(Y[te], pred)
        cos = cosine_sim(Y[te], pred)
        br = brier_multiclass(Y[te], pred)
        acc = (np.argmax(Y[te], axis=1) == np.argmax(pred, axis=1)).mean()

        fold_rows.append(
            {
                "fold": fold,
                "n_test": len(te),
                "kl_mean": float(np.mean(kls)),
                "kl_median": float(np.median(kls)),
                "cosine_mean": float(np.mean(cos)),
                "brier_mean": float(np.mean(br)),
                "top1_acc": float(acc),
            }
        )

    kls = kl_div(Y, oof_pred)
    cos = cosine_sim(Y, oof_pred)
    br = brier_multiclass(Y, oof_pred)
    acc = (np.argmax(Y, axis=1) == np.argmax(oof_pred, axis=1)).mean()

    overall = {
        "fold": "ALL_OOF",
        "n_test": len(rows),
        "kl_mean": float(np.mean(kls)),
        "kl_median": float(np.median(kls)),
        "cosine_mean": float(np.mean(cos)),
        "brier_mean": float(np.mean(br)),
        "top1_acc": float(acc),
    }

    df_fold = pd.DataFrame(fold_rows + [overall])
    df_fold.to_csv(out_dir / "softlabel_metrics_oof.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "doc_id": [d.doc_id for d in rows],
        "main_abuse": [d.main_abuse or "" for d in rows],
        "valence": [d.valence or "" for d in rows],
        "a_entropy": [d.a_entropy if d.a_entropy is not None else np.nan for d in rows],
        "a_gap12": [d.a_gap12 if d.a_gap12 is not None else np.nan for d in rows],
    })

    for j, a in enumerate(ABUSE_ORDER):
        pred_df[f"y_true_{a}"] = Y[:, j]
        pred_df[f"y_pred_{a}"] = oof_pred[:, j]

    pred_df["kl"] = kls
    pred_df["cosine"] = cos
    pred_df["brier"] = br
    pred_df["top1_true"] = [ABUSE_ORDER[i] for i in np.argmax(Y, axis=1)]
    pred_df["top1_pred"] = [ABUSE_ORDER[i] for i in np.argmax(oof_pred, axis=1)]

    pred_df.to_csv(out_dir / "softlabel_predictions_oof.csv", index=False, encoding="utf-8-sig")

    print(f"[SOFT] saved -> {out_dir}")


# ---------------------------------------------------------------------
# 6) Preprocessing sensitivity: tokenizer variants → CA → Procrustes
# ---------------------------------------------------------------------

def tokenize_custom_okt(
    text: str,
    *,
    pos_tags: Sequence[str],
    min_len: int,
    remove_stopwords: bool = True,
    use_stem: bool = True,
) -> List[str]:
    """Customizable tokenization based on project's Okt object (if available)."""
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    stopwords = set(getattr(C, "STOPWORDS_BASE", set())) if remove_stopwords else set()

    if getattr(C, "HAS_OKT", False) and getattr(C, "okt", None) is not None:
        pos_list = C.okt.pos(text, stem=use_stem)
        out: List[str] = []
        for w, pos in pos_list:
            if pos not in set(pos_tags):
                continue
            if len(w) < min_len:
                continue
            if w in stopwords:
                continue
            out.append(w)
        return out

    toks = text.split()
    toks = [t for t in toks if len(t) >= min_len and t not in stopwords]
    return toks


def tokenize_whitespace(text: str, *, min_len: int = 2, remove_stopwords: bool = True) -> List[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    stopwords = set(getattr(C, "STOPWORDS_BASE", set())) if remove_stopwords else set()
    toks = [t for t in text.split() if len(t) >= min_len]
    if remove_stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks


def build_doc_level_counts_from_docs(
    docs: Sequence[Doc],
    tokenizer: Callable[[str], List[str]],
    *,
    min_doc_count: int = 5,
    require_main_abuse: bool = True,
) -> pd.DataFrame:
    """doc-level: 한 아동 문서에서 단어가 1회 이상 등장하면 1 카운트."""
    abuse2idx = {a: i for i, a in enumerate(ABUSE_ORDER)}
    K = len(ABUSE_ORDER)

    acc: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(K, dtype=int))

    for d in docs:
        if require_main_abuse and (d.main_abuse not in ABUSE_ORDER):
            continue
        if not d.raw_text.strip():
            continue

        toks = set(tokenizer(d.raw_text))
        if not toks:
            continue

        idx = abuse2idx.get(d.main_abuse)
        if idx is None:
            continue

        for w in toks:
            acc[w][idx] += 1

    if not acc:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(acc, orient="index", columns=ABUSE_ORDER)
    df.index.name = "word"

    df["total_docs"] = df.sum(axis=1)
    df = df[df["total_docs"] >= int(min_doc_count)].drop(columns=["total_docs"])

    for a in ABUSE_ORDER:
        if a not in df.columns:
            df[a] = 0
    return df[ABUSE_ORDER]


def run_ca_numpy(
    df_counts: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ABUSE_ORDER,
    top_chi_for_ca: int = 200,
    n_components: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Minimal CA (numpy SVD) returning row coords and meta."""
    if df_counts is None or df_counts.empty:
        return pd.DataFrame(), {}

    chi_df = compute_chi_square(df_counts, list(group_cols))
    if chi_df is None or chi_df.empty:
        return pd.DataFrame(), {}

    chi_sorted = chi_df.sort_values("chi2", ascending=False, kind="mergesort")
    words = chi_sorted.head(min(top_chi_for_ca, len(chi_sorted))).index
    df_ca = df_counts.loc[df_counts.index.intersection(words), list(group_cols)]

    X = df_ca.T.astype(float).values  # rows=groups, cols=words
    if X.size == 0 or X.sum() <= 0:
        return pd.DataFrame(), {}

    N = float(X.sum())
    P = X / N
    r = P.sum(axis=1, keepdims=True)  # (K,1)
    c = P.sum(axis=0, keepdims=True)  # (1,M)

    with np.errstate(divide="ignore", invalid="ignore"):
        Dr_inv_sqrt = np.diagflat(1.0 / np.sqrt(np.maximum(r, 1e-12)))
        Dc_inv_sqrt = np.diagflat(1.0 / np.sqrt(np.maximum(c, 1e-12)))

    S = Dr_inv_sqrt @ (P - r @ c) @ Dc_inv_sqrt
    U, s, _ = np.linalg.svd(S, full_matrices=False)
    eig = s ** 2
    total_inertia = float(eig.sum())

    Sigma = np.diag(s)
    F = Dr_inv_sqrt @ U @ Sigma
    F = F[:, :n_components]

    row_coords = pd.DataFrame(F, index=list(group_cols), columns=[f"Dim{i+1}" for i in range(F.shape[1])])

    explained = (eig / total_inertia) if total_inertia > 0 else np.full_like(eig, np.nan)

    meta = {
        "n_words_in_ca": int(df_ca.shape[0]),
        "total_inertia": total_inertia,
        "eig": eig.tolist(),
        "explained": explained.tolist(),
        "top_words": list(df_ca.index),
        "chi2_top": chi_sorted.head(min(top_chi_for_ca, len(chi_sorted))).reset_index().rename(columns={"index": "word"}),
    }

    return row_coords, meta


def procrustes_r2(A: np.ndarray, B: np.ndarray) -> Tuple[float, float, float]:
    """Align B to A with orthogonal Procrustes (translation+scale+rotation)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)

    nA = np.linalg.norm(A0)
    nB = np.linalg.norm(B0)
    if nA <= 0 or nB <= 0:
        return np.nan, np.nan, np.nan

    A0 = A0 / nA
    B0 = B0 / nB

    M = A0.T @ B0
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    B_aligned = B0 @ R.T

    diff = A0 - B_aligned
    ss_diff = float(np.sum(diff ** 2))
    ss_ref = float(np.sum(A0 ** 2))
    r2 = 1.0 - (ss_diff / ss_ref) if ss_ref > 0 else np.nan
    rmsd = float(np.sqrt(np.mean(diff ** 2)))
    return float(r2), float(rmsd), float(ss_diff)


def run_preprocess_sensitivity(
    docs: Sequence[Doc],
    *,
    out_dir: Path,
    min_doc_count: int = 5,
    top_chi_for_ca: int = 200,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: List[Dict[str, Any]] = [
        {
            "name": "baseline_project_tokenize_korean",
            "desc": "project tokenize_korean() (N/V/Adj, len>=2)",
            "tokenizer": lambda s: tokenize_korean(s),
        },
        {
            "name": "no_1char_filter",
            "desc": "Okt N/V/Adj, len>=1 (1글자 제거 OFF)",
            "tokenizer": lambda s: tokenize_custom_okt(s, pos_tags=["Noun", "Verb", "Adjective"], min_len=1),
        },
        {
            "name": "nouns_only",
            "desc": "Okt Noun only, len>=2",
            "tokenizer": lambda s: tokenize_custom_okt(s, pos_tags=["Noun"], min_len=2),
        },
        {
            "name": "include_adverbs",
            "desc": "Okt N/V/Adj/Adv, len>=2",
            "tokenizer": lambda s: tokenize_custom_okt(s, pos_tags=["Noun", "Verb", "Adjective", "Adverb"], min_len=2),
        },
        {
            "name": "whitespace",
            "desc": "simple whitespace tokenizer, len>=2",
            "tokenizer": lambda s: tokenize_whitespace(s, min_len=2),
        },
    ]

    baseline_name = variants[0]["name"]

    row_coords_map: Dict[str, pd.DataFrame] = {}
    topword_map: Dict[str, Set[str]] = {}
    meta_rows = []

    for v in variants:
        name = v["name"]
        tok = v["tokenizer"]

        df_counts = build_doc_level_counts_from_docs(
            docs,
            tokenizer=tok,
            min_doc_count=min_doc_count,
            require_main_abuse=True,
        )

        meta_row: Dict[str, Any] = {
            "variant": name,
            "desc": v.get("desc", ""),
            "n_words_total": int(df_counts.shape[0]) if df_counts is not None else 0,
        }

        row_coords, meta = run_ca_numpy(
            df_counts,
            group_cols=ABUSE_ORDER,
            top_chi_for_ca=top_chi_for_ca,
        )

        if row_coords is None or row_coords.empty:
            meta_row.update({"ca_ok": 0})
            meta_rows.append(meta_row)
            continue

        meta_row.update({
            "ca_ok": 1,
            "n_words_in_ca": meta.get("n_words_in_ca", np.nan),
            "total_inertia": meta.get("total_inertia", np.nan),
            "dim1_explained_pct": (meta.get("explained", [np.nan, np.nan])[0] * 100) if meta.get("explained") else np.nan,
            "dim2_explained_pct": (meta.get("explained", [np.nan, np.nan])[1] * 100) if meta.get("explained") and len(meta.get("explained")) > 1 else np.nan,
        })

        row_coords.to_csv(out_dir / f"ca_row_coords__{name}.csv", encoding="utf-8-sig")

        chi2_top_df = meta.get("chi2_top")
        if isinstance(chi2_top_df, pd.DataFrame) and not chi2_top_df.empty:
            chi2_top_df.to_csv(out_dir / f"chi2_top__{name}.csv", index=False, encoding="utf-8-sig")
            topword_map[name] = set(chi2_top_df["word"].astype(str).tolist())
        else:
            topword_map[name] = set()

        row_coords_map[name] = row_coords
        meta_rows.append(meta_row)

    if baseline_name not in row_coords_map:
        print("[SENS] baseline CA failed -> cannot compute Procrustes")
        pd.DataFrame(meta_rows).to_csv(out_dir / "preprocess_sensitivity_meta.csv", index=False, encoding="utf-8-sig")
        return

    base = row_coords_map[baseline_name].reindex(ABUSE_ORDER)
    A = base[["Dim1", "Dim2"]].values

    sum_rows = []
    for name, row_coords in row_coords_map.items():
        B = row_coords.reindex(ABUSE_ORDER)[["Dim1", "Dim2"]].values
        r2, rmsd, ss = procrustes_r2(A, B)

        w0 = topword_map.get(baseline_name, set())
        w1 = topword_map.get(name, set())
        jacc = (len(w0 & w1) / len(w0 | w1)) if (w0 | w1) else np.nan

        sum_rows.append({
            "variant": name,
            "procrustes_r2_vs_baseline": r2,
            "procrustes_rmsd_vs_baseline": rmsd,
            "ss_diff": ss,
            "jaccard_topwords_vs_baseline": jacc,
        })

    df_meta = pd.DataFrame(meta_rows)
    df_sum = pd.DataFrame(sum_rows)
    df = df_meta.merge(df_sum, on="variant", how="left")
    df.to_csv(out_dir / "preprocess_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    print(f"[SENS] saved -> {out_dir}")


# ---------------------------------------------------------------------
# 7) GT exclusion bias + best/worst-case bounds
# ---------------------------------------------------------------------

def run_gt_exclusion_and_sensitivity(
    docs: Sequence[Doc],
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in docs:
        gt_raw_exists = bool((d.gt_raw_text or "").strip())
        if len(d.gt_mapped) > 0:
            gt_status = "mapped"
        else:
            gt_status = "missing" if not gt_raw_exists else "unmappable"

        algo_set = set()
        if d.main_abuse:
            algo_set.add(d.main_abuse)
        algo_set |= set(d.sub_abuses or [])

        main_hit = int((d.main_abuse in d.gt_mapped) if (d.main_abuse and d.gt_mapped) else 0)
        exact_set_match = int((algo_set == d.gt_mapped) if (gt_status == "mapped") else 0)

        if gt_status == "mapped":
            union = algo_set | d.gt_mapped
            inter = algo_set & d.gt_mapped
            jacc = (len(inter) / len(union)) if union else np.nan
        else:
            jacc = np.nan

        rows.append({
            "doc_id": d.doc_id,
            "valence": d.valence,
            "crisis": d.crisis,
            "total_score": d.total_score,
            "main_abuse": d.main_abuse,
            "n_sub": len(d.sub_abuses or []),
            "a_sum": float(sum(d.a_scores.get(a, 0) for a in ABUSE_ORDER)),
            "a_entropy": d.a_entropy if d.a_entropy is not None else np.nan,
            "a_gap12": d.a_gap12 if d.a_gap12 is not None else np.nan,
            "gt_status": gt_status,
            "gt_mapped": "|".join(sorted(d.gt_mapped)) if d.gt_mapped else "",
            "gt_unmapped": "|".join(sorted(d.gt_unmapped)) if d.gt_unmapped else "",
            "main_hit": main_hit if gt_status == "mapped" else np.nan,
            "exact_set_match": exact_set_match if gt_status == "mapped" else np.nan,
            "jaccard": jacc,
            "source_file": d.source_file,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gt_exclusion_cases.csv", index=False, encoding="utf-8-sig")

    dist_rows = []
    for status, sub in df.groupby("gt_status"):
        dist_rows.append({
            "gt_status": status,
            "n": len(sub),
            "pct": len(sub) / len(df) * 100 if len(df) else np.nan,
            "mean_total_score": sub["total_score"].dropna().mean(),
            "mean_a_entropy": sub["a_entropy"].dropna().mean(),
            "mean_a_gap12": sub["a_gap12"].dropna().mean(),
        })

    df_dist = pd.DataFrame(dist_rows).sort_values("n", ascending=False)
    df_dist.to_csv(out_dir / "gt_exclusion_overview.csv", index=False, encoding="utf-8-sig")

    pd.crosstab(df["gt_status"], df["main_abuse"], dropna=False).to_csv(out_dir / "gt_status_x_mainabuse.csv", encoding="utf-8-sig")
    pd.crosstab(df["gt_status"], df["valence"], dropna=False).to_csv(out_dir / "gt_status_x_valence.csv", encoding="utf-8-sig")

    n_total = len(df)
    mapped = df[df["gt_status"] == "mapped"].copy()
    n_mapped = len(mapped)
    n_excl = n_total - n_mapped

    def _bounds(metric_col: str) -> Tuple[float, float, float]:
        if n_mapped <= 0:
            return np.nan, np.nan, np.nan
        obs = float(mapped[metric_col].mean())
        lower = (obs * n_mapped + 0.0 * n_excl) / n_total if n_total else np.nan
        upper = (obs * n_mapped + 1.0 * n_excl) / n_total if n_total else np.nan
        return obs, lower, upper

    out_bounds = []
    for m in ["main_hit", "exact_set_match"]:
        obs, lower, upper = _bounds(m)
        out_bounds.append({
            "metric": m,
            "n_total": n_total,
            "n_mapped": n_mapped,
            "n_excluded": n_excl,
            "observed_mean_on_mapped": obs,
            "overall_lower_bound": lower,
            "overall_upper_bound": upper,
        })

    pd.DataFrame(out_bounds).to_csv(out_dir / "gt_metric_sensitivity_bounds.csv", index=False, encoding="utf-8-sig")
    print(f"[GT] saved -> {out_dir}")


# ---------------------------------------------------------------------
# 8) CLI
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="data directory containing *.json")
    ap.add_argument("--out_dir", type=str, default="revision_appendix_out", help="output directory")
    ap.add_argument("--only_negative", action="store_true", help="filter valence=='부정'")
    ap.add_argument("--gt_field", type=str, default="학대의심", help="GT field name inside info")

    ap.add_argument("--run_softlabel", action="store_true", help="run A-score soft-label auxiliary analysis")
    ap.add_argument("--soft_k_folds", type=int, default=5)

    ap.add_argument("--run_sensitivity", action="store_true", help="run preprocessing sensitivity (CA invariance)")
    ap.add_argument("--min_doc_count", type=int, default=5)
    ap.add_argument("--top_chi_for_ca", type=int, default=200)

    ap.add_argument("--run_gt", action="store_true", help="run GT exclusion bias + sensitivity bounds")

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args.data_dir)
    json_files = find_json_files(data_dir)

    if not json_files:
        print("[ERROR] no json files found")
        print("  - provided data_dir:", args.data_dir)
        print("  - resolved data_dir:", data_dir)
        print("  - tip: try absolute path, e.g., --data_dir C:/.../Childeren/data")
        raise SystemExit(1)

    print("=" * 72)
    print("[REVISION APPENDIX RUNNER]")
    print("- data_dir:", data_dir)
    print("- n_json:", len(json_files))
    print("- out_dir:", out_dir)
    print("- only_negative:", args.only_negative)
    print("=" * 72)

    docs = build_docs(
        json_files,
        only_negative=args.only_negative,
        gt_field=args.gt_field,
    )
    print(f"[LOAD] docs = {len(docs)}")

    # default: run all if none specified
    if not (args.run_softlabel or args.run_sensitivity or args.run_gt):
        args.run_softlabel = True
        args.run_sensitivity = True
        args.run_gt = True

    if args.run_softlabel:
        run_softlabel_aux_analysis(
            docs,
            out_dir=out_dir / "softlabel_aux",
            k_folds=args.soft_k_folds,
        )

    if args.run_sensitivity:
        run_preprocess_sensitivity(
            docs,
            out_dir=out_dir / "preprocess_sensitivity",
            min_doc_count=args.min_doc_count,
            top_chi_for_ca=args.top_chi_for_ca,
        )

    if args.run_gt:
        run_gt_exclusion_and_sensitivity(
            docs,
            out_dir=out_dir / "gt_exclusion",
        )

    print("\n[DONE] outputs at:", out_dir)


if __name__ == "__main__":
    main()