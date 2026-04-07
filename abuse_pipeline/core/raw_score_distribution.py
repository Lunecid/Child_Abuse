"""raw_score_distribution.py
════════════════════════════════════════════════════════════════
알고리즘 2(classify_abuse_main_sub)와 **독립적으로** 임상가 점수 분포를
분석하는 모듈.  이름에 *raw*를 넣어 "표기 규약 이전의 원점수"임을 코드
수준에서 명시한다.

Import 금지 사항
────────────────
  core.labels 의 어떠한 심볼(classify_abuse_main_sub, classify_child_group,
  SEVERITY_RANK 등)도 이 모듈에 들어와서는 **안 된다**.
  이 제약은 "원점수 분석이 라벨링 알고리즘에 의존하지 않는다"는
  코드-수준 보증을 제공한다.

제공 함수
─────────
  extract_raw_abuse_scores   : JSON → 4영역 점수 DataFrame
  count_nonzero_domains      : 활성 영역 수 계산
  distribution_by_threshold  : 임계값별 활성 영역 수 분포
  pairwise_cooccurrence      : 4×4 동시 출현 행렬
  gt_consistency_check       : GT 보유 사례의 다영역 활성 점검
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

# ── 허용된 import: core.common에서 상수와 경로만 사용 ──────────
from abuse_pipeline.core.common import ABUSE_ORDER, DATA_JSON_DIR

# 영역 컬럼명 (DataFrame 내부 표현)
_DOMAIN_COLS = ["A_neglect", "A_emotional", "A_physical", "A_sexual"]

# ABUSE_ORDER(성학대, 신체학대, 정서학대, 방임) ↔ 컬럼 매핑
_ABUSE_TO_COL: Dict[str, str] = {
    "방임":   "A_neglect",
    "정서학대": "A_emotional",
    "신체학대": "A_physical",
    "성학대":  "A_sexual",
}
_COL_TO_ABUSE: Dict[str, str] = {v: k for k, v in _ABUSE_TO_COL.items()}

# GT 표기 변형 정규화 맵 (labels.py를 import하지 않고 로컬 복제)
_GT_CANON_MAP = {
    "신체적학대": "신체학대",
    "정서적학대": "정서학대",
    "성적학대":   "성학대",
    "성폭력":     "성학대",
    "성폭행":     "성학대",
    "유기":       "방임",
}


# ════════════════════════════════════════════════════════════════
#  내부 헬퍼
# ════════════════════════════════════════════════════════════════

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


def _normalize_gt_label(label: str) -> str:
    s = re.sub(r"\s+", "", str(label))
    s = _GT_CANON_MAP.get(s, s)
    if s.endswith("적학대"):
        s = s.replace("적학대", "학대")
    return _GT_CANON_MAP.get(s, s)


def _extract_gt_labels(info: Dict[str, Any]) -> Set[str]:
    """info['학대의심'] 필드에서 GT 학대유형 집합을 추출한다."""
    raw = info.get("학대의심", "")
    text = _to_text(raw)
    text_nospace = re.sub(r"\s+", "", text)

    if not text_nospace:
        return set()

    found: Set[str] = set()
    for m in re.findall(r"([가-힣]+?)학대", text_nospace):
        cand = _normalize_gt_label(m + "학대")
        if cand in set(ABUSE_ORDER):
            found.add(cand)
    if "방임" in text_nospace:
        found.add("방임")
    if "유기" in text_nospace:
        found.add("방임")
    for a in ABUSE_ORDER:
        if re.sub(r"\s+", "", a) in text_nospace:
            found.add(a)
    return {a for a in found if a in set(ABUSE_ORDER)}


def _extract_abuse_scores_from_rec(
    rec: Dict[str, Any],
    abuse_order: Sequence[str] = ABUSE_ORDER,
) -> Dict[str, int]:
    """한 JSON record에서 학대여부 문항의 4영역 합산점수를 추출한다."""
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


def _read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(json_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(json_obj, dict) and ("info" in json_obj or "list" in json_obj):
        return [json_obj]
    if isinstance(json_obj, list):
        return [x for x in json_obj if isinstance(x, dict)]
    return []


def _find_json_files(data_dir: str | Path) -> List[Path]:
    d = Path(data_dir)
    if not d.exists() or not d.is_dir():
        return []
    return sorted(d.rglob("*.json"))


# ════════════════════════════════════════════════════════════════
#  공개 API
# ════════════════════════════════════════════════════════════════

def extract_raw_abuse_scores(
    json_files: Sequence[str | Path] | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """원자료에서 각 사례의 4영역 점수를 추출한다.

    Parameters
    ----------
    json_files : list of paths, optional
        JSON 파일 경로 리스트.  None이면 *data_dir*에서 자동 탐색.
    data_dir : str or Path, optional
        JSON 디렉토리.  None이면 ``DATA_JSON_DIR`` 사용.

    Returns
    -------
    pd.DataFrame
        컬럼: case_id, A_neglect, A_emotional, A_physical, A_sexual,
              gt_label (str | None), corpus_membership (set)
    """
    if json_files is None:
        data_dir = data_dir or DATA_JSON_DIR
        json_files = _find_json_files(data_dir)

    rows: List[Dict[str, Any]] = []

    for fp in [Path(f) for f in json_files]:
        try:
            obj = _read_json_any(fp)
        except Exception:
            continue

        for ridx, rec in enumerate(_iter_records(obj)):
            info = rec.get("info", {}) or {}
            case_id = info.get("ID") or info.get("id") or info.get("Id")
            if not case_id:
                case_id = f"{fp.stem}__{ridx}"

            a_scores = _extract_abuse_scores_from_rec(rec)

            # GT label (단일: SEVERITY_RANK 없이 첫 번째만 사용)
            gt_set = _extract_gt_labels(info)
            gt_label: Optional[str] = None
            if gt_set:
                # 여러 GT가 있으면 ABUSE_ORDER 기준 가장 앞(= 가장 심각)을 선택
                for a in ABUSE_ORDER:
                    if a in gt_set:
                        gt_label = a
                        break

            # 코퍼스 멤버십 판정 (라벨 알고리즘 비의존)
            membership: Set[str] = {"ALL"}
            has_any_score = any(v > 0 for v in a_scores.values())

            # 위기단계 기반 간이 부정 판정 (labels.py 비의존)
            crisis = info.get("위기단계")
            negative_crisis = {"응급", "위기아동", "학대의심", "상담필요"}
            try:
                total_score = int(info.get("합계점수"))
            except (TypeError, ValueError):
                total_score = 0
            is_neg = (crisis in negative_crisis) or (total_score >= 45)
            if is_neg:
                membership.add("NEG")
                if has_any_score:
                    membership.add("ABUSE_NEG")
                # GT는 NEG 내에서만 인정: 부정군이 아닌 사례의 GT 라벨은
                # 본 분석의 대상이 아니므로 멤버십에서 제외한다.
                if gt_label is not None:
                    membership.add("GT")

            # NEG 밖의 GT 라벨은 gt_label 컬럼에도 반영하지 않음
            effective_gt = gt_label if is_neg else None

            rows.append({
                "case_id": str(case_id),
                "A_neglect": a_scores.get("방임", 0),
                "A_emotional": a_scores.get("정서학대", 0),
                "A_physical": a_scores.get("신체학대", 0),
                "A_sexual": a_scores.get("성학대", 0),
                "gt_label": effective_gt,
                "corpus_membership": membership,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "case_id", *_DOMAIN_COLS, "gt_label", "corpus_membership",
        ])
    return df


def filter_by_corpus(
    scores_df: pd.DataFrame,
    subset: str = "ALL",
) -> pd.DataFrame:
    """corpus_membership 집합에 *subset* 태그가 포함된 행만 필터링한다."""
    mask = scores_df["corpus_membership"].apply(lambda s: subset in s)
    return scores_df.loc[mask].reset_index(drop=True)


def count_nonzero_domains(
    scores_df: pd.DataFrame,
    threshold: int = 0,
) -> pd.DataFrame:
    """각 사례에서 A_k > threshold (threshold=0) 또는 A_k >= threshold
    (threshold>0) 를 만족하는 영역의 수를 계산한다.

    Parameters
    ----------
    scores_df : pd.DataFrame
        extract_raw_abuse_scores()의 반환값.
    threshold : int
        0이면 "비영 점수의 영역 수" (A_k > 0),
        양수이면 "A_k >= threshold인 영역 수".

    Returns
    -------
    pd.DataFrame
        컬럼: case_id, n_domains_active
    """
    mat = scores_df[_DOMAIN_COLS].values
    if threshold <= 0:
        active = (mat > 0).sum(axis=1)
    else:
        active = (mat >= threshold).sum(axis=1)
    return pd.DataFrame({
        "case_id": scores_df["case_id"],
        "n_domains_active": active,
    })


def distribution_by_threshold(
    scores_df: pd.DataFrame,
    thresholds: List[int] | None = None,
    subset: str = "GT",
) -> pd.DataFrame:
    """임계값별로 활성 영역 수의 분포를 산출한다.

    Returns
    -------
    pd.DataFrame  (long-form)
        컬럼: threshold, n_domains, n_cases, proportion, subset
    """
    if thresholds is None:
        thresholds = [1, 2, 4, 6]

    rows: List[Dict[str, Any]] = []
    n_total = len(scores_df)

    for tau in thresholds:
        cnt_df = count_nonzero_domains(scores_df, threshold=tau)
        for nd in range(5):  # 0, 1, 2, 3, 4
            n_cases = int((cnt_df["n_domains_active"] == nd).sum())
            rows.append({
                "threshold": tau,
                "n_domains": nd,
                "n_cases": n_cases,
                "proportion": n_cases / n_total if n_total > 0 else 0.0,
                "subset": subset,
            })

    return pd.DataFrame(rows)


def pairwise_cooccurrence(
    scores_df: pd.DataFrame,
    threshold: int = 4,
    subset: str = "GT",
    normalize: str = "none",
) -> pd.DataFrame:
    """두 영역 간 동시 활성화를 4×4 행렬로 산출한다.

    대각 원소: 해당 영역 단독 활성 사례 수  (다른 영역은 비활성)
    비대각 원소: 두 영역 동시 활성 사례 수

    Parameters
    ----------
    normalize : {'none', 'row', 'all'}
        'none': 원래 사례 수, 'row': 행 합 기준, 'all': 전체 합 기준.
    """
    mat = scores_df[_DOMAIN_COLS].values
    active = mat >= threshold  # (N, 4) bool

    n_domains = len(_DOMAIN_COLS)
    cooc = np.zeros((n_domains, n_domains), dtype=float)

    for i in range(n_domains):
        for j in range(n_domains):
            if i == j:
                # 단독 활성: i가 활성이고 나머지 모두 비활성
                solo = active[:, i] & (~np.any(
                    np.delete(active, i, axis=1), axis=1
                ))
                cooc[i, j] = float(solo.sum())
            else:
                # 동시 활성: i와 j 모두 활성
                cooc[i, j] = float((active[:, i] & active[:, j]).sum())

    domain_labels = [_COL_TO_ABUSE[c] for c in _DOMAIN_COLS]

    if normalize == "row":
        row_sums = cooc.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cooc = cooc / row_sums
    elif normalize == "all":
        total = cooc.sum()
        if total > 0:
            cooc = cooc / total

    df = pd.DataFrame(cooc, index=domain_labels, columns=domain_labels)
    df.index.name = "domain"
    return df


def gt_consistency_check(
    scores_df: pd.DataFrame,
    threshold: int = 4,
) -> dict:
    """GT 보유 사례에서 다영역 활성 패턴을 점검한다.

    "임상가가 GT로 지정한 영역의 A_k"와 "GT 외 영역에서 A_k >= threshold인
    영역의 수와 평균 점수"를 보고한다.

    Returns
    -------
    dict
        n_gt_cases, mean_a_at_gt_domain, n_with_other_active_domains,
        mean_n_other_active, pct_with_at_least_one_other
    """
    gt_df = scores_df[scores_df["gt_label"].notna()].copy()
    n_gt = len(gt_df)

    if n_gt == 0:
        return {
            "n_gt_cases": 0,
            "mean_a_at_gt_domain": float("nan"),
            "n_with_other_active_domains": 0,
            "mean_n_other_active": float("nan"),
            "pct_with_at_least_one_other": 0.0,
        }

    a_at_gt = []
    n_other_active_list = []

    for _, row in gt_df.iterrows():
        gt_label = row["gt_label"]
        gt_col = _ABUSE_TO_COL.get(gt_label)
        if gt_col is None:
            continue

        a_at_gt.append(float(row[gt_col]))

        # GT 외 영역에서 threshold 이상인 수
        n_other = 0
        for col in _DOMAIN_COLS:
            if col == gt_col:
                continue
            if row[col] >= threshold:
                n_other += 1
        n_other_active_list.append(n_other)

    a_at_gt = np.array(a_at_gt)
    n_other_active = np.array(n_other_active_list)
    n_with_other = int((n_other_active >= 1).sum())

    return {
        "n_gt_cases": n_gt,
        "mean_a_at_gt_domain": float(np.mean(a_at_gt)) if len(a_at_gt) > 0 else float("nan"),
        "n_with_other_active_domains": n_with_other,
        "mean_n_other_active": float(np.mean(n_other_active)) if len(n_other_active) > 0 else float("nan"),
        "pct_with_at_least_one_other": n_with_other / n_gt * 100 if n_gt > 0 else 0.0,
    }
