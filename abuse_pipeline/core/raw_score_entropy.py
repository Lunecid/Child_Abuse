"""raw_score_entropy.py
════════════════════════════════════════════════════════════════
∆H(정보 손실)의 입력이 알고리즘 2의 산출물(main, subs)이 **아니라**
정규화된 원점수 A_k 임을 코드 구조에서 입증하는 모듈.

Shannon 엔트로피를 임상가 4영역 점수에서 직접 계산하며,
core.labels의 어떠한 심볼도 import하지 않는다.

Import 금지 사항
────────────────
  core.labels 의 어떠한 심볼(classify_abuse_main_sub, classify_child_group,
  SEVERITY_RANK 등)도 이 모듈에 들어와서는 **안 된다**.

제공 함수
─────────
  normalized_score_distribution : 4영역 점수 → 합 1 정규화
  shannon_entropy_from_raw      : 정규화 분포의 Shannon 엔트로피
  delta_h_dataframe             : 전체 사례에 대한 ∆H DataFrame
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 영역 컬럼명 (raw_score_distribution과 동일)
_DOMAIN_COLS = ["A_neglect", "A_emotional", "A_physical", "A_sexual"]


def normalized_score_distribution(a_scores: np.ndarray) -> np.ndarray:
    """한 사례의 4영역 점수 벡터를 합이 1이 되도록 정규화한다.

    Parameters
    ----------
    a_scores : np.ndarray, shape (4,)
        (A_방임, A_정서, A_신체, A_성) 원점수.

    Returns
    -------
    np.ndarray, shape (4,)
        정규화된 분포.  합이 0인 경우(학대 증거 전무) NaN 배열 반환.
    """
    a = np.asarray(a_scores, dtype=float)
    s = a.sum()
    if s <= 0:
        return np.full_like(a, np.nan)
    return a / s


def shannon_entropy_from_raw(
    a_scores: np.ndarray,
    base: int = 2,
) -> float:
    """정규화된 원점수 분포의 Shannon 엔트로피를 직접 계산한다.

    이 함수는 어떠한 라벨 인자도 받지 않는다.
    입력은 오직 4영역 점수뿐이다.

    Parameters
    ----------
    a_scores : np.ndarray, shape (4,)
        원점수 벡터 (정규화 전).
    base : int
        로그 밑.  2 = bits, e = nats.

    Returns
    -------
    float
        Shannon 엔트로피.  합이 0인 경우 NaN.
    """
    p = normalized_score_distribution(a_scores)
    if np.any(np.isnan(p)):
        return float("nan")

    # 0 확률 원소는 0 * log(0) = 0 으로 처리
    eps = 1e-15
    p_safe = np.where(p > 0, p, eps)

    if base == 2:
        return float(-np.sum(p * np.log2(p_safe)))
    else:
        return float(-np.sum(p * np.log(p_safe))) / np.log(base)


def delta_h_dataframe(scores_df: pd.DataFrame) -> pd.DataFrame:
    """모든 사례에 대해 H_multi, H_single (= 0), delta_h를 계산한다.

    H_multi  : 4영역 정규화 분포의 Shannon 엔트로피 (bits)
    H_single : 단일 라벨의 엔트로피 = 0 (one-hot 분포)
    delta_h  : H_multi - H_single = H_multi

    max_share : 정규화 분포의 최댓값 (단일 라벨이 포착하는 정보의 비중)
    n_active_domains : A_k > 0 인 영역의 수

    Parameters
    ----------
    scores_df : pd.DataFrame
        extract_raw_abuse_scores()의 반환값. _DOMAIN_COLS 컬럼 필수.

    Returns
    -------
    pd.DataFrame
        컬럼: case_id, H_multi, delta_h, max_share, n_active_domains
    """
    rows = []
    for _, row in scores_df.iterrows():
        a_vec = np.array([row[c] for c in _DOMAIN_COLS], dtype=float)
        h = shannon_entropy_from_raw(a_vec, base=2)

        p = normalized_score_distribution(a_vec)
        max_share = float(np.nanmax(p)) if not np.all(np.isnan(p)) else float("nan")
        n_active = int((a_vec > 0).sum())

        rows.append({
            "case_id": row["case_id"],
            "H_multi": h,
            "delta_h": h,  # H_single = 0
            "max_share": max_share,
            "n_active_domains": n_active,
        })

    return pd.DataFrame(rows)
