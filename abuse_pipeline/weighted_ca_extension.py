#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weighted_ca_extension.py
========================
가중 빈도표 기반 CA(대응분석) 확장 모듈

목적
----
기존 CA는 아동의 **주(Main) 학대유형**에만 토큰을 배정하여 빈도표를 구축했다.
이 모듈은 **부(Sub) 학대유형에도 가중치(α)를 부여**하여 빈도표를 구축한 뒤,
동일한 CA를 수행함으로써:
  1) 부 학대유형 정보가 CA 공간 구조에 미치는 영향을 정량화하고,
  2) α 민감도 분석을 통해 결과의 강건성을 검증한다.

핵심 수식
---------
가중 빈도표 F^(α)의 (w, A_k) 셀:

    F^(α)_{w, A_k} = Σ_i  δ(w ∈ T_i) · w(A_k, i)

    여기서:
        w(A_k, i) = 1.0    if A_k = M_i   (주 학대유형)
                  = α      if A_k ∈ S_i   (부 학대유형, 0 < α ≤ 1)
                  = 0      otherwise

α 특수값:
    α = 0.0 → 기존 CA와 동일 (주 학대유형만 반영)
    α = 1.0 → 주와 부를 동등하게 반영

    try:
        from .label_comparison_analysis import run_label_comparison_from_pipeline

        label_comp_out = os.path.join(C.OUTPUT_DIR, "label_comparison")
        os.makedirs(label_comp_out, exist_ok=True)

        label_comp_results = run_label_comparison_from_pipeline(
            json_files=json_files,
            out_dir=label_comp_out,
            abuse_order=C.ABUSE_ORDER,
            sub_threshold=2,
            use_clinical_text=True,
            only_negative=only_negative,
            alpha_values=[0.0, 0.3, 0.5, 0.7, 1.0],
        )
    except ImportError as e:
        print(f"[LABEL-COMP] import 실패 → 라벨 비교 분석을 건너뜁니다: {e}")
    except Exception as e:
        print(f"[LABEL-COMP] 실행 실패: {e}")
        import traceback; traceback.print_exc()
"""

from __future__ import annotations

import os
import json
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── 패키지 내부 import ───
from . import common as C
from .labels import classify_abuse_main_sub
from .text import extract_child_speech, tokenize_korean

# ─── 상수 (common.py에서 가져옴) ───
ABUSE_ORDER = C.ABUSE_ORDER
ABUSE_LABEL_EN = C.ABUSE_LABEL_EN
ABUSE_COLORS = C.ABUSE_COLORS

# ─── 선택적 의존성 ───
try:
    from scipy.spatial import procrustes as scipy_procrustes
    from scipy.stats import chi2 as scipy_chi2
    from scipy.linalg import svd as scipy_svd
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
#  0. 파이프라인 데이터 수집 & 가중 빈도표 구축
# ═══════════════════════════════════════════════════════════════

def collect_child_records(
    json_files: List[str],
    allowed_groups=None,
    only_negative: bool = False,
) -> List[Dict]:
    """
    JSON 파일들에서 아동별 main_abuse, sub_abuses, speech_tokens를 수집한다.

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    allowed_groups : set or None
        허용된 정서군 (예: {"부정"})
    only_negative : bool
        True이면 부정 정서군만 포함

    Returns
    -------
    list[dict] : 각 아동의 {child_id, main_abuse, sub_abuses, speech_tokens}
    """
    from .labels import classify_child_group

    records = []
    for path in json_files:
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        info = rec.get("info", {})
        child_id = info.get("ID")

        group = classify_child_group(rec)
        if group not in C.VALENCE_ORDER:
            continue
        if only_negative and group != "부정":
            continue
        if allowed_groups and group not in allowed_groups:
            continue

        main_abuse, sub_abuses = classify_abuse_main_sub(rec)
        if main_abuse not in C.ABUSE_ORDER:
            continue

        speech_list = extract_child_speech(rec)
        if not speech_list:
            continue

        raw_text = " ".join(speech_list)
        tokens = tokenize_korean(raw_text)
        if not tokens:
            continue

        records.append({
            "child_id": child_id,
            "main_abuse": main_abuse,
            "sub_abuses": list(sub_abuses) if sub_abuses else [],
            "speech_tokens": tokens,
        })

    return records


def build_weighted_frequency_table(
    records: List[Dict],
    alpha: float = 0.5,
    min_total_count: int = 8,
    abuse_order: List[str] = None,
) -> pd.DataFrame:
    """
    가중 빈도표를 구축한다.

    ─── 가중 빈도표 F^(α) ───

    각 아동 i에 대해, 단어 w가 발화에 존재하면(δ(w ∈ T_i) = 1):
      - 주 학대유형 M_i 열에 1.0을 더한다.
      - 부 학대유형 S_i의 각 열에 α를 더한다.

    수식:
        F^(α)_{w, A_k} = Σ_i  δ(w ∈ T_i) · w(A_k, i)

        w(A_k, i) = 1.0  if A_k = M_i
                  = α    if A_k ∈ S_i
                  = 0    otherwise

    Parameters
    ----------
    records : list[dict]
        collect_child_records()의 출력
    alpha : float
        부 학대유형 가중치 (0 ≤ α ≤ 1)
    min_total_count : int
        총 가중 빈도가 이 값 미만인 단어는 제외

    Returns
    -------
    pd.DataFrame : index=word, columns=abuse_order, values=가중 빈도
    """
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    word_counts = defaultdict(lambda: {a: 0.0 for a in abuse_order})

    for r in records:
        main = r["main_abuse"]
        subs = set(r.get("sub_abuses", [])) & set(abuse_order)
        # document-level presence: 중복 토큰 제거
        tokens = set(r["speech_tokens"])

        for w in tokens:
            # 주 학대유형에 1.0
            word_counts[w][main] += 1.0
            # 부 학대유형에 α
            for sub in subs:
                if sub != main:  # 주와 겹치면 중복 방지
                    word_counts[w][sub] += alpha

    rows = []
    for word, counts in word_counts.items():
        row = {"word": word}
        row.update(counts)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("word")
    df = df[[a for a in abuse_order if a in df.columns]]

    # 최소 빈도 필터링
    df["_total"] = df.sum(axis=1)
    df = df[df["_total"] >= min_total_count].drop(columns=["_total"])

    return df


# ═══════════════════════════════════════════════════════════════
#  1. 가중 CA 수행
# ═══════════════════════════════════════════════════════════════

def run_weighted_ca(
    df_weighted: pd.DataFrame,
    abuse_order: List[str] = None,
    n_components: int = 2,
    top_chi_words: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    가중 빈도표에 대해 CA(대응분석)를 수행한다.

    Parameters
    ----------
    df_weighted : pd.DataFrame
        index=word, columns=abuse_order, values=가중 빈도
    abuse_order : list[str]
        학대유형 순서
    n_components : int
        추출할 차원 수 (기본 2)
    top_chi_words : int
        CA에 사용할 카이제곱 상위 단어 수

    Returns
    -------
    dict with keys:
        'row_coords_2d', 'bary_df', 'eigenvalues', 'explained',
        'total_inertia', 'chi2_total', 'df_ca', 'p_value', etc.
    """
    if not HAS_SCIPY:
        print("[WARN] scipy 미설치 → 가중 CA를 건너뜁니다.")
        return None

    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    if df_weighted is None or df_weighted.empty:
        print("[WARN] 가중 빈도표가 비어 있어 CA를 건너뜁니다.")
        return None

    # 학대유형 열만 선택
    cols_present = [a for a in abuse_order if a in df_weighted.columns]
    if len(cols_present) < 2:
        print("[WARN] 유효한 학대유형 열이 2개 미만 → CA 건너뜀.")
        return None

    df_ca_input = df_weighted[cols_present].copy()

    # 행 합계로 유효 단어 필터
    df_ca_input["_row_total"] = df_ca_input.sum(axis=1)
    df_ca_input = df_ca_input[df_ca_input["_row_total"] > 0]

    # 카이제곱 기여도 기준 단어 선택
    if len(df_ca_input) > top_chi_words:
        obs = df_ca_input[cols_present].values
        row_sums = obs.sum(axis=1, keepdims=True)
        col_sums = obs.sum(axis=0, keepdims=True)
        N = obs.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = row_sums * col_sums / max(N, 1e-12)
            chi_contrib = np.nansum(
                np.where(expected > 0, (obs - expected) ** 2 / expected, 0), axis=1
            )
        df_ca_input["_chi_contrib"] = chi_contrib
        df_ca_input = df_ca_input.nlargest(top_chi_words, "_chi_contrib")

    df_ca_input = df_ca_input[cols_present]

    # ─── CA 수행: scipy SVD 직접 구현 ───
    X = df_ca_input.T.values.astype(float)  # (학대유형 × 단어)
    row_labels = list(df_ca_input.columns)  # 학대유형
    col_labels = list(df_ca_input.index)  # 단어
    n_rows, n_cols = X.shape

    N_total = X.sum()
    if N_total <= 0:
        print("[WARN] 빈도표 총합이 0 → CA 건너뜀.")
        return None

    # 대응행렬
    P = X / N_total
    r = P.sum(axis=1)  # 행 질량
    c = P.sum(axis=0)  # 열 질량

    r_safe = np.where(r > 0, r, 1e-12)
    c_safe = np.where(c > 0, c, 1e-12)

    # 표준화 잔차 행렬
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r_safe))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c_safe))
    expected_P = np.outer(r, c)
    residuals = P - expected_P
    S = Dr_inv_sqrt @ residuals @ Dc_inv_sqrt

    # SVD
    n_components_actual = min(n_components, n_rows - 1, n_cols - 1)
    U, sigma, Vt = scipy_svd(S, full_matrices=False)

    eigenvalues = sigma[:n_components_actual] ** 2
    total_inertia = float((sigma ** 2).sum())
    explained = eigenvalues / total_inertia if total_inertia > 0 else np.zeros(n_components_actual)

    # 행 좌표 (학대유형)
    F_coords = Dr_inv_sqrt @ U[:, :n_components_actual] * sigma[:n_components_actual]

    row_coords_2d = pd.DataFrame(
        F_coords[:, :n_components_actual],
        index=row_labels,
        columns=[f"Dim{i + 1}" for i in range(n_components_actual)],
    )

    # 전역 카이제곱
    obs_full = X.astype(float)
    rs_full = obs_full.sum(axis=1, keepdims=True)
    cs_full = obs_full.sum(axis=0, keepdims=True)
    exp_full = rs_full * cs_full / max(N_total, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_mat = np.where(exp_full > 0, (obs_full - exp_full) ** 2 / exp_full, 0)
    chi2_total = float(chi_mat.sum())
    df_val = int((n_rows - 1) * (n_cols - 1))

    p_value = np.nan
    try:
        p_value = float(scipy_chi2.sf(chi2_total, df=df_val))
    except Exception:
        pass

    # ─── 무중심좌표(Barycentric) 계산 ───
    bary_list = []
    for w in col_labels:
        counts = df_ca_input.loc[w, cols_present].values.astype(float)
        tot = float(counts.sum())
        if tot <= 0:
            bx, by = 0.0, 0.0
        else:
            probs = counts / tot
            bx = float(np.dot(probs, row_coords_2d.loc[cols_present, "Dim1"].values))
            by = float(np.dot(probs, row_coords_2d.loc[cols_present, "Dim2"].values))
        bary_list.append((w, bx, by))

    bary_df = pd.DataFrame(bary_list, columns=["word", "Dim1_bary", "Dim2_bary"]).set_index("word")

    return {
        "row_coords_2d": row_coords_2d,
        "bary_df": bary_df,
        "eigenvalues": eigenvalues,
        "explained": explained,
        "total_inertia": total_inertia,
        "chi2_total": chi2_total,
        "df": df_val,
        "p_value": p_value,
        "N": N_total,
        "df_ca": df_ca_input,
        "n_words_used": len(df_ca_input),
    }


# ═══════════════════════════════════════════════════════════════
#  2. Procrustes 거리
# ═══════════════════════════════════════════════════════════════

def compute_procrustes_distance(
    coords_ref: pd.DataFrame,
    coords_comp: pd.DataFrame,
    coord_cols: List[str] = None,
) -> Dict[str, float]:
    """두 CA 공간 사이의 Procrustes 거리를 계산한다."""
    if coord_cols is None:
        coord_cols = ["Dim1", "Dim2"]

    common_idx = coords_ref.index.intersection(coords_comp.index)
    if len(common_idx) < 2:
        return {"procrustes_d": np.nan, "disparity": np.nan}

    X1 = coords_ref.loc[common_idx, coord_cols].values.astype(float)
    X2 = coords_comp.loc[common_idx, coord_cols].values.astype(float)

    if HAS_SCIPY:
        try:
            _, _, disparity = scipy_procrustes(X1, X2)
            return {"procrustes_d": float(np.sqrt(disparity)), "disparity": float(disparity)}
        except Exception:
            pass

    # fallback
    X1c = X1 - X1.mean(axis=0)
    X2c = X2 - X2.mean(axis=0)
    norm1 = np.sqrt((X1c ** 2).sum())
    norm2 = np.sqrt((X2c ** 2).sum())
    if norm1 > 0:
        X1c /= norm1
    if norm2 > 0:
        X2c /= norm2
    d = float(np.sqrt(((X1c - X2c) ** 2).sum()))
    return {"procrustes_d": d, "disparity": d ** 2}


# ═══════════════════════════════════════════════════════════════
#  3. α 민감도 CA 비교
# ═══════════════════════════════════════════════════════════════

def _compute_word_stability(
    ca_results: Dict[float, Dict],
    abuse_order: List[str],
) -> List[Dict]:
    """각 단어가 α 변화에 따라 가장 가까운 학대유형이 바뀌는지 평가."""
    alphas = sorted(ca_results.keys())
    if len(alphas) < 2:
        return []

    common_words = None
    for alpha in alphas:
        bary = ca_results[alpha]["bary_df"]
        if common_words is None:
            common_words = set(bary.index)
        else:
            common_words &= set(bary.index)

    if not common_words:
        return []

    word_nearest = defaultdict(list)
    for alpha in alphas:
        row_coords = ca_results[alpha]["row_coords_2d"]
        bary_df = ca_results[alpha]["bary_df"]

        abuse_pts = {}
        for a in abuse_order:
            if a in row_coords.index:
                abuse_pts[a] = np.array([
                    row_coords.loc[a, "Dim1"],
                    row_coords.loc[a, "Dim2"],
                ])

        for w in common_words:
            w_pt = np.array([bary_df.loc[w, "Dim1_bary"], bary_df.loc[w, "Dim2_bary"]])
            min_dist = float("inf")
            nearest = None
            for a, a_pt in abuse_pts.items():
                d = np.sqrt(((w_pt - a_pt) ** 2).sum())
                if d < min_dist:
                    min_dist = d
                    nearest = a
            word_nearest[w].append(nearest)

    results = []
    n_alpha = len(alphas)
    for w in sorted(common_words):
        assignments = word_nearest[w]
        counts = Counter(assignments)
        most_common_abuse, most_common_count = counts.most_common(1)[0]
        stability = most_common_count / n_alpha

        results.append({
            "word": w,
            "dominant_abuse": most_common_abuse,
            "dominant_abuse_en": ABUSE_LABEL_EN.get(most_common_abuse, most_common_abuse),
            "stability_ratio": stability,
            "n_alpha_values": n_alpha,
            "assignments": "|".join(str(a) for a in assignments),
        })

    return results


def run_alpha_sensitivity_ca(
    weighted_tables: Dict[float, pd.DataFrame],
    abuse_order: List[str] = None,
    top_chi_words: int = 200,
    out_dir: str = "output/weighted_ca",
) -> Dict[str, Any]:
    """
    여러 α값의 가중 빈도표에 대해 CA를 수행하고, 구조 변화를 비교한다.
    """
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    os.makedirs(out_dir, exist_ok=True)

    alphas = sorted(weighted_tables.keys())
    if not alphas:
        print("[WARN] α 값이 없습니다.")
        return {}

    print(f"\n{'=' * 70}")
    print(f"  가중 CA α 민감도 분석")
    print(f"  α 값: {alphas}")
    print(f"{'=' * 70}")

    # ─── Step 1: 각 α에 대해 CA 수행 ───
    ca_results = {}
    for alpha in alphas:
        df_wt = weighted_tables[alpha]
        print(f"\n  [α={alpha:.1f}] CA 수행 중... (단어 {len(df_wt)}개)")
        result = run_weighted_ca(df_wt, abuse_order=abuse_order, top_chi_words=top_chi_words)
        if result is not None:
            ca_results[alpha] = result
            lam1 = result["explained"][0] * 100 if len(result["explained"]) > 0 else 0
            lam2 = result["explained"][1] * 100 if len(result["explained"]) > 1 else 0
            print(f"    → Dim1: {lam1:.1f}%, Dim2: {lam2:.1f}%, "
                  f"총 관성: {result['total_inertia']:.6f}, "
                  f"χ²={result['chi2_total']:.1f} (p={result['p_value']:.2e})")

    if not ca_results:
        print("[WARN] 유효한 CA 결과가 없습니다.")
        return {}

    # ─── Step 2: Procrustes 거리 ───
    ref_alpha = min(ca_results.keys())
    ref_coords = ca_results[ref_alpha]["row_coords_2d"]

    proc_rows = []
    for alpha in sorted(ca_results.keys()):
        comp_coords = ca_results[alpha]["row_coords_2d"]
        proc = compute_procrustes_distance(ref_coords, comp_coords)
        proc_rows.append({
            "alpha_ref": ref_alpha,
            "alpha_comp": alpha,
            "procrustes_d": proc["procrustes_d"],
            "disparity": proc["disparity"],
        })
    procrustes_df = pd.DataFrame(proc_rows)

    # ─── Step 3: 설명 관성 변화 ───
    inertia_rows = []
    for alpha in sorted(ca_results.keys()):
        r = ca_results[alpha]
        row = {
            "alpha": alpha,
            "total_inertia": r["total_inertia"],
            "chi2_total": r["chi2_total"],
            "df": r["df"],
            "p_value": r["p_value"],
            "N": r["N"],
            "n_words": r["n_words_used"],
        }
        for dim_idx, val in enumerate(r["explained"]):
            row[f"lambda_dim{dim_idx + 1}"] = val
            row[f"lambda_dim{dim_idx + 1}_pct"] = val * 100
        for dim_idx, val in enumerate(r["eigenvalues"]):
            row[f"eigenvalue_dim{dim_idx + 1}"] = val
        inertia_rows.append(row)
    inertia_df = pd.DataFrame(inertia_rows)

    # ─── Step 4: 학대유형 좌표 궤적 ───
    traj_rows = []
    for alpha in sorted(ca_results.keys()):
        coords = ca_results[alpha]["row_coords_2d"]
        for abuse_name in coords.index:
            if abuse_name not in abuse_order:
                continue
            traj_rows.append({
                "alpha": alpha,
                "abuse_type": abuse_name,
                "abuse_en": ABUSE_LABEL_EN.get(abuse_name, abuse_name),
                "Dim1": float(coords.loc[abuse_name, "Dim1"]),
                "Dim2": float(coords.loc[abuse_name, "Dim2"]),
            })
    trajectory_df = pd.DataFrame(traj_rows)

    # ─── Step 5: 단어 안정성 ───
    stability_rows = _compute_word_stability(ca_results, abuse_order)
    stability_df = pd.DataFrame(stability_rows) if stability_rows else pd.DataFrame()

    # ─── CSV 저장 ───
    procrustes_df.to_csv(os.path.join(out_dir, "weighted_ca_procrustes.csv"), encoding="utf-8-sig", index=False)
    inertia_df.to_csv(os.path.join(out_dir, "weighted_ca_inertia_by_alpha.csv"), encoding="utf-8-sig", index=False)
    trajectory_df.to_csv(os.path.join(out_dir, "weighted_ca_abuse_trajectory.csv"), encoding="utf-8-sig", index=False)
    if not stability_df.empty:
        stability_df.to_csv(os.path.join(out_dir, "weighted_ca_word_stability.csv"), encoding="utf-8-sig", index=False)

    # ─── 요약 출력 ───
    print(f"\n{'=' * 70}")
    print(f"  가중 CA 결과 요약")
    print(f"{'=' * 70}")
    print(f"\n  [Procrustes 거리 (기준: α={ref_alpha})]")
    for _, row in procrustes_df.iterrows():
        print(f"    α={row['alpha_comp']:.1f} → d={row['procrustes_d']:.4f}")

    print(f"\n  [설명 관성 변화]")
    print(f"    {'α':>5s}  {'Dim1%':>8s}  {'Dim2%':>8s}  {'총 관성':>10s}  {'χ²':>10s}")
    for _, row in inertia_df.iterrows():
        d1 = row.get("lambda_dim1_pct", 0)
        d2 = row.get("lambda_dim2_pct", 0)
        print(f"    {row['alpha']:5.1f}  {d1:8.1f}  {d2:8.1f}  "
              f"{row['total_inertia']:10.6f}  {row['chi2_total']:10.1f}")

    if not stability_df.empty:
        mean_stab = stability_df["stability_ratio"].mean()
        print(f"\n  [단어 안정성] 평균 안정성 비율: {mean_stab:.3f}")

    return {
        "ca_results": ca_results,
        "procrustes_df": procrustes_df,
        "inertia_df": inertia_df,
        "trajectory_df": trajectory_df,
        "stability_df": stability_df,
    }


# ═══════════════════════════════════════════════════════════════
#  4. 시각화
# ═══════════════════════════════════════════════════════════════

def plot_weighted_ca_biplot(
    ca_result: Dict[str, Any],
    alpha: float,
    abuse_order: List[str] = None,
    top_words: int = 60,
    out_path: str = None,
    title_suffix: str = "",
):
    """특정 α 값의 가중 CA biplot을 그린다."""
    if ca_result is None:
        return
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    row_coords = ca_result["row_coords_2d"]
    bary_df = ca_result["bary_df"].copy()
    explained = ca_result["explained"]

    bary_df["_dist"] = np.sqrt(bary_df["Dim1_bary"] ** 2 + bary_df["Dim2_bary"] ** 2)
    words_to_plot = bary_df.nlargest(min(top_words, len(bary_df)), "_dist").index

    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)
    ax.axhline(0, ls="--", lw=0.6, alpha=0.4, color="gray")
    ax.axvline(0, ls="--", lw=0.6, alpha=0.4, color="gray")

    # 학대유형 점
    for abuse_name in abuse_order:
        if abuse_name not in row_coords.index:
            continue
        x = float(row_coords.loc[abuse_name, "Dim1"])
        y = float(row_coords.loc[abuse_name, "Dim2"])
        color = ABUSE_COLORS.get(abuse_name, "black")
        ax.scatter([x], [y], marker="s", s=160, color=color, alpha=0.95, zorder=6)
        label = ABUSE_LABEL_EN.get(abuse_name, abuse_name)
        ax.text(x, y, f"  {label}", fontsize=12, weight="bold", va="center", zorder=7)

    # 학대유형 좌표 dict
    abuse_pts = {}
    for a in abuse_order:
        if a in row_coords.index:
            abuse_pts[a] = np.array([row_coords.loc[a, "Dim1"], row_coords.loc[a, "Dim2"]])

    # 단어 점
    for w in words_to_plot:
        bx = float(bary_df.loc[w, "Dim1_bary"])
        by = float(bary_df.loc[w, "Dim2_bary"])
        w_pt = np.array([bx, by])

        min_d = float("inf")
        nearest_abuse = None
        for a, a_pt in abuse_pts.items():
            d = np.sqrt(((w_pt - a_pt) ** 2).sum())
            if d < min_d:
                min_d = d
                nearest_abuse = a

        color = ABUSE_COLORS.get(nearest_abuse, "black")
        ax.scatter([bx], [by], s=30, color=color, alpha=0.7, zorder=3)
        ax.text(bx, by, f" {w}", fontsize=7, color=color, alpha=0.8, zorder=4)

    # 범례
    legend_handles = [
        Line2D([0], [0], marker="s", ls="None", markerfacecolor=ABUSE_COLORS.get(a, "black"),
               markeredgecolor=ABUSE_COLORS.get(a, "black"),
               label=ABUSE_LABEL_EN.get(a, a), markersize=10)
        for a in abuse_order
    ]

    lam1 = explained[0] * 100 if len(explained) > 0 else 0
    lam2 = explained[1] * 100 if len(explained) > 1 else 0
    ax.set_xlabel(f"Dimension 1 ({lam1:.1f}% inertia)", fontsize=12)
    ax.set_ylabel(f"Dimension 2 ({lam2:.1f}% inertia)", fontsize=12)
    ax.set_title(
        f"Weighted CA Biplot (α={alpha:.1f}){title_suffix}\n"
        f"Total inertia: {ca_result['total_inertia']:.6f}, "
        f"χ²={ca_result['chi2_total']:.1f} (p={ca_result['p_value']:.2e})",
        fontsize=13,
    )
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.22, 1.02),
              title="Maltreatment type", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [SAVE] Weighted CA biplot (α={alpha:.1f}) → {out_path}")
    plt.close(fig)


def plot_alpha_sensitivity_panel(
    ca_results: Dict[float, Dict],
    abuse_order: List[str] = None,
    out_path: str = None,
):
    """α 민감도 분석 4-패널 비교 시각화."""
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    alphas = sorted(ca_results.keys())
    if len(alphas) < 2:
        print("[WARN] α 값이 2개 미만 → 민감도 패널을 건너뜁니다.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)

    # (A) 설명 관성 변화
    ax = axes[0, 0]
    dim1_vals = [ca_results[a]["explained"][0] * 100 if len(ca_results[a]["explained"]) > 0 else 0 for a in alphas]
    dim2_vals = [ca_results[a]["explained"][1] * 100 if len(ca_results[a]["explained"]) > 1 else 0 for a in alphas]
    total_vals = [d1 + d2 for d1, d2 in zip(dim1_vals, dim2_vals)]

    ax.plot(alphas, dim1_vals, "o-", color="#2196F3", label="Dim 1", linewidth=2, markersize=8)
    ax.plot(alphas, dim2_vals, "s-", color="#FF9800", label="Dim 2", linewidth=2, markersize=8)
    ax.plot(alphas, total_vals, "^--", color="#4CAF50", label="Dim 1+2", linewidth=1.5, markersize=7, alpha=0.7)
    ax.set_xlabel("α (sub-type weight)", fontsize=11)
    ax.set_ylabel("Explained inertia (%)", fontsize=11)
    ax.set_title("(A) Explained Inertia by α", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    # (B) Procrustes 거리
    ax = axes[0, 1]
    ref_alpha = alphas[0]
    ref_coords = ca_results[ref_alpha]["row_coords_2d"]
    proc_dists = []
    for a in alphas:
        comp_coords = ca_results[a]["row_coords_2d"]
        proc = compute_procrustes_distance(ref_coords, comp_coords)
        proc_dists.append(proc["procrustes_d"])

    ax.plot(alphas, proc_dists, "D-", color="#9C27B0", linewidth=2, markersize=8)
    ax.fill_between(alphas, proc_dists, alpha=0.15, color="#9C27B0")
    ax.set_xlabel("α (sub-type weight)", fontsize=11)
    ax.set_ylabel(f"Procrustes distance (ref: α={ref_alpha})", fontsize=11)
    ax.set_title("(B) Structural Change vs. α=0", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (C) 학대유형 좌표 궤적
    ax = axes[1, 0]
    ax.axhline(0, ls="--", lw=0.5, alpha=0.4, color="gray")
    ax.axvline(0, ls="--", lw=0.5, alpha=0.4, color="gray")

    for abuse_name in abuse_order:
        xs, ys = [], []
        for a in alphas:
            coords = ca_results[a]["row_coords_2d"]
            if abuse_name in coords.index:
                xs.append(float(coords.loc[abuse_name, "Dim1"]))
                ys.append(float(coords.loc[abuse_name, "Dim2"]))
            else:
                xs.append(np.nan)
                ys.append(np.nan)

        color = ABUSE_COLORS.get(abuse_name, "black")
        label = ABUSE_LABEL_EN.get(abuse_name, abuse_name)
        ax.plot(xs, ys, "o-", color=color, linewidth=1.5, markersize=6, alpha=0.7, label=label)

        if xs and ys:
            ax.scatter([xs[0]], [ys[0]], s=120, color=color, marker="s", zorder=5, edgecolors="black", linewidth=1.5)
            ax.scatter([xs[-1]], [ys[-1]], s=120, color=color, marker="^", zorder=5, edgecolors="black", linewidth=1.5)
            if len(xs) >= 2 and not (np.isnan(xs[0]) or np.isnan(xs[-1])):
                ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[0], ys[0]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.5))

    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)
    ax.set_title("(C) Abuse Type Trajectories (■=α=0, ▲=α=max)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # (D) 단어 안정성 분포
    ax = axes[1, 1]
    stability_rows = _compute_word_stability(ca_results, abuse_order)
    if stability_rows:
        stab_vals = [r["stability_ratio"] for r in stability_rows]
        ax.hist(stab_vals, bins=np.arange(0, 1.05, 0.1), color="#607D8B", edgecolor="white", alpha=0.85)
        mean_stab = np.mean(stab_vals)
        ax.axvline(mean_stab, color="red", ls="--", lw=2, label=f"Mean = {mean_stab:.2f}")
        ax.set_xlabel("Stability Ratio", fontsize=11)
        ax.set_ylabel("Number of Words", fontsize=11)
        ax.set_title("(D) Word Assignment Stability across α", fontsize=13, fontweight="bold")
        ax.legend(frameon=True)
    else:
        ax.text(0.5, 0.5, "No stability data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Weighted CA: α Sensitivity Analysis", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [SAVE] α 민감도 패널 → {out_path}")
    plt.close(fig)


def plot_multi_alpha_biplot(
    ca_results: Dict[float, Dict],
    abuse_order: List[str] = None,
    top_words: int = 40,
    out_path: str = None,
):
    """여러 α 값의 CA biplot을 한 그림에 비교."""
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    alphas = sorted(ca_results.keys())
    n = len(alphas)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=200)
    if n == 1:
        axes = [axes]

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        r = ca_results[alpha]
        row_coords = r["row_coords_2d"]
        bary_df = r["bary_df"].copy()
        explained = r["explained"]

        ax.axhline(0, ls="--", lw=0.4, alpha=0.3, color="gray")
        ax.axvline(0, ls="--", lw=0.4, alpha=0.3, color="gray")

        for a in abuse_order:
            if a not in row_coords.index:
                continue
            x = float(row_coords.loc[a, "Dim1"])
            y = float(row_coords.loc[a, "Dim2"])
            c = ABUSE_COLORS.get(a, "black")
            ax.scatter([x], [y], marker="s", s=100, color=c, zorder=5)
            ax.text(x, y, f" {ABUSE_LABEL_EN.get(a, a)}", fontsize=7, weight="bold", va="center")

        bary_df["_dist"] = np.sqrt(bary_df["Dim1_bary"] ** 2 + bary_df["Dim2_bary"] ** 2)
        words = bary_df.nlargest(min(top_words, len(bary_df)), "_dist").index

        abuse_pts = {
            a: np.array([row_coords.loc[a, "Dim1"], row_coords.loc[a, "Dim2"]])
            for a in abuse_order if a in row_coords.index
        }

        for w in words:
            bx = float(bary_df.loc[w, "Dim1_bary"])
            by = float(bary_df.loc[w, "Dim2_bary"])
            w_pt = np.array([bx, by])
            nearest = min(abuse_pts.items(), key=lambda x: np.sqrt(((w_pt - x[1]) ** 2).sum()))[0]
            c = ABUSE_COLORS.get(nearest, "black")
            ax.scatter([bx], [by], s=12, color=c, alpha=0.5, zorder=2)

        lam1 = explained[0] * 100 if len(explained) > 0 else 0
        lam2 = explained[1] * 100 if len(explained) > 1 else 0
        ax.set_title(f"α = {alpha:.1f}\nDim1: {lam1:.1f}%, Dim2: {lam2:.1f}%", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    fig.suptitle("Weighted CA Biplots across α values", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [SAVE] 다중 α biplot → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  5. Bridge Word 변화 분석
# ═══════════════════════════════════════════════════════════════

def analyze_bridge_shift_across_alpha(
    ca_results: Dict[float, Dict],
    abuse_order: List[str] = None,
    bridge_threshold: float = 0.6,
    out_dir: str = None,
) -> pd.DataFrame:
    """α 변화에 따른 Bridge Word 구성 변화를 분석한다."""
    if abuse_order is None:
        abuse_order = ABUSE_ORDER

    alphas = sorted(ca_results.keys())
    bridge_sets = {}
    all_rows = []

    for alpha in alphas:
        r = ca_results[alpha]
        df_ca = r.get("df_ca")
        if df_ca is None:
            continue

        cols = [a for a in abuse_order if a in df_ca.columns]
        if len(cols) < 2:
            continue

        bridges = set()
        for w in df_ca.index:
            counts = df_ca.loc[w, cols].values.astype(float)
            tot = counts.sum()
            if tot <= 0:
                continue
            probs = counts / tot
            sorted_idx = np.argsort(probs)[::-1]
            p1 = probs[sorted_idx[0]]
            p2 = probs[sorted_idx[1]] if len(sorted_idx) > 1 else 0

            if p1 > 0 and p2 / p1 >= bridge_threshold:
                bridges.add(w)
                all_rows.append({
                    "alpha": alpha,
                    "word": w,
                    "primary_abuse": cols[sorted_idx[0]],
                    "primary_abuse_en": ABUSE_LABEL_EN.get(cols[sorted_idx[0]], cols[sorted_idx[0]]),
                    "secondary_abuse": cols[sorted_idx[1]],
                    "secondary_abuse_en": ABUSE_LABEL_EN.get(cols[sorted_idx[1]], cols[sorted_idx[1]]),
                    "p1": p1,
                    "p2": p2,
                    "p2_over_p1": p2 / p1 if p1 > 0 else 0,
                })

        bridge_sets[alpha] = bridges

    df_all = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    # Jaccard 유사도
    jacc_rows = []
    ref_alpha = alphas[0] if alphas else None
    ref_set = bridge_sets.get(ref_alpha, set())

    for alpha in alphas:
        comp_set = bridge_sets.get(alpha, set())
        union = ref_set | comp_set
        inter = ref_set & comp_set
        jacc = len(inter) / len(union) if union else 1.0
        jacc_rows.append({
            "alpha_ref": ref_alpha,
            "alpha_comp": alpha,
            "n_bridge_ref": len(ref_set),
            "n_bridge_comp": len(comp_set),
            "n_intersection": len(inter),
            "jaccard": jacc,
        })
    jacc_df = pd.DataFrame(jacc_rows)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if not df_all.empty:
            df_all.to_csv(os.path.join(out_dir, "weighted_ca_bridge_words_by_alpha.csv"),
                          encoding="utf-8-sig", index=False)
        jacc_df.to_csv(os.path.join(out_dir, "weighted_ca_bridge_jaccard_by_alpha.csv"),
                       encoding="utf-8-sig", index=False)

    if alphas:
        print(f"\n  [Bridge Word 변화 (threshold={bridge_threshold})]")
        print(f"    {'α':>5s}  {'Bridge수':>8s}  {'Jaccard':>8s}")
        for _, row in jacc_df.iterrows():
            print(f"    {row['alpha_comp']:5.1f}  {int(row['n_bridge_comp']):8d}  {row['jaccard']:8.3f}")

    return df_all


# ═══════════════════════════════════════════════════════════════
#  6. 통합 실행 함수 (파이프라인에서 호출)
# ═══════════════════════════════════════════════════════════════

def run_weighted_ca_analysis(
    json_files: List[str],
    abuse_order: List[str] = None,
    alpha_values: List[float] = None,
    top_chi_words: int = 200,
    min_total_count: int = 8,
    out_dir: str = "output/weighted_ca",
    bridge_threshold: float = 0.6,
    allowed_groups=None,
    only_negative: bool = False,
) -> Dict[str, Any]:
    """
    가중 CA 확장 분석의 전체 파이프라인을 실행한다.

    ─── 실행 순서 ───
    1) JSON에서 아동별 main/sub abuse + 토큰 수집
    2) 각 α에 대해 가중 빈도표 구축
    3) 각 α에 대해 CA 수행 + Procrustes/관성/궤적/안정성 비교
    4) Bridge Word 변화 분석
    5) 시각화 (biplot, 민감도 패널, 다중 α biplot)
    6) CSV/PNG 저장

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    abuse_order : list[str]
        학대유형 순서
    alpha_values : list[float]
        검증할 α 목록 (기본: [0.0, 0.3, 0.5, 0.7, 1.0])
    top_chi_words : int
        CA에 사용할 상위 단어 수
    min_total_count : int
        최소 빈도 기준
    out_dir : str
        결과 저장 디렉토리
    bridge_threshold : float
        Bridge word 판별 임계값
    allowed_groups : set or None
        허용된 정서군
    only_negative : bool
        부정 정서군만 사용할지 여부

    Returns
    -------
    dict: 전체 분석 결과
    """
    if abuse_order is None:
        abuse_order = ABUSE_ORDER
    if alpha_values is None:
        alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]

    os.makedirs(out_dir, exist_ok=True)

    # ─── Step 1: 아동별 데이터 수집 ───
    print(f"\n{'=' * 70}")
    print(f"  [WEIGHTED-CA] 가중 CA 확장 분석 시작")
    print(f"  α 값: {alpha_values}")
    print(f"{'=' * 70}")

    records = collect_child_records(
        json_files=json_files,
        allowed_groups=allowed_groups,
        only_negative=only_negative,
    )
    print(f"  수집된 아동 수: {len(records)}")

    if not records:
        print("[WARN] 유효한 아동 레코드가 없습니다.")
        return {}

    # 주/부 학대유형 분포 로그
    main_counts = Counter(r["main_abuse"] for r in records)
    sub_counts = Counter(s for r in records for s in r.get("sub_abuses", []))
    print(f"  주 학대유형 분포: {dict(main_counts)}")
    print(f"  부 학대유형 분포: {dict(sub_counts)}")
    n_with_subs = sum(1 for r in records if r.get("sub_abuses"))
    print(f"  부 학대유형이 있는 아동 비율: {n_with_subs}/{len(records)} ({n_with_subs / len(records) * 100:.1f}%)")

    # ─── Step 2: 각 α에 대해 가중 빈도표 구축 ───
    weighted_tables = {}
    for alpha in alpha_values:
        df = build_weighted_frequency_table(
            records, alpha=alpha, min_total_count=min_total_count, abuse_order=abuse_order,
        )
        if df is not None and not df.empty:
            weighted_tables[alpha] = df
            print(f"  α={alpha:.1f}: {len(df)} 단어")

    if not weighted_tables:
        print("[WARN] 가중 빈도표가 모두 비어 있습니다.")
        return {}

    # ─── Step 3: α 민감도 CA ───
    sensitivity_results = run_alpha_sensitivity_ca(
        weighted_tables=weighted_tables,
        abuse_order=abuse_order,
        top_chi_words=top_chi_words,
        out_dir=out_dir,
    )

    if not sensitivity_results:
        print("[WARN] 민감도 분석 실패.")
        return {}

    ca_results = sensitivity_results["ca_results"]

    # ─── Step 4: Bridge Word 변화 ───
    bridge_df = analyze_bridge_shift_across_alpha(
        ca_results=ca_results,
        abuse_order=abuse_order,
        bridge_threshold=bridge_threshold,
        out_dir=out_dir,
    )

    # ─── Step 5: 시각화 ───
    for alpha in sorted(ca_results.keys()):
        plot_weighted_ca_biplot(
            ca_result=ca_results[alpha],
            alpha=alpha,
            abuse_order=abuse_order,
            top_words=60,
            out_path=os.path.join(out_dir, f"weighted_ca_biplot_alpha_{alpha:.1f}.png"),
        )

    plot_alpha_sensitivity_panel(
        ca_results=ca_results,
        abuse_order=abuse_order,
        out_path=os.path.join(out_dir, "weighted_ca_sensitivity_panel.png"),
    )

    plot_multi_alpha_biplot(
        ca_results=ca_results,
        abuse_order=abuse_order,
        top_words=40,
        out_path=os.path.join(out_dir, "weighted_ca_multi_alpha_biplot.png"),
    )

    # ─── Step 6: 개별 좌표 CSV 저장 ───
    for alpha in sorted(ca_results.keys()):
        r = ca_results[alpha]
        r["row_coords_2d"].to_csv(
            os.path.join(out_dir, f"weighted_ca_row_coords_alpha_{alpha:.1f}.csv"),
            encoding="utf-8-sig",
        )
        r["bary_df"][["Dim1_bary", "Dim2_bary"]].to_csv(
            os.path.join(out_dir, f"weighted_ca_bary_coords_alpha_{alpha:.1f}.csv"),
            encoding="utf-8-sig",
        )

    # ─── 최종 요약 ───
    n_files = len([f for f in os.listdir(out_dir) if not f.startswith(".")])
    print(f"\n{'=' * 70}")
    print(f"  가중 CA 확장 분석 완료")
    print(f"  출력 디렉토리: {out_dir}")
    print(f"  α 값: {sorted(ca_results.keys())}")
    print(f"  생성 파일 수: {n_files}")
    print(f"{'=' * 70}")

    sensitivity_results["bridge_df"] = bridge_df
    sensitivity_results["records_count"] = len(records)
    return sensitivity_results