"""
revision_extensions.py
======================
논문 리비전 요구사항 3가지를 구현하는 확장 모듈.

요구사항 1: 라벨링 임계값 민감도 분석 (Aₖ > 4, 5, 6, 7, 8)
요구사항 2: 다중비교 보정(FDR) 후 유의미 토큰 수 재보고
요구사항 3: 2개 이상 분류기 비교 → 모호성 지대의 model-invariance 입증

사용법:
    아래 세 함수를 기존 파이프라인의 적절한 위치에서 호출하거나,
    독립 실행(standalone)으로도 사용할 수 있습니다.
"""

from __future__ import annotations

import os
import json
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd

# ── 프로젝트 내부 모듈 import ──────────────────────────────────
# 이 파일을 프로젝트 루트에 두고 실행하거나,
# sys.path에 프로젝트 경로를 추가한 뒤 사용하세요.
from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean
from abuse_pipeline.stats.stats import (
    compute_chi_square, add_bh_fdr, compute_log_odds,
    compute_prob_bridge_for_words,
)
from abuse_pipeline.stats.ca import run_abuse_ca_with_prob_bridges

# ── sklearn / 공유 코드 ─────────────────────────────────────────
try:
    from abuse_pipeline.classifiers.classifier_utils import run_tfidf_classifiers_cv
    HAS_SKLEARN = C.HAS_SKLEARN
except ImportError:
    HAS_SKLEARN = False

# ── prince import (CA용) ────────────────────────────────────────
try:
    import prince
    HAS_PRINCE = True
except ImportError:
    HAS_PRINCE = False


# ═══════════════════════════════════════════════════════════════
# 요구사항 1: 라벨링 임계값 민감도 분석
# ═══════════════════════════════════════════════════════════════

# 공용 상수 참조 (common.py에서 중앙 관리)
try:
    _SEVERITY_RANK = C.SEVERITY_RANK
except Exception:
    _SEVERITY_RANK = {"성학대": 0, "신체학대": 1, "정서학대": 2, "방임": 3}


def classify_abuse_main_sub_with_threshold(
    rec,
    abuse_threshold: int = 6,
    abuse_order=None,
    sub_threshold: int = 2,
    use_clinical_text: bool = True,
):
    """
    classify_abuse_main_sub()의 임계값 파라미터화 버전.

    원래 코드에서 `nonzero = {a: s for a, s in abuse_scores.items() if s > 6}`
    의 하드코딩된 6을 `abuse_threshold` 파라미터로 대체합니다.

    Parameters
    ----------
    rec : dict
        한 아동의 JSON record
    abuse_threshold : int
        학대유형 점수 합산값이 이 값을 초과해야 주(main) 학대유형으로 인정.
        기본값 6은 원래 코드와 동일.
    abuse_order : list[str]
        학대유형 순서 리스트. None이면 ["방임","정서학대","신체학대","성학대"]
    sub_threshold : int
        부(sub) 학대유형 인정 임계값
    use_clinical_text : bool
        임상진단/임상가 종합소견 텍스트에서 학대유형 키워드를 추출할지 여부

    Returns
    -------
    main_abuse : str or None
    sub_abuses : list[str]
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

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

    # ★ 핵심 변경: 하드코딩 6 → abuse_threshold 파라미터
    nonzero = {a: s for a, s in abuse_scores.items() if s > abuse_threshold}
    if nonzero:
        # [FIX] deterministic tie-breaking: (점수, -심각도)
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
        # [FIX] deterministic tie-breaking
        main = sorted(subs,
                      key=lambda x: (abuse_scores.get(x, 0),
                                     -_SEVERITY_RANK.get(x, 999)),
                      reverse=True)[0]
        subs.remove(main)

    if main is None:
        return None, []

    return main, sorted(subs, key=lambda x: _SEVERITY_RANK.get(x, 999))


def run_labeling_threshold_sensitivity(
    json_files: list[str],
    thresholds: list[int] = None,
    abuse_order: list[str] = None,
    out_dir: str = "revision_output/threshold_sensitivity",
    run_ca: bool = True,
    top_chi_for_ca: int = 200,
):
    """
    요구사항 1: 라벨링 임계값 민감도 분석.

    학대여부 점수 임계값 Aₖ를 4, 5, 6, 7, 8로 변동시키면서:
      (a) 각 임계값별 ABUSE 분류 구성 (몇 명이 어떤 유형으로 분류되는지)
      (b) 토큰-학대유형 빈도표의 변화
      (c) CA 대응분석 구조의 안정성 (관성, 차원별 기여율, 학대유형 좌표)

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    thresholds : list[int]
        검증할 임계값 목록. 기본값 [4, 5, 6, 7, 8]
    abuse_order : list[str]
        학대유형 순서. 기본값 ["방임","정서학대","신체학대","성학대"]
    out_dir : str
        결과 저장 디렉토리
    run_ca : bool
        True이면 각 임계값별 CA를 실행하여 구조 안정성 비교
    top_chi_for_ca : int
        CA에 투입할 χ² 상위 단어 수

    Returns
    -------
    dict with keys:
        "composition_df": 각 임계값별 학대유형 분류 구성표
        "ca_stability_df": CA 안정성 비교표 (run_ca=True일 때)
    """
    if thresholds is None:
        thresholds = [4, 5, 6, 7, 8]
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    os.makedirs(out_dir, exist_ok=True)

    # ── (A) 각 임계값별 분류 구성 변화 ──
    composition_rows = []
    per_child_labels = {}  # threshold → {child_id: main_abuse}

    for thresh in thresholds:
        label_counts = {a: 0 for a in abuse_order}
        label_counts["None"] = 0
        child_labels = {}

        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue

            child_id = rec.get("info", {}).get("ID", os.path.basename(path))

            main_abuse, _ = classify_abuse_main_sub_with_threshold(
                rec, abuse_threshold=thresh, abuse_order=abuse_order,
            )

            if main_abuse in abuse_order:
                label_counts[main_abuse] += 1
                child_labels[child_id] = main_abuse
            else:
                label_counts["None"] += 1
                child_labels[child_id] = None

        per_child_labels[thresh] = child_labels

        total_classified = sum(label_counts[a] for a in abuse_order)
        for abuse_type, count in label_counts.items():
            composition_rows.append({
                "threshold": thresh,
                "abuse_type": abuse_type,
                "n_children": count,
                "pct_of_classified": (
                    count / total_classified * 100
                    if total_classified > 0 and abuse_type != "None"
                    else np.nan
                ),
            })

    df_composition = pd.DataFrame(composition_rows)
    comp_path = os.path.join(out_dir, "threshold_composition.csv")
    df_composition.to_csv(comp_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 임계값별 분류 구성표 → {comp_path}")

    # ── (A-2) 임계값 간 라벨 일치도(Cohen's κ) ──
    kappa_rows = []
    for t1, t2 in itertools.combinations(thresholds, 2):
        labels1 = per_child_labels[t1]
        labels2 = per_child_labels[t2]
        common_ids = set(labels1.keys()) & set(labels2.keys())
        if len(common_ids) < 10:
            continue
        y1 = [labels1[cid] for cid in common_ids if labels1[cid] and labels2[cid]]
        y2 = [labels2[cid] for cid in common_ids if labels1[cid] and labels2[cid]]
        if len(y1) < 10:
            continue
        try:
            kappa = cohen_kappa_score(y1, y2)
        except Exception:
            kappa = np.nan
        n_agree = sum(1 for a, b in zip(y1, y2) if a == b)
        kappa_rows.append({
            "threshold_1": t1,
            "threshold_2": t2,
            "n_common": len(y1),
            "n_agree": n_agree,
            "agreement_rate": n_agree / len(y1) if len(y1) > 0 else np.nan,
            "cohen_kappa": kappa,
        })

    df_kappa = pd.DataFrame(kappa_rows)
    kappa_path = os.path.join(out_dir, "threshold_pairwise_kappa.csv")
    df_kappa.to_csv(kappa_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 임계값 간 라벨 일치도(κ) → {kappa_path}")

    # ── (B) 각 임계값별 토큰-학대유형 빈도표 + CA ──
    ca_stability_rows = []
    ca_row_coords_all = {}

    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"[THRESHOLD SENSITIVITY] Aₖ > {thresh}")
        print(f"{'='*60}")

        # 토큰 수집
        abuse_texts = {a: [] for a in abuse_order}
        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue

            main_abuse, _ = classify_abuse_main_sub_with_threshold(
                rec, abuse_threshold=thresh, abuse_order=abuse_order,
            )
            if main_abuse not in abuse_order:
                continue

            speech_list = extract_child_speech(rec)
            if speech_list:
                abuse_texts[main_abuse].extend(speech_list)

        # 빈도표 구축
        rows_wc = []
        for abuse_name in abuse_order:
            texts = abuse_texts.get(abuse_name, [])
            if not texts:
                continue
            joined = " ".join(t for t in texts if isinstance(t, str))
            tokens = tokenize_korean(joined)
            for w in tokens:
                rows_wc.append({"abuse": abuse_name, "word": w})

        df_wc = pd.DataFrame(rows_wc)
        if df_wc.empty:
            print(f"  [SKIP] 임계값 {thresh}: 토큰이 없습니다.")
            continue

        df_counts = (
            df_wc.groupby(["word", "abuse"]).size()
            .unstack("abuse").fillna(0)
        )
        for a in abuse_order:
            if a not in df_counts.columns:
                df_counts[a] = 0
        df_counts = df_counts[abuse_order]

        df_counts["total"] = df_counts.sum(axis=1)
        df_counts = df_counts[df_counts["total"] >= 8].drop(columns=["total"])

        n_words = len(df_counts)
        print(f"  필터링 후 단어 수: {n_words}")

        # 빈도표 저장
        counts_path = os.path.join(out_dir, f"counts_threshold_{thresh}.csv")
        df_counts.to_csv(counts_path, encoding="utf-8-sig")

        # CA 실행 (run_ca=True인 경우)
        if run_ca and HAS_PRINCE and n_words >= 10:
            chi_df = compute_chi_square(df_counts, abuse_order)
            chi_df = add_bh_fdr(chi_df, p_col="p_value", out_col="p_fdr_bh")

            chi_sorted = chi_df.sort_values("chi2", ascending=False)
            ca_words = chi_sorted.head(min(top_chi_for_ca, len(chi_sorted))).index
            df_ca_input = df_counts.loc[df_counts.index.intersection(ca_words)]

            if len(df_ca_input) < 5:
                print(f"  [SKIP] CA: 단어 수 부족 ({len(df_ca_input)})")
                continue

            X = df_ca_input.T  # rows=abuse, cols=word

            ca = prince.CA(
                n_components=min(2, len(abuse_order) - 1),
                n_iter=10, copy=True, check_input=True, random_state=42,
            ).fit(X)

            row_coords = ca.row_coordinates(X)
            row_coords_2d = row_coords.iloc[:, :2].copy()
            row_coords_2d.columns = ["Dim1", "Dim2"]

            # 관성(inertia) 정보
            total_inertia = getattr(ca, "total_inertia_", np.nan)
            eigenvalues = getattr(ca, "eigenvalues_", None)
            if eigenvalues is not None and len(eigenvalues) >= 2:
                lam1 = eigenvalues[0] / total_inertia if total_inertia > 0 else np.nan
                lam2 = eigenvalues[1] / total_inertia if total_inertia > 0 else np.nan
            else:
                lam1 = lam2 = np.nan

            ca_stability_rows.append({
                "threshold": thresh,
                "n_words_in_ca": len(df_ca_input),
                "total_inertia": total_inertia,
                "dim1_pct": lam1 * 100 if np.isfinite(lam1) else np.nan,
                "dim2_pct": lam2 * 100 if np.isfinite(lam2) else np.nan,
            })

            # 학대유형별 CA 좌표 저장
            for abuse_name in abuse_order:
                if abuse_name in row_coords_2d.index:
                    ca_stability_rows[-1][f"{abuse_name}_Dim1"] = row_coords_2d.loc[abuse_name, "Dim1"]
                    ca_stability_rows[-1][f"{abuse_name}_Dim2"] = row_coords_2d.loc[abuse_name, "Dim2"]

            ca_row_coords_all[thresh] = row_coords_2d

            coords_path = os.path.join(out_dir, f"ca_row_coords_threshold_{thresh}.csv")
            row_coords_2d.to_csv(coords_path, encoding="utf-8-sig")
            print(f"  [저장] CA 좌표 → {coords_path}")

    # ── CA 안정성 요약표 ──
    df_ca_stability = pd.DataFrame()
    if ca_stability_rows:
        df_ca_stability = pd.DataFrame(ca_stability_rows)
        stability_path = os.path.join(out_dir, "ca_stability_across_thresholds.csv")
        df_ca_stability.to_csv(stability_path, encoding="utf-8-sig", index=False)
        print(f"\n[저장] CA 안정성 요약 → {stability_path}")

    # ── CA 좌표 간 Procrustes 유사도 (참조 = threshold 6) ──
    if len(ca_row_coords_all) >= 2 and 6 in ca_row_coords_all:
        ref = ca_row_coords_all[6]
        procrustes_rows = []
        for thresh, coords in ca_row_coords_all.items():
            common_idx = ref.index.intersection(coords.index)
            if len(common_idx) < 3:
                continue
            A = ref.loc[common_idx].values
            B = coords.loc[common_idx].values
            # 간이 Procrustes: R² = 1 - ||A-B||² / ||A||²
            ss_diff = float(np.sum((A - B) ** 2))
            ss_ref = float(np.sum(A ** 2))
            r2 = 1 - ss_diff / ss_ref if ss_ref > 0 else np.nan
            procrustes_rows.append({
                "ref_threshold": 6,
                "comp_threshold": thresh,
                "n_common_types": len(common_idx),
                "ss_diff": ss_diff,
                "r_squared": r2,
            })
        if procrustes_rows:
            df_proc = pd.DataFrame(procrustes_rows)
            proc_path = os.path.join(out_dir, "ca_procrustes_vs_threshold6.csv")
            df_proc.to_csv(proc_path, encoding="utf-8-sig", index=False)
            print(f"[저장] CA Procrustes 유사도 → {proc_path}")

    return {
        "composition_df": df_composition,
        "kappa_df": df_kappa,
        "ca_stability_df": df_ca_stability,
    }


# ═══════════════════════════════════════════════════════════════
# 요구사항 2: FDR 보정 후 유의미 토큰 수 재보고
# ═══════════════════════════════════════════════════════════════

def run_fdr_significance_report(
    df_counts: pd.DataFrame,
    group_cols: list[str],
    analysis_name: str = "abuse",
    alpha_levels: list[float] = None,
    out_dir: str = "revision_output/fdr_report",
):
    """
    요구사항 2: 다중비교 보정(FDR) 적용 결과 보고.

    이미 파이프라인에 add_bh_fdr()가 적용되어 있지만,
    리뷰어가 요구하는 것은 "보정 전후 유의미 토큰 수 비교"입니다.

    이 함수는:
      (a) 원래 p-value 기준 유의미 토큰 수
      (b) BH-FDR 보정 후 유의미 토큰 수
      (c) 각 α 수준(0.05, 0.01, 0.001)별 비교표
      (d) 보정으로 탈락한 토큰 목록

    Parameters
    ----------
    df_counts : pd.DataFrame
        단어 × 그룹 빈도표 (index=word, columns=group_cols)
    group_cols : list[str]
        그룹 칼럼명 리스트
    analysis_name : str
        분석 종류 태그 (예: "abuse", "valence")
    alpha_levels : list[float]
        검증할 유의수준 목록
    out_dir : str
        결과 저장 디렉토리

    Returns
    -------
    dict with keys: "summary_df", "dropped_tokens_df"
    """
    if alpha_levels is None:
        alpha_levels = [0.05, 0.01, 0.001]

    os.makedirs(out_dir, exist_ok=True)

    if df_counts.empty:
        print(f"[FDR] df_counts가 비어 있어 분석을 건너뜁니다.")
        return None

    # χ² 통계량 + p-value 계산
    chi_df = compute_chi_square(df_counts, group_cols)

    # BH-FDR 보정 적용
    chi_df = add_bh_fdr(chi_df, p_col="p_value", out_col="p_fdr_bh")

    total_tokens = len(chi_df)

    # ── 요약 테이블 생성 ──
    summary_rows = []
    for alpha in alpha_levels:
        n_raw = int((chi_df["p_value"] < alpha).sum())
        n_fdr = int((chi_df["p_fdr_bh"] < alpha).sum())
        n_dropped = n_raw - n_fdr

        summary_rows.append({
            "analysis": analysis_name,
            "alpha": alpha,
            "total_tokens_tested": total_tokens,
            "n_significant_raw_p": n_raw,
            "pct_significant_raw_p": n_raw / total_tokens * 100 if total_tokens > 0 else 0,
            "n_significant_fdr_q": n_fdr,
            "pct_significant_fdr_q": n_fdr / total_tokens * 100 if total_tokens > 0 else 0,
            "n_dropped_by_fdr": n_dropped,
            "retention_rate_pct": n_fdr / n_raw * 100 if n_raw > 0 else np.nan,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, f"fdr_significance_summary_{analysis_name}.csv")
    df_summary.to_csv(summary_path, encoding="utf-8-sig", index=False)
    print(f"[저장] FDR 보정 전후 유의미 토큰 요약 → {summary_path}")

    # ── α=0.05 기준 FDR로 탈락한 토큰 목록 ──
    mask_raw_sig = chi_df["p_value"] < 0.05
    mask_fdr_sig = chi_df["p_fdr_bh"] < 0.05
    mask_dropped = mask_raw_sig & ~mask_fdr_sig

    df_dropped = chi_df[mask_dropped].copy()
    df_dropped = df_dropped.sort_values("p_value")

    dropped_path = os.path.join(out_dir, f"fdr_dropped_tokens_{analysis_name}_alpha005.csv")
    df_dropped.to_csv(dropped_path, encoding="utf-8-sig")
    print(f"[저장] FDR 보정으로 탈락한 토큰 (α=0.05) → {dropped_path}")

    # ── FDR 보정 후에도 유의한 토큰 목록 ──
    df_survived = chi_df[mask_fdr_sig].copy()
    df_survived = df_survived.sort_values("chi2", ascending=False)

    survived_path = os.path.join(out_dir, f"fdr_survived_tokens_{analysis_name}_alpha005.csv")
    df_survived.to_csv(survived_path, encoding="utf-8-sig")
    print(f"[저장] FDR 보정 후 유의한 토큰 (α=0.05) → {survived_path}")

    # ── 전체 토큰의 p-value vs q-value 비교표 ──
    full_comparison = chi_df[["chi2", "p_value", "p_fdr_bh"]].copy()
    full_comparison["significant_raw_005"] = (full_comparison["p_value"] < 0.05).astype(int)
    full_comparison["significant_fdr_005"] = (full_comparison["p_fdr_bh"] < 0.05).astype(int)
    full_comparison = full_comparison.sort_values("chi2", ascending=False)

    full_path = os.path.join(out_dir, f"fdr_full_comparison_{analysis_name}.csv")
    full_comparison.to_csv(full_path, encoding="utf-8-sig")
    print(f"[저장] 전체 p vs q 비교표 → {full_path}")

    return {
        "summary_df": df_summary,
        "dropped_tokens_df": df_dropped,
        "survived_tokens_df": df_survived,
    }


# ═══════════════════════════════════════════════════════════════
# 요구사항 3: 2개 이상 분류기 비교 → model-invariant 오분류 패턴
# ═══════════════════════════════════════════════════════════════

def run_multi_classifier_comparison(
    df_text: pd.DataFrame,
    label_col: str,
    label_order: list[str],
    out_dir: str = "revision_output/multi_classifier",
    label_name: str = "abuse",
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    요구사항 3: 2개 이상의 분류기 비교.

    Logistic Regression (기존) + Random Forest + Linear SVM 3개 분류기로
    동일한 Stratified K-Fold CV를 수행하여:
      (a) 분류기별 성능(accuracy, Cohen's κ) 비교
      (b) 분류기별 혼동행렬(confusion matrix) 비교
      (c) "모호성 지대" 분석: 2개 이상의 분류기가 공통으로 오분류한 샘플 추출
      (d) 모델 불변(model-invariant) 오분류 패턴 검증

    Parameters
    ----------
    df_text : pd.DataFrame
        columns = ["ID", label_col, "tfidf_text"]
    label_col : str
        라벨 칼럼명 (예: "main_abuse", "group")
    label_order : list[str]
        라벨 순서
    out_dir : str
        결과 저장 디렉토리
    label_name : str
        파일명 태그
    n_splits : int
        K-Fold 분할 수
    random_state : int
        재현성을 위한 랜덤 시드

    Returns
    -------
    dict with keys:
        "performance_df": 분류기별 성능 비교표
        "confusion_matrices": {clf_name: cm_df}
        "ambiguity_zone_df": 모호성 지대 분석 결과
    """
    if not HAS_SKLEARN:
        print("[MULTI-CLF] scikit-learn이 설치되지 않아 분석을 건너뜁니다.")
        return None

    os.makedirs(out_dir, exist_ok=True)

    # 데이터 전처리
    df = df_text.copy()
    df = df.dropna(subset=[label_col, "tfidf_text"])
    df = df[df[label_col].isin(label_order)]

    if df.empty or df[label_col].nunique() < 2:
        print(f"[MULTI-CLF] 유효한 라벨이 2개 미만이라 분석을 건너뜁니다.")
        return None

    texts = df["tfidf_text"].astype(str).tolist()
    y = df[label_col].astype(str).values
    ids = df["ID"].values if "ID" in df.columns else np.arange(len(df))

    # ── classifier_utils.py 의 공유 CV 함수 사용 ──
    _name_map = {"LR": "LogisticRegression", "RF": "RandomForest", "SVM": "LinearSVM"}
    cv_results = run_tfidf_classifiers_cv(
        texts=texts, y=y, label_order=label_order,
        clf_names=["LR", "RF", "SVM"],
        n_splits=n_splits, random_state=random_state,
    )
    if not cv_results:
        print("[MULTI-CLF] CV 결과 없음")
        return None

    # cv_results 키를 표시용 이름으로 변환
    results = {}
    per_sample_preds = {}
    for cname, res in cv_results.items():
        display_name = _name_map.get(cname, cname)
        results[display_name] = res
        per_sample_preds[display_name] = res["per_sample"]

    # ── (A) 분류기별 성능 요약 ──
    perf_rows = []
    cm_dfs = {}

    for clf_name, res in results.items():
        overall_kappa = C.cohen_kappa_score(
            res["all_true"], res["all_pred"], labels=label_order
        )
        overall_acc = float(np.mean(
            [t == p for t, p in zip(res["all_true"], res["all_pred"])]
        ))

        perf_rows.append({
            "classifier": clf_name,
            "mean_accuracy": overall_acc,
            "overall_kappa": overall_kappa,
        })

        # 혼동행렬
        cm = C.confusion_matrix(
            res["all_true"], res["all_pred"], labels=label_order
        )
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in label_order],
            columns=[f"pred_{l}" for l in label_order],
        )
        cm_dfs[clf_name] = cm_df

        cm_path = os.path.join(out_dir, f"confusion_matrix_{clf_name}_{label_name}.csv")
        cm_df.to_csv(cm_path, encoding="utf-8-sig")
        print(f"[저장] {clf_name} 혼동행렬 → {cm_path}")

        # classification report
        report = C.classification_report(
            res["all_true"], res["all_pred"],
            labels=label_order, output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_path = os.path.join(
            out_dir, f"classification_report_{clf_name}_{label_name}.csv"
        )
        report_df.to_csv(report_path, encoding="utf-8-sig")

    # 성능 비교표 저장
    perf_df = pd.DataFrame(perf_rows)
    perf_path = os.path.join(out_dir, f"performance_comparison_{label_name}.csv")
    perf_df.to_csv(perf_path, encoding="utf-8-sig", index=False)
    print(f"\n[저장] 분류기 성능 비교표 → {perf_path}")

    # ── (B) 모호성 지대 분석: 공통 오분류 패턴 ──
    clf_names = list(results.keys())
    all_indices = sorted(
        set.intersection(
            *[set(per_sample_preds[c].keys()) for c in clf_names]
        )
    )

    ambiguity_rows = []
    for idx in all_indices:
        true_label = y[idx]
        preds = {c: per_sample_preds[c][idx] for c in clf_names}
        n_correct = sum(1 for p in preds.values() if p == true_label)
        n_wrong = len(clf_names) - n_correct
        is_all_wrong = int(n_wrong == len(clf_names))
        is_any_wrong = int(n_wrong > 0)

        # 오분류한 분류기들이 동일한 오답 라벨을 예측했는지 확인
        wrong_labels = [p for c, p in preds.items() if p != true_label]
        wrong_consensus = (
            len(set(wrong_labels)) == 1 if wrong_labels else False
        )

        row = {
            "sample_idx": idx,
            "ID": ids[idx] if idx < len(ids) else None,
            "true_label": true_label,
            "n_classifiers": len(clf_names),
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "all_wrong": is_all_wrong,
            "wrong_consensus": int(wrong_consensus),
            "wrong_consensus_label": (
                wrong_labels[0] if wrong_consensus and wrong_labels else None
            ),
        }
        for c in clf_names:
            row[f"pred_{c}"] = preds[c]
            row[f"correct_{c}"] = int(preds[c] == true_label)

        ambiguity_rows.append(row)

    df_ambiguity = pd.DataFrame(ambiguity_rows)

    ambiguity_path = os.path.join(
        out_dir, f"per_sample_predictions_{label_name}.csv"
    )
    df_ambiguity.to_csv(ambiguity_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 샘플별 예측 결과 → {ambiguity_path}")

    # ── 모호성 지대 요약 ──
    n_total = len(df_ambiguity)
    n_all_correct = int((df_ambiguity["n_wrong"] == 0).sum())
    n_some_wrong = int((df_ambiguity["n_wrong"] > 0).sum())
    n_all_wrong = int(df_ambiguity["all_wrong"].sum())
    n_consensus_wrong = int(df_ambiguity["wrong_consensus"].sum())

    # 모든 분류기가 동일하게 오분류한 케이스의 true→pred 패턴
    consensus_df = df_ambiguity[df_ambiguity["wrong_consensus"] == 1].copy()
    if not consensus_df.empty:
        pattern_counts = (
            consensus_df
            .groupby(["true_label", "wrong_consensus_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        pattern_path = os.path.join(
            out_dir, f"model_invariant_misclassification_patterns_{label_name}.csv"
        )
        pattern_counts.to_csv(pattern_path, encoding="utf-8-sig", index=False)
        print(f"[저장] 모델 불변 오분류 패턴 → {pattern_path}")
    else:
        pattern_counts = pd.DataFrame()

    # ── 분류기 간 오분류 일치도 (pairwise) ──
    clf_pairs_kappa = []
    for c1, c2 in itertools.combinations(clf_names, 2):
        preds1 = [per_sample_preds[c1][i] for i in all_indices]
        preds2 = [per_sample_preds[c2][i] for i in all_indices]
        try:
            kappa_inter = C.cohen_kappa_score(preds1, preds2)
        except Exception:
            kappa_inter = np.nan

        # 오분류 일치율: 둘 다 틀린 경우 중 같은 오답을 낸 비율
        both_wrong = [
            (preds1[i], preds2[i])
            for i in range(len(all_indices))
            if preds1[i] != y[all_indices[i]] and preds2[i] != y[all_indices[i]]
        ]
        same_wrong = sum(1 for p1, p2 in both_wrong if p1 == p2)
        wrong_agreement = same_wrong / len(both_wrong) if both_wrong else np.nan

        clf_pairs_kappa.append({
            "classifier_1": c1,
            "classifier_2": c2,
            "inter_classifier_kappa": kappa_inter,
            "n_both_wrong": len(both_wrong),
            "n_same_wrong_label": same_wrong,
            "wrong_label_agreement_rate": wrong_agreement,
        })

    df_inter_clf = pd.DataFrame(clf_pairs_kappa)
    inter_path = os.path.join(
        out_dir, f"inter_classifier_agreement_{label_name}.csv"
    )
    df_inter_clf.to_csv(inter_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 분류기 간 일치도 → {inter_path}")

    # ── 최종 모호성 지대 요약 ──
    zone_summary = pd.DataFrame([{
        "label_name": label_name,
        "n_total_samples": n_total,
        "n_all_classifiers_correct": n_all_correct,
        "pct_all_correct": n_all_correct / n_total * 100 if n_total > 0 else 0,
        "n_any_classifier_wrong": n_some_wrong,
        "pct_any_wrong": n_some_wrong / n_total * 100 if n_total > 0 else 0,
        "n_all_classifiers_wrong": n_all_wrong,
        "pct_all_wrong": n_all_wrong / n_total * 100 if n_total > 0 else 0,
        "n_model_invariant_misclass": n_consensus_wrong,
        "pct_model_invariant": (
            n_consensus_wrong / n_all_wrong * 100 if n_all_wrong > 0 else 0
        ),
        "interpretation": (
            "HIGH model-invariance"
            if n_all_wrong > 0 and n_consensus_wrong / n_all_wrong > 0.7
            else "MODERATE model-invariance"
            if n_all_wrong > 0 and n_consensus_wrong / n_all_wrong > 0.4
            else "LOW model-invariance"
        ),
    }])

    zone_path = os.path.join(
        out_dir, f"ambiguity_zone_summary_{label_name}.csv"
    )
    zone_summary.to_csv(zone_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 모호성 지대 요약 → {zone_path}")

    return {
        "performance_df": perf_df,
        "confusion_matrices": cm_dfs,
        "ambiguity_zone_df": df_ambiguity,
        "model_invariant_patterns": pattern_counts,
        "inter_classifier_df": df_inter_clf,
        "zone_summary": zone_summary,
    }


# ═══════════════════════════════════════════════════════════════
# 통합 실행 함수
# ═══════════════════════════════════════════════════════════════

def run_all_revisions(
    json_files: list[str],
    df_abuse_counts: pd.DataFrame = None,
    df_text_abuse: pd.DataFrame = None,
    abuse_order: list[str] = None,
    base_out_dir: str | None = None,
):
    """
    세 가지 리비전 요구사항을 한 번에 실행하는 통합 함수.

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    df_abuse_counts : pd.DataFrame
        기존 파이프라인에서 생성된 학대유형별 단어 빈도표.
        요구사항 2에서 사용. None이면 건너뜀.
    df_text_abuse : pd.DataFrame
        columns = ["ID", "main_abuse", "tfidf_text"]
        기존 파이프라인의 rows_text_abuse로 만든 DataFrame.
        요구사항 3에서 사용. None이면 건너뜀.
    abuse_order : list[str]
        학대유형 순서
    base_out_dir : str
        결과 저장 기본 디렉토리
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    if base_out_dir is None:
        base_out_dir = getattr(C, "REVISION_DIR", None) or "revision_output"
    os.makedirs(base_out_dir, exist_ok=True)
    results = {}

    # ── 요구사항 1: 임계값 민감도 분석 ──
    print("\n" + "=" * 72)
    print("[REVISION 1] 라벨링 임계값 민감도 분석 시작")
    print("=" * 72)
    try:
        res1 = run_labeling_threshold_sensitivity(
            json_files=json_files,
            thresholds=[4, 5, 6, 7, 8],
            abuse_order=abuse_order,
            out_dir=os.path.join(base_out_dir, "R1_threshold_sensitivity"),
            run_ca=True,
        )
        results["threshold_sensitivity"] = res1
    except Exception as e:
        print(f"[ERROR] 요구사항 1 실행 실패: {e}")
        import traceback; traceback.print_exc()

    # ── 요구사항 2: FDR 보정 재보고 ──
    print("\n" + "=" * 72)
    print("[REVISION 2] FDR 보정 후 유의미 토큰 수 재보고")
    print("=" * 72)
    if df_abuse_counts is not None and not df_abuse_counts.empty:
        try:
            res2 = run_fdr_significance_report(
                df_counts=df_abuse_counts,
                group_cols=abuse_order,
                analysis_name="abuse",
                out_dir=os.path.join(base_out_dir, "R2_fdr_report"),
            )
            results["fdr_report"] = res2
        except Exception as e:
            print(f"[ERROR] 요구사항 2 실행 실패: {e}")
            import traceback; traceback.print_exc()
    else:
        print("[SKIP] df_abuse_counts가 없어 요구사항 2를 건너뜁니다.")

    # ── 요구사항 3: 다중 분류기 비교 ──
    print("\n" + "=" * 72)
    print("[REVISION 3] 다중 분류기 비교 (모호성 지대 분석)")
    print("=" * 72)
    if df_text_abuse is not None and not df_text_abuse.empty:
        try:
            res3 = run_multi_classifier_comparison(
                df_text=df_text_abuse,
                label_col="main_abuse",
                label_order=abuse_order,
                out_dir=os.path.join(base_out_dir, "R3_multi_classifier"),
                label_name="abuse",
            )
            results["multi_classifier"] = res3
        except Exception as e:
            print(f"[ERROR] 요구사항 3 실행 실패: {e}")
            import traceback; traceback.print_exc()
    else:
        print("[SKIP] df_text_abuse가 없어 요구사항 3을 건너뜁니다.")

    print("\n" + "=" * 72)
    print("[REVISION] 모든 리비전 분석 완료")
    print(f"[REVISION] 결과 저장 위치: {base_out_dir}")
    print("=" * 72)

    return results


# ═══════════════════════════════════════════════════════════════
# 파이프라인 통합 가이드 (pipeline.py에 삽입할 코드)
# ═══════════════════════════════════════════════════════════════

INTEGRATION_GUIDE = """
# ============================================================
# pipeline.py에 아래 코드를 추가하면 리비전 분석이 자동 실행됩니다.
# 위치: run_pipeline() 함수의 마지막 (# 8. Frequency-matched
#        baseline 이후)
# ============================================================

    # =================================================
    # 9. [REVISION] 리비전 요구사항 분석
    # =================================================
    from .revision_extensions import run_all_revisions

    revision_out = os.path.join(C.OUTPUT_DIR, "revision")
    os.makedirs(revision_out, exist_ok=True)

    # df_text_abuse 준비 (이미 rows_text_abuse에서 만들어진 상태)
    df_text_abuse_rev = pd.DataFrame(rows_text_abuse) if rows_text_abuse else pd.DataFrame()

    run_all_revisions(
        json_files=json_files,
        df_abuse_counts=df_abuse_counts if not df_abuse_counts.empty else None,
        df_text_abuse=df_text_abuse_rev if not df_text_abuse_rev.empty else None,
        abuse_order=C.ABUSE_ORDER,
        base_out_dir=revision_out,
    )
"""


if __name__ == "__main__":
    print("이 모듈은 직접 실행용이 아닙니다.")
    print("파이프라인에서 import하여 사용하세요.")
    print()
    print("=== 파이프라인 통합 가이드 ===")
    print(INTEGRATION_GUIDE)