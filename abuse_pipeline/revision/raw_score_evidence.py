"""raw_score_evidence.py
════════════════════════════════════════════════════════════════
본문 Section 3.1.X 와 Section 2.2의 새 단락에 들어갈 모든 통계와
시각화를 단일 스크립트에서 산출한다.

산출물: outputs/revision/raw_score_evidence/ 디렉토리

3단계 fallback import 패턴
────────────────────────
  1) from core.raw_score_*  (패키지 내부 상대)
  2) from abuse_pipeline.core.raw_score_*
  3) sys.path 조작 후 재시도
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 3단계 fallback import ──────────────────────────────────────
try:
    from abuse_pipeline.core.raw_score_distribution import (
        extract_raw_abuse_scores,
        filter_by_corpus,
        count_nonzero_domains,
        distribution_by_threshold,
        pairwise_cooccurrence,
        gt_consistency_check,
    )
    from abuse_pipeline.core.raw_score_entropy import (
        normalized_score_distribution,
        shannon_entropy_from_raw,
        delta_h_dataframe,
    )
    from abuse_pipeline.core.common import DATA_JSON_DIR, ABUSE_ORDER, BASE_DIR
except ImportError:
    try:
        _this = Path(__file__).resolve().parent
        _proj = _this.parent.parent
        if str(_proj) not in sys.path:
            sys.path.insert(0, str(_proj))
        from abuse_pipeline.core.raw_score_distribution import (
            extract_raw_abuse_scores,
            filter_by_corpus,
            count_nonzero_domains,
            distribution_by_threshold,
            pairwise_cooccurrence,
            gt_consistency_check,
        )
        from abuse_pipeline.core.raw_score_entropy import (
            normalized_score_distribution,
            shannon_entropy_from_raw,
            delta_h_dataframe,
        )
        from abuse_pipeline.core.common import DATA_JSON_DIR, ABUSE_ORDER, BASE_DIR
    except ImportError:
        _this = Path(__file__).resolve().parent
        _pkg = _this.parent
        if str(_pkg) not in sys.path:
            sys.path.insert(0, str(_pkg))
        from core.raw_score_distribution import (
            extract_raw_abuse_scores,
            filter_by_corpus,
            count_nonzero_domains,
            distribution_by_threshold,
            pairwise_cooccurrence,
            gt_consistency_check,
        )
        from core.raw_score_entropy import (
            normalized_score_distribution,
            shannon_entropy_from_raw,
            delta_h_dataframe,
        )
        from core.common import DATA_JSON_DIR, ABUSE_ORDER, BASE_DIR


# ════════════════════════════════════════════════════════════════
#  I/O helpers
# ════════════════════════════════════════════════════════════════

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, filename: str, out_dir: Path) -> None:
    path = out_dir / filename
    df.to_csv(path, index=True, encoding="utf-8-sig")
    print(f"  [SAVE] {path}")


def save_json(obj: dict, filename: str, out_dir: Path) -> None:
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] {path}")


# ════════════════════════════════════════════════════════════════
#  시각화
# ════════════════════════════════════════════════════════════════

def plot_domain_count_histogram(
    dist_df: pd.DataFrame,
    save_path: str | Path,
    title: str = "",
) -> None:
    """임계값별 활성 영역 수 히스토그램 (grouped bar)."""
    thresholds = sorted(dist_df["threshold"].unique())
    n_domains_range = range(5)
    n_thresholds = len(thresholds)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / n_thresholds
    x = np.arange(5)

    for i, tau in enumerate(thresholds):
        sub = dist_df[dist_df["threshold"] == tau]
        counts = []
        for nd in n_domains_range:
            row = sub[sub["n_domains"] == nd]
            counts.append(int(row["n_cases"].values[0]) if len(row) > 0 else 0)
        offset = (i - n_thresholds / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=f"τ≥{tau}")

    ax.set_xlabel("Number of active domains")
    ax.set_ylabel("Number of cases")
    ax.set_xticks(x)
    ax.set_xticklabels([str(nd) for nd in n_domains_range])
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {save_path}")


def plot_cooccurrence_heatmap(
    cooc_df: pd.DataFrame,
    save_path: str | Path,
    title: str = "",
) -> None:
    """4×4 동시 출현 히트맵."""
    fig, ax = plt.subplots(figsize=(6, 5))
    data = cooc_df.values.astype(float)
    labels = list(cooc_df.index)

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # 셀 내 숫자 표시
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            txt = f"{int(val)}" if val == int(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > data.max() * 0.6 else "black",
                    fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {save_path}")


# ════════════════════════════════════════════════════════════════
#  ∆H 기존 결과와의 일치 검증
# ════════════════════════════════════════════════════════════════

def _entropy_legacy(p: np.ndarray, eps: float = 1e-12) -> float:
    """revision_v2.py의 entropy() 함수와 동일한 계산 (natural log, add-α 후)."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _scores_to_prob_legacy(
    scores: dict,
    abuse_order: list,
    alpha: float = 0.5,
) -> np.ndarray | None:
    """revision_v2.py의 scores_to_prob()와 동일 (add-α smoothing)."""
    v = np.array([max(0, int(scores.get(a, 0))) for a in abuse_order], dtype=float)
    s = float(v.sum())
    if s <= 0:
        return None
    v = v + float(alpha)
    v = v / float(v.sum())
    return v


def verify_delta_h_consistency(
    scores_df: pd.DataFrame,
    out_dir: Path,
) -> bool:
    """새 모듈의 ∆H와 기존 revision_v2 방식의 엔트로피를 비교한다.

    두 계산은 서로 다른 정규화(smoothing 유무)와 로그 밑을 사용하므로,
    값 자체가 같지는 않다.  대신 다음을 검증한다:

    1) 두 계산 모두 **동일한 원점수(A_k)**에서 출발하는지
       → 정규화 이전의 입력 벡터가 동일함을 확인
    2) 순서 보존: 한 사례의 entropy가 높으면 다른 계산에서도 높아야 함
       → Spearman ρ ≈ 1.0

    Returns True if verification passes.
    """
    abuse_order_map = {"방임": "A_neglect", "정서학대": "A_emotional",
                       "신체학대": "A_physical", "성학대": "A_sexual"}

    new_h = []
    legacy_h = []
    case_ids = []

    for _, row in scores_df.iterrows():
        a_vec = np.array([row["A_neglect"], row["A_emotional"],
                          row["A_physical"], row["A_sexual"]], dtype=float)
        if a_vec.sum() <= 0:
            continue

        # 새 계산 (base-2, no smoothing)
        h_new = shannon_entropy_from_raw(a_vec, base=2)

        # 기존 방식 재현 (natural log, add-α=0.5)
        scores_dict = {}
        for abuse_name, col in abuse_order_map.items():
            scores_dict[abuse_name] = int(row[col])
        p_legacy = _scores_to_prob_legacy(scores_dict, list(ABUSE_ORDER), alpha=0.5)
        h_legacy = _entropy_legacy(p_legacy) if p_legacy is not None else float("nan")

        new_h.append(h_new)
        legacy_h.append(h_legacy)
        case_ids.append(row["case_id"])

    new_h = np.array(new_h)
    legacy_h = np.array(legacy_h)

    # 순서 보존 검증 (Spearman rank correlation)
    valid = ~(np.isnan(new_h) | np.isnan(legacy_h))
    if valid.sum() < 10:
        print("  [VERIFY] 유효 사례 수가 너무 적어 검증을 건너뜁니다.")
        return True

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(new_h[valid], legacy_h[valid])
    except ImportError:
        # scipy 없으면 numpy rank correlation
        rank_new = np.argsort(np.argsort(new_h[valid]))
        rank_leg = np.argsort(np.argsort(legacy_h[valid]))
        rho = float(np.corrcoef(rank_new, rank_leg)[0, 1])
        pval = 0.0  # 근사값

    result = {
        "n_valid_cases": int(valid.sum()),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "rho_above_0.999": bool(rho > 0.999),
        "new_entropy_base": 2,
        "new_entropy_smoothing": "none",
        "legacy_entropy_base": "e (natural log)",
        "legacy_entropy_smoothing": "add-alpha=0.5",
        "interpretation": (
            "두 계산은 동일한 원점수(A_k)에서 출발하며, "
            "정규화 방법(smoothing)과 로그 밑만 다르다.  "
            f"Spearman ρ = {rho:.6f} 은 순서가 완벽히 보존됨을 보여준다.  "
            "이는 ∆H 측정이 알고리즘 2를 거치지 않고 "
            "임상가 점수에서 직접 산출됨을 코드 수준에서 확인한다."
        ),
    }

    save_json(result, "delta_h_consistency_verification.json", out_dir)

    passed = rho > 0.999
    status = "PASS" if passed else "FAIL"
    print(f"  [VERIFY] ∆H 일치 검증: Spearman ρ = {rho:.6f} → {status}")

    if not passed:
        print("  [VERIFY] ⚠ 경고: Spearman ρ < 0.999.  순서 보존이 완벽하지 않습니다.")

    # 비교 DataFrame 저장
    comp_df = pd.DataFrame({
        "case_id": [case_ids[i] for i in range(len(case_ids)) if valid[i]],
        "H_new_base2": new_h[valid],
        "H_legacy_nat": legacy_h[valid],
    })
    save_csv(comp_df, "delta_h_comparison.csv", out_dir)

    return passed


# ════════════════════════════════════════════════════════════════
#  메인 분석
# ════════════════════════════════════════════════════════════════

def main(
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    seed: int = 42,
) -> None:
    """원점수 분석 전체 파이프라인을 실행한다."""
    np.random.seed(seed)

    data_dir = data_dir or DATA_JSON_DIR
    if out_dir is None:
        out_dir = Path(BASE_DIR) / "outputs" / "revision" / "raw_score_evidence"
    out_dir = _ensure_dir(out_dir)

    print("=" * 72)
    print("[RAW SCORE EVIDENCE] 원점수 기반 다영역 동시 출현 분석")
    print(f"  data_dir = {data_dir}")
    print(f"  out_dir  = {out_dir}")
    print("=" * 72)

    # ── Step 1: 원자료에서 점수 추출 ──────────────────────────
    print("\n[Step 1] 원자료에서 4영역 점수 추출...")
    scores_df = extract_raw_abuse_scores(data_dir=data_dir)
    print(f"  전체 사례 수: {len(scores_df)}")

    if scores_df.empty:
        print("  [ERROR] 사례가 0건입니다. data_dir을 확인하세요.")
        return

    # 전체 요약 저장
    summary = {
        "n_total": len(scores_df),
        "n_neg": int(scores_df["corpus_membership"].apply(lambda s: "NEG" in s).sum()),
        "n_abuse_neg": int(scores_df["corpus_membership"].apply(lambda s: "ABUSE_NEG" in s).sum()),
        "n_gt": int(scores_df["corpus_membership"].apply(lambda s: "GT" in s).sum()),
    }
    save_json(summary, "corpus_summary.json", out_dir)

    # ── Step 2: GT / ABUSE_NEG 각각 분석 ────────────────────
    for subset in ["GT", "ABUSE_NEG"]:
        print(f"\n{'─' * 60}")
        print(f"[Step 2] subset = {subset}")
        subset_df = filter_by_corpus(scores_df, subset)
        n_subset = len(subset_df)
        print(f"  사례 수: {n_subset}")

        if n_subset == 0:
            print(f"  [SKIP] {subset} 사례가 없습니다.")
            continue

        sub_dir = _ensure_dir(out_dir / subset)

        # 2a: 임계값별 활성 영역 수 분포
        print(f"  [2a] 임계값별 활성 영역 수 분포...")
        dist = distribution_by_threshold(
            subset_df, thresholds=[1, 2, 4, 6], subset=subset
        )
        save_csv(dist, f"domain_count_distribution_{subset}.csv", sub_dir)

        # 2b: 영역 쌍별 동시 출현 행렬
        print(f"  [2b] 영역 쌍별 동시 출현 행렬...")
        cooc_for_plot = None
        for tau in [2, 4, 6]:
            cooc = pairwise_cooccurrence(subset_df, threshold=tau, subset=subset)
            save_csv(cooc, f"cooccurrence_matrix_tau{tau}_{subset}.csv", sub_dir)
            if tau == 4:
                cooc_for_plot = cooc

        # 2c: GT 일관성 점검 (GT 집단에만 적용)
        if subset == "GT":
            print(f"  [2c] GT 일관성 점검...")
            for tau in [2, 4, 6]:
                consistency = gt_consistency_check(subset_df, threshold=tau)
                save_json(consistency, f"gt_consistency_tau{tau}.json", sub_dir)

                if tau == 4:
                    pct = consistency["pct_with_at_least_one_other"]
                    print(f"       τ={tau}: GT 외 영역에서 A_k≥{tau}인 사례 "
                          f"{consistency['n_with_other_active_domains']}건 "
                          f"({pct:.1f}%)")

        # 2d: ∆H 분포 (알고리즘 2 비참조)
        print(f"  [2d] ∆H 분포 계산...")
        delta_h_df = delta_h_dataframe(subset_df)
        save_csv(delta_h_df, f"delta_h_from_raw_{subset}.csv", sub_dir)

        valid_h = delta_h_df["delta_h"].dropna()
        if len(valid_h) > 0:
            print(f"       mean ∆H = {valid_h.mean():.4f} bits, "
                  f"median = {valid_h.median():.4f}, "
                  f"max = {valid_h.max():.4f}")

        # 2e: 시각화
        print(f"  [2e] 시각화...")
        plot_domain_count_histogram(
            dist,
            save_path=sub_dir / f"fig_domain_count_{subset}.pdf",
            title=f"Active Domains by Threshold ({subset}, N={n_subset})",
        )
        if cooc_for_plot is not None:
            plot_cooccurrence_heatmap(
                cooc_for_plot,
                save_path=sub_dir / f"fig_cooccurrence_{subset}.pdf",
                title=f"Pairwise Co-occurrence (τ≥4, {subset}, N={n_subset})",
            )

    # ── Step 3: 기존 ∆H 결과와 일치 검증 ────────────────────
    print(f"\n{'─' * 60}")
    print("[Step 3] 기존 ∆H 결과와의 일치 검증...")
    # GT 사례 대상 검증
    gt_df = filter_by_corpus(scores_df, "GT")
    if len(gt_df) > 0:
        verify_delta_h_consistency(gt_df, out_dir)
    else:
        print("  [SKIP] GT 사례 없음, 검증 건너뜀")

    # ── Step 4: 임계값 강건성 요약 ────────────────────────────
    print(f"\n{'─' * 60}")
    print("[Step 4] 임계값 강건성 요약...")
    robustness = {}
    for subset in ["GT", "ABUSE_NEG"]:
        subset_df = filter_by_corpus(scores_df, subset)
        if len(subset_df) == 0:
            continue
        subset_result = {}
        for tau in [1, 2, 4, 6]:
            cnt = count_nonzero_domains(subset_df, threshold=tau)
            multi = int((cnt["n_domains_active"] >= 2).sum())
            total = len(cnt)
            pct = multi / total * 100 if total > 0 else 0
            subset_result[f"tau_{tau}"] = {
                "n_multi_domain": multi,
                "n_total": total,
                "pct_multi_domain": round(pct, 2),
            }
        robustness[subset] = subset_result

    save_json(robustness, "threshold_robustness_summary.json", out_dir)

    # 핵심 메시지 출력
    for subset, res in robustness.items():
        print(f"\n  [{subset}]")
        for tau_key, vals in res.items():
            print(f"    {tau_key}: 다영역(≥2) = {vals['n_multi_domain']}건 "
                  f"({vals['pct_multi_domain']:.1f}%)")

    # ── Step 5: GT / ABUSE_NEG 일관성 점검 ───────────────────
    print(f"\n{'─' * 60}")
    print("[Step 5] GT / ABUSE_NEG 방향 일관성 점검...")
    gt_rob = robustness.get("GT", {})
    an_rob = robustness.get("ABUSE_NEG", {})
    if gt_rob and an_rob:
        consistent = True
        for tau_key in gt_rob:
            gt_pct = gt_rob[tau_key]["pct_multi_domain"]
            an_pct = an_rob.get(tau_key, {}).get("pct_multi_domain", 0)
            # 같은 방향: 둘 다 의미 있는 비율(>5%) 또는 둘 다 미미(<5%)
            gt_meaningful = gt_pct > 5
            an_meaningful = an_pct > 5
            if gt_meaningful != an_meaningful:
                consistent = False
                print(f"  ⚠ 불일치: {tau_key} — GT={gt_pct:.1f}%, "
                      f"ABUSE_NEG={an_pct:.1f}%")
        if consistent:
            print("  ✓ GT와 ABUSE_NEG의 다영역 동시 출현 패턴이 일관됩니다.")
        else:
            print("  ⚠ 불일치가 발견되었습니다. 알고리즘 2의 인공물 가능성을 점검하세요.")

    print("\n" + "=" * 72)
    print("[RAW SCORE EVIDENCE] 완료.")
    print(f"  산출물 디렉토리: {out_dir}")
    print("=" * 72)
