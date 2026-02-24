"""
bert_abuse_coloring.py
=======================
BERT 단어 임베딩 2D 좌표에 학대유형별 색상을 입혀,
"성학대 vs. 나머지" 이분 구조가 BERT 의미 공간에서도 재현되는지 검증.

기존 procrustes_CA_words_vs_BERT_words.png에서 관찰된
두 개의 연속체(continuum)가 실제로 성학대 vs. {방임, 정서학대, 신체학대}의
분리인지를 시각적 · 통계적으로 입증한다.

사용법 (파이프라인 통합)
------------------------
    from bert_abuse_coloring import plot_bert_words_by_abuse_type

    plot_bert_words_by_abuse_type(
        bert_word_df=bert_word_df,          # BERT PCA 2D 좌표
        bary_df=bary_df,                    # CA barycentric 좌표
        word_main_abuse=word_main_abuse,    # {word: abuse_type} 매핑
        abuse_stats_logodds=abuse_stats_logodds,
        out_dir="revision_output/bert_ca_validation",
    )
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# ── 색상 / 라벨 설정 (common.py와 동일) ──
ABUSE_ORDER = ["방임", "정서학대", "신체학대", "성학대"]
ABUSE_COLORS = {
    "방임": "#1f77b4",  # 파란색
    "정서학대": "#ff7f0e",  # 주황색
    "신체학대": "#2ca02c",  # 초록색
    "성학대": "#d62728",  # 빨간색
}
ABUSE_LABEL_EN = {
    "방임": "Neglect",
    "정서학대": "Emotional",
    "신체학대": "Physical",
    "성학대": "Sexual",
}

# 이분 구조 색상 (성학대 vs 나머지)
BINARY_COLORS = {
    "Sexual abuse": "#d62728",  # 빨간색
    "Other maltreatment": "#1f77b4",  # 파란색
}


# ============================================================
#  핵심 함수 1: BERT 단어 좌표에 학대유형 색상 입히기
# ============================================================

def plot_bert_words_by_abuse_type(
        bert_word_df: pd.DataFrame,
        bary_df: pd.DataFrame,
        word_main_abuse: dict,
        abuse_stats_logodds: pd.DataFrame = None,
        out_dir: str = None,
        n_perm: int = 999,
) -> dict:
    """
    BERT 단어 임베딩 2D 좌표 각 점을 학대유형별 색상으로 착색하여,
    "성학대 단어 클러스터 vs. 나머지 학대유형 단어 클러스터"의
    이분 구조를 시각적·통계적으로 검증한다.

    Parameters
    ----------
    bert_word_df : pd.DataFrame
        index=word, columns=[Dim1_bert_word, Dim2_bert_word]
        (contextual_embedding_ca.compute_bert_word_embeddings 결과)
    bary_df : pd.DataFrame
        index=word, columns=[Dim1_bary, Dim2_bary]
        (ca.py의 CA barycentric 좌표)
    word_main_abuse : dict
        {word: abuse_type} 매핑
        (abuse_stats_logodds에서 각 단어의 log odds가 가장 높은 학대유형)
    abuse_stats_logodds : pd.DataFrame, optional
        log odds 통계 (추가 분석용)
    out_dir : str
        결과 저장 디렉토리
    n_perm : int
        permutation test 반복 횟수

    Returns
    -------
    dict: 통계 검증 결과
    """
    from scipy.spatial.distance import pdist, squareform, cdist
    from scipy.stats import spearmanr, mannwhitneyu, ttest_ind

    # ── 1. 공통 단어 & Procrustes 정렬 (기존 로직 재사용) ──
    common_words = sorted(set(bary_df.index) & set(bert_word_df.index))
    n = len(common_words)
    print(f"[BERT-COLOR] 공통 단어 수: {n}")

    X = bary_df.loc[common_words, ["Dim1_bary", "Dim2_bary"]].values.astype(float)
    Y = bert_word_df.loc[common_words, ["Dim1_bert_word", "Dim2_bert_word"]].values.astype(float)

    # Procrustes 정렬
    X0 = X - X.mean(0, keepdims=True)
    Y0 = Y - Y.mean(0, keepdims=True)
    nX = np.linalg.norm(X0, "fro")
    nY = np.linalg.norm(Y0, "fro")
    X0 /= nX
    Y0 /= nY
    M = X0.T @ Y0
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    Y_aligned = Y0 @ R.T

    # ── 2. 각 단어에 학대유형 배정 ──
    word_colors_4type = []  # 4유형 색상
    word_colors_binary = []  # 이분법 색상
    word_labels = []  # 학대유형명
    word_binary = []  # "Sexual" or "Other"
    unassigned = 0

    for w in common_words:
        abuse = word_main_abuse.get(w)
        if abuse in ABUSE_COLORS:
            word_colors_4type.append(ABUSE_COLORS[abuse])
            word_labels.append(abuse)
            if abuse == "성학대":
                word_colors_binary.append(BINARY_COLORS["Sexual abuse"])
                word_binary.append("Sexual abuse")
            else:
                word_colors_binary.append(BINARY_COLORS["Other maltreatment"])
                word_binary.append("Other maltreatment")
        else:
            word_colors_4type.append("#999999")
            word_colors_binary.append("#999999")
            word_labels.append("unassigned")
            word_binary.append("unassigned")
            unassigned += 1

    print(f"[BERT-COLOR] 학대유형 배정: {n - unassigned}/{n} 단어")
    for abuse in ABUSE_ORDER:
        cnt = sum(1 for l in word_labels if l == abuse)
        print(f"  {ABUSE_LABEL_EN.get(abuse, abuse)}: {cnt} words")

    word_binary = np.array(word_binary)

    # ── 3. 통계 검증: 성학대 vs 나머지 거리 분리 ──
    sexual_mask = word_binary == "Sexual abuse"
    other_mask = word_binary == "Other maltreatment"
    n_sexual = sexual_mask.sum()
    n_other = other_mask.sum()

    results = {
        "n_total": n,
        "n_sexual": int(n_sexual),
        "n_other": int(n_other),
        "n_unassigned": int(unassigned),
    }

    if n_sexual >= 3 and n_other >= 3:
        # BERT 공간에서의 centroid
        bert_sexual_centroid = Y_aligned[sexual_mask].mean(axis=0)
        bert_other_centroid = Y_aligned[other_mask].mean(axis=0)

        # 그룹 내 거리 vs 그룹 간 거리
        within_sexual = pdist(Y_aligned[sexual_mask])
        within_other = pdist(Y_aligned[other_mask])
        between = cdist(Y_aligned[sexual_mask], Y_aligned[other_mask]).ravel()

        within_all = np.concatenate([within_sexual, within_other])

        # Mann-Whitney U test: 그룹 간 거리가 그룹 내 거리보다 유의하게 큰가?
        u_stat, u_p = mannwhitneyu(between, within_all, alternative="greater")

        # 효과크기 (Cohen's d)
        d_between = between.mean()
        d_within = within_all.mean()
        pooled_std = np.sqrt(
            (between.var() * len(between) + within_all.var() * len(within_all))
            / (len(between) + len(within_all))
        )
        cohens_d = (d_between - d_within) / pooled_std if pooled_std > 0 else 0

        # Centroid 간 거리
        centroid_dist = np.linalg.norm(bert_sexual_centroid - bert_other_centroid)

        results.update({
            "mean_within_distance": float(d_within),
            "mean_between_distance": float(d_between),
            "centroid_distance": float(centroid_dist),
            "mann_whitney_U": float(u_stat),
            "mann_whitney_p": float(u_p),
            "cohens_d": float(cohens_d),
        })

        print(f"\n[BERT-COLOR] ── 이분 구조 검증 ──")
        print(f"  성학대 단어: {n_sexual}개, 나머지: {n_other}개")
        print(f"  그룹 내 평균 거리: {d_within:.4f}")
        print(f"  그룹 간 평균 거리: {d_between:.4f}")
        print(f"  Centroid 간 거리:  {centroid_dist:.4f}")
        print(f"  Mann-Whitney U={u_stat:.1f}, p={u_p:.6f}")
        print(f"  Cohen's d={cohens_d:.3f}")

        # Permutation test for centroid distance
        rng = np.random.default_rng(2025)
        perm_dists = []
        for _ in range(n_perm):
            perm_labels = rng.permutation(word_binary)
            perm_sex = perm_labels == "Sexual abuse"
            perm_oth = perm_labels == "Other maltreatment"
            if perm_sex.sum() > 0 and perm_oth.sum() > 0:
                c1 = Y_aligned[perm_sex].mean(axis=0)
                c2 = Y_aligned[perm_oth].mean(axis=0)
                perm_dists.append(np.linalg.norm(c1 - c2))

        perm_dists = np.array(perm_dists)
        perm_p = float((1 + np.sum(perm_dists >= centroid_dist)) / (len(perm_dists) + 1))

        results["centroid_perm_p"] = perm_p
        print(f"  Centroid permutation p={perm_p:.4f} (n_perm={n_perm})")

    # ── 4. 시각화 ──
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        # ===== Figure 1: BERT 4유형 색상 (Procrustes 정렬 후) =====
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # (A) BERT 단어 좌표 — 4유형 색상
        ax = axes[0]
        for abuse in ABUSE_ORDER:
            mask = np.array([l == abuse for l in word_labels])
            if mask.sum() == 0:
                continue
            ax.scatter(
                Y_aligned[mask, 0], Y_aligned[mask, 1],
                s=25, alpha=0.65,
                color=ABUSE_COLORS[abuse],
                label=f"{ABUSE_LABEL_EN[abuse]} (n={mask.sum()})",
                zorder=3,
            )
        # 미배정 단어
        mask_un = np.array([l == "unassigned" for l in word_labels])
        if mask_un.sum() > 0:
            ax.scatter(
                Y_aligned[mask_un, 0], Y_aligned[mask_un, 1],
                s=15, alpha=0.3, color="#999999", label=f"Unassigned (n={mask_un.sum()})",
                zorder=1,
            )

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_xlabel("Dim 1 (Procrustes-aligned)", fontsize=11)
        ax.set_ylabel("Dim 2 (Procrustes-aligned)", fontsize=11)
        ax.set_title(
            "(A) BERT word embeddings colored by\n"
            "primary maltreatment type (log-odds assignment)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.15)

        # (B) 이분법: 성학대 vs 나머지
        ax = axes[1]
        for group_name, color in BINARY_COLORS.items():
            mask = word_binary == group_name
            if mask.sum() == 0:
                continue
            ax.scatter(
                Y_aligned[mask, 0], Y_aligned[mask, 1],
                s=25, alpha=0.65, color=color,
                label=f"{group_name} (n={mask.sum()})",
                zorder=3,
            )

        # Centroids 표시
        if n_sexual >= 3 and n_other >= 3:
            ax.scatter(
                [bert_sexual_centroid[0]], [bert_sexual_centroid[1]],
                s=300, marker="*", color=BINARY_COLORS["Sexual abuse"],
                edgecolor="black", linewidth=1.5, zorder=10,
                label="Sexual centroid",
            )
            ax.scatter(
                [bert_other_centroid[0]], [bert_other_centroid[1]],
                s=300, marker="*", color=BINARY_COLORS["Other maltreatment"],
                edgecolor="black", linewidth=1.5, zorder=10,
                label="Other centroid",
            )
            # Centroid 연결선
            ax.plot(
                [bert_sexual_centroid[0], bert_other_centroid[0]],
                [bert_sexual_centroid[1], bert_other_centroid[1]],
                color="black", linewidth=1.5, linestyle="--", alpha=0.5, zorder=5,
            )
            mid_x = (bert_sexual_centroid[0] + bert_other_centroid[0]) / 2
            mid_y = (bert_sexual_centroid[1] + bert_other_centroid[1]) / 2
            ax.annotate(
                f"d={centroid_dist:.3f}\np={perm_p:.4f}",
                (mid_x, mid_y), fontsize=9, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_xlabel("Dim 1 (Procrustes-aligned)", fontsize=11)
        ax.set_ylabel("Dim 2 (Procrustes-aligned)", fontsize=11)
        ax.set_title(
            "(B) Binary structure: Sexual abuse vs. Other types\n"
            f"Mann-Whitney p={results.get('mann_whitney_p', float('nan')):.4f}, "
            f"Cohen's d={results.get('cohens_d', float('nan')):.3f}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.15)

        fig.suptitle(
            "BERT Contextual Word Embeddings: Maltreatment Type Structure",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig_path = os.path.join(out_dir, "bert_words_abuse_type_coloring.png")
        fig.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[저장] BERT 단어 학대유형 착색 → {fig_path}")

        # ===== Figure 2: CA vs BERT 나란히 (동일 색상) =====
        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))

        # (A) CA barycentric 좌표 — 4유형 색상
        ax = axes2[0]
        for abuse in ABUSE_ORDER:
            mask = np.array([l == abuse for l in word_labels])
            if mask.sum() == 0:
                continue
            ax.scatter(
                X0[mask, 0], X0[mask, 1],
                s=25, alpha=0.65, color=ABUSE_COLORS[abuse],
                label=f"{ABUSE_LABEL_EN[abuse]} (n={mask.sum()})",
                zorder=3,
            )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_xlabel("Dim 1", fontsize=11)
        ax.set_ylabel("Dim 2", fontsize=11)
        ax.set_title(
            "(A) CA barycentric coordinates\n(token frequency-based)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.15)

        # (B) BERT 단어 좌표 — 4유형 색상 (동일 스케일)
        ax = axes2[1]
        for abuse in ABUSE_ORDER:
            mask = np.array([l == abuse for l in word_labels])
            if mask.sum() == 0:
                continue
            ax.scatter(
                Y_aligned[mask, 0], Y_aligned[mask, 1],
                s=25, alpha=0.65, color=ABUSE_COLORS[abuse],
                label=f"{ABUSE_LABEL_EN[abuse]} (n={mask.sum()})",
                zorder=3,
            )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_xlabel("Dim 1 (Procrustes-aligned)", fontsize=11)
        ax.set_ylabel("Dim 2 (Procrustes-aligned)", fontsize=11)
        ax.set_title(
            "(B) BERT word embeddings\n(contextual meaning-based)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.15)

        fig2.suptitle(
            "Token-Frequency CA vs. BERT Contextual Embeddings\n"
            "— Same words, same color assignment (log-odds primary type) —",
            fontsize=14, fontweight="bold", y=1.04,
        )
        fig2.tight_layout()
        fig2_path = os.path.join(out_dir, "CA_vs_BERT_words_abuse_colored.png")
        fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")
        plt.close(fig2)
        print(f"[저장] CA vs BERT 학대유형 착색 비교 → {fig2_path}")

        # ===== Figure 3: 거리 분포 비교 (boxplot) =====
        if n_sexual >= 3 and n_other >= 3:
            fig3, ax3 = plt.subplots(figsize=(8, 5))

            box_data = [within_sexual, within_other, between]
            box_labels = [
                f"Within Sexual\n(n={len(within_sexual)})",
                f"Within Other\n(n={len(within_other)})",
                f"Between groups\n(n={len(between)})",
            ]
            box_colors = ["#d62728", "#1f77b4", "#7f7f7f"]

            bp = ax3.boxplot(
                box_data, labels=box_labels, patch_artist=True,
                showmeans=True, meanline=True,
                medianprops=dict(color="black", linewidth=1.5),
                meanprops=dict(color="darkred", linewidth=1.5, linestyle="--"),
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            ax3.set_ylabel("Pairwise Euclidean distance", fontsize=11)
            ax3.set_title(
                "Distance Distribution: Sexual vs. Other Maltreatment Words\n"
                f"(BERT embedding space, Procrustes-aligned)\n"
                f"Mann-Whitney U={results['mann_whitney_U']:.0f}, "
                f"p={results['mann_whitney_p']:.2e}, "
                f"Cohen's d={results['cohens_d']:.3f}",
                fontsize=11, fontweight="bold",
            )
            ax3.grid(True, alpha=0.2, axis="y")
            fig3.tight_layout()
            fig3_path = os.path.join(out_dir, "bert_distance_boxplot_sexual_vs_other.png")
            fig3.savefig(fig3_path, dpi=200)
            plt.close(fig3)
            print(f"[저장] 거리 분포 비교 → {fig3_path}")

        # ===== 통계 결과 CSV 저장 =====
        results_df = pd.DataFrame([results])
        csv_path = os.path.join(out_dir, "bert_binary_structure_test.csv")
        results_df.to_csv(csv_path, encoding="utf-8-sig", index=False)
        print(f"[저장] 이분 구조 검증 통계 → {csv_path}")

    return results


# ============================================================
#  파이프라인 통합 가이드
# ============================================================

INTEGRATION_GUIDE = """
# ============================================================
# BERT 학대유형 착색 시각화 — 파이프라인 통합 (1곳 수정)
# ============================================================
#
# contextual_embedding_ca.py의 run_bert_ca_validation() 함수
# Analysis D 이후에 다음 코드를 추가:
#
#   # ── Analysis F: BERT 단어 학대유형 착색 (이분 구조 검증) ──
#   if bert_word_df is not None and bary_df is not None:
#       from bert_abuse_coloring import plot_bert_words_by_abuse_type
#
#       # word_main_abuse 생성 (CA에서 사용한 동일한 로그 오즈 기반 매핑)
#       if abuse_stats_logodds is not None:
#           word_main_abuse = (
#               abuse_stats_logodds
#               .sort_values("log_odds", ascending=False)
#               .groupby("word")["group"]
#               .first()
#               .to_dict()
#           )
#       else:
#           word_main_abuse = {}
#
#       binary_results = plot_bert_words_by_abuse_type(
#           bert_word_df=bert_word_df,
#           bary_df=bary_df,
#           word_main_abuse=word_main_abuse,
#           abuse_stats_logodds=abuse_stats_logodds,
#           out_dir=out_dir,
#       )
#       results["binary_structure"] = binary_results
"""

if __name__ == "__main__":
    print("=" * 72)
    print("BERT Word Abuse-Type Coloring Module")
    print("=" * 72)
    print()
    print("이 모듈은 독립 실행용이 아닙니다.")
    print("파이프라인에서 import하여 사용하세요.")
    print()
    print("=== 통합 가이드 ===")
    print(INTEGRATION_GUIDE)