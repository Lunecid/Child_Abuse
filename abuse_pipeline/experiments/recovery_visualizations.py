"""
recovery_visualizations.py
==========================
Step 2: 학대유형 겹침 시각화 (6개 독립 함수).
"""
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정: 사용 가능한 폰트를 탐색하여 설정
def _setup_korean_font():
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
        r"C:\Windows\Fonts\malgun.ttf",
    ]
    for fp in candidates:
        if fp and os.path.exists(fp):
            font_name = fm.FontProperties(fname=fp).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return
    # Fallback: use DejaVu Sans (no Korean but no warnings)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

from abuse_pipeline.core import common as C

# common import 이후에 폰트를 재설정 (plots.py가 NanumGothic을 설정하므로 덮어쓰기)
_setup_korean_font()

ABUSE_COLORS = C.ABUSE_COLORS
ABUSE_LABEL_EN = C.ABUSE_LABEL_EN


# ═══════════════════════════════════════════════════════════════════
#  1. Co-occurrence Heatmap
# ═══════════════════════════════════════════════════════════════════

def plot_cooccurrence_heatmap(dataset_df: pd.DataFrame, output_dir: str | Path):
    """4×4 학대유형 동시출현 히트맵."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    labels = list(C.ABUSE_ORDER)
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for _, row in dataset_df.iterrows():
        label_set = set(row["label_list"]) if isinstance(row["label_list"], list) else set()
        for a, b in combinations(range(n), 2):
            if labels[a] in label_set and labels[b] in label_set:
                matrix[a, b] += 1
                matrix[b, a] += 1
        # diagonal: total count for each type
        for i in range(n):
            if labels[i] in label_set:
                matrix[i, i] += 1

    en_labels = [ABUSE_LABEL_EN.get(a, a) for a in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(matrix, dtype=bool)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(en_labels, rotation=45, ha="right")
    ax.set_yticklabels(en_labels)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Abuse Type Co-occurrence Matrix", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "cooccurrence_heatmap.png", dpi=200)
    plt.close(fig)
    print(f"  [VIS] cooccurrence_heatmap.png saved")


# ═══════════════════════════════════════════════════════════════════
#  2. UpSet Plot
# ═══════════════════════════════════════════════════════════════════

def plot_upset(dataset_df: pd.DataFrame, output_dir: str | Path):
    """UpSet plot of abuse type combinations."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from upsetplot import UpSet
    except ImportError:
        print("  [VIS] upsetplot 미설치 — UpSet plot 건너뜀 (pip install upsetplot)")
        return

    # Build multi-index
    labels = list(C.ABUSE_ORDER)
    records = []
    for _, row in dataset_df.iterrows():
        label_set = set(row["label_list"]) if isinstance(row["label_list"], list) else set()
        records.append(tuple(a in label_set for a in labels))

    idx = pd.MultiIndex.from_tuples(records, names=[ABUSE_LABEL_EN.get(a, a) for a in labels])
    data = pd.Series(1, index=idx).groupby(level=list(range(len(labels)))).sum()

    try:
        upset = UpSet(data, subset_size="count", show_counts=True, sort_by="cardinality")
        fig = plt.figure(figsize=(10, 6))
        upset.plot(fig=fig)
        fig.savefig(output_dir / "upset_plot.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [VIS] upset_plot.png saved")
    except (ValueError, TypeError) as e:
        # upsetplot versions may have color/scatter compatibility issues
        print(f"  [VIS] UpSet plot 렌더링 실패 (라이브러리 호환성): {e}")
        plt.close("all")

        # Fallback: horizontal bar chart of label combinations
        combo_counts = data.reset_index()
        combo_counts.columns = list(combo_counts.columns[:-1]) + ["count"]
        combo_counts = combo_counts.sort_values("count", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(9, 6))
        combo_labels = []
        for _, r in combo_counts.iterrows():
            active = [ABUSE_LABEL_EN.get(labels[i], labels[i])
                      for i in range(len(labels)) if r.iloc[i]]
            combo_labels.append(" + ".join(active) if active else "(none)")
        ax.barh(range(len(combo_labels)), combo_counts["count"].values, color="#4C72B0")
        ax.set_yticks(range(len(combo_labels)))
        ax.set_yticklabels(combo_labels, fontsize=9)
        ax.set_xlabel("Count")
        ax.set_title("Top Abuse Type Combinations (UpSet fallback)")
        plt.tight_layout()
        fig.savefig(output_dir / "upset_plot_fallback.png", dpi=200)
        plt.close(fig)
        print(f"  [VIS] upset_plot_fallback.png saved (fallback)")


# ═══════════════════════════════════════════════════════════════════
#  3. Jaccard Similarity Matrix
# ═══════════════════════════════════════════════════════════════════

def plot_jaccard_matrix(dataset_df: pd.DataFrame, output_dir: str | Path):
    """학대 유형별 아동 집합 Jaccard index 4×4 히트맵."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    labels = list(C.ABUSE_ORDER)
    n = len(labels)

    # Build child sets per type
    child_sets = {a: set() for a in labels}
    for idx, row in dataset_df.iterrows():
        label_set = set(row["label_list"]) if isinstance(row["label_list"], list) else set()
        for a in labels:
            if a in label_set:
                child_sets[a].add(idx)

    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            inter = len(child_sets[labels[i]] & child_sets[labels[j]])
            union = len(child_sets[labels[i]] | child_sets[labels[j]])
            matrix[i, j] = inter / union if union else 0.0

    en_labels = [ABUSE_LABEL_EN.get(a, a) for a in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(en_labels, rotation=45, ha="right")
    ax.set_yticklabels(en_labels)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Jaccard Similarity Between Abuse Type Child Sets", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "jaccard_matrix.png", dpi=200)
    plt.close(fig)
    print(f"  [VIS] jaccard_matrix.png saved")


# ═══════════════════════════════════════════════════════════════════
#  4. Network Graph
# ═══════════════════════════════════════════════════════════════════

def plot_network_graph(dataset_df: pd.DataFrame, output_dir: str | Path):
    """학대 유형 동시출현 네트워크 그래프."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        import networkx as nx
    except ImportError:
        print("  [VIS] networkx 미설치 — 네트워크 그래프 건너뜀")
        return

    labels = list(C.ABUSE_ORDER)
    G = nx.Graph()

    # Node counts
    node_counts = {a: 0 for a in labels}
    edge_counts = {}
    for _, row in dataset_df.iterrows():
        label_set = set(row["label_list"]) if isinstance(row["label_list"], list) else set()
        for a in labels:
            if a in label_set:
                node_counts[a] += 1
        for a, b in combinations(labels, 2):
            if a in label_set and b in label_set:
                key = tuple(sorted([a, b]))
                edge_counts[key] = edge_counts.get(key, 0) + 1

    for a in labels:
        en = ABUSE_LABEL_EN.get(a, a)
        G.add_node(en, count=node_counts[a])

    max_edge = max(edge_counts.values()) if edge_counts else 1
    for (a, b), cnt in edge_counts.items():
        G.add_edge(ABUSE_LABEL_EN.get(a, a), ABUSE_LABEL_EN.get(b, b), weight=cnt)

    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, seed=42, k=2.0)

    node_sizes = [G.nodes[n]["count"] * 8 for n in G.nodes()]
    node_colors = [ABUSE_COLORS.get(
        [k for k, v in ABUSE_LABEL_EN.items() if v == n][0] if any(v == n for v in ABUSE_LABEL_EN.values()) else "",
        "#999"
    ) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    edges = G.edges(data=True)
    widths = [e[2]["weight"] / max_edge * 8 for e in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.5, edge_color="gray")

    # Edge labels
    edge_labels = {(e[0], e[1]): str(e[2]["weight"]) for e in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    ax.set_title("Abuse Type Co-occurrence Network", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "network_graph.png", dpi=200)
    plt.close(fig)
    print(f"  [VIS] network_graph.png saved")


# ═══════════════════════════════════════════════════════════════════
#  5. CA Biplot (기존 인프라 활용)
# ═══════════════════════════════════════════════════════════════════

def plot_ca_biplot(dataset_df: pd.DataFrame, output_dir: str | Path):
    """CA biplot with bridge word overlay (기존 ca.py 활용)."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Build document-level word×abuse frequency table from dataset_df
    labels = list(C.ABUSE_ORDER)
    word_abuse_rows = []
    for _, row in dataset_df.iterrows():
        gt_main = row["gt_main"]
        tokens = set(str(row["text"]).split())
        for tok in tokens:
            word_abuse_rows.append({"word": tok, "abuse_type": gt_main})

    if not word_abuse_rows:
        print("  [VIS] CA biplot: no data")
        return

    raw = pd.DataFrame(word_abuse_rows)
    doc_counts = raw.groupby(["word", "abuse_type"]).size().unstack(fill_value=0)
    for a in labels:
        if a not in doc_counts.columns:
            doc_counts[a] = 0
    doc_counts = doc_counts[labels]
    doc_counts["total"] = doc_counts.sum(axis=1)
    doc_counts = doc_counts[doc_counts["total"] >= C.MIN_DOC_COUNT]

    if doc_counts.empty:
        print("  [VIS] CA biplot: insufficient data after filtering")
        return

    try:
        from abuse_pipeline.stats.stats import compute_chi_square, compute_log_odds
        from abuse_pipeline.stats.ca import run_abuse_ca_with_prob_bridges

        # Compute stats needed by CA function
        chi_df = compute_chi_square(doc_counts[labels], labels)
        log_odds_df = compute_log_odds(doc_counts[labels], labels)

        # Save temporarily to get CA output
        prev_dir = C.CA_PROB_DIR
        C.CA_PROB_DIR = str(output_dir)

        run_abuse_ca_with_prob_bridges(
            df_abuse_counts=doc_counts[labels],
            abuse_stats_logodds=log_odds_df,
            abuse_stats_chi=chi_df,
            top_chi_for_ca=200,
            df_abuse_counts_doc=doc_counts[labels],
            abuse_stats_logodds_doc=log_odds_df,
        )

        C.CA_PROB_DIR = prev_dir
        print(f"  [VIS] CA biplot saved to {output_dir}")

    except Exception as e:
        print(f"  [VIS] CA biplot 실패: {e}")


# ═══════════════════════════════════════════════════════════════════
#  6. Bridge Word Overlay (CA 위에 별도 마커)
# ═══════════════════════════════════════════════════════════════════

def plot_bridge_word_overlay(
    dataset_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    output_dir: str | Path,
):
    """
    별도의 브릿지 워드 scatter plot.
    bridge_df가 비어있으면 건너뜀.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if bridge_df is None or bridge_df.empty:
        print("  [VIS] bridge overlay: no bridge words")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for _, brow in bridge_df.iterrows():
        pa = brow.get("primary_abuse", "")
        sa = brow.get("secondary_abuse", "")
        p1 = float(brow.get("p1", 0))
        p2 = float(brow.get("p2", 0))
        word = brow.get("word", "")

        color1 = ABUSE_COLORS.get(pa, "#999")
        color2 = ABUSE_COLORS.get(sa, "#ccc")

        ax.scatter(p1, p2, s=60, c=color1, edgecolors=color2, linewidths=2, zorder=3)
        ax.annotate(word, (p1, p2), fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("P(primary | word)", fontsize=11)
    ax.set_ylabel("P(secondary | word)", fontsize=11)
    ax.set_title("Bridge Words: Primary vs Secondary Probability", fontsize=12)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=ABUSE_LABEL_EN.get(a, a))
        for a, c in ABUSE_COLORS.items()
    ]
    ax.legend(handles=handles, title="Primary type", loc="upper right")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "bridge_word_overlay.png", dpi=200)
    plt.close(fig)
    print(f"  [VIS] bridge_word_overlay.png saved")
