from __future__ import annotations


import os
import json
import itertools

import numpy as np
import pandas as pd

from abuse_pipeline.core import common as C
from abuse_pipeline.stats.stats import compute_prob_bridge_for_words, compute_bridge_words
from abuse_pipeline.core import plots

# ---------------------------------------------------------------------------
# optional dependencies (prince / scipy / adjustText)
# ---------------------------------------------------------------------------
try:
    import prince  # type: ignore
except Exception:
    prince = None

if getattr(C, "HAS_SCIPY", False):
    try:
        from scipy.stats import chi2  # type: ignore
    except Exception:
        chi2 = None
else:
    chi2 = None

if getattr(C, "HAS_ADJUSTTEXT", False):
    try:
        from adjustText import adjust_text  # type: ignore
    except Exception:
        adjust_text = None
else:
    adjust_text = None

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# CA artifact dump
# ---------------------------------------------------------------------------
def dump_ca_artifacts(
    row_coords_2d: pd.DataFrame,
    bary_df: pd.DataFrame,
    word_main_abuse: dict,
    out_dir: str,
):
    """
    CA separability가 요구하는 3개 파일을 저장:
      - row_coords_2d.csv    (index=abuse, columns: Dim1, Dim2)
      - bary_df.csv          (index=word,  columns: Dim1_bary, Dim2_bary)
      - word_main_abuse.json (word -> main_abuse)
    """
    os.makedirs(out_dir, exist_ok=True)

    if not {"Dim1", "Dim2"}.issubset(set(row_coords_2d.columns)):
        raise ValueError(
            f"row_coords_2d에 Dim1/Dim2 컬럼이 없습니다. columns={list(row_coords_2d.columns)}"
        )

    if not {"Dim1_bary", "Dim2_bary"}.issubset(set(bary_df.columns)):
        raise ValueError(
            f"bary_df에 Dim1_bary/Dim2_bary 컬럼이 없습니다. columns={list(bary_df.columns)}"
        )

    row_fp = os.path.join(out_dir, "row_coords_2d.csv")
    bary_fp = os.path.join(out_dir, "bary_df.csv")
    map_fp = os.path.join(out_dir, "word_main_abuse.json")

    row_coords_2d[["Dim1", "Dim2"]].to_csv(row_fp, encoding="utf-8-sig")
    bary_df[["Dim1_bary", "Dim2_bary"]].to_csv(bary_fp, encoding="utf-8-sig")

    with open(map_fp, "w", encoding="utf-8") as f:
        json.dump(word_main_abuse, f, ensure_ascii=False, indent=2)

    print(f"[CA-DUMP] saved: {row_fp}")
    print(f"[CA-DUMP] saved: {bary_fp}")
    print(f"[CA-DUMP] saved: {map_fp}")


# ---------------------------------------------------------------------------
# geometry helper: convex hull (no scipy dependency)
# ---------------------------------------------------------------------------
def _convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Monotonic chain convex hull.
    points: (n, 2)  →  return hull points ordered (m, 2). if n < 3, returns points.
    """
    pts = np.unique(points.astype(float), axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


# ---------------------------------------------------------------------------
# Main CA + prob-bridge ablation
# ---------------------------------------------------------------------------
def run_abuse_ca_with_prob_bridges(
    df_abuse_counts: pd.DataFrame,
    abuse_stats_logodds: pd.DataFrame,
    abuse_stats_chi: pd.DataFrame,
    top_chi_for_ca: int = 200,
    top_log_per_abuse: int = 15,
    top_freq_per_abuse: int = 20,
    bridge_filter_configs=None,
    *,
    # ★ 추가: 문서 수(doc-level) 기반 교량 단어 판정용
    df_abuse_counts_doc: pd.DataFrame | None = None,
    abuse_stats_logodds_doc: pd.DataFrame | None = None,
    abuse_stats_chi_doc: pd.DataFrame | None = None,
    lang: str = "en",
    display_mode: str = "en_only",          # "en_only" or "en_ko"
    translate_title: bool = False,           # unused here but kept for compat
    plot_words_max: int = 140,               # word label 폭발 방지
    emphasize_clusters: bool = True,
):
    """
    CA biplot + 확률 기반 교량 단어(bridge word) ablation.

    데이터 흐름 (논문 정합성)
    -------------------------
    ┌─ CA 자체 (축 좌표 · 단어 위치) ──────┐
    │  df_abuse_counts   (토큰 수 기반)     │  ← CA가 빈도 구조를 반영
    │  abuse_stats_chi   (토큰 수 기반)     │
    └───────────────────────────────────────┘
    ┌─ 교량 단어 판정 (색칠) ───────────────┐
    │  df_abuse_counts_doc  (문서 수 기반)   │  ← 논문 §3 정의: d_{w,k} 기반
    │  abuse_stats_logodds_doc              │
    └───────────────────────────────────────┘

    df_abuse_counts_doc 가 None이면 토큰 수 기반으로 fallback (하위 호환).

    논문 표기와의 대응
    ------------------
    - k1 : 주 연관 유형 (primary_abuse)   — 논문의 $k_1$
    - k2 : 부 연관 유형 (secondary_abuse) — 논문의 $k_2$
    - 교량 단어 플롯: 면 색(facecolor) = k1 색상, 테두리 색(edgecolor) = k2 색상
    """
    # ----------- guards -----------
    if not getattr(C, "HAS_PRINCE", False):
        print("[WARN] prince not installed -> skip abuse CA(prob-bridge).")
        return None, None
    if df_abuse_counts is None or df_abuse_counts.empty or abuse_stats_chi is None or abuse_stats_chi.empty:
        print("[WARN] df_abuse_counts or abuse_stats_chi empty -> skip CA(prob-bridge).")
        return None, None
    if C.CA_PROB_DIR is None:
        raise RuntimeError("C.CA_PROB_DIR is None. Call C.configure_output_dirs(...) before running pipeline.")

    if bridge_filter_configs is None:
        bridge_filter_configs = C.BRIDGE_FILTER_CONFIGS

    # ★ 추가: 교량 판정에 사용할 데이터 선택 (doc-level 우선)
    if df_abuse_counts_doc is not None and not df_abuse_counts_doc.empty:
        _bridge_counts = df_abuse_counts_doc
        _bridge_logodds = (
            abuse_stats_logodds_doc
            if (abuse_stats_logodds_doc is not None and not abuse_stats_logodds_doc.empty)
            else abuse_stats_logodds
        )
        _bridge_src = "doc-level"
    else:
        _bridge_counts = df_abuse_counts
        _bridge_logodds = abuse_stats_logodds
        _bridge_src = "token-level (fallback)"

    print(f"[CA] bridge detection source: {_bridge_src}")

    # ----------- translator (KO->EN for word labels) -----------
    tt = C.TokenTranslator(C.META_DIR, use_auto=True) if (lang == "en") else None

    # -----------------------------------------------------------------
    # 1) CA with chi-square top words  (토큰 수 기반 — CA 축 산출)
    # -----------------------------------------------------------------
    if (abuse_stats_chi_doc is not None and not abuse_stats_chi_doc.empty
            and df_abuse_counts_doc is not None and not df_abuse_counts_doc.empty):
        _chi_for_ca = abuse_stats_chi_doc.sort_values("chi2", ascending=False, kind="mergesort")
        ca_words = _chi_for_ca.head(min(top_chi_for_ca, len(_chi_for_ca))).index
        # doc-level 빈도표에 존재하는 단어만
        ca_words = [w for w in ca_words if w in df_abuse_counts_doc.index]
        df_ca = df_abuse_counts_doc.loc[ca_words]
        print(f"[CA] using doc-level counts for CA axis ({len(ca_words)} words)")
    else:
        chi_sorted = abuse_stats_chi.sort_values("chi2", ascending=False, kind="mergesort")
        ca_words = chi_sorted.head(min(top_chi_for_ca, len(chi_sorted))).index
        df_ca = df_abuse_counts.loc[ca_words]
        print(f"[CA] using token-level counts for CA axis (fallback, {len(ca_words)} words)")

    # rows = abuse types, cols = words
    X = df_ca.T

    ca = C.prince.CA(
        n_components=2,
        n_iter=10,
        copy=True,
        check_input=True,
        random_state=42,
    ).fit(X)

    row_coords = ca.row_coordinates(X)
    row_coords_2d = row_coords.iloc[:, :2].copy()
    row_coords_2d.columns = ["Dim1", "Dim2"]

    # -----------------------------------------------------------------
    # global chi-square / inertia summary
    # -----------------------------------------------------------------
    obs = X.values
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    N = float(obs.sum())
    expected = row_sums @ col_sums / max(N, 1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        chi_sq = (obs - expected) ** 2 / expected
        chi_sq = np.nan_to_num(chi_sq, nan=0.0, posinf=0.0, neginf=0.0)

    chi_total = float(chi_sq.sum())
    r_ca, c_ca = obs.shape
    df_ca_val = int((r_ca - 1) * (c_ca - 1))

    if getattr(C, "HAS_SCIPY", False):
        try:
            p_ca = float(C.chi2.sf(chi_total, df=df_ca_val))
        except Exception:
            p_ca = np.nan
    else:
        p_ca = np.nan

    total_inertia = getattr(ca, "total_inertia_", np.nan)

    os.makedirs(C.CA_PROB_DIR, exist_ok=True)
    ca_summary = pd.DataFrame({
        "N": [N],
        "n_rows": [r_ca],
        "n_cols": [c_ca],
        "chi2": [chi_total],
        "df": [df_ca_val],
        "p_value": [p_ca],
        "total_inertia": [total_inertia],
    })
    ca_summary_path = os.path.join(C.CA_PROB_DIR, "abuse_CA_summary_probBridge_childtokens.csv")
    ca_summary.to_csv(ca_summary_path, encoding="utf-8-sig", index=False)
    print(f"[SAVE] CA summary (prob-bridge) -> {ca_summary_path}")

    # -----------------------------------------------------------------
    # eigenvalues / explained inertia
    # -----------------------------------------------------------------
    if hasattr(ca, "eigenvalues_"):
        eig = np.array(ca.eigenvalues_, dtype=float).ravel()
        dims = np.arange(1, len(eig) + 1)
    else:
        eig = np.array([], dtype=float)
        dims = np.array([], dtype=int)

    if hasattr(ca, "explained_inertia_"):
        explained = np.array(ca.explained_inertia_, dtype=float).ravel()
    else:
        explained = np.full_like(eig, np.nan, dtype=float)

    if eig.size > 0:
        explained_pct = explained * 100.0
        df_eig = pd.DataFrame({
            "dim": dims,
            "eigenvalue": eig,
            "explained_inertia": explained,
            "explained_inertia_pct": explained_pct,
        })
        eig_path = os.path.join(C.CA_PROB_DIR, "abuse_CA_eigenvalues_probBridge_childtokens.csv")
        df_eig.to_csv(eig_path, encoding="utf-8-sig", index=False)
        print(f"[SAVE] CA eigenvalues/explained (prob-bridge) -> {eig_path}")
    else:
        print("[INFO] CA eigenvalues_ not available (prob-bridge).")

    # -----------------------------------------------------------------
    # row coords save
    # -----------------------------------------------------------------
    row_out = row_coords_2d.copy()
    row_out.index.name = "abuse"
    row_coords_path = os.path.join(C.CA_PROB_DIR, "abuse_CA_rows_probBridge_childtokens.csv")
    row_out.to_csv(row_coords_path, encoding="utf-8-sig")
    print(f"[SAVE] CA row coords (prob-bridge) -> {row_coords_path}")

    # -----------------------------------------------------------------
    # 2) barycentric word coordinates  (토큰 수 기반 — CA 좌표계)
    # -----------------------------------------------------------------
    bary_list = []
    for w in df_ca.index:
        counts = df_ca.loc[w, C.ABUSE_ORDER].values.astype(float)
        tot = float(counts.sum())
        if tot <= 0:
            bx, by = 0.0, 0.0
        else:
            probs = counts / tot
            bx = float(np.dot(probs, row_coords_2d.loc[C.ABUSE_ORDER, "Dim1"].values))
            by = float(np.dot(probs, row_coords_2d.loc[C.ABUSE_ORDER, "Dim2"].values))
        bary_list.append((w, bx, by))

    bary_df = pd.DataFrame(bary_list, columns=["word", "Dim1_bary", "Dim2_bary"]).set_index("word")
    bary_df.index.name = "word"

    bary_coords_path = os.path.join(C.CA_PROB_DIR, "abuse_CA_words_barycentric_probBridge_childtokens.csv")
    bary_df.to_csv(bary_coords_path, encoding="utf-8-sig")
    print(f"[SAVE] CA word coords (barycentric) -> {bary_coords_path}")

    # -----------------------------------------------------------------
    # main-abuse mapping for word color
    # ★ 변경: _bridge_logodds (doc-level) 기반
    # -----------------------------------------------------------------
    if _bridge_logodds is not None and (not _bridge_logodds.empty):
        word_main_abuse = (
            _bridge_logodds
            .sort_values("log_odds", ascending=False)
            .groupby("word")["group"]
            .first()
            .to_dict()
        )
    else:
        word_main_abuse = {}

    # -----------------------------------------------------------------
    # select words (logodds top + freq top)
    # ★ 변경: logodds 기준은 _bridge_logodds (doc-level)
    # -----------------------------------------------------------------
    selected_words = []
    for abuse_name in C.ABUSE_ORDER:
        if abuse_name not in df_ca.columns:
            continue

        sub_stats = _bridge_logodds[
            (_bridge_logodds["group"] == abuse_name) &
            (_bridge_logodds["word"].isin(df_ca.index))
        ]
        top_log_words = (
            sub_stats.sort_values("log_odds", ascending=False)
            .head(top_log_per_abuse)["word"]
            .tolist()
        )

        top_freq_words = (
            df_ca[abuse_name]
            .sort_values(ascending=False)
            .head(top_freq_per_abuse)
            .index
            .tolist()
        )

        selected_words.extend(list(set(top_log_words) | set(top_freq_words)))

    selected_words = sorted(set(w for w in selected_words if w in bary_df.index))

    # -----------------------------------------------------------------
    # plot bounds
    # -----------------------------------------------------------------
    xs_all = list(row_coords_2d["Dim1"]) + list(bary_df["Dim1_bary"])
    ys_all = list(row_coords_2d["Dim2"]) + list(bary_df["Dim2_bary"])
    # aspect='equal' 유지를 위해 x/y 범위를 통일하고 여백을 축소
    x_range = (max(xs_all) - min(xs_all)) if xs_all else 2.0
    y_range = (max(ys_all) - min(ys_all)) if ys_all else 2.0
    max_range = max(x_range, y_range)
    x_center = (min(xs_all) + max(xs_all)) / 2 if xs_all else 0.0
    y_center = (min(ys_all) + max(ys_all)) / 2 if ys_all else 0.0
    half_span = max_range * (0.5 + 0.08)  # 8% margin (reduced from 15%)
    xmin, xmax = x_center - half_span, x_center + half_span
    ymin, ymax = y_center - half_span, y_center + half_span

    lam1 = lam2 = None
    if hasattr(ca, "explained_inertia_") and len(ca.explained_inertia_) >= 2:
        lam1, lam2 = ca.explained_inertia_[:2]

    # =================================================================
    # ablation loop
    # ★ 변경: 교량 단어 판정은 _bridge_counts (doc-level) 기반
    # =================================================================
    ca_bridge_summary = []

    # ★ 추가: CA top-200 중 doc-level 빈도표에도 존재하는 단어만 교량 후보
    if abuse_stats_chi_doc is not None and not abuse_stats_chi_doc.empty:
        _chi_doc_sorted = abuse_stats_chi_doc.sort_values("chi2", ascending=False, kind="mergesort")
        _chi_doc_top = _chi_doc_sorted.head(min(top_chi_for_ca, len(_chi_doc_sorted))).index
        bridge_candidate_words = [w for w in _chi_doc_top if w in _bridge_counts.index]
        print(
            f"[CA] bridge candidates: {len(bridge_candidate_words)}/{len(_chi_doc_top)} "
            f"(doc-level χ² top-{top_chi_for_ca}, present in {_bridge_src} counts)"
        )
    else:
        bridge_candidate_words = [w for w in df_ca.index if w in _bridge_counts.index]
        print(
            f"[CA] bridge candidates: {len(bridge_candidate_words)}/{len(df_ca.index)} "
            f"(CA words present in {_bridge_src} counts) [fallback: no doc-level χ²]"
        )

    for cfg in bridge_filter_configs:
        cfg_name = cfg["name"]
        min_p1 = cfg.get("min_p1", C.BRIDGE_MIN_P1)
        min_p2 = cfg.get("min_p2", C.BRIDGE_MIN_P2)
        max_gap = cfg.get("max_gap", C.BRIDGE_MAX_GAP)
        logodds_min = cfg.get("logodds_min")
        count_min = cfg.get("count_min", C.BRIDGE_MIN_COUNT)
        z_min = cfg.get("z_min")

        print(
            f"[CA-ABLT] config={cfg_name}: "
            f"min_p1={min_p1}, min_p2={min_p2}, max_gap={max_gap}, "
            f"count_min={count_min}, logodds_min={logodds_min}, z_min={z_min}"
        )

        # ★ 핵심 변경: doc-level 빈도표·logodds 로 교량 단어 판정
        bridge_prob = compute_prob_bridge_for_words(
            df_counts=_bridge_counts,                  # ★ was: df_abuse_counts
            words=bridge_candidate_words,              # ★ was: df_ca.index
            logodds_df=_bridge_logodds,                # ★ was: abuse_stats_logodds
            min_p1=min_p1,
            min_p2=min_p2,
            max_gap=max_gap,
            logodds_min=logodds_min,
            count_min=count_min,
            z_min=z_min,
        )

        if bridge_prob is not None and (not bridge_prob.empty):
            df_bridge = bridge_prob.copy()
            df_bridge.set_index("word", inplace=True)
            n_bridge = len(df_bridge)
        else:
            df_bridge = pd.DataFrame(columns=["primary_abuse", "secondary_abuse", "source"])
            df_bridge.index.name = "word"
            n_bridge = 0

        bridge_out_path_cfg = os.path.join(
            C.CA_PROB_DIR, f"abuse_bridge_words_prob_selected_for_CA_childtokens_{cfg_name}.csv"
        )
        df_bridge.to_csv(bridge_out_path_cfg, encoding="utf-8-sig")
        print(f"[SAVE] bridge words ({cfg_name}, {_bridge_src}) -> {bridge_out_path_cfg}")

        if cfg_name == "B0_baseline":
            bridge_out_path_base = os.path.join(
                C.CA_PROB_DIR, "abuse_bridge_words_prob_selected_for_CA_childtokens.csv"
            )
            df_bridge.to_csv(bridge_out_path_base, encoding="utf-8-sig")
            print(f"[SAVE] bridge words (baseline alias) -> {bridge_out_path_base}")

        # --- bridge_map: word → (k1, k2) 매핑 ---
        # k1 = primary_abuse (주 연관 유형, 논문 $k_1$)
        # k2 = secondary_abuse (부 연관 유형, 논문 $k_2$)
        # CSV 컬럼명(primary_abuse, secondary_abuse)은 하류 호환성 유지
        bridge_map = {}
        for w, r in df_bridge.iterrows():
            k1 = r.get("primary_abuse")
            k2 = r.get("secondary_abuse")
            if pd.notna(k1) and pd.notna(k2):
                bridge_map[w] = (str(k1), str(k2))

        # plot words = selected + bridges, limited
        plot_words = sorted(set(selected_words) | {w for w in df_bridge.index if w in bary_df.index})
        if len(plot_words) > plot_words_max:
            bridges = [w for w in plot_words if w in bridge_map]
            rest = [w for w in plot_words if w not in bridge_map]
            plot_words = (bridges + rest)[:plot_words_max]

        if not plot_words:
            ca_bridge_summary.append({
                "config": cfg_name, "min_p1": min_p1, "min_p2": min_p2, "max_gap": max_gap,
                "count_min": count_min, "n_bridge_all": n_bridge, "n_plot_words": 0, "n_bridge_in_plot": 0,
                "bridge_source": _bridge_src,      # ★ 추가
            })
            continue

        col_plot = bary_df.loc[plot_words]
        n_bridge_in_plot = sum(1 for w in col_plot.index if w in bridge_map)

        ca_bridge_summary.append({
            "config": cfg_name, "min_p1": min_p1, "min_p2": min_p2, "max_gap": max_gap,
            "count_min": count_min, "n_bridge_all": n_bridge,
            "n_plot_words": len(col_plot), "n_bridge_in_plot": n_bridge_in_plot,
            "bridge_source": _bridge_src,          # ★ 추가
        })

        # -------------------------------------------------------------
        # PLOT (English)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 11), dpi=220)
        ax.axhline(0, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.axvline(0, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="datalim")

        # cluster connectivity emphasis
        if emphasize_clusters and hasattr(C, "ABUSE_CLUSTERS") and isinstance(C.ABUSE_CLUSTERS, dict):
            for cl_name, members in C.ABUSE_CLUSTERS.items():
                members = [m for m in members if m in row_coords_2d.index]
                if len(members) == 0:
                    continue
                pts = row_coords_2d.loc[members, ["Dim1", "Dim2"]].values.astype(float)

                if len(members) >= 3:
                    hull = _convex_hull(pts)
                    if hull.shape[0] >= 3:
                        ax.fill(hull[:, 0], hull[:, 1], alpha=0.10, linewidth=0.0, zorder=0)

                centroid = pts.mean(axis=0)
                for m in members:
                    p = row_coords_2d.loc[m, ["Dim1", "Dim2"]].values.astype(float)
                    ax.plot([centroid[0], p[0]], [centroid[1], p[1]], linewidth=1.4, alpha=0.35, zorder=1)

                ax.scatter([centroid[0]], [centroid[1]], s=180, marker="x", linewidths=2.0, zorder=4)

        # abuse type별 고유 마커 (시각적 구분 강화)
        _ABUSE_MARKERS = {
            "성학대":   "D",   # Diamond
            "신체학대": "^",   # Triangle up
            "정서학대": "o",   # Circle
            "방임":     "s",   # Square
        }

        # abuse points + English labels
        for abuse_name, r in row_coords_2d.iterrows():
            x, y = float(r["Dim1"]), float(r["Dim2"])
            color = C.ABUSE_COLORS.get(abuse_name, "black")
            mkr = _ABUSE_MARKERS.get(abuse_name, "s")
            ax.scatter([x], [y], marker=mkr, s=160, color=color,
                       edgecolors="black", linewidth=0.8, alpha=0.95, zorder=6)
            ax.text(x, y, f"  {C.abuse_label(abuse_name, lang='en')}",
                    fontsize=12, weight="bold", va="center", zorder=7)

        # word points + label in English (via disp_token)
        # 교량 단어: facecolor = k1(주 연관 유형) 색상, edgecolor = k2(부 연관 유형) 색상
        texts = []
        for word, r in col_plot.iterrows():
            x, y = float(r["Dim1_bary"]), float(r["Dim2_bary"])
            main = word_main_abuse.get(word)
            main_color = C.ABUSE_COLORS.get(main, "black")

            if word in bridge_map:
                # k1(주 연관 유형) → 면 색, k2(부 연관 유형) → 테두리 색
                k1, k2 = bridge_map[word]
                k1_color = C.ABUSE_COLORS.get(k1, "black")
                k2_color = C.ABUSE_COLORS.get(k2, "black")
                ax.scatter(
                    [x], [y], s=70,
                    facecolor=k1_color,
                    edgecolor=k2_color,
                    linewidth=1.8, alpha=0.95, zorder=3,
                )
                text_color = "black"
            else:
                ax.scatter([x], [y], s=45, color=main_color, alpha=0.90, zorder=3)
                text_color = main_color

            wlab = C.disp_token(str(word), lang=lang, tt=tt, mode=display_mode)
            t = ax.text(x, y, wlab, fontsize=8.5, ha="center", va="center", color=text_color, zorder=8)
            texts.append(t)

        if getattr(C, "HAS_ADJUSTTEXT", False) and C.adjust_text is not None and texts:
            C.adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.35),
                expand_points=(1.05, 1.05),
                expand_text=(1.05, 1.05),
                force_points=0.10,
                force_text=0.35,
                lim=200,
            )

        # legend (English labels + visual element descriptions)
        legend_handles = []
        for abuse_name in C.ABUSE_ORDER:
            color = C.ABUSE_COLORS.get(abuse_name, "black")
            mkr = _ABUSE_MARKERS.get(abuse_name, "s")
            legend_handles.append(
                Line2D([0], [0], marker=mkr, linestyle="None",
                       markerfacecolor=color, markeredgecolor="black",
                       markeredgewidth=0.8,
                       label=C.abuse_label(abuse_name, lang="en"), markersize=10)
            )
        # Visual element descriptions
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor="gray", markeredgecolor="gray",
                   label="Word token", markersize=6, alpha=0.7)
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor="gray", markeredgecolor="black",
                   markeredgewidth=1.8,
                   label="Bridge word (dual-color)", markersize=8)
        )

        if lam1 is not None and lam2 is not None:
            xlabel = f"Dimension 1 ({lam1*100:.1f}% inertia)"
            ylabel = f"Dimension 2 ({lam2*100:.1f}% inertia)"
        else:
            xlabel = "Dimension 1"
            ylabel = "Dimension 2"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(
            f"CA biplot (prob-bridge) — config={cfg_name}\n"
            f"Rows: maltreatment types | Columns: words (barycentric)",
            fontsize=13
        )

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            title="Abuse type",
            frameon=True,
            framealpha=0.92,
            fontsize=9,
            title_fontsize=10,
        )

        ax.grid(True, alpha=0.20)
        fig.tight_layout()

        out_path_cfg = os.path.join(C.CA_PROB_DIR, f"abuse_CA_childtokens_probBridge_bary_chiTop_{cfg_name}.png")
        fig.savefig(out_path_cfg, dpi=220)
        plt.close(fig)
        print(f"[SAVE] CA biplot ({cfg_name}) -> {out_path_cfg}")

        if cfg_name == "B0_baseline":
            out_path_base = os.path.join(C.CA_PROB_DIR, "abuse_CA_childtokens_probBridge_bary_chiTop.png")
            try:
                import shutil
                shutil.copyfile(out_path_cfg, out_path_base)
                print(f"[SAVE] CA biplot (baseline alias) -> {out_path_base}")
            except Exception:
                pass

    # summary csv
    if ca_bridge_summary:
        df_ca_br = pd.DataFrame(ca_bridge_summary)
        ca_br_path = os.path.join(C.CA_PROB_DIR, "abuse_CA_bridge_filter_ablation_summary.csv")
        df_ca_br.to_csv(ca_br_path, encoding="utf-8-sig", index=False)
        print(f"[SAVE] CA bridge ablation summary -> {ca_br_path}")

    # save translator cache
    if tt is not None:
        try:
            tt.save()
        except Exception:
            pass

    return bary_df, row_coords_2d


# ---------------------------------------------------------------------------
# δ-ablation (log-odds based bridge robustness)
# ---------------------------------------------------------------------------
def bridge_ablation_and_assignments(
    abuse_stats_logodds: pd.DataFrame,
    df_abuse_counts: pd.DataFrame,
    prefix: str,
    output_dir: str,
    delta_cands=None,
    ref_delta: float | int | None = None,
):
    """
    δ-ablation 공통 함수 (log-odds 기반 bridge robustness).

    논문 표기와의 대응
    ------------------
    - k1          : 주 연관 유형 (log-odds 1순위) — 논문의 $k_1$
    - k2          : 부 연관 유형 (log-odds 2순위, bridge 성립 시) — 논문의 $k_2$
    - diff_thresh : $\\delta$ — $\\ell(k_1) - \\ell(k_2) < \\delta$ 이면 교량 단어
    """
    if delta_cands is None:
        delta_cands = C.BRIDGE_DELTA_CANDS
    if ref_delta is None:
        ref_delta = C.JACCARD_REF_DELTA

    bridge_summary_rows = []
    bridge_pair_rows = []
    bridge_word_sets = {}
    bridge_top50_sets = {}

    for delta in delta_cands:
        wm_tmp, wt2_tmp = compute_bridge_words(abuse_stats_logodds, diff_thresh=delta)
        # wt2_tmp: dict[word, (k1, k2 | None)]
        bridge_words = [w for w, (k1, k2) in wt2_tmp.items() if k2 is not None]
        bridge_word_sets[delta] = set(bridge_words)

        if bridge_words:
            freq = df_abuse_counts.loc[bridge_words, C.ABUSE_ORDER].sum(axis=1)
            top50 = freq.sort_values(ascending=False).head(50).index.tolist()
            bridge_top50_sets[delta] = set(top50)
        else:
            bridge_top50_sets[delta] = set()

        n_total = len(wt2_tmp)
        n_bridge = len(bridge_words)
        bridge_summary_rows.append({
            "delta": delta,
            "n_total_words": n_total,
            "n_bridge_words": n_bridge,
            "bridge_ratio": n_bridge / n_total if n_total > 0 else np.nan,
        })

        # 학대유형 쌍별 교량 단어 수 집계
        pair_counts = {}
        for w in bridge_words:
            k1, k2 = wt2_tmp[w]
            if k2 is None:
                continue
            # 순서 정규화: ABUSE_ORDER 기준으로 작은 인덱스가 앞
            if (k1 in C.ABUSE_ORDER) and (k2 in C.ABUSE_ORDER):
                i_k1 = C.ABUSE_ORDER.index(k1)
                i_k2 = C.ABUSE_ORDER.index(k2)
                pair = f"{k1}–{k2}" if i_k1 <= i_k2 else f"{k2}–{k1}"
            else:
                pair = f"{k1}–{k2}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        for pair, cnt in pair_counts.items():
            bridge_pair_rows.append({
                "delta": delta,
                "pair": pair,
                "n_bridge_words": cnt,
            })

    os.makedirs(output_dir, exist_ok=True)

    df_bridge_summary = pd.DataFrame(bridge_summary_rows)
    bridge_summary_path = os.path.join(
        output_dir, f"bridge_word_ablation_{prefix}_delta_summary.csv"
    )
    df_bridge_summary.to_csv(bridge_summary_path, encoding="utf-8-sig", index=False)
    print(f"[저장] bridge 단어 delta ablation summary ({prefix}) -> {bridge_summary_path}")
    print(f"[LOG] bridge 단어 delta ablation 요약 ({prefix}, 논문용):")
    print(df_bridge_summary.to_string(index=False))

    df_bridge_pairs = pd.DataFrame(bridge_pair_rows)
    if not df_bridge_pairs.empty:
        bridge_pairs_path = os.path.join(
            output_dir, f"bridge_word_ablation_{prefix}_delta_pairs.csv"
        )
        df_bridge_pairs.to_csv(bridge_pairs_path, encoding="utf-8-sig", index=False)
        print(f"[저장] bridge 단어 delta별 학대유형 쌍 분포 ({prefix}) -> {bridge_pairs_path}")
    else:
        print(f"[INFO] ({prefix}) bridge 단어 쌍 분포 데이터가 없어 pairs CSV는 생성하지 않습니다.")

    # ref_delta 기준 Jaccard overlap
    jacc_rows = []
    ref_all = bridge_word_sets.get(ref_delta, set())
    ref_top = bridge_top50_sets.get(ref_delta, set())

    for delta in delta_cands:
        all_set = bridge_word_sets.get(delta, set())
        top_set = bridge_top50_sets.get(delta, set())

        if ref_all or all_set:
            union_all = ref_all | all_set
            jacc_all = len(ref_all & all_set) / len(union_all) if union_all else np.nan
        else:
            jacc_all = np.nan

        if ref_top or top_set:
            union_top = ref_top | top_set
            jacc_top = len(ref_top & top_set) / len(union_top) if union_top else np.nan
        else:
            jacc_top = np.nan

        jacc_rows.append({
            "delta_ref": ref_delta,
            "delta_comp": delta,
            "jacc_all_bridge": jacc_all,
            "jacc_top50_bridge": jacc_top,
        })

    df_jacc = pd.DataFrame(jacc_rows)
    jacc_path = os.path.join(
        output_dir,
        f"bridge_word_ablation_{prefix}_delta_jaccard_vs_{ref_delta}.csv"
    )
    df_jacc.to_csv(jacc_path, encoding="utf-8-sig", index=False)
    print(f"[저장] bridge 단어 delta별 Jaccard overlap ({prefix}) -> {jacc_path}")
    print(f"[LOG] bridge 단어 Jaccard overlap ({prefix}, 논문용):")
    print(df_jacc.to_string(index=False))

    # 최종: 엄격 조건으로 main/top2 다시 산출
    word_main_abuse, word_top2 = compute_bridge_words(
        abuse_stats_logodds,
        diff_thresh=1.0,
        z_min=None,
        min_logodds=1.0,
        min_count_per_group=5,
        min_total=20,
        require_positive=True,
        strict_z=True
    )

    return {
        "word_main_abuse": word_main_abuse,
        "word_top2": word_top2,
        "bridge_word_sets": bridge_word_sets,
        "bridge_top50_sets": bridge_top50_sets,
    }