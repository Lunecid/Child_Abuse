from __future__ import annotations

from .common import *
from .text import tokenize_korean
import matplotlib.pyplot as plt

def train_embedding_models(corpus_tokens, output_dir,
                           vector_size=100, window=5, min_count=5, workers=4):
    """
    corpus_tokens: [["토큰1","토큰2",...], ...] 형태
    gensim Word2Vec / FastText 학습 후 모델 객체와 파일 저장.
    (현재 버전: 스무딩 X, 임베딩 공간 투사용으로만 사용)
    """
    if not HAS_GENSIM:
        print("[EMB] gensim 없음 → 임베딩 학습 건너뜁니다.")
        return None, None

    if not corpus_tokens or len(corpus_tokens) < 10:
        print("[EMB] 임베딩 학습에 사용할 코퍼스가 너무 적어 건너뜁니다.")
        return None, None

    print(f"[EMB] word2vec / fastText 학습 시작 (docs={len(corpus_tokens)})")

    w2v_model = Word2Vec(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        negative=10,
        epochs=20,
    )
    w2v_path = os.path.join(output_dir, "w2v_child_tokens.model")
    w2v_model.save(w2v_path)
    print(f"[EMB] word2vec 모델 저장 -> {w2v_path}")

    ft_model = FastText(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        negative=10,
        epochs=20,
    )
    ft_path = os.path.join(output_dir, "fasttext_child_tokens.model")
    ft_model.save(ft_path)
    print(f"[EMB] fastText 모델 저장 -> {ft_path}")

    return w2v_model, ft_model


def project_embeddings_for_ca_words(model,
                                   abuse_stats_chi,
                                   prefix,
                                   out_dir,
                                   top_chi_for_ca=200):
    """
    chi² 상위 단어(top_chi_for_ca)를 대상으로
    word2vec / fastText 임베딩 벡터를 2D (PCA)로 투사해서 CSV로 저장.
    - 동일 단어 집합에 대해 CA(barycentric) 좌표 CSV와 비교 가능.
    - DataFrame 을 리턴해서 이후 비교/시각화에 바로 사용 가능.
    """
    if model is None:
        print(f"[EMB] {prefix}: 모델이 없어 투사를 건너뜁니다.")
        return None
    if not HAS_SKLEARN:
        print(f"[EMB] {prefix}: scikit-learn(PCA) 미설치 → 투사 건너뜁니다.")
        return None
    if abuse_stats_chi is None or abuse_stats_chi.empty:
        print(f"[EMB] {prefix}: chi² 통계가 비어 있어 투사 건너뜁니다.")
        return None

    # chi² 상위 단어들 중에서 임베딩 vocab에 있는 단어만 사용
    chi_sorted = abuse_stats_chi.sort_values("chi2", ascending=False)
    words = [w for w in chi_sorted.index if w in model.wv.key_to_index]
    if not words:
        print(f"[EMB] {prefix}: chi² 상위 단어 중 임베딩 vocab에 포함된 단어가 없습니다.")
        return None

    words = words[:top_chi_for_ca]
    vecs = np.vstack([model.wv[w] for w in words])

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vecs)

    df = pd.DataFrame({
        "word": words,
        "Dim1_emb": coords[:, 0],
        "Dim2_emb": coords[:, 1],
    })
    path = os.path.join(out_dir, f"embedding_PCA_{prefix}_chiTop{len(words)}.csv")
    df.to_csv(path, encoding="utf-8-sig", index=False)
    print(f"[저장] 임베딩 2D(PCA) 좌표 ({prefix}) -> {path}")

    return df


def _procrustes_and_mantel(
    df,
    ca_cols=("Dim1_bary", "Dim2_bary"),
    emb_cols=("Dim1_emb", "Dim2_emb"),
    model_name="w2v",
    out_dir=EMBED_PROJ_DIR,
    max_perm: int = 499,
    max_n_for_mantel: int = 250,
):
    """
    df: 공통 단어에 대한 CA(barycentric) + 임베딩(PCA) 좌표 포함 DataFrame
        columns에 ca_cols, emb_cols가 모두 있어야 함.

    반환:
        dict(model, n_words, procrustes_disparity, procrustes_rms,
             mantel_r, mantel_p, n_perm)
    """
    os.makedirs(out_dir, exist_ok=True)

    for c in list(ca_cols) + list(emb_cols):
        if c not in df.columns:
            print(f"[PROC] {model_name}: {c} 컬럼이 없어 Procrustes를 건너뜁니다.")
            return None

    X = df[list(ca_cols)].astype(float).values
    Y = df[list(emb_cols)].astype(float).values
    n, d = X.shape
    if n < 5 or d != 2:
        print(f"[PROC] {model_name}: n={n}, d={d} → Procrustes 조건 불충분.")
        return None

    # 중심화 & 스케일링
    X0 = X - X.mean(axis=0, keepdims=True)
    Y0 = Y - Y.mean(axis=0, keepdims=True)
    normX = np.linalg.norm(X0)
    normY = np.linalg.norm(Y0)
    if normX == 0 or normY == 0:
        print(f"[PROC] {model_name}: 좌표 분산이 너무 작습니다.")
        return None
    X0 /= normX
    Y0 /= normY

    # Procrustes: Y를 X에 맞춰 회전
    M = X0.T @ Y0
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    Y_aligned = Y0 @ R.T

    # disparity & RMSD
    diff = X0 - Y_aligned
    disparity = float(np.sum(diff ** 2))
    rms = float(np.sqrt(np.mean(diff ** 2)))

    # Mantel: 거리 행렬 상관 (Spearman), 회전 불변
    mantel_r, mantel_p, used_perm = np.nan, np.nan, 0
    if n <= max_n_for_mantel:
        # 거리 행렬 (유클리드)
        DX = np.sqrt(((X0[:, None, :] - X0[None, :, :]) ** 2).sum(axis=2))
        DY = np.sqrt(((Y0[:, None, :] - Y0[None, :, :]) ** 2).sum(axis=2))

        # 상삼각 부분만 벡터로 펼치기
        iu = np.triu_indices(n, k=1)
        vX = DX[iu]
        vY = DY[iu]

        if HAS_SCIPY:
            from scipy.stats import spearmanr as _sr
            mantel_r, _ = _sr(vX, vY)
        else:
            mantel_r = float(np.corrcoef(vX, vY)[0, 1])

        # permutation 기반 p-value (Mantel)
        if HAS_SCIPY:
            rng = np.random.default_rng(2025)
            perm_rs = []
            used_perm = max_perm
            for _ in range(max_perm):
                perm_idx = rng.permutation(n)
                DY_perm = DY[perm_idx][:, perm_idx]
                vYp = DY_perm[iu]
                r_p, _ = _sr(vX, vYp)
                perm_rs.append(r_p)
            perm_rs = np.array(perm_rs)
            mantel_p = float(
                (1 + np.sum(np.abs(perm_rs) >= np.abs(mantel_r))) / (len(perm_rs) + 1)
            )

    # 정렬된 좌표 산점도도 저장
    plt.figure(figsize=(7, 6))
    plt.scatter(X0[:, 0], X0[:, 1], s=25, alpha=0.7, label="CA (bary)")
    plt.scatter(Y_aligned[:, 0], Y_aligned[:, 1], s=25, alpha=0.7, label=f"{model_name} (aligned)")
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.title(
        f"Procrustes 정렬: CA vs {model_name}\n"
        f"disparity={disparity:.4f}, RMS={rms:.4f}, Mantel r={mantel_r:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"CA_vs_{model_name}_procrustes_scatter.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[저장] Procrustes 정렬 산점도 ({model_name}) -> {fig_path}")

    return {
        "model": model_name,
        "n_words": n,
        "procrustes_disparity": disparity,
        "procrustes_rms": rms,
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
        "n_perm": used_perm,
    }


def compare_ca_and_embedding_spaces(bary_df,
                                    abuse_stats_logodds,
                                    embedding_results,
                                    out_dir):
    """
    CA barycentric 좌표(bary_df: word × (Dim1_bary, Dim2_bary))와
    임베딩 PCA 좌표(embedding_results: {'w2v': df, 'fasttext': df})를 비교.

    - (기존) 공통 단어에 대해 CA vs 임베딩 좌표 상관계수(Pearson, Spearman) + 산점도
    - (NEW) Procrustes 정렬 + Mantel 거리 상관 (회전 불변) 요약 추가
    """
    if bary_df is None or bary_df.empty:
        print("[EMB-COMP] bary_df 가 비어 있어 CA vs 임베딩 비교를 건너뜁니다.")
        return

    if embedding_results is None or not isinstance(embedding_results, dict):
        print("[EMB-COMP] embedding_results 가 비어 있어 비교를 건너뜁니다.")
        return

    # 단어별 main_abuse 매핑 (log-odds 최대 학대유형)
    if abuse_stats_logodds is not None and not abuse_stats_logodds.empty:
        word_main_abuse = (
            abuse_stats_logodds
            .sort_values("log_odds", ascending=False)
            .groupby("word")["group"]
            .first()
        )
    else:
        word_main_abuse = pd.Series(dtype=object)

    summary_rows = []           # dim별 상관 요약
    proc_summary_rows = []      # Procrustes + Mantel 요약

    for model_name, df_emb in embedding_results.items():
        if df_emb is None or df_emb.empty:
            print(f"[EMB-COMP] {model_name}: 투사된 임베딩 좌표가 없어 건너뜁니다.")
            continue

        df_emb_idx = df_emb.set_index("word")
        common_words = bary_df.index.intersection(df_emb_idx.index)

        if len(common_words) < 5:
            print(f"[EMB-COMP] {model_name}: CA·임베딩 공통 단어가 5개 미만 ({len(common_words)}개).")
            continue

        merged = pd.concat(
            [bary_df.loc[common_words], df_emb_idx.loc[common_words]],
            axis=1,
        ).reset_index().rename(columns={"index": "word"})

        # main_abuse 컬럼 추가
        merged["main_abuse"] = merged["word"].map(word_main_abuse).fillna("기타")

        # -------- 1) dim별 상관 (기존) --------
        for dim_pair, label in [
            (("Dim1_bary", "Dim1_emb"), "Dim1"),
            (("Dim2_bary", "Dim2_emb"), "Dim2"),
        ]:
            x_col, y_col = dim_pair
            if x_col not in merged.columns or y_col not in merged.columns:
                continue

            x = merged[x_col].astype(float)
            y = merged[y_col].astype(float)
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                continue

            pearson = float(np.corrcoef(x[mask], y[mask])[0, 1])
            if HAS_SCIPY:
                rho, p_rho = spearmanr(x[mask], y[mask])
            else:
                rho, p_rho = np.nan, np.nan

            summary_rows.append({
                "model": model_name,
                "dimension": label,
                "n_words": int(mask.sum()),
                "pearson": pearson,
                "spearman": rho,
                "spearman_p": p_rho,
            })

            # 산점도 (CA vs EMB)
            plt.figure(figsize=(7, 6))
            for abuse in list(ABUSE_ORDER) + ["기타"]:
                sub = merged[merged["main_abuse"] == abuse]
                if sub.empty:
                    continue
                color = ABUSE_COLORS.get(abuse, "#7f7f7f")
                plt.scatter(
                    sub[x_col],
                    sub[y_col],
                    s=30,
                    alpha=0.8,
                    label=abuse,
                    color=color,
                )

            plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")

            plt.xlabel(f"CA {label}")
            plt.ylabel(f"{model_name} PCA {label}")
            plt.title(
                f"CA vs {model_name} ({label})\n"
                f"Pearson={pearson:.3f}, Spearman={rho:.3f}"
            )

            plt.legend(
                loc="upper right",
                bbox_to_anchor=(1.25, 1.0),
                frameon=True,
                framealpha=0.9,
            )
            plt.tight_layout()

            out_scatter_path = os.path.join(
                out_dir,
                f"CA_vs_{model_name}_{label}_scatter.png",
            )
            plt.savefig(out_scatter_path, dpi=200)
            print(f"[저장] CA vs {model_name} ({label}) scatter -> {out_scatter_path}")
            plt.close()

        # -------- 2) 임베딩 공간 자체 시각화 (기존) --------
        if {"Dim1_emb", "Dim2_emb"}.issubset(merged.columns):
            plt.figure(figsize=(7, 6))
            for abuse in list(ABUSE_ORDER) + ["기타"]:
                sub = merged[merged["main_abuse"] == abuse]
                if sub.empty:
                    continue
                color = ABUSE_COLORS.get(abuse, "#7f7f7f")
                plt.scatter(
                    sub["Dim1_emb"],
                    sub["Dim2_emb"],
                    s=30,
                    alpha=0.8,
                    label=abuse,
                    color=color,
                )
            plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            plt.xlabel("Embedding Dim1 (PCA)")
            plt.ylabel("Embedding Dim2 (PCA)")
            plt.title(f"{model_name} 임베딩 공간 (chi² 상위 단어)")
            plt.legend(
                loc="upper right",
                bbox_to_anchor=(1.25, 1.0),
                frameon=True,
                framealpha=0.9,
            )
            plt.tight_layout()

            out_emb_path = os.path.join(
                out_dir,
                f"embedding_space_{model_name}_chiTop.png",
            )
            plt.savefig(out_emb_path, dpi=200)
            print(f"[저장] {model_name} 임베딩 공간 scatter -> {out_emb_path}")
            plt.close()

        # -------- 3) Procrustes + Mantel (NEW) --------
        proc_res = _procrustes_and_mantel(
            merged,
            ca_cols=("Dim1_bary", "Dim2_bary"),
            emb_cols=("Dim1_emb", "Dim2_emb"),
            model_name=model_name,
            out_dir=out_dir,
        )
        if proc_res is not None:
            proc_summary_rows.append(proc_res)

    # dim별 상관 요약 CSV
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_dir, "CA_vs_embedding_correlation_summary.csv")
        df_summary.to_csv(summary_path, encoding="utf-8-sig", index=False)
        print(f"[저장] CA vs 임베딩 상관 요약 -> {summary_path}")
    else:
        print("[EMB-COMP] CA vs 임베딩 상관 요약을 만들 데이터가 없습니다.")

    # Procrustes + Mantel 요약 CSV
    if proc_summary_rows:
        df_proc = pd.DataFrame(proc_summary_rows)
        proc_path = os.path.join(out_dir, "CA_vs_embedding_procrustes_mantel_summary.csv")
        df_proc.to_csv(proc_path, encoding="utf-8-sig", index=False)
        print(f"[저장] Procrustes + Mantel 요약 -> {proc_path}")
    else:
        print("[EMB-COMP] Procrustes/Mantel 요약을 만들 데이터가 없습니다.")


def compare_ca_and_embeddings(
    ca_bary_path,
    w2v_proj_path,
    ft_proj_path,
    out_dir
):
    """
    CA barycentric 좌표와 W2V/FT 임베딩 PCA 좌표를 같은 단어 기준으로 merge해서
    상관계수 & 산점도를 생성.
    - ca_bary_path: CA 단어 좌표 CSV (abuse_CA_words_barycentric_...)
    - w2v_proj_path: embedding_PCA_w2v_....csv
    - ft_proj_path:  embedding_PCA_fasttext_....csv
    """

    try:
        ca_df = pd.read_csv(ca_bary_path, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"[COMP] CA barycentric CSV를 찾을 수 없습니다: {ca_bary_path}")
        return

    ca_df = ca_df.rename(columns={"Dim1_bary": "Dim1_CA", "Dim2_bary": "Dim2_CA"})
    if "word" not in ca_df.columns:
        ca_df = ca_df.reset_index().rename(columns={"index": "word"})

    def _load_emb(path, prefix):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except FileNotFoundError:
            print(f"[COMP] {prefix} 임베딩 투사 CSV를 찾을 수 없습니다: {path}")
            return None
        df = df.rename(columns={
            "Dim1_emb": f"Dim1_{prefix}",
            "Dim2_emb": f"Dim2_{prefix}",
        })
        return df

    w2v_df = _load_emb(w2v_proj_path, "w2v") if w2v_proj_path is not None else None
    ft_df  = _load_emb(ft_proj_path, "FT") if ft_proj_path is not None else None

    # --- CA vs W2V ---
    if w2v_df is not None:
        df_w2v = ca_df.merge(w2v_df, on="word", how="inner")
        if len(df_w2v) >= 5:
            print(f"[COMP] CA vs W2V 공통 단어 수 = {len(df_w2v)}")
            _analyze_and_plot_space(
                df_w2v,
                x_cols=["Dim1_CA", "Dim2_CA"],
                y_cols=["Dim1_w2v", "Dim2_w2v"],
                prefix="CA_vs_W2V",
                out_dir=out_dir,
            )
        else:
            print("[COMP] CA vs W2V 공통 단어가 너무 적어 비교를 건너뜁니다.")

    # --- CA vs FastText ---
    if ft_df is not None:
        df_ft = ca_df.merge(ft_df, on="word", how="inner")
        if len(df_ft) >= 5:
            print(f"[COMP] CA vs FastText 공통 단어 수 = {len(df_ft)}")
            _analyze_and_plot_space(
                df_ft,
                x_cols=["Dim1_CA", "Dim2_CA"],
                y_cols=["Dim1_FT", "Dim2_FT"],
                prefix="CA_vs_FT",
                out_dir=out_dir,
            )
        else:
            print("[COMP] CA vs FastText 공통 단어가 너무 적어 비교를 건너뜁니다.")


def _analyze_and_plot_space(df, x_cols, y_cols, prefix, out_dir):
    """
    df: word 단위로 CA와 임베딩 좌표가 모두 들어 있는 DataFrame
    x_cols: ["Dim1_CA", "Dim2_CA"]
    y_cols: ["Dim1_w2v", "Dim2_w2v"] 같은 것
    prefix: "CA_vs_W2V" 등
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 상관계수 (각 dim 쌍)
    corrs = []
    for i, xc in enumerate(x_cols):
        for j, yc in enumerate(y_cols):
            s1 = df[xc].astype(float)
            s2 = df[yc].astype(float)
            if HAS_SCIPY:
                rho, p = spearmanr(s1, s2)
            else:
                rho = s1.corr(s2)
                p = np.nan
            corrs.append({
                "x_dim": xc,
                "y_dim": yc,
                "rho": rho,
                "p_value": p,
                "n": len(df),
            })

    df_corr = pd.DataFrame(corrs)
    corr_path = os.path.join(out_dir, f"{prefix}_dim_correlations.csv")
    df_corr.to_csv(corr_path, encoding="utf-8-sig", index=False)
    print(f"[저장] {prefix} 차원별 상관계수 -> {corr_path}")
    print("[LOG] 차원별 상관계수 (논문용):")
    print(df_corr.to_string(index=False))

    # 2) 산점도 (각 dim 쌍, 2x2 그림)
    plt.figure(figsize=(8, 8))
    for idx, (xc, yc) in enumerate([(x_cols[0], y_cols[0]),
                                    (x_cols[0], y_cols[1]),
                                    (x_cols[1], y_cols[0]),
                                    (x_cols[1], y_cols[1])]):
        plt.subplot(2, 2, idx + 1)
        plt.scatter(df[xc], df[yc], s=15, alpha=0.7)
        plt.xlabel(xc)
        plt.ylabel(yc)
    plt.suptitle(f"{prefix}: CA vs Embedding 2D coord scatter")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_scatter_dims.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[저장] {prefix} 차원별 산점도 -> {fig_path}")

    # -----------------------------------------
    # 4. 학대유형별 단어 분석 (token-level)
    # -----------------------------------------
    df_abuse_counts = pd.DataFrame()
    abuse_stats_logodds = pd.DataFrame()
    abuse_stats_chi = pd.DataFrame()
    baseline_bridge_info = None
    bary_df = None   # [NEW] CA barycentric word coordinates (for embedding 비교)
