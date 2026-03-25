from __future__ import annotations

import matplotlib.pyplot as plt
from abuse_pipeline.core.common import *


def compute_hhi_and_cosine(df_counts: pd.DataFrame, group_cols):
    counts = df_counts[group_cols].astype(float).copy()

    col_sums = counts.sum(axis=0)
    hhi_list = []
    for g in group_cols:
        total = col_sums.get(g, 0.0)
        if total <= 0:
            hhi_val = np.nan
        else:
            p = counts[g] / total
            hhi_val = float((p ** 2).sum())
        hhi_list.append({"group": g, "hhi": hhi_val})

    df_hhi = pd.DataFrame(hhi_list)

    vec = counts.T.values
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    vec_norm = vec / norms

    sim = np.matmul(vec_norm, vec_norm.T)
    sim = np.nan_to_num(sim, nan=np.nan)

    df_cos = pd.DataFrame(sim, index=group_cols, columns=group_cols)

    return df_hhi, df_cos


def compute_chi_square(df_counts, cols):
    if df_counts.empty or len(cols) == 0:
        return pd.DataFrame(columns=["chi2", "p_value"])

    col_totals = df_counts[cols].sum(axis=0).values
    row_totals = df_counts[cols].sum(axis=1).values
    N = col_totals.sum()

    if N <= 0:
        return pd.DataFrame(columns=["chi2", "p_value"])

    expected = np.outer(row_totals, col_totals) / N
    observed = df_counts[cols].values

    with np.errstate(divide="ignore", invalid="ignore"):
        chi_sq = (observed - expected) ** 2 / expected
        chi_sq = np.nan_to_num(chi_sq, nan=0.0, posinf=0.0, neginf=0.0)

    chi_total = chi_sq.sum(axis=1)
    df_out = pd.DataFrame({"word": df_counts.index, "chi2": chi_total})
    if HAS_SCIPY:
        df_out["p_value"] = chi2.sf(df_out["chi2"], df=len(cols) - 1)
    else:
        df_out["p_value"] = np.nan
    return df_out.set_index("word")


def merge_permutation_pvalues(chi_df, df_perm, p_perm_col="p_perm"):
    """
    순열 검정 결과(df_perm)의 경험적 p-value를 chi_df에 병합하고
    BH-FDR 보정을 적용한다.

    Parameters
    ----------
    chi_df : pd.DataFrame
        index=word, columns ⊇ ["chi2", "p_value"].
        compute_chi_square()의 반환값.
    df_perm : pd.DataFrame
        columns ⊇ ["word", "p_perm"].
        run_doc_level_label_shuffle_permutation()의 반환값.
    p_perm_col : str
        df_perm에서 순열 p-value 열 이름.

    Returns
    -------
    pd.DataFrame
        chi_df에 p_perm, p_perm_fdr_bh 열이 추가된 DataFrame.
    """
    if df_perm is None or df_perm.empty:
        return chi_df

    # df_perm의 word를 인덱스로 변환하여 chi_df와 병합
    perm_series = df_perm.set_index("word")[p_perm_col]
    chi_df["p_perm"] = chi_df.index.map(perm_series)

    # 순열 p-value에 BH-FDR 보정 적용
    chi_df = add_bh_fdr(chi_df, p_col="p_perm", out_col="p_perm_fdr_bh")

    n_merged = chi_df["p_perm"].notna().sum()
    n_total = len(chi_df)
    print(f"[PERM-MERGE] 순열 p-value 병합: {n_merged}/{n_total}개 단어에 p_perm 적용")

    return chi_df


def add_bh_fdr(df, p_col="p_value", out_col="p_fdr_bh"):
    """
    Benjamini–Hochberg FDR 보정.
    df[p_col]을 이용해 q-value를 계산해서 df[out_col]에 추가한다.
    p_col 에 NaN 이 있으면 그대로 NaN 유지.
    """
    if p_col not in df.columns:
        raise ValueError(f"{p_col} not in DataFrame")

    p = df[p_col].values.astype(float)
    m = np.isfinite(p).sum()
    if m == 0:
        df[out_col] = np.nan
        return df

    # 유한한 p-value 위치만 대상으로 BH 수행
    idx_finite = np.where(np.isfinite(p))[0]
    p_finite = p[idx_finite]

    order = np.argsort(p_finite)              # p 오름차순 인덱스
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, len(p_finite) + 1)

    # 원래 BH: q_i = p_i * m / rank_i, 뒤에서부터 누적 최소
    q = p_finite * m / ranked
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_bh = np.empty_like(q)
    q_bh[order] = q_sorted
    q_bh = np.minimum(q_bh, 1.0)

    # 전체 벡터에 다시 채우기
    q_all = np.full_like(p, np.nan, dtype=float)
    q_all[idx_finite] = q_bh

    df[out_col] = q_all
    return df


def compute_log_odds(df_counts, cols, alpha=0.01):
    """
    안정화된 로그 오즈비(stabilized log-odds ratio)를 산출한다.

    논문 표기와의 대응
    ------------------
    - 대상 집단 g1 과 나머지 집단 g2 = ¬g1 (one-vs-rest)
    - p_{w,g1} = (c_{w,g1} + α) / (N_{g1} + αV)
    - p_{w,g2} = (c_{w,g2} + α) / (N_{g2} + αV)
    - δ̂_w^{(g1,g2)} = logit(p_{w,g1}) − logit(p_{w,g2})

    Parameters
    ----------
    df_counts : pd.DataFrame
        index = word, columns ⊇ cols.
        단위는 호출 시점에 따라 토큰 빈도 또는 문서 수(doc-level)가 될 수 있다.
        bridge 분석에서는 문서 수 기반 테이블을 권장한다.
    cols : list[str]
        비교 대상 집단 이름 목록 (예: ABUSE_ORDER).
    alpha : float
        Dirichlet 평활화 상수 (기본 0.01).

    Returns
    -------
    pd.DataFrame
        columns: word, group, count, log_odds, se_log_odds, z_log_odds
        - count: df_counts에서 해당 집단의 원 빈도 (토큰 또는 문서 수).
        - group: 대상 집단 g1. 비교 상대 g2 = ¬g1 은 one-vs-rest로 자동 구성됨.
    """
    V = df_counts.shape[0]
    totals = df_counts[cols].sum(axis=0)
    N_total = totals.sum()

    records = []
    for word, row in df_counts[cols].iterrows():
        total_word = row.sum()
        for g1 in cols:
            # --- g1 (대상 집단) vs g2 = ¬g1 (나머지 집단) ---
            c_g1 = row[g1]
            c_g2 = total_word - c_g1
            N_g1 = totals[g1]
            N_g2 = N_total - N_g1

            # Dirichlet 평활화
            c_g1_s = c_g1 + alpha
            c_g2_s = c_g2 + alpha

            # 평활화 확률: p_{w,g1}, p_{w,g2}
            p_g1 = c_g1_s / (N_g1 + alpha * V) if (N_g1 + alpha * V) > 0 else 0.0
            p_g2 = c_g2_s / (N_g2 + alpha * V) if (N_g2 + alpha * V) > 0 else 0.0

            # 로그 오즈비: δ̂_w^{(g1,g2)} = logit(p_g1) − logit(p_g2)
            if p_g1 <= 0 or p_g1 >= 1 or p_g2 <= 0 or p_g2 >= 1:
                log_odds = 0.0
                var_log_odds = np.inf
            else:
                log_odds = np.log(p_g1 / (1 - p_g1)) - np.log(p_g2 / (1 - p_g2))
                var_log_odds = 1.0 / c_g1_s + 1.0 / c_g2_s

            se_log_odds = np.sqrt(var_log_odds) if np.isfinite(var_log_odds) else np.inf
            z_log_odds = log_odds / se_log_odds if se_log_odds > 0 and np.isfinite(se_log_odds) else 0.0

            records.append({
                "word": word,
                "group": g1,           # 대상 집단 (g1); 비교 상대는 ¬g1
                "count": c_g1,         # g1에서의 원 빈도 (단위는 df_counts에 의존)
                "log_odds": log_odds,
                "se_log_odds": se_log_odds,
                "z_log_odds": z_log_odds,
            })
    return pd.DataFrame(records)

def compute_prob_bridge_for_words(
    df_counts,
    words,
    logodds_df=None,
    min_p1=BRIDGE_MIN_P1,
    min_p2=BRIDGE_MIN_P2,
    max_gap=BRIDGE_MAX_GAP,
    logodds_min=None,
    count_min=5,
    z_min=None,
):
    """
    P(abuse | word) 기반 교량 단어(bridge word) 식별 함수.

    논문 표기와의 대응
    ------------------
    단어 w에 대해 P(k|w) = c_{w,k} / Σ_{k'} c_{w,k'} 를 산출한 뒤,
    내림차순 1순위 k1(주 연관 유형), 2순위 k2(부 연관 유형)에 대해
    다음 세 조건을 동시에 만족하면 교량 단어로 식별한다:

        P(k1|w) ≥ τ1,    P(k2|w) ≥ τ2,    P(k1|w) − P(k2|w) ≤ γ

    Parameters
    ----------
    df_counts : pd.DataFrame
        index = word, columns ⊇ ABUSE_ORDER.
        **문서 수(doc-level) 기반 빈도표를 권장한다.**
        한 아동(=문서) 내에서 동일 단어는 1회만 카운트(set 처리)된 테이블이어야
        소수 아동의 반복 사용에 의한 불안정성이 억제된다.
    words : list[str]
        교량 단어 후보 목록 (예: χ² 상위 200개).
    logodds_df : pd.DataFrame or None
        compute_log_odds()의 반환값. count_min 필터에 사용.
        columns: word, group, count, log_odds, se_log_odds, z_log_odds.
    min_p1 : float
        τ1 — 주 연관 유형 최소 조건부 확률 (기본 0.40).
    min_p2 : float
        τ2 — 부 연관 유형 최소 조건부 확률 (기본 0.25).
    max_gap : float
        γ — 주·부 유형 간 확률 차이 상한 (기본 0.20).
    logodds_min : float or None
        (선택) k1, k2 양쪽의 log-odds 최소값. None이면 미적용.
    count_min : int or None
        (선택) k1, k2 **양쪽 유형 모두**에서 단어 w가 출현한 문서 수의 최소값.
        예: count_min=5이면 d_{w,k1} ≥ 5 이고 d_{w,k2} ≥ 5 여야 한다.
        None이면 미적용.
    z_min : float or None
        (선택) k1, k2 양쪽의 z_log_odds 최소값. None이면 미적용.

    Returns
    -------
    pd.DataFrame
        columns: word, primary_abuse(k1), secondary_abuse(k2),
                 p1, p2, gap, source
    """
    if logodds_df is not None:
        lod = logodds_df.set_index(["word", "group"])
    else:
        lod = None

    records = []
    for w in words:
        if w not in df_counts.index:
            continue
        row = df_counts.loc[w, ABUSE_ORDER].astype(float).values
        total = row.sum()
        if total <= 0:
            continue

        # P(k | w) = c_{w,k} / Σ c_{w,k'}
        probs = row / total
        idx_sorted = np.argsort(-probs)
        if len(idx_sorted) < 2:
            continue

        # k1(주 연관 유형), k2(부 연관 유형)
        i_k1, i_k2 = idx_sorted[0], idx_sorted[1]
        p_k1, p_k2 = probs[i_k1], probs[i_k2]
        k1, k2 = ABUSE_ORDER[i_k1], ABUSE_ORDER[i_k2]

        # --- 교량 단어 정의 조건 (논문 식 \ref{eq:bridge_cond}) ---
        # 조건 1: P(k1|w) ≥ τ1
        if p_k1 < min_p1:
            continue
        # 조건 2: P(k2|w) ≥ τ2
        if p_k2 < min_p2:
            continue
        # 조건 3: P(k1|w) − P(k2|w) ≤ γ
        if (p_k1 - p_k2) > max_gap:
            continue

        # --- 안정성 필터: 문서 수 / log-odds / z 기반 ---
        if lod is not None and (logodds_min is not None or count_min is not None or z_min is not None):
            try:
                row_k1 = lod.loc[(w, k1)]
                row_k2 = lod.loc[(w, k2)]
            except KeyError:
                continue

            # log-odds 최소값 필터
            if logodds_min is not None:
                if (row_k1.get("log_odds", 0.0) < logodds_min) or \
                   (row_k2.get("log_odds", 0.0) < logodds_min):
                    continue

            # 문서 수 최소값 필터: d_{w,k1} ≥ count_min AND d_{w,k2} ≥ count_min
            if count_min is not None:
                if (row_k1.get("count", 0.0) < count_min) or \
                   (row_k2.get("count", 0.0) < count_min):
                    continue

            # z-score 최소값 필터
            if z_min is not None:
                if (row_k1.get("z_log_odds", 0.0) < z_min) or \
                   (row_k2.get("z_log_odds", 0.0) < z_min):
                    continue

        records.append({
            "word": w,
            "primary_abuse": k1,    # k1: 주 연관 유형
            "secondary_abuse": k2,  # k2: 부 연관 유형
            "p1": p_k1,             # P(k1 | w)
            "p2": p_k2,             # P(k2 | w)
            "gap": p_k1 - p_k2,    # P(k1|w) − P(k2|w)
            "source": "prob",
        })

    return pd.DataFrame(records)

def compute_bridge_words(
    abuse_stats_logodds,
    diff_thresh: float = 1.0,
    z_min: float | None = None,
    *,
    # --- thresholds (override globals if you want) ---
    min_logodds: float | None = None,
    min_count_per_group: int | None = None,
    min_total: int | None = None,
    # --- behavior knobs ---
    require_positive: bool = True,
    drop_nonfinite: bool = True,
    strict_z: bool = True,
):
    """
    δ 기반 log-odds bridge (A-방식):
    p(abuse|word)가 아니라 log-odds(g vs not-g)의 크기/차이에 기반한 bridge.

    목적:
    - 본문 '정의'로 쓰기보다 delta ablation / robustness check 용도에 적합.
    - 희귀단어 불안정 방지를 위해 min_total(단어 총빈도 하한)을 지원.

    Parameters
    ----------
    abuse_stats_logodds : pd.DataFrame
        required columns: ['word','group','log_odds']
        optional columns: ['count','z_log_odds']
    diff_thresh : float
        top1과 top2 log_odds 차이의 상한 (l1 - l2 < diff_thresh)
    z_min : float or None
        z_log_odds 최소값(둘 다 통과해야 bridge). None이면 사용 안 함.
    min_logodds : float or None
        각 top1/top2의 log_odds 최소값. None이면 전역 BRIDGE_MIN_LOGODDS가 있으면 사용.
    min_count_per_group : int or None
        top1/top2 각 그룹에서의 count 최소값. None이면 전역 BRIDGE_MIN_COUNT가 있으면 사용.
    min_total : int or None
        단어 w의 전체 count 합(모든 그룹 합) 최소값. None이면 사용 안 함.
    require_positive : bool
        True면 l1>0 & l2>0 조건을 추가 (해당 그룹에 "양(+)의 과대표현"일 때만 bridge).
    drop_nonfinite : bool
        True면 log_odds가 NaN/Inf인 row를 제거하고 계산.
    strict_z : bool
        z_min을 쓸 때 z_log_odds가 NaN이면 탈락시키는지 여부.
        (True 권장: z 기반 필터가 의미 있으려면 결측을 통과시키지 않는 편이 안전)

    Returns
    -------
    word_main_abuse : dict[str, str]
        각 단어의 top1 group
    word_top2 : dict[str, tuple[str, str | None]]
        (top1 group, bridge 성립 시 top2 group else None)
    """

    # --- fallback to globals if not provided ---
    if min_logodds is None:
        min_logodds = globals().get("BRIDGE_MIN_LOGODDS", 0.0)
    if min_count_per_group is None:
        min_count_per_group = globals().get("BRIDGE_MIN_COUNT", 0)

    required_cols = {"word", "group", "log_odds"}
    missing = required_cols - set(abuse_stats_logodds.columns)
    if missing:
        raise ValueError(f"abuse_stats_logodds is missing required columns: {sorted(missing)}")

    df = abuse_stats_logodds.copy()

    # default count column
    if "count" not in df.columns:
        df["count"] = 0

    # drop non-finite log_odds rows if requested
    if drop_nonfinite:
        df = df[np.isfinite(df["log_odds"].astype(float))]

    word_main_abuse = {}
    word_top2 = {}

    # NOTE: groupby preserves word; ensure stable sorting
    for w, sub in df.groupby("word", sort=False):
        if sub.empty:
            continue

        # total frequency filter (rare-word stabilizer)
        if min_total is not None:
            total_c = float(np.nansum(sub["count"].astype(float).values))
            if total_c < min_total:
                continue

        sub_sorted = sub.sort_values("log_odds", ascending=False)

        row1 = sub_sorted.iloc[0]
        g1 = row1["group"]
        l1 = float(row1["log_odds"])
        z1 = row1["z_log_odds"] if "z_log_odds" in sub_sorted.columns else np.nan
        c1 = float(row1.get("count", 0))

        g2 = None

        if len(sub_sorted) > 1:
            row2 = sub_sorted.iloc[1]
            g2_candidate = row2["group"]
            l2 = float(row2["log_odds"])
            z2 = row2["z_log_odds"] if "z_log_odds" in sub_sorted.columns else np.nan
            c2 = float(row2.get("count", 0))

            # base condition: top2가 "가까워야" bridge (경계/중첩)
            cond_gap = (l1 - l2) < diff_thresh

            # enrichment sign condition
            if require_positive:
                cond_sign = (l1 > 0.0) and (l2 > 0.0)
            else:
                cond_sign = True

            # strength and support
            cond_strength = (l1 >= float(min_logodds)) and (l2 >= float(min_logodds))
            cond_count = (c1 >= float(min_count_per_group)) and (c2 >= float(min_count_per_group))

            # z filter
            cond_z = True
            if z_min is not None:
                if "z_log_odds" not in sub_sorted.columns:
                    cond_z = False  # z_min을 요구하는데 z가 없으면 탈락이 안전
                else:
                    if strict_z:
                        cond_z = np.isfinite(z1) and np.isfinite(z2) and (float(z1) >= z_min) and (float(z2) >= z_min)
                    else:
                        # 결측 z는 무시(완화)하고, 있는 값만 체크
                        z_ok1 = (not np.isfinite(z1)) or (float(z1) >= z_min)
                        z_ok2 = (not np.isfinite(z2)) or (float(z2) >= z_min)
                        cond_z = z_ok1 and z_ok2

            if cond_gap and cond_sign and cond_strength and cond_count and cond_z:
                g2 = g2_candidate

        word_main_abuse[w] = g1
        word_top2[w] = (g1, g2)

    return word_main_abuse, word_top2


def _compute_top2_prob_stats(df_counts, group_cols):
    """
    df_counts: index=word, columns=group_cols (doc-level presence counts 권장)
    반환: word별 total_freq, p1,p2,gap, entropy, top2_mass, primary, secondary
    """
    rows = []
    eps = 1e-12

    use = df_counts[group_cols].astype(float)

    for w, r in use.iterrows():
        total = float(r.sum())
        if total <= 0:
            continue

        probs = (r.values / total).astype(float)
        idx = np.argsort(-probs)
        k1, k2 = int(idx[0]), int(idx[1])
        p1, p2 = float(probs[k1]), float(probs[k2])

        g1, g2 = group_cols[k1], group_cols[k2]
        gap = p1 - p2
        ent = float(-(probs * np.log(probs + eps)).sum())
        top2_mass = float(p1 + p2)

        rows.append({
            "word": w,
            "total_freq": total,
            "log_total_freq": float(np.log1p(total)),
            "primary": g1,
            "secondary": g2,
            "p1": p1,
            "p2": p2,
            "gap": gap,
            "entropy": ent,
            "top2_mass": top2_mass,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("word")


def _assign_freq_bins(stats_df, n_bins=12):
    """
    log_total_freq 기반 qcut으로 빈도 bin 할당 (중복 많으면 자동 축소)
    """
    if stats_df.empty:
        return stats_df

    q = int(max(2, n_bins))
    x = stats_df["log_total_freq"].astype(float)

    # qcut이 중복 때문에 실패하는 케이스를 방어
    try:
        bins = pd.qcut(x, q=q, duplicates="drop")
    except Exception:
        # fallback: unique 개수 기반으로 cut
        uniq = int(pd.Series(x).nunique())
        nb = int(min(q, max(2, uniq)))
        bins = pd.cut(x, bins=nb, duplicates="drop")

    out = stats_df.copy()
    out["freq_bin"] = bins.astype(str)
    return out


def _bridge_set_from_cfg(df_counts, logodds_df, cfg, group_cols, words_scope=None, count_min=None):
    """
    cfg(B0/loose/strict)로 bridge set 생성
    """
    if words_scope is None:
        words_scope = df_counts.index.tolist()

    if count_min is None:
        count_min = cfg.get("count_min", BRIDGE_MIN_COUNT)

    df_bridge = compute_prob_bridge_for_words(
        df_counts=df_counts,
        words=words_scope,
        logodds_df=logodds_df,   # count_min 적용을 위해 권장
        min_p1=cfg["min_p1"],
        min_p2=cfg["min_p2"],
        max_gap=cfg["max_gap"],
        logodds_min=None,
        count_min=count_min,
        z_min=None,
    )
    if df_bridge is None or df_bridge.empty:
        return set(), pd.DataFrame()

    return set(df_bridge["word"].tolist()), df_bridge


def run_frequency_matched_baseline_for_bridge(
    df_counts,
    logodds_df,
    cfg,
    group_cols=ABUSE_ORDER,
    words_scope=None,
    n_bins=12,
    n_iter=1000,
    out_dir=None,
    tag="ALL",
    random_state=42,
):
    """
    저장물:
      1) freqmatched_bin_summary_{tag}_{cfg}.csv
      2) freqmatched_baseline_summary_{tag}_{cfg}.csv
      3) (옵션) freqmatched_baseline_iters_{tag}_{cfg}.csv (n_iter 크면 커짐)
      4) 히스토그램 png 2장 (mean_p2, mean_gap)

    해석 포인트:
      - 같은 빈도 bin에서도 bridge rate이 낮으면: "고빈도면 다 bridge"가 아님
      - frequency-matched random 샘플의 mean_p2/mean_gap 분포 대비,
        bridge의 mean_p2(↑)/mean_gap(↓)가 극단이면: "빈도 통제해도 선별이 작동"
    """
    if out_dir is None:
        out_dir = os.path.join(ABUSE_STATS_DIR, "freq_matched")
    os.makedirs(out_dir, exist_ok=True)

    cfg_name = cfg.get("name", "CFG")
    if words_scope is None:
        words_scope = df_counts.index.tolist()

    # 1) bridge set
    bridge_set, df_bridge = _bridge_set_from_cfg(
        df_counts=df_counts,
        logodds_df=logodds_df,
        cfg=cfg,
        group_cols=group_cols,
        words_scope=words_scope,
        count_min=cfg.get("count_min", BRIDGE_MIN_COUNT),
    )
    if not bridge_set:
        print(f"[FREQMATCH] {tag}/{cfg_name}: bridge가 0개라 실험을 건너뜁니다.")
        return None

    # 2) word stats + bin
    stats = _compute_top2_prob_stats(df_counts.loc[words_scope], group_cols)
    if stats.empty:
        print("[FREQMATCH] stats 생성 실패(빈 테이블).")
        return None

    stats["is_bridge"] = stats.index.isin(bridge_set).astype(int)
    stats = _assign_freq_bins(stats, n_bins=n_bins)

    # 3) bin summary (같은 빈도대에서 bridge 비율이 낮은지)
    bin_rows = []
    for b, sub in stats.groupby("freq_bin"):
        n_all = len(sub)
        n_br = int(sub["is_bridge"].sum())
        rate = (n_br / n_all) if n_all > 0 else np.nan
        bin_rows.append({
            "tag": tag,
            "config": cfg_name,
            "freq_bin": b,
            "n_words": n_all,
            "n_bridge": n_br,
            "bridge_rate": rate,
            "median_total_freq": float(sub["total_freq"].median()),
            "min_total_freq": float(sub["total_freq"].min()),
            "max_total_freq": float(sub["total_freq"].max()),
        })
    df_bin = pd.DataFrame(bin_rows).sort_values("median_total_freq")
    bin_path = os.path.join(out_dir, f"freqmatched_bin_summary_{tag}_{cfg_name}.csv")
    df_bin.to_csv(bin_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 빈도 bin별 bridge 비율 -> {bin_path}")

    # 4) frequency-matched baseline sampling
    #    - bridge set과 동일한 "bin 분포"로 단어를 무작위 샘플링
    rng = np.random.default_rng(random_state)

    # bin -> (words list, bridge_count_in_bin)
    bin_to_words = {b: sub.index.tolist() for b, sub in stats.groupby("freq_bin")}
    bin_to_k = {b: int(sub["is_bridge"].sum()) for b, sub in stats.groupby("freq_bin")}

    # 관측(bridge set)의 요약 통계
    obs = stats.loc[list(bridge_set)]
    obs_mean_p2 = float(obs["p2"].mean())
    obs_mean_gap = float(obs["gap"].mean())
    obs_mean_ent = float(obs["entropy"].mean())

    it_rows = []
    samp_mean_p2 = []
    samp_mean_gap = []
    samp_mean_ent = []
    samp_bridge_rate = []

    for it in range(int(n_iter)):
        sampled_words = []

        for b, k in bin_to_k.items():
            if k <= 0:
                continue
            pool = bin_to_words.get(b, [])
            if not pool:
                continue

            # bin 내 단어수가 k보다 작으면 replacement 허용
            replace = (len(pool) < k)
            picks = rng.choice(pool, size=k, replace=replace)
            sampled_words.extend(list(picks))

        if not sampled_words:
            continue

        samp = stats.loc[stats.index.intersection(sampled_words)]
        if samp.empty:
            continue

        mean_p2 = float(samp["p2"].mean())
        mean_gap = float(samp["gap"].mean())
        mean_ent = float(samp["entropy"].mean())
        br_rate = float(samp["is_bridge"].mean())  # 랜덤 샘플 중 bridge 비율(=기대 base rate)

        samp_mean_p2.append(mean_p2)
        samp_mean_gap.append(mean_gap)
        samp_mean_ent.append(mean_ent)
        samp_bridge_rate.append(br_rate)

        it_rows.append({
            "iter": it + 1,
            "mean_p2": mean_p2,
            "mean_gap": mean_gap,
            "mean_entropy": mean_ent,
            "bridge_rate_in_sample": br_rate,
        })

    if not it_rows:
        print("[FREQMATCH] baseline 샘플 결과가 비어 있습니다.")
        return None

    df_iters = pd.DataFrame(it_rows)
    it_path = os.path.join(out_dir, f"freqmatched_baseline_iters_{tag}_{cfg_name}.csv")
    df_iters.to_csv(it_path, encoding="utf-8-sig", index=False)
    print(f"[저장] frequency-matched baseline 반복분포 -> {it_path}")

    # p-value 스타일 요약 (랜덤 baseline에서 obs보다 “극단”일 확률)
    #  - p2는 obs가 더 크면 극단(>=)
    #  - gap은 obs가 더 작으면 극단(<=)
    #  - entropy는 방향 애매하면 두 방향 다 보고 싶을 때는 양측으로(여기선 obs>=)
    samp_mean_p2 = np.array(samp_mean_p2, dtype=float)
    samp_mean_gap = np.array(samp_mean_gap, dtype=float)
    samp_mean_ent = np.array(samp_mean_ent, dtype=float)
    samp_bridge_rate = np.array(samp_bridge_rate, dtype=float)

    p_p2 = float((1 + np.sum(samp_mean_p2 >= obs_mean_p2)) / (len(samp_mean_p2) + 1))
    p_gap = float((1 + np.sum(samp_mean_gap <= obs_mean_gap)) / (len(samp_mean_gap) + 1))
    p_ent = float((1 + np.sum(samp_mean_ent >= obs_mean_ent)) / (len(samp_mean_ent) + 1))

    summary = pd.DataFrame([{
        "tag": tag,
        "config": cfg_name,
        "n_words_scope": int(len(stats)),
        "n_bridge": int(len(bridge_set)),
        "obs_mean_p2": obs_mean_p2,
        "obs_mean_gap": obs_mean_gap,
        "obs_mean_entropy": obs_mean_ent,
        "baseline_mean_p2_mean": float(np.mean(samp_mean_p2)),
        "baseline_mean_p2_ci2.5": float(np.quantile(samp_mean_p2, 0.025)),
        "baseline_mean_p2_ci97.5": float(np.quantile(samp_mean_p2, 0.975)),
        "p_value_obs_mean_p2_ge_baseline": p_p2,
        "baseline_mean_gap_mean": float(np.mean(samp_mean_gap)),
        "baseline_mean_gap_ci2.5": float(np.quantile(samp_mean_gap, 0.025)),
        "baseline_mean_gap_ci97.5": float(np.quantile(samp_mean_gap, 0.975)),
        "p_value_obs_mean_gap_le_baseline": p_gap,
        "baseline_bridge_rate_mean": float(np.mean(samp_bridge_rate)),
        "baseline_bridge_rate_ci2.5": float(np.quantile(samp_bridge_rate, 0.025)),
        "baseline_bridge_rate_ci97.5": float(np.quantile(samp_bridge_rate, 0.975)),
        "p_value_obs_entropy_ge_baseline": p_ent,
    }])

    sum_path = os.path.join(out_dir, f"freqmatched_baseline_summary_{tag}_{cfg_name}.csv")
    summary.to_csv(sum_path, encoding="utf-8-sig", index=False)
    print(f"[저장] frequency-matched baseline 요약 -> {sum_path}")

    # 간단 시각화(히스토그램) 저장
    plt.figure(figsize=(7, 5))
    plt.hist(samp_mean_p2, bins=30, alpha=0.8)
    plt.axvline(obs_mean_p2, linewidth=2)
    plt.title(f"Freq-matched baseline: mean(p2) vs bridge (tag={tag}, cfg={cfg_name})")
    plt.xlabel("mean p2 (sample)")
    plt.ylabel("count")
    plt.tight_layout()
    p2_fig = os.path.join(out_dir, f"freqmatched_hist_mean_p2_{tag}_{cfg_name}.png")
    plt.savefig(p2_fig, dpi=200)
    plt.close()
    print(f"[저장] 히스토그램(mean p2) -> {p2_fig}")

    plt.figure(figsize=(7, 5))
    plt.hist(samp_mean_gap, bins=30, alpha=0.8)
    plt.axvline(obs_mean_gap, linewidth=2)
    plt.title(f"Freq-matched baseline: mean(gap) vs bridge (tag={tag}, cfg={cfg_name})")
    plt.xlabel("mean gap (sample)")
    plt.ylabel("count")
    plt.tight_layout()
    gap_fig = os.path.join(out_dir, f"freqmatched_hist_mean_gap_{tag}_{cfg_name}.png")
    plt.savefig(gap_fig, dpi=200)
    plt.close()
    print(f"[저장] 히스토그램(mean gap) -> {gap_fig}")

    return {
        "bridge_df": df_bridge,
        "bin_summary": df_bin,
        "baseline_iters": df_iters,
        "baseline_summary": summary,
        "out_dir": out_dir,
    }
