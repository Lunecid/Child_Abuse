from __future__ import annotations

from . import common as C
from .labels import classify_child_group, classify_abuse_main_sub
from .text import extract_child_speech, tokenize_korean
from .stats import *


def build_doc_level_valence_counts(json_files, allowed_groups=None):
    """
    문서 단위 정서군 × 단어 교차표.
    allowed_groups가 주어지면 해당 정서군에 속한 아동만 포함.
    """
    rows = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        group = classify_child_group(rec)
        if group not in VALENCE_ORDER:
            continue
        if allowed_groups is not None and group not in allowed_groups:
            continue

        speech_list = extract_child_speech(rec)
        if not speech_list:
            continue

        joined = " ".join(speech_list)
        tokens = set(tokenize_korean(joined))

        for w in tokens:
            rows.append({"group": group, "word": w})

    if not rows:
        return pd.DataFrame()

    df_doc = pd.DataFrame(rows)
    df_doc_counts = (
        df_doc
        .groupby(["word", "group"])
        .size()
        .unstack("group")
        .fillna(0)
    )

    for g in VALENCE_ORDER:
        if g not in df_doc_counts.columns:
            df_doc_counts[g] = 0
    df_doc_counts = df_doc_counts[VALENCE_ORDER]

    df_doc_counts["total_docs"] = df_doc_counts.sum(axis=1)
    df_doc_counts = df_doc_counts[df_doc_counts["total_docs"] >= MIN_DOC_COUNT]
    df_doc_counts = df_doc_counts.drop(columns=["total_docs"])

    return df_doc_counts


def build_abuse_doc_word_table(json_files, allowed_groups=None):
    """
    문서(사례) 단위 단어-학대 테이블을 생성.
    - doc_word_df : (doc_id, group, main_abuse, word) 행마다 '단어가 한 번 이상 등장함'을 의미
    - doc_meta    : doc_id별 메타 (child_id, group, main_abuse)

    allowed_groups: {"부정"} 등 정서군 필터. None이면 전체.
    """
    doc_rows = []
    meta_rows = []
    seen_docs = set()

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        info = rec.get("info", {})
        child_id = info.get("ID")
        group = classify_child_group(rec)

        if group not in VALENCE_ORDER:
            continue
        if allowed_groups is not None and group not in allowed_groups:
            continue

        main_abuse, _subs = classify_abuse_main_sub(rec)
        if main_abuse not in ABUSE_ORDER:
            continue

        # 파일명을 doc_id로 사용 (유일 보장)
        doc_id = os.path.splitext(os.path.basename(path))[0]

        speech_list = extract_child_speech(rec)
        if not speech_list:
            continue
        joined = " ".join(speech_list)
        tokens = set(tokenize_korean(joined))
        if not tokens:
            continue

        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            meta_rows.append({
                "doc_id": doc_id,
                "child_id": child_id,
                "group": group,
                "main_abuse": main_abuse,
            })

        for w in tokens:
            doc_rows.append({
                "doc_id": doc_id,
                "group": group,
                "main_abuse": main_abuse,
                "word": w,
            })

    if not doc_rows:
        print("[DOC] build_abuse_doc_word_table: 유효한 doc/word가 없습니다.")
        return pd.DataFrame(), pd.DataFrame()

    doc_word_df = pd.DataFrame(doc_rows)
    doc_meta = pd.DataFrame(meta_rows).drop_duplicates(subset=["doc_id"])

    return doc_word_df, doc_meta


def run_doc_level_label_shuffle_permutation(
    doc_word_df,
    doc_meta,
    n_perm: int = 1000,
    min_doc_count: int = MIN_DOC_COUNT,
    out_dir: str = ABUSE_STATS_DIR,
    scope_words=None,
    tag: str = "ALL",
):
    """
    Parameters
    ----------
    doc_word_df : DataFrame
        columns = ["doc_id","group","main_abuse","word"] (presence=1)
    doc_meta : DataFrame
        columns = ["doc_id","child_id","group","main_abuse"]
    n_perm : int
        label shuffle 횟수 (예: 300~1000)
    min_doc_count : int
        한 단어가 등장한 doc 수 최소값
    scope_words : list or None
        None이면 doc_word_df에 등장하는 모든 단어 사용
        아니면 지정된 단어 subset만 사용 (예: chi² 상위 200개)
    tag : str
        파일명에 붙일 태그 (예: "ALL", "NEG_ONLY")
    """
    if doc_word_df.empty or doc_meta.empty:
        print("[DOC-PERM] 입력 데이터가 비어 있어 퍼뮤테이션을 건너뜁니다.")
        return None


    out_dir = os.path.join(C.ABUSE_STATS_DIR, "doc_level", str(tag))
    os.makedirs(out_dir, exist_ok=True)

    # main_abuse 유효 doc만 사용
    meta = doc_meta[doc_meta["main_abuse"].isin(ABUSE_ORDER)].copy()
    if meta.empty:
        print("[DOC-PERM] main_abuse 유효 doc이 없습니다.")
        return None

    # doc_word_df도 해당 doc으로 제한
    doc_ids_valid = set(meta["doc_id"])
    dw = doc_word_df[doc_word_df["doc_id"].isin(doc_ids_valid)].copy()
    if dw.empty:
        print("[DOC-PERM] 유효 doc에 해당하는 단어가 없습니다.")
        return None

    # 각 doc의 abuse label 인덱스화
    meta = meta.sort_values("doc_id")
    doc_ids = meta["doc_id"].tolist()
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    abuse2idx = {a: i for i, a in enumerate(ABUSE_ORDER)}
    K = len(ABUSE_ORDER)

    labels_int = np.array([abuse2idx[ab] for ab in meta["main_abuse"]])
    # abuse별 doc 수
    counts_abuse = np.bincount(labels_int, minlength=K)
    total_docs = int(counts_abuse.sum())

    # (doc, word) 중복 제거 (이미 set으로 만들었지만 혹시 몰라 방어)
    dw = dw[["doc_id", "word"]].drop_duplicates()

    # 단어별 doc index 리스트
    word_docs = {}
    for doc_id, word in dw.itertuples(index=False):
        idx = doc_index.get(doc_id)
        if idx is None:
            continue
        word_docs.setdefault(word, []).append(idx)

    # 분석 대상 단어 스코프 구성
    if scope_words is None:
        candidate_words = list(word_docs.keys())
    else:
        candidate_words = [w for w in scope_words if w in word_docs]

    # doc 빈도 필터
    words_final = []
    for w in candidate_words:
        n_w = len(word_docs[w])
        if n_w >= min_doc_count:
            words_final.append(w)

    if not words_final:
        print("[DOC-PERM] min_doc_count 조건을 만족하는 단어가 없습니다.")
        return None

    print(f"[DOC-PERM] 대상 단어 수 = {len(words_final)}, n_perm={n_perm}")

    # 관측 chi² 계산 함수
    def _chi2_from_contingency(row_present, col_totals):
        """
        row_present: shape (K,)  (단어가 등장한 doc 수, abuse별)
        col_totals: shape (K,)  (전체 doc 수, abuse별)
        """
        row1 = row_present.astype(float)
        n1 = row1.sum()
        if n1 <= 0 or n1 >= col_totals.sum():
            return 0.0

        row0 = col_totals - row1
        obs = np.vstack([row1, row0])  # 2 x K

        row_sums = obs.sum(axis=1, keepdims=True)
        col_sums = obs.sum(axis=0, keepdims=True)
        N = obs.sum()
        expected = row_sums @ col_sums / N

        with np.errstate(divide="ignore", invalid="ignore"):
            chi_sq = (obs - expected) ** 2 / expected
            chi_sq = np.nan_to_num(chi_sq, nan=0.0, posinf=0.0, neginf=0.0)
        return float(chi_sq.sum())

    # 관측 chi²
    obs_chi = {}
    for w in words_final:
        idxs = word_docs[w]
        if not idxs:
            continue
        present_labels = labels_int[idxs]
        counts_present = np.bincount(present_labels, minlength=K)
        obs_chi[w] = _chi2_from_contingency(counts_present, counts_abuse)

    # 퍼뮤테이션 카운트 (chi_perm >= chi_obs)
    ge_count = {w: 0 for w in words_final}

    rng = np.random.default_rng(42)
    for b in range(n_perm):
        perm_labels = rng.permutation(labels_int)
        # abuse별 doc 총수는 그대로라 counts_abuse 재계산 필요 없음
        for w in words_final:
            idxs = word_docs[w]
            if not idxs:
                continue
            labels_b = perm_labels[idxs]
            counts_present_b = np.bincount(labels_b, minlength=K)
            chi_b = _chi2_from_contingency(counts_present_b, counts_abuse)
            if chi_b >= obs_chi[w]:
                ge_count[w] += 1
        if (b + 1) % 50 == 0:
            print(f"[DOC-PERM] permutation {b+1}/{n_perm} 완료")

    rows = []
    for w in words_final:
        chi_obs = obs_chi.get(w, 0.0)
        count_ge = ge_count.get(w, 0)
        p_perm = (count_ge + 1) / (n_perm + 1)  # +1 smoothing
        rows.append({
            "word": w,
            "chi2_doc_level": chi_obs,
            "n_perm": n_perm,
            "n_perm_ge": count_ge,
            "p_perm": p_perm,
        })

    df_perm = pd.DataFrame(rows).sort_values("chi2_doc_level", ascending=False)

    out_path = os.path.join(
        out_dir,
        f"doc_level_label_shuffle_chi2_{tag}.csv",
    )
    df_perm.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 문서 단위 라벨 셔플 퍼뮤테이션 결과 -> {out_path}")

    return df_perm


def build_doc_level_abuse_counts(json_files, allowed_groups=None):
    """
    문서 단위 main_abuse × 단어 교차표.
    allowed_groups가 주어지면 해당 정서군(label)에 속한 아동만 포함.
    allowed_groups가 None이면 전체 JSON을 사용 (기존 동작 유지).
    """
    rows = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        if allowed_groups is not None:
            group = classify_child_group(rec)
            if group not in allowed_groups:
                continue

        main_abuse, _ = classify_abuse_main_sub(rec)
        if main_abuse not in ABUSE_ORDER:
            continue

        speech_list = extract_child_speech(rec)
        if speech_list:
            joined = " ".join(speech_list)
            tokens = set(tokenize_korean(joined))
            for w in tokens:
                rows.append({"abuse": main_abuse, "word": w})

    if not rows:
        return pd.DataFrame()

    df_doc = pd.DataFrame(rows)
    df_counts = (
        df_doc
        .groupby(["word", "abuse"])
        .size()
        .unstack("abuse")
        .fillna(0)
    )

    for a in ABUSE_ORDER:
        if a not in df_counts.columns:
            df_counts[a] = 0
    df_counts = df_counts[ABUSE_ORDER]

    df_counts["total_docs"] = df_counts.sum(axis=1)
    df_counts = df_counts[df_counts["total_docs"] >= MIN_DOC_COUNT]
    df_counts = df_counts.drop(columns=["total_docs"])

    return df_counts


def run_bridge_prob_ablation(df_counts,
                             chi_df,
                             logodds_df=None,
                             output_dir=BRIDGE_PROB_ABLATION_DIR,
                             ca_top_k=200,
                             p_configs=BRIDGE_P_CONFIGS,
                             count_min=BRIDGE_MIN_COUNT):
    configs = p_configs

    summary_rows = []
    pair_rows = []
    bridge_sets = {}

    scopes = []


    # 2) CA용 chi2 상위 단어 스코프
    chi_sorted = chi_df.sort_values("chi2", ascending=False) if not chi_df.empty else chi_df
    if not chi_sorted.empty:
        ca_words = chi_sorted.head(min(ca_top_k, len(chi_sorted))).index.tolist()
        scopes.append((f"ca_top{len(ca_words)}", ca_words))

    for scope_name, words in scopes:
        print(f"[ABLT] scope = {scope_name}, 후보 단어 수 = {len(words)}")

        scope_dir = os.path.join(output_dir, scope_name)
        bridges_dir = os.path.join(scope_dir, "bridge_lists")
        presence_dir = os.path.join(scope_dir, "presence")
        stable_dir = os.path.join(scope_dir, "stable")
        os.makedirs(bridges_dir, exist_ok=True)
        os.makedirs(presence_dir, exist_ok=True)
        os.makedirs(stable_dir, exist_ok=True)

        baseline_set = None

        for cfg in configs:
            cfg_name = cfg["name"]
            min_p1 = cfg["min_p1"]
            min_p2 = cfg["min_p2"]
            max_gap = cfg["max_gap"]

            print(
                f"[ABLT] scope={scope_name}, config={cfg_name}: "
                f"min_p1={min_p1}, min_p2={min_p2}, max_gap={max_gap}, "
                f"count_min={count_min}"
            )

            df_bridge = compute_prob_bridge_for_words(
                df_counts=df_counts,
                words=words,
                logodds_df=logodds_df,
                min_p1=min_p1,
                min_p2=min_p2,
                max_gap=max_gap,
                logodds_min=None,
                count_min=count_min,
                z_min=None,
            )

            n_candidates = len(words)
            n_bridge = len(df_bridge)
            ratio = n_bridge / n_candidates if n_candidates > 0 else 0.0

            out_bridge_path = os.path.join(
                bridges_dir,
                f"bridge_prob_{scope_name}_{cfg_name}.csv",
            )
            df_bridge.to_csv(out_bridge_path, encoding="utf-8-sig", index=False)
            print(f"[저장] {scope_name} / {cfg_name} bridge 목록 -> {out_bridge_path}")

            if df_bridge.empty:
                word_set = set()
            else:
                word_set = set(df_bridge["word"].tolist())
            bridge_sets[(scope_name, cfg_name)] = word_set

            # --- 통계적 검정: chi² / p-value 요약 (bridge vs non-bridge) ---
            mean_chi_bridge = np.nan
            median_chi_bridge = np.nan
            mean_chi_nonbridge = np.nan
            median_chi_nonbridge = np.nan
            n_sig_005 = np.nan
            n_sig_001 = np.nan
            ratio_sig_005 = np.nan
            ratio_sig_001 = np.nan
            t_chi = np.nan
            p_t_chi = np.nan

            if not chi_df.empty and n_bridge > 0:
                # scope 안에서 chi² 정보가 있는 단어만 사용
                chi_scope = chi_df.loc[chi_df.index.intersection(words)].copy()
                if not chi_scope.empty:
                    mask_bridge = chi_scope.index.isin(word_set)
                    chi_bridge = chi_scope[mask_bridge]
                    chi_nonbridge = chi_scope[~mask_bridge]

                    if not chi_bridge.empty:
                        chi_vals_b = chi_bridge["chi2"].astype(float)
                        mean_chi_bridge = float(chi_vals_b.mean())
                        median_chi_bridge = float(chi_vals_b.median())

                        if "p_value" in chi_bridge.columns:
                            pvals_b = chi_bridge["p_value"].astype(float)
                            pvals_b = pvals_b.replace([np.inf, -np.inf], np.nan).dropna()
                            if not pvals_b.empty:
                                n_sig_005 = int((pvals_b < 0.05).sum())
                                n_sig_001 = int((pvals_b < 0.01).sum())
                                ratio_sig_005 = n_sig_005 / len(pvals_b)
                                ratio_sig_001 = n_sig_001 / len(pvals_b)

                    if not chi_nonbridge.empty:
                        chi_vals_nb = chi_nonbridge["chi2"].astype(float)
                        mean_chi_nonbridge = float(chi_vals_nb.mean())
                        median_chi_nonbridge = float(chi_vals_nb.median())

                    # t-test (bridge vs non-bridge chi²)
                    if HAS_SCIPY and len(chi_bridge) >= 2 and len(chi_nonbridge) >= 2:
                        try:
                            t_chi, p_t_chi = ttest_ind(
                                chi_bridge["chi2"].astype(float),
                                chi_nonbridge["chi2"].astype(float),
                                equal_var=False,
                            )
                        except Exception:
                            t_chi, p_t_chi = np.nan, np.nan

                # bridge 단어별 chi² 상세 테이블 (per-word)
                if not df_bridge.empty:
                    chi_detail = df_bridge.merge(
                        chi_df.reset_index(), on="word", how="left"
                    )
                    chi_detail_path = os.path.join(
                        bridges_dir,
                        f"bridge_prob_{scope_name}_{cfg_name}_chi_detail.csv",
                    )
                    chi_detail.to_csv(chi_detail_path, encoding="utf-8-sig", index=False)
                    print(f"[저장] {scope_name} / {cfg_name} bridge chi² 상세 -> {chi_detail_path}")

            # --- 학대유형 pair 카운트 ---
            if not df_bridge.empty:
                df_pairs_cfg = df_bridge.copy()
                df_pairs_cfg["pair"] = df_pairs_cfg.apply(
                    lambda r: "-".join(sorted([r["primary_abuse"], r["secondary_abuse"]])),
                    axis=1,
                )
                pair_counts = df_pairs_cfg.groupby("pair").size().to_dict()
            else:
                pair_counts = {}

            for pair, cnt in pair_counts.items():
                pair_rows.append({
                    "scope": scope_name,
                    "config": cfg_name,
                    "pair": pair,
                    "count": int(cnt),
                })

            # baseline(B0_baseline)를 기준으로 Jaccard
            if cfg_name == "B0_baseline":
                baseline_set = word_set

            if baseline_set is None:
                jaccard = None
            else:
                if baseline_set or word_set:
                    inter = len(baseline_set & word_set)
                    union = len(baseline_set | word_set)
                    jaccard = inter / union if union > 0 else None
                else:
                    jaccard = None

            summary_rows.append({
                "scope": scope_name,
                "config": cfg_name,
                "min_p1": min_p1,
                "min_p2": min_p2,
                "max_gap": max_gap,
                "count_min": count_min,
                "n_candidates": n_candidates,
                "n_bridge": n_bridge,
                "ratio_bridge": ratio,
                "jaccard_vs_B0_baseline": jaccard,
                "mean_chi2_bridge": mean_chi_bridge,
                "median_chi2_bridge": median_chi_bridge,
                "mean_chi2_nonbridge": mean_chi_nonbridge,
                "median_chi2_nonbridge": median_chi_nonbridge,
                "n_sig_p_lt_0_05": n_sig_005,
                "ratio_sig_p_lt_0_05": ratio_sig_005,
                "n_sig_p_lt_0_01": n_sig_001,
                "ratio_sig_p_lt_0_01": ratio_sig_001,
                "t_chi_bridge_vs_nonbridge": t_chi,
                "p_t_chi_bridge_vs_nonbridge": p_t_chi,
            })

        # ------------ presence matrix + stable bridge (config 간 강건성) ------------

        cfg_order = [
            cfg["name"]
            for cfg in configs
            if (scope_name, cfg["name"]) in bridge_sets
        ]
        if not cfg_order:
            continue

        sets = [bridge_sets[(scope_name, cfg_name)] for cfg_name in cfg_order]
        if not any(len(s) > 0 for s in sets):
            # 어떤 config에서도 bridge가 안 나왔으면 presence/stable 파일 만들 필요 없음
            continue

        n_cfg = len(sets)
        all_words_scope = set().union(*sets)
        presence_rows = []

        for w in sorted(all_words_scope):
            row = {
                "scope": scope_name,
                "word": w,
            }
            presence_flags = []
            for cfg_name, s in zip(cfg_order, sets):
                present = int(w in s)
                row[f"in_{cfg_name}"] = present
                presence_flags.append(present)
            n_present = int(sum(presence_flags))
            ratio_present = n_present / n_cfg
            row["n_configs"] = n_present
            row["ratio_configs"] = ratio_present
            presence_rows.append(row)

        df_presence = pd.DataFrame(presence_rows)
        presence_path = os.path.join(
            presence_dir,
            f"bridge_prob_{scope_name}_presence_matrix.csv",
        )
        df_presence.to_csv(presence_path, encoding="utf-8-sig", index=False)
        print(f"[저장] {scope_name} / presence matrix -> {presence_path}")

        # 모든 config 공통 bridge
        stable_all = df_presence[df_presence["n_configs"] == n_cfg].copy()
        stable_all_path = os.path.join(
            stable_dir,
            f"bridge_prob_{scope_name}_stable_all_configs.csv",
        )
        stable_all.to_csv(stable_all_path, encoding="utf-8-sig", index=False)
        print(
            f"[저장] {scope_name} / 모든 config 공통 bridge "
            f"({len(stable_all)}개) -> {stable_all_path}"
        )

        # 다수결 기준 bridge (>= 절반 config)
        threshold = max(2, int(np.ceil(n_cfg * 0.5)))
        stable_majority = df_presence[df_presence["n_configs"] >= threshold].copy()
        stable_maj_path = os.path.join(
            stable_dir,
            f"bridge_prob_{scope_name}_stable_majority_configs.csv",
        )
        stable_majority.to_csv(stable_maj_path, encoding="utf-8-sig", index=False)
        print(
            f"[저장] {scope_name} / 다수결 기준(>= {threshold}/{n_cfg}) "
            f"stable bridge ({len(stable_majority)}개) -> {stable_maj_path}"
        )

    # 전체 summary / pair 분포 저장
    df_summary = pd.DataFrame(summary_rows)
    df_pairs_all = pd.DataFrame(pair_rows)

    summary_path = os.path.join(output_dir, "bridge_prob_ablation_summary.csv")
    pairs_path = os.path.join(output_dir, "bridge_prob_ablation_pairs.csv")

    df_summary.to_csv(summary_path, encoding="utf-8-sig", index=False)
    df_pairs_all.to_csv(pairs_path, encoding="utf-8-sig", index=False)

    print(f"[저장] bridge p-threshold ablation summary -> {summary_path}")
    print(f"[저장] bridge p-threshold pair 분해 결과 -> {pairs_path}")


def run_bridge_bootstrap_and_shuffle_doc_level(
    doc_word_df,
    doc_meta,
    p_configs=BRIDGE_P_CONFIGS,
    n_bootstrap: int = 200,
    n_shuffle: int = 200,
    count_min: int = BRIDGE_MIN_COUNT,
    out_dir: str = BRIDGE_PROB_ABLATION_DIR,
    tag: str = "docLevel",
):
    """
    문서 단위 presence를 기반으로 p-bridge 조건(BRIDGE_P_CONFIGS)을 적용해
    - bootstrap: 문서 부트스트랩으로 bridge 선정 확률
    - shuffle-null: main_abuse 라벨 셔플로 null 하에서의 선정 확률
    을 계산.

    결과: bridge_bootstrap_shuffle_{tag}.csv
        word, config, sel_count_boot, sel_prob_boot, sel_count_shuffle, sel_prob_shuffle
    """
    if doc_word_df.empty or doc_meta.empty:
        print("[BRIDGE-BOOT] 입력 데이터가 비어 있어 분석을 건너뜁니다.")
        return None

    os.makedirs(out_dir, exist_ok=True)

    meta = doc_meta[doc_meta["main_abuse"].isin(ABUSE_ORDER)].copy()
    if meta.empty:
        print("[BRIDGE-BOOT] main_abuse 유효 doc이 없습니다.")
        return None

    # doc 목록 / label index화
    meta = meta.sort_values("doc_id")
    doc_ids = meta["doc_id"].tolist()
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    abuse2idx = {a: i for i, a in enumerate(ABUSE_ORDER)}
    K = len(ABUSE_ORDER)

    labels_int = np.array([abuse2idx[ab] for ab in meta["main_abuse"]])
    N_doc = len(doc_ids)

    # doc_word_df 정리: 유효 doc만, (doc, word) presence
    dw = doc_word_df[doc_word_df["doc_id"].isin(doc_ids)][["doc_id", "word"]]
    dw = dw.drop_duplicates()

    # doc_idx -> 단어 집합
    doc_to_words = {i: set() for i in range(N_doc)}
    for doc_id, word in dw.itertuples(index=False):
        idx = doc_index.get(doc_id)
        if idx is None:
            continue
        doc_to_words[idx].add(word)

    # 단어 전체 vocabulary
    vocab = sorted({w for ws in doc_to_words.values() for w in ws})
    if not vocab:
        print("[BRIDGE-BOOT] 단어 vocabulary가 비어 있습니다.")
        return None

    print(f"[BRIDGE-BOOT] doc 수={N_doc}, vocab size={len(vocab)}, "
          f"n_bootstrap={n_bootstrap}, n_shuffle={n_shuffle}")

    # 선택 횟수 카운터
    boot_counts = {cfg["name"]: {} for cfg in p_configs}
    shuffle_counts = {cfg["name"]: {} for cfg in p_configs}

    rng = np.random.default_rng(123)

    # ---------- (1) 문서 부트스트랩 ----------
    for b in range(n_bootstrap):
        # doc 인덱스를 부트스트랩 샘플링
        sample_indices = rng.integers(0, N_doc, size=N_doc)

        # word × abuse 문서 수 카운트
        counts = {}
        for idx in sample_indices:
            abuse_idx = labels_int[idx]
            abuse_name = ABUSE_ORDER[abuse_idx]
            words = doc_to_words[idx]
            if not words:
                continue
            for w in words:
                key = (w, abuse_name)
                counts[key] = counts.get(key, 0) + 1

        # DataFrame화
        table = {}
        for (w, abuse_name), c in counts.items():
            table.setdefault(w, {a: 0 for a in ABUSE_ORDER})
            table[w][abuse_name] += c

        if not table:
            continue

        df_counts_boot = (
            pd.DataFrame.from_dict(table, orient="index")
            .reindex(columns=ABUSE_ORDER)
            .fillna(0)
        )

        # 최소 문서 수 필터
        df_counts_boot["total_docs"] = df_counts_boot.sum(axis=1)
        df_counts_boot = df_counts_boot[df_counts_boot["total_docs"] >= count_min]
        df_counts_boot = df_counts_boot.drop(columns=["total_docs"])

        if df_counts_boot.empty:
            continue

        # 각 p-config별 bridge 선정
        for cfg in p_configs:
            cfg_name = cfg["name"]
            df_bridge = compute_prob_bridge_for_words(
                df_counts=df_counts_boot,
                words=df_counts_boot.index,
                logodds_df=None,  # doc-level log-odds는 사용하지 않음
                min_p1=cfg["min_p1"],
                min_p2=cfg["min_p2"],
                max_gap=cfg["max_gap"],
                logodds_min=None,
                count_min=count_min,
                z_min=None,
            )
            if df_bridge.empty:
                continue
            for w in df_bridge["word"]:
                boot_counts[cfg_name][w] = boot_counts[cfg_name].get(w, 0) + 1

        if (b + 1) % 20 == 0:
            print(f"[BRIDGE-BOOT] bootstrap {b+1}/{n_bootstrap} 완료")

    # ---------- (2) 셔플-null (라벨만 셔플, doc/단어는 고정) ----------
    for s in range(n_shuffle):
        perm_labels = rng.permutation(labels_int)

        counts = {}
        for idx in range(N_doc):
            abuse_idx = perm_labels[idx]
            abuse_name = ABUSE_ORDER[abuse_idx]
            words = doc_to_words[idx]
            if not words:
                continue
            for w in words:
                key = (w, abuse_name)
                counts[key] = counts.get(key, 0) + 1

        table = {}
        for (w, abuse_name), c in counts.items():
            table.setdefault(w, {a: 0 for a in ABUSE_ORDER})
            table[w][abuse_name] += c

        if not table:
            continue

        df_counts_shuf = (
            pd.DataFrame.from_dict(table, orient="index")
            .reindex(columns=ABUSE_ORDER)
            .fillna(0)
        )
        df_counts_shuf["total_docs"] = df_counts_shuf.sum(axis=1)
        df_counts_shuf = df_counts_shuf[df_counts_shuf["total_docs"] >= count_min]
        df_counts_shuf = df_counts_shuf.drop(columns=["total_docs"])

        if df_counts_shuf.empty:
            continue

        for cfg in p_configs:
            cfg_name = cfg["name"]
            df_bridge = compute_prob_bridge_for_words(
                df_counts=df_counts_shuf,
                words=df_counts_shuf.index,
                logodds_df=None,
                min_p1=cfg["min_p1"],
                min_p2=cfg["min_p2"],
                max_gap=cfg["max_gap"],
                logodds_min=None,
                count_min=count_min,
                z_min=None,
            )
            if df_bridge.empty:
                continue
            for w in df_bridge["word"]:
                shuffle_counts[cfg_name][w] = shuffle_counts[cfg_name].get(w, 0) + 1

        if (s + 1) % 20 == 0:
            print(f"[BRIDGE-BOOT] shuffle {s+1}/{n_shuffle} 완료")

    # ---------- (3) 요약 테이블 생성 ----------
    rows = []
    for cfg in p_configs:
        cfg_name = cfg["name"]
        words_cfg = set(boot_counts[cfg_name].keys()) | set(shuffle_counts[cfg_name].keys())
        for w in words_cfg:
            c_boot = boot_counts[cfg_name].get(w, 0)
            c_shuf = shuffle_counts[cfg_name].get(w, 0)
            rows.append({
                "config": cfg_name,
                "word": w,
                "n_bootstrap": n_bootstrap,
                "sel_count_boot": c_boot,
                "sel_prob_boot": c_boot / n_bootstrap if n_bootstrap > 0 else np.nan,
                "n_shuffle": n_shuffle,
                "sel_count_shuffle": c_shuf,
                "sel_prob_shuffle": c_shuf / n_shuffle if n_shuffle > 0 else np.nan,
                "sel_prob_diff": (c_boot / n_bootstrap if n_bootstrap > 0 else np.nan)
                                 - (c_shuf / n_shuffle if n_shuffle > 0 else np.nan),
            })

    if not rows:
        print("[BRIDGE-BOOT] bridge 선정 결과가 없어 요약 테이블을 만들지 않습니다.")
        return None

    df_stab = pd.DataFrame(rows)
    out_path = os.path.join(
        out_dir,
        f"bridge_bootstrap_shuffle_{tag}.csv",
    )
    df_stab.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"[저장] bridge 문서 부트스트랩 / 셔플-null 안정성 -> {out_path}")

    return df_stab
