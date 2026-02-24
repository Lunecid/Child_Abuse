from __future__ import annotations

import os
import json
import matplotlib.pyplot as plt
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

import numpy as np
import pandas as pd
from abuse_pipeline.core import common as C

from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.text import (
    extract_child_speech, tokenize_korean,
    save_tokenization_examples, extract_bridge_utterances_p_z
)
from abuse_pipeline.stats.stats import (
    compute_hhi_and_cosine, compute_chi_square, add_bh_fdr, compute_log_odds,
    compute_prob_bridge_for_words,
    run_frequency_matched_baseline_for_bridge
)
from abuse_pipeline.data.doc_level import (
    build_doc_level_valence_counts, build_abuse_doc_word_table, build_doc_level_abuse_counts,
    run_doc_level_label_shuffle_permutation,
    run_bridge_prob_ablation, run_bridge_bootstrap_and_shuffle_doc_level
)
from abuse_pipeline.stats.ca import (
    run_abuse_ca_with_prob_bridges,
    bridge_ablation_and_assignments
)
from abuse_pipeline.data.embedding import (
    train_embedding_models, project_embeddings_for_ca_words,
    compare_ca_and_embedding_spaces
)
from abuse_pipeline.core.plots import (
    set_korean_font,
    plot_valence_by_question_radar, run_tfidf_multilogit_no_leak
)
from abuse_pipeline.stats.contextual_embedding_ca import run_bert_ca_validation


def run_pipeline(json_files, subset_name: str = "ALL", only_negative: bool = False) -> None:
    """
    정서군 subset(전체 / 부정-only)에 대해 동일 파이프라인 실행.

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    subset_name : str
        로그/태그용 이름 (예: "ALL", "NEG_ONLY")
    only_negative : bool
        True이면 정서군이 '부정'인 아동만 포함.
    """

    # -------------------------------------------------
    # 0) subset별 출력 디렉토리 설정 (✅ 핵심 FIX)
    # -------------------------------------------------
    subset_flag = "NEG_ONLY" if only_negative else "ALL"
    C.configure_output_dirs(
        subset_name=subset_flag,
        base_dir=C.BASE_DIR,
        version_tag="ver28",
    )

    # (선택) 폰트 세팅
    try:
        set_korean_font()
    except Exception:
        pass

    print("=" * 72)
    print(f"[RUN] subset = {subset_name} (only_negative={only_negative})")
    print(f"[RUN] OUTPUT_DIR = {C.OUTPUT_DIR}")
    print(f"[RUN] JSON 파일 개수: {len(json_files)}")
    print("=" * 72)

    # subset 공통 메타
    allowed_groups = {"부정"} if only_negative else None
    tag = subset_name

    # 워드클라우드/플롯 공통 폰트 경로
    font_path = r"C:\Windows\Fonts\malgun.ttf"  # Windows 기준
    if WordCloud is None:
        print("[WARN] wordcloud 미설치 → 워드클라우드 이미지는 생성하지 않고 표/통계만 저장합니다.")

    # -------------------------------------------------
    # (A) doc-level abuse × 단어 테이블 + 라벨 셔플 퍼뮤테이션
    # -------------------------------------------------
    doc_word_df, doc_meta = None, None
    try:
        doc_word_df, doc_meta = build_abuse_doc_word_table(
            json_files=json_files,
            allowed_groups=allowed_groups,
        )
    except Exception as e:
        print(f"[WARN] build_abuse_doc_word_table 실패 → doc-level 분석 건너뜀: {e}")

    if doc_word_df is not None and not doc_word_df.empty:
        run_doc_level_label_shuffle_permutation(
            doc_word_df=doc_word_df,
            doc_meta=doc_meta,
            n_perm=1000,
            min_doc_count=C.MIN_DOC_COUNT,
            out_dir=C.ABUSE_STATS_DIR,
            scope_words=None,
            tag=tag,
        )

        run_bridge_bootstrap_and_shuffle_doc_level(
            doc_word_df=doc_word_df,
            doc_meta=doc_meta,
            p_configs=C.BRIDGE_P_CONFIGS,
            n_bootstrap=200,
            n_shuffle=200,
            count_min=5,
            out_dir=C.BRIDGE_PROB_ABLATION_DIR,
            tag=tag,
        )
    else:
        print("[DOC-LEVEL] doc_word_df 가 비어 있어 doc-level permutation / bootstrap 을 건너뜁니다.")

    # -------------------------------------------------
    # (B) per-child 텍스트, 통계, 토큰 테이블 준비
    # -------------------------------------------------
    rows_text_valence = []   # ID, group, tfidf_text
    rows_text_abuse = []     # ID, main_abuse, tfidf_text

    rows_info = []
    rows_q = []

    group_texts_valence = {g: [] for g in C.VALENCE_ORDER}
    group_texts_abuse = {a: [] for a in C.ABUSE_ORDER}

    C.EMBEDDING_CORPUS.clear()

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"[WARN] JSON 로드 실패: {path} ({e})")
            continue

        info = rec.get("info", {})
        child_id = info.get("ID")

        group = classify_child_group(rec)
        if group not in C.VALENCE_ORDER:
            continue

        if only_negative and group != "부정":
            continue

        main_abuse, sub_abuses = classify_abuse_main_sub(rec)

        try:
            total_score = int(info.get("합계점수"))
        except (TypeError, ValueError):
            total_score = np.nan

        rows_info.append({
            "ID": child_id,
            "group": group,
            "합계점수": total_score,
            "위기단계": info.get("위기단계"),
            "main_abuse": main_abuse,
            "성별": info.get("성별"),
            "나이": info.get("나이"),
            "학년": info.get("학년"),
        })

        speech_list = extract_child_speech(rec)
        if speech_list:
            group_texts_valence[group].extend(speech_list)

            if main_abuse in C.ABUSE_ORDER:
                joined_abuse = " ".join(speech_list)
                group_texts_abuse[main_abuse].append(joined_abuse)

            for utt in speech_list:
                toks = tokenize_korean(utt)
                if toks:
                    C.EMBEDDING_CORPUS.append(toks)

            raw_text = " ".join(speech_list)
            tokens_for_tfidf = tokenize_korean(raw_text)
            tfidf_text = " ".join(tokens_for_tfidf)

            rows_text_valence.append({"ID": child_id, "group": group, "tfidf_text": tfidf_text})

            if main_abuse in C.ABUSE_ORDER:
                rows_text_abuse.append({"ID": child_id, "main_abuse": main_abuse, "tfidf_text": tfidf_text})

        for q in rec.get("list", []):
            qname = q.get("문항")
            try:
                q_total = int(q.get("문항합계"))
            except (TypeError, ValueError):
                continue
            rows_q.append({"ID": child_id, "group": group, "문항": qname, "문항합계": q_total})

    df_info = pd.DataFrame(rows_info)
    df_q = pd.DataFrame(rows_q)

    if df_info.empty:
        print(f"[ERROR][{subset_name}] df_info가 비어 있습니다. (해당 subset에 포함되는 아동이 없음)")
        return

    if "group" not in df_info.columns:
        print(f"[ERROR][{subset_name}] df_info에 'group' 컬럼이 없습니다. 현재 컬럼 목록:")
        print(df_info.columns)
        return

    df_info = df_info.drop_duplicates("ID")

    # -------------------------------------------------
    # 정서군 / 학대유형 분포 로그 + CSV
    # -------------------------------------------------
    print("[INFO] 정서군 분포:")
    print(df_info["group"].value_counts())
    print(f"[LOG] 정서군별 사례 수 (논문용):\n{df_info['group'].value_counts().to_string()}")

    valence_count_path = os.path.join(C.META_DIR, "valence_group_counts.csv")
    (
        df_info["group"]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="count")
        .to_csv(valence_count_path, encoding="utf-8-sig", index=False)
    )
    print(f"[저장] 정서군 분포 테이블 -> {valence_count_path}")

    if "main_abuse" in df_info.columns:
        print("[INFO] main_abuse 분포:")
        print(df_info["main_abuse"].value_counts(dropna=False))
        print(f"[LOG] main_abuse 분포 (논문용):\n{df_info['main_abuse'].value_counts(dropna=False).to_string()}")

        abuse_count_path = os.path.join(C.META_DIR, "main_abuse_counts.csv")
        (
            df_info["main_abuse"]
            .value_counts(dropna=False)
            .rename_axis("main_abuse")
            .reset_index(name="count")
            .to_csv(abuse_count_path, encoding="utf-8-sig", index=False)
        )
        print(f"[저장] main_abuse 분포 테이블 -> {abuse_count_path}")

    print("[INFO] 문항 레벨 row 수:", len(df_q))

    # =================================================
    # 1. 정서군 라벨 타당도 (합계점수 / 위기단계 / main_abuse)
    # =================================================
    if C.HAS_SCIPY:
        df_sc = df_info.dropna(subset=["합계점수"])
        if not df_sc.empty:
            group_stats = []
            arrays = []
            for g in C.VALENCE_ORDER:
                vals = df_sc.loc[df_sc["group"] == g, "합계점수"].astype(float)
                if len(vals) == 0:
                    continue
                arrays.append(vals.values)
                group_stats.append({"group": g, "n": len(vals), "mean": vals.mean(), "std": vals.std(ddof=1)})

            if len(arrays) >= 2:
                F, p_anova = C.f_oneway(*arrays)
                df1 = len(arrays) - 1
                df2 = len(df_sc) - len(arrays)
            else:
                F = p_anova = df1 = df2 = np.nan

            df_anova = pd.DataFrame(group_stats)
            df_anova["F"] = F
            df_anova["df1"] = df1
            df_anova["df2"] = df2
            df_anova["p_value"] = p_anova
            anova_path = os.path.join(C.VALENCE_STATS_DIR, "valence_total_score_ANOVA.csv")
            df_anova.to_csv(anova_path, encoding="utf-8-sig", index=False)
            print(f"[저장] 정서군 합계점수 ANOVA 요약 -> {anova_path}")

            pairs = [("부정", "평범"), ("부정", "긍정"), ("평범", "긍정")]
            rows_t = []
            for g1, g2 in pairs:
                v1 = df_sc.loc[df_sc["group"] == g1, "합계점수"].astype(float)
                v2 = df_sc.loc[df_sc["group"] == g2, "합계점수"].astype(float)
                if len(v1) < 2 or len(v2) < 2:
                    continue
                t_val, p_val = C.ttest_ind(v1, v2, equal_var=False)
                rows_t.append({
                    "group1": g1, "group2": g2,
                    "n1": len(v1), "n2": len(v2),
                    "mean1": v1.mean(), "mean2": v2.mean(),
                    "t": t_val, "p_value": p_val,
                })
            if rows_t:
                df_t = pd.DataFrame(rows_t)
                t_path = os.path.join(C.VALENCE_STATS_DIR, "valence_total_score_ttest_pairwise.csv")
                df_t.to_csv(t_path, encoding="utf-8-sig", index=False)
                print(f"[저장] 정서군 합계점수 t-test 요약 -> {t_path}")

        if "위기단계" in df_info.columns:
            df_tmp = df_info.dropna(subset=["위기단계"])
            if not df_tmp.empty:
                ctab = pd.crosstab(df_tmp["group"], df_tmp["위기단계"])
                if ctab.shape[0] >= 2 and ctab.shape[1] >= 2:
                    chi2_cr, p_cr, dof_cr, _ = C.chi2_contingency(ctab)
                    n_cr = ctab.to_numpy().sum()
                    r_cr, k_cr = ctab.shape
                    phi2_cr = chi2_cr / n_cr
                    cramers_cr = np.sqrt(phi2_cr / max(1, min(r_cr - 1, k_cr - 1)))
                    cr_path = os.path.join(C.VALENCE_STATS_DIR, "valence_x_crisis_chi2.csv")
                    df_cr = pd.DataFrame({
                        "N": [n_cr], "n_rows": [r_cr], "n_cols": [k_cr],
                        "chi2": [chi2_cr], "df": [dof_cr], "p_value": [p_cr],
                        "cramers_v": [cramers_cr],
                    })
                    df_cr.to_csv(cr_path, encoding="utf-8-sig", index=False)
                    print(f"[저장] 정서군 × 위기단계 chi² 요약 -> {cr_path}")

        if "main_abuse" in df_info.columns:
            df_tmp2 = df_info.dropna(subset=["main_abuse"])
            if not df_tmp2.empty:
                ctab2 = pd.crosstab(df_tmp2["group"], df_tmp2["main_abuse"])
                if ctab2.shape[0] >= 2 and ctab2.shape[1] >= 2:
                    chi2_ab2, p_ab2, dof_ab2, _ = C.chi2_contingency(ctab2)
                    n_ab2 = ctab2.to_numpy().sum()
                    r_ab2, k_ab2 = ctab2.shape
                    phi2_ab2 = chi2_ab2 / n_ab2
                    cramers_ab2 = np.sqrt(phi2_ab2 / max(1, min(r_ab2 - 1, k_ab2 - 1)))
                    v_ma_path = os.path.join(C.VALENCE_STATS_DIR, "valence_x_mainabuse_chi2.csv")
                    df_vma = pd.DataFrame({
                        "N": [n_ab2], "n_rows": [r_ab2], "n_cols": [k_ab2],
                        "chi2": [chi2_ab2], "df": [dof_ab2], "p_value": [p_ab2],
                        "cramers_v": [cramers_ab2],
                    })
                    df_vma.to_csv(v_ma_path, encoding="utf-8-sig", index=False)
                    print(f"[저장] 정서군 × main_abuse chi² 요약 -> {v_ma_path}")

    # =================================================
    # 2. 문항별 정서군 레이더차트
    # =================================================
    if not df_q.empty:
        pivot_q_valence = (
            df_q.groupby(["문항", "group"])["문항합계"]
            .mean()
            .unstack("group")
            .reindex(columns=C.VALENCE_ORDER)
            .fillna(0)
        )
        pivot_q_valence["overall_mean"] = pivot_q_valence.mean(axis=1)
        pivot_q_valence = pivot_q_valence.sort_values("overall_mean", ascending=False).drop(columns=["overall_mean"])

        radar_path = os.path.join(C.VALENCE_FIG_DIR, "radar_question_valence.png")
        plot_valence_by_question_radar(
            pivot_q_valence,
            title="문항별 정서군(부정·평범·긍정) 평균 문항합계",
            out_path=radar_path,
        )
    else:
        print("[RADAR] df_q 가 비어 있어 문항별 레이더 차트를 건너뜁니다.")

    # =================================================
    # 3. 정서군별 단어 분석 (token-level)  ✅ WordCloud 버전
    # =================================================
    df_valence_counts = pd.DataFrame()
    valence_logodds = pd.DataFrame()

    rows_wc_valence = []
    for g in C.VALENCE_ORDER:
        texts = group_texts_valence.get(g, [])
        if not texts:
            print(f"[INFO] 정서군 {g} 에 해당하는 텍스트가 없습니다.")
            continue

        joined = " ".join(t for t in texts if isinstance(t, str))
        tokens = tokenize_korean(joined)
        for w in tokens:
            rows_wc_valence.append({"group": g, "word": w})

    df_wc_valence = pd.DataFrame(rows_wc_valence)

    if df_wc_valence.empty:
        print("[WARN] 정서군 토큰이 없어 정서군 단어 분석을 건너뜁니다.")
    else:
        df_valence_counts = (
            df_wc_valence
            .groupby(["word", "group"])
            .size()
            .unstack("group")
            .fillna(0)
        )

        for g in C.VALENCE_ORDER:
            if g not in df_valence_counts.columns:
                df_valence_counts[g] = 0
        df_valence_counts = df_valence_counts[C.VALENCE_ORDER]

        df_valence_counts["total"] = df_valence_counts.sum(axis=1)
        df_valence_counts = df_valence_counts[
            df_valence_counts["total"] >= C.MIN_TOTAL_COUNT_VALENCE
        ].drop(columns=["total"])

        print("[INFO] 정서군 필터링 후 단어 수:", df_valence_counts.shape[0])

        # 전역 chi² + Cramer's V
        if C.HAS_SCIPY and not df_valence_counts.empty:
            obs_val = df_valence_counts[C.VALENCE_ORDER].values
            if (obs_val.sum(axis=0) > 0).sum() >= 2:
                chi2_val, p_val, dof_val, _ = C.chi2_contingency(obs_val)
                n_val = obs_val.sum()
                r_val, k_val = obs_val.shape
                phi2_val = chi2_val / n_val
                cramers_v_val = np.sqrt(phi2_val / max(1, min(r_val - 1, k_val - 1)))

                val_global_path = os.path.join(C.VALENCE_STATS_DIR, "valence_word_chi2_global.csv")
                pd.DataFrame({
                    "N": [n_val],
                    "n_rows": [r_val],
                    "n_cols": [k_val],
                    "chi2": [chi2_val],
                    "df": [dof_val],
                    "p_value": [p_val],
                    "cramers_v": [cramers_v_val],
                }).to_csv(val_global_path, encoding="utf-8-sig", index=False)
                print(f"[저장] 정서군 × 단어 전역 카이제곱 요약 -> {val_global_path}")

        # HHI / cosine
        if not df_valence_counts.empty:
            df_val_hhi, df_val_cos = compute_hhi_and_cosine(df_valence_counts, C.VALENCE_ORDER)

            val_hhi_path = os.path.join(C.VALENCE_STATS_DIR, "valence_word_hhi_childtokens.csv")
            df_val_hhi.to_csv(val_hhi_path, encoding="utf-8-sig", index=False)
            print(f"[저장] 정서군별 HHI -> {val_hhi_path}")

            val_cos_path = os.path.join(C.VALENCE_STATS_DIR, "valence_word_cosine_similarity_childtokens.csv")
            df_val_cos.to_csv(val_cos_path, encoding="utf-8-sig")
            print(f"[저장] 정서군 간 코사인 유사도 행렬 -> {val_cos_path}")

        # log-odds + chiÂ² (+ BH-FDR)
        valence_chi = compute_chi_square(df_valence_counts, C.VALENCE_ORDER)
        valence_chi = add_bh_fdr(valence_chi, p_col="p_value", out_col="p_fdr_bh")

        valence_logodds = compute_log_odds(df_valence_counts, C.VALENCE_ORDER)

        valence_stats = (
            valence_logodds
            .merge(valence_chi, left_on="word", right_index=True, how="left")
            .sort_values(["group", "log_odds"], ascending=[True, False])
        )

        valence_stats_path = os.path.join(C.VALENCE_STATS_DIR, "valence_word_stats_logodds_childtokens.csv")
        valence_stats.to_csv(valence_stats_path, encoding="utf-8-sig", index=False)
        print(f"[저장] 정서군 단어 통계(log-odds + chi2) -> {valence_stats_path}")

        # 정서군별 WordCloud + 상위 단어 표
        TOP_K_VALENCE_WC = 120
        TOP_N_TABLE_VALENCE = 30
        valence_top_tables = []

        for g in C.VALENCE_ORDER:
            if g not in df_valence_counts.columns:
                continue

            if df_valence_counts[g].sum() == 0:
                print(f"[VALENCE WC] 정서군 {g} 는 토큰 합계가 0이라 워드클라우드를 생성하지 않습니다.")
                continue

            sub_stats = valence_stats[valence_stats["group"] == g].copy()
            if sub_stats.empty:
                continue

            words = sub_stats["word"].values
            sub_stats["count_group"] = df_valence_counts.reindex(words)[g].fillna(0).values

            sub_wc = sub_stats[
                (sub_stats["log_odds"] > 0) & (sub_stats["count_group"] > 0)
            ].sort_values("count_group", ascending=False)

            if sub_wc.empty:
                print(f"[VALENCE WC] 정서군 {g} 에 log-odds>0 이면서 count>0 인 단어가 없습니다.")
                continue

            wc_words = sub_wc.head(TOP_K_VALENCE_WC)
            freq_dict = dict(zip(wc_words["word"], wc_words["count_group"]))

            if not freq_dict or max(freq_dict.values()) <= 0:
                print(f"[VALENCE WC] 정서군 {g} 의 freq_dict 가 비거나 max=0 입니다. 워드클라우드를 건너뜁니다.")
                continue

            cmap = "Reds" if g == "부정" else ("Greys" if g == "평범" else "Blues")

            if WordCloud is None:
                print(f"[VALENCE WC] wordcloud 미설치 → 정서군 {g} 워드클라우드 생성을 건너뜁니다.")
            else:
                wc = WordCloud(
                    font_path=font_path,
                    width=1200,
                    height=900,
                    background_color="white",
                    colormap=cmap,
                ).generate_from_frequencies(freq_dict)

                plt.figure(figsize=(9, 7))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"{g} 정서군 워드클라우드 (chi2·log-odds 기반)", fontsize=18)

                out_wc_path = os.path.join(C.VALENCE_FIG_DIR, f"wordcloud_valence_{g}_chi_logodds.png")
                plt.savefig(out_wc_path, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"[저장] 정서군 {g} 워드클라우드 -> {out_wc_path}")

            want_cols = ["group", "word", "count_group", "log_odds", "chi2", "p_value", "p_fdr_bh"]
            have_cols = [c for c in want_cols if c in wc_words.columns]
            sub_top = wc_words[have_cols].copy()
            if "count_group" in sub_top.columns and "log_odds" in sub_top.columns:
                sub_top = sub_top.sort_values(["count_group", "log_odds"], ascending=[False, False])

            out_csv_path = os.path.join(C.VALENCE_STATS_DIR, f"table_valence_{g}_top{TOP_N_TABLE_VALENCE}.csv")
            sub_top.head(TOP_N_TABLE_VALENCE).to_csv(out_csv_path, encoding="utf-8-sig", index=False)
            print(f"[저장] 정서군 {g} 상위 단어 표 -> {out_csv_path}")

            valence_top_tables.append(sub_top.head(TOP_N_TABLE_VALENCE))

        if valence_top_tables:
            table_valence_all = pd.concat(valence_top_tables, ignore_index=True)
            out_csv_all = os.path.join(C.VALENCE_STATS_DIR, f"table_valence_logodds_top{TOP_N_TABLE_VALENCE}.csv")
            table_valence_all.to_csv(out_csv_all, encoding="utf-8-sig", index=False)
            print(f"[저장] 정서군 전체 상위 단어 통합 표 -> {out_csv_all}")

        # Doc-level valence 교차표 + Spearman
        allowed_groups_doc = {"부정"} if only_negative else None
        df_valence_counts_doc = build_doc_level_valence_counts(json_files, allowed_groups=allowed_groups_doc)

        if not df_valence_counts_doc.empty:
            val_logodds_doc = compute_log_odds(df_valence_counts_doc, C.VALENCE_ORDER)
            val_chi_doc = compute_chi_square(df_valence_counts_doc, C.VALENCE_ORDER)

            val_stats_doc = (
                val_logodds_doc
                .merge(val_chi_doc, left_on="word", right_index=True, how="left")
                .sort_values(["group", "log_odds"], ascending=[True, False])
            )

            val_stats_doc_path = os.path.join(C.VALENCE_STATS_DIR, "valence_word_stats_logodds_doclevel.csv")
            val_stats_doc.to_csv(val_stats_doc_path, encoding="utf-8-sig", index=False)
            print(f"[저장] 정서군 단어 통계(doc-level) -> {val_stats_doc_path}")

            if C.HAS_SCIPY and not valence_logodds.empty:
                token_pv = valence_logodds.pivot(index="word", columns="group", values="log_odds")
                doc_pv = val_logodds_doc.pivot(index="word", columns="group", values="log_odds")

                common = token_pv.index.intersection(doc_pv.index)

                rows_sp = []
                for g in C.VALENCE_ORDER:
                    if g not in token_pv.columns or g not in doc_pv.columns:
                        continue
                    s1 = token_pv.loc[common, g]
                    s2 = doc_pv.loc[common, g]
                    mask = s1.notna() & s2.notna()
                    if mask.sum() < 3:
                        continue

                    rho, p_rho = C.spearmanr(s1[mask], s2[mask])
                    rows_sp.append({"group": g, "n_words": int(mask.sum()), "rho": rho, "p_value": p_rho})

                if rows_sp:
                    sp_path = os.path.join(C.VALENCE_STATS_DIR, "valence_token_vs_doc_logodds_spearman.csv")
                    pd.DataFrame(rows_sp).to_csv(sp_path, encoding="utf-8-sig", index=False)
                    print(f"[저장] 정서군 token vs doc log-odds Spearman 요약 -> {sp_path}")

    # =================================================
    # 4. 학대유형별 단어 분석 (token-level)
    # =================================================
    df_abuse_counts = pd.DataFrame()
    abuse_stats_logodds = pd.DataFrame()
    abuse_stats_chi = pd.DataFrame()
    bary_df = None
    row_coords_2d = None

    rows_wc_abuse = []
    for abuse_name in C.ABUSE_ORDER:
        texts = group_texts_abuse.get(abuse_name, [])
        if not texts:
            print(f"[INFO] 학대유형 {abuse_name} 에 해당하는 텍스트가 없습니다.")
            continue
        joined = " ".join(t for t in texts if isinstance(t, str))
        tokens = tokenize_korean(joined)
        for w in tokens:
            rows_wc_abuse.append({"abuse": abuse_name, "word": w})

    df_wc_abuse = pd.DataFrame(rows_wc_abuse)
    print("[INFO] 학대유형 토큰 row 수:", len(df_wc_abuse))

    if df_wc_abuse.empty:
        print("[WARN] 학대유형 토큰이 없어 학대유형 단어 분석을 건너뜁니다.")
    else:
        df_abuse_counts = df_wc_abuse.groupby(["word", "abuse"]).size().unstack("abuse").fillna(0)
        for a in C.ABUSE_ORDER:
            if a not in df_abuse_counts.columns:
                df_abuse_counts[a] = 0
        df_abuse_counts = df_abuse_counts[C.ABUSE_ORDER]

        df_abuse_counts["total"] = df_abuse_counts.sum(axis=1)
        df_abuse_counts = df_abuse_counts[df_abuse_counts["total"] >= C.MIN_TOTAL_COUNT_ABUSE].drop(columns=["total"])

        print("[INFO] 학대유형 필터링 후 단어 수:", df_abuse_counts.shape[0])

        if not df_abuse_counts.empty:
            df_abuse_hhi, df_abuse_cos = compute_hhi_and_cosine(df_abuse_counts, C.ABUSE_ORDER)
            abuse_hhi_path = os.path.join(C.ABUSE_STATS_DIR, "abuse_word_hhi_clinical_mainOnly_childtokens.csv")
            df_abuse_hhi.to_csv(abuse_hhi_path, encoding="utf-8-sig", index=False)
            print(f"[저장] 학대유형별 HHI -> {abuse_hhi_path}")

            abuse_cos_path = os.path.join(C.ABUSE_STATS_DIR, "abuse_word_cosine_similarity_clinical_mainOnly_childtokens.csv")
            df_abuse_cos.to_csv(abuse_cos_path, encoding="utf-8-sig")
            print(f"[저장] 학대유형 간 코사인 유사도 행렬 -> {abuse_cos_path}")

        if C.HAS_SCIPY and not df_abuse_counts.empty:
            obs_ab = df_abuse_counts[C.ABUSE_ORDER].values
            if (obs_ab.sum(axis=0) > 0).sum() >= 2:
                chi2_ab, p_ab, dof_ab, _ = C.chi2_contingency(obs_ab)
                n_ab = obs_ab.sum()
                r_ab, k_ab = obs_ab.shape
                phi2_ab = chi2_ab / n_ab
                cramers_v_ab = np.sqrt(phi2_ab / max(1, min(r_ab - 1, k_ab - 1)))
                abuse_global_path = os.path.join(C.ABUSE_STATS_DIR, "abuse_word_chi2_global.csv")
                pd.DataFrame({
                    "N": [n_ab], "n_rows": [r_ab], "n_cols": [k_ab],
                    "chi2": [chi2_ab], "df": [dof_ab], "p_value": [p_ab],
                    "cramers_v": [cramers_v_ab],
                }).to_csv(abuse_global_path, encoding="utf-8-sig", index=False)
                print(f"[저장] 학대유형 × 단어 전역 카이제곱 요약 -> {abuse_global_path}")

        abuse_stats_chi = compute_chi_square(df_abuse_counts, C.ABUSE_ORDER)
        abuse_stats_chi = add_bh_fdr(abuse_stats_chi, p_col="p_value", out_col="p_fdr_bh")

        abuse_stats_logodds = compute_log_odds(df_abuse_counts, C.ABUSE_ORDER)
        abuse_stats = (
            abuse_stats_logodds
            .merge(abuse_stats_chi, left_on="word", right_index=True, how="left")
            .sort_values(["group", "log_odds"], ascending=[True, False])
        )

        abuse_stats_path = os.path.join(C.ABUSE_STATS_DIR, "abuse_word_stats_logodds_mainOnly_childtokens.csv")
        abuse_stats.to_csv(abuse_stats_path, encoding="utf-8-sig", index=False)
        print(f"[저장] 학대유형 단어 통계(log-odds + chi2) -> {abuse_stats_path}")

        # (4-1) ✅ Bridge Words (pipeline ↔ integrated_label_bridge_analysis 정렬)
        # -------------------------------------------------
        # integrated_label_bridge_analysis.py(Stage 0)와 동일:
        #   - 문서 기반(doc-level) 빈도표: 한 아동(=문서) 내 동일 단어는 1회만 카운트 (set)
        #   - MIN_DOC_COUNT로 1차 필터
        #   - χ² 상위 200개 단어만 후보
        #   - χ² + BH-FDR 보정 적용 (integrated Stage 0과 동일)
        #   - p-bridge 조건 + count_min=5 적용 (z_min=None, logodds_min=None)
        # -------------------------------------------------

        allowed_groups_doc = {"부정"} if only_negative else None
        df_abuse_counts_doc = build_doc_level_abuse_counts(
            json_files,
            allowed_groups=allowed_groups_doc,
        )

        # doc-level 통계 (bridge 및 Spearman 등에 재사용)
        abuse_logodds_doc = pd.DataFrame()
        abuse_chi_doc = pd.DataFrame()

        if df_abuse_counts_doc is None or df_abuse_counts_doc.empty:
            print("[BRIDGE][DOC] doc-level df_abuse_counts_doc 가 비어 있어 bridge 추출을 건너뜁니다.")
            bridge_pz = pd.DataFrame()
        else:
            abuse_logodds_doc = compute_log_odds(df_abuse_counts_doc, C.ABUSE_ORDER)
            abuse_chi_doc = compute_chi_square(df_abuse_counts_doc, C.ABUSE_ORDER)
            abuse_chi_doc = add_bh_fdr(abuse_chi_doc, p_col="p_value", out_col="p_fdr_bh")  # ★ integrated Stage 0과 통일

            _chi2_top_k = 200
            _chi_sorted = abuse_chi_doc.sort_values("chi2", ascending=False, kind="mergesort")
            # ★ kind="mergesort" → 동점(tie) 시 원래 인덱스 순서 유지 (안정 정렬)
            _bridge_candidate_words = _chi_sorted.head(
                min(_chi2_top_k, len(_chi_sorted))
            ).index.tolist()

            bridge_pz = compute_prob_bridge_for_words(
                df_counts=df_abuse_counts_doc,
                words=_bridge_candidate_words,
                logodds_df=abuse_logodds_doc,
                min_p1=C.BRIDGE_MIN_P1,
                min_p2=C.BRIDGE_MIN_P2,
                max_gap=C.BRIDGE_MAX_GAP,
                logodds_min=None,
                count_min=5,
                z_min=None,
            )

            # integrated(Stage0)와 직접 비교 가능한 bridge 리스트 저장
            bridge_list_path = os.path.join(C.ABUSE_STATS_DIR, "stage0_bridge_words_doclevel.csv")
            bridge_pz.to_csv(bridge_list_path, encoding="utf-8-sig", index=False)
            print(f"[저장] doc-level bridge words (χ² top200, p-only) -> {bridge_list_path}")

            # 발화 추출도 doc-level bridge 기준으로 수행 (단, 발화 검색은 원문 utterance에서 수행)
            bridge_pz_utt_path = os.path.join(
                C.ABUSE_STATS_DIR,
                "bridge_prob_doclevel_top200_utterances.csv",
            )
            extract_bridge_utterances_p_z(
                json_files=json_files,
                bridge_df=bridge_pz,
                out_path=bridge_pz_utt_path,
                allowed_groups=allowed_groups_doc,
            )

        # (4-2) CA + biplot (조건 ablation)
        _ca_result = run_abuse_ca_with_prob_bridges(
            df_abuse_counts=df_abuse_counts,
            abuse_stats_logodds=abuse_stats_logodds,
            abuse_stats_chi=abuse_stats_chi,
            top_chi_for_ca=200,
            top_log_per_abuse=15,
            top_freq_per_abuse=20,
            bridge_filter_configs=C.BRIDGE_FILTER_CONFIGS,
            df_abuse_counts_doc=df_abuse_counts_doc,
            abuse_stats_logodds_doc=abuse_logodds_doc,
            abuse_stats_chi_doc=abuse_chi_doc,         # ★ 추가
        )
        if isinstance(_ca_result, tuple):
            bary_df, row_coords_2d = _ca_result
        else:
            bary_df = _ca_result
            row_coords_2d = None

        # (4-3) δ 기반 log-odds bridge ablation
        _ = bridge_ablation_and_assignments(
            abuse_stats_logodds,
            df_abuse_counts,
            prefix="baseline",
            output_dir=C.BRIDGE_DELTA_DIR,
        )

        # (4-4) p-bridge 기준 ablation
        # ✅ integrated(Stage0)와 동일한 doc-level 기반으로 수행
        #   - df_counts: doc-level (MIN_DOC_COUNT 필터 포함)
        #   - count_min: None (추가 필터 없음)
        if df_abuse_counts_doc is not None and not df_abuse_counts_doc.empty and \
           abuse_chi_doc is not None and not abuse_chi_doc.empty:
            run_bridge_prob_ablation(
                df_counts=df_abuse_counts_doc,
                chi_df=abuse_chi_doc,
                logodds_df=abuse_logodds_doc if abuse_logodds_doc is not None and not abuse_logodds_doc.empty else None,
                output_dir=C.BRIDGE_PROB_ABLATION_DIR,
                ca_top_k=200,
                count_min=5,
            )

            if (doc_word_df is not None and not doc_word_df.empty and
                    doc_meta is not None and not doc_meta.empty and
                    df_abuse_counts_doc is not None and not df_abuse_counts_doc.empty and
                    abuse_chi_doc is not None and not abuse_chi_doc.empty):

                from abuse_pipeline.stats.bridge_threshold_justification import (
                    run_bridge_threshold_justification,
                    prepare_doc_structures,
                    get_chi_top_words,
                )

                # doc_word_df + doc_meta → 내부 자료구조 변환
                _doc_to_words, _labels_int, _doc_ids = prepare_doc_structures(
                    doc_word_df=doc_word_df,
                    doc_meta=doc_meta,
                )

                # chi² top 200 단어 (교량 후보)
                _target_words = get_chi_top_words(abuse_chi_doc, top_k=200)

                # 통합 실행
                _justification = run_bridge_threshold_justification(
                    df_abuse_counts_doc=df_abuse_counts_doc,
                    doc_to_words=_doc_to_words,
                    labels_int=_labels_int,
                    target_words=_target_words,
                    p_configs=C.BRIDGE_P_CONFIGS,
                    n_perm=1000,  # 순열 반복 수 (논문용: 1000 권장)
                    n_boot=500,  # 부트스트랩 반복 수
                    count_min=5,
                    out_dir=os.path.join(C.BRIDGE_PROB_ABLATION_DIR, "threshold_justification"),
                    seed=42,
                )
                print(f"[THRESHOLD] 임계값 근거 확보 완료")
            else:
                print("[THRESHOLD] 데이터 부족으로 임계값 근거 확보를 건너뜁니다.")
        else:
            print("[ABLT][DOC] doc-level counts/chi 가 비어 있어 p-bridge ablation을 건너뜁니다.")

        # (4-5) Doc-level abuse 교차표 + Spearman
        # (4-1)에서 df_abuse_counts_doc / abuse_logodds_doc / abuse_chi_doc 를 이미 만들었으면 재사용
        if df_abuse_counts_doc is not None and not df_abuse_counts_doc.empty:
            abuse_stats_doc = (
                abuse_logodds_doc
                .merge(abuse_chi_doc, left_on="word", right_index=True, how="left")
                .sort_values(["group", "log_odds"], ascending=[True, False])
            )
            abuse_stats_doc_path = os.path.join(C.ABUSE_STATS_DIR, "abuse_word_stats_logodds_doclevel.csv")
            abuse_stats_doc.to_csv(abuse_stats_doc_path, encoding="utf-8-sig", index=False)
            print(f"[저장] 학대유형 단어 통계(doc-level) -> {abuse_stats_doc_path}")

            if C.HAS_SCIPY and not abuse_stats_logodds.empty:
                token_pv_ab = abuse_stats_logodds.pivot(index="word", columns="group", values="log_odds")
                doc_pv_ab = abuse_logodds_doc.pivot(index="word", columns="group", values="log_odds")
                common_ab = token_pv_ab.index.intersection(doc_pv_ab.index)

                rows_sp_ab = []
                for a in C.ABUSE_ORDER:
                    if a not in token_pv_ab.columns or a not in doc_pv_ab.columns:
                        continue
                    s1 = token_pv_ab.loc[common_ab, a]
                    s2 = doc_pv_ab.loc[common_ab, a]
                    mask = s1.notna() & s2.notna()
                    if mask.sum() < 3:
                        continue
                    rho_ab, p_rho_ab = C.spearmanr(s1[mask], s2[mask])
                    rows_sp_ab.append({"abuse": a, "n_words": int(mask.sum()), "rho": rho_ab, "p_value": p_rho_ab})

                if rows_sp_ab:
                    sp_ab_path = os.path.join(C.ABUSE_STATS_DIR, "abuse_token_vs_doc_logodds_spearman.csv")
                    pd.DataFrame(rows_sp_ab).to_csv(sp_ab_path, encoding="utf-8-sig", index=False)
                    print(f"[저장] 학대유형 token vs doc log-odds Spearman 요약 -> {sp_ab_path}")

        # (4-6) ✅ 학대유형별 WordCloud + 상위 단어 표 + 통합 표 (treemap 완전 대체)
        TOP_K_ABUSE_WC = 120
        TOP_N_TABLE_ABUSE = 30
        abuse_top_tables = []

        for abuse_name in C.ABUSE_ORDER:
            if abuse_name not in df_abuse_counts.columns:
                print(f"[WARN] 학대유형 {abuse_name} 컬럼이 df_abuse_counts 에 없습니다.")
                continue

            if df_abuse_counts[abuse_name].sum() == 0:
                print(f"[ABUSE WC] 학대유형 {abuse_name} 는 토큰 합계가 0이라 워드클라우드를 생성하지 않습니다.")
                continue

            sub_stats = abuse_stats[abuse_stats["group"] == abuse_name].copy()
            if sub_stats.empty:
                continue

            # count_abuse: 해당 abuse_name 빈도
            words = sub_stats["word"].values
            sub_stats["count_abuse"] = df_abuse_counts.reindex(words)[abuse_name].fillna(0).values

            # count: 전체 빈도(행 합) -> CSV 스키마 호환용
            sub_stats["count"] = df_abuse_counts.reindex(words).fillna(0).sum(axis=1).values

            sub_wc = sub_stats[
                (sub_stats["log_odds"] > 0) & (sub_stats["count_abuse"] > 0)
            ].sort_values("count_abuse", ascending=False)

            if sub_wc.empty:
                print(f"[ABUSE WC] 학대유형 {abuse_name} 에 log-odds>0 이면서 count_abuse>0 인 단어가 없습니다.")
                continue

            wc_words = sub_wc.head(TOP_K_ABUSE_WC)
            if wc_words.empty:
                continue

            freq_dict = dict(zip(wc_words["word"], wc_words["count_abuse"]))
            if not freq_dict or max(freq_dict.values()) <= 0:
                print(f"[ABUSE WC] 학대유형 {abuse_name} 의 freq_dict 가 비거나 max=0 입니다. 워드클라우드를 건너뜁니다.")
                continue

            cmap = (
                "Greens" if abuse_name == "방임"
                else "Blues" if abuse_name == "정서학대"
                else "Oranges" if abuse_name == "신체학대"
                else "Reds"
            )

            if WordCloud is None:
                print(f"[ABUSE WC] wordcloud 미설치 → 학대유형 {abuse_name} 워드클라우드 생성을 건너뜁니다.")
            else:
                wc = WordCloud(
                    font_path=font_path,
                    width=1200,
                    height=900,
                    background_color="white",
                    colormap=cmap,
                ).generate_from_frequencies(freq_dict)

                plt.figure(figsize=(9, 7))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"{abuse_name} 워드클라우드 (chi2·log-odds 기반)", fontsize=18)

                out_wc_path = os.path.join(
                    C.ABUSE_FIG_DIR,
                    f"wordcloud_abuse_{abuse_name}_clinical_chi_logodds.png",
                )
                plt.savefig(out_wc_path, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"[저장] {abuse_name} 워드클라우드 -> {out_wc_path}")

            want_cols = ["group", "word", "count", "count_abuse", "log_odds", "chi2", "p_value", "p_fdr_bh"]
            have_cols = [c for c in want_cols if c in wc_words.columns]
            sub_top = wc_words[have_cols].copy()
            if "count_abuse" in sub_top.columns and "log_odds" in sub_top.columns:
                sub_top = sub_top.sort_values(["count_abuse", "log_odds"], ascending=[False, False])

            out_csv_path = os.path.join(
                C.ABUSE_STATS_DIR,
                f"table_abuse_{abuse_name}_top{TOP_N_TABLE_ABUSE}.csv",
            )
            sub_top.head(TOP_N_TABLE_ABUSE).to_csv(out_csv_path, encoding="utf-8-sig", index=False)
            print(f"[저장] 학대유형 {abuse_name} 상위 단어 표 -> {out_csv_path}")

            abuse_top_tables.append(sub_top.head(TOP_N_TABLE_ABUSE))

        if abuse_top_tables:
            table_abuse_all = pd.concat(abuse_top_tables, ignore_index=True)
            out_csv_all = os.path.join(
                C.ABUSE_STATS_DIR,
                f"table_abuse_logodds_top{TOP_N_TABLE_ABUSE}.csv",
            )
            table_abuse_all.to_csv(out_csv_all, encoding="utf-8-sig", index=False)
            print(f"[저장] 학대유형 전체 상위 단어 통합 표 -> {out_csv_all}")

    # =================================================
    # 5. 임베딩 학습 + CA vs 임베딩 비교
    # =================================================
    if C.EMBEDDING_CORPUS and abuse_stats_chi is not None and not abuse_stats_chi.empty and C.HAS_GENSIM and C.HAS_SKLEARN:
        print(f"[EMB] 코퍼스 문장 수 (utterances) = {len(C.EMBEDDING_CORPUS)}")
        w2v_model, ft_model = train_embedding_models(
            corpus_tokens=C.EMBEDDING_CORPUS,
            output_dir=C.EMBED_MODEL_DIR,
            vector_size=100,
            window=5,
            min_count=C.BRIDGE_MIN_COUNT,
            workers=4,
        )

        df_w2v = project_embeddings_for_ca_words(
            model=w2v_model,
            abuse_stats_chi=abuse_stats_chi,
            prefix="w2v",
            out_dir=C.EMBED_PROJ_DIR,
            top_chi_for_ca=200,
        ) if w2v_model is not None else None

        df_ft = project_embeddings_for_ca_words(
            model=ft_model,
            abuse_stats_chi=abuse_stats_chi,
            prefix="fasttext",
            out_dir=C.EMBED_PROJ_DIR,
            top_chi_for_ca=200,
        ) if ft_model is not None else None

        embedding_results = {"w2v": df_w2v, "fasttext": df_ft}

        if bary_df is not None:
            compare_ca_and_embedding_spaces(
                bary_df=bary_df,
                abuse_stats_logodds=abuse_stats_logodds,
                embedding_results=embedding_results,
                out_dir=C.EMBED_PROJ_DIR,
            )
        else:
            print("[EMB] bary_df(CA 단어 좌표)가 없어 CA vs 임베딩 비교를 건너뜁니다.")
    else:
        print("[EMB] EMBEDDING_CORPUS 또는 abuse_stats_chi 가 비어 있거나 gensim/sklearn 이 없어 임베딩 비교를 건너뜁니다.")

    # -------------------------------------------------
    # BERT-CA 검증 + 이분 구조 착색 (try/except: transformers/torch 미설치 대비)
    # -------------------------------------------------
    bert_results = None
    bert_ca_out = C.BERT_CA_DIR

    try:
        bert_results = run_bert_ca_validation(
            json_files=json_files,
            df_abuse_counts=df_abuse_counts,
            abuse_stats_chi=abuse_stats_chi,
            bary_df=bary_df,           # CA barycentric word coords
            row_coords_2d=row_coords_2d,  # CA row coords (abuse types)
            out_dir=bert_ca_out,
            model_name="klue/bert-base",
            pooling="mean",
            n_clusters=200,
            batch_size=32,
            n_perm=999,
            abuse_order=C.ABUSE_ORDER,
            allowed_groups=allowed_groups,
        )
    except ImportError:
        print("[BERT-CA] transformers/torch 미설치 → BERT-CA 검증을 건너뜁니다.")
    except Exception as e:
        print(f"[BERT-CA] 실행 실패: {e}")
        import traceback; traceback.print_exc()

    # =================================================
    # 5-1. [REVISION] BERT 단어 학대유형 착색 — 이분 구조 검증
    # =================================================
    if bert_results is not None:
        _bert_word_df = bert_results.get("bert_word_coords")
        if _bert_word_df is not None and bary_df is not None:
            try:
                from abuse_pipeline.classifiers.bert_abuse_coloring import plot_bert_words_by_abuse_type

                if abuse_stats_logodds is not None and not abuse_stats_logodds.empty:
                    _word_main_abuse = (
                        abuse_stats_logodds
                        .sort_values("log_odds", ascending=False)
                        .groupby("word")["group"]
                        .first()
                        .to_dict()
                    )
                else:
                    _word_main_abuse = {}

                binary_results = plot_bert_words_by_abuse_type(
                    bert_word_df=_bert_word_df,
                    bary_df=bary_df,
                    word_main_abuse=_word_main_abuse,
                    abuse_stats_logodds=abuse_stats_logodds,
                    out_dir=bert_ca_out,
                )
                bert_results["binary_structure"] = binary_results
            except Exception as e:
                print(f"[BERT-COLORING] 이분 구조 착색 실패: {e}")
                import traceback; traceback.print_exc()
    # =================================================
    # 6. TF-IDF + Multinomial Logistic Regression
    # =================================================
    if C.HAS_SKLEARN:
        df_valence_text = pd.DataFrame(rows_text_valence)
        if not df_valence_text.empty:
            tfidf_val_dir = os.path.join(C.VALENCE_STATS_DIR, "tfidf_logit")
            os.makedirs(tfidf_val_dir, exist_ok=True)
            run_tfidf_multilogit_no_leak(
                df_text=df_valence_text,
                label_col="group",
                label_order=C.VALENCE_ORDER,
                out_dir=tfidf_val_dir,
                label_name=f"valence_{subset_name}",
            )

        df_abuse_text = pd.DataFrame(rows_text_abuse)
        if not df_abuse_text.empty:
            tfidf_abuse_dir = os.path.join(C.ABUSE_STATS_DIR, "tfidf_logit")
            os.makedirs(tfidf_abuse_dir, exist_ok=True)
            run_tfidf_multilogit_no_leak(
                df_text=df_abuse_text,
                label_col="main_abuse",
                label_order=C.ABUSE_ORDER,
                out_dir=tfidf_abuse_dir,
                label_name=f"abuse_{subset_name}",
            )
    else:
        print("[TFIDF-LOGIT] scikit-learn 미설치 → TF-IDF + 로지스틱 회귀 전체 건너뜁니다.")

    # =================================================
    # 7. 전처리 예시 + STOPWORDS 파일 저장
    # =================================================
    example_path = os.path.join(C.META_DIR, "preprocessing_examples.csv")
    save_tokenization_examples(json_files, example_path, n_examples=10)

    stop_base_path = os.path.join(C.META_DIR, "stopwords_base.txt")
    with open(stop_base_path, "w", encoding="utf-8") as f:
        for w in sorted(C.STOPWORDS_BASE):
            f.write(w + "\n")

    stop_ca_path = os.path.join(C.META_DIR, "stopwords_for_CA.txt")
    with open(stop_ca_path, "w", encoding="utf-8") as f:
        for w in sorted(C.STOPWORDS_FOR_CA):
            f.write(w + "\n")

    print(f"[DONE] 파이프라인 완료 (subset={subset_name}, output_dir={C.OUTPUT_DIR})")

    # =================================================
    # 8. Frequency-matched baseline
    # =================================================
    if df_abuse_counts is not None and not df_abuse_counts.empty and abuse_stats_logodds is not None and not abuse_stats_logodds.empty:
        out_dir = os.path.join(C.ABUSE_STATS_DIR, "freq_matched")
        os.makedirs(out_dir, exist_ok=True)

        for cfg in C.BRIDGE_P_CONFIGS:
            run_frequency_matched_baseline_for_bridge(
                df_counts=df_abuse_counts,
                logodds_df=abuse_stats_logodds,
                cfg=cfg,
                group_cols=C.ABUSE_ORDER,
                words_scope=None,
                n_bins=12,
                n_iter=1000,
                out_dir=out_dir,
                tag=subset_name,
                random_state=42,
            )
    else:
        print("[FREQ-MATCHED] df_abuse_counts 또는 abuse_stats_logodds 가 비어 있어 freq-matched baseline을 건너뜁니다.")

    from abuse_pipeline.revision.revision_extensions import run_all_revisions

    revision_out = C.REVISION_DIR

    df_text_abuse_rev = pd.DataFrame(rows_text_abuse) if rows_text_abuse else pd.DataFrame()

    run_all_revisions(
        json_files=json_files,
        df_abuse_counts=df_abuse_counts if not df_abuse_counts.empty else None,
        df_text_abuse=df_text_abuse_rev if not df_text_abuse_rev.empty else None,
        abuse_order=C.ABUSE_ORDER,
        base_out_dir=revision_out,
    )

    # =================================================
    # 9. 논문용: ABUSE_NEG + GT 기반 다중라벨(main+sub) vs 단일라벨(main) 비교
    # =================================================
    if only_negative:
        try:
            from abuse_pipeline.classifiers.neg_gt_multilabel_analysis import run_neg_gt_multilabel_study

            paper_out = C.NEG_GT_MULTILABEL_DIR
            _ = run_neg_gt_multilabel_study(
                json_files=[str(x) for x in json_files],
                out_dir=paper_out,
                gt_field="학대의심",
                n_splits=5,
                random_state=42,
                threshold=0.5,
            )
        except Exception as e:
            print(f"[NEG-GT-MULTI] 실행 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[NEG-GT-MULTI] only_negative=False 이므로 분석을 건너뜁니다.")
