from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_abuse_main_sub, classify_child_group
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean
from abuse_pipeline.data.doc_level import build_doc_level_abuse_counts
from abuse_pipeline.stats.stats import (
    add_bh_fdr,
    compute_chi_square,
    compute_log_odds,
    compute_prob_bridge_for_words,
)


def _iter_records(json_obj: Any) -> list[dict[str, Any]]:
    if isinstance(json_obj, dict) and ("info" in json_obj or "list" in json_obj):
        return [json_obj]
    if isinstance(json_obj, list):
        return [x for x in json_obj if isinstance(x, dict)]
    return []


def _canonicalize_json_files(json_files: list[str]) -> list[str]:
    """
    common.py에 canonicalize_json_files가 없는 구버전 환경도 지원한다.
    """
    fn = getattr(C, "canonicalize_json_files", None)
    if callable(fn):
        try:
            return fn(json_files)
        except Exception:
            pass

    out: list[str] = []
    seen: set[str] = set()
    for x in json_files:
        try:
            p = Path(str(x)).expanduser().resolve()
        except Exception:
            continue
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return sorted(out)


def _pair_tuple(a: str, b: str) -> tuple[str, str]:
    if C.SEVERITY_RANK.get(a, 999) <= C.SEVERITY_RANK.get(b, 999):
        return a, b
    return b, a


def _pair_key(a: str, b: str) -> str:
    x, y = _pair_tuple(a, b)
    return f"{x}↔{y}"


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else np.nan


def _collect_neg_records(
    json_files: list[str],
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        for idx, rec in enumerate(_iter_records(obj)):
            try:
                if classify_child_group(rec) != "부정":
                    continue
            except Exception:
                continue

            info = rec.get("info", {}) or {}
            doc_id = info.get("ID") or info.get("id") or info.get("Id") or f"{Path(path).stem}__{idx}"

            try:
                main, subs = classify_abuse_main_sub(
                    rec,
                    abuse_order=C.ABUSE_ORDER,
                    sub_threshold=sub_threshold,
                    use_clinical_text=use_clinical_text,
                )
            except Exception:
                continue

            if main not in C.ABUSE_ORDER:
                continue
            subs = [s for s in (subs or []) if s in C.ABUSE_ORDER and s != main]

            speech = extract_child_speech(rec)
            if not speech:
                continue
            toks = set(tokenize_korean(" ".join(speech)))
            if not toks:
                continue

            rows.append(
                {
                    "doc_id": str(doc_id),
                    "source_file": str(path),
                    "algo_main": main,
                    "algo_subs": sorted(set(subs), key=lambda x: C.SEVERITY_RANK.get(x, 999)),
                    "token_set": toks,
                }
            )

    return pd.DataFrame(rows)


def _build_bridge_words_neg(
    json_files: list[str],
    min_p1: float = 0.40,
    min_p2: float = 0.25,
    max_gap: float = 0.20,
    chi2_top_k: int = 200,
    count_min: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ABUSE_NEG 코퍼스에서 문서기반 bridge words를 생성한다.
    """
    df_counts = build_doc_level_abuse_counts(json_files=json_files, allowed_groups={"부정"})
    if df_counts is None or df_counts.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logodds_df = compute_log_odds(df_counts, C.ABUSE_ORDER)
    chi_df = compute_chi_square(df_counts, C.ABUSE_ORDER)
    chi_df = add_bh_fdr(chi_df, p_col="p_value", out_col="p_fdr_bh")

    chi_words = (
        chi_df.sort_values("chi2", ascending=False, kind="mergesort")
        .head(min(int(chi2_top_k), len(chi_df)))
        .index.tolist()
    )
    if not chi_words:
        return pd.DataFrame(), df_counts, chi_df, logodds_df

    bridge_df = compute_prob_bridge_for_words(
        df_counts=df_counts,
        words=chi_words,
        logodds_df=logodds_df,
        min_p1=float(min_p1),
        min_p2=float(min_p2),
        max_gap=float(max_gap),
        logodds_min=None,
        count_min=int(count_min),
        z_min=None,
    )
    if bridge_df is None or bridge_df.empty:
        bridge_df = pd.DataFrame(
            columns=["word", "primary_abuse", "secondary_abuse", "p1", "p2", "gap", "source"]
        )
    return bridge_df, df_counts, chi_df, logodds_df


def _build_bridge_maps(
    bridge_df: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]], pd.DataFrame]:
    """
    pair -> words, word -> pairs 매핑 생성
    """
    pair_to_words: dict[str, set[str]] = defaultdict(set)
    word_to_pairs: dict[str, set[str]] = defaultdict(set)
    inv_rows: list[dict[str, Any]] = []

    if bridge_df is None or bridge_df.empty:
        inv = pd.DataFrame(columns=["pair", "k1", "k2", "n_bridge_words"])
        return pair_to_words, word_to_pairs, inv

    for _, r in bridge_df.iterrows():
        w = str(r.get("word", "")).strip()
        k1 = str(r.get("primary_abuse", "")).strip()
        k2 = str(r.get("secondary_abuse", "")).strip()
        if not w or k1 not in C.ABUSE_ORDER or k2 not in C.ABUSE_ORDER or k1 == k2:
            continue

        p = _pair_key(k1, k2)
        pair_to_words[p].add(w)
        word_to_pairs[w].add(p)

    for p, ws in sorted(pair_to_words.items()):
        k1, k2 = p.split("↔")
        inv_rows.append({"pair": p, "k1": k1, "k2": k2, "n_bridge_words": int(len(ws))})
    inv = pd.DataFrame(inv_rows)

    return pair_to_words, word_to_pairs, inv


def _section22_cross_type_case_hit(
    rec_df: pd.DataFrame,
    pair_to_words: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    정의 A (교차유형 case-level):
      시도 = (아동, algo_main, algo_sub) 하나의 교차쌍
      적중 = 해당 pair에 속한 bridge 단어가 아동 token에 1개 이상 존재
    """
    rows: list[dict[str, Any]] = []
    if rec_df.empty:
        empty = pd.DataFrame()
        return empty, empty, pd.DataFrame(columns=["metric", "value"])

    for _, r in rec_df.iterrows():
        main = r["algo_main"]
        subs = r["algo_subs"] if isinstance(r["algo_subs"], list) else []
        toks = r["token_set"] if isinstance(r["token_set"], set) else set()
        for sub in subs:
            p = _pair_key(main, sub)
            ws = pair_to_words.get(p, set())
            found = sorted(toks & ws)
            rows.append(
                {
                    "doc_id": r["doc_id"],
                    "source_file": r["source_file"],
                    "main": main,
                    "sub": sub,
                    "pair": p,
                    "n_bridge_words_pair": int(len(ws)),
                    "bridge_supported_pair": int(len(ws) > 0),
                    "n_found_bridge_words": int(len(found)),
                    "found_bridge_words": "|".join(found),
                    "hit": int(len(found) > 0),
                }
            )

    attempts_df = pd.DataFrame(rows)
    if attempts_df.empty:
        empty = pd.DataFrame()
        return attempts_df, empty, pd.DataFrame(columns=["metric", "value"])

    pair_df = (
        attempts_df.groupby(["pair", "n_bridge_words_pair", "bridge_supported_pair"], as_index=False)
        .agg(
            n_attempts=("hit", "size"),
            n_hits=("hit", "sum"),
        )
        .sort_values(["bridge_supported_pair", "n_attempts", "pair"], ascending=[False, False, True])
    )
    pair_df["hit_rate"] = pair_df["n_hits"] / pair_df["n_attempts"]

    n_total = int(len(attempts_df))
    n_hits_total = int(attempts_df["hit"].sum())
    n_supported = int(attempts_df["bridge_supported_pair"].sum())
    n_hits_supported = int(attempts_df.loc[attempts_df["bridge_supported_pair"] == 1, "hit"].sum())
    n_unsupported = int(n_total - n_supported)
    n_hits_unsupported = int(n_hits_total - n_hits_supported)

    summary = pd.DataFrame(
        [
            {"metric": "definition", "value": "attempt=(doc,main,sub), hit=exists bridge word for pair in doc"},
            {"metric": "n_attempts_total", "value": n_total},
            {"metric": "n_hits_total", "value": n_hits_total},
            {"metric": "hit_rate_total", "value": _safe_rate(n_hits_total, n_total)},
            {"metric": "n_attempts_bridge_supported_only", "value": n_supported},
            {"metric": "n_hits_bridge_supported_only", "value": n_hits_supported},
            {"metric": "hit_rate_bridge_supported_only", "value": _safe_rate(n_hits_supported, n_supported)},
            {"metric": "n_attempts_bridge_zero_pairs", "value": n_unsupported},
            {"metric": "n_hits_bridge_zero_pairs", "value": n_hits_unsupported},
            {"metric": "hit_rate_bridge_zero_pairs", "value": _safe_rate(n_hits_unsupported, n_unsupported)},
            {
                "metric": "note",
                "value": "교량 단어가 0개인 pair는 hit가 구조적으로 불가능하므로 주요 지표에서 분리 보고",
            },
        ]
    )
    return attempts_df, pair_df, summary


def _section22_word_trial_hit(
    rec_df: pd.DataFrame,
    word_to_pairs: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    정의 B (bridge-word trial-level):
      시도 = (아동, bridge_word, bridge_pair)
      적중 = 아동의 sub set이 bridge_pair와 교집합을 가짐
    """
    rows: list[dict[str, Any]] = []
    if rec_df.empty:
        empty = pd.DataFrame()
        return empty, empty, pd.DataFrame(columns=["metric", "value"])

    has_sub_df = rec_df[rec_df["algo_subs"].map(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    if has_sub_df.empty:
        empty = pd.DataFrame()
        return empty, empty, pd.DataFrame(columns=["metric", "value"])

    # pair별 사전확률: P(sub set intersects pair) on ABUSE_NEG(has_sub)
    pair_prior_rows: list[dict[str, Any]] = []
    pair_set = sorted({p for ps in word_to_pairs.values() for p in ps})
    for p in pair_set:
        k1, k2 = p.split("↔")
        prior_hit = has_sub_df["algo_subs"].map(lambda subs: int((k1 in subs) or (k2 in subs))).mean()
        pair_prior_rows.append({"pair": p, "pair_prior_hit_prob": float(prior_hit)})
    pair_prior_df = pd.DataFrame(pair_prior_rows)
    pair_prior_map = {r["pair"]: float(r["pair_prior_hit_prob"]) for _, r in pair_prior_df.iterrows()}

    for _, r in has_sub_df.iterrows():
        toks = r["token_set"] if isinstance(r["token_set"], set) else set()
        subs = set(r["algo_subs"]) if isinstance(r["algo_subs"], list) else set()

        bridge_words_in_doc = sorted([w for w in toks if w in word_to_pairs])
        for w in bridge_words_in_doc:
            pairs = sorted(word_to_pairs.get(w, set()))
            for p in pairs:
                k1, k2 = p.split("↔")
                hit = int((k1 in subs) or (k2 in subs))
                rows.append(
                    {
                        "doc_id": r["doc_id"],
                        "source_file": r["source_file"],
                        "algo_main": r["algo_main"],
                        "algo_subs": "|".join(sorted(subs, key=lambda x: C.SEVERITY_RANK.get(x, 999))),
                        "bridge_word": w,
                        "pair": p,
                        "k1": k1,
                        "k2": k2,
                        "hit": hit,
                        "pair_prior_hit_prob": float(pair_prior_map.get(p, np.nan)),
                    }
                )

    trial_df = pd.DataFrame(rows)
    if trial_df.empty:
        empty = pd.DataFrame()
        return trial_df, empty, pd.DataFrame(columns=["metric", "value"])

    pair_df = (
        trial_df.groupby(["pair", "k1", "k2"], as_index=False)
        .agg(
            n_trials=("hit", "size"),
            n_hits=("hit", "sum"),
            pair_prior_hit_prob=("pair_prior_hit_prob", "mean"),
        )
        .sort_values("n_trials", ascending=False)
    )
    pair_df["hit_rate"] = pair_df["n_hits"] / pair_df["n_trials"]
    pair_df["expected_hits_pair_prior"] = pair_df["n_trials"] * pair_df["pair_prior_hit_prob"]

    n_trials = int(len(trial_df))
    n_hits = int(trial_df["hit"].sum())
    obs_rate = _safe_rate(n_hits, n_trials)
    exp_rate_pair_prior = float(trial_df["pair_prior_hit_prob"].mean())

    k = len(C.ABUSE_ORDER)
    naive_uniform_single_sub = 2.0 / float(k) if k > 0 else np.nan

    summary = pd.DataFrame(
        [
            {"metric": "definition", "value": "attempt=(doc,bridge_word,pair), hit=sub_set intersects pair"},
            {"metric": "n_trials", "value": n_trials},
            {"metric": "n_hits", "value": n_hits},
            {"metric": "observed_hit_rate", "value": obs_rate},
            {"metric": "expected_hit_rate_pair_prior", "value": exp_rate_pair_prior},
            {
                "metric": "lift_vs_pair_prior",
                "value": float(obs_rate / exp_rate_pair_prior) if exp_rate_pair_prior > 0 else np.nan,
            },
            {
                "metric": "naive_uniform_single_sub_baseline_2_over_K",
                "value": naive_uniform_single_sub,
            },
            {
                "metric": "note",
                "value": "주요 기준선은 pair 사전확률 기반(expected_hit_rate_pair_prior)으로 보고",
            },
        ]
    )
    return trial_df, pair_df, summary


def _load_ambiguity_table(path: str) -> tuple[pd.DataFrame, str, list[str]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("ambiguity csv is empty")

    if "true_label" in df.columns:
        true_col = "true_label"
    elif "gt_main" in df.columns:
        true_col = "gt_main"
    else:
        raise ValueError("true label column not found (expected true_label or gt_main)")

    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("prediction columns not found (expected pred_*)")

    return df, true_col, pred_cols


def _expected_same_wrong_conditional_true(
    sub_df: pd.DataFrame,
    true_col: str,
    pred_a_col: str,
    pred_b_col: str,
) -> float:
    if sub_df.empty:
        return np.nan

    n_all = len(sub_df)
    e_total = 0.0
    for t, g in sub_df.groupby(true_col):
        w_t = len(g) / n_all
        pa = g[pred_a_col].value_counts(normalize=True)
        pb = g[pred_b_col].value_counts(normalize=True)
        labels = set(pa.index.tolist()) | set(pb.index.tolist())
        e_t = 0.0
        for lbl in labels:
            if lbl == t:
                continue
            e_t += float(pa.get(lbl, 0.0) * pb.get(lbl, 0.0))
        e_total += w_t * e_t
    return float(e_total)


def _section23_chance_corrected_path3(
    ambiguity_csv: str,
    label_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    경로 3 보정:
      기존 naive baseline 1/(K-1) 대신 오답 주변분포 기반 기대 일치율을 계산.
    """
    df, true_col, pred_cols = _load_ambiguity_table(ambiguity_csv)
    labels_all = [x for x in (label_order or C.ABUSE_ORDER) if x in set(df[true_col].dropna().unique())]
    k = len(labels_all) if labels_all else len(C.ABUSE_ORDER)
    naive_1_over_k_minus_1 = 1.0 / max(1, (k - 1))

    rows: list[dict[str, Any]] = []
    for a_col, b_col in combinations(pred_cols, 2):
        a_name = a_col.replace("pred_", "")
        b_name = b_col.replace("pred_", "")

        mask = (
            df[true_col].notna()
            & df[a_col].notna()
            & df[b_col].notna()
            & (df[a_col] != df[true_col])
            & (df[b_col] != df[true_col])
        )
        d = df.loc[mask, [true_col, a_col, b_col]].copy()
        n = int(len(d))
        if n == 0:
            rows.append(
                {
                    "clf_a": a_name,
                    "clf_b": b_name,
                    "n_both_wrong": 0,
                    "observed_same_wrong": np.nan,
                    "naive_baseline_1_over_k_minus_1": naive_1_over_k_minus_1,
                    "marginal_baseline_sum_pa_pb": np.nan,
                    "marginal_baseline_sum_q2": np.nan,
                    "conditional_true_baseline": np.nan,
                    "lift_vs_conditional": np.nan,
                    "excess_vs_conditional": np.nan,
                    "kappa_wrong_conditional": np.nan,
                }
            )
            continue

        obs = float((d[a_col] == d[b_col]).mean())
        pa = d[a_col].value_counts(normalize=True)
        pb = d[b_col].value_counts(normalize=True)
        lbls = set(pa.index.tolist()) | set(pb.index.tolist())
        exp_pa_pb = float(sum(pa.get(lbl, 0.0) * pb.get(lbl, 0.0) for lbl in lbls))

        q = pd.concat([d[a_col], d[b_col]], axis=0).value_counts(normalize=True)
        exp_q2 = float(np.sum(np.square(q.values)))

        exp_cond = _expected_same_wrong_conditional_true(d, true_col, a_col, b_col)
        lift_cond = float(obs / exp_cond) if (not np.isnan(exp_cond) and exp_cond > 0) else np.nan
        excess_cond = float(obs - exp_cond) if not np.isnan(exp_cond) else np.nan
        kappa_cond = (
            float((obs - exp_cond) / (1.0 - exp_cond))
            if (not np.isnan(exp_cond) and exp_cond < 1.0)
            else np.nan
        )

        rows.append(
            {
                "clf_a": a_name,
                "clf_b": b_name,
                "n_both_wrong": n,
                "observed_same_wrong": obs,
                "naive_baseline_1_over_k_minus_1": naive_1_over_k_minus_1,
                "marginal_baseline_sum_pa_pb": exp_pa_pb,
                "marginal_baseline_sum_q2": exp_q2,
                "conditional_true_baseline": exp_cond,
                "lift_vs_conditional": lift_cond,
                "excess_vs_conditional": excess_cond,
                "kappa_wrong_conditional": kappa_cond,
            }
        )

    pair_df = pd.DataFrame(rows)
    valid = pair_df[pair_df["n_both_wrong"] > 0].copy()
    if valid.empty:
        summary = pd.DataFrame(
            [
                {"metric": "note", "value": "No pair with both-wrong samples"},
            ]
        )
        return pair_df, summary

    w = valid["n_both_wrong"].astype(float).values
    def _wavg(col: str) -> float:
        vals = valid[col].astype(float).values
        mask = ~np.isnan(vals)
        if np.sum(mask) == 0:
            return np.nan
        return float(np.average(vals[mask], weights=w[mask]))

    obs_w = _wavg("observed_same_wrong")
    cond_w = _wavg("conditional_true_baseline")
    summary = pd.DataFrame(
        [
            {"metric": "n_classifier_pairs", "value": int(len(valid))},
            {"metric": "n_both_wrong_weighted_total", "value": int(valid["n_both_wrong"].sum())},
            {"metric": "observed_same_wrong_weighted", "value": obs_w},
            {"metric": "naive_baseline_1_over_k_minus_1", "value": float(valid["naive_baseline_1_over_k_minus_1"].iloc[0])},
            {"metric": "marginal_baseline_sum_pa_pb_weighted", "value": _wavg("marginal_baseline_sum_pa_pb")},
            {"metric": "marginal_baseline_sum_q2_weighted", "value": _wavg("marginal_baseline_sum_q2")},
            {"metric": "conditional_true_baseline_weighted", "value": cond_w},
            {
                "metric": "lift_vs_conditional_weighted",
                "value": float(obs_w / cond_w) if (not np.isnan(cond_w) and cond_w > 0) else np.nan,
            },
            {
                "metric": "kappa_wrong_conditional_weighted",
                "value": float((obs_w - cond_w) / (1.0 - cond_w)) if cond_w < 1.0 else np.nan,
            },
        ]
    )
    return pair_df, summary


def run_abuse_neg_rebuttal_metrics(
    *,
    json_files: list[str],
    out_dir: str,
    bridge_csv: str | None = None,
    ambiguity_csv: str | None = None,
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
    bridge_min_p1: float = 0.40,
    bridge_min_p2: float = 0.25,
    bridge_max_gap: float = 0.20,
    bridge_chi2_top_k: int = 200,
    bridge_count_min: int = 5,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    json_files = _canonicalize_json_files(json_files)

    rec_df = _collect_neg_records(
        json_files=json_files,
        sub_threshold=int(sub_threshold),
        use_clinical_text=bool(use_clinical_text),
    )
    if rec_df.empty:
        pd.DataFrame([{"message": "No valid ABUSE_NEG records."}]).to_csv(
            os.path.join(out_dir, "analysis_skipped.csv"), encoding="utf-8-sig", index=False
        )
        return {"status": "skipped", "reason": "empty_abuse_neg_records"}

    if bridge_csv:
        bridge_df = pd.read_csv(bridge_csv)
        df_counts = pd.DataFrame()
        chi_df = pd.DataFrame()
        logodds_df = pd.DataFrame()
    else:
        bridge_df, df_counts, chi_df, logodds_df = _build_bridge_words_neg(
            json_files=json_files,
            min_p1=float(bridge_min_p1),
            min_p2=float(bridge_min_p2),
            max_gap=float(bridge_max_gap),
            chi2_top_k=int(bridge_chi2_top_k),
            count_min=int(bridge_count_min),
        )
        bridge_df.to_csv(os.path.join(out_dir, "section22_bridge_words.csv"), encoding="utf-8-sig", index=False)
        if not df_counts.empty:
            df_counts.to_csv(os.path.join(out_dir, "section22_bridge_doc_counts.csv"), encoding="utf-8-sig")
        if not chi_df.empty:
            chi_df.to_csv(os.path.join(out_dir, "section22_bridge_chi_square.csv"), encoding="utf-8-sig")
        if not logodds_df.empty:
            logodds_df.to_csv(os.path.join(out_dir, "section22_bridge_logodds.csv"), encoding="utf-8-sig", index=False)

    pair_to_words, word_to_pairs, bridge_inventory_df = _build_bridge_maps(bridge_df)
    bridge_inventory_df.to_csv(
        os.path.join(out_dir, "section22_bridge_pair_inventory.csv"), encoding="utf-8-sig", index=False
    )

    cross_attempt_df, cross_pair_df, cross_summary_df = _section22_cross_type_case_hit(rec_df, pair_to_words)
    cross_attempt_df.to_csv(
        os.path.join(out_dir, "section22_cross_type_attempts.csv"), encoding="utf-8-sig", index=False
    )
    cross_pair_df.to_csv(
        os.path.join(out_dir, "section22_cross_type_pair_summary.csv"), encoding="utf-8-sig", index=False
    )
    cross_summary_df.to_csv(
        os.path.join(out_dir, "section22_cross_type_overall_summary.csv"), encoding="utf-8-sig", index=False
    )

    word_trial_df, word_trial_pair_df, word_trial_summary_df = _section22_word_trial_hit(rec_df, word_to_pairs)
    word_trial_df.to_csv(
        os.path.join(out_dir, "section22_word_trial_attempts.csv"), encoding="utf-8-sig", index=False
    )
    word_trial_pair_df.to_csv(
        os.path.join(out_dir, "section22_word_trial_pair_summary.csv"), encoding="utf-8-sig", index=False
    )
    word_trial_summary_df.to_csv(
        os.path.join(out_dir, "section22_word_trial_overall_summary.csv"), encoding="utf-8-sig", index=False
    )

    if ambiguity_csv and os.path.exists(ambiguity_csv):
        pairwise_df, path3_summary_df = _section23_chance_corrected_path3(
            ambiguity_csv=ambiguity_csv,
            label_order=list(C.ABUSE_ORDER),
        )
        pairwise_df.to_csv(
            os.path.join(out_dir, "section23_pairwise_chance_corrected.csv"),
            encoding="utf-8-sig",
            index=False,
        )
        path3_summary_df.to_csv(
            os.path.join(out_dir, "section23_overall_summary.csv"), encoding="utf-8-sig", index=False
        )
        sec23_status = "ok"
    else:
        pd.DataFrame(
            [
                {
                    "message": "ambiguity_csv not provided or not found; section 2.3 skipped",
                    "hint": "Provide ambiguity_zone_singlelabel.csv from run_neg_gt_multilabel output.",
                }
            ]
        ).to_csv(
            os.path.join(out_dir, "section23_skipped.csv"), encoding="utf-8-sig", index=False
        )
        sec23_status = "skipped"

    # 간단 텍스트 요약
    summary_lines = [
        "ABUSE_NEG rebuttal metrics",
        f"- n_records_neg_valid: {len(rec_df)}",
        f"- n_bridge_words: {len(bridge_df)}",
    ]
    if not cross_summary_df.empty:
        try:
            v_total = float(cross_summary_df.loc[cross_summary_df["metric"] == "hit_rate_total", "value"].iloc[0])
            v_supported = float(
                cross_summary_df.loc[
                    cross_summary_df["metric"] == "hit_rate_bridge_supported_only", "value"
                ].iloc[0]
            )
            summary_lines.append(
                f"- section2.2 case-level hit: total={v_total:.4f}, bridge-supported-only={v_supported:.4f}"
            )
        except Exception:
            pass
    if not word_trial_summary_df.empty:
        try:
            obs = float(word_trial_summary_df.loc[word_trial_summary_df["metric"] == "observed_hit_rate", "value"].iloc[0])
            exp = float(
                word_trial_summary_df.loc[
                    word_trial_summary_df["metric"] == "expected_hit_rate_pair_prior", "value"
                ].iloc[0]
            )
            summary_lines.append(
                f"- section2.2 trial-level hit: observed={obs:.4f}, pair-prior-expected={exp:.4f}"
            )
        except Exception:
            pass
    summary_lines.append(f"- section2.3 chance baseline: {sec23_status}")

    with open(os.path.join(out_dir, "rebuttal_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    return {
        "status": "ok",
        "section23": sec23_status,
        "n_records_neg_valid": int(len(rec_df)),
        "n_bridge_words": int(len(bridge_df)),
        "out_dir": out_dir,
    }
