#!/usr/bin/env python3
"""Aggregate-only sensitivity analyses.

The script reads the accessible AI--Hub training and validation labels, keeps
only the four maltreatment scores and case-level metadata required for the
analyses, and writes aggregate CSV/JSON outputs. It never writes case-level
records, identifiers, narratives, or utterances.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint


DOMAINS = ("방임", "정서학대", "신체학대", "성학대")
SUMMARY_LEVELS = (*DOMAINS, "해당 없음")
BASELINE_THRESHOLDS = {
    "방임": 4,
    "정서학대": 5,
    "신체학대": 5,
    "성학대": 5,
}


def normalize_summary(value: object) -> str:
    text = str(value or "").strip().replace("(", "").replace(")", "")
    if text in DOMAINS:
        return text
    if text in {"해당 없음", "해당없음", "없음", "none", "None", ""}:
        return "해당 없음"
    return "기타"


def normalize_age(value: object) -> int:
    match = re.search(r"\d+", str(value or ""))
    if match is None:
        raise ValueError(f"Could not parse age value: {value!r}")
    return int(match.group())


def normalize_sex(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("여"):
        return "여성"
    if text.startswith("남"):
        return "남성"
    return text or "미상"


def normalize_grade(value: object) -> str:
    text = str(value or "").strip()
    if "저" in text:
        return "저학년"
    if "고" in text:
        return "고학년"
    match = re.search(r"[1-6]", text)
    if match:
        return "저학년" if int(match.group()) <= 3 else "고학년"
    return text or "미상"


def normalize_crisis(value: object) -> str:
    text = str(value or "").strip()
    aliases = {
        "긴급개입필요": "응급",
        "긴급 개입 필요": "응급",
        "정상": "정상군",
        "학대 의심": "학대의심",
    }
    return aliases.get(text, text or "미상")


def extract_record(obj: dict, split: str) -> dict:
    scores: dict[str, int] = {}
    for area in obj.get("list", []):
        if area.get("문항") != "학대여부":
            continue
        for item in area.get("list", []):
            name = item.get("항목")
            if name in DOMAINS:
                scores[name] = int(item.get("점수", 0))
    if set(scores) != set(DOMAINS):
        raise ValueError("A case does not contain all four maltreatment scores")
    info = obj.get("info", {})
    return {
        "split": split,
        "summary": normalize_summary(info.get("학대의심")),
        "crisis": normalize_crisis(info.get("위기단계", "미상")),
        "age": normalize_age(info.get("나이")),
        "sex": normalize_sex(info.get("성별")),
        "grade": normalize_grade(info.get("학년")),
        **{f"score_{domain}": scores[domain] for domain in DOMAINS},
    }


def load_records(train_dir: Path, validation_zip: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(train_dir.glob("*.json")):
        rows.append(
            extract_record(json.loads(path.read_text(encoding="utf-8-sig")), "train")
        )
    with zipfile.ZipFile(validation_zip) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for name in names:
            obj = json.loads(archive.read(name).decode("utf-8-sig"))
            rows.append(extract_record(obj, "validation"))
    frame = pd.DataFrame(rows)
    if len(frame) != 3236:
        raise ValueError(f"Expected 3,236 accessible cases, found {len(frame):,}")
    if Counter(frame["split"]) != Counter({"train": 2876, "validation": 360}):
        raise ValueError("Training/validation counts do not match the manuscript")
    if set(frame["summary"]) - set(SUMMARY_LEVELS):
        raise ValueError("Unexpected one-label value found in the accessible corpus")
    frame.insert(0, "case_index", np.arange(len(frame), dtype=int))
    return frame


def add_domain_status(frame: pd.DataFrame, thresholds: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    for domain in DOMAINS:
        out[domain] = (out[f"score_{domain}"] >= thresholds[domain]).astype(int)
    out["n_domains"] = out[list(DOMAINS)].sum(axis=1)
    return out


def threshold_metrics(
    frame: pd.DataFrame,
    thresholds: dict[str, int],
    scheme: str,
    varied_threshold: int,
) -> dict:
    data = add_domain_status(frame, thresholds)
    positive = data[data["n_domains"] >= 1]
    row: dict[str, float | int | str] = {
        "scheme": scheme,
        "varied_threshold": varied_threshold,
        "threshold_neglect": thresholds["방임"],
        "threshold_emotional": thresholds["정서학대"],
        "threshold_physical": thresholds["신체학대"],
        "threshold_sexual": thresholds["성학대"],
        "positive_cases": len(positive),
        "multidomain_cases": int((positive["n_domains"] >= 2).sum()),
        "multidomain_percent": 100 * (positive["n_domains"] >= 2).mean(),
        "domain_occurrences": int(positive["n_domains"].sum()),
    }
    reductions: dict[str, float] = {}
    for domain in DOMAINS:
        domain_cases = positive[positive[domain] == 1]
        covered = int((domain_cases["summary"] == domain).sum())
        reduction = 100 * (1 - covered / len(domain_cases)) if len(domain_cases) else np.nan
        key = {
            "방임": "neglect",
            "정서학대": "emotional",
            "신체학대": "physical",
            "성학대": "sexual",
        }[domain]
        row[f"n_{key}"] = len(domain_cases)
        row[f"reduction_{key}"] = reduction
        reductions[domain] = reduction
    row["mean_reduction_emotional_neglect"] = np.mean(
        [reductions["정서학대"], reductions["방임"]]
    )
    row["mean_reduction_physical_sexual"] = np.mean(
        [reductions["신체학대"], reductions["성학대"]]
    )
    row["group_contrast_percentage_points"] = (
        row["mean_reduction_emotional_neglect"]
        - row["mean_reduction_physical_sexual"]
    )
    return row


def build_threshold_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in range(3, 8):
        common = {domain: threshold for domain in DOMAINS}
        rows.append(threshold_metrics(frame, common, "common_all_domains", threshold))
    for threshold in range(3, 8):
        neglect_only = dict(BASELINE_THRESHOLDS)
        neglect_only["방임"] = threshold
        rows.append(threshold_metrics(frame, neglect_only, "neglect_only", threshold))
    rows.append(threshold_metrics(frame, BASELINE_THRESHOLDS, "prespecified", 4))
    return pd.DataFrame(rows)


def build_single_domain_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    positive = data[data["n_domains"] >= 1]
    rows = []
    for domain in DOMAINS:
        domain_cases = positive[positive[domain] == 1]
        single = domain_cases[domain_cases["n_domains"] == 1]
        multi = domain_cases[domain_cases["n_domains"] >= 2]
        single_covered = int((single["summary"] == domain).sum())
        multi_covered = int((multi["summary"] == domain).sum())
        total_reduced = len(domain_cases) - single_covered - multi_covered
        single_reduced = len(single) - single_covered
        multi_reduced = len(multi) - multi_covered
        rows.append(
            {
                "domain": domain,
                "total_n": len(domain_cases),
                "single_n": len(single),
                "single_covered_n": single_covered,
                "single_covered_percent": 100 * single_covered / len(single),
                "single_reduction_contribution_pp": 100 * single_reduced / len(domain_cases),
                "multidomain_n": len(multi),
                "multidomain_covered_n": multi_covered,
                "multidomain_covered_percent": 100 * multi_covered / len(multi),
                "multidomain_reduction_contribution_pp": 100 * multi_reduced / len(domain_cases),
                "total_reduced_n": total_reduced,
                "total_reduction_percent": 100 * total_reduced / len(domain_cases),
            }
        )
    return pd.DataFrame(rows)


def profile_label(row: pd.Series) -> str:
    active = [domain for domain in DOMAINS if row[domain] == 1]
    return "+".join(active) if active else "기준 충족 영역 없음"


def build_combination_record_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    data["profile"] = data.apply(profile_label, axis=1)
    table = pd.crosstab(data["profile"], data["summary"])
    for level in SUMMARY_LEVELS:
        if level not in table.columns:
            table[level] = 0
    table = table[list(SUMMARY_LEVELS)]

    profile_order: list[str] = []
    profile_cardinality: dict[str, int] = {}
    for profile in itertools.product((0, 1), repeat=4):
        active = [domain for domain, present in zip(DOMAINS, profile, strict=True) if present]
        label = "+".join(active) if active else "기준 충족 영역 없음"
        profile_order.append(label)
        profile_cardinality[label] = len(active)
    table = table.reindex(profile_order, fill_value=0)
    table.insert(0, "profile_n", table.sum(axis=1))
    table.insert(0, "n_domains", table.index.to_series().map(profile_cardinality))
    table = table.sort_values(["n_domains", "profile_n"], ascending=[True, False])
    return table.reset_index()


def build_basic_relation(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    data["threshold_status"] = np.where(
        data["n_domains"].ge(1), "one_or_more_domains", "no_positive_domain"
    )
    data["one_label_status"] = np.where(
        data["summary"].isin(DOMAINS), "maltreatment_type", "none"
    )
    table = pd.crosstab(data["threshold_status"], data["one_label_status"])
    table = table.reindex(
        index=["one_or_more_domains", "no_positive_domain"],
        columns=["maltreatment_type", "none"],
        fill_value=0,
    )
    table["row_total"] = table.sum(axis=1)
    total = pd.DataFrame(
        [[int(table["maltreatment_type"].sum()), int(table["none"].sum()), len(data)]],
        index=["column_total"],
        columns=["maltreatment_type", "none", "row_total"],
    )
    return pd.concat([table, total]).reset_index(names="threshold_status")


def build_domain_2x2(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-classify each score-defined domain with the single recorded type."""
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        score_positive = data[domain].eq(1)
        recorded_same = data["summary"].eq(domain)
        both = int((score_positive & recorded_same).sum())
        score_only = int((score_positive & ~recorded_same).sum())
        recorded_only = int((~score_positive & recorded_same).sum())
        neither = int((~score_positive & ~recorded_same).sum())
        union = both + score_only + recorded_only
        rows.append(
            {
                "domain": domain,
                "score_positive_and_same_recorded_n": both,
                "score_positive_and_not_same_recorded_n": score_only,
                "score_negative_and_same_recorded_n": recorded_only,
                "score_negative_and_not_same_recorded_n": neither,
                "score_defined_n": both + score_only,
                "single_type_defined_n": both + recorded_only,
                "jaccard": both / union if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_partition_rq2(
    frame: pd.DataFrame, repetitions: int = 9999, seed: int = 20260810
) -> pd.DataFrame:
    """Domain nonrepresentation and Jaccard overlap by official partition."""
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    rows: list[dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(seed)
    split_seeds = seed_sequence.spawn(2)
    for split, split_seed in zip(("train", "validation"), split_seeds, strict=True):
        subset = data.loc[data["split"].eq(split)].reset_index(drop=True)
        rng = np.random.default_rng(split_seed)
        domain_array = subset[list(DOMAINS)].to_numpy(dtype=np.int8)
        summaries = subset["summary"].to_numpy()
        for domain_index, domain in enumerate(DOMAINS):
            positive = domain_array[:, domain_index] == 1
            same = positive & (summaries == domain)
            recorded = summaries == domain
            positive_n = int(positive.sum())
            same_n = int(same.sum())
            not_same_n = positive_n - same_n
            union_n = int((positive | recorded).sum())
            bootstrap_rates = np.empty(repetitions, dtype=float)
            for repetition in range(repetitions):
                indices = rng.integers(0, len(subset), size=len(subset))
                denominator = int(positive[indices].sum())
                numerator = denominator - int(same[indices].sum())
                bootstrap_rates[repetition] = numerator / denominator
            rows.append(
                {
                    "partition": split,
                    "domain": domain,
                    "score_positive_n": positive_n,
                    "same_type_n": same_n,
                    "not_same_type_n": not_same_n,
                    "not_same_type_percent": 100 * not_same_n / positive_n,
                    "percentile_ci_low": 100
                    * float(np.quantile(bootstrap_rates, 0.025)),
                    "percentile_ci_high": 100
                    * float(np.quantile(bootstrap_rates, 0.975)),
                    "single_type_n": int(recorded.sum()),
                    "jaccard": same_n / union_n,
                    "bootstrap_repetitions": repetitions,
                    "seed": seed,
                    "resampling_unit": "case_within_partition",
                }
            )
    return pd.DataFrame(rows)


def build_domain_direct_contrasts(
    frame: pd.DataFrame, repetitions: int = 9999, seed: int = 20260808
) -> pd.DataFrame:
    """Case-bootstrap risk differences in same-type nonrepresentation."""
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    domain_array = data[list(DOMAINS)].to_numpy(dtype=np.int8)
    summaries = data["summary"].to_numpy()
    same_array = np.column_stack(
        [
            (domain_array[:, j] == 1) & (summaries == domain)
            for j, domain in enumerate(DOMAINS)
        ]
    ).astype(np.int8)

    def rates(indices: np.ndarray) -> np.ndarray:
        den = domain_array[indices].sum(axis=0)
        num_not_same = den - same_array[indices].sum(axis=0)
        return num_not_same / den

    observed = rates(np.arange(len(data)))
    rng = np.random.default_rng(seed)
    boot = np.empty((repetitions, len(DOMAINS)), dtype=float)
    for b in range(repetitions):
        indices = rng.integers(0, len(data), size=len(data))
        boot[b] = rates(indices)

    contrast_specs = [
        (
            "emotional_and_neglect_minus_physical_and_sexual",
            np.array([0.5, 0.5, -0.5, -0.5]),
        ),
        ("neglect_minus_emotional", np.array([1.0, -1.0, 0.0, 0.0])),
        ("neglect_minus_physical", np.array([1.0, 0.0, -1.0, 0.0])),
        ("neglect_minus_sexual", np.array([1.0, 0.0, 0.0, -1.0])),
        ("emotional_minus_physical", np.array([0.0, 1.0, -1.0, 0.0])),
        ("emotional_minus_sexual", np.array([0.0, 1.0, 0.0, -1.0])),
        ("physical_minus_sexual", np.array([0.0, 0.0, 1.0, -1.0])),
    ]
    rows: list[dict[str, object]] = []
    for label, weights in contrast_specs:
        draws = 100 * (boot * weights).sum(axis=1)
        rows.append(
            {
                "contrast": label,
                "risk_difference_percentage_points": float(100 * (observed * weights).sum()),
                "percentile_ci_low": float(np.quantile(draws, 0.025)),
                "percentile_ci_high": float(np.quantile(draws, 0.975)),
                "bootstrap_repetitions": repetitions,
                "seed": seed,
                "resampling_unit": "case",
                "interval_method": "percentile",
                "failed_replicates": 0,
            }
        )
    return pd.DataFrame(rows)


def _profile_count_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS).rename(
        columns={"방임": "N", "정서학대": "E", "신체학대": "P", "성학대": "S"}
    )
    counts = data.groupby(["N", "E", "P", "S"]).size()
    rows = []
    for profile in itertools.product((0, 1), repeat=4):
        rows.append(
            {
                "N": profile[0],
                "E": profile[1],
                "P": profile[2],
                "S": profile[3],
                "count": int(counts.get(profile, 0)),
            }
        )
    return pd.DataFrame(rows)


def _fit_loglinear(counts: pd.DataFrame, pairwise: bool):
    formula = "count ~ N + E + P + S"
    if pairwise:
        formula += " + N:E + N:P + N:S + E:P + E:S + P:S"
    return smf.glm(formula, counts, family=sm.families.Poisson()).fit()


def _conditional15_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the complete 15-cell support conditional on one positive domain."""
    counts = _profile_count_frame(frame)
    return counts.loc[counts[["N", "E", "P", "S"]].sum(axis=1).ge(1)].reset_index(
        drop=True
    )


def _bootstrap_absolute_fit(
    counts: pd.DataFrame,
    fitted,
    pairwise: bool,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    total = int(counts["count"].sum())
    probabilities = np.asarray(fitted.fittedvalues, dtype=float).copy()
    probabilities /= probabilities.sum()
    simulated_deviances: list[float] = []
    failures = 0
    for _ in range(repetitions):
        simulated = counts.copy()
        simulated["count"] = rng.multinomial(total, probabilities)
        try:
            simulated_deviances.append(
                float(_fit_loglinear(simulated, pairwise).deviance)
            )
        except Exception:
            failures += 1
    p_value = (
        1 + sum(value >= fitted.deviance for value in simulated_deviances)
    ) / (1 + len(simulated_deviances))
    return p_value, failures


def build_conditional15_loglinear(
    frame: pd.DataFrame, repetitions: int = 5000, seed: int = 20260807
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Primary conditional analysis on the 15 positive-domain profiles.

    The routine reproduces the independence and all-pairwise model checks in
    the full sample and official training/validation partitions. Odds-ratio
    intervals use fixed-total multinomial resampling of each observed
    conditional profile distribution, with no pseudocount.
    """
    scope_frames = {
        "all": frame,
        "train": frame.loc[frame["split"].eq("train")],
        "validation": frame.loc[frame["split"].eq("validation")],
    }
    pair_terms = ("N:E", "N:P", "N:S", "E:P", "E:S", "P:S")
    pair_labels = {
        "N:E": "방임+정서학대",
        "N:P": "방임+신체학대",
        "N:S": "방임+성학대",
        "E:P": "정서학대+신체학대",
        "E:S": "정서학대+성학대",
        "P:S": "신체학대+성학대",
    }
    model_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(seed)
    scope_seeds = seed_sequence.spawn(len(scope_frames))

    for (scope, scope_frame), scope_seed in zip(
        scope_frames.items(), scope_seeds, strict=True
    ):
        rng = np.random.default_rng(scope_seed)
        counts = _conditional15_counts(scope_frame)
        total = int(counts["count"].sum())
        independence = _fit_loglinear(counts, False)
        pairwise_model = _fit_loglinear(counts, True)
        fitted_models = {
            "conditional_independence": (independence, False),
            "all_pairwise": (pairwise_model, True),
        }
        for model_name, (fitted, pairwise) in fitted_models.items():
            p_value, failures = _bootstrap_absolute_fit(
                counts, fitted, pairwise, repetitions, rng
            )
            model_rows.append(
                {
                    "scope": scope,
                    "positive_cases": total,
                    "model": model_name,
                    "deviance": float(fitted.deviance),
                    "df": int(fitted.df_resid),
                    "bic_case_count": float(
                        -2 * fitted.llf + len(fitted.params) * np.log(total)
                    ),
                    "parametric_bootstrap_p": p_value,
                    "requested_repetitions": repetitions,
                    "successful_replicates": repetitions - failures,
                    "failed_replicates": failures,
                    "seed": seed,
                    "simulation": "fixed_total_multinomial_from_fitted_15_profile_probabilities",
                    "p_value_formula": "(1 + count[simulated_deviance >= observed]) / (1 + successful_replicates)",
                }
            )

        observed_probabilities = counts["count"].to_numpy(dtype=float) / total
        beta_draws: dict[str, list[float]] = {term: [] for term in pair_terms}
        association_failures = 0
        for _ in range(repetitions):
            sampled = counts.copy()
            sampled["count"] = rng.multinomial(total, observed_probabilities)
            try:
                sampled_model = _fit_loglinear(sampled, True)
                for term in pair_terms:
                    beta_draws[term].append(float(sampled_model.params[term]))
            except Exception:
                association_failures += 1
        for term in pair_terms:
            draws = np.asarray(beta_draws[term], dtype=float)
            association_rows.append(
                {
                    "scope": scope,
                    "pair": pair_labels[term],
                    "conditional_or": float(np.exp(pairwise_model.params[term])),
                    "percentile_ci_low": float(
                        np.exp(np.quantile(draws, 0.025))
                    ),
                    "percentile_ci_high": float(
                        np.exp(np.quantile(draws, 0.975))
                    ),
                    "requested_repetitions": repetitions,
                    "successful_replicates": len(draws),
                    "failed_replicates": association_failures,
                    "seed": seed,
                    "resampling": "fixed_total_multinomial_from_observed_15_profile_distribution",
                    "interval_method": "percentile",
                    "pseudocount": 0,
                }
            )

        if scope == "all":
            leverage = np.asarray(pairwise_model.get_hat_matrix_diag(), dtype=float)
            fitted_values = np.asarray(pairwise_model.fittedvalues, dtype=float)
            observed_residuals = (counts["count"].to_numpy() - fitted_values) / np.sqrt(
                fitted_values * np.maximum(1 - leverage, np.finfo(float).eps)
            )
            max_draws: list[float] = []
            diagnostic_failures = 0
            pair_probabilities = fitted_values / fitted_values.sum()
            for _ in range(repetitions - 1):
                simulated = counts.copy()
                simulated["count"] = rng.multinomial(total, pair_probabilities)
                try:
                    simulated_model = _fit_loglinear(simulated, True)
                    sim_fit = np.asarray(simulated_model.fittedvalues, dtype=float)
                    sim_hat = np.asarray(
                        simulated_model.get_hat_matrix_diag(), dtype=float
                    )
                    sim_resid = (simulated["count"].to_numpy() - sim_fit) / np.sqrt(
                        sim_fit * np.maximum(1 - sim_hat, np.finfo(float).eps)
                    )
                    max_draws.append(float(np.max(np.abs(sim_resid))))
                except Exception:
                    diagnostic_failures += 1
            for row_index, row in counts.iterrows():
                adjusted_p = (
                    1
                    + sum(
                        value >= abs(observed_residuals[row_index])
                        for value in max_draws
                    )
                ) / (1 + len(max_draws))
                diagnostic_rows.append(
                    {
                        "N": int(row["N"]),
                        "E": int(row["E"]),
                        "P": int(row["P"]),
                        "S": int(row["S"]),
                        "observed": int(row["count"]),
                        "fitted": float(fitted_values[row_index]),
                        "leverage_adjusted_pearson_residual": float(
                            observed_residuals[row_index]
                        ),
                        "familywise_adjusted_p": adjusted_p,
                        "requested_repetitions": repetitions - 1,
                        "successful_replicates": len(max_draws),
                        "failed_replicates": diagnostic_failures,
                        "seed": seed,
                    }
                )

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(association_rows),
        pd.DataFrame(diagnostic_rows),
    )


def build_full16_loglinear(
    frame: pd.DataFrame, repetitions: int = 4999, seed: int = 20260809
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity analysis on all 16 profiles, including the all-negative cell."""
    counts = _profile_count_frame(frame)
    total = int(counts["count"].sum())
    rng = np.random.default_rng(seed)
    model_rows: list[dict[str, object]] = []
    fitted_models = {}
    for model_name, pairwise in (("independence", False), ("all_pairwise", True)):
        fitted = _fit_loglinear(counts, pairwise)
        fitted_models[model_name] = fitted
        fitted_values = np.asarray(fitted.fittedvalues)
        probabilities = fitted_values / fitted_values.sum()
        simulated_deviances: list[float] = []
        failures = 0
        for _ in range(repetitions):
            simulated = counts.copy()
            simulated["count"] = rng.multinomial(total, probabilities)
            try:
                simulated_deviances.append(float(_fit_loglinear(simulated, pairwise).deviance))
            except Exception:
                failures += 1
        p_value = (1 + sum(value >= fitted.deviance for value in simulated_deviances)) / (
            1 + len(simulated_deviances)
        )
        model_rows.append(
            {
                "analysis_population": "all_3236_cases_all_16_profiles",
                "model": model_name,
                "deviance": float(fitted.deviance),
                "df": int(fitted.df_resid),
                "parametric_bootstrap_p": p_value,
                "requested_repetitions": repetitions,
                "successful_replicates": len(simulated_deviances),
                "failed_replicates": failures,
                "seed": seed,
                "simulation": "fixed_total_multinomial_from_fitted_profile_probabilities",
                "p_value_formula": "(1 + count[simulated_deviance >= observed]) / (1 + successful_replicates)",
            }
        )

    pair_model = fitted_models["all_pairwise"]
    observed_probabilities = counts["count"].to_numpy() / total
    pair_terms = ("N:E", "N:P", "N:S", "E:P", "E:S", "P:S")
    bootstrap_betas: dict[str, list[float]] = {term: [] for term in pair_terms}
    association_failures = 0
    for _ in range(repetitions):
        sampled = counts.copy()
        sampled["count"] = rng.multinomial(total, observed_probabilities)
        try:
            fitted = _fit_loglinear(sampled, True)
            for term in pair_terms:
                bootstrap_betas[term].append(float(fitted.params[term]))
        except Exception:
            association_failures += 1
    labels = {
        "N:E": "방임+정서학대",
        "N:P": "방임+신체학대",
        "N:S": "방임+성학대",
        "E:P": "정서학대+신체학대",
        "E:S": "정서학대+성학대",
        "P:S": "신체학대+성학대",
    }
    association_rows = []
    for term in pair_terms:
        draws = np.asarray(bootstrap_betas[term])
        association_rows.append(
            {
                "pair": labels[term],
                "conditional_or": float(np.exp(pair_model.params[term])),
                "percentile_ci_low": float(np.exp(np.quantile(draws, 0.025))),
                "percentile_ci_high": float(np.exp(np.quantile(draws, 0.975))),
                "requested_repetitions": repetitions,
                "successful_replicates": len(draws),
                "failed_replicates": association_failures,
                "seed": seed,
                "resampling": "multinomial_case_profile_resampling_across_all_16_profiles",
                "interval_method": "percentile",
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(association_rows)


def build_sample_characteristics(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    populations = {
        "all_accessible_records": data,
        "threshold_positive_records": data[data["n_domains"].ge(1)],
    }
    rows: list[dict[str, object]] = []
    for population, subset in populations.items():
        rows.append(
            {
                "population": population,
                "characteristic": "age",
                "level": "years",
                "n": len(subset),
                "percent": np.nan,
                "mean": subset["age"].mean(),
                "sd": subset["age"].std(ddof=1),
                "min": subset["age"].min(),
                "max": subset["age"].max(),
            }
        )
        for characteristic in ("sex", "grade", "crisis", "split"):
            counts = subset[characteristic].value_counts(dropna=False, sort=False)
            for level, count in counts.items():
                rows.append(
                    {
                        "population": population,
                        "characteristic": characteristic,
                        "level": level,
                        "n": int(count),
                        "percent": 100 * count / len(subset),
                        "mean": np.nan,
                        "sd": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    }
                )
        rows.extend(
            [
                {
                    "population": population,
                    "characteristic": "threshold_positive",
                    "level": "one_or_more_domains",
                    "n": int(subset["n_domains"].ge(1).sum()),
                    "percent": 100 * subset["n_domains"].ge(1).mean(),
                    "mean": np.nan,
                    "sd": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                },
                {
                    "population": population,
                    "characteristic": "multidomain",
                    "level": "two_or_more_domains",
                    "n": int(subset["n_domains"].ge(2).sum()),
                    "percent": 100 * subset["n_domains"].ge(2).mean(),
                    "mean": np.nan,
                    "sd": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def _composition_metrics(subset: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": len(subset),
        "age_mean": subset["age"].mean(),
        "age_sd": subset["age"].std(ddof=1),
        "female_percent": 100 * subset["sex"].eq("여성").mean(),
        "lower_grade_percent": 100 * subset["grade"].eq("저학년").mean(),
        "observation_percent": 100 * subset["crisis"].eq("관찰필요").mean(),
        "counseling_percent": 100 * subset["crisis"].eq("상담필요").mean(),
        "emergency_percent": 100 * subset["crisis"].eq("응급").mean(),
        "normal_percent": 100 * subset["crisis"].eq("정상군").mean(),
        "suspected_percent": 100 * subset["crisis"].eq("학대의심").mean(),
    }


def build_sample_definition_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        domain_sample = data[data[domain].eq(1)]
        one_label_sample = data[data["summary"].eq(domain)]
        overlap = int((data[domain].eq(1) & data["summary"].eq(domain)).sum())
        domain_metrics = _composition_metrics(domain_sample)
        label_metrics = _composition_metrics(one_label_sample)
        row: dict[str, object] = {
            "domain": domain,
            "overlap_n": overlap,
            "domain_occurrence_retained_percent": 100 * overlap / len(domain_sample),
            "one_label_n_as_percent_of_domain_n": 100 * len(one_label_sample) / len(domain_sample),
        }
        for key, value in domain_metrics.items():
            row[f"domain_defined_{key}"] = value
        for key, value in label_metrics.items():
            row[f"one_label_defined_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_crisis_strata(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    rows = []
    for crisis, stratum in data.groupby("crisis", sort=True):
        positive = stratum[stratum["n_domains"] >= 1]
        row: dict[str, float | int | str] = {
            "crisis": crisis,
            "all_cases": len(stratum),
            "positive_cases": len(positive),
            "multidomain_cases": int((positive["n_domains"] >= 2).sum()),
            "multidomain_percent": 100 * (positive["n_domains"] >= 2).mean(),
        }
        for domain in DOMAINS:
            domain_cases = positive[positive[domain] == 1]
            key = {
                "방임": "neglect",
                "정서학대": "emotional",
                "신체학대": "physical",
                "성학대": "sexual",
            }[domain]
            row[f"n_{key}"] = len(domain_cases)
            reduced = int((domain_cases["summary"] != domain).sum())
            row[f"reduced_n_{key}"] = reduced
            if len(domain_cases):
                low, high = proportion_confint(reduced, len(domain_cases), method="wilson")
                row[f"reduction_{key}"] = 100 * reduced / len(domain_cases)
                row[f"reduction_ci_low_{key}"] = 100 * low
                row[f"reduction_ci_high_{key}"] = 100 * high
            else:
                row[f"reduction_{key}"] = np.nan
                row[f"reduction_ci_low_{key}"] = np.nan
                row[f"reduction_ci_high_{key}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_rq2_crisis_gee(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    eligible_stages = ("상담필요", "응급", "학대의심")
    occurrence_rows: list[dict[str, object]] = []
    for _, case in data[data["crisis"].isin(eligible_stages)].iterrows():
        for domain in DOMAINS:
            if int(case[domain]) == 1:
                occurrence_rows.append(
                    {
                        "case_index": int(case["case_index"]),
                        "domain": domain,
                        "crisis": case["crisis"],
                        "retained": int(case["summary"] == domain),
                    }
                )
    occurrences = pd.DataFrame(occurrence_rows)
    model = smf.gee(
        "retained ~ C(domain) * C(crisis)",
        groups="case_index",
        data=occurrences,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    interaction_terms = [
        index for index, name in enumerate(model.params.index) if ":" in name
    ]
    restriction = np.zeros((len(interaction_terms), len(model.params)))
    for row_index, parameter_index in enumerate(interaction_terms):
        restriction[row_index, parameter_index] = 1
    test = model.wald_test(restriction, scalar=True)
    return pd.DataFrame(
        [
            {
                "included_crisis_stages": "|".join(eligible_stages),
                "domain_occurrences": len(occurrences),
                "unique_cases": occurrences["case_index"].nunique(),
                "interaction_wald_chi2": float(test.statistic),
                "interaction_df": int(len(interaction_terms)),
                "interaction_p": float(test.pvalue),
                "working_correlation": 0.0,
                "converged": bool(model.converged),
            }
        ]
    )


def build_crisis_adjusted_associations(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    data = data[data["n_domains"] >= 1].rename(
        columns={"방임": "N", "정서학대": "E", "신체학대": "P", "성학대": "S"}
    )
    profiles = [profile for profile in itertools.product((0, 1), repeat=4) if any(profile)]
    count_rows = []
    for crisis, stratum in data.groupby("crisis"):
        counts = stratum.groupby(["N", "E", "P", "S"]).size()
        for profile in profiles:
            count_rows.append(
                {
                    "crisis": crisis,
                    "N": profile[0],
                    "E": profile[1],
                    "P": profile[2],
                    "S": profile[3],
                    "count": int(counts.get(profile, 0)),
                }
            )
    stratified = pd.DataFrame(count_rows)
    pooled = stratified.groupby(["N", "E", "P", "S"], as_index=False)["count"].sum()
    pair_terms = ("N:E", "N:P", "N:S", "E:P", "E:S", "P:S")
    pair_formula = " + ".join(pair_terms)
    pooled_model = smf.glm(
        "count ~ N + E + P + S + " + pair_formula,
        pooled,
        family=sm.families.Poisson(),
    ).fit()
    crisis_model = smf.glm(
        "count ~ C(crisis) * (N + E + P + S) + " + pair_formula,
        stratified,
        family=sm.families.Poisson(),
    ).fit()
    pair_labels = {
        "N:E": "방임+정서학대",
        "N:P": "방임+신체학대",
        "N:S": "방임+성학대",
        "E:P": "정서학대+신체학대",
        "E:S": "정서학대+성학대",
        "P:S": "신체학대+성학대",
    }
    rows = []
    for term in pair_terms:
        beta_pooled = pooled_model.params[term]
        se_pooled = pooled_model.bse[term]
        beta_crisis = crisis_model.params[term]
        se_crisis = crisis_model.bse[term]
        rows.append(
            {
                "pair": pair_labels[term],
                "or_without_crisis": np.exp(beta_pooled),
                "ci_low_without_crisis": np.exp(beta_pooled - 1.96 * se_pooled),
                "ci_high_without_crisis": np.exp(beta_pooled + 1.96 * se_pooled),
                "or_adjusted_for_crisis": np.exp(beta_crisis),
                "ci_low_adjusted_for_crisis": np.exp(beta_crisis - 1.96 * se_crisis),
                "ci_high_adjusted_for_crisis": np.exp(beta_crisis + 1.96 * se_crisis),
                "crisis_adjusted_model_deviance": crisis_model.deviance,
                "crisis_adjusted_model_df": crisis_model.df_resid,
                "crisis_adjusted_model_gof_p": chi2.sf(
                    crisis_model.deviance, crisis_model.df_resid
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_baseline(frame: pd.DataFrame) -> dict:
    data = add_domain_status(frame, BASELINE_THRESHOLDS)
    positive = data[data["n_domains"] >= 1]
    checks = {
        "all_cases": len(data),
        "positive_cases": len(positive),
        "multidomain_cases": int((positive["n_domains"] >= 2).sum()),
        "domain_occurrences": int(positive["n_domains"].sum()),
        "case_matches": int(
            sum(
                row["summary"] in {d for d in DOMAINS if row[d] == 1}
                for _, row in positive.iterrows()
            )
        ),
        "domain_counts": {domain: int(positive[domain].sum()) for domain in DOMAINS},
        "covered_counts": {
            domain: int(((positive[domain] == 1) & (positive["summary"] == domain)).sum())
            for domain in DOMAINS
        },
    }
    expected = {
        "all_cases": 3236,
        "positive_cases": 1479,
        "multidomain_cases": 708,
        "domain_occurrences": 2348,
        "case_matches": 1331,
        "domain_counts": {"방임": 505, "정서학대": 921, "신체학대": 633, "성학대": 289},
        "covered_counts": {"방임": 228, "정서학대": 339, "신체학대": 500, "성학대": 264},
    }
    if checks != expected:
        raise ValueError(f"Baseline aggregates do not reproduce the manuscript: {checks}")

    relation = build_basic_relation(frame).set_index("threshold_status")
    expected_relation = {
        ("one_or_more_domains", "maltreatment_type"): 1335,
        ("one_or_more_domains", "none"): 144,
        ("no_positive_domain", "maltreatment_type"): 15,
        ("no_positive_domain", "none"): 1742,
    }
    for (row, column), value in expected_relation.items():
        if int(relation.loc[row, column]) != value:
            raise ValueError("Basic relation table does not match frozen aggregates")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = load_records(args.train_dir, args.validation_zip)
    checks = validate_baseline(frame)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conditional_models, conditional_associations, conditional_diagnostics = (
        build_conditional15_loglinear(frame)
    )
    full16_models, full16_associations = build_full16_loglinear(frame)

    outputs = {
        "threshold_sensitivity.csv": build_threshold_sensitivity(frame),
        "single_domain_decomposition.csv": build_single_domain_decomposition(frame),
        "combination_record_matrix.csv": build_combination_record_matrix(frame),
        "basic_relation.csv": build_basic_relation(frame),
        "sample_characteristics.csv": build_sample_characteristics(frame),
        "sample_definition_comparison.csv": build_sample_definition_comparison(frame),
        "crisis_strata.csv": build_crisis_strata(frame),
        "rq2_crisis_gee.csv": build_rq2_crisis_gee(frame),
        "crisis_adjusted_associations.csv": build_crisis_adjusted_associations(frame),
        "domain_2x2.csv": build_domain_2x2(frame),
        "partition_rq2.csv": build_partition_rq2(frame),
        "domain_direct_contrasts.csv": build_domain_direct_contrasts(frame),
        "conditional15_loglinear_models.csv": conditional_models,
        "conditional15_pairwise_associations.csv": conditional_associations,
        "conditional15_cell_diagnostics.csv": conditional_diagnostics,
        "full16_loglinear_models.csv": full16_models,
        "full16_pairwise_associations.csv": full16_associations,
    }
    for name, table in outputs.items():
        table.to_csv(args.out_dir / name, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary = {
        "baseline_validation": checks,
        "source_counts": {"training": 2876, "validation": 360},
        "privacy": "Aggregate outputs only; no identifiers, narratives, or utterances written.",
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validated baseline and wrote {len(outputs) + 1} aggregate files to {args.out_dir}")


if __name__ == "__main__":
    main()
