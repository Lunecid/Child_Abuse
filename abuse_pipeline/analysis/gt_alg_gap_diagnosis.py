from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from abuse_pipeline.analysis.compare_abuse_labels import extract_gt_abuse_types_from_info
from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_abuse_main_sub, classify_child_group


def _iter_records(json_obj: Any) -> list[dict[str, Any]]:
    if isinstance(json_obj, dict) and ("info" in json_obj or "list" in json_obj):
        return [json_obj]
    if isinstance(json_obj, list):
        return [x for x in json_obj if isinstance(x, dict)]
    return []


def _labels_to_str(labels: set[str] | list[str], label_order: list[str]) -> str:
    if not labels:
        return ""
    return "|".join(sorted(set(labels), key=lambda x: C.SEVERITY_RANK.get(x, 999)))


def _extract_abuse_scores(rec: dict[str, Any], label_order: list[str]) -> dict[str, int]:
    scores = {a: 0 for a in label_order}
    for q in rec.get("list", []) or []:
        if q.get("문항") != "학대여부":
            continue
        for it in q.get("list", []) or []:
            name = it.get("항목")
            try:
                sc = int(it.get("점수"))
            except (TypeError, ValueError):
                sc = 0
            if not isinstance(name, str):
                continue
            for a in label_order:
                if a in name:
                    scores[a] += sc
    return scores


def _infer_main_source(
    main_label: str | None,
    scores: dict[str, int],
    clinical_keywords: set[str],
    main_threshold: int,
    sub_threshold: int,
    use_clinical_text: bool,
) -> str:
    if not main_label:
        return "none"
    if scores.get(main_label, 0) > main_threshold:
        return "score"
    if use_clinical_text and main_label in clinical_keywords:
        return "clinical"
    if scores.get(main_label, 0) >= sub_threshold:
        return "fallback_sub"
    return "unknown"


def _resolve_reason(
    *,
    gt_set: set[str],
    main_on: str | None,
    subs_on: list[str],
    main_off: str | None,
    main_source_on: str,
    scores: dict[str, int],
    clinical_keywords: set[str],
    main_threshold: int,
    sub_threshold: int,
) -> tuple[str, str]:
    if not main_on:
        return "no_algorithm_main", "score"
    if main_on in gt_set:
        return "main_match", "match"

    gt_in_sub = any(g in set(subs_on) for g in gt_set)
    main_off_match = bool(main_off and main_off in gt_set)
    clinical_changed_main = main_on != main_off

    if clinical_changed_main and main_off_match:
        return "clinical_override_from_gt", "clinical"

    if main_source_on == "clinical":
        if gt_in_sub:
            return "clinical_main_with_gt_in_sub", "clinical"
        return "clinical_main_not_in_gt", "clinical"

    if gt_in_sub:
        return "score_priority_gt_in_sub", "score"

    gt_max_score = max((scores.get(g, 0) for g in gt_set), default=0)
    top_score = max(scores.values()) if scores else 0

    if gt_max_score < sub_threshold:
        return "gt_below_sub_threshold", "score"
    if gt_max_score <= main_threshold and top_score > main_threshold:
        return "gt_not_top_over_main_threshold", "score"
    if top_score <= main_threshold and not clinical_keywords:
        return "no_score_or_clinical_signal", "score"

    return "other_mismatch", "mixed"


def run_gt_alg_gap_diagnosis(
    json_files: list[str],
    out_dir: str,
    gt_field: str = "학대의심",
    only_negative: bool = True,
    dedupe_by_doc_id: bool = True,
    sub_threshold: int = 4,
    main_threshold: int = 6,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    label_order = list(C.ABUSE_ORDER)
    rows: list[dict[str, Any]] = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        recs = _iter_records(obj)
        for ridx, rec in enumerate(recs):
            if only_negative:
                try:
                    if classify_child_group(rec) != "부정":
                        continue
                except Exception:
                    continue

            info = rec.get("info", {}) or {}
            doc_id = info.get("ID") or info.get("id") or info.get("Id")
            if not doc_id:
                doc_id = f"{Path(path).stem}__{ridx}"

            gt_set_raw = extract_gt_abuse_types_from_info(info, field=gt_field)
            gt_set = {x for x in gt_set_raw if x in label_order}
            if not gt_set:
                continue
            gt_main = sorted(gt_set, key=lambda x: C.SEVERITY_RANK.get(x, 999))[0]

            scores = _extract_abuse_scores(rec, label_order)
            clin_text = " ".join(
                str(info.get(k, "")) for k in ["임상진단", "임상가 종합소견"] if k in info
            ).strip()
            clinical_keywords = {a for a in label_order if a in clin_text}

            try:
                main_on, subs_on = classify_abuse_main_sub(
                    rec,
                    abuse_order=label_order,
                    sub_threshold=sub_threshold,
                    use_clinical_text=True,
                )
            except Exception:
                main_on, subs_on = None, []

            try:
                main_off, subs_off = classify_abuse_main_sub(
                    rec,
                    abuse_order=label_order,
                    sub_threshold=sub_threshold,
                    use_clinical_text=False,
                )
            except Exception:
                main_off, subs_off = None, []

            subs_on = subs_on or []
            subs_off = subs_off or []
            set_on = set(subs_on)
            if main_on:
                set_on.add(main_on)
            set_off = set(subs_off)
            if main_off:
                set_off.add(main_off)

            main_source_on = _infer_main_source(
                main_label=main_on,
                scores=scores,
                clinical_keywords=clinical_keywords,
                main_threshold=main_threshold,
                sub_threshold=sub_threshold,
                use_clinical_text=True,
            )
            main_source_off = _infer_main_source(
                main_label=main_off,
                scores=scores,
                clinical_keywords=clinical_keywords,
                main_threshold=main_threshold,
                sub_threshold=sub_threshold,
                use_clinical_text=False,
            )

            reason, cause_group = _resolve_reason(
                gt_set=gt_set,
                main_on=main_on,
                subs_on=subs_on,
                main_off=main_off,
                main_source_on=main_source_on,
                scores=scores,
                clinical_keywords=clinical_keywords,
                main_threshold=main_threshold,
                sub_threshold=sub_threshold,
            )

            row = {
                "doc_id": str(doc_id),
                "source_file": str(path),
                "gt_types": _labels_to_str(gt_set, label_order),
                "gt_main": gt_main,
                "alg_main": main_on or "",
                "alg_subs": _labels_to_str(subs_on, label_order),
                "alg_set": _labels_to_str(set_on, label_order),
                "main_match": int(bool(main_on and main_on in gt_set)),
                "set_match": int(set_on == gt_set),
                "main_source_on": main_source_on,
                "main_source_off": main_source_off,
                "main_changed_by_clinical": int(main_on != main_off),
                "set_changed_by_clinical": int(set_on != set_off),
                "gt_in_alg_sub": int(any(g in set(subs_on) for g in gt_set)),
                "main_off_match_gt": int(bool(main_off and main_off in gt_set)),
                "clinical_keywords": _labels_to_str(clinical_keywords, label_order),
                "clinical_has_gt": int(any(g in clinical_keywords for g in gt_set)),
                "clinical_has_alg_main": int(bool(main_on and main_on in clinical_keywords)),
                "reason": reason,
                "cause_group": cause_group,
                "top_score_label": "",
                "top_score_value": 0,
                "gt_main_score": int(scores.get(gt_main, 0)),
                "alg_main_score": int(scores.get(main_on, 0)) if main_on else 0,
            }

            if scores:
                top_label = sorted(
                    scores.keys(),
                    key=lambda x: (scores.get(x, 0), -C.SEVERITY_RANK.get(x, 999)),
                    reverse=True,
                )[0]
                row["top_score_label"] = top_label
                row["top_score_value"] = int(scores.get(top_label, 0))

            for a in label_order:
                row[f"score_{a}"] = int(scores.get(a, 0))

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        pd.DataFrame([{"message": "No valid GT records for diagnosis."}]).to_csv(
            os.path.join(out_dir, "gt_alg_gap_skipped.csv"),
            encoding="utf-8-sig",
            index=False,
        )
        return {"status": "skipped", "reason": "empty_dataset"}

    if dedupe_by_doc_id:
        df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)

    df_mismatch = df[df["main_match"] == 0].copy()
    reason_summary = (
        df_mismatch.groupby(["cause_group", "reason"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["n", "cause_group", "reason"], ascending=[False, True, True])
    )
    if not reason_summary.empty:
        reason_summary["pct_of_mismatch"] = reason_summary["n"] / len(df_mismatch)

    cause_summary = (
        df_mismatch.groupby("cause_group", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )
    if not cause_summary.empty:
        cause_summary["pct_of_mismatch"] = cause_summary["n"] / len(df_mismatch)

    source_summary = (
        df.groupby("main_source_on", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )
    if not source_summary.empty:
        source_summary["pct_of_total"] = source_summary["n"] / len(df)

    summary_rows = [
        {"item": "n_total_gt_records", "value": int(len(df))},
        {"item": "n_main_match", "value": int(df["main_match"].sum())},
        {"item": "main_match_rate", "value": float(df["main_match"].mean())},
        {"item": "n_main_mismatch", "value": int(len(df_mismatch))},
        {"item": "main_mismatch_rate", "value": float((df["main_match"] == 0).mean())},
        {"item": "n_set_match", "value": int(df["set_match"].sum())},
        {"item": "set_match_rate", "value": float(df["set_match"].mean())},
        {
            "item": "n_main_changed_by_clinical",
            "value": int(df["main_changed_by_clinical"].sum()),
        },
        {
            "item": "pct_main_changed_by_clinical",
            "value": float(df["main_changed_by_clinical"].mean()),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    df.to_csv(os.path.join(out_dir, "gt_alg_gap_case_report.csv"), encoding="utf-8-sig", index=False)
    df_mismatch.to_csv(
        os.path.join(out_dir, "gt_alg_gap_main_mismatches.csv"), encoding="utf-8-sig", index=False
    )
    reason_summary.to_csv(
        os.path.join(out_dir, "gt_alg_gap_reason_summary.csv"), encoding="utf-8-sig", index=False
    )
    cause_summary.to_csv(
        os.path.join(out_dir, "gt_alg_gap_cause_group_summary.csv"), encoding="utf-8-sig", index=False
    )
    source_summary.to_csv(
        os.path.join(out_dir, "gt_alg_gap_main_source_summary.csv"), encoding="utf-8-sig", index=False
    )
    summary_df.to_csv(
        os.path.join(out_dir, "gt_alg_gap_overview.csv"), encoding="utf-8-sig", index=False
    )

    print("==============================================================")
    print("[GT-ALG-GAP] diagnosis completed")
    print(f"  out_dir            : {out_dir}")
    print(f"  n_total_gt_records : {len(df)}")
    print(f"  n_main_mismatch    : {len(df_mismatch)} ({(len(df_mismatch) / len(df)):.1%})")
    print(f"  n_set_mismatch     : {(df['set_match'] == 0).sum()} ({(df['set_match'] == 0).mean():.1%})")
    if not cause_summary.empty:
        print("  mismatch cause groups:")
        for _, r in cause_summary.iterrows():
            print(f"    - {r['cause_group']}: {int(r['n'])} ({r['pct_of_mismatch']:.1%})")
    print("==============================================================")

    return {
        "status": "ok",
        "n_total_gt_records": int(len(df)),
        "n_main_mismatch": int(len(df_mismatch)),
        "out_dir": str(out_dir),
    }
