from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_abuse_main_sub


# -----------------------------
# 1) Ground-truth(원본) 학대유형 추출
# -----------------------------
# 아동 안전 우선 위계: 성학대 > 신체학대 > 정서학대 > 방임
# 앞쪽 = 더 심각 → classify_abuse_main_sub에서 동점 시 우선 선택
DEFAULT_ABUSE_ORDER = ["성학대", "신체학대", "정서학대", "방임"]

_CANON_MAP = {
    # 공백/변형/유사표현 정규화
    "신체적학대": "신체학대" ,
    "정서적학대": "정서학대",
    "성적학대": "성학대",
    "성폭력": "성학대",
    "성폭행": "성학대",
    "유기": "방임",
}

def _to_text(x: Any) -> str:
    """학대의심 값이 str/list/dict 등으로 올 수 있어 안전하게 문자열로."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_to_text(v) for v in x)
    if isinstance(x, dict):
        # 흔한 패턴: {"val": "..."} 또는 {"text": "..."}
        for k in ("val", "text", "value"):
            if k in x:
                return _to_text(x.get(k))
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def normalize_abuse_label(label: str) -> str:
    s = re.sub(r"\s+", "", str(label))
    s = _CANON_MAP.get(s, s)
    # "신체적" 같은 꼬리 제거 후 다시 매핑
    if s.endswith("적학대"):
        s = s.replace("적학대", "학대")
    return _CANON_MAP.get(s, s)

def extract_gt_abuse_types_from_info(info: Dict[str, Any], field: str = "학대의심") -> Set[str]:
    """
    info[field] 텍스트에서:
      - '...학대' 패턴을 찾아 '신체학대' 같은 형태로 복원
      - '방임'이 있으면 포함
    """
    raw = info.get(field, "")
    text = _to_text(raw)
    text_nospace = re.sub(r"\s+", "", text)

    found: Set[str] = set()

    # 1) '신체학대', '신체적학대', '신체 학대' 등 포괄
    #    캡처 그룹은 '신체' or '신체적' 같은 앞부분
    for m in re.findall(r"([가-힣]+?)학대", text_nospace):
        cand = normalize_abuse_label(m + "학대")
        found.add(cand)

    # 2) '방임'은 보통 '학대' 접미사가 없으므로 별도 처리
    if "방임" in text_nospace:
        found.add(normalize_abuse_label("방임"))
    if "유기" in text_nospace:
        found.add(normalize_abuse_label("유기"))

    # 3) 아주 흔한 케이스: 값 자체가 이미 "신체학대"처럼 깔끔한 단일 라벨
    #    (정규식에서 이미 잡히지만, 혹시 '학대' 단어가 없는 경우 대비)
    for a in DEFAULT_ABUSE_ORDER:
        if re.sub(r"\s+", "", a) in text_nospace:
            found.add(normalize_abuse_label(a))

    return {normalize_abuse_label(x) for x in found if x}


# -----------------------------
# 2) JSON 로딩 유틸
# -----------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def iter_records(json_obj: Any) -> List[Dict[str, Any]]:
    """
    파일 구조가
    - dict 1개(rec)
    - list[rec]
    어떤 형태든 rec 리스트로 통일.
    """
    if isinstance(json_obj, dict) and ("info" in json_obj or "list" in json_obj):
        return [json_obj]
    if isinstance(json_obj, list):
        return [x for x in json_obj if isinstance(x, dict)]
    # 알 수 없는 구조면 빈 리스트
    return []


# -----------------------------
# 3) 비교/평가
# -----------------------------
def compare_one_record(
    rec: Dict[str, Any],
    abuse_order: List[str],
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
    gt_field: str = "학대의심",
) -> Dict[str, Any]:
    info = rec.get("info", {}) or {}
    doc_id = info.get("ID") or info.get("id") or info.get("Id") or None

    # GT 추출
    gt_types = extract_gt_abuse_types_from_info(info, field=gt_field)

    # Pred(네 알고리즘)
    # -> 이 함수는 네가 이미 정의해둔 것을 사용한다고 가정
    pred_main, pred_subs = classify_abuse_main_sub(  # noqa: F821
        rec,
        abuse_order=abuse_order,
        sub_threshold=sub_threshold,
        use_clinical_text=use_clinical_text,
    )

    pred_main_n = normalize_abuse_label(pred_main) if pred_main else None
    pred_subs_n = [normalize_abuse_label(x) for x in (pred_subs or [])]
    pred_set = set([pred_main_n]) if pred_main_n else set()
    pred_set |= set(pred_subs_n)

    # ── 세트 구성: Main only / Sub only / Main+Sub ──
    main_only_set = {pred_main_n} if pred_main_n else set()
    sub_only_set = set(pred_subs_n)
    full_set = main_only_set | sub_only_set          # = pred_set

    # ── Ablation 1: Main only vs GT ──
    mo_inter = gt_types & main_only_set
    mo_union = gt_types | main_only_set
    main_hit = (pred_main_n in gt_types) if (pred_main_n and gt_types) else False
    main_exact = (main_only_set == gt_types) if (main_only_set or gt_types) else False
    main_jaccard = (len(mo_inter) / len(mo_union)) if mo_union else None

    # ── Ablation 2: Sub only vs GT ──
    so_inter = gt_types & sub_only_set
    so_union = gt_types | sub_only_set
    sub_any_hit = bool(so_inter) if (sub_only_set and gt_types) else False
    sub_exact = (sub_only_set == gt_types) if (sub_only_set or gt_types) else False
    sub_jaccard = (len(so_inter) / len(so_union)) if so_union else None

    # ── Ablation 3: Main+Sub vs GT ──
    fs_inter = gt_types & full_set
    fs_union = gt_types | full_set
    full_any_hit = bool(fs_inter) if (full_set and gt_types) else False
    full_exact = (full_set == gt_types) if (full_set or gt_types) else False
    full_jaccard = (len(fs_inter) / len(fs_union)) if fs_union else None

    missing = sorted(gt_types - full_set)
    extra = sorted(full_set - gt_types)

    return {
        "doc_id": doc_id,
        "gt_types": "|".join(sorted(gt_types)) if gt_types else "",
        "pred_main": pred_main_n or "",
        "pred_subs": "|".join(sorted(sub_only_set)) if sub_only_set else "",
        "pred_set": "|".join(sorted(full_set)) if full_set else "",
        # Ablation 1: Main only
        "main_hit": int(main_hit),
        "main_exact": int(main_exact),
        "main_jaccard": main_jaccard if main_jaccard is not None else "",
        # Ablation 2: Sub only
        "sub_any_hit": int(sub_any_hit),
        "sub_exact": int(sub_exact),
        "sub_jaccard": sub_jaccard if sub_jaccard is not None else "",
        # Ablation 3: Main+Sub (full set)
        "full_any_hit": int(full_any_hit),
        "full_exact": int(full_exact),
        "full_jaccard": full_jaccard if full_jaccard is not None else "",
        # 메타
        "missing_from_pred": "|".join(missing),
        "extra_in_pred": "|".join(extra),
        "gt_has_label": int(bool(gt_types)),
        "pred_has_label": int(bool(pred_main_n or pred_subs_n)),
        "has_sub": int(bool(sub_only_set)),
    }

def evaluate_folder_or_file(
    input_path: str,
    abuse_order: Optional[List[str]] = None,
    sub_threshold: int = 4,
    use_clinical_text: bool = True,
    gt_field: str = "학대의심",
    out_csv: str = "abuse_label_compare_report.csv",
    out_mismatch_csv: str = "abuse_label_compare_mismatches.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    abuse_order = abuse_order or DEFAULT_ABUSE_ORDER
    p = Path(input_path)

    files: List[Path]
    if p.is_dir():
        files = sorted(list(p.rglob("*.json")))
    else:
        files = [p]

    rows: List[Dict[str, Any]] = []
    for fp in files:
        obj = load_json(fp)
        recs = iter_records(obj)
        for rec in recs:
            r = compare_one_record(
                rec,
                abuse_order=abuse_order,
                sub_threshold=sub_threshold,
                use_clinical_text=use_clinical_text,
                gt_field=gt_field,
            )
            r["source_file"] = str(fp)
            rows.append(r)

    df = pd.DataFrame(rows)

    # ================================================================
    # 요약: 3-way Ablation (Main abuse 존재 + GT 라벨 존재 코퍼스 기준)
    # ================================================================
    df_both = df[(df["gt_has_label"] == 1) & (df["pred_main"] != "")].copy()
    denom = len(df_both)

    n_gt_only = int(((df["gt_has_label"] == 1) & (df["pred_main"] == "")).sum())
    n_pred_only = int(((df["gt_has_label"] == 0) & (df["pred_main"] != "")).sum())

    print("=" * 62)
    print(f"  Total records                    : {len(df)}")
    print(f"  Records with GT label            : {int(df['gt_has_label'].sum())}")
    print(f"  Records with Main abuse          : {int((df['pred_main'] != '').sum())}")
    print(f"  Records with BOTH (GT + Main)    : {denom}")
    print(f"    GT only (no main abuse)        : {n_gt_only}")
    print(f"    Main abuse only (no GT)        : {n_pred_only}")
    print("=" * 62)

    # ── Ablation 비교 테이블 ──
    def _safe_mean(series: pd.Series) -> float:
        return series.mean() if len(series) > 0 else 0.0

    def _safe_jaccard_mean(series: pd.Series) -> float:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.mean() if numeric.notna().any() else 0.0

    ablations = []

    # (A) GT vs Main only
    ablations.append({
        "condition": "GT vs Main",
        "n": denom,
        "hit_rate": _safe_mean(df_both["main_hit"]),
        "exact_match": _safe_mean(df_both["main_exact"]),
        "jaccard": _safe_jaccard_mean(df_both["main_jaccard"]),
    })

    # (B) GT vs Sub only — sub가 존재하는 케이스 기준
    df_sub = df_both[df_both["has_sub"] == 1].copy()
    ablations.append({
        "condition": "GT vs Sub",
        "n": len(df_sub),
        "hit_rate": _safe_mean(df_sub["sub_any_hit"]),
        "exact_match": _safe_mean(df_sub["sub_exact"]),
        "jaccard": _safe_jaccard_mean(df_sub["sub_jaccard"]),
    })

    # (C) GT vs Main+Sub (full set)
    ablations.append({
        "condition": "GT vs Main+Sub",
        "n": denom,
        "hit_rate": _safe_mean(df_both["full_any_hit"]),
        "exact_match": _safe_mean(df_both["full_exact"]),
        "jaccard": _safe_jaccard_mean(df_both["full_jaccard"]),
    })

    df_abl = pd.DataFrame(ablations)

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│           Label Comparison Ablation Table               │")
    print("├────────────────┬───────┬──────────┬────────────┬────────┤")
    print("│ Condition      │   N   │ Hit Rate │ Exact Match│ Jaccard│")
    print("├────────────────┼───────┼──────────┼────────────┼────────┤")
    for _, row in df_abl.iterrows():
        print(f"│ {row['condition']:<14s} │ {row['n']:>5d} │ {row['hit_rate']:>8.4f} │ {row['exact_match']:>10.4f} │ {row['jaccard']:>6.4f} │")
    print("└────────────────┴───────┴──────────┴────────────┴────────┘")

    # ── 불일치 리포트 (Main vs GT 기준) ──
    mism = df_both[df_both["main_hit"] == 0].copy()

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    mism.to_csv(out_mismatch_csv, index=False, encoding="utf-8-sig")

    print(f"\n[Saved] Full report     -> {out_csv}")
    print(f"[Saved] Mismatch report -> {out_mismatch_csv}")

    # 샘플 출력(상위 10개)
    if len(mism) > 0:
        print("\n--- Top mismatch examples (up to 10) ---")
        show_cols = ["doc_id", "gt_types", "pred_main", "pred_subs",
                     "missing_from_pred", "extra_in_pred", "source_file"]
        print(mism[show_cols].head(10).to_string(index=False))

    # ── Confusion Matrix: GT(단일 라벨) vs Main ──
    df_single = df_both[~df_both["gt_types"].str.contains(r"\|", na=False)].copy()
    if len(df_single) > 0:
        df_single["gt_single"] = df_single["gt_types"]
        df_single["pred_main2"] = df_single["pred_main"].replace("", pd.NA)
        cm = pd.crosstab(df_single["gt_single"], df_single["pred_main2"], dropna=False)
        print("\n--- Confusion matrix: GT vs Main (single GT label) ---")
        print(cm)

    # ── Confusion Matrix: GT(단일 라벨) vs Main+Sub 최다 라벨 ──
    #    (Main+Sub set에서 GT와 겹치는 항목이 있으면 해당, 없으면 main 사용)
    if len(df_single) > 0:
        print("\n--- Confusion matrix: GT vs Main+Sub (single GT label) ---")
        df_single["pred_best"] = df_single.apply(
            lambda r: r["gt_single"] if r["gt_single"] in (r["pred_set"].split("|") if r["pred_set"] else []) else r["pred_main"],
            axis=1,
        )
        df_single["pred_best"] = df_single["pred_best"].replace("", pd.NA)
        cm2 = pd.crosstab(df_single["gt_single"], df_single["pred_best"], dropna=False)
        print(cm2)

    return df, mism


# -----------------------------
# 4) 실행 예시
# -----------------------------
if __name__ == "__main__":
    df, mism = evaluate_folder_or_file(
        input_path= C.DATA_JSON_DIR,
        sub_threshold=4,
        use_clinical_text=True,
        gt_field="학대의심",
        out_csv="report.csv",
        out_mismatch_csv="mismatch.csv",
    )
