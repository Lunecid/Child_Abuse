"""
bert_hyperparam_search.py
=========================
KLUE-BERT 하이퍼파라미터 그리드 서치

목적
----
기존 tfidf_vs_bert_comparison.py의 run_bert_classifier()를
여러 (learning_rate × epochs) 조합으로 반복 실행하여
최적 하이퍼파라미터를 탐색한다.

논문 기여
---------
리뷰어의 "단일 하이퍼파라미터 세트로 BERT 비교는 불충분하다"는
잠재적 지적에 대응하기 위한 보완 실험이다.
최적 세팅에서도 TF-IDF 기반 방법과 CA 구조적 결과 간
불일치가 유지된다면, 이는 단순 파인튜닝 부족이 아니라
"이질적 특징 표현"에서 기인한다는 주장을 강화한다.

탐색 공간
---------
  learning_rate : [1e-5, 2e-5, 5e-5]          (3개)
  epochs        : [3, 5, 10]                    (3개)
  → 총 9개 조합 (각 조합마다 5-fold CV)

사용법
------
  # 독립 실행
  python bert_hyperparam_search.py

  # 기존 파이프라인에서 호출
  from bert_hyperparam_search import run_bert_grid_search
  best_cfg, summary_df = run_bert_grid_search(json_files, abuse_order, out_dir)
"""

from __future__ import annotations

import os
import sys
import json
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 기존 모듈 import ──
try:
    from .tfidf_vs_bert_comparision import (
        prepare_classification_data,
        run_bert_classifier,
    )
    _IMPORT_OK = True
except ImportError:
    try:
        from abuse_pipeline.tfidf_vs_bert_comparision import (
            prepare_classification_data,
            run_bert_classifier,
        )
        _IMPORT_OK = True
    except ImportError:
        _IMPORT_OK = False


# ═══════════════════════════════════════════════════════════════════
#  그리드 서치 설정
# ═══════════════════════════════════════════════════════════════════

DEFAULT_GRID = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "epochs":        [3, 5, 10],
}

# 논문 원래 설정 (비교 기준)
BASELINE_CONFIG = {
    "learning_rate": 2e-5,
    "epochs":        10,
}


# ═══════════════════════════════════════════════════════════════════
#  메인 그리드 서치 함수
# ═══════════════════════════════════════════════════════════════════

def run_bert_grid_search(
    json_files: List[str],
    abuse_order: List[str] = None,
    out_dir: str = "bert_gridsearch_output",
    only_negative: bool = True,
    n_splits: int = 5,
    bert_model_name: str = "klue/bert-base",
    bert_max_length: int = 256,
    bert_batch_size: int = 16,
    param_grid: Optional[Dict] = None,
    random_state: int = 42,
) -> Tuple[Dict, pd.DataFrame]:
    """
    KLUE-BERT 하이퍼파라미터 그리드 서치를 실행한다.

    Parameters
    ----------
    json_files : List[str]
        data/*.json 파일 경로 리스트
    abuse_order : List[str]
        학대유형 순서 (기본값: ["성학대","신체학대","정서학대","방임"])
    out_dir : str
        결과 저장 디렉토리
    only_negative : bool
        True이면 정서군이 '부정'인 아동만 포함
    n_splits : int
        Stratified K-Fold 분할 수
    bert_model_name : str
        HuggingFace 모델명
    bert_max_length : int
        최대 시퀀스 길이
    bert_batch_size : int
        배치 크기
    param_grid : dict or None
        {"learning_rate": [...], "epochs": [...]}
        None이면 DEFAULT_GRID 사용
    random_state : int
        재현성 시드

    Returns
    -------
    best_config : dict
        최적 하이퍼파라미터 {"learning_rate": ..., "epochs": ...}
    summary_df : pd.DataFrame
        전체 탐색 결과 요약 테이블
    """
    if not _IMPORT_OK:
        raise ImportError(
            "tfidf_vs_bert_comparision 모듈을 찾을 수 없습니다. "
            "abuse_pipeline 패키지 경로에서 실행하세요."
        )

    if abuse_order is None:
        abuse_order = ["성학대", "신체학대", "정서학대", "방임"]

    if param_grid is None:
        param_grid = DEFAULT_GRID

    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 데이터 준비 (1회만) ──
    print("=" * 72)
    print("  [STEP 1] 데이터 준비 (그리드 서치 전 1회)")
    print("=" * 72)
    df = prepare_classification_data(json_files, abuse_order, only_negative)

    if df.empty or df["main_abuse"].nunique() < 2:
        raise ValueError("[ERROR] 유효한 데이터가 부족합니다.")

    # ── 2. 하이퍼파라미터 조합 생성 ──
    lr_list = param_grid["learning_rate"]
    ep_list = param_grid["epochs"]
    all_combinations = list(itertools.product(lr_list, ep_list))
    n_total = len(all_combinations)

    print(f"\n  탐색 조합: {n_total}개")
    print(f"  learning_rate: {lr_list}")
    print(f"  epochs       : {ep_list}")
    print(f"  (각 조합마다 {n_splits}-fold CV)\n")

    summary_rows = []

    # ── 3. 그리드 서치 루프 ──
    for combo_idx, (lr, ep) in enumerate(all_combinations, start=1):
        is_baseline = (lr == BASELINE_CONFIG["learning_rate"] and
                       ep == BASELINE_CONFIG["epochs"])

        tag = " ← 논문 기존 설정" if is_baseline else ""
        print("\n" + "═" * 72)
        print(f"  [{combo_idx}/{n_total}] lr={lr:.0e}, epochs={ep}{tag}")
        print("═" * 72)

        try:
            result = run_bert_classifier(
                df=df,
                label_col="main_abuse",
                label_order=abuse_order,
                model_name=bert_model_name,
                n_splits=n_splits,
                max_length=bert_max_length,
                batch_size=bert_batch_size,
                epochs=ep,
                learning_rate=lr,
                random_state=random_state,
            )
        except Exception as e:
            print(f"  [ERROR] lr={lr}, epochs={ep} 실패: {e}")
            summary_rows.append({
                "learning_rate": lr,
                "epochs": ep,
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "cohen_kappa": None,
                "is_baseline": is_baseline,
                "status": f"ERROR: {e}",
            })
            continue

        if result is None:
            continue

        overall = result["overall_metrics"]
        clf_name = list(result["all_true"].keys())[0]
        row = overall.iloc[0]

        # 라벨별 F1도 저장 (논문 표 작성용)
        per_label = result["per_label_metrics"]
        label_f1 = {}
        for _, r in per_label.iterrows():
            label_f1[f"f1_{r['label']}"] = round(r["f1_score"], 4)

        summary_row = {
            "learning_rate": lr,
            "epochs": ep,
            "accuracy": round(row["accuracy"], 4),
            "macro_f1": round(row["macro_f1"], 4),
            "weighted_f1": round(row["weighted_f1"], 4),
            "cohen_kappa": round(row["cohen_kappa"], 4),
            "is_baseline": is_baseline,
            "status": "OK",
        }
        summary_row.update(label_f1)
        summary_rows.append(summary_row)

        print(f"\n  → 결과: Accuracy={row['accuracy']:.4f}, "
              f"Macro F1={row['macro_f1']:.4f}, κ={row['cohen_kappa']:.4f}")

        # 중간 저장 (실험 도중 중단 대비)
        _save_intermediate(summary_rows, out_dir, combo_idx, n_total)

    # ── 4. 최종 요약 ──
    summary_df = pd.DataFrame(summary_rows)
    _finalize_results(summary_df, out_dir, abuse_order)

    # 최적 설정 추출
    valid = summary_df[summary_df["status"] == "OK"]
    if valid.empty:
        print("[WARN] 성공한 실험이 없습니다.")
        return {}, summary_df

    best_idx = valid["macro_f1"].idxmax()
    best_row = valid.loc[best_idx]
    best_config = {
        "learning_rate": best_row["learning_rate"],
        "epochs": int(best_row["epochs"]),
        "macro_f1": best_row["macro_f1"],
        "accuracy": best_row["accuracy"],
        "cohen_kappa": best_row["cohen_kappa"],
    }

    print("\n" + "═" * 72)
    print("  ✅ 그리드 서치 완료")
    print("═" * 72)
    print(f"  최적 설정 : lr={best_config['learning_rate']:.0e}, "
          f"epochs={best_config['epochs']}")
    print(f"  Macro F1  : {best_config['macro_f1']:.4f}")
    print(f"  Accuracy  : {best_config['accuracy']:.4f}")
    print(f"  Cohen's κ : {best_config['cohen_kappa']:.4f}")

    baseline_rows = summary_df[summary_df["is_baseline"] == True]
    if not baseline_rows.empty:
        b = baseline_rows.iloc[0]
        print(f"\n  논문 기존 설정 (lr=2e-5, ep=10):")
        print(f"  Macro F1  : {b['macro_f1']:.4f}  "
              f"(최적 대비 {best_config['macro_f1'] - b['macro_f1']:+.4f})")
        print(f"  Cohen's κ : {b['cohen_kappa']:.4f}")

    print(f"\n  결과 저장: {out_dir}/")

    return best_config, summary_df


# ═══════════════════════════════════════════════════════════════════
#  보조 함수
# ═══════════════════════════════════════════════════════════════════

def _save_intermediate(
    rows: List[Dict],
    out_dir: str,
    current: int,
    total: int,
) -> None:
    """실험 도중 중단 대비 중간 결과 저장."""
    path = os.path.join(out_dir, "gridsearch_intermediate.csv")
    pd.DataFrame(rows).to_csv(path, encoding="utf-8-sig", index=False)
    print(f"  [중간저장] {current}/{total} → {path}")


def _finalize_results(
    summary_df: pd.DataFrame,
    out_dir: str,
    abuse_order: List[str],
) -> None:
    """최종 결과 저장 및 출력."""

    # CSV 저장
    path = os.path.join(out_dir, "gridsearch_summary.csv")
    summary_df.to_csv(path, encoding="utf-8-sig", index=False)
    print(f"\n  [저장] 그리드 서치 요약 → {path}")

    # 논문용 정렬 테이블 출력
    valid = summary_df[summary_df["status"] == "OK"].copy()
    if valid.empty:
        return

    valid_sorted = valid.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    print("\n" + "═" * 72)
    print("  📊 그리드 서치 결과 (Macro F1 기준 정렬)")
    print("═" * 72)
    print(f"  {'Rank':<5} {'LR':>8} {'Epochs':>7} {'Accuracy':>10} "
          f"{'Macro F1':>10} {'Kappa':>8} {'Baseline':>9}")
    print(f"  {'─'*65}")

    for rank, (_, row) in enumerate(valid_sorted.iterrows(), start=1):
        baseline_mark = "★" if row["is_baseline"] else ""
        print(f"  {rank:<5} {row['learning_rate']:>8.0e} {int(row['epochs']):>7} "
              f"{row['accuracy']:>10.4f} {row['macro_f1']:>10.4f} "
              f"{row['cohen_kappa']:>8.4f} {baseline_mark:>9}")

    # 라벨별 F1 히트맵 테이블
    label_cols = [f"f1_{l}" for l in abuse_order if f"f1_{l}" in valid.columns]
    if label_cols:
        print(f"\n  📊 학대유형별 F1 (lr × epochs)")
        print(f"  {'LR':>8} {'Epochs':>7}", end="")
        for l in abuse_order:
            print(f"  {l:>8}", end="")
        print()
        print(f"  {'─'*60}")

        for _, row in valid_sorted.iterrows():
            baseline_mark = "★" if row["is_baseline"] else " "
            print(f"  {row['learning_rate']:>8.0e} {int(row['epochs']):>7}{baseline_mark}", end="")
            for l in abuse_order:
                col = f"f1_{l}"
                val = row.get(col, float("nan"))
                print(f"  {val:>8.4f}", end="")
            print()

    # 논문용 Latex 테이블 출력
    _print_latex_table(valid_sorted, abuse_order, out_dir)


def _print_latex_table(
    df: pd.DataFrame,
    abuse_order: List[str],
    out_dir: str,
) -> None:
    """논문 appendix용 LaTeX 테이블 생성."""

    label_map = {
        "성학대": "Sexual",
        "신체학대": "Physical",
        "정서학대": "Emotional",
        "방임": "Neglect",
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{KLUE-BERT Hyperparameter Grid Search Results (5-fold CV). "
        r"$\star$ denotes the configuration used in the main analysis.}",
        r"\label{tab:bert_gridsearch}",
        r"\begin{tabular}{cccccccc}",
        r"\hline",
    ]

    # 헤더
    header_cols = ["LR", "Epochs", "Acc.", "Macro F1", r"$\kappa$"] + \
                  [label_map.get(l, l) for l in abuse_order]
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        lr_str = f"\\num{{{row['learning_rate']:.0e}}}"
        ep_str = str(int(row["epochs"]))
        baseline_mark = r"$\star$" if row["is_baseline"] else ""

        cols = [
            f"{lr_str}{baseline_mark}",
            ep_str,
            f"{row['accuracy']:.4f}",
            f"{row['macro_f1']:.4f}",
            f"{row['cohen_kappa']:.4f}",
        ]
        for l in abuse_order:
            col = f"f1_{l}"
            val = row.get(col, float("nan"))
            cols.append(f"{val:.4f}" if not np.isnan(val) else "--")

        lines.append(" & ".join(cols) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex_str = "\n".join(lines)
    path = os.path.join(out_dir, "bert_gridsearch_table.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"\n  [저장] LaTeX 테이블 → {path}")
    print(f"\n  --- LaTeX 미리보기 ---")
    print(latex_str)


# ═══════════════════════════════════════════════════════════════════
#  모호성 지대 강화 분석 (논문 검증~7 보완)
# ═══════════════════════════════════════════════════════════════════

def analyze_ambiguity_zone_across_configs(
    summary_csv: str,
    out_dir: str = "bert_gridsearch_output",
) -> None:
    """
    그리드 서치 결과를 바탕으로,
    "최적 BERT 설정에서도 동일한 오분류 경계가 유지되는가"를 확인한다.

    이 분석 결과는 논문의 핵심 주장---
    "BERT와 CA의 구조적 불일치는 파인튜닝 부족이 아니라
    방법론적 이질성에서 기인한다"---을 지지하는 보강 증거가 된다.

    Parameters
    ----------
    summary_csv : str
        gridsearch_summary.csv 경로
    out_dir : str
        결과 저장 디렉토리
    """
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    valid = df[df["status"] == "OK"]

    if valid.empty:
        print("[WARN] 유효한 결과 없음")
        return

    best = valid.loc[valid["macro_f1"].idxmax()]
    baseline = valid[(valid["learning_rate"] == BASELINE_CONFIG["learning_rate"]) &
                     (valid["epochs"] == BASELINE_CONFIG["epochs"])]

    print("\n" + "═" * 72)
    print("  📊 모호성 지대 일관성 분석")
    print("═" * 72)
    print(f"  최적 설정: lr={best['learning_rate']:.0e}, epochs={int(best['epochs'])}")
    print(f"    Macro F1: {best['macro_f1']:.4f}")

    if not baseline.empty:
        b = baseline.iloc[0]
        delta = best["macro_f1"] - b["macro_f1"]
        print(f"\n  기존 설정: lr=2e-5, epochs=10")
        print(f"    Macro F1: {b['macro_f1']:.4f}")
        print(f"    성능 차이: {delta:+.4f}")

        if abs(delta) < 0.02:
            print("\n  ✅ 결론: 최적 BERT 세팅과 기존 세팅 간 성능 차이가 미미하다.")
            print("     이는 BERT와 CA의 구조적 불일치가 하이퍼파라미터 선택의")
            print("     문제가 아님을 시사한다.")
        else:
            print(f"\n  ⚠️  주의: 성능 차이 {delta:+.4f}가 유의미할 수 있다.")
            print("     최적 설정으로 검증~7을 재실행하는 것을 권장한다.")


# ═══════════════════════════════════════════════════════════════════
#  __main__
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    print("=" * 72)
    print("  KLUE-BERT 하이퍼파라미터 그리드 서치")
    print("=" * 72)

    _this = Path(__file__).resolve()
    _project_root = _this.parent
    for _ in range(5):
        if (_project_root / "data").exists():
            break
        _project_root = _project_root.parent

    _data_dir = _project_root / "data"
    _out = str(_project_root / "output" / "bert_gridsearch")

    print(f"\n  프로젝트: {_project_root}")
    print(f"  데이터  : {_data_dir}")

    if not _data_dir.exists():
        print(f"  ❌ data 폴더 없음: {_data_dir}")
        sys.exit(1)

    _jsons = sorted(glob.glob(str(_data_dir / "*.json")))
    if not _jsons:
        print("  ❌ JSON 없음")
        sys.exit(1)

    print(f"  JSON    : {len(_jsons)}개")

    abuse_order = ["성학대", "신체학대", "정서학대", "방임"]

    # 그리드 서치 실행
    best_config, summary_df = run_bert_grid_search(
        json_files=_jsons,
        abuse_order=abuse_order,
        out_dir=_out,
        only_negative=True,
        n_splits=5,
        bert_model_name="klue/bert-base",
        bert_max_length=256,
        bert_batch_size=16,
        param_grid={
            "learning_rate": [1e-5, 2e-5, 5e-5],
            "epochs":        [3, 5, 10],
        },
        random_state=42,
    )

    # 모호성 지대 일관성 분석
    summary_path = os.path.join(_out, "gridsearch_summary.csv")
    if os.path.exists(summary_path):
        analyze_ambiguity_zone_across_configs(summary_path, _out)

    print("\n" + "=" * 72)
    print("  ✅ 완료!")
    print(f"  결과: {_out}")
    print("=" * 72)
