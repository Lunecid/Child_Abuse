#!/usr/bin/env python3
"""
정보 복원 실험 실행.

기록 시스템의 단일 레이블 제약으로 소실된 동반 학대 정보를,
아동 발화 텍스트로부터 복원하는 실험 파이프라인.

사용법:
  python run_information_recovery.py --data_dir ./data
  python run_information_recovery.py --data_dir ./data --output_dir ./results/recovery
  python run_information_recovery.py --data_dir ./data --ablation
  python run_information_recovery.py --data_dir ./data --skip_viz
"""
from __future__ import annotations
import matplotlib
matplotlib.rcParams['font.family'] = 'NanumGothic'
import argparse
import glob
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Information Recovery Pipeline: GT-anchored multi-label experiment"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="JSON 데이터 디렉토리 (기본: ./data)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="결과 출력 디렉토리 (기본: ./results/recovery)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Ablation study만 실행",
    )
    parser.add_argument(
        "--skip_viz", action="store_true",
        help="시각화 건너뛰기",
    )
    parser.add_argument(
        "--skip_experiment", action="store_true",
        help="분류 실험 건너뛰기 (데이터셋 분석만)",
    )
    args = parser.parse_args()

    # Data directory
    from abuse_pipeline.core import common as C

    data_dir = args.data_dir or C.DATA_JSON_DIR or "/Users/lunecid/project/Child_Abuse/data"
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not json_files:
        print(f"[ERROR] JSON 파일 없음: {data_dir}")
        sys.exit(1)
    print(f"[INFO] JSON 파일 {len(json_files)}개 로드: {data_dir}")

    # Output directory
    output_dir = Path(args.output_dir or os.path.join(C.BASE_DIR, "results", "recovery"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "figures", exist_ok=True)

    # ── Step 1: GT 기반 데이터셋 구축 ──
    from abuse_pipeline.experiments.information_recovery import (
        build_gt_anchored_dataset,
        compute_label_composition,
        quantify_information_loss,
        run_recovery_experiment,
        analyze_recovery_failures,
        RecoveryConfig,
    )

    print("\n" + "=" * 60)
    print("  Step 1: GT 기반 데이터셋 구축")
    print("=" * 60)
    dataset_df = build_gt_anchored_dataset(json_files, only_negative=True)
    if dataset_df.empty:
        print("[ERROR] 유효 데이터 0건. 종료.")
        sys.exit(1)

    dataset_df.to_csv(output_dir / "dataset_gt_anchored.csv", encoding="utf-8-sig", index=False)
    print(f"  GT 기준 데이터셋: {len(dataset_df)}건")
    print(f"  gt_main 분포:\n{dataset_df['gt_main'].value_counts().to_string()}")
    print(f"  n_labels 분포:\n{dataset_df['n_labels'].value_counts().sort_index().to_string()}")

    # ── Step 3: 레이블 구성 분석 ──
    print("\n" + "=" * 60)
    print("  Step 3: 레이블 구성 분석 (main-only vs main+sub)")
    print("=" * 60)
    composition = compute_label_composition(dataset_df)
    print(f"  Total: {composition['total']}")
    print(f"  Group A (main only): {composition['group_a_count']} ({composition['group_a_pct']:.1f}%)")
    print(f"  Group B (main+sub):  {composition['group_b_count']} ({composition['group_b_pct']:.1f}%)")
    print(f"  Group C (algo blind): {composition['group_c_count']} ({composition['group_c_pct']:.1f}%)")
    composition["detail_df"].to_csv(
        output_dir / "label_composition_report.csv", encoding="utf-8-sig", index=False
    )

    # ── Step 4: 정보 손실 정량화 ──
    print("\n" + "=" * 60)
    print("  Step 4: 정보 손실 정량화")
    print("=" * 60)
    loss = quantify_information_loss(dataset_df)
    print(f"  소실 레이블 총 건수: {loss['total_lost_labels']}")
    print(f"  소실 있는 아동 수: {loss['n_children_with_loss']}")
    print(f"  아동 당 평균 소실: {loss['avg_lost_per_child']:.2f}")
    print(f"  유형별 소실: {loss['lost_by_type']}")

    loss["loss_detail_df"].to_csv(
        output_dir / "information_loss_detail.csv", encoding="utf-8-sig", index=False
    )
    loss["loss_pair_matrix"].to_csv(
        output_dir / "information_loss_pair_matrix.csv", encoding="utf-8-sig"
    )
    summary_loss = pd.DataFrame([{
        "total_lost_labels": loss["total_lost_labels"],
        "n_children_with_loss": loss["n_children_with_loss"],
        "avg_lost_per_child": loss["avg_lost_per_child"],
        **{f"lost_{k}": v for k, v in loss["lost_by_type"].items()},
    }])
    summary_loss.to_csv(
        output_dir / "information_loss_summary.csv", encoding="utf-8-sig", index=False
    )

    # ── Step 2: 시각화 ──
    if not args.skip_viz:
        print("\n" + "=" * 60)
        print("  Step 2: 시각화")
        print("=" * 60)
        from abuse_pipeline.experiments.recovery_visualizations import (
            plot_cooccurrence_heatmap,
            plot_upset,
            plot_jaccard_matrix,
            plot_network_graph,
        )
        fig_dir = output_dir / "figures"
        plot_cooccurrence_heatmap(dataset_df, fig_dir)
        plot_upset(dataset_df, fig_dir)
        plot_jaccard_matrix(dataset_df, fig_dir)
        plot_network_graph(dataset_df, fig_dir)

    # ── Step 5+6+7: 분류 실험 ──
    config = RecoveryConfig(output_dir=output_dir)

    if args.ablation:
        print("\n" + "=" * 60)
        print("  Ablation Study")
        print("=" * 60)
        from abuse_pipeline.experiments.recovery_ablation import run_recovery_ablation
        run_recovery_ablation(dataset_df, config)

    elif not args.skip_experiment:
        print("\n" + "=" * 60)
        print("  Step 5-7: 복원 실험")
        print("=" * 60)
        results = run_recovery_experiment(dataset_df, config)

        if results.get("status") == "done":
            analyze_recovery_failures(results, dataset_df, output_dir)

    print("\n" + "=" * 60)
    print(f"  완료. 결과: {output_dir}")
    print("=" * 60)


# Need pandas for the summary DataFrame
import pandas as pd

if __name__ == "__main__":
    main()
