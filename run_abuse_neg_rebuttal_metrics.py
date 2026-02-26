from __future__ import annotations

import argparse
from pathlib import Path

from abuse_pipeline.analysis.abuse_neg_rebuttal_metrics import run_abuse_neg_rebuttal_metrics
from abuse_pipeline.core import common as C


def _collect_json_files(data_dir: Path) -> list[str]:
    return [str(p) for p in sorted(data_dir.glob("*.json")) if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ABUSE_NEG 단일 실행: "
            "교량 단어 적중률 정의 분리(전체 vs bridge-가능쌍) + "
            "경로3 우연 기준선(1/3) 주변분포 보정"
        )
    )
    parser.add_argument("--data_dir", type=str, required=True, help="JSON 디렉토리 경로")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="산출물 디렉토리 (기본: <project>/ver28_negOnly/rebuttal_metrics)",
    )
    parser.add_argument(
        "--bridge_csv",
        type=str,
        default=None,
        help="기존 bridge csv 경로 (없으면 ABUSE_NEG에서 재생성)",
    )
    parser.add_argument(
        "--ambiguity_csv",
        type=str,
        default=None,
        help=(
            "경로3 보정용 예측 테이블(csv). "
            "예: neg_gt_multilabel/ambiguity_zone_singlelabel.csv"
        ),
    )
    parser.add_argument("--sub_threshold", type=int, default=4, help="algo sub threshold")
    parser.add_argument("--no_clinical_text", action="store_true", help="algo 라벨링에서 임상 텍스트 비활성화")

    # bridge 생성 옵션
    parser.add_argument("--bridge_min_p1", type=float, default=0.40)
    parser.add_argument("--bridge_min_p2", type=float, default=0.25)
    parser.add_argument("--bridge_max_gap", type=float, default=0.20)
    parser.add_argument("--bridge_chi2_top_k", type=int, default=200)
    parser.add_argument("--bridge_count_min", type=int, default=5)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser().resolve()

    C.configure_output_dirs(subset_name="NEG_ONLY", base_dir=str(project_root))
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = Path(C.OUTPUT_DIR) / "rebuttal_metrics"

    json_files = _collect_json_files(data_dir)
    print("==============================================================")
    print("[RUN] ABUSE_NEG rebuttal metrics")
    print("[RUN] data_dir :", data_dir)
    print("[RUN] out_dir  :", out_dir)
    print("[RUN] n_json   :", len(json_files))
    print("==============================================================")

    if not json_files:
        print(f"[WARN] JSON 파일이 없습니다: {data_dir}")
        return

    ambiguity_csv = None
    if args.ambiguity_csv:
        ambiguity_csv = str(Path(args.ambiguity_csv).expanduser().resolve())
    else:
        auto_ambiguity = Path(C.NEG_GT_MULTILABEL_DIR) / "ambiguity_zone_singlelabel.csv"
        if auto_ambiguity.exists():
            ambiguity_csv = str(auto_ambiguity)

    result = run_abuse_neg_rebuttal_metrics(
        json_files=json_files,
        out_dir=str(out_dir),
        bridge_csv=str(Path(args.bridge_csv).expanduser().resolve()) if args.bridge_csv else None,
        ambiguity_csv=ambiguity_csv,
        sub_threshold=int(args.sub_threshold),
        use_clinical_text=not bool(args.no_clinical_text),
        bridge_min_p1=float(args.bridge_min_p1),
        bridge_min_p2=float(args.bridge_min_p2),
        bridge_max_gap=float(args.bridge_max_gap),
        bridge_chi2_top_k=int(args.bridge_chi2_top_k),
        bridge_count_min=int(args.bridge_count_min),
    )
    print("[RESULT]", result)


if __name__ == "__main__":
    main()
