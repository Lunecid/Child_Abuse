from __future__ import annotations

import argparse
from pathlib import Path

from abuse_pipeline.neg_gt_multilabel_analysis import run_neg_gt_multilabel_study


def _collect_json_files(data_dir: Path) -> list[str]:
    return [str(p) for p in sorted(data_dir.glob("*.json")) if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABUSE_NEG + GT 존재 케이스에서 main+sub(다중라벨) vs main(단일라벨) 비교"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="JSON 디렉토리 경로")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="산출물 디렉토리 (기본값: <project_root>/paper_neg_gt_multilabel)",
    )
    parser.add_argument("--gt_field", type=str, default="학대의심", help="GT 필드명")
    parser.add_argument("--n_splits", type=int, default=5, help="Stratified K-fold 분할 수")
    parser.add_argument("--threshold", type=float, default=0.5, help="다중라벨 확률 임계값")
    parser.add_argument("--random_state", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (project_root / "paper_neg_gt_multilabel")
    )

    json_files = _collect_json_files(data_dir)
    print("========================================================================")
    print("[RUN] data_dir =", data_dir)
    print("[RUN] out_dir  =", out_dir)
    print("[RUN] JSON 파일 개수:", len(json_files))
    print("========================================================================")

    if not json_files:
        print(f"[WARN] JSON 파일이 없습니다: {data_dir}")
        return

    result = run_neg_gt_multilabel_study(
        json_files=json_files,
        out_dir=str(out_dir),
        gt_field=args.gt_field,
        n_splits=args.n_splits,
        random_state=args.random_state,
        threshold=args.threshold,
    )
    print("[RESULT]", result)


if __name__ == "__main__":
    main()
