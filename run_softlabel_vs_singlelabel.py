from __future__ import annotations

import argparse
from pathlib import Path

from abuse_pipeline.core import common as C
from abuse_pipeline.classifiers.softlabel_vs_singlelabel_analysis import (
    run_softlabel_vs_singlelabel_study,
)


def _collect_json_files(data_dir: Path) -> list[str]:
    return [str(p) for p in sorted(data_dir.glob("*.json")) if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A점수 벡터 soft-label 예측 모델 vs 단일라벨(main) 예측 모델 비교"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="JSON 디렉토리 경로")
    parser.add_argument("--out_dir", type=str, default=None, help="산출물 디렉토리")
    parser.add_argument("--only_negative", action="store_true", help="정서군='부정'만 포함")
    parser.add_argument("--dedupe_by_doc_id", action="store_true", help="doc_id 기준 중복 제거")
    parser.add_argument("--alpha", type=float, default=0.5, help="A점수 soft label 변환 add-alpha")
    parser.add_argument("--min_a_sum", type=int, default=1, help="A점수 합 최소값")
    parser.add_argument("--n_splits", type=int, default=5, help="Stratified K-fold 분할 수")
    parser.add_argument("--random_state", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser().resolve()

    subset_name = "NEG_ONLY" if args.only_negative else "ALL"
    C.configure_output_dirs(subset_name=subset_name, base_dir=str(project_root))

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = Path(C.MODEL_COMPARISON_DIR) / "a_score_softlabel_vs_singlelabel"

    json_files = _collect_json_files(data_dir)
    print("========================================================================")
    print("[RUN] data_dir =", data_dir)
    print("[RUN] out_dir  =", out_dir)
    print("[RUN] JSON 파일 개수:", len(json_files))
    print(
        "[RUN] settings:",
        f"only_negative={args.only_negative}, dedupe_by_doc_id={args.dedupe_by_doc_id},",
        f"alpha={args.alpha}, min_a_sum={args.min_a_sum}",
    )
    print("========================================================================")

    if not json_files:
        print(f"[WARN] JSON 파일이 없습니다: {data_dir}")
        return

    result = run_softlabel_vs_singlelabel_study(
        json_files=json_files,
        out_dir=str(out_dir),
        only_negative=args.only_negative,
        dedupe_by_doc_id=args.dedupe_by_doc_id,
        alpha=args.alpha,
        min_a_sum=args.min_a_sum,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    print("[RESULT]", result)


if __name__ == "__main__":
    main()
