from __future__ import annotations

import argparse
from pathlib import Path

from abuse_pipeline.core import common as C
from abuse_pipeline.classifiers.neg_gt_multilabel_analysis import run_neg_gt_multilabel_study


def _collect_json_files(data_dir: Path) -> list[str]:
    return [str(p) for p in sorted(data_dir.glob("*.json")) if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABUSE_NEG + GT 존재 케이스에서 4종 분류기(LR, RF, SVM, KLUE-BERT)로 "
        "main+sub(다중라벨) vs main(단일라벨) 비교"
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

    # BERT 하이퍼파라미터
    parser.add_argument("--bert_model", type=str, default="klue/bert-base", help="BERT 모델명")
    parser.add_argument("--bert_max_length", type=int, default=256, help="BERT 최대 시퀀스 길이")
    parser.add_argument("--bert_batch_size", type=int, default=16, help="BERT 배치 크기")
    parser.add_argument("--bert_epochs", type=int, default=10, help="BERT 학습 에포크")
    parser.add_argument("--bert_lr", type=float, default=2e-5, help="BERT 학습률")
    parser.add_argument("--skip_bert", action="store_true", help="BERT 분류기 건너뛰기")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser().resolve()

    # configure_output_dirs 호출 → 통합 출력 디렉토리 설정
    C.configure_output_dirs(subset_name="NEG_ONLY", base_dir=str(project_root))

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = Path(C.NEG_GT_MULTILABEL_DIR)

    json_files = _collect_json_files(data_dir)
    print("========================================================================")
    print("[RUN] data_dir =", data_dir)
    print("[RUN] out_dir  =", out_dir)
    print("[RUN] JSON 파일 개수:", len(json_files))
    print(f"[RUN] 분류기: LR, RF, SVM" + (", KLUE-BERT" if not args.skip_bert else ""))
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
        bert_model_name=args.bert_model,
        bert_max_length=args.bert_max_length,
        bert_batch_size=args.bert_batch_size,
        bert_epochs=args.bert_epochs,
        bert_lr=args.bert_lr,
        skip_bert=args.skip_bert,
    )
    print("[RESULT]", result)


if __name__ == "__main__":
    main()
