from __future__ import annotations

import argparse
from pathlib import Path

from abuse_pipeline.analysis.gt_alg_gap_diagnosis import run_gt_alg_gap_diagnosis
from abuse_pipeline.core import common as C


def _collect_json_files(data_dir: Path) -> list[str]:
    return [str(p) for p in sorted(data_dir.glob("*.json")) if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why ALG labels and GT labels differ: "
            "clinical-text effect vs score-based threshold/priority effect."
        )
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to JSON directory")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <project>/ver28_negOnly/label_comparison)",
    )
    parser.add_argument("--gt_field", type=str, default="학대의심", help="GT field name")
    parser.add_argument(
        "--only_negative",
        action="store_true",
        help="Use only ABUSE_NEG corpus (classify_child_group == 부정)",
    )
    parser.add_argument(
        "--no_dedupe_by_doc_id",
        action="store_true",
        help="Do not deduplicate by doc_id (default dedupe enabled)",
    )
    parser.add_argument("--sub_threshold", type=int, default=4, help="Sub-label threshold")
    parser.add_argument(
        "--main_threshold",
        type=int,
        default=6,
        help="Main-label threshold is score > main_threshold",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser().resolve()

    C.configure_output_dirs(subset_name="NEG_ONLY", base_dir=str(project_root))
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = Path(C.LABEL_COMPARISON_DIR)

    json_files = _collect_json_files(data_dir)
    print("==============================================================")
    print("[RUN] GT vs ALG gap diagnosis")
    print("[RUN] data_dir :", data_dir)
    print("[RUN] out_dir  :", out_dir)
    print("[RUN] n_json   :", len(json_files))
    print("==============================================================")

    if not json_files:
        print(f"[WARN] No JSON files found: {data_dir}")
        return

    result = run_gt_alg_gap_diagnosis(
        json_files=json_files,
        out_dir=str(out_dir),
        gt_field=args.gt_field,
        only_negative=bool(args.only_negative),
        dedupe_by_doc_id=not bool(args.no_dedupe_by_doc_id),
        sub_threshold=int(args.sub_threshold),
        main_threshold=int(args.main_threshold),
    )
    print("[RESULT]", result)


if __name__ == "__main__":
    main()
