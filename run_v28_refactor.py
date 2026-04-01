from __future__ import annotations

import argparse
from pathlib import Path

from abuse_pipeline.pipeline import run_pipeline


def _collect_json_files(data_dir: Path) -> list[str]:
    return [str(p) for p in sorted(data_dir.glob("*.json")) if p.is_file()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="JSON 파일들이 있는 디렉토리 경로 (기본값: <project_root>/data)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else (project_root / "data")
    json_files = _collect_json_files(data_dir)

    print("========================================================================")
    print("[RUN] data_dir =", data_dir)
    print("[RUN] JSON 파일 개수:", len(json_files))
    print("========================================================================")

    if not json_files:
        print(f"[WARN] JSON 파일이 없습니다: {data_dir}")
        return

    # run_pipeline(json_files, subset_name="ALL", only_negative=False)
    run_pipeline(json_files, subset_name="NEG_ONLY", only_negative=True)

if __name__ == "__main__":
    main()
