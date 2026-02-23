from __future__ import annotations

import glob
import os
from pathlib import Path
from abuse_pipeline import common as C
from abuse_pipeline.pipeline import run_pipeline
from abuse_pipeline.common import DATA_JSON_DIR
data_dir = Path(DATA_JSON_DIR)


def _collect_json_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    return [p for p in paths if os.path.isfile(p)]


def main():
    project_root = Path(__file__).resolve().parent.parent  # .../Childeren
    data_dir = project_root / "data"

    # ✅ 반드시 먼저 출력 디렉토리 세팅
    C.configure_output_dirs(subset_name="NEG")   # 또는 out_root를 직접 지정해도 됨

    print("========================================================================")
    print("[RUN] subset = ALL (only_negative=False)")
    print("[RUN] OUTPUT_DIR =", C.OUTPUT_DIR)  # ✅ C.OUTPUT_DIR로 확인
    json_files = list(data_dir.glob("*.json"))
    print("[RUN] JSON 파일 개수:", len(json_files))
    print("========================================================================")

    run_pipeline(json_files, subset_name="NEG", only_negative=True)

if __name__ == "__main__":
    main()
