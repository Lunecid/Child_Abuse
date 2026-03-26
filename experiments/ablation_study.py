from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.multilabel_experiment import build_config, run_bridge_ablation_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bridge-word ablations for the multilabel experiment.")
    parser.add_argument("--source-root", type=str, default=None, help="Root of the original Child_Abuse repo.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for experiment outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(source_root=args.source_root, output_dir=args.output_dir)
    run_bridge_ablation_study(config)


if __name__ == "__main__":
    main()
