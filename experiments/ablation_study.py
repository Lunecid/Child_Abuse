from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abuse_pipeline.experiments.ablation_study import cli_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    _, remaining = parser.parse_known_args()
    return argparse.Namespace(remaining=remaining)


def main() -> None:
    args = parse_args()
    cli_main(args.remaining)


if __name__ == "__main__":
    main()
