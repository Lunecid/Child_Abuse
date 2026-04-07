#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_raw_score_evidence.py
════════════════════════════════════════════════════════════════
원점수 기반 다영역 동시 출현 분석의 진입점.

PyCharm standalone 실행과 패키지 모듈 실행 모두를 지원하는 얇은 wrapper.
실제 분석 로직은 raw_score_evidence.main()에 있다.

실행 예:
  python run_raw_score_evidence.py
  python run_raw_score_evidence.py --data_dir ./data
  python run_raw_score_evidence.py --out_dir ./outputs/revision/raw_score_evidence
  python -m abuse_pipeline.revision.run_raw_score_evidence
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_syspath() -> None:
    """PyCharm / 직접 실행 시 프로젝트 루트를 sys.path에 추가."""
    this_file = Path(__file__).resolve()
    revision_dir = this_file.parent          # abuse_pipeline/revision
    pkg_dir = revision_dir.parent            # abuse_pipeline
    proj_root = pkg_dir.parent               # project root

    for p in [proj_root, pkg_dir]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


_bootstrap_syspath()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="원점수 기반 다영역 동시 출현 분석 (본문 Section 2.2 / 3.1.X)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="JSON 데이터 디렉토리 (기본: project/data)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="산출물 디렉토리 (기본: outputs/revision/raw_score_evidence)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="난수 시드 (기본: 42)",
    )
    args = parser.parse_args()

    from abuse_pipeline.revision.raw_score_evidence import main
    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
