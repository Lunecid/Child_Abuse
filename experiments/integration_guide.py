from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_ROOT = Path("/Users/lunecid/project/Child_Abuse")


def resolve_source_root() -> Path:
    env_root = os.environ.get("CHILD_ABUSE_SOURCE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if (PROJECT_ROOT / "data").is_dir():
        return PROJECT_ROOT
    return DEFAULT_SOURCE_ROOT


def default_stage_paths(source_root: Path) -> dict[str, Path]:
    neg_root = source_root / "ver28_negOnly"
    main_sub_root = neg_root / "main_sub_analysis"
    abuse_stats_root = neg_root / "abuse" / "stats"
    return {
        "data_dir": source_root / "data",
        "sub_scores": main_sub_root / "stage1_sub_scores.csv",
        "entropy": main_sub_root / "stage4_entropy.csv",
        "bridge_words": abuse_stats_root / "stage0_bridge_words.csv",
        "bridge_words_doclevel": abuse_stats_root / "stage0_bridge_words_doclevel.csv",
    }


def render_guide(config_source_root: str | None = None, output_dir: str | None = None) -> str:
    source_root = (
        Path(config_source_root).expanduser().resolve()
        if config_source_root
        else resolve_source_root()
    )
    resolved_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (PROJECT_ROOT / "results")
    )
    paths = default_stage_paths(source_root)
    return "\n".join(
        [
            "Multilabel Experiment Integration Guide",
            f"- working_copy: {Path(__file__).resolve().parents[1]}",
            f"- source_root: {source_root}",
            f"- data_dir: {paths['data_dir']}",
            f"- sub_scores: {paths['sub_scores']}",
            f"- entropy_reference: {paths['entropy']}",
            f"- bridge_reference: {paths['bridge_words_doclevel'] if paths['bridge_words_doclevel'].exists() else paths['bridge_words']}",
            f"- output_dir: {resolved_output}",
            "",
            "Recommended commands",
            f"- .venv python: {source_root / '.venv' / 'bin' / 'python'} experiments/run_experiment.py --source-root \"{source_root}\" --output-dir \"{resolved_output}\"",
            f"- ablation: {source_root / '.venv' / 'bin' / 'python'} experiments/ablation_study.py --source-root \"{source_root}\" --output-dir \"{resolved_output}\"",
        ]
    )


def main() -> None:
    print(render_guide())


if __name__ == "__main__":
    main()
