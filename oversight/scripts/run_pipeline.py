"""
End-to-end pipeline: Stage 1 → 11.

Usage:
    python -m oversight.scripts.run_pipeline --data_dir data/ --output_dir oversight/outputs/
    python -m oversight.scripts.run_pipeline --skip_diagnosis --skip_evaluation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oversight.config import OversightConfig

# Stage 1-2
from oversight.core.corpus import stratify_corpus, ensure_tokenized, save_corpus_summary

# Stage 3
from oversight.stats.logodds import compute_all_logodds, save_logodds

# Stage 4
from oversight.stats.bridge import select_bridge_words, save_bridge

# Stage 5
from oversight.stats.diagnosis import run_part1_diagnosis, save_diagnosis

# Stage 6
from oversight.models.classifier import train_primary_classifier, save_classifier_results

# Stage 7
from oversight.models.ambiguous import identify_ambiguous_cases, save_ambiguous_results

# Stage 8
from oversight.models.correction import apply_corrections, save_correction_results

# Stage 9
from oversight.models.prediction import assemble_final_predictions, save_predictions

# Stage 10
from oversight.models.evidence import extract_evidence_utterances, save_evidence

# Stage 11
from oversight.evaluation.metrics import evaluate_predictions, save_evaluation
from oversight.evaluation.baselines import run_baselines, save_baselines
from oversight.evaluation.ablation import run_ablation, save_ablation


def _log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}")


def run_pipeline(config: OversightConfig, args: argparse.Namespace) -> None:
    """Run the full 11-stage pipeline."""
    t0 = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1-2: Corpus ──
    _log("Stage 1", "Loading and stratifying corpus...")
    corpus = stratify_corpus(config)
    corpus = ensure_tokenized(corpus)
    save_corpus_summary(corpus, output_dir)
    _log("Stage 1-2", (
        f"Done. NEG={corpus.neg_count}, NEU={corpus.neu_count}, POS={corpus.pos_count}, "
        f"ABUSE_NEG={len(corpus.abuse_neg_df)}, train={len(corpus.train_df)}"
    ))

    # ── Stage 3: Log-odds ──
    _log("Stage 3", "Computing log-odds for 4 sub-corpora...")
    logodds_result = compute_all_logodds(corpus, config)
    save_logodds(logodds_result, output_dir)
    _log("Stage 3", (
        f"Done. Rows: ALL={len(logodds_result.logodds_all_5class)}, "
        f"NEG={len(logodds_result.logodds_neg_4class)}, "
        f"ABUSE_NEG={len(logodds_result.logodds_abuse_neg_4class)}, "
        f"Train={len(logodds_result.logodds_train_5class)}"
    ))

    # ── Stage 4: Bridge words ──
    _log("Stage 4", "Selecting bridge words...")
    bridge_result = select_bridge_words(logodds_result, config)
    save_bridge(bridge_result, output_dir)
    _log("Stage 4", (
        f"Done. Diagnostic={len(bridge_result.bridge_diagnostic)}, "
        f"Correction={len(bridge_result.bridge_correction)}, "
        f"Overlap={bridge_result.overlap_count} ({bridge_result.overlap_ratio:.1%})"
    ))

    # ── Stage 5: Part 1 diagnostics ──
    if not args.skip_diagnosis:
        _log("Stage 5", "Running Part 1 diagnostics...")
        diagnosis_result = run_part1_diagnosis(corpus, logodds_result, bridge_result, config)
        save_diagnosis(diagnosis_result, output_dir)
        _log("Stage 5", "Done.")
    else:
        _log("Stage 5", "Skipped (--skip_diagnosis)")

    # ── Stage 6: Primary classifier ──
    _log("Stage 6", "Training primary multi-label classifier (5-fold CV)...")
    classifier_result = train_primary_classifier(corpus, config)
    save_classifier_results(classifier_result, corpus.train_df, output_dir)
    _log("Stage 6", f"Done. OOF shape: {classifier_result.oof_probs.shape}")

    # ── Stage 7: Ambiguous case identification ──
    _log("Stage 7", "Identifying ambiguous cases...")
    ambiguous_result = identify_ambiguous_cases(
        corpus, classifier_result, bridge_result, config
    )
    save_ambiguous_results(ambiguous_result, output_dir)
    _log("Stage 7", f"Done. Ambiguous: {len(ambiguous_result.ambiguous_indices)}/{len(corpus.train_df)}")

    # ── Stage 8: Correction ──
    _log("Stage 8", "Applying correction layers (Method A + Method C)...")
    correction_result = apply_corrections(
        corpus, classifier_result, ambiguous_result,
        logodds_result, bridge_result, config
    )
    save_correction_results(correction_result, corpus.train_df, output_dir)
    _log("Stage 8", "Done.")

    # ── Stage 9: Final predictions ──
    _log("Stage 9", "Assembling final predictions...")
    prediction_result = assemble_final_predictions(
        corpus, classifier_result, ambiguous_result, correction_result, config
    )
    save_predictions(prediction_result, corpus.train_df, output_dir)
    _log("Stage 9", "Done.")

    # ── Stage 10: Evidence ──
    _log("Stage 10", "Extracting evidence utterances...")
    for method in ("method_a", "method_c"):
        evidence = extract_evidence_utterances(
            corpus, prediction_result, logodds_result, bridge_result, config,
            method=method,
        )
        save_evidence(evidence, output_dir, method=method)
    _log("Stage 10", "Done.")

    # ── Stage 11: Evaluation ──
    if not args.skip_evaluation:
        _log("Stage 11", "Evaluating predictions...")
        for method in ("method_a", "method_c"):
            metrics = evaluate_predictions(prediction_result, method=method)
            save_evaluation(metrics, output_dir, method=method)
            if metrics:
                _log("Stage 11", (
                    f"{method}: micro-F1={metrics.get('micro_f1', 0):.4f}, "
                    f"macro-F1={metrics.get('macro_f1', 0):.4f}"
                ))

        _log("Stage 11", "Running baselines...")
        baseline_results = run_baselines(
            corpus, classifier_result, prediction_result, logodds_result, config
        )
        save_baselines(baseline_results, output_dir)

        _log("Stage 11", "Running ablation (sweep → component → method comparison)...")
        ablation_result = run_ablation(
            corpus, classifier_result, ambiguous_result,
            logodds_result, bridge_result, config
        )
        save_ablation(ablation_result, output_dir)

        # Log component ablation results
        if not ablation_result.component_table.empty:
            _log("Stage 11", "Component ablation (Method C):")
            for _, row in ablation_result.component_table.iterrows():
                delta = row.get("delta_macro_f1_pp", 0)
                delta_str = f" ({delta:+.1f} pp)" if delta != 0 else ""
                _log("Stage 11", (
                    f"  {row['condition']}: "
                    f"macro-F1={row['macro_f1']:.4f}{delta_str}"
                ))

        # Log method comparison
        if not ablation_result.method_table.empty:
            _log("Stage 11", "Method comparison at full correction:")
            for _, row in ablation_result.method_table.iterrows():
                _log("Stage 11", (
                    f"  Method {row['method']} ({row['params']}): "
                    f"macro-F1={row['macro_f1']:.4f}"
                ))

        _log("Stage 11", (
            f"Optimal: β_B*={ablation_result.beta_b_star}, "
            f"β_L*={ablation_result.beta_l_star}, "
            f"λ*={ablation_result.lambda_star}"
        ))
        _log("Stage 11", "Done.")
    else:
        _log("Stage 11", "Skipped (--skip_evaluation)")

    elapsed = time.time() - t0
    _log("DONE", f"Pipeline completed in {elapsed:.1f}s. Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 11-stage oversight pipeline."
    )
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="Path to JSON data directory (default: data/)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Path to output directory (default: oversight/outputs/)"
    )
    parser.add_argument(
        "--skip_diagnosis", action="store_true",
        help="Skip Stage 5 (Part 1 diagnostics)"
    )
    parser.add_argument(
        "--skip_evaluation", action="store_true",
        help="Skip Stage 11 (evaluation, baselines, ablation)"
    )
    parser.add_argument(
        "--classifier_type", choices=["lr", "svm"], default="lr",
        help="Classifier type (default: lr)"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )

    args = parser.parse_args()

    config = OversightConfig()
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    config.classifier_type = args.classifier_type
    config.n_splits = args.n_splits

    run_pipeline(config, args)


if __name__ == "__main__":
    main()
