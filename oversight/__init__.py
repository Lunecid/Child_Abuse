"""
Oversight pipeline — 11-stage multi-label classification with correction layers.

This package implements the two-part research pipeline:

    Part 1 (Stages 1-5): Diagnostic analysis
        Stage 1 : Corpus stratification (ALL → NEG → ABUSE_NEG)
        Stage 2 : Tokenization (Korean morphological analysis via Okt)
        Stage 3 : Log-odds computation (4 sub-corpora × stabilized log-odds)
        Stage 4 : Bridge word selection (diagnostic + correction sets)
        Stage 5 : Part 1 diagnostics (info loss, label composition, CA, stability)

    Part 2 (Stages 6-11): Multi-label classifier with correction
        Stage 6 : Primary multi-label classifier (TF-IDF + OneVsRest LR)
        Stage 7 : Ambiguous case identification
        Stage 8 : Correction layers (Method A: log-linear, Method C: separated)
        Stage 9 : Final prediction assembly
        Stage 10: Evidence utterance extraction
        Stage 11: Evaluation, baselines, and ablation

Stages 1-5 are thin wrappers around existing ``abuse_pipeline/`` functions.
Stages 6-11 are new code written in this package.

Legacy code (``abuse_pipeline/``) is never modified.

Package layout::

    oversight/
    ├── config.py          Configuration (OversightConfig dataclass)
    ├── _imports.py         Single entry point for all legacy imports
    ├── core/               Stage 1-2: corpus loading, stratification, tokenization
    ├── stats/              Stage 3-5: log-odds, bridge words, Part 1 diagnostics
    ├── models/             Stage 6-10: classifier, correction, prediction, evidence
    ├── evaluation/         Stage 11: metrics, baselines, ablation
    ├── scripts/            Entry-point scripts (run_pipeline, run_ablation, etc.)
    ├── outputs/            Generated outputs (CSV, LaTeX tables, figures)
    └── tests/              Unit / integration tests
"""

from oversight.config import OversightConfig

__all__ = ["OversightConfig"]
