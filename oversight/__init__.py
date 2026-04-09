"""
Oversight prevention tool package.

This package implements the 3-layer pipeline described in PROJECT_PLAN.md
Section 2 (Layer 1: multi-channel candidate retrieval, Layer 2: utterance-level
classification, Layer 3: child-level aggregation).

The goal is NOT classification but oversight prevention: given a clinician's
main label assignment, alert when evidence for companion abuse types is present
in the child's utterances.

This package is independent from the legacy `abuse_pipeline/` package but
imports utility functions from it where appropriate (see `_imports.py`).
Legacy code is never modified.
"""

from oversight.config import OversightConfig

__all__ = ["OversightConfig"]
