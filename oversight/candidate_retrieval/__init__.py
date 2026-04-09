"""
Multi-channel candidate utterance retrieval (Layer 1).

Channels:
    A: Bridge word anchored retrieval
    B: Type-specific log-odds anchored retrieval
    C: Model uncertainty / salience based retrieval

Each channel is independent. If one channel fails on a given boundary pair,
others still contribute candidates. This avoids the single-point-of-failure
problem described in PROJECT_PLAN.md Section 4.4.

Stage 0 only creates this package skeleton. The actual channel implementations
are added in Stage 3.
"""
