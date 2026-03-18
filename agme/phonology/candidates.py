"""Candidate SR enumeration for a given UR.

Generates a set of plausible surface forms by applying single and double
edit operations (substitution, insertion, deletion) to the UR.

# ============================================================
# DECISION POINT: Candidate enumeration vs. FST sampling
# ============================================================
# Currently we enumerate candidates by exhaustive single-edit + random
# multi-edit operations on the UR.  This is fast and dependency-free but
# has two limitations:
#
#   1. The candidate set may miss low-probability-but-important SRs if
#      they require more than 2 edits from the UR.
#   2. The partition function Z = Σ_SR exp(-H(UR, SR)) is approximated
#      over the finite candidate set, not computed exactly.
#
# A cleaner alternative is to compile the *MAP constraints into a
# weighted FST (UR → SR) and:
#   a) sample SRs directly from the FST distribution (exact sampling)
#   b) compute the exact partition function via the shortest-distance
#      algorithm on the composed FST.
#
# This would require pynini or hfst.  The current approach is an
# approximation that is acceptable for small alphabets and short words.
#
# If switching to FST sampling, replace `candidates_for` with a function
# that returns weighted samples from the FST, and update
# `MaxEntPhonology.log_prob` to use the exact Z.
#
# ASK THE USER before implementing: the FST approach also enables the
# RT/production-task extension (see Future Extensions #7 in the plan).
# ============================================================
"""

from __future__ import annotations

import numpy as np

from agme.utils import random_edit


def candidates_for(
    ur: str,
    alphabet: list[str],
    max_length_delta: int = 2,
    n_random: int = 30,
    rng: np.random.Generator | None = None,
) -> set[str]:
    """Return candidate SRs for the given UR.

    Includes:
    - The UR itself (faithful candidate)
    - All single-operation edits within max_length_delta of len(ur)
    - n_random multi-edit random candidates
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates: set[str] = {ur}
    min_len = max(1, len(ur) - max_length_delta)
    max_len = len(ur) + max_length_delta

    # All single-substitution candidates
    for i in range(len(ur)):
        for c in alphabet:
            if c != ur[i]:
                cand = ur[:i] + c + ur[i + 1:]
                if min_len <= len(cand) <= max_len:
                    candidates.add(cand)

    # All single-deletion candidates
    for i in range(len(ur)):
        cand = ur[:i] + ur[i + 1:]
        if min_len <= len(cand) <= max_len:
            candidates.add(cand)

    # All single-insertion candidates
    for i in range(len(ur) + 1):
        for c in alphabet:
            cand = ur[:i] + c + ur[i:]
            if min_len <= len(cand) <= max_len:
                candidates.add(cand)

    # Random multi-edit candidates for better mixing
    for _ in range(n_random):
        cand = ur
        n_edits = int(rng.integers(1, 3))
        for _ in range(n_edits):
            cand = random_edit(cand, alphabet, rng)
        if min_len <= len(cand) <= max_len:
            candidates.add(cand)

    return candidates
