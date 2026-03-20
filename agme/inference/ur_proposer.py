"""UR proposal distribution for a given SR span.

Role in inference
-----------------
The segmenter cannot enumerate all possible URs — the space is infinite.
URProposer acts as an importance-sampling proposal distribution q(ur | sr).
For each span, it returns a small weighted set of candidate URs.

The segmenter then computes the *unnormalized* joint scores
    p(ur, sr | model) ∝ P(ur | PYP_cache) · P(sr | ur, grammar)
for each proposal and marginalises via log-sum-exp to get the span score.
The UR is finally *sampled* (not MAP'd) from the conditional proportional
to these scores during the backward pass.

Proposal sources
----------------
1. The SR itself — the "faithful" candidate.  Given highest weight so that
   the model always seriously considers no alternation.
2. Existing UR types in the PYP lexicon — rich-get-richer effect; known
   morphemes are more likely candidates than novel ones.
3. Single-operation P-map-weighted edits of the SR — substitutions,
   insertions, deletions, weighted by 1/(panphon_distance + ε) so that
   perceptually cheap alternations (e.g. s→z) are tried more than costly
   ones (e.g. s→m).
4. A small number of random multi-edit candidates for mixing.
"""

from __future__ import annotations

import math

import numpy as np

from agme.features import build_distance_matrix
from agme.utils import random_edit


class URProposer:
    """Proposes candidate URs for an observed SR span.

    Proposal sources (in priority order):
    1. The SR itself (faithful candidate, high weight)
    2. Existing UR types in the morpheme-class lexicon
    3. P-map-weighted single-operation edits of the SR

    Parameters
    ----------
    alphabet : list[str]
        Shared phoneme inventory.
    distance_matrix : dict[(x,y), float]
        Pairwise panphon distances used to weight proposals.
    faithful_weight : float
        Proposal weight assigned to the faithful candidate (SR itself).
    n_random : int
        Number of random multi-edit candidates to include.
    """

    def __init__(
        self,
        alphabet: list[str],
        distance_matrix: dict[tuple[str, str], float],
        faithful_weight: float = 100.0,
        n_random: int = 20,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.alphabet = alphabet
        self.distance_matrix = distance_matrix
        self.faithful_weight = faithful_weight
        self.n_random = n_random
        self.rng = rng or np.random.default_rng()
        # Cache of pre-computed single-edit candidates + weights per SR span.
        # Single-edit weights depend only on (sr_span, candidate, alphabet,
        # distance_matrix) — not on the lexicon or sweep count — so they are
        # identical every time the same SR span appears.  With type-level
        # inference each span string recurs every sweep, so after the first
        # encounter (first sweep) all subsequent propose() calls for that span
        # skip the O(len × |alph|) _edit_weight enumeration entirely.
        self._single_edit_cache: dict[str, list[tuple[str, float]]] = {}

    def _precompute_single_edits(self, sr_span: str) -> list[tuple[str, float]]:
        """Return (candidate, weight) for all single-edit neighbours of sr_span.

        Result is cached permanently — weights depend only on the SR span,
        the alphabet, and the distance matrix, none of which change during
        training.
        """
        if sr_span not in self._single_edit_cache:
            results: list[tuple[str, float]] = []
            for i in range(len(sr_span)):
                for c in self.alphabet:
                    if c != sr_span[i]:
                        cand = sr_span[:i] + c + sr_span[i + 1:]
                        results.append((cand, self._edit_weight(sr_span, cand)))
                # Deletion
                cand = sr_span[:i] + sr_span[i + 1:]
                if cand:
                    results.append((cand, self._edit_weight(sr_span, cand)))
            # Insertions
            for i in range(len(sr_span) + 1):
                for c in self.alphabet:
                    cand = sr_span[:i] + c + sr_span[i:]
                    results.append((cand, self._edit_weight(sr_span, cand)))
            self._single_edit_cache[sr_span] = results
        return self._single_edit_cache[sr_span]

    def _edit_weight(self, sr: str, ur_candidate: str) -> float:
        """Proposal weight proportional to 1 / (pmap_cost + ε)."""
        if ur_candidate == sr:
            return self.faithful_weight
        total_dist = 0.0
        for sc, uc in zip(sr, ur_candidate):
            if sc != uc:
                total_dist += self.distance_matrix.get((sc, uc), 1.0)
        # Penalise length differences
        total_dist += abs(len(sr) - len(ur_candidate)) * 0.5
        return 1.0 / (total_dist + 0.5)

    def propose(
        self,
        sr_span: str,
        lexicon: dict[str, int],
        top_k: int = 15,
    ) -> list[tuple[str, float]]:
        """Return a weighted list of (ur_candidate, proposal_weight).

        Parameters
        ----------
        sr_span : str
            The observed surface form for this span.
        lexicon : dict[str, int]
            Current morpheme-class lexicon (ur → count).
        top_k : int
            Keep only the top-k candidates by weight.
        """
        candidates: dict[str, float] = {}

        # 1. Faithful candidate
        candidates[sr_span] = self.faithful_weight

        # 2. Existing lexicon entries
        for ur, count in lexicon.items():
            w = self._edit_weight(sr_span, ur) * (count ** 0.5)
            candidates[ur] = candidates.get(ur, 0.0) + w

        # 3. Single-operation edits of the SR (cached per span)
        for cand, w in self._precompute_single_edits(sr_span):
            candidates[cand] = candidates.get(cand, 0.0) + w

        # 4. Random multi-edit candidates
        for _ in range(self.n_random):
            cand = random_edit(sr_span, self.alphabet, self.rng)
            if cand:
                candidates[cand] = candidates.get(cand, 0.0) + self._edit_weight(sr_span, cand)

        # Trim to top-k
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])[:top_k]
        return sorted_cands
