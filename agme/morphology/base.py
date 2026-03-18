"""Base distributions G₀ over morpheme strings.

Role in the PYP
---------------
Each PitmanYorCache needs a base distribution G₀ from which novel morpheme
types are drawn when a customer opens a new table (i.e. proposes a new
lexical type).  CharacterBaseDistribution implements G₀ as:

    P(word) = P(|word|) × Π_i P(word[i])

where:
  - P(|word|) is a length prior (Poisson for stems, Geometric for affixes)
  - P(word[i]) is a Dirichlet-multinomial character distribution, updated
    online as morphemes are added/removed from the cache

Design: shared alphabet, class-specific distributions
------------------------------------------------------
All morpheme classes share the same phoneme inventory (alphabet), but each
class has its own CharacterBaseDistribution so that stems and suffixes can
develop different character frequency profiles from data.

Length priors
-------------
- Stems: Poisson(λ) — λ defaults to 2, learned from observed stem lengths
- Affixes (prefix/suffix): Geometric(p_end) — p_end defaults to 0.5,
  learned from observed affix lengths
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


LengthPriorType = Literal["poisson", "geometric"]


class CharacterBaseDistribution:
    """Base distribution G₀ for a single morpheme class.

    Parameters
    ----------
    alphabet : list[str]
        Shared phoneme inventory.
    length_prior : "poisson" | "geometric"
        - "poisson": for stems (λ learnable, default 2)
        - "geometric": for affixes (p_end learnable, default 0.5)
    length_param : float
        λ for Poisson; p_end for Geometric.
    char_alpha : float
        Dirichlet concentration for character frequencies.
    """

    def __init__(
        self,
        alphabet: list[str],
        length_prior: LengthPriorType = "poisson",
        length_param: float = 2.0,
        char_alpha: float = 0.1,
    ) -> None:
        self.alphabet = alphabet
        self.length_prior = length_prior
        self.length_param = length_param
        self.char_alpha = char_alpha

        # Character counts (updated online)
        self._char_counts: dict[str, float] = {c: 0.0 for c in alphabet}
        self._total_chars: float = 0.0

    # ------------------------------------------------------------------
    # Length prior
    # ------------------------------------------------------------------

    def _log_length_prob(self, k: int) -> float:
        """Log P(length = k) under the configured prior."""
        if k <= 0:
            return float("-inf")
        if self.length_prior == "poisson":
            lam = self.length_param
            return k * math.log(lam) - lam - math.lgamma(k + 1)
        else:  # geometric
            p = self.length_param
            return math.log(1 - p) * (k - 1) + math.log(p)

    # ------------------------------------------------------------------
    # Character distribution (Dirichlet-multinomial)
    # ------------------------------------------------------------------

    def _log_char_prob(self, char: str) -> float:
        count = self._char_counts.get(char, 0.0)
        total = self._total_chars
        alpha = self.char_alpha
        V = len(self.alphabet)
        return math.log((count + alpha) / (total + alpha * V))

    # ------------------------------------------------------------------
    # Joint probability
    # ------------------------------------------------------------------

    def word_log_prob(self, word: str) -> float:
        """log P(word) = log P(|word|) + Σ log P(char_i)."""
        if not word:
            return float("-inf")
        lp = self._log_length_prob(len(word))
        for char in word:
            lp += self._log_char_prob(char)
        return lp

    def word_prob(self, word: str) -> float:
        return math.exp(self.word_log_prob(word))

    # ------------------------------------------------------------------
    # Online updates
    # ------------------------------------------------------------------

    def update_counts(self, word: str, delta: float = 1.0) -> None:
        """Increment (delta=+1) or decrement (delta=-1) character counts."""
        for char in word:
            self._char_counts[char] = self._char_counts.get(char, 0.0) + delta
        self._total_chars += delta * len(word)

    # ------------------------------------------------------------------
    # Hyperparameter learning (stub — update length_param from data)
    # ------------------------------------------------------------------

    def update_length_param(self, observed_lengths: list[int]) -> None:
        """Update length hyperparameter from a list of observed morpheme lengths.

        For Poisson: MLE is mean(lengths).
        For Geometric: MLE is 1/mean(lengths).
        """
        if not observed_lengths:
            return
        mean_len = sum(observed_lengths) / len(observed_lengths)
        if self.length_prior == "poisson":
            self.length_param = max(mean_len, 0.5)
        else:
            self.length_param = min(max(1.0 / mean_len, 0.01), 0.99)
