"""Morphological grammar: configurable morpheme classes + template prior.

The grammar defines:
- A set of named morpheme classes in canonical positional order
  (e.g. ["prefix", "stem", "suffix"])
- Valid word templates = non-empty subsequences of that list preserving order
- One PitmanYorCache per class
- One CharacterBaseDistribution per class
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.special import gammaln

from agme.morphology.base import CharacterBaseDistribution, LengthPriorType
from agme.morphology.pyp import PitmanYorCache


# Default length prior per class type
_DEFAULT_PRIOR: dict[str, LengthPriorType] = {
    "stem": "poisson",
    "prefix": "geometric",
    "suffix": "geometric",
}


class MorphologicalGrammar:
    """Grammar over morpheme-class sequences.

    Parameters
    ----------
    morpheme_classes : list[str]
        Ordered list of morpheme class names (positional order is canonical).
    alphabet : list[str]
        Shared phoneme inventory.
    pyp_discount : float
        PYP discount parameter (shared across classes).
    pyp_concentration : float
        PYP concentration parameter (shared across classes).
    length_priors : dict[str, LengthPriorType] | None
        Override length prior type per class name.  Defaults to "poisson"
        for "stem" and "geometric" for anything else.
    char_alpha : float
        Dirichlet concentration for character base distributions.
    template_prior : dict[tuple[str,...], float] | None
        Explicit prior over templates.  If None, uniform over all valid
        subsequence templates.
    """

    def __init__(
        self,
        morpheme_classes: list[str],
        alphabet: list[str],
        pyp_discount: float = 0.5,
        pyp_concentration: float = 1.0,
        length_priors: dict[str, LengthPriorType] | None = None,
        char_alpha: float = 0.1,
        template_prior: dict[tuple[str, ...], float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.morpheme_classes = list(morpheme_classes)
        self.alphabet = alphabet

        # Build PYP caches and base distributions
        self.caches: dict[str, PitmanYorCache] = {}
        self.base_dists: dict[str, CharacterBaseDistribution] = {}
        lp_map = length_priors or {}
        for cls in morpheme_classes:
            lp_type = lp_map.get(cls, _DEFAULT_PRIOR.get(cls, "geometric"))
            # Default length_param depends on prior type
            default_param = 2.0 if lp_type == "poisson" else 0.5
            # Pass the shared RNG so PYP table-assignment sampling is deterministic.
            self.caches[cls] = PitmanYorCache(pyp_discount, pyp_concentration, rng=rng)
            self.base_dists[cls] = CharacterBaseDistribution(
                alphabet, length_prior=lp_type, length_param=default_param, char_alpha=char_alpha
            )

        # Build template set: all non-empty subsequences of morpheme_classes
        # preserving order
        self.templates: list[tuple[str, ...]] = []
        for r in range(1, len(morpheme_classes) + 1):
            for combo in combinations(range(len(morpheme_classes)), r):
                self.templates.append(tuple(morpheme_classes[i] for i in combo))

        # Template prior (log-normalised)
        if template_prior is not None:
            raw = {t: template_prior.get(t, 1.0) for t in self.templates}
        else:
            raw = {t: 1.0 for t in self.templates}
        total = sum(raw.values())
        self._log_template_prior: dict[tuple[str, ...], float] = {
            t: __import__('math').log(v / total) for t, v in raw.items()
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def morpheme_log_prob(self, morpheme: str, cls: str) -> float:
        """log P(morpheme | cache[cls], base[cls])."""
        base_prob = self.base_dists[cls].word_prob(morpheme)
        return math.log(
            max(self.caches[cls].predictive_score(morpheme, base_prob), 1e-300)
        )

    def morpheme_log_prob_n(self, morpheme: str, cls: str, n: int) -> float:
        """Type-level log P for n tokens of *morpheme* under the PYP.

        Computes the log-numerator of the CRP probability for adding n identical
        tokens to the cache (minus the shared denominator which cancels when
        comparing options in the segmenter's softmax).

        For an *existing* UR with customer count C and table count t:
            log-numerator ≈ n × log(max(C - d*t, 0) + tiny)
            (dominant term of Σ_{i=0}^{n-1} log(C - d*t + i))

        For a *new* UR (C == 0, base probability p0):
            log-numerator = log((α + d·T) · p0)
                          + gammaln(n - d) - gammaln(1 - d)
            where T = current total tables, and the gammaln terms come from
            Π_{k=1}^{n-1} (k - d), the self-reinforcing table benefit.

        When n == 1, this reduces to morpheme_log_prob() up to the shared
        denominator log(N + α).

        The denominator Σ_{i=0}^{n-1} log(N + α + i) ≈ n × log(N + α) is
        included as a constant subtracted from both branches — it cancels in
        comparisons but ensures the scores are on the correct absolute scale.
        """
        cache = self.caches[cls]
        base_prob = self.base_dists[cls].word_prob(morpheme)
        d = cache.discount
        alpha = cache.concentration
        T = cache._total_tables
        N = cache._total_customers
        C = cache._n(morpheme)
        t = cache._t(morpheme)

        # Shared denominator: n × log(N + α)  [same for all candidate URs]
        log_denom = n * math.log(max(N + alpha, 1e-300))

        if C > 0:
            # Existing UR: approximate Σ_{i=0}^{n-1} log(C - d*t + i) ≈ n×log(C - d*t)
            log_num = n * math.log(max(C - d * t, 1e-10))
        else:
            # New UR: log((α + d·T)·p0) + Σ_{k=1}^{n-1} log(k - d)
            #       = log((α + d·T)·p0) + gammaln(n - d) - gammaln(1 - d)
            creation = math.log(max((alpha + d * T) * base_prob, 1e-300))
            table_bonus = float(gammaln(n - d) - gammaln(1 - d))
            log_num = creation + table_bonus

        return log_num - log_denom

    def template_log_prior(self, template: tuple[str, ...]) -> float:
        """log prior probability of a given template."""
        return self._log_template_prior.get(template, float("-inf"))

    # ------------------------------------------------------------------
    # Cache updates
    # ------------------------------------------------------------------

    def add_parse(self, parse: list[tuple[str, str]]) -> None:
        """Add a list of (morpheme_ur, class) pairs to the caches."""
        for ur, cls in parse:
            base_prob = self.base_dists[cls].word_prob(ur)
            self.caches[cls].add(ur, base_prob)
            self.base_dists[cls].update_counts(ur, delta=1.0)

    def remove_parse(self, parse: list[tuple[str, str]]) -> None:
        """Remove a list of (morpheme_ur, class) pairs from the caches."""
        for ur, cls in parse:
            self.caches[cls].remove(ur)
            self.base_dists[cls].update_counts(ur, delta=-1.0)

    def add_parse_n(self, parse: list[tuple[str, str]], n: int) -> None:
        """Add a parse n times (for type-level inference)."""
        for _ in range(n):
            self.add_parse(parse)

    def remove_parse_n(self, parse: list[tuple[str, str]], n: int) -> None:
        """Remove a parse n times (for type-level inference)."""
        for _ in range(n):
            self.remove_parse(parse)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def morpheme_lexicon(self) -> dict[str, dict[str, int]]:
        """Return {class: {morpheme_ur: count}} for all classes."""
        return {cls: cache.lexicon() for cls, cache in self.caches.items()}
