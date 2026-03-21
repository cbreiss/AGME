"""Public API: Model and ParseResult.

Entry point for users of the AGME package.  All internal components
(phonological grammar, morphological grammar, inference loop) are
assembled and wired together here.

Typical usage
-------------
    from agme import Model

    model = Model(morpheme_classes=["stem", "suffix"])
    model.fit(["dɑgz", "kæts", "bʊks"], n_sweeps=200, burn_in=50)

    result = model.parse("dɑgz")
    # result.morphemes      → ["dɑg", "z"]
    # result.ur             → "dɑgz"
    # result.mappings       → [("dɑg","dɑg",[]), ("z","z",[])]

    model.phonology.faithfulness_weights()   # inspect *MAP weights
    model.morphology.morpheme_lexicon()      # inspect learned morphemes

Component wiring
----------------
Model.fit() builds:
  - alphabet (inferred or user-supplied)
  - features.build_distance_matrix(alphabet)  → pairwise panphon distances
  - phonology.constraints.build_star_map_constraints(...)  → *MAP constraint set
  - phonology.grammar.MaxEntPhonology  → scores P(SR|UR), learns weights
  - morphology.grammar.MorphologicalGrammar  → PYP caches + template prior
  - inference.ur_proposer.URProposer  → UR candidate generator
  - inference.training.run_training(...)  → runs the Gibbs + MaxEnt loop
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agme.features import build_distance_matrix
from agme.inference.segmenter import SpanParse, sample_segmentation
from agme.inference.training import TrainingState, run_training
from agme.inference.ur_proposer import URProposer
from agme.morphology.grammar import MorphologicalGrammar
from agme.phonology.constraints import build_all_constraints
from agme.phonology.grammar import MaxEntPhonology


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """The result of parsing a single surface form.

    Attributes
    ----------
    morphemes : list[str]
        UR strings for each morpheme (left to right).
    morpheme_classes : list[str]
        Class label for each morpheme.
    segmentation : list[int]
        Boundary positions in the SR string (0-indexed start of each span).
    ur : str
        Concatenated full UR string.
    mappings : list[tuple[str, str, list[str]]]
        Per-morpheme (ur_morpheme, sr_morpheme, constraints_fired).
    log_prob : float
        log P(SR, parse) under the current model.
    """

    morphemes: list[str]
    morpheme_classes: list[str]
    segmentation: list[int]
    ur: str
    mappings: list[tuple[str, str, list[str]]]
    log_prob: float


def _spans_to_parse_result(
    spans: list[SpanParse],
    phon_grammar: MaxEntPhonology,
) -> ParseResult:
    """Convert a list of SpanParse objects (internal) to a ParseResult (public).

    For each span, collects which *MAP constraints actually fired (weight > 0
    and violation count > 0), and accumulates the log P(SR|UR) contribution.
    """
    morphemes = [sp.ur for sp in spans]
    classes = [sp.morpheme_class for sp in spans]
    segmentation = [sp.start for sp in spans]
    ur = "".join(morphemes)
    mappings = []
    log_prob = 0.0
    for sp in spans:
        # "fired" = constraints with nonzero weight that have ≥1 violation on this span
        fired = [
            repr(c)
            for c, w in zip(phon_grammar.constraints, phon_grammar.weights)
            if w > 0 and c.violations(sp.ur, sp.sr) > 0
        ]
        mappings.append((sp.ur, sp.sr, fired))
        log_prob += phon_grammar.log_prob(sp.ur, sp.sr)
    return ParseResult(
        morphemes=morphemes,
        morpheme_classes=classes,
        segmentation=segmentation,
        ur=ur,
        mappings=mappings,
        log_prob=log_prob,
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Model:
    """Joint morphological segmentation + phonological grammar model.

    Parameters
    ----------
    morpheme_classes : list[str]
        Ordered morpheme class names (e.g. ["stem", "suffix"]).
    alphabet : list[str] | None
        Phoneme inventory.  If None, inferred from training data.
    pyp_discount : float
        PYP discount parameter.
    pyp_concentration : float
        PYP concentration parameter.
    """

    def __init__(
        self,
        morpheme_classes: list[str] | None = None,
        alphabet: list[str] | None = None,
        pyp_discount: float = 0.5,
        pyp_concentration: float = 1.0,
        identity_phonology: bool = False,
        ipa_map: dict[str, str] | None = None,
        prior_scale: float = 1.0,
        epenthesis_prior: float = 1.0,
        deletion_prior: float = 1.0,
        prior_sigma: float = 1.0,
    ) -> None:
        self._morpheme_classes = morpheme_classes or ["stem", "suffix"]
        self._alphabet = alphabet
        self._pyp_discount = pyp_discount
        self._pyp_concentration = pyp_concentration
        # When True, phonology is bypassed: only faithful UR==SR mappings
        # are permitted.  Useful for testing segmentation independently.
        self._identity_phonology = identity_phonology
        # Optional phoneme-symbol → IPA mapping for non-standard encodings
        # (e.g. Klattbet: C→tʃ, W→aʊ).  Passed to build_distance_matrix so
        # panphon can compute correct P-map prior weights for each constraint.
        self._ipa_map = ipa_map
        # Global multiplier on all constraint prior_weights: raises initial
        # MaxEnt weights and widens the half-normal regularisation uniformly.
        self._prior_scale = prior_scale
        # Independent base priors for epenthesis (*MAP(∅,y)) and deletion
        # (*MAP(x,∅)) constraints, before prior_scale is applied.
        self._epenthesis_prior = epenthesis_prior
        self._deletion_prior = deletion_prior
        # Fixed-width σ for the normal regulariser (w − μ)²/(2σ²).
        # Shared across all constraints; controls how tightly weights are
        # pulled back toward their P-map targets after each MaxEnt update.
        self._prior_sigma = prior_sigma

        # Set after fit()
        self.phonology: MaxEntPhonology | None = None
        self.morphology: MorphologicalGrammar | None = None
        self._training_state: TrainingState | None = None
        self._proposer: URProposer | None = None
        self._dist_matrix: dict | None = None
        # Seed stored for reproducibility inspection and re-fitting
        self.seed: int | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        surface_forms: list[str],
        n_sweeps: int = 100,
        burn_in: int = 20,
        maxent_update_every: int = 10,
        max_morpheme_len: int = 8,
        top_k_urs: int = 8,
        print_every: int = 10,
        progress_bar: bool = False,
        seed: int | None = None,
    ) -> "Model":
        """Fit the model on a list of surface forms.

        Parameters
        ----------
        surface_forms : list[str]
            Observed surface forms (unsegmented).
        n_sweeps, burn_in, maxent_update_every, print_every : int
            Training hyperparameters.
        max_morpheme_len : int
            Maximum span length (in characters) considered for a single morpheme.
            Longer words must be split; shorter values speed up the DP.
        top_k_urs : int
            Number of UR candidates evaluated per (span, class) pair in the
            segmenter.  Higher values improve accuracy at the cost of speed.
        seed : int | None
            Random seed.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(seed)
        self.seed = seed  # stored for reproducibility

        # Build alphabet
        if self._alphabet is None:
            alphabet = sorted(set("".join(surface_forms)))
        else:
            alphabet = list(self._alphabet)

        # Build constraint set: *MAP (substitution) + *DEP (epenthesis) + *MAX (deletion)
        # All three types are trained jointly via MaxEnt (L-BFGS-B).
        dist_matrix = build_distance_matrix(alphabet, ipa_map=self._ipa_map)
        constraints = build_all_constraints(
            alphabet, dist_matrix,
            prior_scale=self._prior_scale,
            epenthesis_prior=self._epenthesis_prior,
            deletion_prior=self._deletion_prior,
        )

        # Build model components.  Pass the seeded rng so candidates_for() is
        # deterministic (the candidate cache is populated on first use per UR).
        self.phonology = MaxEntPhonology(constraints, alphabet,
                                         identity_only=self._identity_phonology,
                                         rng=rng,
                                         prior_sigma=self._prior_sigma)
        self.morphology = MorphologicalGrammar(
            self._morpheme_classes,
            alphabet,
            pyp_discount=self._pyp_discount,
            pyp_concentration=self._pyp_concentration,
            rng=rng,  # shared RNG for deterministic PYP table-assignment sampling
        )
        self._dist_matrix = dist_matrix
        self._proposer = URProposer(alphabet, dist_matrix, rng=rng)

        # Run training
        self._training_state = run_training(
            surface_forms,
            self.morphology,
            self.phonology,
            alphabet,
            n_sweeps=n_sweeps,
            burn_in=burn_in,
            maxent_update_every=maxent_update_every,
            max_morpheme_len=max_morpheme_len,
            top_k_urs=top_k_urs,
            print_every=print_every,
            progress_bar=progress_bar,
            rng=rng,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def parse(self, sr: str) -> ParseResult:
        """Return the MAP parse for a surface form."""
        self._check_fitted()
        spans = sample_segmentation(
            sr,
            self.morphology,  # type: ignore[arg-type]
            self.phonology,   # type: ignore[arg-type]
            self._proposer,   # type: ignore[arg-type]
        )
        return _spans_to_parse_result(spans, self.phonology)  # type: ignore[arg-type]

    def sample_parses(self, sr: str, n: int = 10) -> list[ParseResult]:
        """Return *n* posterior samples of parses for *sr*."""
        self._check_fitted()
        results = []
        for _ in range(n):
            spans = sample_segmentation(
                sr,
                self.morphology,  # type: ignore[arg-type]
                self.phonology,   # type: ignore[arg-type]
                self._proposer,   # type: ignore[arg-type]
            )
            results.append(_spans_to_parse_result(spans, self.phonology))  # type: ignore[arg-type]
        return results

    def predict(
        self, novel_sr: str, candidate_urs: list[str]
    ) -> list[tuple[str, float]]:
        """Score candidate URs for a novel surface form.

        Returns
        -------
        list of (ur, log_score) sorted descending.
        """
        self._check_fitted()
        scored = []
        for ur in candidate_urs:
            lp = self.phonology.log_prob(ur, novel_sr)  # type: ignore[union-attr]
            scored.append((ur, lp))
        return sorted(scored, key=lambda x: -x[1])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def print_sr_types(self) -> None:
        """Print all surface-form types observed in the posterior, by morpheme class.

        Draws from the post-burn-in parse_posterior accumulated during fit().
        Each line shows: the SR span string, its relative frequency within
        the class, and raw count.  Sorted by frequency descending.

        Use this to see what surface shapes the model assigns to each class
        (e.g. are stems always faithful? do suffixes cluster around /z/ and
        /s/?).
        """
        self._check_fitted()
        from collections import Counter, defaultdict

        # Aggregate SR frequencies per morpheme class
        sr_by_class: dict[str, Counter] = defaultdict(Counter)
        for (cls, _ur, sr), count in self._training_state.parse_posterior.items():
            sr_by_class[cls][sr] += count

        for cls in self.morphology.morpheme_classes:
            print(f"\n--- SR types ({cls}) ---")
            counter = sr_by_class.get(cls, Counter())
            if not counter:
                print("  (none observed)")
                continue
            total = sum(counter.values())
            for sr, count in counter.most_common():
                print(f"  {sr!r:20s}  freq={count / total:.3f}  (n={count:.0f})")

    def print_ur_report(self) -> None:
        """Print URs with posterior probabilities and phonological mappings.

        For each (morpheme class, UR, SR) triple observed post burn-in prints:
          - The UR→SR mapping (marking faithful pairs)
          - Posterior probability within the class
          - log P(SR | UR) under current phonological weights
          - Active constraints: *MAP / *DEP / *MAX that have weight > 0
            AND at least one violation on this (UR, SR) pair

        The active-constraint column shows the learned phonological analysis
        of each observed mapping — e.g. a suffix /z/ → [s] will show the
        *MAP(z,s) constraint, and an epenthetic [ɪ] will show *DEP(ɪ) firing
        on the UR that lacks it.

        At most 20 entries per class are shown (sorted by posterior prob).
        """
        self._check_fitted()
        from collections import defaultdict

        # Compute per-class totals for normalisation
        totals: dict[str, float] = defaultdict(float)
        for (cls, _ur, _sr), count in self._training_state.parse_posterior.items():
            totals[cls] += count

        # Build sorted entry lists per class
        by_class: dict[str, list] = defaultdict(list)
        for (cls, ur, sr), count in self._training_state.parse_posterior.items():
            prob = count / totals[cls] if totals[cls] > 0 else 0.0
            by_class[cls].append((prob, ur, sr))

        width = 65
        for cls in self.morphology.morpheme_classes:
            print(f"\n{'=' * width}")
            print(f"  Morpheme class: {cls}")
            print(f"{'=' * width}")
            print(f"  {'UR → SR':<28s}  {'post':>6}  {'logP':>8}  active constraints")
            print(f"  {'-' * (width - 2)}")

            entries = sorted(by_class.get(cls, []), reverse=True)[:20]
            if not entries:
                print("  (none observed)")
                continue

            for prob, ur, sr in entries:
                log_p = self.phonology.log_prob(ur, sr)
                # Constraints with nonzero weight that fired on this mapping
                fired = [
                    repr(c)
                    for c, w in zip(
                        self.phonology.constraints, self.phonology.weights
                    )
                    if w > 0 and c.violations(ur, sr) > 0
                ]
                mapping = f"/{ur}/ → [{sr}]" if ur != sr else f"/{ur}/ (faithful)"
                fired_str = ", ".join(fired) if fired else "—"
                print(
                    f"  {mapping:<28s}  {prob:>6.3f}  {log_p:>8.3f}  {fired_str}"
                )

    def _check_fitted(self) -> None:
        if self.phonology is None or self.morphology is None:
            raise RuntimeError("Model has not been fitted. Call model.fit() first.")
