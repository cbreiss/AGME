"""Public API: Model and ParseResult."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agme.features import build_distance_matrix
from agme.inference.segmenter import SpanParse, sample_segmentation
from agme.inference.training import TrainingState, run_training
from agme.inference.ur_proposer import URProposer
from agme.morphology.grammar import MorphologicalGrammar
from agme.phonology.constraints import build_star_map_constraints
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
    morphemes = [sp.ur for sp in spans]
    classes = [sp.morpheme_class for sp in spans]
    segmentation = [sp.start for sp in spans]
    ur = "".join(morphemes)
    mappings = []
    log_prob = 0.0
    for sp in spans:
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
    ) -> None:
        self._morpheme_classes = morpheme_classes or ["stem", "suffix"]
        self._alphabet = alphabet
        self._pyp_discount = pyp_discount
        self._pyp_concentration = pyp_concentration

        # Set after fit()
        self.phonology: MaxEntPhonology | None = None
        self.morphology: MorphologicalGrammar | None = None
        self._training_state: TrainingState | None = None
        self._proposer: URProposer | None = None
        self._dist_matrix: dict | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        surface_forms: list[str],
        n_sweeps: int = 100,
        burn_in: int = 20,
        maxent_update_every: int = 10,
        print_every: int = 10,
        seed: int | None = None,
    ) -> "Model":
        """Fit the model on a list of surface forms.

        Parameters
        ----------
        surface_forms : list[str]
            Observed surface forms (unsegmented).
        n_sweeps, burn_in, maxent_update_every, print_every : int
            Training hyperparameters.
        seed : int | None
            Random seed.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(seed)

        # Build alphabet
        if self._alphabet is None:
            alphabet = sorted(set("".join(surface_forms)))
        else:
            alphabet = list(self._alphabet)

        # Build constraint set
        dist_matrix = build_distance_matrix(alphabet)
        constraints = build_star_map_constraints(alphabet, dist_matrix)

        # Build model components
        self.phonology = MaxEntPhonology(constraints, alphabet)
        self.morphology = MorphologicalGrammar(
            self._morpheme_classes,
            alphabet,
            pyp_discount=self._pyp_discount,
            pyp_concentration=self._pyp_concentration,
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
            print_every=print_every,
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

    def _check_fitted(self) -> None:
        if self.phonology is None or self.morphology is None:
            raise RuntimeError("Model has not been fitted. Call model.fit() first.")
