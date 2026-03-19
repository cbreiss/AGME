"""Outer training loop: interleaved Gibbs segmentation + MaxEnt weight updates.

Training algorithm
------------------
The model alternates two kinds of updates:

1.  Gibbs sweeps over utterances (discrete sampling)
    For each surface form:
      a. Remove the current parse from all PYP caches (decrement counts).
      b. Re-sample the segmentation + UR assignment via forward-backward DP
         (see inference/segmenter.py), conditioning on the current grammar
         weights and the counts of all *other* utterances' parses.
      c. Add the new parse back into the PYP caches.
      d. Accumulate (ur, sr_span) pairs for the next MaxEnt update.

2.  MaxEnt weight update (continuous optimisation)
    Every `maxent_update_every` sweeps, call phon_grammar.run_weight_update()
    which runs L-BFGS-B on the accumulated (ur, sr) pairs and clears the
    accumulation buffer.

After `burn_in` sweeps, UR posterior counts are accumulated for reporting.

State
-----
TrainingState.parses : list[list[SpanParse]]
    Current parse (segmentation + UR assignment) for each utterance.
    Length = len(surface_forms).  Updated in-place each sweep.
TrainingState.ur_posterior : dict[str, float]
    Normalised posterior frequency of each UR type (post burn-in).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agme.features import build_distance_matrix
from agme.inference.segmenter import SpanParse, sample_segmentation
from agme.inference.ur_proposer import URProposer
from agme.morphology.grammar import MorphologicalGrammar
from agme.phonology.grammar import MaxEntPhonology


@dataclass
class TrainingState:
    """Mutable state accumulated during training.

    Fields
    ------
    parses : list[list[SpanParse]]
        Current parse (segmentation + UR assignment) for each utterance.
        Updated in-place every Gibbs sweep.
    ur_posterior : dict[str, float]
        Normalised posterior frequency of each UR string (post burn-in).
        Summed over all morpheme classes and utterances.
    parse_posterior : dict[tuple[str,str,str], float]
        Raw posterior counts of (morpheme_class, ur, sr) triples observed
        post burn-in.  Not normalised — raw counts allow per-class
        normalisation in introspection methods.  Key: (cls, ur, sr_span).
    sweep_count : int
        Number of sweeps completed so far.
    """

    parses: list[list[SpanParse]] = field(default_factory=list)
    ur_posterior: dict[str, float] = field(default_factory=dict)
    # Per-class (ur, sr) posterior counts — used by print_sr_types / print_ur_report
    parse_posterior: dict[tuple[str, str, str], float] = field(default_factory=dict)
    sweep_count: int = 0


def run_training(
    surface_forms: list[str],
    morph_grammar: MorphologicalGrammar,
    phon_grammar: MaxEntPhonology,
    alphabet: list[str],
    n_sweeps: int = 100,
    burn_in: int = 20,
    maxent_update_every: int = 10,
    max_morpheme_len: int = 10,
    top_k_urs: int = 10,
    print_every: int = 10,
    rng: np.random.Generator | None = None,
) -> TrainingState:
    """Run the joint training loop.

    Parameters
    ----------
    surface_forms : list[str]
        Observed surface forms (unsegmented).
    morph_grammar : MorphologicalGrammar
        Morphological grammar with PYP caches.
    phon_grammar : MaxEntPhonology
        MaxEnt phonological grammar.
    alphabet : list[str]
        Shared phoneme inventory.
    n_sweeps : int
        Total number of Gibbs sweeps.
    burn_in : int
        Number of sweeps before accumulating posterior statistics.
    maxent_update_every : int
        Frequency (in sweeps) of MaxEnt weight updates.
    max_morpheme_len : int
        Maximum morpheme span length in characters.
    top_k_urs : int
        UR candidates per span.
    print_every : int
        Print progress every this many sweeps.
    rng : np.random.Generator | None

    Returns
    -------
    TrainingState
    """
    if rng is None:
        rng = np.random.default_rng()

    dist_matrix = build_distance_matrix(alphabet)
    proposer = URProposer(alphabet, dist_matrix, rng=rng)

    state = TrainingState()

    # Initialise parses: treat each surface form as a single-morpheme stem
    stem_cls = morph_grammar.morpheme_classes[0]  # first class = stem (by convention)
    for sf in surface_forms:
        initial_parse = [SpanParse(0, len(sf), stem_cls, sf, sf)]
        state.parses.append(initial_parse)
        morph_grammar.add_parse([(sf, stem_cls)])

    for sweep in range(n_sweeps):
        state.sweep_count = sweep

        # --- Gibbs sweep over utterances ---
        for utt_idx, sf in enumerate(surface_forms):
            old_parse = state.parses[utt_idx]

            # Remove current parse from caches
            morph_grammar.remove_parse([(sp.ur, sp.morpheme_class) for sp in old_parse])

            # Sample new segmentation
            new_parse = sample_segmentation(
                sf,
                morph_grammar,
                phon_grammar,
                proposer,
                max_morpheme_len=max_morpheme_len,
                top_k_urs=top_k_urs,
                rng=rng,
            )
            state.parses[utt_idx] = new_parse

            # Add new parse to caches
            morph_grammar.add_parse([(sp.ur, sp.morpheme_class) for sp in new_parse])

            # Accumulate (ur, sr) pairs for MaxEnt update
            for sp in new_parse:
                phon_grammar.accumulate(sp.ur, sp.sr)

        # --- MaxEnt weight update ---
        if (sweep + 1) % maxent_update_every == 0:
            phon_grammar.run_weight_update()

        # --- Posterior accumulation (post burn-in) ---
        if sweep >= burn_in:
            for parse in state.parses:
                for sp in parse:
                    # UR marginal posterior (summed over classes and SRs)
                    state.ur_posterior[sp.ur] = (
                        state.ur_posterior.get(sp.ur, 0.0) + 1.0
                    )
                    # Detailed (class, ur, sr) triple — used for introspection
                    triple = (sp.morpheme_class, sp.ur, sp.sr)
                    state.parse_posterior[triple] = (
                        state.parse_posterior.get(triple, 0.0) + 1.0
                    )

        # --- Progress reporting ---
        if print_every > 0 and (sweep + 1) % print_every == 0:
            n_types = sum(
                len(morph_grammar.caches[cls].lexicon())
                for cls in morph_grammar.morpheme_classes
            )
            top_constraints = sorted(
                zip(phon_grammar.weights.tolist(), phon_grammar.constraints),
                key=lambda x: -x[0],
            )[:3]
            top_str = ", ".join(
                f"{repr(c)}={w:.3f}" for w, c in top_constraints
            )
            print(
                f"Sweep {sweep + 1}/{n_sweeps} | "
                f"UR types: {n_types} | "
                f"Top weights: {top_str}"
            )

    # Normalise posterior
    total = sum(state.ur_posterior.values())
    if total > 0:
        state.ur_posterior = {
            k: v / total for k, v in state.ur_posterior.items()
        }

    return state
