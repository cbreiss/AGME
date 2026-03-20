"""Outer training loop: interleaved Gibbs segmentation + MaxEnt weight updates.

Training algorithm
------------------
The model alternates two kinds of updates:

1.  Gibbs sweeps over surface types (type-level inference)
    For each unique surface type (with frequency n):
      a. Remove the n copies of the current parse from all PYP caches.
      b. Re-sample the segmentation + UR assignment via forward-backward DP
         (see inference/segmenter.py), conditioning on the current grammar
         weights and the counts of all *other* types' parses.
      c. Add n copies of the new parse back into the PYP caches.
      d. Accumulate (ur, sr_span) pairs weighted by n for the next MaxEnt
         update.

    Type-level (vs. token-level) inference is an approximation that treats
    all tokens of the same surface type as having the same parse.  Under CRP
    exchangeability this is valid; it reduces the per-sweep cost from O(N)
    tokens to O(T) unique types, which is a large speedup for natural corpora
    where many types repeat.

2.  MaxEnt weight update (continuous optimisation)
    Every `maxent_update_every` sweeps, call phon_grammar.run_weight_update()
    which runs L-BFGS-B on the accumulated (ur, sr) pairs and clears the
    accumulation buffer.

After `burn_in` sweeps, UR posterior counts are accumulated for reporting.

State
-----
TrainingState.parses : dict[str, list[SpanParse]]
    Current parse (segmentation + UR assignment) for each surface type.
    Keys are unique surface forms.  Updated in-place each sweep.
TrainingState.ur_posterior : dict[str, float]
    Normalised posterior frequency of each UR type (post burn-in).
"""

from __future__ import annotations

from collections import Counter
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
    parses : dict[str, list[SpanParse]]
        Current parse (segmentation + UR assignment) for each surface type.
        Keys are unique surface forms.  Updated in-place every Gibbs sweep.
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

    parses: dict[str, list[SpanParse]] = field(default_factory=dict)
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
    max_morpheme_len: int = 8,
    top_k_urs: int = 8,
    print_every: int = 10,
    rng: np.random.Generator | None = None,
) -> TrainingState:
    """Run the joint training loop (type-level Gibbs).

    Parameters
    ----------
    surface_forms : list[str]
        Observed surface forms (unsegmented; may contain duplicates).
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
        Maximum morpheme span length in characters (EC-1).
    top_k_urs : int
        UR candidates per span (EC-3).
    print_every : int
        Print progress every this many sweeps.
    rng : np.random.Generator | None

    Returns
    -------
    TrainingState
    """
    import time as _time

    if rng is None:
        rng = np.random.default_rng()

    # Count token frequencies per surface type.
    # Type-level inference iterates over unique types; each update is weighted
    # by the type's token count.
    type_counts: Counter[str] = Counter(surface_forms)

    dist_matrix = build_distance_matrix(alphabet)
    proposer = URProposer(alphabet, dist_matrix, rng=rng)

    state = TrainingState()
    _t_start = _time.perf_counter()
    _t_last  = _t_start

    # Initialise parses: treat each surface type as a single-morpheme stem.
    # Add n copies to the PYP cache (one per token of that type).
    stem_cls = morph_grammar.morpheme_classes[0]  # first class = stem (by convention)
    for sf, n in type_counts.items():
        initial_parse = [SpanParse(0, len(sf), stem_cls, sf, sf)]
        state.parses[sf] = initial_parse
        morph_grammar.add_parse_n([(sf, stem_cls)], n)

    for sweep in range(n_sweeps):
        state.sweep_count = sweep

        # --- Gibbs sweep over unique surface types ---
        for sf, n in type_counts.items():
            old_parse = state.parses[sf]

            # Remove all n copies of the current parse from caches.
            morph_grammar.remove_parse_n(
                [(sp.ur, sp.morpheme_class) for sp in old_parse], n
            )

            # Sample a new segmentation conditioned on the remaining caches.
            new_parse = sample_segmentation(
                sf,
                morph_grammar,
                phon_grammar,
                proposer,
                max_morpheme_len=max_morpheme_len,
                top_k_urs=top_k_urs,
                rng=rng,
            )
            state.parses[sf] = new_parse

            # Add n copies of the new parse to caches.
            morph_grammar.add_parse_n(
                [(sp.ur, sp.morpheme_class) for sp in new_parse], n
            )

            # Accumulate (ur, sr) pairs weighted by token count.
            for sp in new_parse:
                phon_grammar.accumulate(sp.ur, sp.sr, count=n)

        # --- MaxEnt weight update ---
        if (sweep + 1) % maxent_update_every == 0:
            phon_grammar.run_weight_update()

        # --- Posterior accumulation (post burn-in) ---
        if sweep >= burn_in:
            for sf, parse in state.parses.items():
                n = type_counts[sf]
                for sp in parse:
                    # UR marginal posterior (weighted by token count)
                    state.ur_posterior[sp.ur] = (
                        state.ur_posterior.get(sp.ur, 0.0) + n
                    )
                    # Detailed (class, ur, sr) triple — used for introspection
                    triple = (sp.morpheme_class, sp.ur, sp.sr)
                    state.parse_posterior[triple] = (
                        state.parse_posterior.get(triple, 0.0) + n
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
            _t_now   = _time.perf_counter()
            _elapsed = _t_now - _t_last
            _total   = _t_now - _t_start
            _t_last  = _t_now
            print(
                f"Sweep {sweep + 1}/{n_sweeps} | "
                f"UR types: {n_types} | "
                f"sweep {_elapsed:.1f}s | total {_total:.1f}s | "
                f"Top weights: {top_str}",
                flush=True,
            )

    # Normalise posterior
    total = sum(state.ur_posterior.values())
    if total > 0:
        state.ur_posterior = {
            k: v / total for k, v in state.ur_posterior.items()
        }

    return state
