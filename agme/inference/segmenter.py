"""Forward-backward DP segmenter.

Samples a morphological segmentation + UR assignment for a surface form
by exactly marginalising over:
  - segmentation boundaries (where morphemes begin and end)
  - morpheme class assignments (which class each span belongs to)
  - underlying representations (which UR each span corresponds to)

Algorithm
---------
1.  Pre-compute, for every (span, class, UR) triple:
      joint_log_score = log P(ur | PYP_cache[class]) + log P(sr_span | ur)
    Marginalise over UR proposals via log-sum-exp to get span_log_score[(i,j,ci)].

2.  Forward (sum-product) pass:
    State = (position_in_surface, frozenset_of_class_indices_used_so_far).
    Enforces that class indices are always added in increasing order
    (maintaining the canonical left→right positional ordering of morpheme classes).
    Computes alpha_at[pos][used_classes] = log Σ P(all parses of surface[0:pos]
                                                   using exactly used_classes).

3.  At position n: weight each end state by the template prior P(template).

4.  Backward sampling:
    - Sample end state proportional to alpha[n][used] + log P(template(used)).
    - Since class ordering is enforced, max(used_classes) is always the last
      class added; peel it off at each backward step.
    - For each chosen span, sample the UR from the conditional P(ur | span_score).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agme.inference.ur_proposer import URProposer
from agme.morphology.grammar import MorphologicalGrammar
from agme.phonology.grammar import MaxEntPhonology
from agme.utils import logsumexp


@dataclass
class SpanParse:
    """A single morpheme span within a segmentation.

    Attributes
    ----------
    start, end : int
        Character indices into the surface form (end is exclusive).
    morpheme_class : str
        Class label (e.g. "stem", "suffix").
    ur : str
        The sampled underlying representation for this morpheme.
    sr : str
        The observed surface sub-string surface[start:end].
    """
    start: int
    end: int
    morpheme_class: str
    ur: str
    sr: str

    @property
    def length(self) -> int:
        return self.end - self.start


def sample_segmentation(
    surface: str,
    morph_grammar: MorphologicalGrammar,
    phon_grammar: MaxEntPhonology,
    proposer: URProposer,
    max_morpheme_len: int = 10,
    top_k_urs: int = 10,
    rng: np.random.Generator | None = None,
    token_count: int = 1,
) -> list[SpanParse]:
    """Sample a segmentation + UR assignment for *surface* via forward-backward DP.

    Parameters
    ----------
    surface : str
        The observed surface form (single utterance / word).
    morph_grammar : MorphologicalGrammar
        Provides PYP predictive scores and template priors.
    phon_grammar : MaxEntPhonology
        Provides log P(SR span | UR).
    proposer : URProposer
        Generates weighted candidate UR proposals for each span.
    max_morpheme_len : int
        Maximum number of characters in a single morpheme span.
    top_k_urs : int
        Number of UR candidates to evaluate per (span, class) pair.
    rng : np.random.Generator | None
        Random number generator; created fresh if None.
    token_count : int
        Number of tokens this surface type represents (for type-level inference).
        The phonological cost log P(SR|UR) is multiplied by token_count because
        each of the n tokens is independently generated from the phonological
        grammar, making the total phonological contribution n × log P(SR|UR).
        The PYP term (morph_lp) is kept unscaled — it is the predictive
        probability for one new draw, evaluated at the current cache state.
        Default 1 (token-level behaviour).

    Returns
    -------
    list[SpanParse]
        A sampled segmentation in left-to-right order.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(surface)
    classes = morph_grammar.morpheme_classes
    n_classes = len(classes)

    # ------------------------------------------------------------------
    # Step 1: Pre-compute span scores and UR proposal lists.
    #
    # span_log_score[(i, j, ci)] = log Σ_ur P(ur | cache[ci]) · P(sr[i:j] | ur)
    #   This is the marginal log probability of span [i,j) as class ci,
    #   summed over the top-k UR proposals (importance-sampled approximation).
    #
    # span_ur_proposals[(i, j, ci)] = [(ur, log_joint_score), ...]
    #   Stored so the backward pass can sample the UR from its conditional.
    # ------------------------------------------------------------------
    span_log_score: dict[tuple[int, int, int], float] = {}
    span_ur_proposals: dict[tuple[int, int, int], list[tuple[str, float]]] = {}

    for i in range(n):
        for j in range(i + 1, min(i + max_morpheme_len + 1, n + 1)):
            sr_span = surface[i:j]
            for ci, cls in enumerate(classes):
                lexicon = morph_grammar.caches[cls].lexicon()
                proposals = proposer.propose(sr_span, lexicon, top_k=top_k_urs)
                if not proposals:
                    continue

                log_scores: list[float] = []
                for ur, _proposal_weight in proposals:
                    morph_lp = morph_grammar.morpheme_log_prob(ur, cls)
                    # Use unnormalized phonological score -H(ur, sr_span) to
                    # avoid partition-function artifacts: log_prob(ur, sr)
                    # includes -log Z(ur) which varies per-UR (approximation
                    # artifact from finite candidate sets) and can make
                    # unfaithful URs appear phonologically preferred when
                    # multiplied by large token counts.  Using -H ensures
                    # faithful URs (H=0) always score at least as well.
                    phon_lp = phon_grammar.unnorm_log_prob(ur, sr_span)
                    log_scores.append(morph_lp + token_count * phon_lp)

                key = (i, j, ci)
                # Marginal over URs (for the forward pass)
                span_log_score[key] = logsumexp(log_scores)
                # Per-UR scores (for UR sampling in the backward pass)
                span_ur_proposals[key] = [
                    (proposals[k][0], log_scores[k]) for k in range(len(proposals))
                ]

    # ------------------------------------------------------------------
    # Step 2: Forward (sum-product) pass.
    #
    # DP state: (position, frozenset of class indices used so far).
    # alpha_at[pos][used_classes] = log-sum over all parse prefixes of
    #   surface[0:pos] that use exactly the classes in used_classes,
    #   in canonical order (enforced by requiring ci > max(prev_used)).
    #
    # Initialise at position 0 with the empty class set.
    # ------------------------------------------------------------------
    alpha_at: dict[int, dict[frozenset, float]] = {0: {frozenset(): 0.0}}

    for pos in range(1, n + 1):
        alpha_at[pos] = {}
        for i in range(max(0, pos - max_morpheme_len), pos):
            if not alpha_at.get(i):
                continue
            for ci in range(n_classes):
                key = (i, pos, ci)
                if key not in span_log_score:
                    continue
                span_lp = span_log_score[key]
                for prev_used, prev_lp in alpha_at[i].items():
                    # Enforce: ci must not already be used, and must be the
                    # largest class index so far (canonical left→right order).
                    if ci in prev_used:
                        continue
                    if prev_used and max(prev_used) >= ci:
                        continue
                    new_used = prev_used | frozenset({ci})
                    new_lp = prev_lp + span_lp
                    cur = alpha_at[pos].get(new_used, float("-inf"))
                    alpha_at[pos][new_used] = logsumexp([cur, new_lp])

    # ------------------------------------------------------------------
    # Step 3: Weight end states by template prior.
    #
    # The template prior P(template) favours certain morpheme-class
    # sequences over others.  It is applied here rather than in the
    # forward pass because it is a global property of the full parse,
    # not factorisable per span.
    # ------------------------------------------------------------------
    end_alpha = alpha_at.get(n, {})
    if not end_alpha:
        # Fallback: no valid segmentation found — treat the whole surface
        # as a single stem.
        stem_ci = classes.index("stem") if "stem" in classes else 0
        return [SpanParse(0, n, classes[stem_ci], surface, surface)]

    # Build (used_classes, final_log_score) pairs incorporating the template prior
    end_items: list[tuple[frozenset, float]] = []
    for used_classes, lp in end_alpha.items():
        template = tuple(classes[ci] for ci in sorted(used_classes))
        prior_lp = morph_grammar.template_log_prior(template)
        end_items.append((used_classes, lp + prior_lp))

    # ------------------------------------------------------------------
    # Step 4: Backward sampling.
    #
    # (a) Sample the end state (which class set covers the whole surface).
    # (b) Peel off spans right-to-left.  Because class ordering is enforced,
    #     max(cur_used) is always the class of the rightmost span.
    # (c) For each chosen span, sample the UR from P(ur | span_score).
    # ------------------------------------------------------------------

    # (a) Sample end state
    end_lps = [lp for _, lp in end_items]
    end_probs = np.exp(np.array(end_lps) - logsumexp(end_lps))
    chosen_idx = int(rng.choice(len(end_items), p=end_probs))
    cur_used, _ = end_items[chosen_idx]

    pos = n
    result: list[SpanParse] = []

    while pos > 0 and cur_used:
        # (b) The rightmost class is always max(cur_used) due to ordering
        ci = max(cur_used)
        prev_used = frozenset(cur_used - {ci})

        # Collect all spans (i → pos, ci) that are compatible with the
        # predecessor state (i, prev_used)
        candidates_back: list[tuple[int, float]] = []
        for i in range(max(0, pos - max_morpheme_len), pos):
            key = (i, pos, ci)
            if key not in span_log_score:
                continue
            prev_lp = alpha_at.get(i, {}).get(prev_used, float("-inf"))
            if not math.isfinite(prev_lp):
                continue
            candidates_back.append((i, prev_lp + span_log_score[key]))

        if not candidates_back:
            break  # should not happen in a well-formed DP, but guard against it

        # Sample span start position
        back_lps = [c[1] for c in candidates_back]
        back_probs = np.exp(np.array(back_lps) - logsumexp(back_lps))
        chosen = int(rng.choice(len(candidates_back), p=back_probs))
        i_chosen = candidates_back[chosen][0]

        # (c) Sample UR for this span from its conditional distribution
        key = (i_chosen, pos, ci)
        ur_proposals = span_ur_proposals[key]
        ur_lps = [lp for _, lp in ur_proposals]
        ur_probs = np.exp(np.array(ur_lps) - logsumexp(ur_lps))
        ur_idx = int(rng.choice(len(ur_proposals), p=ur_probs))
        chosen_ur = ur_proposals[ur_idx][0]

        sr_span = surface[i_chosen:pos]
        result.append(SpanParse(i_chosen, pos, classes[ci], chosen_ur, sr_span))

        pos = i_chosen
        cur_used = prev_used

    result.reverse()
    return result
