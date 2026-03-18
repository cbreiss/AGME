"""Forward-backward DP segmenter.

Samples a morphological segmentation + UR assignment for a surface form
by marginalising over possible URs at each span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agme.morphology.grammar import MorphologicalGrammar
from agme.phonology.grammar import MaxEntPhonology
from agme.inference.ur_proposer import URProposer
from agme.utils import logsumexp


@dataclass
class SpanParse:
    """A single span in a segmentation."""
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
) -> list[SpanParse]:
    """Sample a segmentation + UR assignment for *surface* via forward-backward DP.

    Parameters
    ----------
    surface : str
        The observed surface form.
    morph_grammar : MorphologicalGrammar
        Provides PYP scores and template priors.
    phon_grammar : MaxEntPhonology
        Provides P(SR span | UR).
    proposer : URProposer
        Generates candidate URs for each span.
    max_morpheme_len : int
        Maximum character length of a single morpheme span.
    top_k_urs : int
        Number of UR candidates to consider per span.
    rng : np.random.Generator | None

    Returns
    -------
    list[SpanParse]
        Sampled segmentation: one SpanParse per morpheme in left-to-right order.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(surface)
    classes = morph_grammar.morpheme_classes
    n_classes = len(classes)

    # ------------------------------------------------------------------
    # Pre-compute span scores: span_score[(i, j, cls_idx)] = log p(span)
    # and best_ur[(i, j, cls_idx)] = MAP UR for that span
    # where log p(span) = log P(ur | cache[cls]) + log P(sr[i:j] | ur)
    # marginalised over top-k UR proposals
    # ------------------------------------------------------------------
    span_log_score: dict[tuple[int, int, int], float] = {}
    span_best_ur: dict[tuple[int, int, int], str] = {}

    for i in range(n):
        for j in range(i + 1, min(i + max_morpheme_len + 1, n + 1)):
            sr_span = surface[i:j]
            for ci, cls in enumerate(classes):
                lexicon = morph_grammar.caches[cls].lexicon()
                proposals = proposer.propose(sr_span, lexicon, top_k=top_k_urs)
                if not proposals:
                    continue
                log_scores = []
                for ur, prop_w in proposals:
                    morph_lp = morph_grammar.morpheme_log_prob(ur, cls)
                    phon_lp = phon_grammar.log_prob(ur, sr_span)
                    log_scores.append(morph_lp + phon_lp)
                # Marginal log probability (log-sum-exp over URs)
                span_log_score[(i, j, ci)] = logsumexp(log_scores)
                # MAP UR (highest joint score)
                best_idx = int(np.argmax(log_scores))
                span_best_ur[(i, j, ci)] = proposals[best_idx][0]

    # ------------------------------------------------------------------
    # Forward pass: alpha[pos][cls_idx] = log prob of best segmentation
    # of surface[0:pos] where the last morpheme had class cls_idx
    # We use a flat representation: alpha[pos] = log prob of reaching pos
    # summed over all valid class sequences (template-consistent).
    #
    # For simplicity we track: alpha[pos] = log prob of surface[0:pos]
    # under any valid template prefix, along with the last class used.
    # ------------------------------------------------------------------

    # alpha[pos] = log prob of segmenting surface[0:pos]
    # We track all (log_prob, last_cls_idx, last_span_start) for sampling
    NEG_INF = float("-inf")

    # alpha[(pos, last_cls_idx)] = log prob of being at pos with last class ci
    # Initial state: position 0, no class yet (use sentinel -1)
    alpha: dict[tuple[int, int], float] = {(0, -1): 0.0}

    for pos in range(1, n + 1):
        for j in [pos]:
            for i in range(max(0, pos - max_morpheme_len), pos):
                sr_span = surface[i:j]
                for ci, cls in enumerate(classes):
                    key = (i, j, ci)
                    if key not in span_log_score:
                        continue
                    span_lp = span_log_score[key]
                    # Check template consistency: ci must come after last_ci
                    for last_ci in range(-1, n_classes):
                        if last_ci >= ci:
                            continue  # class order must be non-decreasing
                        prev_key = (i, last_ci)
                        if prev_key not in alpha:
                            continue
                        new_lp = alpha[prev_key] + span_lp
                        cur_key = (pos, ci)
                        if cur_key not in alpha or alpha[cur_key] < new_lp:
                            alpha[cur_key] = new_lp

    # ------------------------------------------------------------------
    # Check if any valid segmentation was found
    # ------------------------------------------------------------------
    end_states = [(n, ci) for ci in range(n_classes) if (n, ci) in alpha]
    if not end_states:
        # Fallback: treat entire surface as a single stem
        stem_ci = classes.index("stem") if "stem" in classes else 0
        return [SpanParse(0, n, classes[stem_ci], surface, surface)]

    # ------------------------------------------------------------------
    # Backward sampling
    # ------------------------------------------------------------------
    # Sample end state proportional to exp(alpha)
    end_lps = [alpha[(n, ci)] for _, ci in end_states]
    end_probs = np.exp(np.array(end_lps) - logsumexp(end_lps))
    chosen_end_idx = int(rng.choice(len(end_states), p=end_probs))
    pos, last_ci = end_states[chosen_end_idx]

    result: list[SpanParse] = []
    while pos > 0:
        # Find all spans that could have ended here with class last_ci
        candidates_back = []
        for i in range(max(0, pos - max_morpheme_len), pos):
            key = (i, pos, last_ci)
            if key not in span_log_score:
                continue
            for prev_ci in range(-1, last_ci):
                prev_alpha_key = (i, prev_ci)
                if prev_alpha_key not in alpha:
                    continue
                combined = alpha[prev_alpha_key] + span_log_score[key]
                candidates_back.append((i, prev_ci, combined))

        if not candidates_back:
            break

        lps = [c[2] for c in candidates_back]
        probs = np.exp(np.array(lps) - logsumexp(lps))
        chosen = int(rng.choice(len(candidates_back), p=probs))
        i, prev_ci, _ = candidates_back[chosen]
        sr_span = surface[i:pos]
        ur = span_best_ur.get((i, pos, last_ci), sr_span)
        result.append(SpanParse(i, pos, classes[last_ci], ur, sr_span))
        pos = i
        last_ci = prev_ci

    result.reverse()
    return result
