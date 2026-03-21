"""Boundary-level evaluation for AGME morphological segmentation on Brent corpus.

The gold standard (words_gold.txt) marks two kinds of internal boundaries:
  |  true morpheme boundary (/ in original S_Model.txt)
  +  accidental match (> in original) — string-identical to a suffix but
     NOT a real morphological boundary

This module computes:
  - Boundary precision / recall / F1 (treating | as positive, + as negative)
  - Suffix-type precision / recall on the morpheme-lexicon level
  - A breakdown separating true-positive (+) and false-positive discoveries

Evaluation protocol
-------------------
After training an AGME model on surface forms from words_train.txt, call
``evaluate_model(model, surface_forms, gold_segmentations)`` to get a
full EvalResult.

The model is asked to produce a MAP parse for each word; the predicted
boundary positions are compared to the gold boundaries.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> list[str]:
    """Read one-item-per-line text file, stripping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def gold_boundary_positions(segmented: str) -> tuple[set[int], set[int]]:
    """Extract sets of true and accidental boundary positions from a gold string.

    Gold strings use '|' for true boundaries and '+' for accidental ones.
    A position is defined as the index (0-based) of the phoneme AFTER which
    the boundary falls — i.e., end position of the left morpheme.

    Example: 'kIt|i' → surface='kIti', true_boundaries={2}, acc_boundaries={}
             'w^nt|s' → surface='w^nts', true_boundaries={3}, acc_boundaries={}
             'W>t' → surface='Wt', true_boundaries={}, acc_boundaries={1}
    """
    true_bs: set[int] = set()
    acc_bs: set[int] = set()
    pos = 0  # current position in surface string
    for ch in segmented:
        if ch == "|":
            if pos > 0:
                true_bs.add(pos)
        elif ch == "+":
            if pos > 0:
                acc_bs.add(pos)
        else:
            pos += 1
    return true_bs, acc_bs


def parse_to_boundary_positions(morphemes: list[str]) -> set[int]:
    """Convert a list of morpheme strings to a set of boundary positions.

    Boundary positions are defined as the cumulative length of each prefix
    of the morpheme list (except the last one).

    Example: ['kIt', 'i'] → {3}  (boundary after position 3, i.e. 'kIt' has 3 chars)
             ['w^nt', 's'] → {4}
             ['kIti'] → set()  (no internal boundary)
    """
    positions: set[int] = set()
    cumlen = 0
    for m in morphemes[:-1]:  # all but last
        cumlen += len(m)
        positions.add(cumlen)
    return positions


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Boundary-level evaluation of AGME segmentation vs gold standard.

    Attributes
    ----------
    n_words : int
        Total words evaluated.
    tp : int
        Predicted boundaries that match a true gold boundary (|).
    fp_true : int
        Predicted boundaries at neither-| nor + positions (completely wrong).
    fp_acc : int
        Predicted boundaries that match an accidental position (+) — the
        model found a spurious pattern that is string-identical to a real
        suffix but is not actually morphological.
    fn : int
        Gold true boundaries (|) that the model missed.
    tn : int
        Positions the model correctly left unsegmented.

    True-positive rate:  recall = tp / (tp + fn)
    Precision:           precision = tp / (tp + fp_true + fp_acc)
    F1:                  2 * P * R / (P + R)
    Accidental rate:     fp_acc / (fp_true + fp_acc)  — fraction of FPs that
                         are at accidental positions (measures how well the
                         model avoids spurious pattern matching)
    """
    n_words: int = 0
    tp: int = 0
    fp_true: int = 0   # predicted, not in gold at all
    fp_acc: int = 0    # predicted, at accidental (+) position
    fn: int = 0        # gold boundary missed
    tn: int = 0        # correctly unsegmented position

    # Detailed per-word results
    details: list[dict] = field(default_factory=list, repr=False)

    @property
    def fp(self) -> int:
        return self.fp_true + self.fp_acc

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accidental_rate(self) -> float:
        """Fraction of false positives that land on accidental positions (+)."""
        return self.fp_acc / self.fp if self.fp > 0 else 0.0

    def __str__(self) -> str:
        lines = [
            f"=== Boundary-level evaluation (n={self.n_words} words) ===",
            f"  Precision: {self.precision:.3f}  ({self.tp} TP / {self.tp+self.fp} predicted)",
            f"  Recall:    {self.recall:.3f}  ({self.tp} TP / {self.tp+self.fn} gold boundaries)",
            f"  F1:        {self.f1:.3f}",
            f"",
            f"  TP (correct boundaries):         {self.tp}",
            f"  FP total:                        {self.fp}",
            f"    FP at accidental (+) positions: {self.fp_acc}  (accidental rate={self.accidental_rate:.2%})",
            f"    FP at random positions:         {self.fp_true}",
            f"  FN (missed boundaries):          {self.fn}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    surface_forms: list[str],
    gold_segmentations: list[str],
    verbose: bool = False,
    progress_bar: bool = False,
) -> EvalResult:
    """Evaluate AGME segmentation against gold-standard boundaries.

    Parameters
    ----------
    model : agme.Model
        A fitted AGME model.
    surface_forms : list[str]
        Surface forms (one per word, same order as gold_segmentations).
    gold_segmentations : list[str]
        Gold segmentation strings with | (true) and + (accidental) markers.
    verbose : bool
        If True, print per-word diagnostics for mismatch cases.
    progress_bar : bool
        If True, show a tqdm progress bar over words.

    Returns
    -------
    EvalResult
    """
    try:
        from tqdm.auto import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    result = EvalResult()

    pairs = zip(surface_forms, gold_segmentations)
    if progress_bar and _has_tqdm:
        pairs = tqdm(list(pairs), desc="Evaluating", unit="word")

    for surface, gold_seg in pairs:
        result.n_words += 1

        # Gold boundaries
        true_bs, acc_bs = gold_boundary_positions(gold_seg)
        n_surface = len(surface)
        all_possible_positions = set(range(1, n_surface))  # positions 1..n-1

        # Predicted boundaries from AGME MAP parse
        try:
            parse = model.parse(surface)
            pred_bs = parse_to_boundary_positions(parse.morphemes)
        except Exception:
            pred_bs = set()

        # Classify each predicted boundary
        for pos in pred_bs:
            if pos in true_bs:
                result.tp += 1
            elif pos in acc_bs:
                result.fp_acc += 1
            else:
                result.fp_true += 1

        # Count missed true boundaries
        for pos in true_bs:
            if pos not in pred_bs:
                result.fn += 1

        # Count true negatives (positions with no boundary, correctly)
        non_boundary_positions = all_possible_positions - true_bs - acc_bs
        for pos in non_boundary_positions:
            if pos not in pred_bs:
                result.tn += 1

        if verbose and (pred_bs != true_bs):
            detail = {
                "surface": surface,
                "gold": gold_seg,
                "pred_morphemes": parse.morphemes if "parse" in dir() else [],
                "pred_bs": sorted(pred_bs),
                "true_bs": sorted(true_bs),
                "acc_bs": sorted(acc_bs),
            }
            result.details.append(detail)

    return result


# ---------------------------------------------------------------------------
# Suffix-type level evaluation
# ---------------------------------------------------------------------------

class SuffixEval(NamedTuple):
    """Suffix-type level evaluation results."""
    true_suffix_types: dict[str, int]   # suffix_str → count among gold true boundaries
    found_suffix_types: dict[str, int]  # suffix_str → count in AGME posterior lexicon
    tp_types: set[str]                  # suffix types found that are real
    fp_types: set[str]                  # suffix types found but not real
    fn_types: set[str]                  # real suffix types not found


def evaluate_suffix_types(
    model,
    gold_segmentations: list[str],
    surface_forms: list[str],
) -> SuffixEval:
    """Compare learned suffix types to gold-standard suffix inventory.

    Extracts the gold suffix inventory from gold_segmentations (the string
    after the last | boundary in each word).  Compares to the suffix class
    lexicon in the trained AGME model.

    Returns a SuffixEval with type-level precision/recall information.
    """
    # Gold suffix types: the phoneme string following each | boundary
    gold_suffix_counts: dict[str, int] = defaultdict(int)
    for surface, gold_seg in zip(surface_forms, gold_segmentations):
        # Rebuild the segments split by | boundaries
        parts = gold_seg.split("|")
        if len(parts) > 1:
            # Last part (after final |) is the suffix
            suffix = parts[-1].replace("+", "")  # strip any accidental markers
            if suffix:
                gold_suffix_counts[suffix] += 1

    # Model suffix types from the posterior lexicon
    lexicon = model.morphology.morpheme_lexicon()
    model_suffix_counts = lexicon.get("suffix", {})

    gold_types = set(gold_suffix_counts.keys())
    model_types = set(model_suffix_counts.keys())

    return SuffixEval(
        true_suffix_types=dict(gold_suffix_counts),
        found_suffix_types=dict(model_suffix_counts),
        tp_types=gold_types & model_types,
        fp_types=model_types - gold_types,
        fn_types=gold_types - model_types,
    )


# ---------------------------------------------------------------------------
# CLI entry point (for quick command-line evaluation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import collections
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agme import Model

    print("Loading corpus ...")
    surfaces = load_corpus("data/words_train.txt")
    golds = load_corpus("data/words_gold.txt")

    # Use top-500 types (manageable size for a quick CLI smoke test)
    type_counts = collections.Counter(surfaces)
    top_types = {t for t, _ in type_counts.most_common(500)}
    pairs = [(s, g) for s, g in zip(surfaces, golds) if s in top_types]
    surfaces = [p[0] for p in pairs]
    golds    = [p[1] for p in pairs]

    print(f"Training AGME on {len(surfaces)} tokens ({len(set(surfaces))} types, identity_only, 50 sweeps) ...")
    m = Model(morpheme_classes=["stem", "suffix"], identity_phonology=True)
    m.fit(surfaces, n_sweeps=50, burn_in=10, print_every=10, seed=42)

    print("\nEvaluating ...")
    result = evaluate_model(m, surfaces, golds)
    print(result)

    se = evaluate_suffix_types(m, golds, surfaces)
    print(f"\n=== Suffix-type evaluation ===")
    print(f"  Gold suffix types ({len(se.true_suffix_types)}): {sorted(se.true_suffix_types, key=se.true_suffix_types.get, reverse=True)[:15]}")
    print(f"  Model suffix types ({len(se.found_suffix_types)}): {sorted(se.found_suffix_types, key=se.found_suffix_types.get, reverse=True)[:15]}")
    print(f"  TP (correct) types: {sorted(se.tp_types)}")
    print(f"  FP (spurious) types: {sorted(se.fp_types)}")
    print(f"  FN (missed) types: {sorted(se.fn_types)}")
