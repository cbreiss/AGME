"""Phonological feature lookup via panphon.

Provides:
- segment_features(ipa) -> np.ndarray  (binary or averaged feature vector)
- pairwise_distance(x, y, ipa_map) -> float  (normalised distance [0, 1])
- build_distance_matrix(alphabet, ipa_map) -> dict[(x,y), float]

ipa_map support
---------------
Corpora that use custom phoneme encodings (e.g. Klattbet: C=tʃ, W=aʊ) can
pass an ``ipa_map`` dict that translates each symbol to its IPA string before
panphon lookup.  Multi-segment IPA strings (affricates like "tʃ", diphthongs
like "aʊ") are handled by averaging the panphon feature vectors of the
component segments — treating the whole as a single phoneme whose featural
profile is the centroid of its parts.  For distance purposes, the averaged
vector is compared with abs-difference > 0.5 thresholding so that features
on which both components agree count as "same".

Unknown symbols fall back to 1.0 (maximum distance) as before.
"""

from __future__ import annotations

import numpy as np
import panphon


_ft = panphon.FeatureTable()

# Threshold used when comparing averaged feature vectors.
# A feature "matches" if the two averaged values are within this distance.
# For binary (+1/-1) features, 0.5 means both components must agree
# with the other segment's value.
_AVG_THRESHOLD: float = 0.5


def segment_features(ipa: str) -> np.ndarray | None:
    """Return a panphon feature vector for *ipa*, or None if unknown.

    For multi-character IPA strings (affricates, diphthongs), the feature
    vectors of the component segments are averaged into a single vector so
    the composite is treated as one phoneme in distance calculations.
    """
    vecs = _ft.word_to_vector_list(ipa, numeric=True)
    if not vecs:
        return None
    if len(vecs) == 1:
        return np.array(vecs[0], dtype=np.float32)
    # TODO: averaging is a rough approximation for compound phonemes (affricates,
    # diphthongs).  A principled fix is to let the user supply a custom feature
    # matrix (e.g. Hayes 2009 features, or a sparse matrix with composite entries)
    # and pass it in place of panphon.  For now we average the component vectors.
    return np.mean([np.array(v, dtype=np.float32) for v in vecs], axis=0)


def pairwise_distance(
    x: str,
    y: str,
    ipa_map: dict[str, str] | None = None,
) -> float:
    """Normalised distance between the panphon feature vectors of *x* and *y*.

    Parameters
    ----------
    x, y : str
        Phoneme symbols (may be IPA or custom encoding).
    ipa_map : dict[str, str] | None
        Optional mapping from custom symbol → IPA string.  Applied before
        panphon lookup so that e.g. Klattbet 'C' → 'tʃ' gets proper features.

    Returns
    -------
    float in [0, 1].  Returns 1.0 if either symbol is unknown to panphon.
    """
    if x == y:
        return 0.0
    x_ipa = ipa_map[x] if (ipa_map and x in ipa_map) else x
    y_ipa = ipa_map[y] if (ipa_map and y in ipa_map) else y
    vx = segment_features(x_ipa)
    vy = segment_features(y_ipa)
    if vx is None or vy is None or len(vx) != len(vy):
        return 1.0
    n = len(vx)
    if n == 0:
        return 1.0
    # Use abs-difference with threshold so averaged vectors work correctly.
    # For ordinary single-segment pairs this is equivalent to Hamming distance.
    return float(np.sum(np.abs(vx - vy) > _AVG_THRESHOLD)) / n


def build_distance_matrix(
    alphabet: list[str],
    ipa_map: dict[str, str] | None = None,
) -> dict[tuple[str, str], float]:
    """Return a dict mapping every ordered (x, y) pair (x ≠ y) to their distance.

    Parameters
    ----------
    alphabet : list[str]
        Phoneme inventory (may use custom encoding).
    ipa_map : dict[str, str] | None
        Optional custom-symbol → IPA mapping (see pairwise_distance).
    """
    matrix: dict[tuple[str, str], float] = {}
    for x in alphabet:
        for y in alphabet:
            if x != y:
                matrix[(x, y)] = pairwise_distance(x, y, ipa_map)
    return matrix
