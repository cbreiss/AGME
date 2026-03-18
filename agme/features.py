"""Phonological feature lookup via panphon.

Provides:
- segment_features(char) -> np.ndarray  (binary feature vector)
- pairwise_distance(x, y) -> float      (normalised Hamming distance [0, 1])
- build_distance_matrix(alphabet) -> dict[(x,y), float]
"""

from __future__ import annotations

import numpy as np
import panphon


_ft = panphon.FeatureTable()


def segment_features(char: str) -> np.ndarray | None:
    """Return the panphon feature vector for *char*, or None if unknown."""
    vecs = _ft.word_to_vector_list(char, numeric=True)
    if not vecs:
        return None
    return np.array(vecs[0], dtype=np.float32)


def pairwise_distance(x: str, y: str) -> float:
    """Normalised Hamming distance between the panphon feature vectors of *x* and *y*.

    Returns a float in [0, 1].  If either segment is unknown to panphon,
    returns 1.0 (maximally dissimilar) so the *MAP prior strongly disfavours
    that alternation.
    """
    if x == y:
        return 0.0
    vx = segment_features(x)
    vy = segment_features(y)
    if vx is None or vy is None or len(vx) != len(vy):
        return 1.0
    n = len(vx)
    if n == 0:
        return 1.0
    return float(np.sum(vx != vy)) / n


def build_distance_matrix(alphabet: list[str]) -> dict[tuple[str, str], float]:
    """Return a dict mapping every ordered (x, y) pair (x ≠ y) to their distance."""
    matrix: dict[tuple[str, str], float] = {}
    for x in alphabet:
        for y in alphabet:
            if x != y:
                matrix[(x, y)] = pairwise_distance(x, y)
    return matrix
