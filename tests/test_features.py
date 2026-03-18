"""Tests for features.py."""

import numpy as np
import pytest
from agme.features import pairwise_distance, build_distance_matrix, segment_features


def test_identity_distance():
    assert pairwise_distance("s", "s") == 0.0


def test_distance_symmetric():
    d_sz = pairwise_distance("s", "z")
    d_zs = pairwise_distance("z", "s")
    assert abs(d_sz - d_zs) < 1e-9


def test_distance_range():
    d = pairwise_distance("s", "z")
    assert 0.0 < d <= 1.0


def test_voicing_pair_close():
    # s/z differ only in voicing — should be closer than s/m
    d_sz = pairwise_distance("s", "z")
    d_sm = pairwise_distance("s", "m")
    assert d_sz < d_sm


def test_unknown_segment_returns_one():
    d = pairwise_distance("s", "∅")  # ∅ not in panphon
    assert d == 1.0


def test_build_distance_matrix():
    alpha = ["s", "z", "t"]
    mat = build_distance_matrix(alpha)
    assert ("s", "z") in mat
    assert ("s", "s") not in mat  # only x ≠ y
    assert len(mat) == 3 * 2  # 3 segments × 2 others each
