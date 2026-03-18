"""Tests for *MAP constraints."""

import pytest
from agme.phonology.constraints import StarMapConstraint, build_star_map_constraints
from agme.features import build_distance_matrix


def test_faithful_no_violations():
    c = StarMapConstraint(x="s", y="z")
    assert c.violations("dɑgs", "dɑgs") == 0.0


def test_alternation_violation():
    c = StarMapConstraint(x="s", y="z")
    # UR: dɑgs, SR: dɑgz — s maps to z once
    assert c.violations("dɑgs", "dɑgz") == 1.0


def test_no_violation_wrong_pair():
    c = StarMapConstraint(x="k", y="g")
    assert c.violations("dɑgs", "dɑgz") == 0.0


def test_deletion_no_false_violation():
    # *MAP(s, z) should not fire when s is deleted (s→None)
    c = StarMapConstraint(x="s", y="z")
    assert c.violations("dɑgs", "dɑg") == 0.0


def test_build_constraint_set():
    alpha = ["s", "z", "d"]
    dist = build_distance_matrix(alpha)
    constraints = build_star_map_constraints(alpha, dist)
    # Should have one constraint per ordered (x,y) with x≠y
    assert len(constraints) == 3 * 2
    pairs = {(c.x, c.y) for c in constraints}
    assert ("s", "z") in pairs
    assert ("s", "s") not in pairs


def test_prior_weight_from_distance():
    alpha = ["s", "z", "m"]
    dist = build_distance_matrix(alpha)
    constraints = build_star_map_constraints(alpha, dist)
    c_sz = next(c for c in constraints if c.x == "s" and c.y == "z")
    c_sm = next(c for c in constraints if c.x == "s" and c.y == "m")
    # s/z are closer (only voicing) so prior weight should be smaller
    assert c_sz.prior_weight < c_sm.prior_weight
