"""Tests for MaxEntPhonology."""

import math
import pytest
from agme.features import build_distance_matrix
from agme.phonology.constraints import build_star_map_constraints
from agme.phonology.grammar import MaxEntPhonology


ALPHA = ["s", "z", "d", "g", "æ", "ɑ"]


@pytest.fixture
def grammar():
    dist = build_distance_matrix(ALPHA)
    constraints = build_star_map_constraints(ALPHA, dist)
    return MaxEntPhonology(constraints, ALPHA)


def test_log_prob_faithful_high(grammar):
    # Faithful mapping should have higher probability than unfaithful
    lp_faithful = grammar.log_prob("dɑg", "dɑg")
    lp_unfaithful = grammar.log_prob("dɑg", "dæg")
    assert lp_faithful > lp_unfaithful


def test_log_prob_is_log(grammar):
    # log_prob should return a non-positive value (log of a probability ≤ 1)
    lp = grammar.log_prob("sg", "zg")
    assert lp <= 0.0


def test_harmony_zero_faithful(grammar):
    # All weights > 0, faithful mapping → 0 violations → harmony = 0
    h = grammar.harmony("sg", "sg")
    assert h == 0.0


def test_violation_vector_cached(grammar):
    v1 = grammar.violation_vector("sg", "zg")
    v2 = grammar.violation_vector("sg", "zg")
    assert (v1 == v2).all()
    assert ("sg", "zg") in grammar._viol_cache


def test_accumulate_and_fit(grammar):
    # Should not raise; weights should still be non-negative after update
    grammar.accumulate("dɑgs", "dɑgz")
    grammar.accumulate("kæts", "kæts")
    grammar.fit_weights()
    assert (grammar.weights >= 0).all()
