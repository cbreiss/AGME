"""Tests for PitmanYorCache."""

import math
import pytest
from agme.morphology.pyp import PitmanYorCache


def test_empty_cache_predictive():
    cache = PitmanYorCache(discount=0.5, concentration=1.0)
    base_prob = 0.1
    score = cache.predictive_score("dog", base_prob)
    # With no customers: score = (α + d·0) · base_prob / (0 + α) = base_prob
    assert abs(score - base_prob) < 1e-9


def test_add_increases_score():
    cache = PitmanYorCache(discount=0.5, concentration=1.0)
    base_prob = 0.05
    s0 = cache.predictive_score("dog", base_prob)
    cache.add("dog", base_prob)
    s1 = cache.predictive_score("dog", base_prob)
    assert s1 > s0


def test_add_remove_roundtrip():
    cache = PitmanYorCache(discount=0.5, concentration=1.0)
    base_prob = 0.05
    cache.add("dog", base_prob)
    cache.remove("dog")
    assert cache.total_customers == 0


def test_lexicon_after_add():
    cache = PitmanYorCache(discount=0.5, concentration=1.0)
    cache.add("dog", 0.1)
    cache.add("cat", 0.1)
    lex = cache.lexicon()
    assert "dog" in lex
    assert "cat" in lex


def test_remove_last_customer_clears_entry():
    cache = PitmanYorCache(discount=0.5, concentration=1.0)
    cache.add("dog", 0.1)
    cache.remove("dog")
    assert "dog" not in cache.lexicon()


def test_invalid_params():
    with pytest.raises(ValueError):
        PitmanYorCache(discount=1.0, concentration=1.0)
    with pytest.raises(ValueError):
        PitmanYorCache(discount=0.5, concentration=-1.0)
