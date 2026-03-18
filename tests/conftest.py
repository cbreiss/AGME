"""Shared fixtures for AGME tests."""

import pytest


SMALL_ALPHABET = ["d", "g", "z", "k", "t", "s", "æ", "ɑ", "ɪ", "ʌ", "ʊ"]

PLURALS = [
    ("dɑg",  "dɑgz"),
    ("kæt",  "kæts"),
    ("bʌs",  "bʌsɪz"),
    ("bʊk",  "bʊks"),
]


@pytest.fixture
def small_alphabet():
    return list(SMALL_ALPHABET)


@pytest.fixture
def plural_pairs():
    return list(PLURALS)
