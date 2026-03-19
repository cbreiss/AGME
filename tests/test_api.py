"""Tests for the public Model API: fit, parse, sample_parses, predict,
faithfulness_weights, morpheme_lexicon, and the two introspection methods.

All tests use a small ASCII corpus and a short training run so the suite
finishes quickly.  They check structural correctness of return types and
error handling, not convergence to the correct linguistic analysis.
"""

import pytest

from agme import Model, ParseResult


# ---------------------------------------------------------------------------
# Shared corpus
# ---------------------------------------------------------------------------

SIMPLE_CORPUS = ["abz", "cdz", "efz", "ghz"]
ALPHABET = list("abcdefghijklmnopz")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_model():
    """A small fitted model shared across API tests.

    Uses identity_phonology=True so that log_prob = 0 for faithful UR==SR
    and -inf otherwise.  This makes training fast (no candidate generation
    or MaxEnt updates) and keeps API tests focused on structural correctness
    rather than phonological convergence.  The full phonological grammar is
    tested in test_training.py and test_phonology_grammar.py.
    """
    m = Model(
        morpheme_classes=["stem", "suffix"],
        alphabet=ALPHABET,
        identity_phonology=True,
    )
    m.fit(SIMPLE_CORPUS, n_sweeps=60, burn_in=15, print_every=0)
    return m


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    """fit() should return the model itself, enabling method chaining."""
    m = Model(morpheme_classes=["stem", "suffix"], alphabet=ALPHABET)
    result = m.fit(SIMPLE_CORPUS, n_sweeps=10, burn_in=3, print_every=0)
    assert result is m


def test_fit_populates_components():
    """After fit(), phonology and morphology components should be set."""
    m = Model(morpheme_classes=["stem", "suffix"], alphabet=ALPHABET)
    assert m.phonology is None
    assert m.morphology is None
    m.fit(SIMPLE_CORPUS, n_sweeps=5, burn_in=1, print_every=0)
    assert m.phonology is not None
    assert m.morphology is not None


def test_fit_with_seed_is_deterministic():
    """Two models fitted with the same seed should produce identical weights.

    The seed is now threaded through all stochastic components (Gibbs sampler,
    UR proposer, and candidate enumeration), so full-grammar models are also
    deterministic — not just identity_phonology=True ones.
    """
    import numpy as np

    m1 = Model(morpheme_classes=["stem", "suffix"], alphabet=ALPHABET)
    m2 = Model(morpheme_classes=["stem", "suffix"], alphabet=ALPHABET)
    m1.fit(SIMPLE_CORPUS, n_sweeps=20, burn_in=5, print_every=0, seed=42)
    m2.fit(SIMPLE_CORPUS, n_sweeps=20, burn_in=5, print_every=0, seed=42)
    np.testing.assert_array_almost_equal(
        m1.phonology.weights, m2.phonology.weights,
        err_msg="Same seed should produce identical weights",
    )


def test_seed_stored_on_model():
    """fit() should store the seed on model.seed for reproducibility."""
    m = Model(morpheme_classes=["stem", "suffix"], alphabet=ALPHABET)
    m.fit(SIMPLE_CORPUS, n_sweeps=5, burn_in=1, print_every=0, seed=99)
    assert m.seed == 99, f"Expected model.seed=99, got {m.seed}"


def test_seed_none_stored_on_model():
    """fit() with no seed should store None on model.seed."""
    m = Model(morpheme_classes=["stem", "suffix"], alphabet=ALPHABET)
    m.fit(SIMPLE_CORPUS, n_sweeps=5, burn_in=1, print_every=0)
    assert m.seed is None


# ---------------------------------------------------------------------------
# parse()
# ---------------------------------------------------------------------------

def test_parse_returns_parse_result(fitted_model):
    """parse() should return a ParseResult with all expected fields."""
    result = fitted_model.parse("abz")
    assert isinstance(result, ParseResult)
    assert isinstance(result.morphemes, list)
    assert isinstance(result.morpheme_classes, list)
    assert isinstance(result.segmentation, list)
    assert isinstance(result.ur, str)
    assert isinstance(result.mappings, list)
    assert isinstance(result.log_prob, float)


def test_parse_fields_consistent(fitted_model):
    """morphemes, morpheme_classes, segmentation, and mappings should all
    have the same length (one entry per morpheme)."""
    result = fitted_model.parse("abz")
    n = len(result.morphemes)
    assert len(result.morpheme_classes) == n
    assert len(result.segmentation) == n
    assert len(result.mappings) == n
    assert n >= 1, "Parse must have at least one morpheme"


def test_parse_ur_is_string(fitted_model):
    """The concatenated UR should be a non-empty string."""
    result = fitted_model.parse("abz")
    assert result.ur, "UR should not be empty"
    assert isinstance(result.ur, str)


def test_parse_log_prob_is_finite(fitted_model):
    """log_prob should be a finite float (not inf or NaN)."""
    import math
    result = fitted_model.parse("abz")
    assert math.isfinite(result.log_prob), (
        f"log_prob should be finite, got {result.log_prob}"
    )


# ---------------------------------------------------------------------------
# sample_parses()
# ---------------------------------------------------------------------------

def test_sample_parses_returns_list(fitted_model):
    """sample_parses() should return a list of n ParseResult objects."""
    parses = fitted_model.sample_parses("abz", n=5)
    assert isinstance(parses, list)
    assert len(parses) == 5
    for p in parses:
        assert isinstance(p, ParseResult)


def test_sample_parses_variation(fitted_model):
    """Different samples should not all be identical (model has uncertainty)."""
    parses = fitted_model.sample_parses("abz", n=30)
    ur_set = {p.ur for p in parses}
    # With 30 samples the model should propose at least 1 distinct analysis
    assert len(ur_set) >= 1, "sample_parses returned zero unique parses"


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_returns_sorted_descending(fitted_model):
    """predict() should return (ur, score) pairs sorted by score descending."""
    candidates = ["abz", "ab", "abzs", "z"]
    scored = fitted_model.predict("abz", candidates)
    assert len(scored) == len(candidates)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True), (
        "predict() results should be sorted by log-score descending"
    )


def test_predict_faithful_scores_highest(fitted_model):
    """The faithful candidate (UR == SR) should generally score highest."""
    scored = fitted_model.predict("abz", ["abz", "xyz", "abc"])
    top_ur, _ = scored[0]
    assert top_ur == "abz", (
        f"Expected faithful 'abz' to score highest, got {top_ur!r}"
    )


# ---------------------------------------------------------------------------
# faithfulness_weights()
# ---------------------------------------------------------------------------

def test_faithfulness_weights_is_dict(fitted_model):
    """faithfulness_weights() should return a non-empty dict."""
    fw = fitted_model.phonology.faithfulness_weights()
    assert isinstance(fw, dict)
    assert fw, "faithfulness_weights should be non-empty after training"


def test_faithfulness_weights_entry_structure(fitted_model):
    """Every entry should have 'weight', 'prior', and 'deviation' keys."""
    fw = fitted_model.phonology.faithfulness_weights()
    for constraint_name, info in fw.items():
        assert "weight" in info,    f"{constraint_name}: missing 'weight'"
        assert "prior" in info,     f"{constraint_name}: missing 'prior'"
        assert "deviation" in info, f"{constraint_name}: missing 'deviation'"
        assert isinstance(info["weight"], float)
        assert isinstance(info["prior"], float)
        assert isinstance(info["deviation"], float)


def test_faithfulness_weights_includes_insertion_deletion(fitted_model):
    """The weight dict should include *MAP(∅,y) and *MAP(x,∅) constraint keys."""
    fw = fitted_model.phonology.faithfulness_weights()
    keys = list(fw.keys())
    ins_keys = [k for k in keys if k.startswith("*MAP(∅,")]
    del_keys = [k for k in keys if k.endswith(",∅)")]
    assert ins_keys, "No *MAP(∅,y) insertion constraint found in faithfulness report"
    assert del_keys, "No *MAP(x,∅) deletion constraint found in faithfulness report"


# ---------------------------------------------------------------------------
# morpheme_lexicon()
# ---------------------------------------------------------------------------

def test_morpheme_lexicon_structure(fitted_model):
    """morpheme_lexicon() should return {class: {morpheme_ur: count}}."""
    lexicon = fitted_model.morphology.morpheme_lexicon()
    assert isinstance(lexicon, dict)
    assert "stem" in lexicon
    assert "suffix" in lexicon
    for cls, entries in lexicon.items():
        assert isinstance(entries, dict), (
            f"Expected dict for class '{cls}', got {type(entries)}"
        )
        for morpheme, count in entries.items():
            assert isinstance(morpheme, str)
            assert isinstance(count, int) and count >= 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_parse_before_fit_raises():
    """parse() before fit() should raise RuntimeError with helpful message."""
    m = Model()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.parse("abz")


def test_sample_parses_before_fit_raises():
    """sample_parses() before fit() should raise RuntimeError."""
    m = Model()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.sample_parses("abz", n=5)


def test_predict_before_fit_raises():
    """predict() before fit() should raise RuntimeError."""
    m = Model()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict("abz", ["abz"])


def test_print_sr_types_before_fit_raises():
    """print_sr_types() before fit() should raise RuntimeError."""
    m = Model()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.print_sr_types()


def test_print_ur_report_before_fit_raises():
    """print_ur_report() before fit() should raise RuntimeError."""
    m = Model()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.print_ur_report()


# ---------------------------------------------------------------------------
# ParseResult mapping format
# ---------------------------------------------------------------------------

def test_parse_result_mappings_are_triples(fitted_model):
    """Each entry in ParseResult.mappings should be a (ur, sr, list[str]) triple."""
    result = fitted_model.parse("abz")
    for ur_m, sr_m, fired in result.mappings:
        assert isinstance(ur_m, str),   "ur in mapping should be str"
        assert isinstance(sr_m, str),   "sr in mapping should be str"
        assert isinstance(fired, list), "fired constraints should be a list"
        for c_str in fired:
            assert isinstance(c_str, str), "constraint names should be strings"


def test_parse_result_sr_spans_cover_surface(fitted_model):
    """The SR spans in mappings should concatenate to the input surface form."""
    surface = "abz"
    result = fitted_model.parse(surface)
    sr_concat = "".join(sr for _, sr, _ in result.mappings)
    assert sr_concat == surface, (
        f"Concatenated SR spans '{sr_concat}' don't match surface '{surface}'"
    )
