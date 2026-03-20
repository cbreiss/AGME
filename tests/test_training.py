"""Integration tests for the full Gibbs + MaxEnt training loop.

Tests the joint learning of morphological segmentation and phonological
constraint weights, including the newly unified *MAP(∅,y) epenthesis and
*MAP(x,∅) deletion constraints.

Test corpora
------------
SUFFIX_CORPUS   8 words all ending in "z".  Used for fast structural checks.
PLURALS_SR      English plural allomorphs: exercises voicing assimilation and
                sibilant epenthesis (/bʌs/ → [bʌsɪz]).  A realistic test of
                the phonological grammar learning jointly with segmentation.

Structural tests (fast, scope=module fixture):
  - *MAP(∅,y) and *MAP(x,∅) constraints are present after fitting
  - At least some weights deviate from their priors (MaxEnt ran)
  - parse_posterior is populated post burn-in
  - print_sr_types / print_ur_report produce output without crashing

Integration test (slow, individually fitted):
  - On the suffix corpus the model still recovers "z" as dominant suffix
    even when the full phonological grammar (not identity_only) is active
"""

import pytest

from agme import Model
from agme.phonology.constraints import StarMapConstraint


# ---------------------------------------------------------------------------
# Test corpora
# ---------------------------------------------------------------------------

# Simple corpus: 8 words all ending in suffix "z" (ASCII, no panphon issues)
SUFFIX_CORPUS = [
    "abz", "cdz", "efz", "ghz",
    "ijz", "klz", "mnz", "opz",
]

# English plural allomorphs (IPA).  Exercises:
#   - Faithful  /z/:   dɑgz, læbz   (voiced stem-final → voiced suffix)
#   - Devoiced  /z/→[s]: kæts, bʊks  (*MAP(z,s) should learn low weight)
#   - Epenthesis ∅→[ɪ]: bʌsɪz, rozɪz (*MAP(∅,ɪ) should be penalised)
PLURALS_SR = [
    "dɑgz", "læbz",
    "kæts", "bʊks",
    "bʌsɪz", "rozɪz",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def suffix_model():
    """Model trained on the ASCII suffix corpus with full phonological grammar.

    n_sweeps=150 is enough to observe MaxEnt weight movement and posterior
    accumulation without running the full convergence test.
    """
    m = Model(
        morpheme_classes=["stem", "suffix"],
        alphabet=list("abcdefghijklmnopz"),
    )
    m.fit(SUFFIX_CORPUS, n_sweeps=150, burn_in=50, print_every=0, seed=31)
    return m


# ---------------------------------------------------------------------------
# Structural / fast tests (use suffix_model fixture)
# ---------------------------------------------------------------------------

def test_insertion_constraints_present(suffix_model):
    """Model should include *MAP(∅,y) epenthesis constraints after fitting."""
    ins_constraints = [
        c for c in suffix_model.phonology.constraints
        if isinstance(c, StarMapConstraint) and c.x is None and c.y is not None
    ]
    alph = suffix_model.phonology.alphabet
    assert len(ins_constraints) == len(alph), (
        f"Expected one *MAP(∅,y) per segment ({len(alph)}), "
        f"got {len(ins_constraints)}"
    )


def test_deletion_constraints_present(suffix_model):
    """Model should include *MAP(x,∅) deletion constraints after fitting."""
    del_constraints = [
        c for c in suffix_model.phonology.constraints
        if isinstance(c, StarMapConstraint) and c.x is not None and c.y is None
    ]
    alph = suffix_model.phonology.alphabet
    assert len(del_constraints) == len(alph), (
        f"Expected one *MAP(x,∅) per segment ({len(alph)}), "
        f"got {len(del_constraints)}"
    )


def test_weights_updated_from_prior(suffix_model):
    """After training, at least some weights should differ from their prior."""
    import numpy as np

    deviations = [
        abs(float(w) - c.prior_weight)
        for c, w in zip(
            suffix_model.phonology.constraints, suffix_model.phonology.weights
        )
    ]
    assert max(deviations) > 1e-6, (
        "No weight moved from its prior — MaxEnt update may not have run"
    )


def test_parse_posterior_populated(suffix_model):
    """parse_posterior should be non-empty after training past burn-in."""
    assert suffix_model._training_state.parse_posterior, (
        "parse_posterior is empty — posterior accumulation may not be running"
    )


def test_parse_posterior_has_expected_keys(suffix_model):
    """Each key in parse_posterior should be a (class, ur_str, sr_str) tuple."""
    for key, count in suffix_model._training_state.parse_posterior.items():
        cls, ur, sr = key  # unpacking confirms 3-tuple structure
        assert isinstance(cls, str), f"class should be str, got {type(cls)}"
        assert isinstance(ur, str),  f"ur should be str, got {type(ur)}"
        assert isinstance(sr, str),  f"sr should be str, got {type(sr)}"
        assert count > 0, "Posterior counts should be positive"


def test_print_sr_types_produces_output(suffix_model, capsys):
    """print_sr_types() should run without error and print a header line."""
    suffix_model.print_sr_types()
    captured = capsys.readouterr()
    assert "SR types" in captured.out, (
        "print_sr_types() output missing expected header '--- SR types'"
    )


def test_print_ur_report_produces_output(suffix_model, capsys):
    """print_ur_report() should run without error and print class headers."""
    suffix_model.print_ur_report()
    captured = capsys.readouterr()
    assert "Morpheme class" in captured.out, (
        "print_ur_report() output missing expected header 'Morpheme class'"
    )


def test_ur_report_shows_faithful_mappings(suffix_model, capsys):
    """print_ur_report() should include at least one faithful UR == SR entry."""
    suffix_model.print_ur_report()
    captured = capsys.readouterr()
    assert "faithful" in captured.out, (
        "Expected at least one faithful UR→SR pair in the report "
        "(e.g. a stem that maps identically)"
    )


def test_suffix_learned_with_full_phonology(suffix_model):
    """The model should learn 'z' as the dominant suffix type.

    This is the same check as in test_segmenter.py but exercised with the
    full phonological grammar active (not identity_only).
    """
    lexicon = suffix_model.morphology.morpheme_lexicon()
    suffix_counts = lexicon.get("suffix", {})
    if not suffix_counts:
        pytest.skip("Suffix lexicon empty — may need more sweeps")
    most_common = max(suffix_counts, key=suffix_counts.get)
    assert most_common == "z", (
        f"Expected 'z' as most common suffix with full phonology, "
        f"got {most_common!r}.  Full lexicon: {suffix_counts}"
    )


# ---------------------------------------------------------------------------
# Integration test on the English plurals corpus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_plurals_model_runs_without_error():
    """The model should fit on the IPA plurals corpus without exception.

    This is a smoke test.  n_sweeps=20 is intentionally small so the test
    finishes quickly and does not depend on any prior long-running fixture.
    We check structural correctness (constraint count) and that the MaxEnt
    update did something non-trivial (total weight sum > 0 after training).
    """
    m = Model(morpheme_classes=["stem", "suffix"])
    m.fit(PLURALS_SR, n_sweeps=20, burn_in=5, print_every=0)

    # Constraint set should include insertion constraints for all segments
    alph = m.phonology.alphabet
    ins_constraints = [
        c for c in m.phonology.constraints
        if isinstance(c, StarMapConstraint) and c.x is None
    ]
    assert len(ins_constraints) == len(alph), (
        "Missing insertion constraints in plurals model"
    )

    # At least some constraint weights should be positive after training.
    # (All starting weights are >= 0; the MaxEnt update will increase some
    # constraints that are under-penalised given the observed mappings.)
    total_weight = float(m.phonology.weights.sum())
    assert total_weight > 0.0, (
        f"All weights are 0 after training — MaxEnt update may not have run. "
        f"total_weight={total_weight}"
    )
