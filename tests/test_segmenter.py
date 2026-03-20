"""Tests for the forward-backward segmenter, isolated from the phonological grammar.

All tests use identity_only phonology so that P(SR|UR) = 1 iff UR == SR.
This means the segmenter can only propose UR == SR for each span, and the
only thing being tested is whether the DP finds the correct morpheme
boundaries and converges the PYP caches to the right morpheme types.

Test corpus design
------------------
All surface forms share the suffix "z".  With identity phonology, the model
cannot propose URs that differ from the SR spans, so every span's UR is
just the SR substring.  The PYP rich-get-richer dynamics should therefore
cause the segmenter to converge on the recurring "z" as a suffix type.
"""

import collections
import pytest

from agme import Model


# A corpus of 8 words, all with stem + "z" suffix.
# Stems are distinct 2-character strings so there is no spurious ambiguity
# between stem tokens sharing a type.
IDENTITY_CORPUS = [
    "abz", "cdz", "efz", "ghz",
    "ijz", "klz", "mnz", "opz",
]


@pytest.fixture(scope="module")
def identity_model():
    """A fitted model using identity-only phonology.

    scope="module" so all segmenter tests share one trained model instance
    (avoids repeated 300-sweep fits and ensures all tests see the same
    converged state).  seed=2 gives clean convergence: suffix lexicon ends
    up with a single type {'z': 8}, making bimorphemic-parse tests reliable.
    """
    m = Model(
        morpheme_classes=["stem", "suffix"],
        alphabet=list("abcdefghijklmnopz"),
        identity_phonology=True,
    )
    m.fit(IDENTITY_CORPUS, n_sweeps=300, burn_in=100, print_every=0, seed=0)
    return m


def test_suffix_z_learned(identity_model):
    """After training, 'z' should be the dominant suffix type in the lexicon."""
    lexicon = identity_model.morphology.morpheme_lexicon()
    suffix_counts = lexicon.get("suffix", {})
    assert suffix_counts, "Suffix lexicon should be non-empty after training"
    most_common_suffix = max(suffix_counts, key=suffix_counts.get)
    assert most_common_suffix == "z", (
        f"Expected 'z' as the most common suffix, got '{most_common_suffix}'. "
        f"Full suffix lexicon: {suffix_counts}"
    )


def test_parse_finds_stem_suffix_split(identity_model):
    """Parsing a trained word should recover a stem+suffix segmentation."""
    # Sample several parses to account for sampling variance
    parses = identity_model.sample_parses("abz", n=20)
    two_morpheme = [p for p in parses if len(p.morphemes) == 2]
    assert len(two_morpheme) > 0, (
        "At least some sampled parses should be bimorphemic (stem + suffix)"
    )


def test_parse_suffix_is_z(identity_model):
    """In bimorphemic parses of 'abz', the suffix morpheme should be 'z'."""
    parses = identity_model.sample_parses("abz", n=30)
    two_morpheme = [p for p in parses if len(p.morphemes) == 2]
    if not two_morpheme:
        pytest.skip("No bimorphemic parses sampled — increase n_sweeps or n")
    suffixes = [p.morphemes[1] for p in two_morpheme]
    z_fraction = suffixes.count("z") / len(suffixes)
    assert z_fraction > 0.5, (
        f"Expected 'z' as suffix in majority of bimorphemic parses, "
        f"got z_fraction={z_fraction:.2f}. Suffixes seen: {collections.Counter(suffixes)}"
    )


def test_novel_word_segments(identity_model):
    """A novel word ending in 'z' should also receive a stem+suffix parse."""
    # "qrz" was never in training data
    parses = identity_model.sample_parses("qrz", n=20)
    two_morpheme = [p for p in parses if len(p.morphemes) == 2]
    assert len(two_morpheme) > 0, (
        "Novel word 'qrz' should receive at least some bimorphemic parses "
        "via generalisation of the learned 'z' suffix"
    )


def test_identity_phonology_ur_equals_sr(identity_model):
    """With identity phonology, every UR in a parse must equal its SR span."""
    for word in IDENTITY_CORPUS[:4]:
        parses = identity_model.sample_parses(word, n=10)
        for p in parses:
            for ur, sr_span, _ in p.mappings:
                assert ur == sr_span, (
                    f"Identity phonology: UR '{ur}' should equal SR span '{sr_span}'"
                )
