# AGME Wishlist

Desirable extensions, roughly ordered by scientific payoff.
None of these are blocking for current experiments.

---

## Phonological grammar

### User-supplied feature matrix
**Where:** `agme/features.py:49`
Averaging panphon vectors for compound phonemes (affricates tʃ/dʒ, diphthongs
aʊ/aɪ/ɔɪ) is a rough approximation.  The right fix is to let the user pass in
a custom feature matrix (e.g. Hayes 2009 SPE-style features, or any sparse
matrix where compound segments have their own rows) instead of relying on
panphon.  The `ipa_map` pathway already exists; the next step is a
`feature_matrix` parameter to `build_distance_matrix`.

### HMC weight sampling
**Where:** `agme/phonology/grammar.py:290` (`run_weight_update`)
Replace the L-BFGS-B MAP estimate with full posterior sampling over constraint
weights using Hamiltonian Monte Carlo.  The hook is already stubbed — swap
`fit_weights` for an HMC sampler.  Relevant for quantifying uncertainty in
learned *MAP weights and for the production/RT modeling extension below.

### FST-based exact partition function
**Where:** `agme/phonology/candidates.py` (DECISION POINT note)
Currently Z(UR) is approximated by summing over a finite candidate set.
An exact computation via pynini (log-semiring WFST compose + shortestdistance)
would remove the approximation error.  Optional dependency; pure-Python
fallback stays as-is.

### Context-sensitive \*MAP constraints
**Where:** `agme/phonology/constraints.py:210`
`StarMapConstraint(x, y, left_ctx, right_ctx)` already accepts context
arguments and `count_from_alignment` handles them, but no builder currently
generates context-sensitive constraints automatically.  Needed to model e.g.
place assimilation (*MAP(n, m) / __ [labial]).

---

## Morphological grammar

### Within-domain morpheme reordering
Currently multiple prefixes or multiple suffixes must appear in the fixed order
given at init.  Flexible within-domain ordering would support languages where
affix order is partly free.

### Learned markedness constraints
Swap the fixed *MAP constraint set for one discovered by a Hayes & Wilson
(2008) phonotactic learner run on the training data.  Classmethod stub
`MarkednessFST.from_phonotactic_learner()` is reserved in the plan.

---

## Phonology / lexicon interface

### Lexically-specific constraint weights
**Where:** `agme/api.py` (`predict` / `log_prob` signatures)
`PhonologicalGrammar.score(..., lexical_index=None)` stub is reserved.
Needed to model Zuraw (2010)-style lexical conservatism where individual
lexical items have their own weight perturbations around the population mean.

---

## Corpus / evaluation / modeling

### Production and RT modeling
Given a novel UR, sample candidate SRs from the phonological model and relate
sampling probabilities to wug-test response times (slower RT ≈ lower P(SR|UR)).
Requires HMC weights (for uncertainty) and the lexically-specific weights
extension for individual-level predictions.

### Brent corpus — full-scale run
Current notebooks use top-500 types for speed.  A full run (all ~3 K types,
500+ sweeps) would give publishable word-segmentation F-scores comparable to
Goldwater et al. (2009).  Needs either a multi-hour background job or a
compute cluster.
