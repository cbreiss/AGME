# Engineering Choices (Tractability vs. Scientific Realism)

This file records implementation decisions made for computational tractability
that have theoretical significance and may be worth revisiting when scaling
or when scientific accuracy is the priority.

---

## EC-1: Type-level Gibbs inference

**What**: The training loop iterates over **unique surface types** rather than
individual tokens.  All tokens of the same surface form are assigned the same
parse; PYP caches are updated by adding/removing `n` copies at once (where `n`
is the token frequency).

**Tractability gain**: Per-sweep cost drops from O(N_tokens) to O(N_types).
For a natural corpus where many types repeat (Zipfian distribution), this is
typically a 5–20× speedup.

**Scientific caveat**: The type-level approximation treats all tokens of the
same type as exchangeable.  Under CRP exchangeability this is a valid
approximation, but it prevents the model from assigning different parses to
different tokens of the same surface string — which could matter if the corpus
contains phonological ambiguity (e.g. the same surface string mapping to two
different URs with different frequencies).  For most morphological learning
tasks this is not an issue.

**To revert**: Change `run_training` in `agme/inference/training.py` to
iterate over all tokens with `n=1` at each step (token-level Gibbs).

---

## EC-2: SR candidate pruning (max_sr_candidates = 50)

**What**: After generating all candidate surface forms for a UR via
`candidates_for()` (exhaustive single-edit + 30 random multi-edit), the
violation matrix for that UR is pruned to the **top 50 candidates** ranked
by P-map prior harmony.  Lower harmony = higher P-map probability = more
phonologically natural candidate.

Pruning uses the **P-map prior weights** (not the current learned weights)
so that the candidate set is stable across weight updates.

**Tractability gain**: Reduces per-UR violation matrix size from ~200–400 rows
to 50.  This shrinks the `V @ w` matmul in `log_prob`, the `probs @ V` matmul
in `fit_weights`, and the memory footprint of the violation matrix cache.

**Scientific caveat**: The approximate partition function Z(UR) is now computed
over a smaller candidate set.  Low-probability candidates (those penalised by
the P-map prior) are excluded.  If the true SR for a UR is phonologically
unusual (requires a rare mapping penalised by the P-map), it may be
under-represented.  The observed SR is always added to the matrix even if
pruned, so observed data is never excluded, but the normalisation constant Z
may be slightly underestimated.

**Parameter**: `MaxEntPhonology(max_sr_candidates=50)`.  Set to `0` to disable.

**To revert**: Set `max_sr_candidates=0` (or increase the limit) in
`agme/api.py` when constructing `MaxEntPhonology`.

---

## EC-3: Reduced UR candidates per span (top_k_urs = 8)

**What**: The `URProposer` generates at most **8 candidate URs per morpheme
span** during the forward-backward DP segmentation step.  Previously 10.

**Tractability gain**: Reduces the inner loop in `segmenter.py` from 10 to 8
UR evaluations per span, a ~20% reduction in phonological scoring calls.

**Scientific caveat**: The UR candidate set for each span is already a
heuristic approximation (importance-sampling from a proposal distribution over
small edits of the observed SR span).  Reducing from 10 to 8 slightly narrows
the search.  For morphemes with large UR–SR discrepancy (e.g. heavy epenthesis
or long-distance metathesis), the true UR may occasionally be missed.

**Parameter**: `run_training(top_k_urs=8)`.

**To revert**: Increase `top_k_urs` in `agme/api.py`'s call to
`run_training()`, or expose it as a `Model.fit()` parameter.

---

## EC-4: Maximum morpheme length (max_morpheme_len = 8)

**What**: The forward-backward segmentation DP considers morpheme spans of at
most **8 characters**.  Previously 10.

**Tractability gain**: Reduces the number of (start, end, class, UR) cells in
the DP chart.  The per-sweep cost is roughly quadratic in `max_morpheme_len`,
so reducing from 10 to 8 is a ~36% reduction in DP cells for words longer
than 8 characters.

**Scientific caveat**: Words or morphemes longer than 8 characters cannot be
parsed as a single morpheme.  This is generally not a problem for English or
other morphologically simple languages, but may matter for polysynthetic
languages with long stems.

**Parameter**: `run_training(max_morpheme_len=8)`.

**To revert**: Increase `max_morpheme_len` in `agme/api.py`'s call to
`run_training()`, or expose it as a `Model.fit()` parameter.

---

## EC-5: Levenshtein distance as proxy for P-map harmony in candidate pruning

**What**: When pruning the SR candidate set to `max_sr_candidates` (EC-2),
candidates are ranked by **Levenshtein edit distance** from the UR rather than
by true P-map harmony (i.e., `Σ w_c · violations_c`).

Levenshtein distance counts the total number of insertions, deletions, and
substitutions needed to reach the SR from the UR, without weighting by
featural distance.  True P-map harmony weights each edit by the panphon
distance between the segments involved — so a substitution between featurally
similar segments (e.g. /p/→[b]) costs less than one between dissimilar segments
(e.g. /p/→[z]).

Ranking is done before violation vectors are computed, using
`agme.utils.levenshtein_distance` (O(min(m,n)) space, no traceback).

**Tractability gain**: Avoids computing full violation vectors (which require
`levenshtein_alignment` + count-matrix construction) for the ~85% of candidates
that will be discarded.  Reduces alignment calls from ~2.5M to ~300K per run
on a 100-token Brent sample.

**Scientific caveat**: The proxy ranking slightly misorders candidates relative
to the true P-map prior.  Specifically, a short easy substitution (e.g. /p/→[b],
distance 1) and a long hard substitution (e.g. /p/→[z], distance 1) receive the
same rank, even though P-map harmony would prefer /p/→[b] because panphon
similarity(p, b) > panphon similarity(p, z).  In practice, since all single-edit
candidates have distance 1 and typically outnumber `max_sr_candidates`, the kept
set is a random subset of single-edit candidates — which is still linguistically
reasonable.

**To revert**: Replace the `levenshtein_distance` ranking in
`MaxEntPhonology._ensure_cand_matrix` with full P-map harmony ranking (compute
all violation vectors, rank by `prior_w @ v`, then prune).  This is only correct
if the extra computation cost is acceptable.

---

## EC-6: Reduced random multi-edit candidates (n_random = 5)

**What**: Both the SR candidate generator (`candidates_for` in
`agme/phonology/candidates.py`) and the UR proposer (`URProposer` in
`agme/inference/ur_proposer.py`) include a small number of random multi-edit
candidates to improve mixing beyond the single-edit neighbourhood.  The default
was 30 (SR candidates) and 20 (UR proposer); both are reduced to **5**.

**Tractability gain**: Reduces `random_edit` calls from ~992K to ~250K per run
(~4× reduction), saving ~6s on 100 tokens.  Also shrinks the pre-pruning
candidate set for each UR from ~420 to ~395, slightly reducing the
`levenshtein_distance` calls in `_ensure_cand_matrix` during cold start.

**Scientific caveat**: Random multi-edit candidates are the only way the model
can propose SR candidates (or UR hypotheses) that differ from the observed form
by more than one edit — e.g., a stem with both a deletion and a substitution, or
a UR that requires two simultaneous changes to reach the SR.  Reducing from 20–30
to 5 makes it less likely that the sampler explores such candidates in any given
sweep.  For languages with simple phonology (Brent corpus, English plurals) this
is harmless.  For languages with complex multi-step alternations (e.g. vowel
harmony + consonant mutation), the reduced exploration could slow convergence or
miss valid parse hypotheses.

**Parameters**: `candidates_for(n_random=5)` and `URProposer(n_random=5)`.

**To revert**: Increase `n_random` in both call sites, or expose it as a
`Model.fit()` parameter.
