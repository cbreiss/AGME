"""Experiment: N_TRAIN=100, PRIOR_SCALE=2.25, N_SWEEPS=250, BURN_IN=50, IDENTITY=False.

Metrics reported:
  1. UR diversity — are URs collapsing to one form?
  2. Phonological weight deviation from P-map priors.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TRAIN        = 100
N_SWEEPS       = 250
BURN_IN        = 50
SEED           = 42
IDENTITY       = False
PYP_DISCOUNT   = 0.5
PYP_CONC       = 1.0
PRIOR_SCALE    = 2.25
EPENTH_PRIOR   = 1.0
DELET_PRIOR    = 1.0
PRIOR_SIGMA    = 1.0

KLATTBET_IPA = {
    "D": "ð", "T": "θ", "S": "ʃ", "Z": "ʒ", "G": "ŋ",
    "C": "tʃ", "J": "dʒ", "I": "ɪ", "E": "ɛ", "U": "ʊ",
    "O": "ɔɪ", "W": "aʊ", "Y": "aɪ", "R": "ɝ", "@": "æ", "^": "ʌ",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_dir = Path("data")
surface_forms = (data_dir / "words_train.txt").read_text(encoding="utf-8").splitlines()
gold_segs     = (data_dir / "words_gold.txt").read_text(encoding="utf-8").splitlines()
orthos        = (data_dir / "words_ortho.txt").read_text(encoding="utf-8").splitlines()

# Select top-N_TRAIN types by token frequency
from collections import Counter
type_counts = Counter(surface_forms)
top_types = set(w for w, _ in type_counts.most_common(N_TRAIN))
sf_train = [w for w in surface_forms if w in top_types]

print(f"Training on {len(sf_train)} tokens ({len(set(sf_train))} types)")
print(f"Config: N_SWEEPS={N_SWEEPS}, BURN_IN={BURN_IN}, PRIOR_SCALE={PRIOR_SCALE}, "
      f"IDENTITY={IDENTITY}, PYP_DISCOUNT={PYP_DISCOUNT}")
print()

# ---------------------------------------------------------------------------
# Fit model
# ---------------------------------------------------------------------------
from agme import Model
import time

t0 = time.perf_counter()
model = Model(
    morpheme_classes=["stem", "suffix"],
    pyp_discount=PYP_DISCOUNT,
    pyp_concentration=PYP_CONC,
    identity_phonology=IDENTITY,
    ipa_map=KLATTBET_IPA,
    prior_scale=PRIOR_SCALE,
    epenthesis_prior=EPENTH_PRIOR,
    deletion_prior=DELET_PRIOR,
    prior_sigma=PRIOR_SIGMA,
)
model.fit(
    sf_train,
    n_sweeps=N_SWEEPS,
    burn_in=BURN_IN,
    maxent_update_every=10,
    print_every=25,
    seed=SEED,
)
elapsed = time.perf_counter() - t0
print(f"\nTraining finished in {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Metric 1: UR diversity
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("METRIC 1: UR posterior diversity")
print("="*65)

ur_post = model._training_state.ur_posterior
if ur_post:
    top_urs = sorted(ur_post.items(), key=lambda x: -x[1])[:20]
    print(f"Total unique URs in posterior: {len(ur_post)}")
    print(f"\nTop-20 URs by posterior probability:")
    for ur, prob in top_urs:
        print(f"  {ur:20s}  {prob:.4f}")
    top1_prob = top_urs[0][1]
    print(f"\nTop-1 UR mass: {top1_prob:.4f}  "
          f"({'COLLAPSED' if top1_prob > 0.5 else 'OK'})")
else:
    print("  (no posterior accumulated — check burn_in vs n_sweeps)")

# ---------------------------------------------------------------------------
# Metric 2: Phonological weight deviation from priors
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("METRIC 2: Phonological weight deviation from P-map priors")
print("="*65)

import numpy as np
weights  = model.phonology.weights
prior_mu = model.phonology._prior_mu
deviations = weights - prior_mu
n_constraints = len(weights)

print(f"Total constraints: {n_constraints}")
print(f"Mean |deviation|:  {np.mean(np.abs(deviations)):.4f}")
print(f"Max  |deviation|:  {np.max(np.abs(deviations)):.4f}")
print(f"Constraints with |dev| > 0.01: "
      f"{np.sum(np.abs(deviations) > 0.01)} / {n_constraints}")
print(f"Constraints with |dev| > 0.05: "
      f"{np.sum(np.abs(deviations) > 0.05)} / {n_constraints}")

print(f"\nTop-15 constraints by |deviation from prior|:")
idx = np.argsort(-np.abs(deviations))[:15]
for i in idx:
    c = model.phonology.constraints[i]
    print(f"  {repr(c):30s}  w={weights[i]:.4f}  μ={prior_mu[i]:.4f}  "
          f"Δ={deviations[i]:+.4f}")

overall_ok = (np.max(np.abs(deviations)) > 0.01)
print(f"\nWeight deviation check: {'OK — weights moved' if overall_ok else 'FAIL — weights frozen at priors'}")

# ---------------------------------------------------------------------------
# Per-class summary
# ---------------------------------------------------------------------------
print()
model.print_sr_types()

print("\n\nTop-10 UR report (stem):")
model.print_ur_report()
