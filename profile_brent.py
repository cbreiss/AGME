"""Profile AGME (full model, phonology active) on top-100 Brent types.

Run with:
    conda run -n AGME python profile_brent.py

Outputs a sorted cProfile report to profile_brent_results.txt.
"""

import cProfile
import pstats
import io
import sys
import os
from collections import Counter

# Locate packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

from agme import Model
from klattbet import KLATTBET_TO_IPA

# ── Configuration ──────────────────────────────────────────────────────────────
N_TYPES   = 100   # number of tokens to sample (regardless of type count)
N_SWEEPS  = 30    # enough to get meaningful profile data
BURN_IN   = 10
SEED      = 42
OUTPUT    = "profile_brent_results.txt"
# ──────────────────────────────────────────────────────────────────────────────


def load_random_tokens(path: str, n: int, seed: int) -> list[str]:
    """Return n randomly-sampled tokens (not types) from the corpus."""
    import random
    rng = random.Random(seed)
    with open(path, encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return rng.sample(tokens, min(n, len(tokens)))


def run():
    surfaces = load_random_tokens("data/words_train.txt", N_TYPES, SEED)
    n_unique = len(set(surfaces))
    alphabet_size = len(set("".join(surfaces)))
    print(f"Loaded {len(surfaces)} tokens, {n_unique} unique types, "
          f"alphabet size {alphabet_size}")
    print(f"Training: {N_SWEEPS} sweeps, burn_in={BURN_IN}, seed={SEED}")
    print(f"Full phonological grammar (identity_only=False)")
    print()

    model = Model(
        morpheme_classes=["stem", "suffix"],
        identity_phonology=False,    # full phonology — the expensive path
        ipa_map=KLATTBET_TO_IPA,
    )
    model.fit(
        surfaces,
        n_sweeps=N_SWEEPS,
        burn_in=BURN_IN,
        maxent_update_every=5,
        print_every=1,          # flush after every sweep so progress is visible
        seed=SEED,
    )


# ── Profile ───────────────────────────────────────────────────────────────────
pr = cProfile.Profile()
pr.enable()
run()
pr.disable()

# ── Report ────────────────────────────────────────────────────────────────────
buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf)
ps.strip_dirs()
ps.sort_stats("cumulative")
ps.print_stats(40)   # top-40 by cumulative time
ps.sort_stats("tottime")
ps.print_stats(40)   # top-40 by self time

report = buf.getvalue()

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(report)

# Print a concise human-readable summary to stdout (safe for any encoding)
lines = report.splitlines()
print("=" * 70)
print("TOP FUNCTIONS BY CUMULATIVE TIME")
print("=" * 70)
in_cumul = False
cumul_count = 0
for line in lines:
    if "cumulative" in line and "filename" in line:
        in_cumul = True
        print(line)
        continue
    if in_cumul and line.strip():
        print(line)
        cumul_count += 1
        if cumul_count >= 25:
            break

print()
print("=" * 70)
print("TOP FUNCTIONS BY SELF TIME")
print("=" * 70)
in_self = False
self_count = 0
# Re-parse for tottime section
for line in lines:
    if "tottime" in line and "filename" in line and not in_self:
        in_self = True
        print(line)
        continue
    if in_self and line.strip():
        print(line)
        self_count += 1
        if self_count >= 25:
            break

print()
print(f"Full profile written to: {OUTPUT}")
