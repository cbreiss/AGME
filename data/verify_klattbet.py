"""Verify panphon coverage after the IPA map fix."""
import sys
sys.path.insert(0, ".")
from klattbet import KLATTBET_TO_IPA
import panphon
ft = panphon.FeatureTable()

# All symbols seen in the corpus (from prepare_brent survey)
CORPUS_SYMBOLS = list("@CDEGIJORSTUWYZabcdefghijklmnoprstuvwxyz^")

missing = [s for s in CORPUS_SYMBOLS if s not in KLATTBET_TO_IPA]
if missing:
    print(f"WARNING: no IPA mapping for: {missing}")
else:
    print("All corpus symbols have IPA mappings.")

results = []
for sym in sorted(KLATTBET_TO_IPA):
    ipa = KLATTBET_TO_IPA[sym]
    vecs = ft.word_to_vector_list(ipa, numeric=True)
    n = len(vecs)
    status = "OK" if n >= 1 else "MISSING"
    results.append((sym, ipa, n, status))

with open("data/klattbet_coverage.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Sym':4s} {'IPA':12s} {'Segs':5s} {'Status'}\n")
    f.write("-" * 35 + "\n")
    for sym, ipa, n, status in results:
        f.write(f"{sym!r:4s} {repr(ipa):12s} {n:5d}  {status}\n")

n_ok = sum(1 for _, _, n, _ in results if n >= 1)
n_multi = sum(1 for _, _, n, _ in results if n > 1)
print(f"Coverage: {n_ok}/{len(results)} symbols have panphon features ({n_multi} multi-segment, will be averaged)")
print("Full table in data/klattbet_coverage.txt")
