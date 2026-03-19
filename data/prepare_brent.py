"""Prepare the BreissHayesBrent corpus for AGME word-level morphology learning.

Input files (all 28,391 lines, one utterance per line):
  BreissHayesBrent/Training.txt   — unsegmented Klattbet utterance strings
  BreissHayesBrent/S_Model.txt    — space-separated words with boundary markers:
                                    /  true morpheme boundary (suffix or prefix)
                                    >  accidental match (NOT a real boundary)
  BreissHayesBrent/Orthography.txt — corresponding English orthography

Output files (written to data/):
  words_train.txt      — one word surface-form per line (Klattbet, no markers)
                         sorted by descending token frequency; top N_WORDS words
  words_gold.txt       — corresponding gold-standard segmentations
                         Uses '|' to mark TRUE boundaries (/) and
                         '+' to mark accidental boundaries (>) for evaluation
  words_ortho.txt      — corresponding orthographic word forms

Klattbet symbol legend (one char per phoneme):
  Uppercase: D=ð  T=θ  S=ʃ  Z=ʒ  G=ŋ  C=tʃ  J=dʒ  I=ɪ  E=ɛ  U=ʊ
             O=ɔɪ  W=aʊ  Y=aɪ  R=ɝ  @=æ  ^=ʌ
  Lowercase: a=ɑ  c=ɔ  e=eɪ  i=iː  o=oʊ  u=uː  x=ə  y=j
             (all other lowercase = standard IPA)
"""

import collections, re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA = Path("data/BreissHayesBrent")
OUT = Path("data")

# Maximum number of word TYPES to include (by token frequency).
# Set to None to include all types.  500 is large enough to capture the main
# morphological patterns while keeping AGME training tractable (<10 min).
N_WORDS = 500


# ---------------------------------------------------------------------------
# Step 1: Parse S_Model.txt to extract words with boundary annotations
# ---------------------------------------------------------------------------

def parse_word(klattbet_with_markers: str) -> tuple[str, str, list[int], list[int]]:
    """Parse a Klattbet word with / and > boundary markers.

    Returns
    -------
    surface : str
        Phoneme string with markers removed (pure Klattbet).
    segmented : str
        Readable segmentation: '|' for true boundaries, '+' for accidentals.
        E.g. 'kIt/i' → surface='kIti', segmented='kIt|i'
    true_boundaries : list[int]
        Character positions (0-indexed, in surface string) AFTER which a
        true morpheme boundary falls (marked by / in original).
    acc_boundaries : list[int]
        Character positions AFTER which an accidental boundary falls (>).
    """
    surface_chars = []
    segmented_chars = []
    true_boundaries = []
    acc_boundaries = []

    i = 0
    while i < len(klattbet_with_markers):
        ch = klattbet_with_markers[i]
        if ch == "/":
            # True boundary: record position AFTER the previous character
            if surface_chars:
                true_boundaries.append(len(surface_chars) - 1)
            segmented_chars.append("|")
        elif ch == ">":
            # Accidental boundary
            if surface_chars:
                acc_boundaries.append(len(surface_chars) - 1)
            segmented_chars.append("+")
        else:
            surface_chars.append(ch)
            segmented_chars.append(ch)
        i += 1

    return (
        "".join(surface_chars),
        "".join(segmented_chars),
        true_boundaries,
        acc_boundaries,
    )


print("Parsing S_Model.txt ...")
with open(DATA / "S_Model.txt", encoding="utf-8") as f:
    smodel_lines = [line.rstrip("\n") for line in f]
with open(DATA / "Orthography.txt", encoding="utf-8") as f:
    ortho_lines = [line.rstrip("\n") for line in f]

# Collect (surface, segmented, ortho) for all word tokens
all_tokens: list[tuple[str, str, str]] = []  # (surface, segmented, ortho)

n_parse_errors = 0
for sline, oline in zip(smodel_lines, ortho_lines):
    s_words = sline.split()
    o_words = oline.split()
    if len(s_words) != len(o_words):
        n_parse_errors += 1
        # Use empty ortho for mismatched lines
        o_words = s_words  # fallback

    for sw, ow in zip(s_words, o_words):
        surface, segmented, true_bs, acc_bs = parse_word(sw)
        if surface:  # skip empty strings
            all_tokens.append((surface, segmented, ow))

print(f"  {len(all_tokens)} word tokens parsed ({n_parse_errors} utterance-level length mismatches)")

# ---------------------------------------------------------------------------
# Step 2: Select top-N word types by token frequency
# ---------------------------------------------------------------------------

# Count token frequency for each surface form
surface_count = collections.Counter(tok[0] for tok in all_tokens)
print(f"  {len(surface_count)} unique word types")

# Select top-N surface types
if N_WORDS is not None:
    top_types = set(w for w, _ in surface_count.most_common(N_WORDS))
    print(f"  Keeping top {N_WORDS} types ({sum(c for w,c in surface_count.items() if w in top_types)} tokens)")
else:
    top_types = set(surface_count.keys())

# Filter tokens to those in the top types
filtered_tokens = [(s, seg, o) for s, seg, o in all_tokens if s in top_types]
print(f"  {len(filtered_tokens)} tokens after filtering")

# ---------------------------------------------------------------------------
# Step 3: Write output files
# ---------------------------------------------------------------------------

# Shuffle order to avoid any corpus-order effects (sort by frequency for readability)
# We keep the original token order for reproducible evaluation

surfaces = [t[0] for t in filtered_tokens]
segmenteds = [t[1] for t in filtered_tokens]
orthos = [t[2] for t in filtered_tokens]

(OUT / "words_train.txt").write_text("\n".join(surfaces) + "\n", encoding="utf-8")
(OUT / "words_gold.txt").write_text("\n".join(segmenteds) + "\n", encoding="utf-8")
(OUT / "words_ortho.txt").write_text("\n".join(orthos) + "\n", encoding="utf-8")

print(f"\nWrote:")
print(f"  data/words_train.txt  — {len(surfaces)} word tokens (surface forms only)")
print(f"  data/words_gold.txt   — {len(segmenteds)} word tokens (with | and + markers)")
print(f"  data/words_ortho.txt  — {len(orthos)} word tokens (orthography)")

# ---------------------------------------------------------------------------
# Step 4: Summary statistics
# ---------------------------------------------------------------------------

n_mono = sum(1 for s in segmenteds if "|" not in s and "+" not in s)
n_true_boundary = sum(s.count("|") for s in segmenteds)
n_acc_boundary = sum(s.count("+") for s in segmenteds)

print(f"\nBoundary statistics (in selected {len(segmenteds)} tokens):")
print(f"  Monomorphemic words (no boundary):        {n_mono} ({100*n_mono/len(segmenteds):.1f}%)")
print(f"  Words with ≥1 true boundary (|):          {len(segmenteds)-n_mono} ({100*(len(segmenteds)-n_mono)/len(segmenteds):.1f}%)")
print(f"  Total true boundaries:                    {n_true_boundary}")
print(f"  Total accidental boundary-like positions: {n_acc_boundary}")

# Show unique alphabet
all_phones = sorted(set("".join(surfaces)))
print(f"\nPhoneme inventory ({len(all_phones)} symbols):")
print(f"  {' '.join(all_phones)}")

# Show 10 example words with gold segmentation
print("\nExample words (surface → gold segmentation):")
seen = set()
for s, seg, o in filtered_tokens[:200]:
    if o not in seen and "|" in seg:
        print(f"  {s:15s} → {seg:15s}  ({o})")
        seen.add(o)
    if len(seen) >= 10:
        break
