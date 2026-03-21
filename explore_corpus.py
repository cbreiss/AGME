"""Explore the training corpus to find morphologically related forms."""
from collections import Counter

with open('data/words_train.txt') as f:
    words = [l.strip() for l in f if l.strip()]

counts = Counter(words)
print(f'Total tokens: {len(words)}')
print(f'Unique types: {len(counts)}')

# ---------------------------------------------------------
# Strategy: find paradigm clusters.
# A stem S has a paradigm if we can find S, S+"z"/S+"s", S+"t"/S+"d", etc.
# In Klattbet: z=z, s=s, t=t, d=d, I=ɪ, Z=ʒ
#
# Past tense allomorphs: -t, -d, -Id (Klattbet: -t, -d, -Id)
# Plural/3sg allomorphs:  -z, -s, -Iz (Klattbet: -z, -s, -Iz)
# ---------------------------------------------------------

# Find words ending in each target suffix
def strip_suffix(word, suffix):
    if word.endswith(suffix) and len(word) > len(suffix):
        return word[:-len(suffix)]
    return None

# Collect stems for each suffix
past_suffixes   = ['t', 'd', 'Id']  # walked, played, wanted
plural_suffixes = ['z', 's', 'Iz']  # dogs, cats, buses

# For each word ending in a past suffix, check if the stem also appears
print("\n--- Looking for past-tense paradigms (stem + {t, d, Id}) ---")
paradigm_stems_past = []
for word, cnt in counts.items():
    for suf in past_suffixes:
        stem = strip_suffix(word, suf)
        if stem and len(stem) >= 2 and stem in counts:
            # Check if any other past-tense allomorph also present
            others = [s for s in past_suffixes if s != suf and stem + s in counts]
            paradigm_stems_past.append((stem, suf, word, cnt, others))

# Sort by frequency of the suffixed form
paradigm_stems_past.sort(key=lambda x: -x[2+1])
seen = set()
for stem, suf, word, cnt, others in paradigm_stems_past[:40]:
    if stem not in seen:
        seen.add(stem)
        variants = [stem] + [stem+s for s in past_suffixes if stem+s in counts]
        variant_str = ', '.join(f'{v}({counts[v]})' for v in variants)
        print(f'  {variant_str}')

print("\n--- Looking for plural/3sg paradigms (stem + {z, s, Iz}) ---")
paradigm_stems_plur = []
for word, cnt in counts.items():
    for suf in plural_suffixes:
        stem = strip_suffix(word, suf)
        if stem and len(stem) >= 2 and stem in counts:
            others = [s for s in plural_suffixes if s != suf and stem + s in counts]
            paradigm_stems_plur.append((stem, suf, word, cnt, others))

paradigm_stems_plur.sort(key=lambda x: -x[2+1])
seen = set()
for stem, suf, word, cnt, others in paradigm_stems_plur[:40]:
    if stem not in seen:
        seen.add(stem)
        variants = [stem] + [stem+s for s in plural_suffixes if stem+s in counts]
        variant_str = ', '.join(f'{v}({counts[v]})' for v in variants)
        print(f'  {variant_str}')

# Show all unique types ending in t/d/s/z
print("\n--- All types ending in t (sorted by freq) ---")
t_words = [(w,c) for w,c in counts.items() if w.endswith('t') and len(w)>1]
t_words.sort(key=lambda x:-x[1])
for w,c in t_words[:30]:
    print(f'  {w!r}: {c}')

print("\n--- All types ending in d (sorted by freq) ---")
d_words = [(w,c) for w,c in counts.items() if w.endswith('d') and len(w)>1]
d_words.sort(key=lambda x:-x[1])
for w,c in d_words[:30]:
    print(f'  {w!r}: {c}')

print("\n--- All types ending in z (sorted by freq) ---")
z_words = [(w,c) for w,c in counts.items() if w.endswith('z') and len(w)>1]
z_words.sort(key=lambda x:-x[1])
for w,c in z_words[:30]:
    print(f'  {w!r}: {c}')

print("\n--- All types ending in s (sorted by freq) ---")
s_words = [(w,c) for w,c in counts.items() if w.endswith('s') and len(w)>1]
s_words.sort(key=lambda x:-x[1])
for w,c in s_words[:20]:
    print(f'  {w!r}: {c}')
