"""Build a subset of the Brent corpus focused on morphological paradigms.

Strategy
--------
1. Find stem/suffix paradigm pairs: stem + {z,s,Iz} and stem + {t,d,Id}
2. Include all words in those paradigms
3. Pad with enough high-frequency content to give morphological context

Output: data/words_paradigm.txt  (one token per line, Klattbet)
"""
from collections import Counter

with open('data/words_train.txt') as f:
    words = [l.strip() for l in f if l.strip()]

counts = Counter(words)
print(f"Full corpus: {len(words)} tokens, {len(counts)} types\n")

# ---- Paradigm detection ------------------------------------------------

past_suffs   = ['Id', 't', 'd']   # longest first so Id is tried before d
plural_suffs = ['Iz', 'z', 's']

def find_paradigms(suffix_list):
    """Return list of (stem, {suffix: count}, stem_count) triples."""
    stems = {}
    for word in counts:
        for suf in suffix_list:
            if word.endswith(suf) and len(word) > len(suf):
                stem = word[:-len(suf)]
                if stem in counts and len(stem) >= 2:
                    if stem not in stems:
                        stems[stem] = {}
                    stems[stem][suf] = counts[word]
    result = []
    for stem, suf_counts in stems.items():
        result.append((stem, suf_counts, counts[stem]))
    result.sort(key=lambda x: -max(x[1].values()))
    return result

plur_paradigms = find_paradigms(plural_suffs)
past_paradigms = find_paradigms(past_suffs)

print("=== Plural/3sg paradigms (stem + {Iz,z,s}) ===")
for stem, suf_counts, stem_cnt in plur_paradigms[:30]:
    parts = [f"{stem}({stem_cnt})"]
    for s in plural_suffs:
        if s in suf_counts:
            parts.append(f"{stem+s}({suf_counts[s]})")
    print(f"  {', '.join(parts)}")

print("\n=== Past-tense paradigms (stem + {Id,t,d}) ===")
for stem, suf_counts, stem_cnt in past_paradigms[:30]:
    parts = [f"{stem}({stem_cnt})"]
    for s in past_suffs:
        if s in suf_counts:
            parts.append(f"{stem+s}({suf_counts[s]})")
    print(f"  {', '.join(parts)}")

# ---- Build selected type set ------------------------------------------

# All words in plural paradigms
selected_types = set()
for stem, suf_counts, _ in plur_paradigms:
    selected_types.add(stem)
    for s in suf_counts:
        selected_types.add(stem + s)

# All words in past paradigms
for stem, suf_counts, _ in past_paradigms:
    selected_types.add(stem)
    for s in suf_counts:
        selected_types.add(stem + s)

print(f"\nTypes from paradigms: {len(selected_types)}")
print(f"Tokens from paradigms: {sum(counts[t] for t in selected_types if t in counts)}")

# Pad with high-frequency types to give the model contextual mass
all_by_freq = sorted(counts.items(), key=lambda x: -x[1])
for word, cnt in all_by_freq:
    if len(selected_types) >= 200:
        break
    selected_types.add(word)

print(f"\nAfter padding to 200 types: {len(selected_types)} types")
total_tokens = sum(counts[t] for t in selected_types if t in counts)
print(f"Total tokens: {total_tokens}")

# Verify paradigm coverage
print("\n--- z/s/Iz coverage in selected set ---")
three_way = [(s, sc, sc2) for s, sc, sc2 in plur_paradigms if len(sc) >= 2]
for stem, suf_counts, stem_cnt in three_way[:20]:
    in_set = all(w in selected_types for w in [stem] + [stem+s for s in suf_counts])
    parts = [f"{stem}({stem_cnt})"] + [f"{stem+s}({suf_counts[s]})" for s in plural_suffs if s in suf_counts]
    mark = "✓" if in_set else "✗"
    print(f"  {mark} {', '.join(parts)}")

# Write output
out_tokens = []
for word in words:
    if word in selected_types:
        out_tokens.append(word)

print(f"\nWriting {len(out_tokens)} tokens ({len(selected_types)} types) to data/words_paradigm.txt")
with open('data/words_paradigm.txt', 'w') as f:
    for w in out_tokens:
        f.write(w + '\n')

print("Done.")
