"""Klattbet → IPA mapping for the BreissHayesBrent corpus.

One character per phoneme in the Klattbet encoding.  For complex segments
(affricates C=tʃ, J=dʒ; diphthongs O=ɔɪ, W=aʊ, Y=aɪ), the IPA string has
two characters; agme.features averages their panphon feature vectors so the
pair is treated as a single phoneme in the P-map prior.

ASCII g maps to script-ɡ (U+0261) which panphon recognises.
R (American r-colored vowel ɝ) maps to ɹ (U+0279, voiced alveolar approximant)
— the closest single-segment panphon approximation to the rhotacized vowel.

Import this dict and pass it as ``ipa_map`` to Model() to get correct P-map
priors for every Klattbet symbol.
"""

KLATTBET_TO_IPA: dict[str, str] = {
    # --- Special uppercase symbols ---
    "@": "\u00e6",        # æ  TRAP (that, Sammy)
    "C": "t\u0283",       # tʃ CHAR (such, chew, watched) — affricate
    "D": "\u00f0",        # ð  THERE (the, that)
    "E": "\u025b",        # ɛ  DRESS (get, said, then)
    "G": "\u014b",        # ŋ  RING (eating, doing, standing)
    "I": "\u026a",        # ɪ  KIT (is, sit, kitty)
    "J": "d\u0292",       # dʒ JUDGE (just, juice, orange) — affricate
    "O": "\u0254\u026a",  # ɔɪ CHOICE (toy, toys, oink) — diphthong
    "R": "\u0279",        # ɹ  NURSE/r-colored (girl, hurt, yeah)
                          #    ɝ not in panphon; ɹ is closest single-seg approx
    "S": "\u0283",        # ʃ  SHOE (she, show, wash)
    "T": "\u03b8",        # θ  THIN (thing, three, mouth)
    "U": "\u028a",        # ʊ  FOOT (put, good, look)
    "W": "a\u028a",       # aʊ MOUTH (out, down, ow) — diphthong
    "Y": "a\u026a",       # aɪ PRICE (I, like, hi) — diphthong
    "Z": "\u0292",        # ʒ  MEASURE (usually, measuring)
    "^": "\u028c",        # ʌ  STRUT (what, wants, the-reduced)
    # --- Special lowercase symbols ---
    "a": "\u0251",        # ɑ  LOT (want, father)
    "c": "\u0254",        # ɔ  THOUGHT (dog, for, Morgie)
    "e": "e",             # e  FACE nucleus (okay, say)
    "i": "i",             # i  FLEECE (hear, eat, kitty-final)
    "o": "o",             # o  GOAT nucleus (go, no, okay)
    "u": "u",             # u  GOOSE (you, do, food)
    "x": "\u0259",        # ə  COMMA/schwa (gonna, wanna)
    "y": "j",             # j  YES-consonant (you, yeah)
    # --- Standard consonants (same in Klattbet and IPA) ---
    "b": "b",
    "d": "d",
    "f": "f",
    "g": "\u0261",        # ɡ  voiced velar plosive (go, get, dog)
                          #    ASCII 'g' not in panphon; script ɡ (U+0261) is
    "h": "h",
    "j": "j",             # j  palatal glide (rare in corpus; same as y)
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "v": "v",
    "w": "w",
    "z": "z",
}
