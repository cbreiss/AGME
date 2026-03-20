import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

import random
from agme import Model
from klattbet import KLATTBET_TO_IPA

def load_random_tokens(path, n, seed):
    rng = random.Random(seed)
    with open(path, encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return rng.sample(tokens, min(n, len(tokens)))

surfaces = load_random_tokens("data/words_train.txt", 100, seed=42)
print(f"Loaded {len(surfaces)} tokens, {len(set(surfaces))} unique types", flush=True)

model = Model(
    morpheme_classes=["stem", "suffix"],
    identity_phonology=False,
    ipa_map=KLATTBET_TO_IPA,
)
model.fit(surfaces, n_sweeps=20, burn_in=5, maxent_update_every=5,
          print_every=1, seed=42)
