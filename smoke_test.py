from agme import Model

corpus = ["dɑgz", "kæts", "bʊks", "læbz"]
m = Model(morpheme_classes=["stem", "suffix"])
m.fit(corpus, n_sweeps=5, burn_in=2, print_every=5)
result = m.parse("dɑgz")
print("morphemes:", result.morphemes)
print("classes:  ", result.morpheme_classes)
print("ur:       ", result.ur)
print("log_prob: ", result.log_prob)
print("OK")
