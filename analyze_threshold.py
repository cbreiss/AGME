"""Analyze the mathematical threshold for z/s alternation learning.

With n=1 PYP (current code) + n-scaled unnorm phon:
  /z/ beats new /s/ for saks iff:
    log P_PYP(/z/ | cache) - n × dist(z,s) × prior_scale > log P_PYP(/s/ new)

With n-token PYP + n-scaled unnorm phon:
  /z/ beats new /s/ iff:
    n × log(C_z - d×t_z) - n × dist(z,s) × prior_scale
    > gammaln(n-d) - gammaln(1-d) + log(creation_s)
"""
import math
from scipy.special import gammaln

from agme.features import build_distance_matrix

KLATTBET_IPA = {
    "D": "ð", "T": "θ", "S": "ʃ", "Z": "ʒ", "G": "ŋ",
    "C": "tʃ", "J": "dʒ", "I": "ɪ", "E": "ɛ", "U": "ʊ",
    "O": "ɔɪ", "W": "aʊ", "Y": "aɪ", "R": "ɝ", "@": "æ", "^": "ʌ",
}

# Full Brent alphabet
from collections import Counter
with open('data/words_train.txt') as f:
    words = [l.strip() for l in f if l.strip()]
alphabet = sorted(set(''.join(words)))
print(f"Alphabet ({len(alphabet)}): {alphabet}")

dist = build_distance_matrix(alphabet, ipa_map=KLATTBET_IPA)

# Key distances for the alternation
pairs = [('z','s'), ('z','I'), ('z','t'), ('d','t'), ('d','I')]
print("\nKey panphon distances:")
for a, b in pairs:
    d = dist.get((a,b), None)
    print(f"  dist({a},{b}) = {d:.4f}" if d is not None else f"  dist({a},{b}) = NOT FOUND")

# Threshold analysis for saks (n=41), h@ndz already in cache
n = 41            # tokens of saks
prior_scale = 2.25
d_pyp = 0.5      # PYP discount
alpha = 1.0      # PYP concentration
C_z = 800        # approx suffix /z/ count after processing most -z types
t_z = 15         # approx number of /z/ tables
C_s_new = 0      # /s/ is new for this experiment
N = 1500         # approx total suffix customers
T = 50           # approx total suffix tables
p0_s = 0.5 * (1 - 0.5)  # geometric prior for 1-char suffix /s/ (rough)

print(f"\n--- Threshold analysis for saks (n={n}) ---")
print(f"Assumed: C_z={C_z}, t_z={t_z}, N={N}, T={T}, p0_s≈{p0_s:.4f}")

dist_zs = dist.get(('z','s'), 0.1)  # fallback if not found
print(f"dist(z,s) = {dist_zs:.4f}")

# n=1 PYP threshold
# /z/ wins iff: log(C_z - d*t_z) - n × dist_zs × prior_scale > log((alpha + d*T) * p0_s)
pyp_advantage_n1 = math.log(max(C_z - d_pyp * t_z, 1e-10)) - math.log(N + alpha)
new_s_score_n1 = math.log(max((alpha + d_pyp * T) * p0_s, 1e-300)) - math.log(N + alpha)
phon_cost_per_token = dist_zs * prior_scale
required_phon_budget = pyp_advantage_n1 - new_s_score_n1
threshold_ps_n1 = required_phon_budget / n

print(f"\n  [n=1 PYP formula]")
print(f"  /z/ PYP score:    {pyp_advantage_n1:.3f}")
print(f"  /s/ new score:    {new_s_score_n1:.3f}")
print(f"  PYP advantage:    {required_phon_budget:.3f} nats")
print(f"  Phon cost/token:  dist(z,s) × prior_scale = {dist_zs:.4f} × {prior_scale} = {phon_cost_per_token:.4f}")
print(f"  Total phon cost:  n × {phon_cost_per_token:.4f} = {n * phon_cost_per_token:.3f} nats")
print(f"  → /z/ wins iff: prior_scale < {threshold_ps_n1:.4f}")
print(f"  → At prior_scale={prior_scale}: {'WINS (/z/ preferred)' if n * phon_cost_per_token < required_phon_budget else 'LOSES (/s/ preferred)'}")

# n-token PYP threshold
log_num_z = n * math.log(max(C_z - d_pyp * t_z, 1e-10))
log_denom = n * math.log(max(N + alpha, 1e-300))
log_num_s_new = (float(gammaln(n - d_pyp) - gammaln(1 - d_pyp))
                 + math.log(max((alpha + d_pyp * T) * p0_s, 1e-300)))
phon_cost_z = n * dist_zs * prior_scale

print(f"\n  [n-token PYP formula (morpheme_log_prob_n)]")
print(f"  /z/ PYP numerator:  n × log(C_z - d×t_z) = {log_num_z:.1f}")
print(f"  /s/ new numerator:  gammaln({n}-{d_pyp}) - gammaln({1-d_pyp}) + log(creation)")
print(f"                    = {float(gammaln(n-d_pyp)-gammaln(1-d_pyp)):.1f} + {math.log(max((alpha+d_pyp*T)*p0_s,1e-300)):.1f} = {log_num_s_new:.1f}")
print(f"  PYP advantage for /z/: {log_num_z - log_num_s_new:.1f} nats")
print(f"  Phon cost for /z/: n × dist(z,s) × prior_scale = {phon_cost_z:.1f} nats")
net = (log_num_z - log_num_s_new) - phon_cost_z
print(f"  Net advantage for /z/: {net:.1f} nats → {'WINS' if net > 0 else 'LOSES'}")
threshold_ps_n_token = (log_num_z - log_num_s_new) / (n * dist_zs) if dist_zs > 0 else float('inf')
print(f"  → /z/ wins iff: prior_scale < {threshold_ps_n_token:.2f}")

# Also check: could /D@t/ (C=1854) beat /sak/ (n=44) as stem UR?
print("\n--- Collapse check: could /D@t/ beat /sak/ as stem UR? ---")
n_sak = 44
C_dat = 1854
t_dat = 10  # rough table estimate
N_stem = 50000  # total stem customers
T_stem = 200
dist_dat_sak = (dist.get(('D','s'), 1.0) + dist.get(('@','a'), 1.0) + dist.get(('t','k'), 1.0))
print(f"dist(D,s)={dist.get(('D','s'),None):.4f}, dist(@,a)={dist.get(('@','a'),None):.4f}, dist(t,k)={dist.get(('t','k'),None):.4f}")
print(f"Sum of distances D@t→sak: {dist_dat_sak:.4f}")

# With n-token PYP
log_num_dat = n_sak * math.log(max(C_dat - d_pyp * t_dat, 1e-10))
p0_sak = 0.5 * (1-0.5)**2  # geometric prior for 3-char stem (rough)
log_num_sak_new = (float(gammaln(n_sak - d_pyp) - gammaln(1 - d_pyp))
                   + math.log(max((alpha + d_pyp * T_stem) * p0_sak, 1e-300)))
phon_cost_dat = n_sak * dist_dat_sak * prior_scale
net_dat = (log_num_dat - log_num_sak_new) - phon_cost_dat

print(f"\n  [n-token PYP formula]")
print(f"  /D@t/ PYP numerator: {log_num_dat:.1f}")
print(f"  /sak/ new numerator: {log_num_sak_new:.1f}")
print(f"  PYP advantage for /D@t/: {log_num_dat - log_num_sak_new:.1f} nats")
print(f"  Phon cost for /D@t/: {phon_cost_dat:.1f} nats")
print(f"  Net advantage: {net_dat:.1f} → {'COLLAPSE RISK' if net_dat > 0 else 'OK (faithful wins)'}")
