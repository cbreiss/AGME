"""Analyze the z/s equilibrium: can /z/ beat /s/ for voiceless-stem words?

Uses actual corpus-derived C_z and C_s counts.
"""
import math
from scipy.special import gammaln
from agme.features import build_distance_matrix

KLATTBET_IPA = {
    "D": "ð", "T": "θ", "S": "ʃ", "Z": "ʒ", "G": "ŋ",
    "C": "tʃ", "J": "dʒ", "I": "ɪ", "E": "ɛ", "U": "ʊ",
    "O": "ɔɪ", "W": "aʊ", "Y": "aɪ", "R": "ɝ", "@": "æ", "^": "ʌ",
}

from collections import Counter
with open('data/words_paradigm.txt') as f:
    words = [l.strip() for l in f if l.strip()]
alphabet = sorted(set(''.join(words)))
dist = build_distance_matrix(alphabet, ipa_map=KLATTBET_IPA)

# Approximate suffix token counts from paradigm corpus
# /z/-suffix types and their token counts
z_types = {
    'h@ndz': 106, 'TIGz': 43, 'lEgz': 23, 'mamiz': 123, 'bebiz': 58,
    'morgInz': 38, 'mcrgiz': 26, 'fIGgRz': 25, 'Suz': 34, 'huz': 64,
    'tOz': 49, 'Doz': 141, 'yuz': 21, 'werz': 221, 'Derz': 219,
    'noz': 73, 'goz': 63, 'k^mz': 62, 'hirz': 160, 'hiz': 174,
    'h@z': 83, 'd^z': 88, 'nOz': 34, 'Diz': 105,
}
# /s/-suffix types
s_types = {'saks': 41, 'bUks': 29, 'hWs': 60}
# /Iz/-suffix types
Iz_types = {'kIsIz': 44}

C_z = sum(z_types.values())
C_s = sum(s_types.values())
C_Iz = sum(Iz_types.values())
print(f"Suffix token counts:")
print(f"  /z/ cache: {C_z} tokens from {len(z_types)} types")
print(f"  /s/ cache: {C_s} tokens from {len(s_types)} types")
print(f"  /Iz/ cache: {C_Iz} tokens from {len(Iz_types)} types")
print(f"  Ratio C_z/C_s = {C_z/C_s:.1f}")

d_pyp = 0.5
alpha = 1.0
# Rough table counts (PYP with d=0.5: expected tables ≈ 2*sqrt(C))
t_z = int(2 * math.sqrt(C_z))
t_s = int(2 * math.sqrt(C_s))
N_total = C_z + C_s + C_Iz
T_total = t_z + t_s + 4

dist_zs = dist.get(('z','s'), 0.0417)
prior_scale = 2.25

print(f"\ndist(z,s) = {dist_zs:.4f}, *MAP(z,s) initial weight = {dist_zs * prior_scale:.4f}")

print("\n" + "="*60)
print("Scenario: scoring suffix UR for saks (n=41, SR suffix='s')")
print("="*60)

n = 41  # saks token count
# CASE 1: n=1 PYP
lp_z_n1 = math.log(C_z - d_pyp * t_z) - math.log(N_total + alpha)
lp_s_n1 = math.log(C_s - d_pyp * t_s) - math.log(N_total + alpha)
phon_cost_n1 = n * dist_zs * prior_scale
adv_n1 = lp_z_n1 - lp_s_n1
print(f"\n[n=1 PYP]")
print(f"  /z/ PYP: {lp_z_n1:.3f},  /s/ PYP: {lp_s_n1:.3f}")
print(f"  PYP advantage for /z/ over /s/: {adv_n1:.3f} nats")
print(f"  Phonological cost for /z/→[s]: {phon_cost_n1:.3f} nats")
print(f"  Net: {adv_n1 - phon_cost_n1:.3f} → {'WINS' if adv_n1 > phon_cost_n1 else 'LOSES'}")

# CASE 2: n-token PYP (morpheme_log_prob_n)
log_num_z = n * math.log(C_z - d_pyp * t_z)
log_num_s = n * math.log(C_s - d_pyp * t_s)
log_denom = n * math.log(N_total + alpha)
phon_cost_n = n * dist_zs * prior_scale
adv_n = log_num_z - log_num_s
print(f"\n[n-token PYP]")
print(f"  /z/ numerator: {log_num_z:.1f},  /s/ numerator: {log_num_s:.1f}")
print(f"  PYP advantage for /z/ over /s/: {adv_n:.1f} nats")
print(f"  Phonological cost for /z/→[s]: {phon_cost_n:.3f} nats")
print(f"  Net: {adv_n - phon_cost_n:.1f} → {'WINS' if adv_n > phon_cost_n else 'LOSES'}")

print("\n" + "="*60)
print("Collapse check: /D@t/ (C=1854) vs /sak/ (new) for stem 'sak'")
print("="*60)
n_sak = 44  # saks=41 + sak=3 (approximate)
# Actually, let's use sak's own count: 44
C_dat = 1854  # D@t stem count
t_dat = int(2 * math.sqrt(C_dat))
N_stem = 63000  # rough total stem customers
T_stem = int(2 * math.sqrt(N_stem))
p0_sak = 0.5 * (0.5)**2  # geometric length prior for 3-char word, rough
alpha_stem = 1.0
d_stem = 0.5
# dist for alignment D@t → sak (both 3 chars, substitutions)
dist_Ds = dist.get(('D','s'), 1.0)
dist_at_a = dist.get(('@','a'), 1.0)
dist_tk = dist.get(('t','k'), 1.0)
H_dat_sak = dist_Ds + dist_at_a + dist_tk
print(f"\ndist(D,s)={dist_Ds:.4f} + dist(@,a)={dist_at_a:.4f} + dist(t,k)={dist_tk:.4f} = {H_dat_sak:.4f}")

# n=1 PYP for stems
lp_dat_n1_stem = math.log(C_dat - d_stem * t_dat) - math.log(N_stem + alpha_stem)
lp_sak_new_n1 = math.log(max((alpha_stem + d_stem * T_stem) * p0_sak, 1e-300)) - math.log(N_stem + alpha_stem)
phon_dat_n1 = n_sak * H_dat_sak * prior_scale
print(f"\n[n=1 PYP for stems]")
print(f"  /D@t/ PYP: {lp_dat_n1_stem:.3f},  /sak/ new: {lp_sak_new_n1:.3f}")
print(f"  Advantage for /D@t/: {lp_dat_n1_stem - lp_sak_new_n1:.3f} nats")
print(f"  Phon cost for /D@t/→[sak]: {phon_dat_n1:.3f} nats")
print(f"  Net: {(lp_dat_n1_stem - lp_sak_new_n1) - phon_dat_n1:.3f} → {'COLLAPSE RISK' if (lp_dat_n1_stem-lp_sak_new_n1) > phon_dat_n1 else 'OK'}")

# n-token PYP for stems
log_num_dat_n = n_sak * math.log(C_dat - d_stem * t_dat)
log_num_sak_n = (float(gammaln(n_sak - d_stem) - gammaln(1 - d_stem))
                 + math.log(max((alpha_stem + d_stem * T_stem) * p0_sak, 1e-300)))
phon_dat_n = n_sak * H_dat_sak * prior_scale
print(f"\n[n-token PYP for stems]")
print(f"  /D@t/ numerator: {log_num_dat_n:.1f}")
print(f"  /sak/ new numerator: {log_num_sak_n:.1f}")
print(f"  PYP advantage for /D@t/: {log_num_dat_n - log_num_sak_n:.1f} nats")
print(f"  Phon cost for /D@t/→[sak]: {phon_dat_n:.1f} nats")
net_collapse = (log_num_dat_n - log_num_sak_n) - phon_dat_n
print(f"  Net: {net_collapse:.1f} → {'COLLAPSE RISK' if net_collapse > 0 else 'OK'}")

# Proposal weight check
w_dat_for_sak = (1 / (H_dat_sak + 0.5)) * math.sqrt(C_dat)
w_sak_faithful = 100  # faithful_weight
print(f"\n  Proposal weights: /sak/(faithful)={w_sak_faithful:.1f}, /D@t/(lexicon)={w_dat_for_sak:.1f}")
print(f"  → /D@t/ {'WOULD' if w_dat_for_sak > 1 else 'would NOT'} be proposed as candidate for 'sak'")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
n=1 PYP (current code):
  z/s alternation: {'POSSIBLE' if adv_n1 > phon_cost_n1 else 'BLOCKED'}
    /z/ advantage = {adv_n1:.3f} nats < phon cost = {phon_cost_n1:.3f} nats

  Root cause: C_z/C_s = {C_z/C_s:.1f} but needs exp({phon_cost_n1:.1f}) = {math.exp(phon_cost_n1):.0f}× ratio.
  The corpus naturally equilibrates z/s counts ∼ {C_z/C_s:.1f}:1, far short.

n-token PYP (morpheme_log_prob_n):
  z/s alternation: {'POSSIBLE' if adv_n > phon_cost_n else 'BLOCKED'}
    /z/ advantage = {adv_n:.1f} nats >> phon cost = {phon_cost_n:.3f} nats
  Stem collapse risk: {'YES — D@t beats sak by ' + str(round(net_collapse,1)) + ' nats' if net_collapse > 0 else 'NO'}
""")
