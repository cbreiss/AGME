"""Shared numeric and string utilities.

Contents
--------
logsumexp(log_vals)
    Numerically stable log-sum-exp; used everywhere a distribution over
    log-probabilities needs to be normalised.

levenshtein_alignment(s, t)
    Optimal character alignment of two strings (UR vs SR).  The result is
    consumed by StarMapConstraint.violations() to identify which segments
    correspond to which and whether any are deleted/inserted.

random_edit(word, alphabet, rng)
    Single random substitution, insertion, or deletion.  Used by
    candidates_for() and URProposer to generate exploratory UR candidates.
"""

from __future__ import annotations

import numpy as np


def logsumexp(log_vals: list[float]) -> float:
    """Numerically stable log-sum-exp over a list of log-values."""
    if not log_vals:
        return float("-inf")
    arr = np.array(log_vals, dtype=np.float64)
    m = arr.max()
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(arr - m))))


def levenshtein_alignment(s: str, t: str) -> list[tuple[str | None, str | None]]:
    """Return an optimal Levenshtein alignment of strings *s* (UR) and *t* (SR).

    Each element is a pair (s_char, t_char) where None denotes a gap.
    Insertions (None, t_char) and deletions (s_char, None) are included.
    """
    m, n = len(s), len(t)
    # DP cost table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
            )
    # Traceback
    alignment: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if s[i - 1] == t[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                alignment.append((s[i - 1], t[j - 1]))
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((s[i - 1], None))
            i -= 1
        else:
            alignment.append((None, t[j - 1]))
            j -= 1
    alignment.reverse()
    return alignment


def random_edit(word: str, alphabet: list[str], rng: np.random.Generator) -> str:
    """Return a random single-operation edit (sub/ins/del) of *word*."""
    if not word:
        return rng.choice(alphabet)  # type: ignore[return-value]
    op = rng.integers(3)
    pos = int(rng.integers(len(word)))
    if op == 0:  # substitution
        new_char = rng.choice([c for c in alphabet if c != word[pos]] or alphabet)
        return word[:pos] + new_char + word[pos + 1:]
    elif op == 1:  # insertion
        new_char = rng.choice(alphabet)
        return word[:pos] + new_char + word[pos:]
    else:  # deletion
        return word[:pos] + word[pos + 1:]
