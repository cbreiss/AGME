"""Pitman-Yor Process (PYP) cache for morpheme lexicons.

One cache is created per morpheme class (stem, suffix, prefix, ...) by
MorphologicalGrammar.  The PYP induces a power-law frequency distribution
over morpheme types, matching the Zipfian shape of real lexicons.

Data structure
--------------
The primary structure is:
    _table_sizes: dict[str, list[int]]
where _table_sizes[word] = [n_customers_at_table_0, n_customers_at_table_1, ...]
Each "table" in the Chinese Restaurant Process represents one token-level draw
from the base distribution G₀.  Multiple tables per word allow the PYP to
model the rich-get-richer effect with power-law type frequencies.

Key design choice: table assignments are tracked explicitly so that remove()
is exact.  The naive alternative (heuristic t/n removal) is statistically
incorrect for Gibbs sampling.

References
----------
Johnson et al. (NeurIPS 2006); Goldwater et al. (Cognition 2009).
"""

from __future__ import annotations

import random


class PitmanYorCache:
    """Pitman-Yor Process cache.

    Parameters
    ----------
    discount : float
        Discount parameter d ∈ [0, 1).
    concentration : float
        Concentration parameter α > -d.
    """

    def __init__(self, discount: float = 0.5, concentration: float = 1.0) -> None:
        if not (0.0 <= discount < 1.0):
            raise ValueError(f"discount must be in [0, 1), got {discount}")
        if concentration <= -discount:
            raise ValueError(f"concentration must be > -discount")
        self.discount = discount
        self.concentration = concentration

        # Primary data structure: {word: [customers_at_table_0, customers_at_table_1, ...]}
        # Each element is the count of customers seated at that table.
        self._table_sizes: dict[str, list[int]] = {}
        self._total_tables: int = 0
        self._total_customers: int = 0
        self._lexicon_dirty: bool = True
        self._lexicon_cache: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Derived counts (computed from _table_sizes)
    # ------------------------------------------------------------------

    def _n(self, word: str) -> int:
        """Total customers for word."""
        sizes = self._table_sizes.get(word)
        return sum(sizes) if sizes else 0

    def _t(self, word: str) -> int:
        """Total tables for word."""
        sizes = self._table_sizes.get(word)
        return len(sizes) if sizes else 0

    # ------------------------------------------------------------------
    # Core CRP operations
    # ------------------------------------------------------------------

    def predictive_score(self, word: str, base_prob: float) -> float:
        """P(word | cache, G₀=base_prob) under the PYP predictive.

        Implements the standard PYP predictive:
          P(word) = [max(n_word - d·t_word, 0) + (α + d·T) · base_prob] / (N + α)
        """
        N = self._total_customers
        T = self._total_tables
        d = self.discount
        alpha = self.concentration

        n = self._n(word)
        t = self._t(word)

        numerator = max(n - d * t, 0.0) + (alpha + d * T) * base_prob
        denominator = N + alpha
        return max(numerator / denominator, 1e-300)

    def add(self, word: str, base_prob: float) -> None:
        """Seat one customer for *word*, possibly opening a new table."""
        d = self.discount
        alpha = self.concentration
        T = self._total_tables

        if word not in self._table_sizes or not self._table_sizes[word]:
            # First customer: open a new table
            self._table_sizes[word] = [1]
            self._total_tables += 1
        else:
            sizes = self._table_sizes[word]
            n = sum(sizes)
            t = len(sizes)

            # Weight for sitting at each existing table k: max(sizes[k] - d, 0)
            # Weight for opening a new table: (alpha + d*T) * base_prob
            new_table_weight = (alpha + d * T) * base_prob
            existing_weights = [max(s - d, 0.0) for s in sizes]
            total_weight = new_table_weight + sum(existing_weights)

            if total_weight <= 0 or random.random() < new_table_weight / total_weight:
                # Open a new table
                sizes.append(1)
                self._total_tables += 1
            else:
                # Sit at an existing table, sampled proportional to existing_weights
                r = random.random() * sum(existing_weights)
                cumulative = 0.0
                for k, ew in enumerate(existing_weights):
                    cumulative += ew
                    if r < cumulative:
                        sizes[k] += 1
                        break
                else:
                    sizes[-1] += 1  # fallback: last table

        self._total_customers += 1
        self._lexicon_dirty = True

    def remove(self, word: str) -> None:
        """Remove one customer for *word*, exactly tracking table assignment.

        Samples which table the departing customer was at, proportional
        to table size (uniform customer selection), then decrements.
        Closes the table if it becomes empty.
        """
        if word not in self._table_sizes or not self._table_sizes[word]:
            raise ValueError(f"Cannot remove '{word}': no customers present")

        sizes = self._table_sizes[word]
        n = sum(sizes)

        # Sample which table the departing customer belongs to,
        # proportional to table size (each customer equally likely to leave)
        r = random.random() * n
        cumulative = 0.0
        chosen_k = len(sizes) - 1  # fallback
        for k, s in enumerate(sizes):
            cumulative += s
            if r < cumulative:
                chosen_k = k
                break

        sizes[chosen_k] -= 1
        self._total_customers -= 1

        if sizes[chosen_k] == 0:
            # Close the empty table
            sizes.pop(chosen_k)
            self._total_tables -= 1

        if not sizes:
            del self._table_sizes[word]

        self._lexicon_dirty = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def lexicon(self) -> dict[str, int]:
        """Return {word: customer_count} for all active words."""
        if self._lexicon_dirty or self._lexicon_cache is None:
            self._lexicon_cache = {
                w: sum(sizes)
                for w, sizes in self._table_sizes.items()
                if sizes
            }
            self._lexicon_dirty = False
        return dict(self._lexicon_cache)

    @property
    def total_customers(self) -> int:
        return self._total_customers

    @property
    def total_tables(self) -> int:
        return self._total_tables
