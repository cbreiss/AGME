"""Pitman-Yor Process cache for morpheme lexicons.

One PYP cache per morpheme class (stem, suffix, prefix, ...).
Implements the Chinese Restaurant Process predictive distribution.
"""

from __future__ import annotations


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

        # {word: number of tables}
        self._table_counts: dict[str, int] = {}
        # {word: number of customers}
        self._customer_counts: dict[str, int] = {}
        self._total_tables: int = 0
        self._total_customers: int = 0
        self._lexicon_dirty: bool = True
        self._lexicon_cache: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Core CRP operations
    # ------------------------------------------------------------------

    def predictive_score(self, word: str, base_prob: float) -> float:
        """P(word | cache, G₀=base_prob) under the PYP predictive.

        Implements the standard PYP predictive:
          P(word) = [max(n_word - d·t_word, 0) + (α + d·T) · base_prob] / (N + α)

        where:
          n_word = customers at *word*
          t_word = tables for *word*
          T = total tables
          N = total customers
        """
        N = self._total_customers
        T = self._total_tables
        d = self.discount
        alpha = self.concentration

        n = self._customer_counts.get(word, 0)
        t = self._table_counts.get(word, 0)

        numerator = max(n - d * t, 0.0) + (alpha + d * T) * base_prob
        denominator = N + alpha
        return max(numerator / denominator, 1e-300)

    def add(self, word: str, base_prob: float) -> None:
        """Seat one customer for *word*, possibly opening a new table."""
        n = self._customer_counts.get(word, 0)
        t = self._table_counts.get(word, 0)
        N = self._total_customers
        T = self._total_tables
        d = self.discount
        alpha = self.concentration

        self._customer_counts[word] = n + 1
        self._total_customers += 1

        if n == 0:
            # First customer for this word: must open a new table
            self._table_counts[word] = 1
            self._total_tables += 1
        else:
            # Open a new table with probability proportional to
            # (alpha + d·T) · base_prob / (n + (alpha + d·T) · base_prob)
            new_table_weight = (alpha + d * T) * base_prob
            existing_weight = max(n - d * t, 0.0)
            denom = new_table_weight + existing_weight
            if denom > 0 and (new_table_weight / denom) > __import__('random').random():
                self._table_counts[word] = t + 1
                self._total_tables += 1

        self._lexicon_dirty = True

    def remove(self, word: str) -> None:
        """Remove one customer for *word*, possibly closing a table."""
        n = self._customer_counts.get(word, 0)
        t = self._table_counts.get(word, 0)
        if n <= 0:
            raise ValueError(f"Cannot remove '{word}': no customers present")

        self._customer_counts[word] = n - 1
        self._total_customers -= 1

        if n == 1:
            # Last customer: remove all tables
            del self._customer_counts[word]
            self._total_tables -= t
            del self._table_counts[word]
        elif t > 1:
            # Probabilistically remove one table
            prob_remove_table = t / n  # simplified heuristic
            if __import__('random').random() < prob_remove_table:
                self._table_counts[word] = t - 1
                self._total_tables -= 1

        self._lexicon_dirty = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def lexicon(self) -> dict[str, int]:
        """Return {word: customer_count} for all active words."""
        if self._lexicon_dirty or self._lexicon_cache is None:
            self._lexicon_cache = {
                w: c for w, c in self._customer_counts.items() if c > 0
            }
            self._lexicon_dirty = False
        return dict(self._lexicon_cache)

    @property
    def total_customers(self) -> int:
        return self._total_customers

    @property
    def total_tables(self) -> int:
        return self._total_tables
