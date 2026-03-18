"""*MAP constraint definitions (Zuraw 2013).

A StarMapConstraint(x, y) penalises every UR→SR correspondence where
segment *x* maps to segment *y*.  Weights are initialised from panphon
featural distance (P-map prior) and learned via MaxEnt gradient ascent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agme.utils import levenshtein_alignment


@dataclass(frozen=True)
class StarMapConstraint:
    """*MAP(x, y[, left_ctx, right_ctx]) constraint.

    Parameters
    ----------
    x : str
        UR segment (input side of the correspondence).
    y : str
        SR segment (output side of the correspondence).
    left_ctx : str | None
        Required left-context segment in the SR, or None for context-free.
    right_ctx : str | None
        Required right-context segment in the SR, or None for context-free.
    prior_weight : float
        Initial weight / prior σ parameter (set from panphon distance at
        model init time; not part of equality/hashing).
    """

    x: str
    y: str
    left_ctx: str | None = None
    right_ctx: str | None = None
    prior_weight: float = field(default=1.0, compare=False, hash=False)

    # ------------------------------------------------------------------
    # Violation counting
    # ------------------------------------------------------------------

    def violations(self, ur: str, sr: str) -> float:
        """Count the number of *MAP(x, y) violations in the UR→SR mapping.

        Uses Levenshtein alignment so insertions and deletions are handled:
        - Deletion (x, None): counted if self.y is None (future MAX hook)
        - Insertion (None, y): counted if self.x is None (future DEP hook)
        - Substitution / match (x, y): counted when both match
        Context conditions (left_ctx, right_ctx) are checked in the SR.
        """
        alignment = levenshtein_alignment(ur, sr)
        count = 0.0
        sr_pos = 0  # tracks position in SR for context lookup
        for ur_seg, sr_seg in alignment:
            if ur_seg == self.x and sr_seg == self.y:
                # Check context in SR if specified
                if self._context_matches(sr, sr_pos):
                    count += 1.0
            if sr_seg is not None:
                sr_pos += 1
        return count

    def _context_matches(self, sr: str, sr_pos: int) -> bool:
        if self.left_ctx is not None:
            if sr_pos == 0 or sr[sr_pos - 1] != self.left_ctx:
                return False
        if self.right_ctx is not None:
            if sr_pos + 1 >= len(sr) or sr[sr_pos + 1] != self.right_ctx:
                return False
        return True

    def __repr__(self) -> str:
        ctx = ""
        if self.left_ctx or self.right_ctx:
            ctx = f" / {self.left_ctx or ''}___{self.right_ctx or ''}"
        return f"*MAP({self.x},{self.y}{ctx})"


def build_star_map_constraints(
    alphabet: list[str],
    distance_matrix: dict[tuple[str, str], float],
) -> list[StarMapConstraint]:
    """Build the full *MAP constraint set for the given alphabet.

    One constraint per ordered (x, y) pair with x ≠ y.
    prior_weight is set to the panphon distance, so perceptually distant
    alternations start with a stronger penalty.
    """
    constraints = []
    for x in alphabet:
        for y in alphabet:
            if x != y:
                d = distance_matrix.get((x, y), 1.0)
                constraints.append(StarMapConstraint(x=x, y=y, prior_weight=d))
    return constraints
