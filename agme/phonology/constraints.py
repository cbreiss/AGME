"""Phonological constraint definitions: the *MAP family.

All faithfulness constraints belong to a single family, StarMapConstraint(x, y),
parameterised by two sides of a correspondence:

    x : str | None   —  input (UR) segment; None = ∅ (no UR correspondent)
    y : str | None   —  output (SR) segment; None = ∅ (no SR correspondent)

This covers three linguistically distinct cases:

    *MAP(a, b)     x=str, y=str   substitution penalty  (e.g. /d/ → [t])
    *MAP(∅, b)     x=None, y=str  epenthesis penalty    (e.g. ∅ → [ɪ])
    *MAP(a, ∅)     x=str, y=None  deletion penalty      (e.g. /z/ → ∅)

Context sensitivity
-------------------
All three variants inherit the same left_ctx / right_ctx mechanism.  For
example, *MAP(∅, ɪ, left_ctx=ʃ, right_ctx=z) penalises inserting ɪ
specifically in the environment ʃ___z — capturing the English sibilant
epenthesis context in one constraint.

Weight initialisation
---------------------
Substitution weights (*MAP(a,b) with a,b≠None) are initialised from the
panphon featural distance between a and b (P-map prior: perceptually
similar alternations start cheap).

Insertion / deletion weights (*MAP(∅,b) and *MAP(a,∅)) are initialised
with a uniform prior (no P-map analog for insertion/deletion cost).

All weights are learned jointly via MaxEnt (L-BFGS-B) in MaxEntPhonology.

Building constraint sets
------------------------
build_star_map_constraints(alphabet, distance_matrix)
    → substitution constraints: one *MAP(a,b) per ordered pair a≠b

build_insertion_deletion_constraints(alphabet)
    → context-free insertion/deletion: one *MAP(∅,b) + *MAP(a,∅) per segment

build_all_constraints(alphabet, distance_matrix)
    → full set: substitution + insertion + deletion  [use this in Model.fit()]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agme.utils import levenshtein_alignment

# Sentinel displayed in __repr__ for the null (∅) side of a correspondence
_NULL = "∅"


@dataclass(frozen=True)
class StarMapConstraint:
    """*MAP(x, y) constraint — unified substitution, insertion, and deletion.

    Parameters
    ----------
    x : str | None
        UR segment (input side).  None = ∅ (epenthesis constraint).
    y : str | None
        SR segment (output side).  None = ∅ (deletion constraint).
    left_ctx : str | None
        Required left-context segment in the SR, or None for context-free.
        For deletion (y=None) the context is checked at the deletion site
        in the SR (the position just before the deleted segment would have
        appeared).
    right_ctx : str | None
        Required right-context segment in the SR, or None for context-free.
    prior_weight : float
        Initial weight / prior σ parameter.
        Substitution constraints: set from panphon distance at model init.
        Insertion/deletion constraints: uniform default (1.0).
        Not part of equality/hashing.

    Notes
    -----
    x and y cannot both be None — that would be a null→null correspondence.
    """

    x: str | None
    y: str | None
    left_ctx: str | None = None
    right_ctx: str | None = None
    prior_weight: float = field(default=1.0, compare=False, hash=False)

    # ------------------------------------------------------------------
    # Violation counting
    # ------------------------------------------------------------------

    def violations(self, ur: str, sr: str) -> float:
        """Count violations of *MAP(x, y) in the Levenshtein alignment of ur→sr.

        Computes the alignment internally.  For bulk scoring of many constraints
        against the same (ur, sr) pair, prefer count_from_alignment() so the
        alignment is computed only once and shared across all constraints.

        The alignment produces (ur_seg, sr_seg) pairs where None indicates a gap:
          *MAP(a, b)   — match when ur_seg==a and sr_seg==b  (substitution)
          *MAP(∅, b)   — match when ur_seg is None and sr_seg==b  (epenthesis)
          *MAP(a, ∅)   — match when ur_seg==a and sr_seg is None  (deletion)
        """
        return self.count_from_alignment(levenshtein_alignment(ur, sr), sr)

    def count_from_alignment(
        self, alignment: list[tuple[str | None, str | None]], sr: str
    ) -> float:
        """Count violations given a pre-computed Levenshtein alignment.

        Separating alignment from counting allows violation_vector() in
        MaxEntPhonology to compute levenshtein_alignment(ur, sr) exactly once
        per (ur, sr) pair and share it across all 300+ constraints, avoiding
        the dominant O(N_constraints) alignment bottleneck.

        Parameters
        ----------
        alignment : list of (ur_seg, sr_seg) pairs from levenshtein_alignment()
        sr : the surface form string, needed for context checking
        """
        count = 0.0
        sr_pos = 0  # position in SR; updated only when an SR segment is consumed
        for ur_seg, sr_seg in alignment:
            if ur_seg == self.x and sr_seg == self.y:
                # Both sides match (handles substitution and exact None matches)
                if self._context_matches(sr, sr_pos):
                    count += 1.0
            # Advance SR position whenever an SR segment is present
            if sr_seg is not None:
                sr_pos += 1
        return count

    def _context_matches(self, sr: str, sr_pos: int) -> bool:
        """Check whether the left and right SR contexts are satisfied at sr_pos.

        Context semantics differ by constraint type:

        Substitution / insertion  (y is not None)
            sr_pos is the index of the current output segment in SR.
            left_ctx:  sr[sr_pos - 1]  (character before current segment)
            right_ctx: sr[sr_pos + 1]  (character after current segment)

        Deletion  (y is None)
            Deletions do not advance sr_pos; sr_pos points to the NEXT SR
            character after the deletion site.
            left_ctx:  sr[sr_pos - 1]  (same — character before deletion site)
            right_ctx: sr[sr_pos]      (next SR character after deletion site)
        """
        if self.left_ctx is not None:
            if sr_pos == 0 or sr[sr_pos - 1] != self.left_ctx:
                return False
        if self.right_ctx is not None:
            if self.y is None:
                # Deletion: right context is the next actual SR segment
                if sr_pos >= len(sr) or sr[sr_pos] != self.right_ctx:
                    return False
            else:
                # Substitution / insertion: right context is after current segment
                if sr_pos + 1 >= len(sr) or sr[sr_pos + 1] != self.right_ctx:
                    return False
        return True

    def __repr__(self) -> str:
        x_str = self.x if self.x is not None else _NULL
        y_str = self.y if self.y is not None else _NULL
        ctx = ""
        if self.left_ctx or self.right_ctx:
            ctx = f" / {self.left_ctx or ''}___{self.right_ctx or ''}"
        return f"*MAP({x_str},{y_str}{ctx})"


# ---------------------------------------------------------------------------
# Constraint set builders
# ---------------------------------------------------------------------------

def build_star_map_constraints(
    alphabet: list[str],
    distance_matrix: dict[tuple[str, str], float],
    prior_scale: float = 1.0,
) -> list[StarMapConstraint]:
    """Build the substitution *MAP constraint set for the given alphabet.

    One constraint per ordered (x, y) pair with x ≠ y, both from the alphabet.
    prior_weight is set to panphon_distance * prior_scale so that perceptually
    distant alternations start with a stronger penalty (P-map prior).
    Increasing prior_scale raises all initial weights uniformly and widens the
    half-normal regularisation, making it harder for L-BFGS-B to collapse them.
    """
    constraints = []
    for x in alphabet:
        for y in alphabet:
            if x != y:
                d = distance_matrix.get((x, y), 1.0) * prior_scale
                constraints.append(StarMapConstraint(x=x, y=y, prior_weight=d))
    return constraints


def build_insertion_deletion_constraints(
    alphabet: list[str],
    epenthesis_prior: float = 1.0,
    deletion_prior: float = 1.0,
) -> list[StarMapConstraint]:
    """Build context-free insertion and deletion constraints for the alphabet.

    For each segment s in the alphabet, builds:
      *MAP(∅, s)   — penalises inserting s  (epenthesis); weight = epenthesis_prior
      *MAP(s, ∅)   — penalises deleting s   (deletion);   weight = deletion_prior

    Separate priors allow the model to treat epenthesis and deletion as
    independently costly (e.g. epenthesis_prior=2.0, deletion_prior=1.0 to
    strongly disfavour segments appearing in the SR with no UR correspondent).

    Context-sensitive variants (e.g. *MAP(∅, ɪ, left_ctx=ʃ, right_ctx=z))
    can be added manually or via a future extension that discovers contexts
    from data — the constraint machinery already supports them.
    """
    constraints = []
    for seg in alphabet:
        # Epenthesis: ∅ → seg
        constraints.append(
            StarMapConstraint(x=None, y=seg, prior_weight=epenthesis_prior)
        )
        # Deletion: seg → ∅
        constraints.append(
            StarMapConstraint(x=seg, y=None, prior_weight=deletion_prior)
        )
    return constraints


def build_all_constraints(
    alphabet: list[str],
    distance_matrix: dict[tuple[str, str], float],
    prior_scale: float = 1.0,
    epenthesis_prior: float = 1.0,
    deletion_prior: float = 1.0,
) -> list[StarMapConstraint]:
    """Build the full constraint set: substitution + insertion + deletion.

    Returns *MAP(a,b) for all segment pairs plus *MAP(∅,b) and *MAP(a,∅) for
    all segments.  All three groups are trained jointly via MaxEnt (L-BFGS-B).

    Parameters
    ----------
    alphabet : list[str]
        The phoneme inventory.
    distance_matrix : dict[tuple[str,str], float]
        Pairwise panphon distances for P-map prior on substitution constraints.
    prior_scale : float
        Global multiplier on all substitution prior_weights (panphon distances).
        Also applied to epenthesis_prior and deletion_prior so that the entire
        constraint set is raised uniformly.  Increasing this widens the
        half-normal regularisation and raises initial weights, counteracting the
        PYP mode-collapse attractor.  Default 1.0 (no scaling).
    epenthesis_prior : float
        Base prior weight for *MAP(∅,y) epenthesis constraints before scaling.
    deletion_prior : float
        Base prior weight for *MAP(x,∅) deletion constraints before scaling.

    Returns
    -------
    list[StarMapConstraint]
        Substitution constraints first, then insertion/deletion pairs per segment.
    """
    substitution = build_star_map_constraints(alphabet, distance_matrix,
                                              prior_scale=prior_scale)
    ins_del = build_insertion_deletion_constraints(
        alphabet,
        epenthesis_prior=epenthesis_prior * prior_scale,
        deletion_prior=deletion_prior * prior_scale,
    )
    return substitution + ins_del
