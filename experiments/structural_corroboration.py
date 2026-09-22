"""
Gated corroboration: the architectural role for structural_features.py's
similarity signal, decided after scratch_structural_validation.py showed raw
top-ranked structural pairs are dominated by a cluster of unrelated,
permanently-all-zero-feature problems (closed-form/O(1) solutions with no
loop-like structure at all -- unique-paths, valid-boomerang, and friends).
IDF-weighted Dice mathematically scores any two IDENTICAL tag sets 1.0
regardless of how generic those tags are, so this cluster cannot be fixed by
finer bucketing; it's a fundamental consequence of those problems having
nothing for structural features to describe.

Rather than trying to make structural similarity a standalone, universally
discriminating primary ranker (which the evidence says it can't be), it is
used here as a secondary signal that only SPEAKS when it has something real
to say:

  - richness(name) = how many structural tags a problem has. Restricting to
    pairs where BOTH sides clear a minimum richness (empirically >=8 tags,
    per scratch_structural_validation.py: below this the top-200 structural
    pairs score *below* the random-pair baseline on mechanism-embedding
    agreement; at/above it they score well above baseline) turns "worse than
    random" into "meaningfully better than random," and this remains true
    after fixing comprehension/generator-expression blindness (cause 2) --
    the richness gate is doing real work independent of that fix, because it
    is specifically screening out cause-1 (permanently sparse) problems, not
    just problems that happened to be under-featurized.
  - Below that gate, the signal abstains (contributes exactly zero to a
    blended score) rather than contributing a diluted, near-meaningless
    number computed over 0-2 shared generic tags.
  - The gate is symmetric (both sides must be rich) for the corroboration
    bonus, but richness of the QUERY alone is what should drive an honest
    "we have no structural opinion on this query" confidence label -- see
    corroboration_status().

This intentionally does NOT try to fix cause-1 problems to be structurally
distinguishable -- see structural_features.py's module docstring. A query
with no loop-like structure gets no structural corroboration, ever, and that
is the correct behavior, not a gap to close.
"""

import numpy as np
import pandas as pd

RICHNESS_THRESHOLD = 8


def richness(name: str, structural_tags: dict[str, set]) -> int:
    return len(structural_tags.get(name, set()))


def gated_corroboration_matrix(
    structural_sim: pd.DataFrame,
    structural_tags: dict[str, set],
    threshold: int = RICHNESS_THRESHOLD,
) -> pd.DataFrame:
    """Returns a copy of `structural_sim` with every cell zeroed out except
    where both the row and column problem clear the richness threshold. This
    is the "abstain unless both sides have something to say" gate -- safe to
    add directly onto an existing blended score matrix since an abstained
    cell contributes exactly 0, never penalizing a pair for silence."""
    names = list(structural_sim.index)
    rich = np.array([richness(n, structural_tags) >= threshold for n in names])
    mask = np.outer(rich, rich)
    gated = structural_sim.to_numpy() * mask
    return pd.DataFrame(gated, index=structural_sim.index, columns=structural_sim.columns)


def apply_gated_corroboration(
    base_sim: pd.DataFrame,
    structural_sim: pd.DataFrame,
    structural_tags: dict[str, set],
    bonus_weight: float,
    threshold: int = RICHNESS_THRESHOLD,
) -> pd.DataFrame:
    """base_sim + bonus_weight * gated_structural_sim, aligned to base_sim's
    index/columns. This is role 2 (confidence-boost corroboration): a rich
    pair that also agrees structurally gets nudged up in the ranking; a pair
    where either side is feature-sparse is left exactly as the base blend
    ranked it."""
    gated = gated_corroboration_matrix(structural_sim, structural_tags, threshold)
    gated = gated.reindex(index=base_sim.index, columns=base_sim.columns).fillna(0.0)
    return base_sim + bonus_weight * gated


def corroboration_status(
    query: str,
    candidate: str,
    structural_sim: pd.DataFrame,
    structural_tags: dict[str, set],
    threshold: int = RICHNESS_THRESHOLD,
    agree_cutoff: float = 0.5,
) -> str:
    """Role 3/4: a per-recommendation confidence label, independent of
    whether the corroboration bonus fired. Distinguishes "no opinion" from
    "opinion" so a caller can be honest about which case it's in, rather than
    silently treating both as equally-uncorroborated.

      "no_signal"     -- the query itself is feature-sparse (richness below
                          threshold); structural analysis has nothing to say
                          about this query at all, for any candidate. This is
                          the expected, permanent state for closed-form/O(1)
                          queries (role 4: an honest "can't corroborate" label
                          beats a silently-absent bonus that looks identical
                          to "checked and it didn't match").
      "candidate_sparse" -- the query is rich but this specific candidate
                          isn't, so the pair can't be judged either way.
      "corroborated"   -- both rich, and structurally similar (>= agree_cutoff).
      "contradicted"   -- both rich, but structurally dissimilar (< agree_cutoff)
                          -- an active disagreement worth surfacing for audit,
                          not merely an absence of agreement.
    """
    if richness(query, structural_tags) < threshold:
        return "no_signal"
    if richness(candidate, structural_tags) < threshold:
        return "candidate_sparse"
    sim = structural_sim.loc[query, candidate] if (query in structural_sim.index and candidate in structural_sim.columns) else 0.0
    return "corroborated" if sim >= agree_cutoff else "contradicted"
