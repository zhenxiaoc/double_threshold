"""Cross-fitting utilities: deterministic K-fold splits that keep a participant's
rows in a single fold, plus an inner cross-fitting layer for honest pseudo-outcomes
(task sections 8, 20).
"""

from __future__ import annotations

import numpy as np


def make_folds(
    n: int,
    n_folds: int,
    *,
    seed: int,
    groups: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return a list of index arrays, one per fold.

    If ``groups`` is given, all rows sharing a group id land in the same fold, so
    repeated observations from one participant never straddle the train/test split.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n)
        return [np.sort(f) for f in np.array_split(idx, n_folds)]
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    perm = rng.permutation(uniq.size)
    uniq_shuffled = uniq[perm]
    group_fold = {g: (i % n_folds) for i, g in enumerate(uniq_shuffled)}
    fold_of_row = np.array([group_fold[g] for g in groups])
    return [np.where(fold_of_row == k)[0] for k in range(n_folds)]


def fold_assignments(folds: list[np.ndarray], n: int) -> np.ndarray:
    """Inverse map: row index -> fold id."""
    out = np.full(n, -1, dtype=int)
    for k, f in enumerate(folds):
        out[f] = k
    return out


def check_no_group_leak(folds: list[np.ndarray], groups: np.ndarray) -> bool:
    """True iff no group appears in more than one fold."""
    seen: dict = {}
    for k, f in enumerate(folds):
        for g in np.unique(np.asarray(groups)[f]):
            if g in seen and seen[g] != k:
                return False
            seen[g] = k
    return True


def child_seed(base_seed: int, *tags: int) -> int:
    """Deterministic child seed from a base seed and integer tags."""
    ss = np.random.SeedSequence([base_seed, *[int(t) for t in tags]])
    return int(ss.generate_state(1)[0])
