"""Uniform background noise generator.

Points are sampled uniformly within a bounding box that covers the entire
cluster scene (with a small margin).  Any candidate that falls within
*min_separation* of an already-occupied position is rejected.

Setting *min_separation* = ε ensures that uniform noise points cannot be
reached from cluster core points, so DBSCAN will always label them as noise.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.spatial import KDTree


def generate_uniform_noise(
    n_points: int,
    occupied: np.ndarray,
    eps: float,
    rng: Generator,
    scene_margin: float = 5.0,
    min_separation: float | None = None,
) -> np.ndarray:
    """Generate *n_points* background noise that does not overlap *occupied*.

    Parameters
    ----------
    n_points:
        Number of noise points to generate.
    occupied:
        (M, 3) array of all currently occupied positions (cluster points +
        any previously generated noise).
    eps:
        DBSCAN ε in world units.
    rng:
        NumPy random Generator.
    scene_margin:
        Extra margin added around the bounding box of *occupied*, in ε units.
    min_separation:
        Minimum distance from any occupied point.  Defaults to *eps*.
    """
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    if min_separation is None:
        min_separation = eps

    margin = scene_margin * eps
    lo = occupied.min(axis=0) - margin
    hi = occupied.max(axis=0) + margin

    tree = KDTree(occupied)
    collected: list[np.ndarray] = []
    n_collected = 0

    while n_collected < n_points:
        batch = max((n_points - n_collected) * 4, 256)

        candidates = rng.uniform(lo, hi, (batch, 3)).astype(np.float32)

        dists, _ = tree.query(candidates, workers=-1)
        valid = candidates[dists >= min_separation]

        if len(valid) > 0:
            collected.append(valid)
            n_collected += len(valid)

    return np.vstack(collected)[:n_points]
