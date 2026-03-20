"""Place cluster centers in world space.

Algorithm
---------
1. Start with the first center at the origin.
2. For each subsequent center, pick a random existing center as an anchor,
   sample a direction on the unit sphere, and a distance in [min_dist, max_dist].
3. Accept the candidate only if it is at least min_dist away from every
   already-placed center.  If no valid candidate is found after many attempts,
   emit a warning and force-place at the minimum distance.
4. After all centers are placed, shift the centroid to the origin so that the
   entire scene is centered around (0, 0, 0).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.random import Generator

from .utils import FloatOrRange


def place_cluster_centers(
    n_total: int,
    cluster_distance: FloatOrRange,
    eps: float,
    rng: Generator,
    max_attempts: int = 500,
) -> np.ndarray:
    """Return an (n_total, 3) float32 array of cluster center positions.

    Parameters
    ----------
    n_total:
        Total number of individual clusters to place.
    cluster_distance:
        Minimum distance between any two centers, expressed in ε units.
        If a [min, max] pair, new centers are sampled at a distance in that range.
    eps:
        DBSCAN ε — used to convert the relative distances to world units.
    rng:
        NumPy random Generator for reproducibility.
    max_attempts:
        How many random candidates to try before giving up on the constraint.
    """
    if n_total == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n_total == 1:
        return np.zeros((1, 3), dtype=np.float32)

    # Resolve min/max distance in world units
    if isinstance(cluster_distance, (int, float)):
        min_dist = float(cluster_distance) * eps
        max_dist = min_dist * 2.5
    else:
        min_dist = cluster_distance[0] * eps
        max_dist = cluster_distance[1] * eps

    centers: list[np.ndarray] = [np.zeros(3)]

    for i in range(1, n_total):
        placed = False

        # Allow the sampling radius to grow so we never get stuck in dense configs
        adaptive_max = max(max_dist, min_dist * (1.0 + i * 0.4))

        for _ in range(max_attempts):
            # Anchor on a random already-placed center
            anchor = centers[rng.integers(len(centers))]

            # Random direction on the unit sphere
            direction = rng.standard_normal(3)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                continue
            direction /= norm

            distance = rng.uniform(min_dist, adaptive_max)
            candidate = anchor + direction * distance

            # Accept only if the minimum separation is satisfied everywhere
            if all(np.linalg.norm(candidate - c) >= min_dist for c in centers):
                centers.append(candidate)
                placed = True
                break

        if not placed:
            warnings.warn(
                f"Could not satisfy cluster_distance constraint for cluster {i} "
                f"after {max_attempts} attempts.  Consider reducing n_clusters or "
                "cluster_distance.  Placing at minimum distance from a random center.",
                UserWarning,
                stacklevel=2,
            )
            anchor = centers[rng.integers(len(centers))]
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            centers.append(anchor + direction * min_dist * 1.05)

    result = np.array(centers, dtype=np.float32)

    # Center the entire scene around the origin
    result -= result.mean(axis=0)

    return result
