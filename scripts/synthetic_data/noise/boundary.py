"""Boundary noise generator.

Points are placed in a shell around the cluster point cloud.  For each
generated point:
  1. A random cluster point is chosen as a seed.
  2. A random direction is sampled uniformly on the unit sphere.
  3. A distance in [distance_min, distance_max] is sampled and applied.

This places noise near but outside the cluster regions.  Candidates that
would overlap an already-occupied position are rejected.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.spatial import KDTree

from ..utils import FloatOrRange


def generate_boundary_noise(
    n_points: int,
    cluster_points: np.ndarray,
    occupied: np.ndarray,
    distance_from_cluster: FloatOrRange,
    eps: float,
    rng: Generator,
    min_separation: float | None = None,
) -> np.ndarray:
    """Generate *n_points* of boundary noise around *cluster_points*.

    Parameters
    ----------
    n_points:
        Number of noise points to generate.
    cluster_points:
        (M, 3) array of all cluster points — used as seed positions.
    occupied:
        (K, 3) array of positions to avoid (cluster + previous noise).
    distance_from_cluster:
        Distance range [min, max] in ε units at which to place noise relative
        to the nearest cluster point.
    eps:
        DBSCAN ε in world units.
    rng:
        NumPy random Generator.
    min_separation:
        Minimum distance from any occupied point.  Defaults to ``eps * 0.5``.
        Boundary noise is intentionally near clusters, so a looser threshold
        than for uniform noise is appropriate.
    """
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    if min_separation is None:
        min_separation = eps * 0.5

    # Resolve distance range in world units
    if isinstance(distance_from_cluster, (int, float)):
        d_min = float(distance_from_cluster) * eps
        d_max = d_min * 1.5
    else:
        d_min = distance_from_cluster[0] * eps
        d_max = distance_from_cluster[1] * eps

    tree = KDTree(occupied)
    collected: list[np.ndarray] = []
    n_collected = 0

    while n_collected < n_points:
        batch = max((n_points - n_collected) * 4, 256)

        # Pick random cluster points as seeds
        seed_idx = rng.integers(0, len(cluster_points), batch)
        seeds = cluster_points[seed_idx]

        # Random directions on the unit sphere
        directions = rng.standard_normal((batch, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.where(norms < 1e-12, 1.0, norms)

        # Random distances in [d_min, d_max]
        distances = rng.uniform(d_min, d_max, batch)
        candidates = (seeds + directions * distances[:, np.newaxis]).astype(np.float32)

        # Reject candidates that overlap occupied space
        dists, _ = tree.query(candidates, workers=-1)
        valid = candidates[dists >= min_separation]

        if len(valid) > 0:
            collected.append(valid)
            n_collected += len(valid)

    return np.vstack(collected)[:n_points]
