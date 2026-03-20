"""Noise generation pipeline.

``generate_all_noise`` processes the noise configs in order, accumulating
generated points into the *occupied* set so that each noise type avoids the
positions of previous ones.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..models import BoundaryNoiseConfig, SceneConfig, UniformNoiseConfig
from .boundary import generate_boundary_noise
from .uniform import generate_uniform_noise

__all__ = ["generate_all_noise"]


def generate_all_noise(
    config: SceneConfig,
    cluster_points: np.ndarray,
    rng: Generator,
) -> np.ndarray:
    """Generate all noise types defined in *config* and return combined points.

    Processing rules:
    - Each noise type generates ``ratio * len(cluster_points)`` points.
    - Noise is rejected against all previously occupied positions (cluster
      points + any earlier noise type).
    - Noise types therefore never overlap each other or the cluster space.

    Parameters
    ----------
    config:
        Full scene configuration.
    cluster_points:
        (N, 3) array of all cluster points in world space.
    rng:
        NumPy random Generator.

    Returns
    -------
    np.ndarray
        (K, 3) float32 array of all noise points (all types concatenated).
    """
    eps = config.dbscan.eps
    n_cluster = len(cluster_points)

    # Occupied positions grow as each noise type is added
    occupied = cluster_points.copy()
    all_noise_parts: list[np.ndarray] = []

    for noise_cfg in config.noise:
        n_noise = max(0, round(noise_cfg.ratio * n_cluster))

        if n_noise == 0:
            continue

        if isinstance(noise_cfg, UniformNoiseConfig):
            noise_pts = generate_uniform_noise(
                n_points=n_noise,
                occupied=occupied,
                eps=eps,
                rng=rng,
            )
        elif isinstance(noise_cfg, BoundaryNoiseConfig):
            noise_pts = generate_boundary_noise(
                n_points=n_noise,
                cluster_points=cluster_points,
                occupied=occupied,
                distance_from_cluster=noise_cfg.distance_from_cluster,
                eps=eps,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_cfg.type!r}")

        if len(noise_pts) > 0:
            all_noise_parts.append(noise_pts)
            occupied = np.vstack([occupied, noise_pts])

    if not all_noise_parts:
        return np.zeros((0, 3), dtype=np.float32)

    return np.vstack(all_noise_parts).astype(np.float32)
