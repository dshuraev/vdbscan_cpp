"""Solid sphere cluster generator."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..utils import FloatOrRange, sample_float
from .base import ClusterGenerator


class BallGenerator(ClusterGenerator):
    """Solid sphere with uniform point distribution.

    Parameters
    ----------
    radius:
        Sphere radius, in ε units.
    """

    def __init__(self, radius: FloatOrRange) -> None:
        self.radius = radius

    def generate(self, density_factor: float, min_pts: int, eps: float, rng: Generator) -> np.ndarray:
        import math
        radius_eps = sample_float(self.radius, rng)

        self._warn_small_dimension("radius", radius_eps)

        # Volume of sphere: (4/3)π r³  (in ε³ units)
        volume_eps = (4.0 * math.pi / 3.0) * radius_eps ** 3
        n_points = self._compute_n_points(volume_eps, density_factor, min_pts)

        radius_w = radius_eps * eps

        # Uniform direction on the unit sphere
        directions = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.where(norms < 1e-12, 1.0, norms)  # guard against zero vectors

        # Uniform radial distribution: r = R * cbrt(U)  so that volume ∝ r²
        r = radius_w * rng.uniform(0.0, 1.0, n_points) ** (1.0 / 3.0)

        points = directions * r[:, np.newaxis]
        return points.astype(np.float32)
