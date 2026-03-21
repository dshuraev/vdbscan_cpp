"""Isotropic Gaussian cluster generator."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..utils import FloatOrRange, sample_float
from .base import ClusterGenerator


class GaussianGenerator(ClusterGenerator):
    """Sample points from an isotropic 3-D Gaussian distribution.

    Parameters
    ----------
    std:
        Standard deviation in ε units (scalar or [min, max] range).
    """

    def __init__(self, std: FloatOrRange) -> None:
        self.std = std

    def generate(self, density_factor: float, min_pts: int, eps: float, rng: Generator) -> np.ndarray:
        import math
        std_eps = sample_float(self.std, rng)
        actual_std = std_eps * eps

        # Effective volume derived from center density:
        #   ρ(0) = N / (2π)^(3/2) / σ³,  E[k] = ρ(0) * V_ε  →  V_eff = (2π)^(3/2) · σ³
        volume_eps = (2.0 * math.pi) ** 1.5 * std_eps ** 3
        n_points = self._compute_n_points(volume_eps, density_factor, min_pts)

        points = rng.normal(0.0, actual_std, (n_points, 3))
        return points.astype(np.float32)
