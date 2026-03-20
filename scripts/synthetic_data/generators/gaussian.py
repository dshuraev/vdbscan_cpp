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

    def generate(self, n_points: int, eps: float, rng: Generator) -> np.ndarray:
        std_eps = sample_float(self.std, rng)
        actual_std = std_eps * eps

        points = rng.normal(0.0, actual_std, (n_points, 3))
        return points.astype(np.float32)
