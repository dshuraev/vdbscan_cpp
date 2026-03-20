"""Solid tube (thin cylinder) cluster generator."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..utils import FloatOrRange, sample_float
from .base import ClusterGenerator


class TubeGenerator(ClusterGenerator):
    """Solid cylinder with a small radius — like a rod or pencil.

    Points are distributed uniformly within the cylinder volume.

    Parameters
    ----------
    length:
        Cylinder length along the Z axis, in ε units.
    radius:
        Cross-section radius, in ε units.
    """

    def __init__(self, length: FloatOrRange, radius: FloatOrRange) -> None:
        self.length = length
        self.radius = radius

    def generate(self, n_points: int, eps: float, rng: Generator) -> np.ndarray:
        length_eps = sample_float(self.length, rng)
        radius_eps = sample_float(self.radius, rng)

        self._warn_small_dimension("length", length_eps)

        length_w = length_eps * eps
        radius_w = radius_eps * eps

        # Uniform in a disk (sqrt gives uniform area distribution)
        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        r = np.sqrt(rng.uniform(0.0, 1.0, n_points)) * radius_w
        z = rng.uniform(-length_w / 2.0, length_w / 2.0, n_points)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.column_stack([x, y, z]).astype(np.float32)
