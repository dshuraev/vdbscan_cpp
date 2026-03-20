"""Base class and shared helpers for all cluster generators."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from ..models import RotationSpec


class ClusterGenerator(ABC):
    """Generate points for a single cluster instance in local space.

    "Local space" means:
    - centered at the origin
    - coordinates are in world units (already scaled by ε)

    The caller is responsible for applying rotation and translating to the
    cluster's world-space center.
    """

    @abstractmethod
    def generate(self, n_points: int, eps: float, rng: Generator) -> np.ndarray:
        """Return (n_points, 3) float32 array of cluster points in local space."""
        ...

    def _warn_small_dimension(self, name: str, value_in_eps_units: float) -> None:
        """Emit a warning when a geometric dimension is below 2 ε.

        Small dimensions (< 2 ε) risk having too few neighbors within the ε
        ball for DBSCAN to classify points as core points, even with alpha > 1.
        """
        if value_in_eps_units < 2.0:
            warnings.warn(
                f"{type(self).__name__}: '{name}' = {value_in_eps_units:.3f} ε "
                "is below 2 ε.  The cluster may not be reliably detected by DBSCAN.",
                UserWarning,
                stacklevel=4,
            )


def apply_rotation(
    points: np.ndarray,
    rotation_spec: RotationSpec,
    rng: Generator,
) -> np.ndarray:
    """Rotate (N, 3) *points* according to *rotation_spec*.

    Parameters
    ----------
    rotation_spec:
        - ``"random"``       — uniformly random 3-D rotation (default).
        - ``"none"``         — identity, no rotation applied.
        - ``[w, x, y, z]``  — specific quaternion (scalar-first convention).
    """
    if rotation_spec == "none":
        return points

    if rotation_spec == "random":
        rot = Rotation.random(random_state=int(rng.integers(2**31)))
    else:
        # Config uses [w, x, y, z]; scipy expects [x, y, z, w]
        w, x, y, z = rotation_spec
        rot = Rotation.from_quat([x, y, z, w])

    return rot.apply(points).astype(np.float32)
