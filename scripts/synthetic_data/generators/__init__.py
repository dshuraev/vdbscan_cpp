"""Cluster generator registry and factory.

All generators live in their own modules.  This file provides a single
``create_generator`` factory so the rest of the pipeline never needs to
import individual generator classes directly.
"""

from __future__ import annotations

from ..models import (
    BallClusterConfig,
    BoxClusterConfig,
    ClusterConfig,
    CrescentClusterConfig,
    CylinderClusterConfig,
    GaussianClusterConfig,
    TorusClusterConfig,
    TubeClusterConfig,
)
from .ball import BallGenerator
from .base import ClusterGenerator, apply_rotation
from .box import BoxGenerator
from .crescent import CrescentGenerator
from .cylinder import CylinderGenerator
from .gaussian import GaussianGenerator
from .torus import TorusGenerator
from .tube import TubeGenerator

__all__ = ["ClusterGenerator", "apply_rotation", "create_generator"]


def create_generator(config: ClusterConfig) -> ClusterGenerator:
    """Instantiate the appropriate generator for *config*.

    Parameters
    ----------
    config:
        Any of the ``*ClusterConfig`` Pydantic models.

    Returns
    -------
    ClusterGenerator
        A generator whose ``generate(n_points, eps, rng)`` method produces
        points in local space (centered at origin, in world units).
    """
    match config:
        case GaussianClusterConfig():
            return GaussianGenerator(config.std)

        case CrescentClusterConfig():
            return CrescentGenerator(
                radius=config.radius,
                thickness=config.thickness,
                arc_fraction=config.arc_fraction,
            )

        case TorusClusterConfig():
            return TorusGenerator(
                major_radius=config.major_radius,
                aspect_ratio=config.aspect_ratio,
            )

        case TubeClusterConfig():
            return TubeGenerator(length=config.length, radius=config.radius)

        case CylinderClusterConfig():
            return CylinderGenerator(radius=config.radius, height=config.height)

        case BallClusterConfig():
            return BallGenerator(radius=config.radius)

        case BoxClusterConfig():
            return BoxGenerator(
                width=config.width,
                height=config.height,
                depth=config.depth,
            )

        case _:
            raise ValueError(f"Unknown cluster type: {config.type!r}")
