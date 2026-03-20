"""Assign per-label RGB colors using the golden-ratio hue sequence.

The golden ratio φ = (1 + √5) / 2 gives maximally spread hues when successive
labels are multiplied by φ and taken mod 1.  Label -1 (DBSCAN noise) is always
rendered as neutral grey.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
_NOISE_COLOR: tuple[int, int, int] = (128, 128, 128)

# HSV parameters for cluster colors
_SATURATION = 0.75
_VALUE = 0.90


def label_to_rgb(label: int) -> tuple[int, int, int]:
    """Return (R, G, B) uint8 tuple for a DBSCAN *label*.

    Noise (label == -1) → grey (128, 128, 128).
    Clusters          → evenly spread hues via the golden-ratio sequence.
    """
    if label < 0:
        return _NOISE_COLOR
    hue = (label * _GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, _SATURATION, _VALUE)
    return (round(r * 255), round(g * 255), round(b * 255))


def assign_colors(labels: np.ndarray) -> np.ndarray:
    """Return an (N, 3) uint8 array of RGB colors for each point.

    Parameters
    ----------
    labels:
        1-D integer array of DBSCAN labels (-1 = noise).
    """
    unique = np.unique(labels)
    color_map = {int(lbl): label_to_rgb(int(lbl)) for lbl in unique}
    colors = np.array([color_map[int(lbl)] for lbl in labels], dtype=np.uint8)
    return colors
