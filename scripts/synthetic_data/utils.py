"""Sampling utilities shared across generators and noise modules."""

from __future__ import annotations

from numpy.random import Generator

FloatOrRange = float | list[float]


def sample_float(value: FloatOrRange, rng: Generator) -> float:
    """Return *value* directly or sample uniformly from [value[0], value[1]]."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(rng.uniform(value[0], value[1]))
