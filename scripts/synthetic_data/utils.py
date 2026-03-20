"""Sampling utilities shared across generators and noise modules."""

from __future__ import annotations

from numpy.random import Generator

FloatOrRange = float | list[float]
IntOrRange = int | list[int]


def sample_float(value: FloatOrRange, rng: Generator) -> float:
    """Return *value* directly or sample uniformly from [value[0], value[1]]."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(rng.uniform(value[0], value[1]))


def sample_int(value: IntOrRange, rng: Generator) -> int:
    """Return *value* directly or sample uniformly from {value[0], …, value[1]}."""
    if isinstance(value, int):
        return value
    lo, hi = int(value[0]), int(value[1])
    return int(rng.integers(lo, hi + 1))
