"""Shared metric helpers for benchmark artifacts."""

from __future__ import annotations

import statistics


def mean(values: list[int] | list[float]) -> float:
    """Return the arithmetic mean, or 0.0 when no samples are available."""
    return float(sum(values) / len(values)) if values else 0.0


def observed_percentile(values: list[float], percentile: float) -> float:
    """Return a lower nearest-rank observed sample for a percentile.

    Unlike interpolated percentiles, this always returns a value that was actually
    measured. The helper accepts percentiles in the inclusive [0, 100] range.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if not values:
        return 0.0
    sorted_values = sorted(values)
    sample_index = int((percentile / 100.0) * (len(sorted_values) - 1))
    return sorted_values[sample_index]


def timing_stats(values: list[float]) -> dict[str, float]:
    """Return stable timing summary fields used across benchmark artifacts.

    p95 uses the lower nearest-rank sample from the sorted measurements. This keeps
    small fixture tests deterministic and avoids interpolating latency samples that
    were never observed during a benchmark run.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": statistics.median(values),
        "p95": observed_percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }
