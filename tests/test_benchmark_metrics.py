import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks.metrics import mean, observed_percentile, timing_stats


def test_mean_returns_zero_for_empty_values():
    assert mean([]) == 0.0


def test_timing_stats_uses_lower_nearest_rank_p95():
    stats = timing_stats([float(index) for index in range(100)])

    assert stats == {
        "mean": 49.5,
        "median": 49.5,
        "p95": 94.0,
        "min": 0.0,
        "max": 99.0,
    }


def test_observed_percentile_returns_measured_sample():
    values = [float(index) for index in range(100)]

    assert observed_percentile(values, 0) == 0.0
    assert observed_percentile(values, 90) == 89.0
    assert observed_percentile(values, 95) == 94.0
    assert observed_percentile(values, 99) == 98.0
    assert observed_percentile(values, 100) == 99.0


def test_observed_percentile_rejects_out_of_range_percentiles():
    for percentile in (-1, 101):
        try:
            observed_percentile([], percentile)
        except ValueError as exc:
            assert "between 0 and 100" in str(exc)
        else:
            raise AssertionError(f"invalid percentile should fail: {percentile}")


def test_timing_stats_empty_values_are_zeroed():
    assert timing_stats([]) == {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
