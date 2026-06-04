import numpy as np
import pytest

from oceanicospy.analysis.extremes import POT_method


class TestPOTMethod:
    def test_all_zeros_returns_nan(self):
        ts = np.zeros(100)
        result = POT_method(np.array([10, 50, 100]), ts, threshold=0.5, num_years=10)
        assert np.all(np.isnan(result))

    def test_output_shape_matches_exceedances(self):
        rng = np.random.default_rng(42)
        ts = rng.exponential(scale=2.0, size=500) + 1.0
        exceedances = np.array([2, 10, 50, 100])
        result = POT_method(exceedances, ts, threshold=0.5, num_years=50)
        assert result.shape == exceedances.shape

    def test_return_levels_are_finite(self):
        rng = np.random.default_rng(0)
        ts = rng.exponential(scale=2.0, size=1000) + 1.0
        result = POT_method(np.array([10, 100]), ts, threshold=0.5, num_years=50)
        assert np.all(np.isfinite(result))

    def test_single_exceedance_value(self):
        rng = np.random.default_rng(7)
        ts = rng.exponential(scale=1.5, size=300) + 1.0
        result = POT_method(np.array([50]), ts, threshold=0.5, num_years=30)
        assert result.shape == (1,)
