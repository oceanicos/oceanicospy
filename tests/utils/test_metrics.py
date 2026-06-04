import numpy as np
import pytest

from oceanicospy.utils.metrics import stats


class TestStats:
    def test_perfect_prediction(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, bias, rmse, si = stats(x, x)
        assert r == pytest.approx(1.0)
        assert bias == pytest.approx(0.0, abs=1e-12)
        assert rmse == pytest.approx(0.0, abs=1e-12)
        assert si == pytest.approx(0.0, abs=1e-12)

    def test_constant_positive_bias(self):
        x = np.array([1.0, 2.0, 3.0])
        y = x + 2.0
        r, bias, rmse, si = stats(x, y)
        assert r == pytest.approx(1.0)
        assert bias == pytest.approx(2.0)
        assert rmse == pytest.approx(2.0)

    def test_constant_negative_bias(self):
        x = np.array([2.0, 4.0, 6.0])
        y = x - 1.0
        _, bias, rmse, _ = stats(x, y)
        assert bias == pytest.approx(-1.0)
        assert rmse == pytest.approx(1.0)

    def test_scatter_index_definition(self):
        """SI = RMSE / mean(x)."""
        x = np.array([2.0, 4.0, 6.0])
        y = x + 1.0
        _, _, rmse, si = stats(x, y)
        assert si == pytest.approx(rmse / np.mean(x))

    def test_short_series_returns_nan(self):
        r, bias, rmse, si = stats([1.0], [1.0])
        assert np.isnan(r)
        assert np.isnan(bias)
        assert np.isnan(rmse)
        assert np.isnan(si)

    def test_zero_mean_x_gives_nan_si(self):
        x = np.array([-1.0, 1.0])
        y = np.array([0.0, 0.0])
        _, _, _, si = stats(x, y)
        assert np.isnan(si)

    def test_rmse_non_negative(self):
        x = np.array([1.0, 3.0, 5.0])
        y = np.array([2.0, 1.0, 6.0])
        _, _, rmse, _ = stats(x, y)
        assert rmse >= 0

    def test_accepts_lists(self):
        r, bias, rmse, si = stats([1.0, 2.0], [1.0, 2.0])
        assert r == pytest.approx(1.0)

    def test_anticorrelated(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = -x + 5.0
        r, _, _, _ = stats(x, y)
        assert r == pytest.approx(-1.0)
