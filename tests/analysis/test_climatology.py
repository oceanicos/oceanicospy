import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="skipped")

from oceanicospy.analysis.climatology import compute_annual_cycle


@pytest.fixture
def monthly_series():
    """3 years of daily data where the value equals the calendar month number."""
    dates = pd.date_range("2020-01-01", periods=365 * 3, freq="D")
    values = np.array([d.month for d in dates], dtype=float)
    return pd.Series(values, index=dates), dates


class TestMonthlyMean:
    def test_shape(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data.values, time, freq="Monthly", method="mean")
        assert len(result) == 12

    def test_columns(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data.values, time, freq="Monthly", method="mean")
        assert set(result.columns) == {"month", "mean", "std"}

    def test_mean_values_equal_month_number(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data.values, time, freq="Monthly", method="mean")
        for _, row in result.iterrows():
            assert row["mean"] == pytest.approx(row["month"], abs=1e-9)

    def test_pandas_series_input(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data, time, freq="Monthly", method="mean")
        assert len(result) == 12


class TestMonthlyMedian:
    def test_columns(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data.values, time, freq="Monthly", method="median")
        assert "median" in result.columns
        assert "std" in result.columns

    def test_shape(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data.values, time, freq="Monthly", method="median")
        assert len(result) == 12


class TestSeasonalCycle:
    def test_mean_shape(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data.values, time, freq="Seasonal", method="mean")
        assert len(result) == 4

    def test_mean_columns(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(data.values, time, freq="Seasonal", method="mean")
        assert set(result.columns) == {"season", "mean", "std"}

    def test_median_shape(self, monthly_series):
        data, time = monthly_series
        result = compute_annual_cycle(
            data.values, time, freq="Seasonal", method="median"
        )
        assert len(result) == 4


class TestInputValidation:
    def test_2d_array_raises_value_error(self, monthly_series):
        data, time = monthly_series
        with pytest.raises(ValueError):
            compute_annual_cycle(data.values.reshape(3, -1), time[:3])
