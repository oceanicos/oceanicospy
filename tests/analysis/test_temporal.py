import numpy as np
import pandas as pd
import pytest

from oceanicospy.analysis.temporal import WaveTemporalAnalyzer


@pytest.fixture
def analyzer(single_burst_df, sampling_data):
    return WaveTemporalAnalyzer(single_burst_df, sampling_data, zero_centered=True)


class TestInit:
    def test_burst_length(self, analyzer, sampling_data):
        assert analyzer.burst_length_s == sampling_data["burst_length_s"]

    def test_surface_level_column_default(self, analyzer):
        assert analyzer.surface_level_column == "eta[m]"

    def test_zero_centered_flag(self, analyzer):
        assert analyzer.zero_centered is True


class TestCheckBurstLength:
    def test_correct_length_returns_true(self, analyzer, single_burst_df):
        assert analyzer._check_burst_length(single_burst_df["eta[m]"]) is True

    def test_short_series_returns_false(self, analyzer, single_burst_df):
        short = single_burst_df["eta[m]"].iloc[:50]
        assert analyzer._check_burst_length(short) is False

    def test_empty_series_returns_false(self, analyzer, single_burst_df):
        empty = single_burst_df["eta[m]"].iloc[:0]
        assert analyzer._check_burst_length(empty) is False


class TestZeroUpcrossingBurst:
    def test_all_outputs_positive(self, analyzer, sampling_data, single_burst_df):
        signal = single_burst_df["eta[m]"].values
        H13, Hmax, Tmean, Lmean = analyzer.apply_zero_upcrossing_burst(
            signal,
            sampling_data["anchoring_depth"],
            sampling_data["sensor_height"],
        )
        assert H13 > 0
        assert Hmax > 0
        assert Tmean > 0
        assert Lmean > 0

    def test_hmax_geq_h13(self, analyzer, sampling_data, single_burst_df):
        signal = single_burst_df["eta[m]"].values
        H13, Hmax, _, _ = analyzer.apply_zero_upcrossing_burst(
            signal,
            sampling_data["anchoring_depth"],
            sampling_data["sensor_height"],
        )
        assert Hmax >= H13

    def test_period_near_10s(self, analyzer, sampling_data, single_burst_df):
        """0.1 Hz sine → mean period ≈ 10 s."""
        signal = single_burst_df["eta[m]"].values
        _, _, Tmean, _ = analyzer.apply_zero_upcrossing_burst(
            signal,
            sampling_data["anchoring_depth"],
            sampling_data["sensor_height"],
        )
        assert Tmean == pytest.approx(10.0, abs=1.0)


class TestComputeParamsFromZeroUpcrossing:
    def test_returns_dataframe(self, analyzer):
        result = analyzer.compute_params_from_zero_upcrossing()
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, analyzer):
        result = analyzer.compute_params_from_zero_upcrossing()
        assert "H1/3" in result.columns
        assert "Tmean" in result.columns

    def test_one_burst_one_row(self, analyzer):
        result = analyzer.compute_params_from_zero_upcrossing()
        assert len(result) == 1

    def test_hs_positive(self, analyzer):
        result = analyzer.compute_params_from_zero_upcrossing()
        assert result["H1/3"].iloc[0] > 0

    def test_invalid_burst_length_dropped(self, sampling_data):
        """A burst with wrong length should be dropped before processing."""
        n = int(sampling_data["burst_length_s"])
        t = np.arange(n) / sampling_data["sampling_freq"]
        eta = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        idx = pd.date_range("2024-01-01", periods=n, freq="s")
        df_good = pd.DataFrame({"eta[m]": eta, "burstId": 0}, index=idx)

        # second burst with wrong length (n-10 samples)
        idx2 = pd.date_range("2024-01-01 01:00", periods=n - 10, freq="s")
        df_bad = pd.DataFrame({"eta[m]": eta[: n - 10], "burstId": 1}, index=idx2)

        df = pd.concat([df_good, df_bad])
        wta = WaveTemporalAnalyzer(df, sampling_data, zero_centered=True)
        result = wta.compute_params_from_zero_upcrossing()
        assert len(result) == 1  # only the valid burst survives
