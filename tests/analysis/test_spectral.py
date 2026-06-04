import numpy as np
import pandas as pd
import pytest

from oceanicospy.analysis.spectral import WaveSpectralAnalyzer


@pytest.fixture
def analyzer(single_burst_df, sampling_data):
    return WaveSpectralAnalyzer(single_burst_df, sampling_data, logger=False)


class TestInit:
    def test_sampling_freq(self, analyzer, sampling_data):
        assert analyzer.sampling_freq == sampling_data["sampling_freq"]

    def test_anchoring_depth(self, analyzer, sampling_data):
        assert analyzer.anchoring_depth == sampling_data["anchoring_depth"]

    def test_sensor_height(self, analyzer, sampling_data):
        assert analyzer.sensor_height == sampling_data["sensor_height"]

    def test_burst_length(self, analyzer, sampling_data):
        assert analyzer.burst_length_s == sampling_data["burst_length_s"]


class TestCheckBurstLength:
    def test_correct_length_returns_true(self, analyzer, single_burst_df):
        burst = single_burst_df["eta[m]"]
        assert analyzer._check_burst_length(burst) is True

    def test_short_series_returns_false(self, analyzer, single_burst_df):
        short = single_burst_df["eta[m]"].iloc[:100]
        assert analyzer._check_burst_length(short) is False

    def test_one_extra_sample_returns_false(self, analyzer, single_burst_df):
        extra = single_burst_df["eta[m]"].iloc[:513]
        assert analyzer._check_burst_length(extra) is False


class TestSmoothPSD:
    def test_output_length_preserved(self, analyzer):
        psd = np.ones(200)
        smoothed = analyzer._smooth_psd_spectrum(psd, 5)
        assert len(smoothed) == 200

    def test_uniform_psd_stays_uniform(self, analyzer):
        psd = np.ones(100)
        smoothed = analyzer._smooth_psd_spectrum(psd, 7)
        assert np.allclose(smoothed, 1.0, atol=0.01)


class TestFFTSpectrum:
    def test_output_shapes(self, analyzer, single_burst_df):
        signal = single_burst_df["eta[m]"].values
        freqs, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        n = len(signal)
        assert len(freqs) == n // 2 + 1
        assert len(psd) == n // 2 + 1

    def test_psd_non_negative(self, analyzer, single_burst_df):
        signal = single_burst_df["eta[m]"].values
        _, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        assert np.all(psd >= 0)

    def test_freqs_start_at_zero(self, analyzer, single_burst_df):
        signal = single_burst_df["eta[m]"].values
        freqs, _ = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        assert freqs[0] == pytest.approx(0.0)

    def test_peak_frequency_near_01hz(self, analyzer, single_burst_df):
        """Sine wave at 0.1 Hz must dominate the spectrum."""
        signal = single_burst_df["eta[m]"].values
        freqs, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        peak_freq = freqs[np.argmax(psd)]
        assert peak_freq == pytest.approx(0.1, abs=0.005)


class TestWaveParams:
    def test_all_params_positive(self, analyzer, single_burst_df):
        signal = single_burst_df["eta[m]"].values
        freqs, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        Hs, Hrms, Hmean, Tp, Tm01, Tm02 = analyzer.get_wave_params_from_spectrum(psd, freqs)
        assert Hs > 0
        assert Hrms > 0
        assert Hmean > 0
        assert Tp > 0
        assert Tm01 > 0
        assert Tm02 > 0

    def test_tp_near_10s(self, analyzer, single_burst_df):
        """Peak period should be near 10 s (1/0.1 Hz)."""
        signal = single_burst_df["eta[m]"].values
        freqs, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        _, _, _, Tp, _, _ = analyzer.get_wave_params_from_spectrum(psd, freqs)
        assert Tp == pytest.approx(10.0, abs=1.0)

    def test_hs_physically_reasonable(self, analyzer, single_burst_df):
        """0.5 m amplitude sine → Hs ≈ 1.41 m (Hs = 2√2·A)."""
        signal = single_burst_df["eta[m]"].values
        freqs, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        Hs, _, _, _, _, _ = analyzer.get_wave_params_from_spectrum(psd, freqs)
        assert Hs == pytest.approx(2 * np.sqrt(2) * 0.5, rel=0.05)


class TestIGBandSplit:
    def test_outputs_non_negative(self, analyzer, single_burst_df):
        signal = single_burst_df["eta[m]"].values
        freqs, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        Hm0_ig, Hm0_sw = analyzer._compute_hs_ig_band(psd, freqs, freq_split=0.05)
        assert Hm0_ig >= 0
        assert Hm0_sw >= 0

    def test_sw_dominates_for_short_wave_signal(self, analyzer, single_burst_df):
        """Signal at 0.1 Hz is in the short-wave band (> 0.05 Hz split)."""
        signal = single_burst_df["eta[m]"].values
        freqs, psd = analyzer.compute_spectrum_from_direct_fft(signal, kp_correction=False)
        Hm0_ig, Hm0_sw = analyzer._compute_hs_ig_band(psd, freqs, freq_split=0.05)
        assert Hm0_sw > Hm0_ig


class TestGetSpectraAndParams:
    def test_spectra_keys(self, analyzer):
        spectra, _ = analyzer.get_spectra_and_params_for_bursts(
            method="fft", kp_correction=False
        )
        assert "S" in spectra
        assert "freq" in spectra
        assert "time" in spectra

    def test_params_columns(self, analyzer):
        _, params = analyzer.get_spectra_and_params_for_bursts(
            method="fft", kp_correction=False
        )
        for col in ("Hm0", "Hrms", "Hmean", "Tp", "Tm01", "Tm02"):
            assert col in params.columns

    def test_one_burst_one_row(self, analyzer):
        _, params = analyzer.get_spectra_and_params_for_bursts(
            method="fft", kp_correction=False
        )
        assert len(params) == 1

    def test_missing_burst_id_raises(self, single_burst_df, sampling_data):
        df_no_id = single_burst_df.drop(columns=["burstId"])
        wsa = WaveSpectralAnalyzer(df_no_id, sampling_data, logger=False)
        with pytest.raises(ValueError, match="burstId"):
            wsa.get_spectra_and_params_for_bursts(method="fft", kp_correction=False)
