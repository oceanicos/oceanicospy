import logging
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks,welch

from ..utils import wave_props,extras

class WaveSpectralAnalyzer():
    def __init__(self,measured_signal,sampling_data,surface_level_column='eta[m]',logger=True):
        """
        Initializes the analysis object with measurement signal and sampling data.

        Parameters
        ----------
        measured_signal : array-like
            The input signal data to be analyzed.
        sampling_data : dict
            Dictionary containing sampling parameters with the following keys:
                - ``sampling_freq`` (float): Sampling frequency of the signal.
                - ``anchoring_depth`` (float): Depth at which the sensor is anchored.
                - ``sensor_height`` (float): Height of the sensor above the bottom.
                - ``burst_length_s`` (float): Duration of each burst in seconds.
        surface_level_column : str
            Column name in measured_signal that contains the surface level data (default is ``eta[m]``).
        logger : bool
            If True, initializes a logger for the class (default is True).

        Notes
        -----
        **Development history**

        - 01-Ago-2025 : Origination - Franklin Ayala
        - 01-Sep-2025 : FFT method - Juan Diego Toro
        - 10-Oct-2025 : Kp correction - Franklin Ayala/Juan Diego Toro/Camilo Cabrera
        - 12-Nov-2025 : Welch's method - Franklin Ayala
        - 10-Dec-2025 : Wavelets analysis - Franklin Ayala
        """

        self.measured_signal = measured_signal
        self.sampling_data = sampling_data
        self.sampling_freq = self.sampling_data['sampling_freq']
        self.anchoring_depth = self.sampling_data['anchoring_depth']
        self.sensor_height = self.sampling_data['sensor_height']
        self.burst_length_s = self.sampling_data['burst_length_s']
        self.surface_level_column = surface_level_column

        if logger:
            self.logfile = 'wave_spectral_analyzer.log'
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)

            if not self.logger.handlers:
                handler = logging.FileHandler(self.logfile)
                formatter = logging.Formatter(
                    "%(asctime)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

    def _check_burst_length(self,burst_series):
        """Verify that the burst has the expected number of samples based on the sampling frequency and burst length.
        
        Parameters
        ----------
        burst_series : pandas.Series
            The burst data as a pandas Series.

        Returns
        -------
        bool
            True if the burst has the expected number of samples, False otherwise.

        Raises
        ------
        ValueError
            If the burst is missing timestamps.
        """
        expected_samples = int(self.burst_length_s)
        if len(burst_series) != expected_samples:
            return False
        else:
            return True
        
    def _verify_bursts_in_signal(self,measured_signal):
        """
        Verify that each burst in the measured signal has the expected number of samples. If not, remove the burst and log a warning.

        Parameters
        ----------
        measured_signal : pandas.DataFrame
            The input measurement signal containing a 'burstId' column.
        
        Returns
        -------
        pandas.DataFrame
            The measurement signal with bursts of incorrect length removed.
        """
        burst_to_delete = []
        for burst_id in measured_signal["burstId"].unique():
            burst_series = measured_signal[measured_signal["burstId"] == burst_id]
            if self._check_burst_length(burst_series) == False:
                burst_to_delete.append(int(burst_id))
        
        if burst_to_delete:
            if self.logger:
                self.logger.warning(f"The following bursts have been removed due to incorrect length: {burst_to_delete}")
            measured_signal = measured_signal[~measured_signal["burstId"].isin(burst_to_delete)]
        return measured_signal       
        
    def _compute_spectrum_for_burst(self, burst_signal, method, kp_correction, window_type, window_length, smoothing_bins):
        """Calculate the spectrum for the burst using the specified method and applying Kp correction if needed.
        
        Parameters
        ----------
        burst_signal : pandas.Series
            The burst signal as a pandas Series.
        method : str
            The method to compute the spectrum: ``'fft'`` or ``'welch'``.
        kp_correction : bool
            Whether to apply Kp correction to the spectrum.
        window_type : str
            The type of window to use for Welch's method (e.g., ``'hamming'``, ``'hann'``).
        window_length : int
            The length of the window in samples for Welch's method.
        smoothing_bins : int
            The number of bins for moving-average smoothing (Welch only).
        
        Returns
         -------
         freqs : ndarray
             The frequency array corresponding to the spectrum.
         spectrum : ndarray
             The computed power spectral density (PSD) for the burst.
            
        Raises        
        ------
        ValueError
            If the specified method is not recognized.
         """
        if self._check_burst_length(burst_signal):
            burst_signal = burst_signal.values
            if method == 'fft':
                result = self.compute_spectrum_from_direct_fft(burst_signal, kp_correction)
            elif method == 'welch':
                # welch's method requires at least 2 segments, so we set the default overlap to 50% of the window length if not provided
                result = self.compute_spectrum_from_welch(burst_signal, kp_correction, window_type, window_length)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'fft' or 'welch'.")

            if kp_correction:
                freqs, spectrum, _ = result  # (freqs, PSD_kp, PSD)
            else:
                freqs, spectrum = result  # (freqs, PSD)

            # If using Welch's method and smoothing_bins is provided, apply smoothing to the PSD
            if method == 'welch' and smoothing_bins is not None:
                spectrum = self._smooth_psd_spectrum(spectrum, smoothing_bins)
            return freqs, spectrum

    def _smooth_psd_spectrum(self,PSD,smoothing_bins):
        """Smooth the power spectral density (PSD) spectrum using a moving average filter.
        Parameters
        ----------
        PSD : ndarray
            The power spectral density to be smoothed.
        smoothing_bins : int
            The number of bins to use for the moving average smoothing.
        
        Returns
        -------
        PSD_smoothed : ndarray
            The smoothed power spectral density.
        """

        kernel = np.ones(smoothing_bins) / smoothing_bins
        PSD_smoothed = np.convolve(PSD, kernel, mode='same')
        return PSD_smoothed

    def _compute_nonadaptive_Kp(self,freqs):
        """Compute non-adaptive Kp correction factor based on linear wave theory.

        Parameters
        ----------
        freqs : ndarray
            Frequency array corresponding to the PSD.
        
        Returns
        -------
        Kp : ndarray
            The non-adaptive Kp correction factor for each frequency.
        """

        total_depth = self.anchoring_depth + self.sensor_height
        L = np.array([wave_props.wavelength(1/f, total_depth) for f in freqs])
        k = 2*np.pi/L
        Kp = np.cosh(k * self.sensor_height) / np.cosh(k * total_depth)
        if freqs[0]==0:
            Kp[0] = 1

        Kp_min = (np.cosh(np.pi/(total_depth - self.sensor_height)*self.sensor_height)) / \
                (np.cosh(np.pi/(total_depth - self.sensor_height)*total_depth))
        Kp = np.clip(Kp, Kp_min, 1)
        return Kp
    
    def _compute_adaptive_Kp(self,freqs,PSD):
        """Compute adaptive Kp correction factor based on the spectrum shape.
        Parameters
        ----------
        freqs : ndarray
            Frequency array corresponding to the PSD.
        PSD : ndarray
            Power spectral density to be corrected.

        Returns
        -------
        Kp_final : ndarray
            The adaptive Kp correction factor for each frequency.
        """

        total_depth = self.anchoring_depth + self.sensor_height
        L = np.array([wave_props.wavelength(1/f, total_depth) for f in freqs])
        k = 2*np.pi/L
        Kp = np.cosh(k * self.sensor_height) / np.cosh(k * total_depth)
        Kp_min = (np.cosh(np.pi/(total_depth - self.sensor_height)*self.sensor_height)) / \
                (np.cosh(np.pi/(total_depth - self.sensor_height)*total_depth))

        if freqs[0] == 0:
            Kp[0] = 1

        PSD_Kp = PSD / (Kp**2)
        PSD_Kp_smoothed = self._smooth_psd_spectrum(PSD_Kp,24)
        minima_idx, _ = find_peaks(-np.log(PSD_Kp_smoothed),prominence=0.5)
        last_min_idx = minima_idx[-1]
        f_maxpcorr = freqs[last_min_idx]

        # Assuming f_maxpcorr is always lower than fcmax and fmax_Kp
        total_depth = self.anchoring_depth + self.sensor_height
        L_max = wave_props.wavelength(1/f_maxpcorr, total_depth)
        k_max = 2*np.pi/L_max
        Kp_min = np.cosh(k_max * self.sensor_height) / np.cosh(k_max * total_depth)

        Kp_final = Kp.copy()
        Kp_final[0:last_min_idx] =  Kp[0:last_min_idx]
        Kp_final[last_min_idx:] = Kp_min
        return Kp_final
    
    def correction_by_Kp(self,freqs,PSD,kp_method='adaptive'):
        """Apply Kp correction to the power spectral density (PSD) based on the specified method.
        
        Parameters
        ----------
        freqs : ndarray
            Frequency array corresponding to the PSD.
        PSD : ndarray
            Power spectral density to be corrected.
        kp_method : str, optional
            Method for Kp correction: ``'adaptive'`` or ``'nonadaptive'``. Default is ``'adaptive'``
        
        Returns
        -------
        freqs : ndarray
            Frequency array corresponding to the PSD.
        PSD_Kp : ndarray
            The Kp-corrected power spectral density.
        PSD : ndarray
            The original power spectral density (returned for reference).
            
        Notes
        -----
        Most of the equations used on the methods are based on [1]_

        .. [1] Karimpour, A., & Chen, Q. (2017). Wind wave analysis in depth limited water using OCEANLYZ,
            A MATLAB toolbox. Computers & Geosciences, 106, 181-189. https://doi.org/10.1016/j.cageo.2017.06.010
        """
        
        if kp_method == 'nonadaptive':
            Kp = self._compute_nonadaptive_Kp(freqs)
        else:
            Kp = self._compute_adaptive_Kp(freqs,PSD)
        PSD_Kp = PSD / (Kp**2)

        return freqs, PSD_Kp, PSD

    def _compute_hs_ig_band(self,PSD,freqs,freq_split):
        """Computes significant wave height in the infragravity and short-wave band.
        Parameters
        ----------
        PSD : ndarray
            Power spectral density to be analyzed.
        freqs : ndarray
            Frequency array corresponding to the PSD.
        freq_split : float
            The frequency that separates the infragravity band from the short-wave band.
        
        Returns
        -------
        Hm0_ig : float
            Significant wave height in the infragravity band [m].
        Hm0_sw : float
            Significant wave height in the short-wave band [m].

        Notes
        -----
        The upper limit for the short-wave band is set to 0.2 Hz while the lower limit for the infragravity band is set to 0 Hz.
        """

        freq_upper_sw = 0.2
        freq_lower_ig = 0.
        ig_band_mask = (freqs >= freq_lower_ig) & (freqs <= freq_split)
        sw_band_mask = (freqs > freq_split) & (freqs <= freq_upper_sw)
        m0_ig = np.trapezoid(PSD[ig_band_mask], freqs[ig_band_mask])
        m0_sw = np.trapezoid(PSD[sw_band_mask], freqs[sw_band_mask])
        Hm0_ig = 4.004 * np.sqrt(m0_ig)
        Hm0_sw = 4.004 * np.sqrt(m0_sw)

        return Hm0_ig,Hm0_sw

    @extras.timing_decorator
    def _compute_wavelet_scalogram_for_burst(self,burst_signal,window_length,overlap,mother_wavelet,scales):
        """Compute wavelet scalogram for a single burst signal using overlapping windows.
        
        Parameters
        ----------
        burst_signal : ndarray
            The signal for which to compute the wavelet scalogram.
        window_length : int
            The length of each window.
        overlap : float
            The overlap between consecutive windows.
        mother_wavelet : str
            The mother wavelet to use.
        scales : ndarray
            The scales for the wavelet transform.

        Returns
        -------
        stitched : ndarray
            The stitched wavelet scalogram.
        freqs : ndarray
            The corresponding frequencies.

        """

        if self._check_burst_length(burst_signal):
            step = int(window_length * (1 - overlap))
            if len(burst_signal)==window_length:
                n_segments =1
            else:
                n_segments = (len(burst_signal) - window_length) // step + 1
            window = np.hanning(window_length)

            stitched = np.zeros((len(scales), len(burst_signal)))
            weight = np.zeros(len(burst_signal))

            for idx_seg in range(n_segments):
                start = idx_seg * step
                end = start + window_length
                segment = burst_signal[start:end]
                coef, freqs = pywt.cwt(segment, scales, mother_wavelet, sampling_period=1/self.sampling_freq)
                coef_mag = np.abs(coef) * window  # Apply window to smooth overlap
                stitched[:, start:end] += coef_mag
                weight[start:end] += window

            stitched /= np.maximum(weight, 1e-8)
            return stitched,freqs

    def get_wave_params_from_spectrum(self,PSD,freqs):
        """
        This function computes different wave integral parameters from the spectrum
        
        Parameters
        ----------
        PSD : list or ndarray
            Density variance spectrum
        freqs : list or ndarray
            Frequencies of the spectrum
        
        Returns
        -------
        Hs : float
            Significant wave height [m]
        Hrms : float
            Root-mean squared wave height [m]
        Hmean : float
            Mean wave height [m]
        Tp : float
            Peak period [s]
        Tm01 : float
            Mean period - first order [s]
        Tm02 : float
            Mean period - second order [s]
        """

        m0 = np.trapezoid(PSD, freqs.flatten())
        m1 = np.trapezoid(freqs.flatten()*PSD, freqs.flatten())
        m2 = np.trapezoid((freqs.flatten()**2)*PSD, freqs.flatten())

        Hs = 4.004*np.sqrt(m0)
        Hrms = np.sqrt(8*m0)
        Hmean = np.sqrt(2*np.pi*m0)

        Tp = 1/freqs[np.argmax(PSD)]
        Tm01 = m0/m1
        Tm02 = np.sqrt(m0/m2)

        return Hs,Hrms,Hmean,Tp,Tm01,Tm02

    @extras.timing_decorator
    def compute_spectrum_from_direct_fft(self,signal,kp_correction):
        """
        Computes the density variance spectrum based on the Fast Fourier transform. 
        
        Parameters
        ----------
        signal : list or ndarray
            An array containing the signal
        kp_correction : bool
            If True, applies Kp correction to the spectrum.     
        
        Returns
        -------
        freqs: ndarray
            Frequency of the spectrum
        PSD : ndarray
            Density variance spectrum    
        PSD_kp : ndarray (optional)
            Density variance spectrum corrected by Kp    

        Notes
        -----
        Based on https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html

        """

        length_signal = len(signal)
        freqs = np.fft.rfftfreq(length_signal,1/self.sampling_freq)
        fourier = np.fft.rfft(signal)

        # Compute power spectrum: contribution of each frequency to the total variance in the time series (Parseval's theorem)
        amplitude = np.abs(fourier)
        power_spectrum_raw = (amplitude**2)*2
        power_spectrum_norm =  power_spectrum_raw/(length_signal**2) # power per bin

        # Compute the power spectral density (PSD)
        PSD = power_spectrum_norm * length_signal * (1/self.sampling_freq) # power per Hz

        if kp_correction == False:
            return freqs,PSD
        else:
            return self.correction_by_Kp(freqs,PSD)

    @extras.timing_decorator
    def compute_spectrum_from_welch(self,signal,kp_correction,window_type,window_length,overlap=None):
        """
        Compute PSD using Welch's method and smooth across frequency bins.

        Parameters
        ----------
        signal : ndarray
            1D numpy array containing the signal.
        kp_correction : bool
            If True, applies Kp correction to the spectrum.
        window_type : str, optional
            Type of window to use (default is ``'hamming'``).
            Can be any window name supported by scipy.signal.windows, e.g.,
            ``'hann'``, ``'blackman'``, ``'boxcar'``, etc.
        window_length : int
            Length of the Hamming window in samples.
        overlap: int, optional
            Number of overlapping samples between segments (default is half of window_length).

        Returns
        -------
        freqs : ndarray
            Frequency array.
        PSD : ndarray
            Power spectral density.
        PSD_kp : ndarray (optional)
            Density variance spectrum corrected by Kp
        """

        freqs, PSD = welch(x=signal,fs=self.sampling_freq,window=window_type,
                            nperseg=window_length,
                            noverlap=overlap,
                            scaling='density')
                
        if kp_correction == False:
            return freqs,PSD
        else:
            return self._correction_by_Kp(freqs,PSD)

    def get_spectra_and_params_for_bursts(self, method, kp_correction=True, ig_split=False, freq_split=None, 
                                          window_type=None, window_length=None, overlap=None, smoothing_bins=None):
        """
        Compute wave spectra and integral parameters for each burst in the measurement signal.

        Parameters
        ----------
        method : str
            Spectrum computation method: ``'fft'`` or ``'welch'``.
        kp_correction : bool, optional
            Whether to apply Kp pressure correction. Default is True.
        ig_split : bool, optional
            Whether to compute infragravity and wind wave Hm0 separately. Default is False.
        freq_split : float, optional
            Frequency that separates infragravity from short waves (required if ``ig_split`` is True). Default is None.
        window_type : str, optional
            Window type for Welch method (e.g., ``'hamming'``, ``'hann'``). Default is None.
        window_length : int, optional
            Window length for Welch method in samples. Default is None.
        overlap : int, optional
            Number of overlapping samples for Welch method. Default is None.
        smoothing_bins : int, optional
            Number of bins for moving-average smoothing (Welch only). Default is None.

        Returns
        -------
        wave_spectra_data : dict
            Dictionary with keys:

            - ``S``: ndarray of shape (n_bursts, n_freqs) containing power spectral densities.
            - ``freq``: ndarray of frequency values.
            - ``dir``: empty list (placeholder for directional info).
            - ``time``: DatetimeIndex of hourly timestamps.
        wave_params_data : pd.DataFrame
            Wave parameters indexed by time, with columns:

            - ``Hm0``: Zero-moment wave height [m].
            - ``Hrms``: Root-mean-square wave height [m].
            - ``Hmean``: Mean wave height [m].
            - ``Tp``: Peak period [s].
            - ``Tm01``: Mean period (first moment) [s].
            - ``Tm02``: Mean period (second moment) [s].
            - ``Hm0_ig``: Infragravity wave height [m] (if ig_split is True).
            - ``Hm0_sw``: Short wave height [m] (if ig_split is True).
        
        Raises
        ------
        ValueError
            If ``burstId`` column is missing in the measurement signal.
        """

        if 'burstId' not in self.measured_signal.columns:
            raise ValueError("Measurement signal must contain 'burstId' column.")

        self.measured_signal = self._verify_bursts_in_signal(self.measured_signal)
        hourly_timeindex = self.measured_signal.index.floor('h').unique().sort_values()

        wave_param_names = ["Hm0", "Hrms", "Hmean", "Tp", "Tm01", "Tm02"]
        
        wave_params_data = {param: np.zeros(len(hourly_timeindex)) for param in wave_param_names}
        wave_spectra_data = {"S": [], "dir": [], "freq": None, "time": hourly_timeindex}
        wave_params_data["time"] = hourly_timeindex

        for idx, burst_id in enumerate(self.measured_signal["burstId"].unique()):
            burst_series = self.measured_signal[self.measured_signal["burstId"] == burst_id]
            burst_signal = burst_series[self.surface_level_column]
            freqs, spectrum = self._compute_spectrum_for_burst(burst_signal, method, 
                                                            kp_correction, window_type, 
                                                            window_length, smoothing_bins)
            wave_spectra_data["S"].append(spectrum)

            # Compute wave parameters
            wave_params = self.get_wave_params_from_spectrum(spectrum, freqs)
            for param_idx, param_name in enumerate(wave_param_names):
                wave_params_data[param_name][idx] = wave_params[param_idx]

            if ig_split:
                Hm0_ig, Hm0_sw = self._compute_hs_ig_band(spectrum, freqs, freq_split)
                wave_params_data["Hm0_ig"] = wave_params_data.get("Hm0_ig", np.zeros(len(hourly_timeindex)))
                wave_params_data["Hm0_sw"] = wave_params_data.get("Hm0_sw", np.zeros(len(hourly_timeindex)))
                wave_params_data["Hm0_ig"][idx] = Hm0_ig
                wave_params_data["Hm0_sw"][idx] = Hm0_sw
                
            if idx == 0:
                wave_spectra_data["freq"] = freqs

        wave_spectra_data["S"] = np.array(wave_spectra_data["S"])
        wave_params_data = pd.DataFrame(wave_params_data, index=hourly_timeindex)

        return wave_spectra_data, wave_params_data

    def compute_wavelet_scalograms(self,mother_wavelet,points_scale,burst_mode=False,window_length=None,overlap=None):
        """Compute wavelet scalograms for all bursts in the measurement signal.
        
        Parameters
        ----------
        mother_wavelet : str
            The mother wavelet to use (e.g., ``'morl'``, ``'cmor'``, etc.).
        points_scale : int
            The number of frequency points.
        burst_mode : bool, optional
            If True, computes scalograms for each burst separately using overlapping windows. Default is False.
            If False, computes a single scalogram for the entire measurement signal without windowing.
        window_length : int, optional
            The length of each window in samples (required if ``burst_mode`` is True). Default is None.
        overlap : float, optional
            The overlap between consecutive windows (required if ``burst_mode`` is True). Default is None.

        Returns
        -------
        coefs_all : ndarray
            The computed wavelet scalograms for all bursts.
        freqs : ndarray
            The corresponding frequencies in Hz.
        
        Raises
        ------
        ValueError
            If ``burst_mode`` is True and ``window_length`` or ``overlap`` is not provided.
            If ``burst_mode`` is True and ``'burstId'`` column is missing in the measurement signal.
        
        """

        if burst_mode and (window_length is None or overlap is None):
            raise ValueError("window_length and overlap must be provided when burst_mode is True.")
        if burst_mode and 'burstId' not in self.measured_signal.columns:
            raise ValueError("Measurement signal must contain 'burstId' column when burst_mode is True.")
    
        # scale construction
        frequencies = np.logspace(np.log10(0.001), np.log10(self.sampling_freq/2), points_scale)
        # scales = np.arange(self.sampling_freq*0.5, maximum_scale, 20*int(self.sampling_freq))
        f_c = pywt.central_frequency(mother_wavelet)
        dt = 1 / self.sampling_freq
        scales = f_c / (frequencies * dt)

        if not burst_mode:
            coeffs, freqs = pywt.cwt(self.measured_signal[self.surface_level_column].values,scales,
                                 wavelet=mother_wavelet, sampling_period=1/self.sampling_freq)
            coeffs_mag = np.abs(coeffs)
            return coeffs_mag,freqs
        
        # burst mode
        self.measured_signal = self._verify_bursts_in_signal(self.measured_signal)
        hourly_timeindex = self.measured_signal.index.floor('h').unique().sort_values()

        # scale = np.arange(self.sampling_freq*0.5,maximum_scale,20*int(self.sampling_freq))
        coefs_all = np.zeros((len(hourly_timeindex),len(scales),self.burst_length_s))

        for idx,burst in enumerate(self.measured_signal["burstId"].unique()):
            burst_series = self.measured_signal[self.measured_signal['burstId'] == burst]
            coefs,freqs = self._compute_wavelet_scalogram_for_burst(burst_series[self.surface_level_column].values,window_length,overlap,
                                                                        mother_wavelet,scales)
            coefs_all[idx,:,:] = coefs
        return coefs_all,freqs