import numpy as np
import pandas as pd

from ..utils import wave_props,constants,extras
from scipy.signal import welch
from PyEMD import EMD,EEMD,CEEMDAN
import pywt

class WaveSpectralAnalyzer():
    def __init__(self,measurement_signal,sampling_data):
        """
        Initializes the analysis object with measurement signal and sampling data.

        Parameters
        ----------
        measurement_signal : array-like
            The input signal data to be analyzed.
        sampling_data : dict
            Dictionary containing sampling parameters with the following keys:
                - 'sampling_freq' (float): Sampling frequency of the signal.
                - 'anchoring_depth' (float): Depth at which the sensor is anchored.
                - 'sensor_height' (float): Height of the sensor above the bottom.
                - 'burst_length_s' (float): Duration of each burst in seconds.
        Attributes
        ----------
        measurement_signal : array-like
            Stores the input measurement signal.
        sampling_data : dict
            Stores the sampling parameters.
        sampling_freq : float
            Sampling frequency extracted from sampling_data.
        anchoring_depth : float
            Anchoring depth extracted from sampling_data.
        sensor_height : float
            Sensor height extracted from sampling_data.
        burst_length_s : float
            Burst length in seconds extracted from sampling_data.
        """

        self.measurement_signal = measurement_signal
        self.sampling_data = sampling_data
        self.sampling_freq = self.sampling_data['sampling_freq']
        self.anchoring_depth = self.sampling_data['anchoring_depth']
        self.sensor_height = self.sampling_data['sensor_height']
        self.burst_length_s = self.sampling_data['burst_length_s']

    def smooth_psd_spectrum(self,psd,smoothing_bins):
        """Smooth the power spectral density (PSD) spectrum using a moving average filter."""
        kernel = np.ones(smoothing_bins) / smoothing_bins
        psd_smoothed = np.convolve(psd, kernel, mode='same')
        return psd_smoothed

    def get_wave_params_from_spectrum_v1(self,psd,freqs):
        """
        This function computes different wave integral parameters from the spectrum
        
        Parameters
        ----------
        psd : list or ndarray
            Density variance spectrum
        freq : list or ndarray
            Frequencies of the spectrum
        
        Returns
        -------
        Hs : float
            Significant wave heigth [m]
        Hrms : float
            Root-mean squared wave heigth [m]
        Hmean : float
            Mean wave heigth [m]
        Tp : float
            Peak period [s]
        Tm01 : float
            Mean period - fisrt order [s]
        Tm02 : float
            Mean period - second order [s]
        
        Notes
        -----
        10-Dec-2024 : Origination - Franklin Ayala

        """

        m0 = np.trapz(psd, freqs.flatten())
        m1 = np.trapz(freqs.flatten()*psd, freqs.flatten())
        m2 = np.trapz((freqs.flatten()**2)*psd, freqs.flatten())

        i0 = np.trapz(np.abs(psd)**4, freqs.flatten())
        i1 = np.trapz(freqs.flatten() * np.abs(psd)**4, freqs.flatten())

        Hs = 4.004*np.sqrt(m0)
        Hrms = np.sqrt(8*m0)
        Hmean = np.sqrt(2*np.pi*m0)

        # Tp = i0/i1
        Tp = 1/freqs[np.argmax(psd)]
        Tm01 = m0/m1
        Tm02 = np.sqrt(m0/m2)

        return Hs,Hrms,Hmean,Tp,Tm01,Tm02

    @extras.timing_decorator
    def compute_spectrum_from_direct_fft(self,signal):
        """
        Computes the density variance spectrum based on the Fast Fourier transform. 
        
        Parameters
        ----------
        signal : list or ndarray
            An array of the signal
        sampling_freq : float
            Sampling frequency for the records    
        anchoring_depth : float
            Depth at where the device was settled on the bottom.
        sensor_height: float
            Distance from the sensor to the bottom.        
        
        Returns
        -------
        power : ndarray
            Density variance spectrum    
        power_kp : ndarray
            Density variance spectrum corrected by Kp    
        freq: ndarray
            Frequency of the spectrum

        Notes
        -----
        Based on https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html

        01-Sep-2023 : Origination - Juan Diego Toro

        """
        freq = np.fft.fftfreq(len(signal),1/self.sampling_freq)
        T = 1/freq
        fourier = np.fft.fft(signal)

        # Filtering frequencies 
        target_freqs = np.where((np.abs(freq>=0.0009))&((np.abs(freq<=1))))[0]
        freq = freq[target_freqs]
        fourier= fourier[target_freqs]

        # Density variance spectrum
        amplitude = np.abs(fourier)
        power = (amplitude**2)*2/len(signal)*(1/self.sampling_freq)

        # Correction by Kp
        Kp, Kpmin = self._compute_kp(freq, self.anchoring_depth, self.sensor_height)
        fmax_kp = 1/(2*np.pi)*np.sqrt(9.8*np.pi/(self.anchoring_depth - self.sensor_height)*np.tanh(np.pi/(self.anchoring_depth-self.sensor_height)*self.anchoring_depth))

        power_kp = power / (Kp**2)

        return power,power_kp,freq,T,Kpmin,fmax_kp

    # @extras.timing_decorator
    # def compute_spectrum_from_welch(self,signal,window_type,window_length,overlap=None):
    #     """
    #     Compute PSD using Welch method and smooth across frequency bins.

    #     Parameters
    #     ----------
    #     signal : ndarray
    #         1D numpy array containing the signal.
    #     window_type : str, optional
    #         Type of window to use (default is 'hamming').
    #         Can be any window name supported by scipy.signal.windows, e.g., 
    #         'hann', 'blackman', 'boxcar', etc.
    #     window_length : int
    #         Length of the Hamming window in samples.
    #     smoothing_bins : int
    #         Number of adjacent frequency bins to smooth over.

    #     Returns
    #     -------
    #     freqs : ndarray
    #         Frequency array.
    #     psd_smoothed : ndarray
    #         PSD after smoothing.
    #     dof : float
    #         Estimated degrees of freedom.
    #     """

    #     # Welch PSD
    #     freqs, psd = welch(x=signal,fs=self.sampling_freq,window=window_type,
    #                         nperseg=window_length,
    #                         noverlap=overlap,
    #                         scaling='density')
    #     # Estimate degrees of freedom:
    #     # DOF ≈ (2 × number of segments) × (effective freq bins averaged / total bins)
    #     # n_segments = 1 + (len(signal) - window_length) // (window_length - overlap)
    #     # dof = 2 * n_segments * (1 / smoothing_bins)
    #     return freqs, psd


#     def get_spectra_and_params_for_bursts(self,method,window_type=None,window_length=None,overlap=None,smoothing_bins=None):
#         """
#         Get wave spectra for each burst in the cleaned records.

#         Parameters
#         ----------

#         Returns
#         -------
#         wave_spectra_data : dict
#             Dictionary containing wave spectra data with keys:
#             - 'S': List of power spectral densities for each burst.
#             - 'dir': List of directions (currently not computed, placeholder).
#             - 'freq': List of frequency arrays for each burst.
#             - 'time': List of timestamps corresponding to each burst.
#         wave_params_data : pandas.DataFrame
#             DataFrame containing wave parameters with columns:
#             - 'Hm0': Zero-moment wave height.
#             - 'Hrms': Root mean square wave height.
#             - 'Hmean': Mean wave height.
#             - 'Tp': Peak period.
#             - 'Tm01': Mean period (first moment).
#             - 'Tm02': Mean period (second moment).
#             - Index is the timestamp corresponding to each burst.
#         """

#         hourly_timeindex = self.measurement_signal.index.floor('h').unique().sort_values()
#         wave_params=["Hm0","Hrms","Hmean","Tp","Tm01","Tm02"]
#         wave_params_data={param:np.zeros((hourly_timeindex.shape)) for param in wave_params}

#         wave_spectra_vars=["S","dir","freq","time"]
#         wave_spectra_data={var:[] for var in wave_spectra_vars}

#         wave_spectra_data['time'] = hourly_timeindex
#         wave_params_data['time'] = hourly_timeindex

#         if 'burstId' in self.measurement_signal.columns:
#             for idx,burst in enumerate(self.measurement_signal["burstId"].unique()):
#                 burst_series = self.measurement_signal[self.measurement_signal['burstId'] == burst]
#                 len_burst_series = len(burst_series)

#                 # Create a time index for each expected burst based on the sampling frequency and burst length
#                 # burst_start_time = burst_series.index[0]
#                 # burst_end_time = burst_series.index[-1]
#                 # expected_times = pd.date_range(start=burst_start_time, end=burst_end_time, freq=pd.Timedelta(seconds=1/sampling_data['sampling_freq']))

#                 # # Find which expected times are missing in the burst
#                 # missing_times = expected_times.difference(burst_series.index)
#                 # if not missing_times.empty:
#                 #     print(f"Missing timestamps in burst {hourly_timeindex[idx]}: {missing_times}")

#                 if method == 'fft':
#                     # Compute the spectrum using FFT
#                     power, power_kp, freqs, T, kpmin, fmax_kp = self.compute_spectrum_from_direct_fft(burst_series['eta[m]'].values)
#                     wave_spectra_data["S"].append(power_kp)

#                 elif method == 'welch':
#                     # Compute the spectrum using Welch method
#                     freqs,power = self.compute_spectrum_from_welch(burst_series['eta[m]'].values, window_type, window_length)
#                     power = self.smooth_psd_spectrum(power,smoothing_bins)
#                     wave_spectra_data["S"].append(power)
#                 else:
#                     pass

#                 # Compute wave parameters from the spectrum
#                 Hm0, Hrms, Hmean, Tp, Tm01, Tm02 = self.get_wave_params_from_spectrum_v1(power, freqs)

#                 # Store wave parameters
#                 wave_params_data['Hm0'][idx] = Hm0
#                 wave_params_data['Hrms'][idx] = Hrms
#                 wave_params_data['Hmean'][idx] = Hmean
#                 wave_params_data['Tp'][idx] = Tp
#                 wave_params_data['Tm01'][idx] = Tm01
#                 wave_params_data['Tm02'][idx] = Tm02
            
#             wave_spectra_data['freq'] = freqs
#             wave_spectra_data['S']=np.array(wave_spectra_data['S'])

#             wave_params_data=pd.DataFrame(wave_params_data).set_index('time')
#         return wave_spectra_data,wave_params_data

#     def decompose_into_IMFs_for_bursts(self,emd_type,maximum_IMFs,number_ensembles=None,amplitude_noise_std=None):
#         hourly_timeindex = self.measurement_signal.index.floor('h').unique().sort_values()
#         time_seconds = np.arange(0,self.burst_length_s,1)

#         IMFs_all = np.zeros((len(hourly_timeindex),maximum_IMFs,self.burst_length_s))

#         if 'burstId' in self.measurement_signal.columns:
#             for idx,burst in enumerate(self.measurement_signal["burstId"].unique()):
#                 burst_series = self.measurement_signal[self.measurement_signal['burstId'] == burst]
#                 if emd_type == 'EMD':
#                     emd = EMD()
#                     IMFs = emd(burst_series['eta[m]'].values, time_seconds, max_imf=maximum_IMFs)[:maximum_IMFs, :]
#                 elif emd_type == 'EEMD':
#                     eemd = EEMD(trials=number_ensembles, epsilon=amplitude_noise_std)
#                     IMFs = eemd(burst_series['eta[m]'].values, time_seconds, max_imf=maximum_IMFs)[:maximum_IMFs, :]
#                 else:
#                     ceemd = CEEMDAN(DTYPE=np.float16,trials=number_ensembles,epsilon=amplitude_noise_std,parallel=True,processes=48)
#                     IMFs = ceemd(burst_series['eta[m]'].values,time_seconds,max_imf=maximum_IMFs)[:maximum_IMFs,:]
#                 IMFs_all[idx,:,:] = IMFs
#         return IMFs_all

#     def compute_wavelet_scalogram(self,signal,mother_wavelet,maximum_scale):
#         coefficients, frequencies = pywt.cwt(signal, scales, mother_wavelet, sampling_period=1/self.sampling_freq)
#         return coefficients,frequencies

#     def compute_wavelet_scalogram_windowed(self,signal,window_length,overlap,mother_wavelet,maximum_scale):
#         step = int(window_length * (1 - overlap))
#         if len(signal)==window_length:
#             n_segments =1
#         else:
#             n_segments = (len(signal) - window_length) // step + 1
#         window = np.hanning(window_length)

#         scales = np.arange(1, maximum_scale, 20)
#         stitched = np.zeros((len(scales), len(signal)))
#         weight = np.zeros(len(signal))

#         for idx_seg in range(n_segments):
#             start = idx_seg * step
#             end = start + window_length
#             segment = signal[start:end]
#             coef, freqs = pywt.cwt(segment, scales, mother_wavelet, sampling_period=1/self.sampling_freq)
#             coef_mag = np.abs(coef) * window  # Apply window to smooth overlap
#             stitched[:, start:end] += coef_mag
#             weight[start:end] += window

#         stitched /= np.maximum(weight, 1e-8)
#         return stitched,freqs

#     def compute_all_wavelet_scalograms(self,window_length,overlap,mother_wavelet,maximum_scale):
#         hourly_timeindex = self.measurement_signal.index.floor('h').unique().sort_values()
#         scale = np.arange(1,maximum_scale,20)
#         coefs_all = np.zeros((len(hourly_timeindex),len(scale),self.burst_length_s))

#         if 'burstId' in self.measurement_signal.columns:
#             for idx,burst in enumerate(self.measurement_signal["burstId"].unique()):
#                 burst_series = self.measurement_signal[self.measurement_signal['burstId'] == burst]
#                 coefs,freqs = self.compute_wavelet_scalogram_windowed(burst_series['eta[m]'].values,window_length,overlap,
#                                                                             mother_wavelet,maximum_scale)
#                 coefs_all[idx,:,:] = coefs
#         return coefs_all,freqs

#     @staticmethod
#     def _compute_kp(freqs, anchoring_depth, sensor_height):
#         L = [wave_props.wavelength(1/f, anchoring_depth) for f in freqs]
#         k = [2*np.pi/l for l in L]
#         Kp = np.array([np.cosh(ki*sensor_height)/np.cosh(ki*anchoring_depth) for ki in k])
#         Kpmin = (np.cosh(np.pi/(anchoring_depth - sensor_height)*sensor_height)) / \
#                 (np.cosh(np.pi/(anchoring_depth - sensor_height)*anchoring_depth))
#         Kp = np.clip(Kp, Kpmin, 10)
#         return Kp, Kpmin