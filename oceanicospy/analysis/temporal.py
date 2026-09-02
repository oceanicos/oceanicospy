import numpy as np
import pandas as pd
from PyEMD import EMD, EEMD, CEEMDAN
from scipy.signal import resample,detrend

from ..utils import wave_props

class WaveTemporalAnalyzer:
    def __init__(self,measured_signal,sampling_data,surface_level_column='eta[m]',zero_centered=False): 
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
        surface_level_column : str, optional
            The name of the column in measured_signal that contains surface level data (default is ``eta[m]``).
        zero_centered: bool
            Indicates whether the measured signal is centred at zero; if not, the signal is centred at zero (default is ``False``).

        Notes
        -----

        **Development history**
        
        - 23-Feb-2014 : First Matlab version - Daniel Peláez
        - 01-Sep-2023 : First Python version - Alejandro Henao
        - 10-Dec-2025 : Empirical Mode Decomposition - Franklin Ayala
        """

        self.measured_signal = measured_signal
        self.sampling_data = sampling_data
        self.burst_length_s = self.sampling_data['burst_length_s']
        self.surface_level_column = surface_level_column
        self.zero_centered = zero_centered

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
        Verify that each burst in the measured signal has the expected number of samples. If not, remove the burst and print a warning.

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
            print(f"The following bursts have been removed due to incorrect length: {burst_to_delete}")
            measured_signal = measured_signal[~measured_signal["burstId"].isin(burst_to_delete)]
        return measured_signal       

    def apply_zero_upcrossing_burst(self, burst_signal, anchoring_depth, sensor_height):
        """
        This function calculates the significant wave height, the period, and the wavelength
        with the zero-upcrossing method.
        
        Parameters
        ----------
        burst_signal : array_like
            A series of data without a trend.
        anchoring_depth : float
            The measurement depth.
        sensor_height : float
            The distance from the bottom to the sensor.

        Returns
        -------
        H_top_third : float
            The significant wave height (top third).
        Hmax : float
            The maximum wave height.
        Tmean : float
            The mean period.
        Lmean : float
            The mean wavelength.
        """

        ratio = 10
        # Resample the signal with a certain ratio to increase the resolution of zero-crossing detection
        burst_signal_upsampled = resample(burst_signal, len(burst_signal) * ratio)
        time = np.linspace(1, len(burst_signal), len(burst_signal_upsampled)) 
        sign = np.sign(burst_signal_upsampled)
        idxs_cross = np.where(np.diff(sign) > 0)[0] # indices of zero-crossings (up-crossings)

        H_cross = []
        T = []

        # Loop through pairs of consecutive up-crossings to calculate wave heights and periods
        for idx_cross in range(len(idxs_cross)-1):
            first_up = idxs_cross[idx_cross] 
            second_up = idxs_cross[idx_cross+1]
            wave = burst_signal_upsampled[first_up:second_up+1]
            H_cross.append(np.max(wave) - np.min(wave))
            T.append(time[second_up] - time[first_up])
        H_cross = np.array(H_cross)
        T = np.array(T)

        # Determine the wavenumber based on the dispersion relation
        L = np.array([wave_props.wavelength(t, anchoring_depth+sensor_height) for t in T], dtype=np.float64) 
        k = 2.*np.pi/L

        # Computing non-adaptive transference factor Kp
        Kp = np.cosh(k * sensor_height) / np.cosh(k * anchoring_depth) 
        Kp_min = (np.cosh(np.pi/(anchoring_depth - sensor_height)*sensor_height)) / \
                (np.cosh(np.pi/(anchoring_depth - sensor_height)*anchoring_depth)) 
        # Clip Kp to avoid unrealistic amplification of wave heights for very long waves
        Kp = np.clip(Kp, Kp_min, 1)

        H = H_cross/(Kp)
        n = min(1,len(H) // 3) # to prevent errors when there are fewer than 3 waves in the burst

        H_sorted = np.sort(H)
        H_top_third = np.nanmean(H_sorted[-n:])  # top third
        Hmax = H_sorted[-1]
        Tmean = np.nanmean(T)
        Lmean= np.nanmean(L)

        return H_top_third,Hmax,Tmean,Lmean

    def compute_params_from_zero_upcrossing(self):
        """
        Calculate wave parameters from zero-crossing analysis of pressure data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with wave parameters indexed by time, containing columns:

            - ``H1/3``: Significant wave height [m].
            - ``Tmean``: Mean wave period [s].
        """
        wave_params = ["time","H1/3","Tmean"]
        wave_params_data = {param:[] for param in wave_params}

        self.measured_signal = self._verify_bursts_in_signal(self.measured_signal)

        for i in self.measured_signal['burstId'].unique():
            burst_signal = self.measured_signal[self.measured_signal['burstId'] == i]

            if self.zero_centered:
                burst_signal_detrended = burst_signal.copy()
            else:
                burst_signal_detrended = burst_signal.iloc[:,:-1].apply(lambda x: detrend(x, type='constant'), axis=0)
                burst_signal_detrended[self.measured_signal.columns[-1]] = burst_signal.iloc[:, -1]

            H_top_third, Hmax, Tmean, Lmean = self.apply_zero_upcrossing_burst(burst_signal_detrended[self.surface_level_column].values,
                                    self.sampling_data['anchoring_depth'], self.sampling_data['sensor_height'])

            wave_params_data['time'].append(burst_signal_detrended.index[0])
            wave_params_data['H1/3'].append(H_top_third)
            wave_params_data['Tmean'].append(Tmean)

        wave_params_data=pd.DataFrame(wave_params_data).set_index('time')

        return wave_params_data

    def decompose_into_IMFs_for_bursts(self,EMD_type,maximum_IMFs,number_ensembles=None,amplitude_noise_std=None,
                                       parallel=True,nb_processes=2):
        """
        Decomposes the signal into Intrinsic Mode Functions (IMFs) for each burst using Empirical Mode Decomposition (EMD) 
        or its variants.

        Parameters
        ----------
        EMD_type : str
            The type of EMD to use. Options are ``'EMD'``, ``'EEMD'``, or ``'CEEMDAN'``.
        maximum_IMFs : int
            The maximum number of IMFs to compute for each burst.
        number_ensembles : int, optional
            The number of ensembles to use for EEMD or CEEMDAN (required if ``EMD_type`` is ``'EEMD'`` or ``'CEEMDAN'``).
        amplitude_noise_std : float, optional
            The standard deviation of the added noise for EEMD or CEEMDAN.
        parallel : bool, optional
            Whether to use parallel processing for CEEMDAN (default is True).
        nb_processes : int, optional
            The number of processes to use for parallel processing in CEEMDAN (default is 2).
        
        Returns
        -------
        numpy.ndarray
            A 3D array containing the IMFs for each burst, with shape (number of bursts, maximum_IMFs, burst_length_s).

        Notes
        -----
        This function is based on the decomposition methods implemented in the PyEMD library [1]_.

        .. [1] Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear
            and non-stationary time series analysis. Proceedings of the Royal Society of London. Series A, 454(1971), 903-995.

        Raises
        ------
        ValueError
            If ``'burstId'`` is not found in the signal columns.
            If ``EMD_type`` is not one of ``'EMD'``, ``'EEMD'``, or ``'CEEMDAN'``.
            If ``EMD_type`` is ``'EEMD'`` or ``'CEEMDAN'`` and ``number_ensembles`` or
            ``amplitude_noise_std`` are not provided.
        """

        if 'burstId' not in self.measured_signal.columns:
            raise ValueError("'burstId' column not found in measured_signal.")

        valid_emd_types = {'EMD', 'EEMD', 'CEEMDAN'}
        if EMD_type not in valid_emd_types:
            raise ValueError(f"EMD_type must be one of {valid_emd_types}, got '{EMD_type}'.")

        if EMD_type in {'EEMD', 'CEEMDAN'}:
            if number_ensembles is None or amplitude_noise_std is None:
                raise ValueError(
                    f"'number_ensembles' and 'amplitude_noise_std' are required for {EMD_type}."
                )

        self.measured_signal = self._verify_bursts_in_signal(self.measured_signal)
        burst_ids = self.measured_signal['burstId'].unique()
        n_bursts = len(burst_ids)
        time_seconds = np.arange(0,self.sampling_data['burst_length_s'],1)
        IMFs_all = np.zeros((n_bursts,maximum_IMFs,self.sampling_data['burst_length_s']))

        # --- Decompose each burst ---
        for idx, burst_id in enumerate(burst_ids):
            burst_series = self.measured_signal[self.measured_signal['burstId'] == burst_id]
            signal = burst_series[self.surface_level_column].values

            IMFs = self._compute_IMFs(
                signal, time_seconds, EMD_type,
                maximum_IMFs, number_ensembles,
                amplitude_noise_std, parallel, nb_processes
            )

            # Pad with zeros if fewer IMFs than maximum_IMFs were produced
            n_imfs_found = IMFs.shape[0]
            IMFs_all[idx, :n_imfs_found, :] = IMFs[:, :self.sampling_data['burst_length_s']]

        return IMFs_all

    def _compute_IMFs(self,signal,time_seconds,EMD_type,maximum_IMFs,number_ensembles,amplitude_noise_std,
                      parallel,nb_processes):
        """
        Internal helper: runs the appropriate EMD variant on a single burst signal.

        Returns
        -------
        numpy.ndarray
            2D array of shape (n_imfs, n_samples).
        """
        if EMD_type == 'EMD':
            decomposer = EMD()

        elif EMD_type == 'EEMD':
            decomposer = EEMD(trials=number_ensembles, epsilon=amplitude_noise_std)

        else:
            # np.float16 is used to reduce memory usage during ensemble averaging
            decomposer = CEEMDAN(DTYPE=np.float16,trials=number_ensembles,epsilon=amplitude_noise_std,
                                 parallel=parallel,processes=nb_processes)

        return decomposer(signal, time_seconds, max_imf=maximum_IMFs)[:maximum_IMFs, :]
