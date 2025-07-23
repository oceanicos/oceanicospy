import numpy as np
import pandas as pd

from scipy.signal import resample,detrend

from ..utils import wave_props

class WaveTemporalAnalyzer:
    """
    Class for analyzing wave parameters from temporal data.
    """

    def __init__(self,measurement_signal,sampling_data): 
        """"
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
        """
        self.measurement_signal = measurement_signal
        self.sampling_data = sampling_data

    def params_from_zero_crossing(self, clean_records, sampling_data):
        """
        Calculate wave parameters from zero-crossing analysis of pressure data.

        Parameters
        ----------
        clean_records : pandas.DataFrame
            DataFrame containing cleaned pressure records with a 'burstId' column to identify different bursts.
        sampling_data : dict
            Dictionary containing sampling information with keys:
            - 'sampling_freq': Sampling frequency of the pressure data.
            - 'anchoring_depth': Depth at which the sensor is anchored.
            - 'sensor_height': Height of the sensor from the seabed.

        Returns
        -------
        pandas.DataFrame
            DataFrame with wave parameters indexed by time, containing columns:
            - 'H1/3': Significant wave height (H1/3).
            - 'Tmean': Mean wave period (Tmean).
        """
        wave_params=["time","H1/3","Tmean"]
        wave_params_data={param:[] for param in wave_params}

        clean_data=clean_records.copy()

        for i in clean_data['burstId'].unique():
            burst_series=clean_data[clean_data['burstId']==i]

            burst_series_detrended = burst_series.iloc[:,:-1].apply(lambda x: detrend(x,type='constant'), axis=0)
            burst_series_detrended[clean_records.columns[-1]] = burst_series.iloc[:, -1]

            H13, Tm, Lm, Hmax = self.zero_crossing(burst_series_detrended['pressure'], sampling_data['sampling_freq'],
                                    sampling_data['anchoring_depth'], sampling_data['sensor_height'])

            wave_params_data['time'].append(burst_series_detrended.index[0])
            wave_params_data['H1/3'].append(H13)
            wave_params_data['Tmean'].append(Tm)

        wave_params_data=pd.DataFrame(wave_params_data).set_index('time')

        return wave_params_data

    def zero_crossing(self, burst, fs, h, zp):
        """
        This function calculates the significant wave height, the period and the wavelength
        with the zero-crossing method.
        
        Parameters
        ----------
        burst : array_like
            A series of data without trend.
        fs : float
            The sampling frequency.
        h : float
            The measurement depth.
        zp : float
            The distance from the bottom to the sensor.

        Returns
        -------
        Hs : float
            The significant wave height.
        Tm : float
            The mean period.

        Notes
        -----

        23-Feb-2014 : First Matlab version - Daniel Peláez
        01-Sep-2023 : First Python version - Alejandro Henao
        10-Dec-2024 : Polishing            - Franklin Ayala 

        """

        tt = np.arange(1,len(burst)+1,1/fs)
        ratio = 100
        pp = resample(burst, len(burst) * ratio)  # Resample p by the ratio
        tt = np.linspace(1, 1024, len(pp)+1)  # Create the time vector
        sign = np.sign(burst)
        index_cross = np.where(np.diff(sign) > 0)[0]

        Hp = []
        T = []
        for p in range(0,len(index_cross)-1):
            a = index_cross[p]
            b = index_cross[p+1]
            Hp.append(np.max(burst[a:b+1])-np.min(burst[a:b+1]))
            T.append(tt[b]-tt[a])
        Hp = np.array(Hp)
        T = np.array(T)

        # Determine the wavenumber based on the dispersion relation
        L=np.array([wave_props.wavelength(t,h) for t in T],dtype=np.float64)
        k = 2.*np.pi/L

        # Transference factor Kp
        Kp=np.cosh(k*zp)/np.cosh(k*h)
        Kpmin=(np.cosh(np.pi/(h-zp)*zp))/(np.cosh(np.pi/(h-zp)*h))
        for i in range(0,len(Kp)):
            if (Kp[i]<Kpmin):
                Kp[i]=Kpmin

        H = Hp/(Kp)
        H0 = np.sort(H)[::-1]
        H13 = np.nanmean(H[:int(len(H0)/3)])
        Hmx = H0[0]

        Tm = np.nanmean(T)
        Lm = np.nanmean(L)
        return (H13,Tm,Lm,Hmx)