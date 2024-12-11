from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import detrend

import pandas as pd
import numpy as np
import glob

import warnings
# Suppress all warnings globally
warnings.filterwarnings('ignore')

from ..analysis import spectral,temporal
from ..utils import constants

class AQUAlogger():
    """
    A class to handle reading and processing the data files recorded by AQUAlogger. 

    Notes
    -----
    10-Dec-2024 : Origination - Franklin Ayala

    """
    def __init__(self,directory_path,sampling_data,obs_name):
        """
        Initializes the AQUAlogger class with the given directory path, sampling data, and observation name.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the .hdr and .wad files.
        sampling_data : dict
            Dictionary containing the information about the device installation
        obs_name : str
            The observation name.
        """
        self.directory_path = directory_path
        self.sampling_data=sampling_data
        self.obs_name=obs_name
        
    def reading_records(self):
        """
        Reads the .csv file from the device to create a DataFrame containing data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the concatenated data from all the .wad files with an added 'burstId' column.
        """

        # Write a conditional to know whether or not the depth series has already been calculated with the device software

        self.filepath=glob.glob(self.directory_path+'*.csv')[0]
        self.raw_data=pd.read_csv(self.filepath,names=['UNITS','date','Raw1','T[C]','Raw2','P[bar]','Raw3','Depth[m]','nan'],
                                    header=21,encoding='latin-1')
      
        self.raw_data['date']=pd.to_datetime(self.raw_data['date'])
        self.raw_data=self.raw_data.drop(['Raw1','Raw2','Raw3','nan'],axis=1)
        self.raw_data = self.raw_data.set_index('date')
        self.raw_data=self.raw_data[self.sampling_data['start_time']:self.sampling_data['end_time']]

        # self.raw_data['depth']=((self.raw_data['P[bar]']-constants.ATM_PRESSURE_BAR)*10000)/(constants.WATER_DENSITY*constants.GRAVITY)
        return self.raw_data

    def getting_clean_records(self):
        """
        Processes the raw data by grouping the series per each burst

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame containing the columns '....', filtered by the specified time range.
        """

        self.clean_data=self.reading_records()

        # Initialize burstId column
        self.clean_data['burstId'] = 0

        # Iterate through the 'UNITS' column and increment burstId when 'BURSTSTART' is found
        self.burst_count = 0
        for index, row in self.clean_data.iterrows():
            if row['UNITS'] == 'BURSTSTART':
                self.burst_count += 1
            self.clean_data.loc[index:index+timedelta(seconds=1024), 'burstId'] = self.burst_count

        self.clean_data = self.clean_data.drop(['UNITS'],axis=1)   
        self.clean_data[self.clean_data.columns[:-1]] = self.clean_data.groupby('burstId')[self.clean_data.columns[:-1]].transform(lambda x: x - x.mean())
        self.clean_data_detrended = self.clean_data

        return self.clean_data

    def spectra_from_puv(self,clean_records):
        self.wave_params=["time","Hm0","Hrms","Hmean","Tp","Tm01","Tm02"]
        self.wave_params_data={param:[] for param in self.wave_params}

        self.wave_spectra_vars=["S","dir","freq","time"]
        self.wave_spectra_data={var:[] for var in self.wave_spectra_vars}

        self.clean_data=clean_records

        for i in self.clean_data['burstId'].unique():

            self.burst_series = self.clean_data[self.clean_data['burstId'] == i]

            self.burst_series_detrended = self.burst_series.iloc[:,:-1].apply(lambda x: detrend(x,type='constant'), axis=0)
            self.burst_series_detrended[clean_records.columns[-1]] = self.burst_series.iloc[:, -1]

            # Compute the spectrum
            power, direction, freqs, Su, Sv = spectral.spectrum_puv_method(self.burst_series_detrended['pressure'],self.burst_series_detrended['u'],
                                                                            self.burst_series_detrended['v'],self.sampling_data['sampling_freq'], 
                                                                            self.sampling_data['anchoring_depth'], self.sampling_data['sensor_height'])                                                     
            self.wave_spectra_data["S"].append(power)
            self.wave_spectra_data["dir"].append(direction)
            self.wave_spectra_data["freq"].append(freqs)
            self.wave_spectra_data["time"].append(self.burst_series_detrended.index[0])

            # Compute wave parameters from the spectrum
            Hm0, Hrms, Hmean, Tp, Tm01, Tm02 = spectral.wave_params_from_spectrum_v1(power, freqs)

            # Store wave parameters
            self.wave_params_data['time'].append(self.burst_series_detrended.index[0])
            self.wave_params_data['Hm0'].append(Hm0)
            self.wave_params_data['Hrms'].append(Hrms)
            self.wave_params_data['Hmean'].append(Hmean)
            self.wave_params_data['Tp'].append(Tp)
            self.wave_params_data['Tm01'].append(Tm01)
            self.wave_params_data['Tm02'].append(Tm02)

        self.wave_params_data=pd.DataFrame(self.wave_params_data).set_index('time')

        return self.wave_spectra_data,self.wave_params_data

    def spectra_from_fft(self,clean_records):

        # The depth is computed dividing the pressure by the density and gravity. 
        # The atmospheric pressure is also subtracted and it is converted from bars to pascals.
        self.clean_data=clean_records.copy()

        if np.all(['depth' not in column.lower() for column in self.clean_data.columns]):
            self.clean_data['depth']=((self.clean_data['pressure']-constants.ATM_PRESSURE_BAR)*10000)/(constants.WATER_DENSITY*constants.GRAVITY)

        print(self.clean_data)
   
        # To eliminate the trend of the series, it is grouped by burst and the average prof of each burst is found.        
        self.clean_data[self.clean_data.columns[:-1]] = self.clean_data.groupby('burstId')[self.clean_data.columns[:-1]].transform(lambda x: x - x.mean())

        # Subtracting the mean depth in each burst
        try:
            self.clean_data['n']=self.clean_data['depth']
        except:
            self.clean_data['n']=self.clean_data['Depth[m]']

        self.wave_params=["time","Hm0","Hrms","Hmean","Tp","Tm01","Tm02"]
        self.wave_params_data={param:[] for param in self.wave_params}

        self.wave_spectra_vars=["S","dir","freq","time"]
        self.wave_spectra_data={var:[] for var in self.wave_spectra_vars}

        for i in self.clean_data['burstId'].unique():
            self.burst_series = self.clean_data[self.clean_data['burstId'] == i]

            # Compute the spectrum
            power, power_kp, freqs, T, kpmin, fmax_kp = spectral.spectrum_from_surflevel(self.burst_series['n'][::2], self.sampling_data['sampling_freq']/2,
                                                                                    self.sampling_data['anchoring_depth'], 
                                                                                    self.sampling_data['sensor_height'])
            self.wave_spectra_data["S"].append(power_kp)
            self.wave_spectra_data["time"].append(self.burst_series.index[0])
            self.wave_spectra_data["freq"].append(freqs)

            # Compute wave parameters from the spectrum
            Hm0, Hrms, Hmean, Tp, Tm01, Tm02 = spectral.wave_params_from_spectrum_v1(power, freqs)

            # Store wave parameters
            self.wave_params_data['time'].append(self.burst_series.index[0])
            self.wave_params_data['Hm0'].append(Hm0)
            self.wave_params_data['Hrms'].append(Hrms)
            self.wave_params_data['Hmean'].append(Hmean)
            self.wave_params_data['Tp'].append(Tp)
            self.wave_params_data['Tm01'].append(Tm01)
            self.wave_params_data['Tm02'].append(Tm02)

        self.wave_params_data=pd.DataFrame(self.wave_params_data).set_index('time')

        return self.wave_spectra_data,self.wave_params_data

    def params_from_zero_crossing(self,clean_records):
        self.wave_params=["time","H1/3","Tmean"]
        self.wave_params_data={param:[] for param in self.wave_params}

        self.clean_data=clean_records.copy()

        for i in self.clean_data['burstId'].unique():
            self.burst_series=self.clean_data[self.clean_data['burstId']==i]

            self.burst_series_detrended = self.burst_series.iloc[:,:-1].apply(lambda x: detrend(x,type='constant'), axis=0)
            self.burst_series_detrended[clean_records.columns[-1]] = self.burst_series.iloc[:, -1]

            try:
                H13, Tm, Lm, Hmax = temporal.zero_crossing(self.burst_series_detrended['pressure'], self.sampling_data['sampling_freq'],
                                        self.sampling_data['anchoring_depth'], self.sampling_data['sensor_height'])
            except:
                H13, Tm, Lm, Hmax = temporal.zero_crossing(self.burst_series_detrended['P[bar]'], self.sampling_data['sampling_freq'],
                                        self.sampling_data['anchoring_depth'], self.sampling_data['sensor_height'])
                

            self.wave_params_data['time'].append(self.burst_series_detrended.index[0])
            self.wave_params_data['H1/3'].append(H13)
            self.wave_params_data['Tmean'].append(Tm)

        self.wave_params_data=pd.DataFrame(self.wave_params_data).set_index('time')

        return self.wave_params_data
