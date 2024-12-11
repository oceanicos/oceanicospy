import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import detrend
import pandas as pd

from ..utils import constants
from ..analysis import spectral,temporal

import warnings
# Suppress all warnings globally
warnings.filterwarnings('ignore')


class Awac():
    """
    A class to handle reading and processing the data files recorded by an ADCP AWAC (Nortek S.A). 

    Notes
    -----
    04-Jan-2018 : Origination - Daniel Peláez
    01-Sep-2023 : Migration to Python - Alejandro Henao
    10-Dec-2024 : Class implementation - Franklin Ayala

    """
    def __init__(self,directory_path,sampling_data,obs_name):
        """
        Initializes the Awac class with the given directory path, sampling data, and observation name.

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
        
    def reading_header(self):
        """
        Reads and parses the header file (.hdr) to extract the column names.

        The function filters and processes the header information to generate a list of column names.

        Returns
        -------
        numpy.ndarray
            An array of column names extracted from the .hdr file.
        """
    
        self.header=glob.glob(self.directory_path+'*.hdr') # File with headers
        self.headers=open(self.header[0],'r')
        self.headers=self.headers.read().split('\n')

        # Replacing the title with a mask
        self.tf=[]
        self.control=False
        for i in self.headers:
                
            if self.control==True:
                if i.endswith('-')==True:
                    self.control=False
                    self.tf.append(False)
                else:
                    self.tf.append(True)
            else:
                self.tf.append(False)
                
            if i.endswith('.wad]')==True:
                self.control=True
        self.headers=list(np.array(self.headers)[self.tf])
        self.columns=np.array([' '.join(i.split()) for i in self.headers])
        if self.columns[-1]=='':
            self.columns=self.columns[:-1]
        return self.columns

    def reading_records(self):
        """
        Reads and processes the .wad files to create a DataFrame containing the burst data.

        For each .wad file, the function reads the data, adds a 'burstId' column, and combines all the data into a single DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the concatenated data from all the .wad files with an added 'burstId' column.
        """

        self.columns_=self.reading_header()
        self.data=sorted(glob.glob(self.directory_path+'*.wad')) #Each .wad file represents one burst

        self.wads=[]
        self.burst_id=1
        for i in self.data:
            self.e = pd.read_csv(i,header=0,delim_whitespace=True,names=self.columns_)
            self.e['burstId']=(np.ones(len(self.e))*self.burst_id).astype(int)
            self.burst_id+=1
            self.wads.append(self.e.dropna())
        self.wads=pd.concat(self.wads)

        self.columns_=np.append(self.columns_,['burstId'])

        return self.wads
    
    def getting_clean_records(self):
        """
        Processes the raw data by converting certain columns to numeric types, adding a timestamp, and filtering the data 
        by the specified time range.

        The function also renames columns and returns a cleaned DataFrame that includes only relevant data.

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame containing 'pressure', 'u', 'v', and 'burstId' columns, filtered by the specified time range.
        """

        self.wads=self.reading_records()
        self.wads.iloc[:,[11,12]]=self.wads.iloc[:,[11,12]].astype(float)
        self.raw_data=self.wads.iloc[:,[0,1,2,3,4,5,6,11,12,17]]

        self.raw_data['date'] = pd.to_datetime(self.raw_data.iloc[:,[2,0,1,3,4,5]].astype(str).agg('-'.join, axis=1),
                                format='%Y-%m-%d-%H-%M-%S.%f',errors='ignore')
        self.raw_data=self.raw_data.set_index('date')
        self.clean_data=self.raw_data[self.columns_[[6,11,12,17]].tolist()]
        self.clean_data.columns=['pressure','u','v','burstId']
        self.clean_data['pressure']=self.clean_data['pressure'].astype(float)
        self.clean_data=self.clean_data[self.sampling_data['start_time']:self.sampling_data['end_time']]

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
        self.clean_data[self.clean_data.columns[[0,1,2,4]]] = self.clean_data.groupby('burstId')[self.clean_data.columns[[0,1,2,4]]].transform(lambda x: x - x.mean())

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

            H13, Tm, Lm, Hmax = temporal.zero_crossing(self.burst_series_detrended['pressure'], self.sampling_data['sampling_freq'],
                                    self.sampling_data['anchoring_depth'], self.sampling_data['sensor_height'])

            self.wave_params_data['time'].append(self.burst_series_detrended.index[0])
            self.wave_params_data['H1/3'].append(H13)
            self.wave_params_data['Tmean'].append(Tm)

        self.wave_params_data=pd.DataFrame(self.wave_params_data).set_index('time')

        return self.wave_params_data
    
    def getting_currents(self,filename,measurement_origin=0,distance_to_sensor=0.5):

        self.datos = np.loadtxt(os.path.join(self.directory_path,filename))

        # Number of cells (assumed to be stored in the 19th column of 'datos')
        self.num_cells = int(self.datos[0, 18])

        # Separate metadata profiles from the main data
        self.metadata = self.datos[::(self.num_cells + 1), :]

        # Identify the profiles where the 19th column value is 0
        self.zero_value_profiles = np.where(self.datos[:, 18] == 0)[0]
        self.profiles = self.datos[self.zero_value_profiles, :]
        