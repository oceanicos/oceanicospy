from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import detrend

import pandas as pd
import numpy as np
import glob

import warnings
# Suppress all warnings globally
warnings.filterwarnings('ignore')

from ..analysis import *
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

        print(self.raw_data)
        self.raw_data[self.raw_data['UNITS']=='BURSTSTART']

        # self.raw_data['depth']=((self.raw_data['pressure']-ATM_PRESSURE_BAR)*10000)/(WATER_DENSITY*GRAVITY)
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

        self.clean_data['Day-hour']=[str(i.day)+'-'+str(i.hour) for i in self.clean_data.index]
        self.clean_data=self.clean_data.drop(['UNITS'],axis=1)

        mean_depths=self.clean_data.groupby('Day-hour').mean().drop(['P[bar]','T[C]'],axis=1)

        self.clean_data['mean_depth']=[mean_depths['Depth[m]'][i] for i in self.clean_data['Day-hour']]

        self.clean_data['n']=self.clean_data['Depth[m]']-self.clean_data['mean_depth']

        return self.clean_data