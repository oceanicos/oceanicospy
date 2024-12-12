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
    def __init__(self,directory_path,sampling_data):
        """
        Initializes the AQUAlogger class with the given directory path, sampling data, and observation name.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the .hdr and .wad files.
        sampling_data : dict
            Dictionary containing the information about the device installation
        """
        self.directory_path = directory_path
        self.sampling_data=sampling_data
        
    def read_records(self):
        """
        Reads the .csv file from the device to create a DataFrame containing data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the concatenated data from all the .wad files with an added 'burstId' column.
        """

        # Write a conditional to know whether or not the depth series has already been calculated with the device software

        self.filepath=glob.glob(self.directory_path+'*.csv')[0]
        self.raw_data=pd.read_csv(self.filepath,names=['UNITS','date','Raw1','temperature','Raw2','pressure','Raw3','depth','nan'],
                                    header=21,encoding='latin-1')
      
        self.raw_data['date']=pd.to_datetime(self.raw_data['date'])
        self.raw_data=self.raw_data.drop(['Raw1','Raw2','Raw3','nan'],axis=1)
        self.raw_data = self.raw_data.set_index('date')
        self.raw_data=self.raw_data[self.sampling_data['start_time']:self.sampling_data['end_time']]

        # self.raw_data['depth']=((self.raw_data['P[bar]']-constants.ATM_PRESSURE_BAR)*10000)/(constants.WATER_DENSITY*constants.GRAVITY)
        return self.raw_data

    def get_clean_records(self):
        """
        Processes the raw data by grouping the series per each burst

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame containing the columns '....', filtered by the specified time range.
        """

        self.clean_data=self.read_records()

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