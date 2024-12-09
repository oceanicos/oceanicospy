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
    def __init__(self,directory_path,sampling_data,obs_name):
        self.directory_path = directory_path
        self.sampling_data=sampling_data
        self.obs_name=obs_name
        
    def reading_records(self):

        # Esscribir un condicional para saber si ya trae o no la serie de profundidad calculada.

        self.filepath=glob.glob(self.directory_path+'*.csv')[0]
        self.raw_data=pd.read_csv(self.filepath,names=['UNITS','date','Raw1','T[C]','Raw2','P[bar]','Raw3','Depth[m]','nan'],
                                    header=21,encoding='latin-1')
      
        self.raw_data['date']=pd.to_datetime(self.raw_data['date'])
        self.raw_data=self.raw_data.drop(['Raw1','Raw2','Raw3','nan'],axis=1)
        self.raw_data = self.raw_data.set_index('date')
        self.raw_data=self.raw_data[self.sampling_data['start_time']:self.sampling_data['end_time']]

        self.raw_data[self.raw_data['UNITS']=='BURSTSTART']

        # #Calculamos la profundidad dividiendo la presion por la densidad y la gravedad
        # #Se resta la presion atmosferica, se pasa de bar a Pascales!
        # self.raw_data['depth']=((self.raw_data['pressure']-ATM_PRESSURE_BAR)*10000)/(WATER_DENSITY*GRAVITY)
        return self.raw_data

    def getting_clean_records(self):
        self.clean_data=self.reading_records()

        #Para eliminar la tendencia de la serie se agrupara por burst y se halla su prof media de cada uno
        self.clean_data['Day-hour']=[str(i.day)+'-'+str(i.hour) for i in self.clean_data.index]
        self.clean_data=self.clean_data.drop(['UNITS'],axis=1)

        mean_depths=self.clean_data.groupby('Day-hour').mean().drop(['P[bar]','T[C]'],axis=1)

        self.clean_data['mean_depth']=[mean_depths['Depth[m]'][i] for i in self.clean_data['Day-hour']]

        # #Restamos la profundidad media a cada burst
        self.clean_data['n']=self.clean_data['Depth[m]']-self.clean_data['mean_depth']

        return self.clean_data


# campaigns_directories=['/homes/medellin/ffayalac/data/LR1_LittleReef-out/']

# sampling_data_LR1_2019=dict(anchoring_depth=11.6,sensor_height=1.20,sampling_freq=1,
#                             start_time=datetime(2019,11,16,13,0,0),end_time=datetime(2019,11,20,8,0,0))

# sampling_data_list=[sampling_data_LR1_2019]
# metadata_list=['aqualogger_LR1_2019']

# for idx,path_folder in enumerate(campaigns_directories):
#     print(metadata_list[idx])
#     AWAC_measurements=AQUAlogger(path_folder,sampling_data_list[idx],metadata_list[idx])
#     records=AWAC_measurements.getting_clean_records()}
