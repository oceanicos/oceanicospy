import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import numpy as np
import glob as glob
import os

from .. import utils
from ..init_setup import InitialSetup
from ....retrievals import *

class WaterLevelForcing(InitialSetup):
    """
    Class representing wind forcing in SWAN model.
    Args:
        input_filename (str): Path to the input file.
    Attributes:
        input_filename (str): Path to the input file.
        wind_params (dict): Dictionary containing wind parameters.
    Methods:
        fill_wl_section: Fills the water level section in the SWAN input file.
    """
    def __init__ (self,domain_number,wl_info=None,input_filename=None,*args,**kwargs):
        """
        Initialize WaterLevelForcing object.
        Args:
            input_filename (str): Path to the input file.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args,**kwargs)
        self.domain_number=domain_number
        self.wl_info=wl_info
        self.input_filename=input_filename
  
    def waterlevels_from_user(self):
        wl_file_path = glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.wl')[0]
        wl_filename=wl_file_path.split('/')[-1]

        os.system(f'cp {self.dict_folders["input"]}domain_0{self.domain_number}/{wl_filename}\
                                 {self.dict_folders["run"]}domain_0{self.domain_number}/')

        if self.wl_info!=None:
            self.wl_info.update({"water_levels.wl":wl_filename})
            return self.wl_info
    
    def waterlevel_from_uhslc(self,station_code=737):
        """
        Downloads water level data from the UH-SLC service and saves it to a file.
        """
                
        if not utils.verify_file(f'{self.dict_folders["input"]}domain_0{self.domain_number}/h{station_code}.csv'):
            getting_wl_data(station_code,f'{self.dict_folders["input"]}domain_0{self.domain_number}')
        return None
    
    def write_wl_ascii(self,wl_filepath,ascii_filepath):
        """
        Writes water level data to an ASCII file.
        Args:
            wl_filepath (str): Path to the water level data file.
            ascii_filepath (str): Path to the output ASCII file.
        Returns:
            dict: Dictionary containing water level parameters.
        """
        self.dataset = pd.read_csv(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{wl_filepath}',header=None,
                                            names=["year","month","day","hour","water_level"],sep=',')
        print(self.dataset)
                                    
        self.dataset.index = pd.to_datetime(self.dataset[['year', 'month', 'day', 'hour']])
        self.dataset = self.dataset.drop(columns=['year', 'month', 'day', 'hour'])
        self.dataset.index = self.dataset.index- pd.DateOffset(hours=5)  # Adjusting for UTC-5
        self.dataset_filtered = self.dataset[(self.dataset.index >= self.ini_date) & (self.dataset.index <= self.end_date)]
        self.dataset_filtered['water_level']=(self.dataset_filtered['water_level']-np.mean(self.dataset_filtered['water_level']))/1000

        bathymetry_grid = np.genfromtxt(glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.bot')[0])

        file = open(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{ascii_filepath}','w')

        for date in self.dataset_filtered.index:
            file.write("%s\n" % date.strftime("%Y%m%d %H%M%S"))
            water_level=np.full(np.shape(bathymetry_grid),-9999).astype(float)
            water_level[bathymetry_grid>=0]=float(self.dataset_filtered['depth'][date])
            np.savetxt(file,water_level,fmt='%12.4f')
        file.close()

        if not utils.verify_link(ascii_filepath,f'{self.dict_folders["run"]}domain_0{self.domain_number}/'):
            utils.create_link(ascii_filepath,f'{self.dict_folders["input"]}domain_0{self.domain_number}/',
                                f'{self.dict_folders["run"]}domain_0{self.domain_number}/')

        return {'water_levels.wl':ascii_filepath}

    def fill_wl_section(self,dict_wl_data):
        print (f'\n*** Adding/Editing water level information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',dict_wl_data)
