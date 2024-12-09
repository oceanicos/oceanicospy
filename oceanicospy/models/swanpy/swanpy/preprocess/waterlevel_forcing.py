import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import numpy as np
import glob as glob
import os

from .. import utils
from ..init_setup import InitialSetup

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
        if self.dict_ini_data["nested_domains"]>0:
            wl_file_path = glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.wl')[0]
        else:
            wl_file_path = glob.glob(f'{self.dict_folders["input"]}*.wl')[0]
        wl_filename=wl_file_path.split('/')[-1]

        if not utils.verify_link(wl_filename,f'{self.dict_folders["run"]}domain_0{self.domain_number}/'):
            utils.create_link(wl_filename,f'{self.dict_folders["input"]}domain_0{self.domain_number}/',
                                f'{self.dict_folders["run"]}domain_0{self.domain_number}/')

        # os.system(f'cp {self.dict_folders["input"]}domain_0{self.domain_number}/{wl_filename}\
        #                         {self.dict_folders["run"]}domain_0{self.domain_number}/')

        if self.wl_info!=None:
            self.wl_info.update({"water_levels.wl":wl_filename})
            return self.wl_info

    def fill_wl_section(self,dict_wl_data):

        if self.dict_ini_data["nested_domains"]>0:
            print (f'\n*** Adding/Editing water level information for domain {self.domain_number} in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',dict_wl_data)
        else:
            print ('\n*** Adding/Editing water level information in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}run.swn',dict_wl_data)
