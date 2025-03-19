import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import numpy as np
import glob as glob
import os

from .. import utils
from ..init_setup import InitialSetup

class Friction(InitialSetup):
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
    def __init__ (self,domain_number,friction_info=None,input_filename=None,*args,**kwargs):
        """
        Initialize WaterLevelForcing object.
        Args:
            input_filename (str): Path to the input file.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args,**kwargs)
        self.domain_number=domain_number
        self.friction_info=friction_info
        self.input_filename=input_filename
  
    def friction_from_user(self):
        friction_file_path = glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.fric')[0]
        friction_filename=friction_file_path.split('/')[-1]

        if not utils.verify_link(friction_filename,f'{self.dict_folders["run"]}domain_0{self.domain_number}/'):
            utils.create_link(friction_filename,f'{self.dict_folders["input"]}domain_0{self.domain_number}/',
                                f'{self.dict_folders["run"]}domain_0{self.domain_number}/')

        # os.system(f'cp {self.dict_folders["input"]}domain_0{self.domain_number}/{friction_filename}\
        #                         {self.dict_folders["run"]}domain_0{self.domain_number}/')

        if self.friction_info!=None:
            self.friction_info.update({"friction.fric":friction_filename})
            return self.friction_info

    def fill_friction_section(self,dict_fric_data):
        print (f'\n*** Adding/Editing friction information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',dict_fric_data)