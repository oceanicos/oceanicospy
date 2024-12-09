import numpy as np
import glob as glob
from .. import utils
from ..init_setup import InitialSetup

class MakeGrid(InitialSetup):
    """
    A class for creating a SWAN computational grid from bathymetry data and filling grid information in a SWAN file.
    Args:
        root_path (str): The root path of the project.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
    Attributes:
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
    Methods:

    """
    def __init__ (self,dx,dy,grid_info,domain_number,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dx=dx
        self.dy=dy     
        self.grid_info=grid_info
        self.domain_number=domain_number

    def params_from_bathy(self):
        if self.dict_ini_data["nested_domains"]>0:
            bathy_file_path = glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.dat')[0]
        else:
            bathy_file_path = glob.glob(f'{self.dict_folders["input"]}*.dat')[0]

        data = np.loadtxt(bathy_file_path)
        longitude = data[:, 0]
        latitude = data[:, 1]
        elevation = data[:, 2]

        min_longitude = np.min(longitude)
        min_latitude = np.min(latitude)

        max_longitude = np.max(longitude)
        max_latitude = np.max(latitude)
        min_longitude = int(np.ceil(min_longitude / 100) * 100)
        max_longitude = int(np.floor(max_longitude / 100) * 100)
        min_latitude = int(np.ceil(min_latitude / 100) * 100)
        max_latitude = int(np.floor(max_latitude / 100) * 100)

        x_extent=max_longitude-min_longitude
        y_extent=max_latitude-min_latitude

        nx = int(x_extent/self.dx)
        ny = int(y_extent/self.dy)
        
        grid_dict={'lon_ll_corner':min_longitude,'lat_ll_corner':min_latitude,'x_extent':x_extent,'y_extent':y_extent,'nx':nx,'ny':ny}
        for key,value in grid_dict.items():
            grid_dict[key]=str(value)

        return grid_dict
    
    def params_from_user(self):
        if self.grid_info!=None:
            return self.grid_info

    def fill_grid_section(self,dict_grid_data):
        if self.dict_ini_data["nested_domains"]>0:
            print (f'\n*** Adding/Editing grid information for domain {self.domain_number} in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',dict_grid_data)
        else:
            print ('\n*** Adding/Editing grid information in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}run.swn',dict_grid_data)
