import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import numpy as np
import glob as glob
import os

from .. import utils
from ..init_setup import InitialSetup
from ....retrievals import *

class WindForcing(InitialSetup):
    """
    Class representing wind forcing in SWAN model.
    Args:
        input_filename (str): Path to the input file.
    Attributes:
        input_filename (str): Path to the input file.
        wind_params (dict): Dictionary containing wind parameters.
    Methods:
        write_ERA5_ascii: Writes ERA5 wind data to an ASCII file.
        write_constant_wind: Writes constant wind data to an ASCII file.
        fill_wind_section: Fills the wind section in the SWAN input file.
    """
    def __init__ (self,domain_number,wind_info=None,input_filename=None,*args,**kwargs):
        """
        Initialize WindForcing object.
        Args:
            input_filename (str): Path to the input file.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args,**kwargs)
        self.domain_number=domain_number
        self.wind_info=wind_info
        self.input_filename=input_filename
  
    def write_ERA5_ascii(self,era5_filepath,ascii_filepath):
        """
        Writes ERA5 wind data to an ASCII file.
        Args:
            era5_filepath (str): Path to the ERA5 wind data file.
            ascii_filepath (str): Path to the output ASCII file.
        Returns:
            dict: Dictionary containing wind parameters.
        """
        ds_era5 = xr.load_dataset(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{era5_filepath}',engine='netcdf4')
        
        # print(ds_era5)
        lat_era5 = ds_era5['latitude'].values
        lon_era5 = ds_era5['longitude'].values

        v10 = ds_era5.variables['v10'].values
        u10 = ds_era5.variables['u10'].values
        time= pd.DatetimeIndex(ds_era5.valid_time)

        time_s = pd.date_range(self.ini_date,self.end_date, freq='1h')
        time_to_write = time_s.format(formatter=lambda x: x.strftime('%Y%m%d.%H%M'))

        v10 = v10[(time >= self.ini_date) & (time <= self.end_date),:,:]
        u10 = u10[(time >= self.ini_date) & (time <= self.end_date),:,:]

        file = open(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{ascii_filepath}','w')
        for idx,t in enumerate(time_to_write):
            file.write(t)
            file.write('\n')
            
            u10_to_write=u10[idx]
            v10_to_write=v10[idx]
            file.write(pd.DataFrame(u10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))
            file.write(pd.DataFrame(v10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))
        file.close()

        if not utils.verify_link(ascii_filepath,f'{self.dict_folders["run"]}domain_0{self.domain_number}/'):
            utils.create_link(ascii_filepath,f'{self.dict_folders["input"]}domain_0{self.domain_number}/',
                                f'{self.dict_folders["run"]}domain_0{self.domain_number}/')

        if self.wind_info!=None:
            self.wind_info.update({"winds.wnd":"winds.wnd"})
            return self.wind_info
        return None

    def write_constant_wind(self, ascii_filepath):
        """
        Writes constant wind data to an ASCII file.
        Args:
            ascii_filepath (str): Path to the output ASCII file.
        Returns:
            dict: Dictionary containing wind parameters.
        """
        ds=pd.read_csv(f'{self.dict_folders["input"]}{self.input_filename}',delimiter=' ')
        dates=pd.date_range(self.ini_date,self.end_date,freq='1h')
        ds=ds.set_index(dates)
        ds_to_save=ds[['Dir','Vel']]
        ds_to_save.index=ds_to_save.index.strftime('%Y%m%d.%H%M%S')
        
        file = open(f'{self.dict_folders["run"]}{ascii_filepath}','w')
        for idx,t in enumerate(ds_to_save.index):
            file.write(t)
            file.write('\n')
            ds_to_save['Dir_to'] = (ds_to_save['Dir'] + 180) % 360
            u10_to_write = ds_to_save['Vel'].iloc[idx] * np.sin(np.deg2rad(ds_to_save['Dir_to'].iloc[idx]))
            v10_to_write = ds_to_save['Vel'].iloc[idx] * np.cos(np.deg2rad(ds_to_save['Dir_to'].iloc[idx]))

            u10_to_write = round(u10_to_write,2)
            v10_to_write = round(v10_to_write,2)

            u10_to_write = u10_to_write*np.ones((25,25))
            v10_to_write = v10_to_write*np.ones((25,25))
                        
            file.write(pd.DataFrame(u10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))
            file.write(pd.DataFrame(v10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))

        file.close()
    
        ll_lon_on,ll_lat_on=4931900,2799400 #quemado

        self.wind_params=dict(lon_ll_wind=ll_lon_on,lat_ll_wind=ll_lat_on,
                              meshes_x_wind=24,meshes_y_wind=24,
                              dx_wind=2000,dy_wind=2000,ini_wind_date=ds_to_save.index[0],
                              dt_wind_hours=1,end_wind_date=ds_to_save.index[-1])
        return self.wind_params

    def winds_from_user(self):
        wind_file_path = glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.wnd')[0]
        wind_filename=wind_file_path.split('/')[-1]

        if not utils.verify_link(wind_filename,f'{self.dict_folders["run"]}domain_0{self.domain_number}/'):
            utils.create_link(wind_filename,f'{self.dict_folders["input"]}domain_0{self.domain_number}/',
                                f'{self.dict_folders["run"]}domain_0{self.domain_number}/')

        # os.system(f'rsync {self.dict_folders["input"]}domain_0{self.domain_number}/{wind_filename}\
        #                         {self.dict_folders["run"]}domain_0{self.domain_number}/')

        if self.wind_info!=None:
            self.wind_info.update({"winds.wnd":wind_filename})
            return self.wind_info

    def winds_from_era5(self,data_path='winds_era5'):
        """
        Downloads ERA5 wind data and writes it to an ASCII file.
        Args:
            data_path (str): Path to save the ERA5 wind data.
        Returns:
            dict: Dictionary containing wind parameters.
        """
        
        if not utils.verify_file(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{data_path}.nc'):
            getting_data(self.wind_info['lat_ll_wind'], self.wind_info['lon_ll_wind'],
                        self.wind_info['meshes_x_wind'], self.wind_info['meshes_y_wind'],
                        self.wind_info['dx_wind'], self.wind_info['dy_wind'],
                        f'{self.dict_folders["input"]}domain_0{self.domain_number}/{data_path}')
        return None

    def fill_wind_section(self,dict_wind_data):
        print (f'\n*** Adding/Editing winds information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',dict_wind_data)
