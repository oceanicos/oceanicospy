import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import glob as glob
import numpy as np

from .. import utils
from ..init_setup import InitialSetup

class WindForcing(InitialSetup):
    def __init__ (self,input_filename=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
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
        ds_era5 = xr.load_dataset(f'{self.dict_folders["input"]}{era5_filepath}',engine='netcdf4')
        # print(ds_era5)
        lat_era5 = ds_era5['latitude'].values
        lon_era5 = ds_era5['longitude'].values
        v10 = ds_era5.variables['v10'].values
        u10 = ds_era5.variables['u10'].values
        time= pd.DatetimeIndex(ds_era5.valid_time)

        time_s = pd.date_range(self.ini_date,self.end_date, freq='1h')

        time_to_write = (time_s - time_s[0]).total_seconds().astype(int).tolist()

        v10 = v10[(time >= self.ini_date) & (time <= self.end_date)]
        u10 = u10[(time >= self.ini_date) & (time <= self.end_date)]

        wind_speed=np.sqrt(v10**2+u10**2)
        wind_dir_cart=np.degrees(np.arctan2(u10,v10))
        wind_dir_naut=(90-wind_dir_cart)%360

        df_to_save=pd.DataFrame({'Time':time_to_write,'Vel':wind_speed[:,-1,0],'Dir':wind_dir_naut[:,-1,0]},index=time_s)
        df_to_save.to_csv(f'{self.dict_folders["run"]}{ascii_filepath}',sep=' ',header=False,index=False)

        self.wind_params=dict(windfilepath=f'{ascii_filepath}')
        return self.wind_params

    def write_constant_from_ERA5(self,era5_filepath):
        ds_era5 = xr.load_dataset(f'{self.dict_folders["input"]}{era5_filepath}',engine='netcdf4')

        v10 = ds_era5.variables['v10'].values
        u10 = ds_era5.variables['u10'].values
        time= pd.DatetimeIndex(ds_era5.valid_time)

        v10 = v10[(time >= self.ini_date) & (time < self.end_date)]
        u10 = u10[(time >= self.ini_date) & (time < self.end_date)]

        wind_speed=np.sqrt(v10**2+u10**2)
        wind_dir_cart=np.degrees(np.arctan2(v10,u10))
        wind_dir_naut=(90-wind_dir_cart)%360

        self.wind_params=dict(wind_speed=wind_speed[0,-1,0],wind_direction=wind_dir_naut[0,-1,0])
        return self.wind_params


    # def write_constant_wind(self, ascii_filepath):
    #     ds=pd.read_csv(f'{self.dict_folders["input"]}{self.input_filename}',delimiter=' ')
    #     dates=pd.date_range(self.ini_date,self.end_date,freq='1h')
    #     ds=ds.set_index(dates)
    #     ds_to_save=ds[['Dir','Vel']]
    #     ds_to_save.index=ds_to_save.index.strftime('%Y%m%d.%H%M%S')
        
    #     file = open(f'{self.dict_folders["run"]}{ascii_filepath}','w')
    #     for idx,t in enumerate(ds_to_save.index):
    #         file.write(t)
    #         file.write('\n')
    #         ds_to_save['Dir_to'] = (ds_to_save['Dir'] + 180) % 360
    #         u10_to_write = ds_to_save['Vel'].iloc[idx] * np.sin(np.deg2rad(ds_to_save['Dir_to'].iloc[idx]))
    #         v10_to_write = ds_to_save['Vel'].iloc[idx] * np.cos(np.deg2rad(ds_to_save['Dir_to'].iloc[idx]))

    #         u10_to_write = round(u10_to_write,2)
    #         v10_to_write = round(v10_to_write,2)

    #         u10_to_write = u10_to_write*np.ones((25,25))
    #         v10_to_write = v10_to_write*np.ones((25,25))
                        
    #         file.write(pd.DataFrame(u10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))
    #         file.write(pd.DataFrame(v10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))

    #     file.close()
    
    #     ll_lon_on,ll_lat_on=4931900,2799400 #quemado

    #     self.wind_params=dict(lon_ll_wind=ll_lon_on,lat_ll_wind=ll_lat_on,
    #                           meshes_x_wind=24,meshes_y_wind=24,
    #                           dx_wind=2000,dy_wind=2000,ini_wind_date=ds_to_save.index[0],
    #                           dt_wind_hours=1,end_wind_date=ds_to_save.index[-1])
    #     return self.wind_params

    def txt_from_user(self):
        """
        Reads wind parameters from a user-defined file and returns them as a dictionary.
        """
        wind_file_path = glob.glob(f'{self.dict_folders["input"]}*.wnd')[0]
        wind_filename = wind_file_path.split('/')[-1]

        if not utils.verify_link(wind_filename,f'{self.dict_folders["run"]}/'):
            utils.create_link(wind_filename,f'{self.dict_folders["input"]}/',
                                f'{self.dict_folders["run"]}/')

        self.wind_info = {"windfilepath":wind_filename}
        return self.wind_info

    def fill_wind_section(self,dict_wind_data):
        print (f'\n*** Adding/Editing winds information for domain in configuration file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}params.txt',dict_wind_data)
