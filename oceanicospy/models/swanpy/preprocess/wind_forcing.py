import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import numpy as np
import glob as glob
import os

from .. import utils
from ..init_setup import InitialSetup

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
        ds_era5 = xr.load_dataset(f'{self.dict_folders["input"]}{era5_filepath}',engine='netcdf4')
        # print(ds_era5)
        lat_era5 = ds_era5['latitude'].values
        lon_era5 = ds_era5['longitude'].values
        v10 = ds_era5.variables['v10'].values
        u10 = ds_era5.variables['u10'].values
        time= pd.DatetimeIndex(ds_era5.valid_time)

        time_s = pd.date_range(self.ini_date,self.end_date, freq='1h')
        time_to_write = time_s.format(formatter=lambda x: x.strftime('%Y%m%d.%H%M'))

        v10 = v10[(time >= self.ini_date) & (time <= self.end_date)]
        u10 = u10[(time >= self.ini_date) & (time <= self.end_date)]

        file = open(f'{self.dict_folders["run"]}{ascii_filepath}','w')
        for idx,t in enumerate(time_to_write):
            file.write(t)
            file.write('\n')
            
            u10_to_write=u10[idx]
            v10_to_write=v10[idx]
            file.write(pd.DataFrame(u10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))
            file.write(pd.DataFrame(v10_to_write).to_csv(index=False, header=False, na_rep=0, float_format='%7.3f').replace(',', ' '))
        file.close()

        wgs84 = Proj(init='epsg:4326')
        origen_nacional = Proj(init='epsg:9377')

        # print(lon_era5, lat_era5)
    
        ll_lon_on,ll_lat_on=transform(wgs84, origen_nacional, lon_era5[0], lat_era5[-1])
        ll_lon_on,ll_lat_on=round(ll_lon_on,2),round(ll_lat_on,2)

        self.wind_params=dict(path_wind=f'{self.ini_date.year}_{self.end_date.year}.wnd',lon_ll_wind=ll_lon_on,lat_ll_wind=ll_lat_on,
                              meshes_x_wind=2,meshes_y_wind=2,
                              dx_wind=27500,dy_wind=27500,ini_wind_date=time_to_write[0],
                              dt_wind_hours=1,end_wind_date=time_to_write[-1])
        return self.wind_params

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
        if self.dict_ini_data["nested_domains"]>0:
            wind_file_path = glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.wnd')[0]
        else:
            wind_file_path = glob.glob(f'{self.dict_folders["input"]}*.wnd')[0]
        wind_filename=wind_file_path.split('/')[-1]

        if not utils.verify_link(wind_filename,f'{self.dict_folders["run"]}domain_0{self.domain_number}/'):
            utils.create_link(wind_filename,f'{self.dict_folders["input"]}domain_0{self.domain_number}/',
                                f'{self.dict_folders["run"]}domain_0{self.domain_number}/')

        # os.system(f'rsync {self.dict_folders["input"]}domain_0{self.domain_number}/{wind_filename}\
        #                         {self.dict_folders["run"]}domain_0{self.domain_number}/')

        if self.wind_info!=None:
            self.wind_info.update({"winds.wnd":wind_filename})
            return self.wind_info

    def fill_wind_section(self,dict_wind_data):

        if self.dict_ini_data["nested_domains"]>0:
            print (f'\n*** Adding/Editing winds information for domain {self.domain_number} in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',dict_wind_data)
        else:
            print ('\n*** Adding/Editing winds information in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}run.swn',dict_wind_data)
