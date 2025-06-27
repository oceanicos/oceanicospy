import numpy as np
import pandas as pd
import xarray as xr
import subprocess
import os

from .. import utils
from ..init_setup import InitialSetup
from ....retrievals import *

class BoundaryConditions(InitialSetup):
    """
    Class representing the boundary conditions for a simulation.
    Args:
        input_filename (str): The name of the input file.
        dict_bounds_params (dict): A dictionary containing the boundary parameters.
        list_sides (list): A list of sides.
    Attributes:
        input_filename (str): The name of the input file.
        dict_bounds_params (dict): A dictionary containing the boundary parameters.
        list_sides (list): A list of sides.
        tpar_method_invoked (bool): Indicates whether the tpar method has been invoked.
    Methods:
        tpar_from_rawdata(output_filename): Generate TPAR data from raw data.
        tpar_from_ERA5_wave_data(output_filename): Generate TPAR data from ERA5 wave data.
        fill_boundaries_section(*args): Fill the boundaries section of the simulation.
    """
    def __init__ (self,domain_number,input_filename=None,dict_bounds_params=None,list_sides=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input_filename=input_filename
        self.dict_bounds_params=dict_bounds_params
        self.list_sides=list_sides
        self.tpar_method_invoked=False
        self.bounds_var=False
        self.domain_number=domain_number

    def tpar_from_rawdata(self,output_filename):
        """
        Generate TPAR data from raw data.
        Args:
            output_filename (str): The name of the output file.
        Returns:
            None
        """
        self.tpar_method_invoked=True
        file=pd.read_csv(f'{self.dict_folders["input"]}{self.input_filename}',delimiter=' ')
        dates=pd.date_range(self.ini_date,self.end_date,freq='1h')
        file=file.set_index(dates)
        file['Dir_spread'] = 2  
        file_to_save=file[['Hs','Tp','Dir','Dir_spread']]
        file_to_save.index=file_to_save.index.strftime('%Y%m%d.%H%M%S')

        output_file_path = f'{self.dict_folders["run"]}{output_filename}'
        
        # Open the file in write mode and write the first line
        with open(output_file_path, 'w') as f:
            f.write('TPAR\n')
    
        # Append the dataset to the file
        formatted_data = file_to_save.to_string(header=False,index=True,float_format='{:7.2f}'.format)

        with open(output_file_path, 'a') as f:
            f.write(formatted_data)

    def tpar_from_ERA5_wave_data(self,output_filename):
        """
        Generate TPAR data from ERA5 wave data.
        Args:
            output_filename (str): The name of the output file.
        Returns:
            None
        """
        self.tpar_method_invoked=True
        era5_wave_data = xr.open_dataset(f'{self.dict_folders["input"]}waveparams_1980_2023.nc')
        era5_wave_data = era5_wave_data.sel(valid_time=slice(self.ini_date,self.end_date))
        dict_vars={}
        for variable in era5_wave_data.data_vars:
            lat_interp = np.array([11.5, 12])
            var=era5_wave_data[variable][:,:,0]
            var_interp = np.array([np.interp([11.7], lat_interp, var[x,:])[0] for x in range(len(var))])
            var_interp = np.round(var_interp, 3)
            dict_vars[variable]=var_interp
        file_to_save = pd.DataFrame(dict_vars,index=pd.date_range(self.ini_date,self.end_date,freq='1h'))
        file_to_save['Dir_spread'] = 2  
        file_to_save.index=file_to_save.index.strftime('%Y%m%d.%H%M%S')

        output_file_path = f'{self.dict_folders["run"]}{output_filename}'
        
        # Open the file in write mode and write the first line
        with open(output_file_path, 'w') as f:
            f.write('TPAR\n')
    
        # Append the dataset to the file
        formatted_data = file_to_save.to_string(header=False,index=True,float_format='{:7.2f}'.format)

        with open(output_file_path, 'a') as f:
            f.write(formatted_data)

    def single_tpar_from_era5(self,filename,lati,long,wave_file='waves_era5'):
        ds = xr.open_dataset(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{wave_file}.nc')
        time = ds.valid_time.values
        time_UTC_offset = pd.Timedelta(hours=5)  # Adjusting for UTC-5
        tiempo = pd.to_datetime(time) - time_UTC_offset
        tiempo = tiempo[(tiempo >= self.ini_date) & (tiempo <= self.end_date)]
        strtime = [tiempo[i].strftime("%Y%m%d.%H%M") for i in range(len(tiempo))]
        lat_idx = np.where(np.isclose(ds.latitude.values, lati, atol=1e-4))[0]
        lon_idx = np.where(np.isclose(ds.longitude.values, long, atol=1e-4))[0]
        swh = np.zeros(len(tiempo))
        pp = np.zeros(len(tiempo))
        mwd = np.zeros(len(tiempo))
        for i in range(len(tiempo)):
            datai = ds.isel(latitude=lat_idx,longitude=lon_idx,valid_time=np.where(ds.valid_time.values==np.datetime64(tiempo[i]))[0][0])
            swh[i] = datai.swh.values[0][0]
            pp[i] = datai.pp1d.values[0][0]
            mwd[i] = datai.mwd.values[0][0]
        df=pd.DataFrame({'Tiempo':strtime,'Altura':swh,'Periodo':pp,'Direccion':mwd,'dd':40})
        with open (filename+'.bnd', "w") as f:
                f.write("TPAR \n")
                np.savetxt(f,df,fmt =('%s  %7.9f  %8.9f  %9.9f  %5.1f'))
        return df
    
    def waves_from_era5(self,wind_info,data_path='waves_era5'):
        """
        Downloads ERA5 wave data and writes it to an ASCII file.
        Args:
            data_path (str): Path to save the ERA5 wave data.
        Returns:
            None
-        """
        
        if not utils.verify_file(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{data_path}.nc'):
            getting_wave_data(wind_info['lat_ll_wind'], wind_info['lon_ll_wind'],
                        wind_info['meshes_x_wind'], wind_info['meshes_y_wind'],
                        wind_info['dx_wind'], wind_info['dy_wind'],
                        f'{self.dict_folders["input"]}domain_0{self.domain_number}/{data_path}')
        return None

    def tpar_from_ERA5_wave_data_2(self,points_lat,points_lon):

        for i in range(len(points_lon)):
            self.single_tpar_from_era5(f'{self.dict_folders["input"]}domain_0{self.domain_number}/TparN2025_{i+1}',max(points_lat),points_lon[i])
            self.single_tpar_from_era5(f'{self.dict_folders["input"]}domain_0{self.domain_number}/TparS2025_{i+1}',min(points_lat),points_lon[i])

        for j in range(len(points_lat)):
            self.single_tpar_from_era5(f'{self.dict_folders["input"]}domain_0{self.domain_number}/TparE2025_{j+1}',points_lat[j],max(points_lon))
            self.single_tpar_from_era5(f'{self.dict_folders["input"]}domain_0{self.domain_number}/TparO2025_{j+1}',points_lat[j],min(points_lon))
        return None

    def tpar_from_user(self):
        self.tpar_method_invoked=True
        self.bounds_var=True

    def fill_boundaries_section(self,*args):
        """
        Fill the boundaries section of the simulation.
        Args:
            *args: Variable length argument list.
        Returns:
            None
        """

        if self.bounds_var==True:
            var="VAR"
        else:
            var="CON"

        # if self.domain_number==1:
        #     os.system(f'cp {self.dict_folders["input"]}domain_0{self.domain_number}/*.bnd \
        #                 {self.dict_folders["run"]}domain_0{self.domain_number}/')

        if self.domain_number==1:
            os.system(f'ln -sf {self.dict_folders["input"]}domain_0{self.domain_number}/*.bnd \
                        {self.dict_folders["run"]}domain_0{self.domain_number}/')

        # if self.domain_number==1:
        #     if self.tpar_method_invoked:
        #         if len(args)>0:
        #             output_filename=args[0]
        #             if len(self.list_sides)==1:
        #                     filename_parts = output_filename.split('.')
        #                     filename_parts.insert(1, self.list_sides[0])
        #                     string_to_replace = f"BOU SIDE {self.list_sides[0]} CLOCW {var} FILE '{'.'.join(filename_parts)}'"
        #                     subprocess.run(['cp',f'{self.dict_folders["run"]}{output_filename}',f'{self.dict_folders["run"]}{output_filename.replace(".",f".{self.list_sides[0]}.")}'])
        #             else:
        #                 string_to_replace=''
        #                 for idx,side in enumerate(self.list_sides):
        #                     filename_parts = output_filename.split('.')
        #                     filename_parts.insert(1, side)
        #                     if idx!=len(self.list_sides)-1:
        #                         string_to_replace += f"BOU SIDE {side} CLOCKW {var} FILE '{'.'.join(filename_parts)}'\n"
        #                     else:
        #                         string_to_replace += f"BOU SIDE {side} CLOCKW {var} FILE '{'.'.join(filename_parts)}'"
        #                     subprocess.run(['cp',f'{self.dict_folders["run"]}{output_filename}',f'{self.dict_folders["run"]}{output_filename.replace(".",f".{side}.")}'])
        #         else:
        #             string_to_replace=''
                
        #     else:
        #         for param in self.dict_bounds_params:
        #             if type(self.dict_bounds_params[param])==float:
        #                 self.dict_bounds_params[param]=str(self.dict_bounds_params[param])

        #         if type(self.dict_bounds_params['dir'])==str:
        #             if self.dict_bounds_params['dir'].isalpha():
        #                 self.segment_dir=self.dict_bounds_params['dir']

        #                 if self.dict_bounds_params['dir']=='ENE':
        #                     self.dict_bounds_params['dir']=60.
        #                 elif self.dict_bounds_params['dir']=='E':
        #                     self.dict_bounds_params['dir']=90.
        #                 elif self.dict_bounds_params['dir']=='SE':
        #                     self.dict_bounds_params['dir']=135.
        #             else:
        #                 self.dict_bounds_params[param]=str(self.dict_bounds_params[param])
        #         list_string=[str(value) for key,value in self.dict_bounds_params.items() if key!='segment_dir']
        #         string=' '.join(list_string)
        #         string_to_replace = f"BOU SIDE {self.list_sides[0]} CLOCW {var} PAR {string}"
        # else:
        #     string_to_replace='NEST'
        
        if isinstance(args,(list,tuple)):
            values_bounds=args[0]
        else:
            values_bounds=string_to_replace

        self.dict_boundaries={'values_bounds':values_bounds}

        print (f'\n*** Adding/Editing boundary information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.dict_boundaries)
