import numpy as np
import pandas as pd
import xarray as xr
import subprocess
import os

from .. import utils
from ..init_setup import InitialSetup

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
