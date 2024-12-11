import pandas as pd
import numpy as np

from .. import utils
from ..init_setup import InitialSetup

class SeaLevelForcing(InitialSetup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_data_from_CECOLDO(self, input_filename, output_filename):
        """
        Convert data from CECOLDO to XBeach format.
        Args:
            path_cecoldo (str): The path to the CECOLDO data.
            path_output (str): The path to the output data.
        Returns:
            None
        """
        data_cecoldo=pd.read_csv(f'{self.dict_folders["input"]}{input_filename}',sep='\t',skiprows=[1],
                                 na_values=['-99999.00'])
        data_cecoldo['Fecha'] = pd.to_datetime(data_cecoldo['Fecha [aaaa-mm-dd UT-5]'] + ' ' + data_cecoldo['Hora [hh:mm:ss UT-5]'])
        data_cecoldo.set_index('Fecha',inplace=True)
        data_cecoldo.drop(columns=['Fecha [aaaa-mm-dd UT-5]','Hora [hh:mm:ss UT-5]',
                                'Latitud [deg]','Longitud [deg]','Estacion [#]','QF [IODE]'],inplace=True)
        
        data_cecoldo=data_cecoldo[(data_cecoldo.index >= self.ini_date) & (data_cecoldo.index <= self.end_date)]

        # data_cecoldo = data_cecoldo.resample('1H').ffill()
        data_cecoldo['Nivel_mar [m]'] = data_cecoldo['Nivel_mar [m]']-np.nanmean(data_cecoldo['Nivel_mar [m]'])

        time_to_write = (data_cecoldo.index - data_cecoldo.index[0]).total_seconds().astype(int).tolist()
        df_to_save=pd.DataFrame({'time [s]':time_to_write,'sealevel':data_cecoldo['Nivel_mar [m]']},index=None)
        df_to_save=df_to_save.round(3)
        df_to_save.to_csv(f'{self.dict_folders["run"]}{output_filename}',sep='\t',header=None,index=None)
        self.dict_sealevel={'sealevelfilepath':'sealevel.txt'}
        return self.dict_sealevel

    def constant_from_CECOLDO(self,input_filename):
        data_cecoldo=pd.read_csv(f'{self.dict_folders["input"]}{input_filename}',sep='\t',skiprows=[1],
                                 na_values=['-99999.00'])
        data_cecoldo['Fecha'] = pd.to_datetime(data_cecoldo['Fecha [aaaa-mm-dd UT-5]'] + ' ' + data_cecoldo['Hora [hh:mm:ss UT-5]'])
        data_cecoldo.set_index('Fecha',inplace=True)
        data_cecoldo.drop(columns=['Fecha [aaaa-mm-dd UT-5]','Hora [hh:mm:ss UT-5]',
                                'Latitud [deg]','Longitud [deg]','Estacion [#]','QF [IODE]'],inplace=True)
        
        # data_cecoldo = data_cecoldo.resample('1H').ffill()
        data_cecoldo['Nivel_mar [m]'] = data_cecoldo['Nivel_mar [m]']-np.nanmean(data_cecoldo['Nivel_mar [m]'])
        data_cecoldo=data_cecoldo[(data_cecoldo.index >= self.ini_date) & (data_cecoldo.index <= self.end_date)]

        self.dict_sealevel={'sealevelvalue':round(data_cecoldo['Nivel_mar [m]'].values[0],3)}
        return self.dict_sealevel


    def fill_sealevel_section(self):
        """
        Fill the sealevel section of the simulation.
        Returns:
            None
        """
        for param in self.dict_sealevel:
            self.dict_sealevel[param]=str(self.dict_sealevel[param])

        utils.fill_files(f'{self.dict_folders["run"]}params.txt',self.dict_sealevel)        