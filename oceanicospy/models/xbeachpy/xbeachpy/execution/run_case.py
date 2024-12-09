import shutil
import subprocess
import pandas as pd
import datetime as dt
import numpy as np
import glob as glob
import scipy.interpolate

from .. import utils
from ..init_setup import InitialSetup

class RunCase(InitialSetup):
    def __init__(self,dict_comp_data,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dict_comp_data=dict_comp_data

    def output_definition(self,filename=None):
        self.dict_comp_data['outputfilepath']=f'../output/{filename}'
    
    def output_points(self,list_y_points):
        dat_files=glob.glob(f'{self.dict_folders["input"]}*.dat')
        bathy_file = [file for file in dat_files if 'Perfil_0' in file][0]
        data=np.loadtxt(bathy_file)
        x=data[:,0]
        h=data[:,3][::-1]
        f=scipy.interpolate.interp1d(h,x)
        x_points=f(list_y_points)

        self.dict_comp_data['lenpoints']=len(list_y_points)+1
        string_points=[f'{round(x,2)} 0\n' for x in x_points]
        self.dict_comp_data['string_points']=''.join(string_points)

    def fill_computation_section(self):
        shutil.copy('/home/fayalacruz/runs/modelling/inp_templates/launcher_xbeach_base.slurm',
                    f'{self.dict_folders["run"]}launcher_xbeach.slurm')

        launch_dict=dict(path_case=f'{self.dict_folders["run"]}',case_number='matthew')
        utils.fill_files(f'{self.dict_folders["run"]}launcher_xbeach.slurm',launch_dict)

        ini_comp_date = dt.datetime.strptime(self.dict_comp_data['ini_comp_date'], '%Y%m%d.%H%M%S')
        end_comp_date = dt.datetime.strptime(self.dict_comp_data['end_comp_date'], '%Y%m%d.%H%M%S')

        seconds = (end_comp_date - ini_comp_date).total_seconds()
        self.dict_comp_data['tstop_value'] = int(seconds)
        for param in self.dict_comp_data:
            self.dict_comp_data[param]=str(self.dict_comp_data[param])

        utils.fill_files(f'{self.dict_folders["run"]}params.txt',self.dict_comp_data)