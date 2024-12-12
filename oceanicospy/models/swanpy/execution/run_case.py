import shutil
import subprocess
import pandas as pd
from pathlib import Path

from .. import utils
from ..init_setup import InitialSetup

class RunCase(InitialSetup):
    def __init__(self,domain_number,dict_comp_data,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dict_comp_data=dict_comp_data
        self.domain_number=domain_number

    def output_definition(self,filename=None):
        if self.dict_ini_data["nested_domains"]>0:
            ds=pd.read_csv(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{filename}',delimiter=',')
        else:
            ds=pd.read_csv(f'{self.dict_folders["input"]}{filename}',delimiter=',')

        ds=ds[['X','Y']]
        # ds['id']=ds['id'].apply(lambda x:f'! {x}')
        if self.dict_ini_data["nested_domains"]>0:
            ds.to_csv(f'{self.dict_folders["run"]}domain_0{self.domain_number}/points.loc',index=False, header=False, na_rep=0, float_format='%7.7f',sep=' ')
        else:
            ds.to_csv(f'{self.dict_folders["run"]}points.loc',index=False, header=False, na_rep=0, float_format='%7.7f',sep=' ')
    
    def write_nest_section(self,child_grid_info=None):
        if child_grid_info==None:
            utils.delete_line(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn','NGRID')
            utils.delete_line(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn','NESTOUT')
        else:
            self.child_grid_info=child_grid_info
            for key in self.child_grid_info.copy().keys():
                self.child_grid_info[f'child_{key}']=self.child_grid_info[key]
            self.child_grid_info.update(nest_id=f'nest0{self.domain_number}',nest_grid_file=f'child_0{self.domain_number}.NEST')
            utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.child_grid_info)

    def fill_computation_section(self):
        self.script_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.script_dir.parent.parent.parent / 'data'

        if self.dict_ini_data["nested_domains"]>0:
            shutil.copy(f'{self.data_dir}/model_config_templates/swan/launcher_base_nest_cecc.slurm',
                        f'{self.dict_folders["run"]}launcher_swan.slurm')
            shutil.copy(f'{self.data_dir}/model_config_templates/swan/swaninit',
                        f'{self.dict_folders["run"]}domain_0{self.domain_number}/swaninit')
            launch_dict=dict(path_case=self.root_path,simulation_name=self.dict_ini_data["name"].replace(" ","_"),number_domains=self.dict_ini_data["nested_domains"])

        else:
            shutil.copy(f'{self.data_dir}/model_config_templates/swan/launcher_base_cecc.slurm',
                        f'{self.dict_folders["run"]}launcher_swan.slurm')
            shutil.copy(f'{self.data_dir}/model_config_templates/swan/swaninit',
                        f'{self.dict_folders["run"]}swaninit')
            launch_dict=dict(path_case=self.root_path,simulation_name=self.dict_ini_data["name"].replace(" ","_"),number_domains=self.dict_ini_data["nested_domains"])
        utils.fill_files(f'{self.dict_folders["run"]}launcher_swan.slurm',launch_dict,strict=False)

        if self.dict_comp_data['stat_comp'] in (0,"0"):
            self.stat_label='NONSTAT'
            self.string_comp=f'COMP {self.stat_label} {self.dict_comp_data["ini_comp_date"]} {self.dict_comp_data["dt_minutes"]} MIN {self.dict_comp_data["end_comp_date"]}'
        else:
            self.stat_label='STAT'
            self.string_comp=''
            for idx,date in enumerate(self.dict_comp_data['comp_dates']):
                self.date=date.strftime('%Y%m%d.%H%M%S')
                if idx==len(self.dict_comp_data['comp_dates'])-1:
                    self.string_comp+=f'COMP {self.stat_label} {self.date}'
                else:
                    if self.dict_comp_data['init_intermediate']:
                        self.string_comp+=f'COMP {self.stat_label} {self.date}\nINIT\n'
                    else:
                        self.string_comp+=f'COMP {self.stat_label} {self.date}\n'
        self.dict_comp_data['string_comp']=self.string_comp
        self.dict_comp_data['stat_label_comp']=self.stat_label

        for param in self.dict_comp_data:
            self.dict_comp_data[param]=str(self.dict_comp_data[param])

        if self.dict_ini_data["nested_domains"]>0:
            print (f'\n*** Adding/Editing compilation information for domain {self.domain_number} in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.dict_comp_data)
        else:
            print ('\n*** Adding/Editing compilation information in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}run.swn',self.dict_comp_data)

        # subprocess.run([f'rm -rf {self.dict_folders["run"]}run.erf-*'],shell=True)
