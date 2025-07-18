import shutil
import subprocess
import pandas as pd
from pathlib import Path
import os

from .. import utils
from ..init_setup import InitialSetup

class RunCase(InitialSetup):
    def __init__(self,domain_number,dict_comp_data,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dict_comp_data=dict_comp_data
        self.domain_number=domain_number

    def output_definition(self,filename=None):
        ds=pd.read_csv(f'{self.dict_folders["input"]}domain_0{self.domain_number}/{filename}',delimiter=',')

        ds=ds[['X','Y']]
        if (ds['X'] < 0).any():
            ds.loc[ds['X'] < 0, 'X'] += 360
        ds.to_csv(f'{self.dict_folders["run"]}domain_0{self.domain_number}/points.loc',index=False, header=False, na_rep=0, float_format='%7.7f',sep=' ')
    
    def write_nest_section(self,nested_doms=[],nested_doms_info=[]):
        if len(nested_doms)!=0:
            if self.domain_number==2:
                utils.duplicate_lines(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn', 56)
            for nested_dom_id,nested_dom_info in zip(nested_doms,nested_doms_info):
                nested_dom_info_=dict()
                for key in nested_dom_info.copy().keys():
                    nested_dom_info_[f'child_{key}']= nested_dom_info[key]
                nested_dom_info_.update(nest_id=f'n0{self.domain_number}_0{nested_dom_id}',nest_grid_file=f'child0{self.domain_number}_0{nested_dom_id}.NEST')
                print(nested_dom_info_)
                utils.fill_files_only_once(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',nested_dom_info_)
        else:
            utils.delete_line(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn','NGRID')
            utils.delete_line(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn','NESTOUT')
               
    def fill_computation_section(self):
        self.script_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.script_dir.parent.parent.parent / 'data'

        shutil.copy(f'{self.data_dir}/model_config_templates/swan/launcher_base_nest_cecc.slurm',
                    f'{self.dict_folders["run"]}launcher_swan.slurm')
        shutil.copy(f'{self.data_dir}/model_config_templates/swan/swaninit',
                    f'{self.dict_folders["run"]}domain_0{self.domain_number}/swaninit')
        bash_code = "declare -a bash_dict\n"
        for key, value in self.dict_ini_data["parent_domains"].items():
            bash_value = "" if value is None else value
            bash_code += f'bash_dict[{key}]={bash_value}\n'
        launch_dict=dict(path_case=self.root_path,simulation_name=self.dict_ini_data["name"].replace(" ","_"),
                        number_domains=self.dict_ini_data["number_domains"],parent_domains = bash_code)
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

        print (f'\n*** Adding/Editing compilation information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.dict_comp_data)

        # subprocess.run([f'rm -rf {self.dict_folders["run"]}run.erf-*'],shell=True)
