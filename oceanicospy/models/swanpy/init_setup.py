import subprocess
from pathlib import Path
from . import utils
import shutil
import os

class InitialSetup():
    def __init__ (self,root_path,dict_ini_data,ini_date=None,end_date=None):
        self.root_path=root_path
        self.ini_date=ini_date
        self.end_date=end_date
        self.dict_ini_data=dict_ini_data
        self.folder_names=['input','pros','run','output']
        self.dict_folders={}
        for folder_name in self.folder_names:
            self.dict_folders[folder_name]=f'{self.root_path}{folder_name}/'

    def create_folders_l1(self):
        print ('\n*** Creating project structure ***\n')
        for folder_name in self.folder_names:
            if not os.path.exists(self.dict_folders[folder_name]):
                subprocess.call(['mkdir','-p',f'{self.dict_folders[folder_name]}'])
            else:
                if folder_name in ['output','run']:
                    os.system(f'rm -rf {self.dict_folders[folder_name]}*')


    def create_folders_l2(self):
        if self.dict_ini_data["nested_domains"]>0:
            for domain in range(1,self.dict_ini_data["nested_domains"]+1):
                if not os.path.exists(f'{self.dict_folders["output"]}domain_0{domain}/'):
                    subprocess.call(['mkdir','-p',f'{self.dict_folders["output"]}domain_0{domain}/'])
                else:
                    os.system(f'rm -rf {self.dict_folders["output"]}domain_0{domain}/*')
                    
                if not os.path.exists(f'{self.dict_folders["run"]}domain_0{domain}/'):
                    subprocess.call(['mkdir','-p',f'{self.dict_folders["run"]}domain_0{domain}/'])
                else:
                    os.system(f'rm -rf {self.dict_folders["run"]}domain_0{domain}/*')


    def replace_ini_data(self):

        if self.dict_ini_data['stat_id']==0:
            self.stat_label='NONSTAT'
        else:
            self.stat_label='STAT'
        self.dict_ini_data['stat_label']=self.stat_label

        print ('\n*** Copying base swan configuration file into run folder ***\n')

        self.script_dir = Path(__file__).resolve().parent
        self.data_dir = self.script_dir.parent.parent.parent / 'data'

        if self.dict_ini_data["nested_domains"]==0:

            shutil.copy(f'{self.data_dir}/model_config_templates/swan/run_base_{self.stat_label.lower()}_SAI.swn', f'{self.dict_folders["run"]}run.swn')
            utils.fill_files(f'{self.dict_folders["run"]}run.swn',self.dict_ini_data)

        else:
            for domain in range(1,self.dict_ini_data["nested_domains"]+1):
                shutil.copy(f'{self.data_dir}/model_config_templates/swan/run_base_{self.stat_label.lower()}_SAI.swn', f'{self.dict_folders["run"]}domain_0{domain}/run.swn')
                utils.fill_files(f'{self.dict_folders["run"]}domain_0{domain}/run.swn',self.dict_ini_data)
