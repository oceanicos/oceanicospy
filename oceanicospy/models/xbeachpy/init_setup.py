import subprocess
from . import utils
from pathlib import Path
import shutil
import os

class InitialSetup():
    def __init__ (self,root_path,dic_ini_data,ini_date=None,end_date=None):
        self.root_path=root_path
        self.ini_date=ini_date
        self.end_date=end_date
        self.dic_ini_data=dic_ini_data
        self.folder_names=['input','pros','run','output']
        self.dict_folders={}
        for folder_name in self.folder_names:
            self.dict_folders[folder_name]=f'{self.root_path}{folder_name}/'

    def create_folders_l1(self):
        print ('\n*** Creating project structure ***\n')
        for folder_name in self.folder_names:
            if not os.path.exists(self.dict_folders[folder_name]):
                subprocess.call(['mkdir','-p',f'{self.dict_folders[folder_name]}'])

    def replace_ini_data(self):
        if self.dic_ini_data['activate_flow']==True:
            self.stat_label='flow'
            if self.dic_ini_data['activate_morf']==True:
                self.stat_label=self.stat_label+'+morf'

        self.script_dir = Path(__file__).resolve().parent
        self.data_dir = self.script_dir.parent.parent.parent / 'data'

        shutil.copy(f'{self.data_dir}/model_config_templates/xbeach/params_base_{self.stat_label.lower()}.txt', f'{self.dict_folders["run"]}/params.txt')
        #shutil.copy(f'{self.data_dir}/model_config_templates/xbeach/params_base_{self.stat_label.lower()}_2.txt', f'{self.dict_folders["run"]}/params.txt')
        print ('\n*** Copying base swan configuration file into run folder ***\n')

        for key,value in self.dic_ini_data.items():
            if (type(value)==float) or (type(value)==int):
                self.dic_ini_data[key]=str(value)
            self.dic_ini_data[key]=str(value)
        utils.fill_files(f'{self.dict_folders["run"]}/params.txt',self.dic_ini_data)
