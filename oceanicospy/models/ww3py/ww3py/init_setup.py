import subprocess
from . import utils
import shutil
import os

class InitialSetup():
    def __init__ (self,root_path,exe_path,dict_ini_data,ini_date=None,end_date=None):
        self.root_path=root_path
        self.exe_path=exe_path
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
        
        print ('\n*** Copying base ww3 configuration files into run folder ***\n')
        os.system(f'cp /home/fayalacruz/runs/modelling/inp_templates/ww3/ww3_* {self.dict_folders["input"]}')

    def link_exe_files(self):
        print ('\n*** Linking the executable files ***\n')
        for item in os.listdir(self.dict_folders["run"]):
            item_path = os.path.join(self.dict_folders["run"], item)
            # Check if the item is a symbolic link
            if item in os.listdir(self.exe_path):
                if os.path.islink(item_path):
                # print(f"Deleting symbolic link: {item_path}")
                    os.remove(item_path)  # Delete the sy
        os.system(f'ln -s {self.exe_path}* {self.dict_folders["run"]}')