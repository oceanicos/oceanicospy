import numpy as np
import pandas as pd
import subprocess

class create_folders():
    """
    A class for creating folders in a specified root path.

    Args:
        root_path (str): The root path where the folders will be created.
        ini_date (datetime, optional): The initial date for creating subfolders. Defaults to None.

    Attributes:
        root_path (str): The root path where the folders will be created.
        ini_date (datetime): The initial date for creating subfolders.
        folder_names (list): A list of folder names to be created.

    Methods:
        crear_carpetas_l1(): Creates the folders specified in folder_names.

            Creates the folders specified in folder_names.

            If ini_date is not None, subfolders are created with the format 'root_path/folder/ini_date.strftime("%Y%m%d%H")/'.
        Otherwise, subfolders are created with the format 'root_path/folder/'.
    """
    def __init__ (self,root_path,ini_date=None):
        self.root_path=root_path
        self.init_date=ini_date
        self.folder_names=['input','pros','run','output']

    def crear_carpetas_l1(self):
        for folder in self.folder_names:
            if self.ini_date is not None:
                subprocess.call(['mkdir','-p',f'{self.root_path}{folder}/{self.ini_date.strftime("%Y%m%d%H")}/'])
            else:
                subprocess.call(['mkdir','-p',f'{self.root_path}{folder}/'])

    # def crear_carpetas_l2(self):
    #     subprocess.call(['mkdir','-p',f'{self.info_path}forc/'])
    #     subprocess.call(['mkdir','-p',f'{self.info_path}gridgen/'])
    #     subprocess.call(['mkdir','-p',f'{self.data_path}plots/'])