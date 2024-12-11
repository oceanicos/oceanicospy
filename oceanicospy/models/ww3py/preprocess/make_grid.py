import shutil

from .. import utils
from ..init_setup import InitialSetup


class MakeGrid(InitialSetup):
    def __init__(self,grid_dict,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.grid_dict=grid_dict

    def fill_grid_file(self):
        print ('\n*** Editing ww3_grid.inp ***\n')
        self.grid_dict.update({'NAME':self.dict_ini_data["name"]})
        shutil.copy(f'{self.dict_folders["input"]}ww3_grid.inp_code', f'{self.dict_folders["run"]}ww3_grid.inp')
        utils.fill_files(f'{self.dict_folders["run"]}ww3_grid.inp',self.grid_dict)