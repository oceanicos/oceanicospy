import numpy as np
import glob as glob
from scipy.interpolate import griddata
import os
import shapefile

from .. import utils
from ..init_setup import InitialSetup

class Vegetation(InitialSetup):

    def __init__ (self,dict_species,dict_locations,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dict_species=dict_species
        self.dict_locations=dict_locations
        self.dict_veggie={}

    def definition_species(self):
        os.system(f'touch {self.dict_folders["run"]}veggielist.txt')

        with open(f'{self.dict_folders["run"]}veggielist.txt','w') as f:
            for specie in self.dict_species.keys():
                f.write(f'{specie}.txt\n')
        f.close()

        # for specie in self.dict_species.keys():
        #     os.system(f'touch {specie}.txt')

        self.dict_veggie={'number_species':str(len(self.dict_species.keys())),'vegetation_file':'veggielist.txt'}

        for key,value in self.dict_veggie.items():
            self.dict_veggie[key]=str(value)
    
    def params_per_specie(self):
        for specie in self.dict_species.keys():
            with open(f'{self.dict_folders["run"]}{specie}.txt','w') as f:
                for key,value in self.dict_species[specie].items():
                    if type(value)==list:
                        value_to_write=' '.join([str(i) for i in value])
                    else:
                        value_to_write=value
                    f.write(f'{key} = {value_to_write}\n')
            f.close()
    
    def create_veggie_map(self):
        x=np.genfromtxt(f'{self.dict_folders["run"]}x_profile.grd')
        max_x=np.nanmax(x)
        veggie_locs=np.zeros(x.shape)
        for idx,dic_space in enumerate(self.dict_locations.values()):
            abscisa_start=max_x+dic_space['loc']
            abscisa_end=abscisa_start+dic_space['length']
            index_end = np.argmin(np.abs(x - abscisa_end))
            index_start = np.argmin(np.abs(x - abscisa_start))
            veggie_locs[index_start:index_end]=int(idx+1)

        np.savetxt(f'{self.dict_folders["run"]}veggiemapfile.txt',veggie_locs,fmt='%d')

        self.dict_veggie.update({'vegetation_map_file':'veggiemapfile.txt'})

        
    def fill_vegetation_section(self):
        """
        Fill the boundaries section of the simulation.
        Args:
            *args: Variable length argument list.
        Returns:
            None
        """
        for param in self.dict_veggie:
            self.dict_veggie[param]=str(self.dict_veggie[param])
        utils.fill_files(f'{self.dict_folders["run"]}params.txt',self.dict_veggie)

