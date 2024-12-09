import numpy as np
import glob as glob
from scipy.interpolate import griddata
import os


from .. import utils
from ..init_setup import InitialSetup

class MakeBathy(InitialSetup):
    """
    A class for preprocessing bathymetry data.

    Args:
        filename (str): The name of the output file.
        dx_bat (float): The grid spacing for the bathymetry data.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        filename (str): The name of the output file.
        dx_bat (float): The grid spacing for the bathymetry data.

    Methods:
        xyz2asc(nodata_value): Converts bathymetry data from XYZ format to ESRI ASCII Grid format.
        fill_bathy_section(dict_data): Adds or edits bathymetry information in a SWAN file.
    """
    def __init__ (self,domain_number,bathy_info=None,filename=None,dx_bat=None,*args,**kwargs):
        """
        Initialize MakeBathy class.

        Args:
            filename (str): The name of the output file.
            dx_bat (float): The grid spacing for the bathymetry data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args,**kwargs)
        self.filename=filename
        self.dx_bat=dx_bat
        self.bathy_info=bathy_info
        self.domain_number=domain_number

    def xyz2asc(self,nodata_value):
        """
        Converts bathymetry data from XYZ format to ESRI ASCII Grid format.

        Args:
            nodata_value (float): The value to replace NaN values in the grid.

        Returns:
            dict: A dictionary containing the metadata of the generated grid.
        """
        bathy_xyz_path = glob.glob(f'{self.dict_folders["input"]}*.dat')[0]
        ascfile = f'{self.dict_folders["run"]}{self.filename}.bot'
        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        # Read bathymetry file
        longitude,latitude,z = np.loadtxt(bathy_xyz_path, delimiter=' ', unpack=True)

        min_longitude = np.min(longitude)
        min_latitude = np.min(latitude)

        max_longitude = np.max(longitude)
        max_latitude = np.max(latitude)

        min_longitude = int(np.ceil(min_longitude / 100) * 100)
        max_longitude = int(np.floor(max_longitude / 100) * 100)
        min_latitude = int(np.ceil(min_latitude / 100) * 100)
        max_latitude = int(np.floor(max_latitude / 100) * 100)

        xmax=max_longitude
        xmin=min_longitude
        ymax=max_latitude
        ymin=min_latitude

        nx_bathy = int((xmax - xmin)/self.dx_bat)
        ny_bathy = int((ymax - ymin)/self.dx_bat)
        # Generate grid with data
        xi, yi = np.mgrid[xmin:xmax:(nx_bathy+1)*1j, ymin:ymax:(ny_bathy+1)*1j]

        # Interpolate bathymetry. Method can be 'linear', 'nearest' or 'cubic'
        zi = griddata((longitude,latitude), z, (xi, yi), method='linear')
        # Change Nans for values
        zi[np.isnan(zi)] = nodata_value
        # Flip array in the left/right direction
        zi = np.fliplr(zi)
        # Transpose it
        zi = zi.T
        # Write ESRI ASCII Grid file
        zi_str = np.where(zi == nodata_value, str(nodata_value), np.round(zi, 3))
        np.savetxt(ascfile, zi_str, fmt='%8s', delimiter=' ')
        print('File %s saved successfuly.' % ascfile)

        dict_asc={'lon_ll_bat_corner':min_longitude,'lat_ll_bat_corner':min_latitude,'x_bot':nx_bathy,'y_bot':ny_bathy,'spacing_x':self.dx_bat,'spacing_y':self.dx_bat}
        for key,value in dict_asc.items():
            dict_asc[key]=str(value)
        return dict_asc
    
    def asc_from_user(self):
        if self.dict_ini_data["nested_domains"]>0:
            bathy_file_path = glob.glob(f'{self.dict_folders["input"]}domain_0{self.domain_number}/*.bot')[0]
        else:
            bathy_file_path = glob.glob(f'{self.dict_folders["input"]}*.bot')[0]
        bathy_filename=bathy_file_path.split('/')[-1]

        if not utils.verify_link(bathy_filename,f'{self.dict_folders["run"]}domain_0{self.domain_number}/'):
            utils.create_link(bathy_filename,f'{self.dict_folders["input"]}domain_0{self.domain_number}/',
                                f'{self.dict_folders["run"]}domain_0{self.domain_number}/')
        # os.system(f'cp {self.dict_folders["input"]}domain_0{self.domain_number}/{bathy_filename}\
        #                         {self.dict_folders["run"]}domain_0{self.domain_number}/')


        if self.bathy_info!=None:
            self.bathy_info.update({"bathymetry.bot":bathy_filename})
            return self.bathy_info

    def fill_bathy_section(self,dict_bathy_data):



        if self.dict_ini_data["nested_domains"]>0:
            print (f'\n*** Adding/Editing bathymetry information for domain {self.domain_number} in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}domain_0{self.domain_number}/run.swn',dict_bathy_data)
        else:
            print ('\n*** Adding/Editing bathymetry information in configuration file ***\n')
            utils.fill_files(f'{self.dict_folders["run"]}run.swn',dict_bathy_data)

