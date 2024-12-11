import numpy as np
import glob as glob
from scipy.interpolate import griddata
import os
import shapefile

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
        profile2asc(nodata_value): Converts bathymetry data text format to ESRI ASCII Grid format.
        fill_bathy_section(dict_data): Adds or edits bathymetry information in a SWAN file.
    """
    def __init__ (self,filename,*args,**kwargs):
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

    def profile2asc(self):
        dat_files=glob.glob(f'{self.dict_folders["input"]}*.dat')
        bathy_file = [file for file in dat_files if 'Perfil_0' in file][0]
        print(f'Using bathymetry file: {bathy_file}')
        data=np.loadtxt(bathy_file)
        ascfile = f'{self.dict_folders["run"]}{self.filename}.dep'
        np.savetxt(ascfile,data[:,3][::-1],fmt='%f')
        print('File %s saved successfuly.' % ascfile)

        dict_asc={'depfilepath':f'{self.filename}.dep'}

        ne_layer=np.ones(data[:,1].shape)*1
        np.savetxt(f'{self.dict_folders["run"]}nelayer.dep',ne_layer,fmt='%f')

        for key,value in dict_asc.items():
            dict_asc[key]=str(value)
            
        return dict_asc


    def xyz2asc(self,dx_bat,nodata_value):
        """
        Converts bathymetry data from XYZ format to ESRI ASCII Grid format.

        Args:
            nodata_value (float): The value to replace NaN values in the grid.

        Returns:
            dict: A dictionary containing the metadata of the generated grid.
        """
        bathy_xyz_path = glob.glob(f'{self.dict_folders["input"]}*.csv')[0]
        ascfile = f'{self.dict_folders["run"]}{self.filename}.dep'
        ascfile_ne_ones = f'{self.dict_folders["run"]}ne_layer_ones.dep'
        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        # Read bathymetry file
        longitude,latitude,z = np.loadtxt(bathy_xyz_path, delimiter=',', unpack=True, skiprows=1)

        # print(longitude,latitude,z)

        # Load the shapefile
        sf = shapefile.Reader(f'{self.dict_folders["input"]}Modelo_2D.shp')

        # Extract the shapes and records
        shapes = sf.shapes()
        records = sf.records()

        # Assuming the shapefile contains only one rectangle
        shape = shapes[0]

        # Extract the bounding box (min_lon, min_lat, max_lon, max_lat)
        min_lon, min_lat, max_lon, max_lat = shape.bbox

        # Print the bounding box
        # print(f'Bounding box: min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}')

        min_longitude = int(np.ceil(min_lon / 50) * 50)-50
        max_longitude = int(np.floor(max_lon / 50) * 50)+50
        min_latitude = int(np.ceil(min_lat / 50) * 50)-50
        max_latitude = int(np.floor(max_lat / 50) * 50)+50   

        ymax=max_longitude
        ymin=min_longitude
        xmin=max_latitude
        xmax=min_latitude

        nx_bathy = int((xmin - xmax)/dx_bat)
        ny_bathy = int((ymax - ymin)/dx_bat)
        # # Generate grid with data
        xi, yi = np.mgrid[xmin:xmax:(nx_bathy+1)*1j, ymin:ymax:(ny_bathy+1)*1j]  # Caution

        # # Interpolate bathymetry. Method can be 'linear', 'nearest' or 'cubic'
        zi = griddata((latitude,longitude), z, (xi, yi), method='linear')
        # # Change Nans for values
        zi[np.isnan(zi)] = nodata_value
        # Flip array in the left/right direction
        # zi = np.fliplr(zi)
        # Transpose it
        zi = zi.T
        # Write ESRI ASCII Grid file
        zi_str = np.where(zi == nodata_value, str(nodata_value), np.round(zi, 3))

        zi_ones=np.ones(zi.shape)
        np.savetxt(ascfile, zi_str, fmt='%8s', delimiter=' ')
        np.savetxt(ascfile_ne_ones, zi_ones, fmt='%8s', delimiter=' ')

        print('File %s saved successfuly.' % ascfile)

        dict_asc={'depfilepath':f'{self.filename}.dep','x_bot':nx_bathy,'y_bot':ny_bathy,'spacing_x':dx_bat,'spacing_y':dx_bat,
                  'nelayerfilepath':'ne_layer.dep'}
        for key,value in dict_asc.items():
            dict_asc[key]=str(value)
        return dict_asc
    
    def read_dry_index(self):
        index_dry = glob.glob(f'{self.dict_folders["input"]}indexes_dry_points.txt')[0]
        points= np.loadtxt(index_dry)
        xi,yi=points[:,1],points[:,2]
        ne_layer_ones=np.loadtxt(f'{self.dict_folders["run"]}ne_layer_ones.dep')
        for idx_x,idx_y in zip(xi,yi):
            ne_layer_ones[int(idx_y),-int(idx_x)]=0
        # ne_layer_ones = np.fliplr(ne_layer_ones)
        # Transpose it
        # ne_layer_ones = ne_layer_ones.T
        ascfile_ne = f'{self.dict_folders["run"]}ne_layer.dep'
        np.savetxt(ascfile_ne, ne_layer_ones, fmt='%8s', delimiter=' ')
        print('File %s saved successfuly.' % ascfile_ne)

    def bathy_from_DELFT3D(self,filename_dep,filename_nedep):
        os.system(f'cp {self.dict_folders["input"]}{filename_dep}.dep {self.dict_folders["run"]}')
        os.system(f'cp {self.dict_folders["input"]}{filename_nedep}.dep {self.dict_folders["run"]}')

        dict_asc={'depfilepath':f'{filename_dep}.dep','model_origin':'delft3d',
                  'nelayerfilepath':f'{filename_nedep}.dep'}
        return dict_asc

    def fill_bathy_section(self,dict_data):

        print ('\n*** Adding/Editing bathymetry information in params file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}params.txt',dict_data)


