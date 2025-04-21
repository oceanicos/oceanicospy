import numpy as np
import glob as glob
import pandas as pd
from .. import utils
import shapefile
import os
from ..init_setup import InitialSetup

class MakeGrid(InitialSetup):
    """
    A class for creating a Xbeach computational grid from bathymetry data and filling grid information in a SWAN file.
    Args:
        root_path (str): The root path of the project.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
    Attributes:
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
    Methods:

    """
    def __init__ (self,dx,dy,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dx=dx
        self.dy=dy   

    def params_1D_from_bathy(self):
        dat_files=glob.glob(f'{self.dict_folders["input"]}*.dat')
        print(f'Using bathymetry file: {dat_files}')
        bathy_file = [file for file in dat_files if 'Perfil_0' in file][0]
        data=np.loadtxt(bathy_file)
        x=data[:,0]  # No esta reversado, caution!
        y=np.zeros(data[:,1].shape) 

        np.savetxt(f'{self.dict_folders["run"]}x_profile.grd',x,fmt='%f')
        np.savetxt(f'{self.dict_folders["run"]}y_profile.grd',y,fmt='%f')

        grid_dict={'xfilepath':'x_profile.grd','yfilepath':'y_profile.grd','meshes_x':len(x)-1,'meshes_y':0}
        for key,value in grid_dict.items():
            grid_dict[key]=str(value)
        return grid_dict

    def params_2D_from_bathy(self):
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


        x=np.arange(xmin-xmin,xmin-xmax+2,2)  # Caution
        y=np.arange(ymin-ymin,ymax-ymin+2,2)

        X,Y=np.meshgrid(x,y)

        np.savetxt(f'{self.dict_folders["run"]}x_profile.grd',X,fmt='%f')
        np.savetxt(f'{self.dict_folders["run"]}y_profile.grd',Y,fmt='%f')

        grid_dict={'xfilepath':'x_profile.grd','yfilepath':'y_profile.grd','meshes_x':len(x)-1,'meshes_y':len(y)-1}
        for key,value in grid_dict.items():
            grid_dict[key]=str(value)
        return grid_dict        

    def params_2D_from_xyz(self):
        bathy_file_path = glob.glob(f'{self.dict_folders["input"]}*.csv')[0]

        bathy_data = pd.read_csv(bathy_file_path)
        print(bathy_data)
        min_lon = np.min(bathy_data.X)
        max_lon = np.max(bathy_data.X)
        min_lat = np.min(bathy_data.Y)
        max_lat = np.max(bathy_data.Y)

        ymax=max_lon
        ymin=min_lon
        xmin=max_lat
        xmax=min_lat

        print(xmin-xmin,xmin-xmax)

        x=np.arange(xmin-xmin,xmin-xmax+(10/110000),10/110000)  # Caution
        y=np.arange(ymin-ymin,ymax-ymin+(10/110000),10/110000)

        X,Y=np.meshgrid(x,y)

        np.savetxt(f'{self.dict_folders["run"]}x_profile.grd',X,fmt='%f')
        np.savetxt(f'{self.dict_folders["run"]}y_profile.grd',Y,fmt='%f')

        grid_dict={'xfilepath':'x_profile.grd','yfilepath':'y_profile.grd','meshes_x':len(x)-1,'meshes_y':len(y)-1}
        for key,value in grid_dict.items():
            grid_dict[key]=str(value)
        return grid_dict        

    # def params_from_bathy(self):
    #     bathy_file_path = glob.glob(f'{self.dict_folders["input"]}*.dat')[0]
    #     data = np.loadtxt(bathy_file_path)
    #     longitude = data[:, 0]
    #     latitude = data[:, 1]
    #     elevation = data[:, 2]

    #     min_longitude = np.min(longitude)
    #     min_latitude = np.min(latitude)

    #     max_longitude = np.max(longitude)
    #     max_latitude = np.max(latitude)
    #     min_longitude = int(np.ceil(min_longitude / 100) * 100)
    #     max_longitude = int(np.floor(max_longitude / 100) * 100)
    #     min_latitude = int(np.ceil(min_latitude / 100) * 100)
    #     max_latitude = int(np.floor(max_latitude / 100) * 100)

    #     x_extent=max_longitude-min_longitude
    #     y_extent=max_latitude-min_latitude

    #     nx = int(x_extent/self.dx)
    #     ny = int(y_extent/self.dy)
        
    #     grid_dict={'lon_ll_corner':min_longitude,'lat_ll_corner':min_latitude,'x_extent':x_extent,'y_extent':y_extent,'nx':nx,'ny':ny}
    #     for key,value in grid_dict.items():
    #         grid_dict[key]=str(value)

    #     return grid_dict

    def grid_from_DELFT3D(self,filename_grd):
        os.system(f'cp {self.dict_folders["input"]}{filename_grd}.grd {self.dict_folders["run"]}')
        dict_asc={'grdfilepath':f'{filename_grd}.grd','model_origin':'delft3d'}
        return dict_asc

    def fill_grid_section(self,grid_dict):
        print ('\n*** Adding/Editing grid information in params file ***\n')
        utils.fill_files(f'{self.dict_folders["run"]}params.txt',grid_dict)