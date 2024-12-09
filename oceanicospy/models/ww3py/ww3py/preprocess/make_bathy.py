import numpy as np
import glob as glob
from .. import utils
import shutil
from ..init_setup import InitialSetup

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

class MakeBathy(InitialSetup):
    def __init__(self,gridgen_path,resolution,subset_grd,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gridgen_path = gridgen_path
        self.resolution = resolution
        self.subset_grd = subset_grd
        self.bath_files = ['bottom.inp','mask.inp','obstr.inp']

    def generate_bathy_files(self):
        print ('\n *** Generating bathymetry data with gridgen ***\n')
        self.vrf_list = np.array([utils.verify_file(f'{self.dict_folders["input"]}{file}') for file in self.bath_files])

        if np.all(self.vrf_list):
            print(f'The bathimetric files : {self.bath_files} already exist')
        else:
            # Using gridgen to create bathymetry        
            shutil.copy('/home/fayalacruz/runs/modelling/inp_templates/ww3/create_grid_code.m', f'{self.dict_folders["input"]}create_grid.m')
            self.gridgen_dict={'bin_path':f'{self.gridgen_path}bin/','reference_path':f'{self.gridgen_path}reference_data/',\
                            'output_path':self.dict_folders["input"],'res_x':self.resolution,'res_y':self.resolution,'name_case':self.dict_ini_data["name"]}
            self.gridgen_dict.update(self.subset_grd)
            utils.fill_files(f'{self.dict_folders["input"]}create_grid.m',self.gridgen_dict)
            print('Please run the following commands in a node in Spartan before to continue:\n')
            print(f'cd {self.dict_folders["input"]}')
            print('matlab -nodisplay -nodesktop -batch "run create_grid.m"\n')
            raise UserWarning('The model will fail if the previous commands are not runned')

    def plt_bottom(self):
        self.bottom=np.genfromtxt(f'{self.dict_folders["input"]}bottom.inp')
        self.bottom[self.bottom>=1000]=np.nan
        self.lat=np.arange(float(self.subset_grd['latmin']),float(self.subset_grd['latmax'])+float(self.resolution),float(self.resolution))
        self.lon=np.arange(float(self.subset_grd['lonmin']),float(self.subset_grd['lonmax'])+float(self.resolution),float(self.resolution))

        self.fig,self.ax1 = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

        self.land = cfeature.NaturalEarthFeature('physical', 'land', \
                scale='50m', edgecolor='k', facecolor=cfeature.COLORS['land'])

        self.ax1.add_feature(self.land, facecolor='gray',alpha=0.5)

        cf=self.ax1.contourf(self.lon,self.lat,self.bottom[:,:],30,transform=ccrs.PlateCarree(),extend='both',cmap='Blues_r')

        self.ax1.set(xlabel='Longitude',ylabel='Latitude',xticks=(np.arange(-98,-53,4).tolist()), \
                        yticks=(np.arange(8,33,3).tolist()),ylim=(float(self.subset_grd['latmin']),float(self.subset_grd['latmax'])),xlim=(float(self.subset_grd['lonmin'])-360,float(self.subset_grd['lonmax'])-360),\
                            title='Bottom from gridgen')
        lon_formatter = LongitudeFormatter(number_format='g',
                                        degree_symbol='°')
        lat_formatter = LatitudeFormatter(number_format='g',
                                        degree_symbol='°')
        self.ax1.xaxis.set_major_formatter(lon_formatter)
        self.ax1.yaxis.set_major_formatter(lat_formatter)

        self.cax = self.fig.add_axes([self.ax1.get_position().x1+0.02,self.ax1.get_position().y0,\
                                0.015,self.ax1.get_position().height])
        self.cbar=plt.colorbar(cf,cax=self.cax,orientation="vertical",pad=0.12)
        self.cbar.ax.set_ylabel('Bathimetry [m]',rotation=90,labelpad=0.45)

        self.fig.savefig(f'{self.dict_folders["input"]}bottom_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
        return self.fig,self.ax1

    def plt_mask(self):
        self.mask=np.genfromtxt(f'{self.dict_folders["input"]}mask.inp')
        self.lat=np.arange(float(self.subset_grd['latmin']),float(self.subset_grd['latmax'])+float(self.resolution),float(self.resolution))
        self.lon=np.arange(float(self.subset_grd['lonmin']),float(self.subset_grd['lonmax'])+float(self.resolution),float(self.resolution))

        self.fig,self.ax1 = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

        cf=self.ax1.contourf(self.lon,self.lat,self.mask[:,:],1,transform=ccrs.PlateCarree(),cmap='Set2')

        self.ax1.set(xlabel='Longitude',ylabel='Latitude',xticks=(np.arange(-98,-57,4).tolist()), \
                        yticks=(np.arange(8,33,3).tolist()),ylim=(8.5,32.5),xlim=(-98,-58),\
                            title='Mask from gridgen')
        lon_formatter = LongitudeFormatter(number_format='g',
                                        degree_symbol='°')
        lat_formatter = LatitudeFormatter(number_format='g',
                                        degree_symbol='°')
        self.ax1.xaxis.set_major_formatter(lon_formatter)
        self.ax1.yaxis.set_major_formatter(lat_formatter)

        self.cax = self.fig.add_axes([self.ax1.get_position().x1+0.02,self.ax1.get_position().y0,\
                                0.015,self.ax1.get_position().height])
        self.cbar=plt.colorbar(cf,cax=self.cax,orientation="vertical",pad=0.12)
        self.cbar.ax.set_ylabel('Mask label',rotation=90,labelpad=0.45)

        self.fig.savefig(f'{self.dict_folders["input"]}mask_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
        return self.fig,self.ax1

    def plt_obstr(self):
        self.obstr=np.genfromtxt(f'{self.dict_folders["input"]}obstr.inp')
        self.obstr=self.obstr/100
        self.lat=np.arange(float(self.subset_grd['latmin']),float(self.subset_grd['latmax'])+float(self.resolution),float(self.resolution))
        self.lon=np.arange(float(self.subset_grd['lonmin']),float(self.subset_grd['lonmax'])+float(self.resolution),float(self.resolution))

        self.fig,self.ax1 = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

        self.land = cfeature.NaturalEarthFeature('physical', 'land', \
                scale='50m', edgecolor='k', facecolor=cfeature.COLORS['land'])

        self.ax1.add_feature(self.land, facecolor='gray',alpha=0.5)

        cf=self.ax1.contourf(self.lon,self.lat,self.obstr[:int(self.obstr.shape[0]/2),:],30,transform=ccrs.PlateCarree())

        self.ax1.set(xlabel='Longitude',ylabel='Latitude',xticks=(np.arange(-98,-57,4).tolist()), \
                        yticks=(np.arange(8,33,3).tolist()),ylim=(8.5,32.5),xlim=(-98,-58),\
                            title='Obstr from gridgen')
        lon_formatter = LongitudeFormatter(number_format='g',
                                        degree_symbol='°')
        lat_formatter = LatitudeFormatter(number_format='g',
                                        degree_symbol='°')
        self.ax1.xaxis.set_major_formatter(lon_formatter)
        self.ax1.yaxis.set_major_formatter(lat_formatter)

        self.cax = self.fig.add_axes([self.ax1.get_position().x1+0.02,self.ax1.get_position().y0,\
                                0.015,self.ax1.get_position().height])
        self.cbar=plt.colorbar(cf,cax=self.cax,orientation="vertical",pad=0.12)
        self.cbar.ax.set_ylabel('obstruction scale',rotation=90,labelpad=0.45)

        self.fig.savefig(f'{self.dict_folders["input"]}Sx_obstr_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

        self.fig2,self.ax2 = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

        self.ax2.add_feature(self.land, facecolor='gray',alpha=0.5)

        cf2=self.ax2.contourf(self.lon,self.lat,self.obstr[int(self.obstr.shape[0]/2):,:],30,transform=ccrs.PlateCarree())

        self.ax2.set(xlabel='Longitude',ylabel='Latitude',xticks=(np.arange(-98,-57,4).tolist()), \
                        yticks=(np.arange(8,33,3).tolist()),ylim=(8.5,32.5),xlim=(-98,-58),\
                            title='Obstr from gridgen')

        self.ax2.xaxis.set_major_formatter(lon_formatter)
        self.ax2.yaxis.set_major_formatter(lat_formatter)

        self.cax2 = self.fig2.add_axes([self.ax2.get_position().x1+0.02,self.ax2.get_position().y0,\
                                0.015,self.ax2.get_position().height])
        self.cbar2=plt.colorbar(cf2,cax=self.cax2,orientation="vertical",pad=0.12)
        self.cbar2.ax.set_ylabel('obstruction scale',rotation=90,labelpad=0.45)

        self.fig.savefig(f'{self.dict_folders["input"]}Sy_obstr_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

    def link_bathy_data(self):
        for file in self.bath_files:
            if utils.verify_link(file,self.dict_folders["run"]):
                continue
            else:
                utils.create_link(file,self.dict_folders["input"],self.dict_folders["run"])