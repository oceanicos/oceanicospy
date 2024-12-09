import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import subprocess
import cdsapi
from datetime import datetime
import pandas as pd
import numpy as np
import shutil
import util
import glob as glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

class oper_carpetas():
    def __init__ (self,root_path,ini_date):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
        self.info_path = f'{root_path}info/'
    def crear_carpetas_l1(self):
        subprocess.call(['mkdir','-p',self.run_path])
        subprocess.call(['mkdir','-p',self.data_path])
    def crear_carpetas_l2(self):
        subprocess.call(['mkdir','-p',f'{self.info_path}forc/'])
        subprocess.call(['mkdir','-p',f'{self.info_path}gridgen/'])
        subprocess.call(['mkdir','-p',f'{self.data_path}plots/'])

class dwnd_forcing():
    def __init__(self,root_path,ini_date,end_date,sbst_grd):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.forc_path = f'{root_path}info/forc/'
        self.idate = ini_date
        self.edate = end_date
        self.sbst_grd = sbst_grd
        self.raw_name = self.idate.strftime("%Y%m%d")+'_era5_raw.nc'
        self.fin_name = self.idate.strftime("%Y%m%d")+'_era5.nc'
        self.raw_path = f'{self.forc_path}{self.raw_name}'
        self.fin_path = f'{self.forc_path}{self.fin_name}'

    def link_nc_wind(self):
        util.verify_links('HrdExtra_1min_e117_Grid_gf_1.16.nc',self.forc_path,self.run_path)

    def link_nc_current(self):
        util.verify_links('currents_hycom.nc',self.forc_path,self.run_path)
    
    def dwnd_era5(self):
        print ('\n *** Downloading and modifying ERA5 data via cdsapi *** \n')
        self.check_file=util.verify_files(self.raw_path)
        if not self.check_file:
            self.era5_var_ids = ['10m_u_component_of_wind','10m_v_component_of_wind']
            self.c = cdsapi.Client()
            self.hours = list(pd.date_range(datetime(1900,1,1,0,0),\
                    datetime(1900,1,1,23,0),freq='H').strftime('%H:%M'))

            self.dates=list(pd.date_range(self.idate.date(),self.edate.date(),freq='d').strftime('%Y-%m-%d'))

            self.area = self.sbst_grd['latmax']+'/'+self.sbst_grd['lonmin']+'/'+self.sbst_grd['latmin']+'/'+self.sbst_grd['lonmax']
            self.c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'variable':self.era5_var_ids,
                'product_type':'reanalysis',
                'area':self.area,		#N/W/S/E
                'date':self.dates[:],
                'time':self.hours[:],
                'format':'netcdf'
            },
                self.raw_path)

    def mdf_era5(self):
        self.check_file=util.verify_files(self.fin_path)
        if not self.check_file:
            # This can be done with xarray
            os.system(f'ncpdq -h -O -a -latitude {self.raw_path} {self.fin_path}') 

    def plt_era5(self):
        pass
    
    def lnk_era5(self):
        os.system(f'rm -rf {self.run_path}*.nc')
        util.verify_links(self.fin_name,self.forc_path,self.run_path)

class bthm_data():
    def __init__(self,root_path,gridgen_path,ini_date,name,res,sbst_grd):
        self.gridgen_path = gridgen_path 
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.inp_path = f'{root_path}info/inp/'
        self.info_grid_path = f'{root_path}info/gridgen/'
        self.name = name
        self.res = res
        self.sbst_grd = sbst_grd
        self.bath_files = ['bottom.inp','mask.inp','obstr.inp']

    def generate_bthm(self):
        print ('\n *** Generating bathymetry data with gridgen ***\n')
        self.vrf_list = np.array([util.verify_files(f'{self.info_grid_path}{file}') for file in self.bath_files])

        if np.all(self.vrf_list):
            print(f'The bathimetric files : {self.bath_files} already exist')
        else:        
            shutil.copy(f'{self.inp_path}create_grid_code.m', f'{self.info_grid_path}create_grid.m')
            self.gridgen_dict={'bin_path':f'{self.gridgen_path}bin/','reference_path':f'{self.gridgen_path}reference_data/',\
                            'output_path':self.info_grid_path,'res_x':self.res,'res_y':self.res,'name_case':self.name}
            self.gridgen_dict.update(self.sbst_grd)
            util.fill_files(f'{self.info_grid_path}create_grid.m',self.gridgen_dict)
            print('Please run the following commands in a node in Spartan before to continue:\n')
            print(f'cd {self.info_grid_path}')
            print('matlab -nodisplay -nodesktop -batch "run create_grid.m"\n')
            raise UserWarning('The model will fail if the previous commands are not runned')

    def plt_bottom(self):
        self.bottom=np.genfromtxt(f'{self.info_grid_path}bottom.inp')
        self.bottom[self.bottom>=1000]=np.nan
        self.lat=np.arange(float(self.sbst_grd['latmin']),float(self.sbst_grd['latmax'])+float(self.res),float(self.res))
        self.lon=np.arange(float(self.sbst_grd['lonmin']),float(self.sbst_grd['lonmax'])+float(self.res),float(self.res))

        self.fig,self.ax1 = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

        self.land = cfeature.NaturalEarthFeature('physical', 'land', \
                scale='50m', edgecolor='k', facecolor=cfeature.COLORS['land'])

        self.ax1.add_feature(self.land, facecolor='gray',alpha=0.5)

        cf=self.ax1.contourf(self.lon,self.lat,self.bottom[:,:],30,transform=ccrs.PlateCarree(),extend='both',cmap='Blues_r')

        self.ax1.set(xlabel='Longitude',ylabel='Latitude',xticks=(np.arange(-98,-53,4).tolist()), \
                        yticks=(np.arange(8,33,3).tolist()),ylim=(float(self.sbst_grd['latmin']),float(self.sbst_grd['latmax'])),xlim=(float(self.sbst_grd['lonmin'])-360,float(self.sbst_grd['lonmax'])-360),\
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

        self.fig.savefig(f'{self.info_grid_path}bottom_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
        return self.fig,self.ax1

    def plt_mask(self):
        self.mask=np.genfromtxt(f'{self.info_grid_path}mask.inp')
        self.lat=np.arange(float(self.sbst_grd['latmin']),float(self.sbst_grd['latmax'])+float(self.res),float(self.res))
        self.lon=np.arange(float(self.sbst_grd['lonmin']),float(self.sbst_grd['lonmax'])+float(self.res),float(self.res))

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

        self.fig.savefig(f'{self.info_grid_path}mask_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
        return self.fig,self.ax1

    def plt_obstr(self):
        self.obstr=np.genfromtxt(f'{self.info_grid_path}obstr.inp')
        self.obstr=self.obstr/100
        self.lat=np.arange(float(self.sbst_grd['latmin']),float(self.sbst_grd['latmax'])+float(self.res),float(self.res))
        self.lon=np.arange(float(self.sbst_grd['lonmin']),float(self.sbst_grd['lonmax'])+float(self.res),float(self.res))

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

        self.fig.savefig(f'{self.info_grid_path}Sx_obstr_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

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

        self.fig.savefig(f'{self.info_grid_path}Sy_obstr_gridgen.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

    def lnk_bthm_data(self):
        for file in self.bath_files:
            util.verify_links(file,self.info_grid_path,self.run_path)

class exec_files():
    def __init__(self,root_path,exe_path,ini_date):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.exe_path = exe_path
    
    def copy_exe(self):
        for item in os.listdir(self.run_path):
            item_path = os.path.join(self.run_path, item)
            # Check if the item is a symbolic link
            if item in os.listdir(self.exe_path):
                if os.path.islink(item_path):
                # print(f"Deleting symbolic link: {item_path}")
                    os.remove(item_path)  # Delete the sy
        os.system(f'ln -s {self.exe_path}* {self.run_path}')

class strt_grd_inp():
    def __init__(self,root_path,ini_date,start_dict,grid_dict):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.inp_path = f'{root_path}info/inp/'
        self.start_dict = start_dict
        self.grid_dict = grid_dict

    def fill_grd(self):
        print ('\n*** Editing ww3_grid.inp and ww3_strt.inp ***\n')
        shutil.copy(f'{self.inp_path}ww3_grid.inp_code', f'{self.run_path}ww3_grid.inp')
        util.fill_files(f'{self.run_path}ww3_grid.inp',self.grid_dict)

    def fill_strt(self):
        shutil.copy(f'{self.inp_path}ww3_strt.inp_code', f'{self.run_path}ww3_strt.inp')
        util.fill_files(f'{self.run_path}ww3_strt.inp',self.start_dict)

class prnc_inp():
    def __init__(self,root_path,ini_date,prnc_dict):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.inp_path = f'{root_path}info/inp/'
        self.prnc_dict = prnc_dict

    def fill_prnc(self):
        print ('\n*** Editing ww3_prnc.inp ***\n')
        shutil.copy(f'{self.inp_path}ww3_prnc_wind.inp_code', f'{self.run_path}ww3_prnc_wind.inp')
        # shutil.copy(f'{self.inp_path}ww3_prnc_current.inp_code', f'{self.run_path}ww3_prnc_current.inp')
        util.fill_files(f'{self.run_path}ww3_prnc_wind.inp',self.prnc_dict)
        # util.fill_files(f'{self.run_path}ww3_prnc_current.inp',self.prnc_dict)


class exe_pre():
    def __init__(self,root_path,exe_path,ini_date):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.exe_path = exe_path
        self.pre_programs = ['ww3_grid','ww3_strt','ww3_prnc']
        self.forcings = ['wind','current']

    def run_exe_pre(self):
        print ('\n*** Running pre-processing programs of ww3 ***\n')
        for program in self.pre_programs:
            if program == 'ww3_prnc':
                for forc in self.forcings:
                    f = open(f'{self.run_path}{program[4:]}_{forc}.log', "w") # this creates the file
                    os.system(f'cp {self.run_path}{program}_{forc}.inp {self.run_path}{program}.inp' )
                    subprocess.call([f'{self.run_path}{program}'],cwd=self.run_path,stdout=f)  
            else:
                f = open(f'{self.run_path}{program[4:]}.log', "w") # this creates the file
                subprocess.call([f'{self.run_path}{program}'],cwd=self.run_path,stdout=f)  
        os.system(f'rm -rf {self.run_path}out_*.ww3')



if __name__ == '__main__':
    # print_a() is only executed when the module is run directly.
    print('executed directly')