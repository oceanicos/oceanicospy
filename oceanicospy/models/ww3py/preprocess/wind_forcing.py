import cdsapi
from datetime import datetime
import pandas as pd
import os,sys
import shutil

from .. import utils
from ..init_setup import InitialSetup


class WindForcing():
    def __init__(self,root_path,ini_date,end_date,sbst_grd,prnc_dict):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.forc_path = f'{root_path}info/forc/'
        self.idate = ini_date
        self.edate = end_date
        self.sbst_grd = sbst_grd
        self.raw_name = self.idate.strftime("%Y%m%d")+'_era5_raw.nc'
        self.fin_name = self.idate.strftime("%Y%m%d")+'_era5.nc'
        self.raw_path = f'{self.forc_path}{self.raw_name}'
        self.fin_path = f'{self.forc_path}{self.fin_name}'
        self.prnc_dict = prnc_dict

    def link_nc_wind(self):
        utils.verify_links('HrdExtra_1min_e117_Grid_gf_1.16.nc',self.forc_path,self.run_path)
    
    def dwnd_era5(self):
        print ('\n *** Downloading and modifying ERA5 data via cdsapi *** \n')
        self.check_file=utils.verify_files(self.raw_path)
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
        self.check_file=utils.verify_files(self.fin_path)
        if not self.check_file:
            # This can be done with xarray
            os.system(f'ncpdq -h -O -a -latitude {self.raw_path} {self.fin_path}') 

    def plt_era5(self):
        pass
    
    def lnk_era5(self):
        os.system(f'rm -rf {self.run_path}*.nc')
        utils.verify_links(self.fin_name,self.forc_path,self.run_path)

    def fill_prnc(self):
        print ('\n*** Editing ww3_prnc.inp ***\n')
        shutil.copy(f'{self.inp_path}ww3_prnc_wind.inp_code', f'{self.run_path}ww3_prnc_wind.inp')
        # shutil.copy(f'{self.inp_path}ww3_prnc_current.inp_code', f'{self.run_path}ww3_prnc_current.inp')
        utils.fill_files(f'{self.run_path}ww3_prnc_wind.inp',self.prnc_dict)
        # util.fill_files(f'{self.run_path}ww3_prnc_current.inp',self.prnc_dict)