import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import util
from . import init_custom

import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.dates as mdates
import math
import numpy as np
import datetime as dt
from matplotlib.lines import Line2D
import xarray as xr

class Dir_spread():
        def __init__(self,root_path,ini_date,end_date,buoys_id,locs_buoys) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/spectra_2d/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date
                self.buoys_id = buoys_id
                self.locs_buoys = locs_buoys

        def preparing_data(self):  
                self.data_ounp=util.read_data_int_stations(f'{self.data_path}ww3.{self.idate.year}_tab_params.nc')

                self.spread={}

                for buoy in list(self.locs_buoys.keys()):
                        self.lon=self.locs_buoys[buoy][1]+360
                        self.lat=self.locs_buoys[buoy][0]

                        self.data_buoy = util.ord_buoy_data(buoy)

                        self.hs_buoy=self.data_buoy.hs[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)] # First day is cutted off due to spin-up
                        self.new_x_index=(self.hs_buoy.index-pd.Timedelta(minutes=40))

                        # Directiona spread from buoy
                        self.time_buoy,self.sprm_values_buoy=util.read_dir_spread_buoys(buoy)  # reading buoy results
                        self.sprm_buoy=pd.Series(index=self.time_buoy,data=self.sprm_values_buoy)
                        self.sprm_buoy=self.sprm_buoy[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]

                        # Directional spread from the model
                        self.sprp_model=self.data_ounp[buoy].sprp[self.new_x_index]
                        self.sprm_model=self.data_ounp[buoy].sprm[self.new_x_index]

                        self.spread[buoy]=dict(model_mean=self.sprm_model,model_peak=self.sprp_model,
                                               buoy=self.sprm_buoy)

                return  self.spread

        def plotting_one_plot(self,axes,dict_var,id_buoy,label,color):
                
                self.colors_dict = dict(buoy='k',ERA5='c',wind_buoy='sandybrown',wind_era='salmon')
                spr_dict=dict_var['spr'][id_buoy]

                for key in spr_dict.keys():
                        if key == 'model_mean':
                                line_type='-.'
                        else:
                                line_type='-'
                        axes.plot(spr_dict[key],line_type,label=key,color=color)

                #         if flag == 'compare':
                #                 if label=='winp' or label=='cos2':
                #                         label='$cos^2$'
                #                 elif label=='winp4' or label=='cos4':
                #                         label='$cos^4$'
                #                 elif label=='bim2' or label=='bim':
                #                         label='bim'
                #                 elif label =='all':
                #                         label='$cos^4$+bim'
                #                 elif label=='expB':
                #                         label='expA'
                #                 elif label=='expC':
                #                         label='expB'
                #                 elif label=='expD':
                #                         label='expC'
                        
                self.myFmt = mdates.DateFormatter('%m-%d-%y')
                axes.set_xlim(dt.datetime(2020,5,1,0),dt.datetime(2020,6,2))

                self.fmt_day = mdates.DayLocator()        
                axes.xaxis.set_major_formatter(self.myFmt)
                axes.xaxis.set_minor_locator(self.fmt_day)
                axes.grid(True,linestyle='dotted',zorder=0)

                return axes

        def setting_up_plot(self,label,color):
                self.all_data_vbles=dict(spr=self.preparing_data())
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                self.figs={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.fig,self.axes=plt.subplots(1,1,figsize=(11,3))        
                        self.dict_axes[id_buoy][0]=self.plotting_one_plot(self.axes,self.all_data_vbles,id_buoy,label,color)
                        self.fig.legend()
                        self.figs[id_buoy]=self.fig
                        self.fig.savefig(f'{self.plots_path}Fig11_dir_spread_{id_buoy}.pdf',dpi=400,bbox_inches='tight',pad_inches=0.05)
                return self.dict_axes,self.figs
        
        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.adding_variables=dict(spr=obj2.preparing_data())
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.dict_axes[id_buoy][0]=self.plotting_one_plot(dict2[id_buoy][0],self.adding_variables,id_buoy,label,color)

                        figs2[id_buoy].savefig(f'{self.plots_path}Fig11_dir_spread_{id_buoy}.pdf',dpi=400,bbox_inches='tight',pad_inches=0.05)
