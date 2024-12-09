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

class Stickplot():
        def __init__(self,root_path,ini_date,end_date,buoys_id,locs_buoys) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/series/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date
                self.buoys_id = buoys_id
                self.locs_buoys = locs_buoys
                self.vbles_to_plot = ['dir']

        def preparing_data(self):  
                self.data_ounp=util.read_data_int_stations(f'{self.data_path}ww3.{self.idate.year}_tab_params.nc')

                self.hs={}
                self.u10={}
                self.dirs={}

                self.wind_params=util.read_data_extra_stations(f'{self.data_path}ww3.{self.idate.year}_extra_params.nc')

                # Getting ERA5 series in the location of the buoys
                self.series_era5_buoys ={}
                self.series_era5_buoys_u ={}
                self.series_era5_buoys_v ={}

                for buoy in list(self.locs_buoys.keys()):
                        self.lon=self.locs_buoys[buoy][1]+360
                        self.lat=self.locs_buoys[buoy][0]

                        self.data_buoy = util.ord_buoy_data(buoy)

                        self.hs_buoy=self.data_buoy.hs[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)] # First day is cutted off due to spin-up
                        self.new_x_index=(self.hs_buoy.index-pd.Timedelta(minutes=40))

                        self.result,self.result_u,self.result_v=util.read_era5_buoys(f'{self.run_path}/{self.idate.strftime("%Y%m%d")}_era5.nc',self.lon,self.lat)
                        self.series_era5_buoys[buoy]=self.result
                        self.series_era5_buoys_u[buoy]=self.result_u
                        self.series_era5_buoys_v[buoy]=self.result_v

                        # u10 data
                        self.wnd_spd_buoy=self.data_buoy.wspd[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]

                        self.wnd_spd_era = self.series_era5_buoys[buoy][self.idate+relativedelta(hours=24):]
                        self.u10[buoy] = dict(ERA5=self.wnd_spd_era,buoy=self.wnd_spd_buoy,u_comp_era=self.series_era5_buoys_u[buoy][self.idate+relativedelta(hours=24):],
                                                v_comp_era=self.series_era5_buoys_v[buoy][self.idate+relativedelta(hours=24):])

                        # wind and wave dir data
                        self.wvdir_buoy=self.data_buoy.dir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wndir_buoy=self.data_buoy.wndir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wvdir_model=self.data_ounp[buoy].dirm[self.new_x_index]
                        self.wndir_era5=(270-np.degrees(np.arctan2(self.series_era5_buoys_v[buoy][self.idate+relativedelta(hours=24):],self.series_era5_buoys_u[buoy][self.idate+relativedelta(hours=24):])))%360
                        self.wndir_model=self.wind_params[buoy].wnddir

                        self.dirs[buoy]=dict(model=self.wvdir_model,buoy=self.wvdir_buoy,wind_buoy=self.wndir_buoy,wind_era=self.wndir_era5)

                return  self.u10,self.dirs

        def plotting_one_stickplot(self,axes,dict_var,id_buoy,legends,label,color):
                
                self.colors_dict = dict(buoy='k',ERA5='c',wind_buoy='sandybrown',wind_era='salmon')
                self.u_comp_era=dict_var['u10'][id_buoy]['u_comp_era']
                self.v_comp_era=dict_var['u10'][id_buoy]['v_comp_era']
                self.U_era=dict_var['u10'][id_buoy]['ERA5']

                self.wvdir=dict_var['dir'][id_buoy]['model']
                self.inv_magnitud_series=np.ones(dict_var['dir'][id_buoy]['model'].values.shape)*10
                self.u_comp_model=self.inv_magnitud_series*-np.sin(np.radians(self.wvdir))
                self.v_comp_model=self.inv_magnitud_series*np.cos(np.radians(self.wvdir))

                props = {'units' : "dots",
                        'width' : 2,
                        'headwidth': 3,
                        'headlength': 3,
                        'headaxislength': 3,
                        'scale' : 0.5,
                        }

                label_scale = 10
                unit_label = "%3g %s"%(label_scale, 'm/s')

                y_wind = np.zeros_like(self.U_era.values)
                y_wave = np.zeros_like(self.inv_magnitud_series)

                self.Q_wind = axes.quiver(self.U_era.index[::24], y_wind[::24], 
                                          self.u_comp_era.values[::24], self.v_comp_era.values[::24],
                                          zorder=3,**props)
                axes.quiverkey(self.Q_wind, X=0.1, Y=0.95, U=label_scale, label=unit_label, coordinates='axes', labelpos='S')

                self.Q_wave = axes.quiver(self.U_era.index[::24], y_wave[::24], 
                                          self.u_comp_model.values[::24], self.v_comp_model.values[::24],
                                          zorder=4,**props,color=color)
                yaxis = axes.yaxis
                axes.set_ylim(-0.2,0.2)
                yaxis.set_ticklabels([])


                if label =='all':
                        label='$cos^4$+bim'
                        axes.text(0.97, 0.87, '(a)', transform=axes.transAxes, size=12)

                patch=mpatches.Patch(color=color,label=label)             
                line=Line2D([], [], color=color,label=label,lw=2)
                legends.append(line)

                # for key in self.dict_vars.keys():

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

                #                 # self.ax.plot(self.dict_vars['model'],c=color,label=f'{first_line}\n{second_line.rjust(total_spaces+2)}',lw=0.9)
                #                 self.ax.plot(self.dict_vars['model'],c=color,lw=0.9)

                #                 break
                #         else:

                #                 if key =='model':
                #                         if label == 'all':
                #                                 label='$cos^4$+bim'
                #                         # self.ax.plot(self.dict_vars[key],c=color,label=f'{first_line}\n{second_line.rjust(total_spaces+2)}',lw=0.9)
                #                         self.ax.plot(self.dict_vars[key],c=color,lw=0.9)
                #                         label_scale = 10
                #                         unit_label = "%3g %s"%(label_scale, 'm/s')
                                        
                #                         # y = np.zeros_like(speeds)

                #                         # self.Q = self.ax.quiver(times, y, u, v, **props)
                #                         # self.ax.quiverkey(self.Q, X=0.1, Y=0.95, U=label_scale, label=unit_label, coordinates='axes', labelpos='S')


                #                 # else:
                #                 #         self.ax.plot(self.dict_vars[key],c=self.colors_dict[key],label=key,markersize=3,lw=0.9)


                # if param =='u10':
                #         if label=='ctrl':
                #                 self.ax.text(0.97, 0.87, "b)",fontsize=17,transform=self.ax.transAxes)
                #         self.ax.set(ylabel="Wind speed [m/s]")
                #         self.ax.tick_params(which='minor', length=3, color='k')
                #         self.ax.tick_params(which='major', length=5, color='k')
                # elif param == 'dir':
                #         if label=='ctrl':
                #                 self.ax.text(0.97, 0.87, "c)",fontsize=17,transform=self.ax.transAxes)

                #         self.ax.set(ylabel="Direction [°]")
                #         self.ax.tick_params(which='minor', length=3, color='k')
                #         self.ax.tick_params(which='major', length=5, color='k')
                # else:
                #         self.ax.set(ylabel="Wave energy $[m^{2}]$")

                # if self.idate.year == 2020:
                #         self.myFmt = mdates.DateFormatter('%m-%d-%y')
                #         self.ax.set_xlim(dt.datetime(2020,5,1,0),dt.datetime(2020,6,2))

                # else:
                #         self.myFmt = mdates.DateFormatter('%d-%b')
                #         self.ax.set_xlim(dt.datetime(2004,9,14,0),dt.datetime(2004,9,16,7))
                #         self.ax.set_yticks(np.arange(0,13,3))
                #         self.ax.set_yticklabels(np.arange(0,13,3))
                #         self.ax.set_xticks([dt.datetime(2004,9,14,0),dt.datetime(2004,9,15,0),dt.datetime(2004,9,16,0)])
                        
                self.myFmt = mdates.DateFormatter('%m-%d-%y')
                axes.set_xlim(dt.datetime(2020,5,1,0),dt.datetime(2020,6,2))

                self.fmt_day = mdates.DayLocator()        
                axes.xaxis.set_major_formatter(self.myFmt)
                axes.xaxis.set_minor_locator(self.fmt_day)
                axes.grid(True,alpha=0.5,zorder=0)

                return axes,legends

        def setting_up_plot(self,label,color):
                self.all_data_vbles=dict(u10=self.preparing_data()[0],dir=self.preparing_data()[1])
                self.dict_axes={buoy:[1,2] for buoy in self.buoys_id}
                self.figs={buoy:[1,2] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.fig,self.axes=plt.subplots(1,1,figsize=(11,1.5))        
                        legends=[] 
                        self.dict_axes[id_buoy][0],self.dict_axes[id_buoy][1]=self.plotting_one_stickplot(self.axes,self.all_data_vbles,id_buoy,legends,label,color)
                        # plt.legend(handles=legends,loc=3)
                        self.figs[id_buoy]=self.fig
                        self.fig.savefig(f'{self.plots_path}Fig8a_stickplot_{id_buoy}.eps',dpi=800,bbox_inches='tight',pad_inches=0.05)
                return self.dict_axes,self.figs
        
        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.adding_variables=dict(u10=obj2.preparing_data()[0],dir=obj2.preparing_data()[1])
                self.dict_axes={buoy:[1,2] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.dict_axes[id_buoy][0],self.dict_axes[id_buoy][1]=self.plotting_one_stickplot(dict2[id_buoy][0],self.adding_variables,id_buoy,dict2[id_buoy][1],label,color)
                        plt.legend(handles=self.dict_axes[id_buoy][1],loc=3,ncol=2)

                        figs2[id_buoy].savefig(f'{self.plots_path}Fig8a_stickplot_{id_buoy}.svg',dpi=800,bbox_inches='tight',pad_inches=0.05)
