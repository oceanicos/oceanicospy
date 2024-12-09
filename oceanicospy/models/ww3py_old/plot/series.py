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
import pandas as pd
import matplotlib.dates as mdates
import math
import numpy as np
import datetime as dt
from matplotlib.lines import Line2D
import xarray as xr

class Series():
        def __init__(self,root_path,ini_date,end_date,vbles_to_plot,buoys_id,locs_buoys,hdcast) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/series/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date
                self.buoys_id = buoys_id
                self.locs_buoys = locs_buoys
                self.vbles_to_plot = vbles_to_plot
                self.hdcast=hdcast

        def preparing_data(self):

                time_ERA5_hcst,hs_hindcast_ERA5_buoys=util.read_hindcast('/home/fayalacruz/ww3.202005_hs.nc')
                time_CSFR_hcst,hs_hindcast_CSFR_buoys=util.read_hindcast('/home/fayalacruz/ww3.202005_hs_CSFR.nc')
  
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

                        # Hs data
                        if self.idate.year == 2020:
                                self.result,self.result_u,self.result_v=util.read_era5_buoys(f'{self.run_path}/{self.idate.strftime("%Y%m%d")}_era5.nc',self.lon,self.lat)
                                self.series_era5_buoys[buoy]=self.result
                                self.series_era5_buoys_u[buoy]=self.result_u
                                self.series_era5_buoys_v[buoy]=self.result_v

                                self.hs_buoy=self.data_buoy.hs[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)] # First day is cutted off due to spin-up
                                self.new_x_index=(self.hs_buoy.index-pd.Timedelta(minutes=40))
                        else:
                                self.hs_buoy=self.data_buoy.hs[self.idate:self.edate] 
                                self.new_x_index=self.hs_buoy.index

                        self.hs_model=self.data_ounp[buoy].hs[self.new_x_index]
                        self.hs_hindcast_ERA5=pd.Series(hs_hindcast_ERA5_buoys[buoy],index=time_ERA5_hcst)
                        self.hs_hindcast_CSFR=pd.Series(hs_hindcast_CSFR_buoys[buoy],index=time_CSFR_hcst)
                        
                        if self.hdcast==True:
                                self.hs[buoy] = dict(model=self.hs_model,buoy=self.hs_buoy,hindcast_ERA5=self.hs_hindcast_ERA5,
                                                     hindcast_CSFR=self.hs_hindcast_CSFR)   
                        else:
                                self.hs[buoy] = dict(model=self.hs_model,buoy=self.hs_buoy)   


                        # u10 data
                        self.wnd_spd_buoy=self.data_buoy.wspd[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]

                        if self.idate.year == 2020:
                                self.wnd_spd_era = self.series_era5_buoys[buoy][self.idate+relativedelta(hours=24):]
                                self.u10[buoy] = dict(ERA5=self.wnd_spd_era,buoy=self.wnd_spd_buoy)
                                # self.u10[buoy] = dict(buoy=self.wnd_spd_buoy)
                        else:
                                self.u10[buoy] = dict(buoy=self.wnd_spd_buoy)

                        # wind and wave dir data
                        self.wvdir_buoy=self.data_buoy.dir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wndir_buoy=self.data_buoy.wndir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wvdir_model=self.data_ounp[buoy].dirp[self.new_x_index]
                        self.wndir_era5=(270-np.degrees(np.arctan2(self.series_era5_buoys_v[buoy][self.idate+relativedelta(hours=24):],self.series_era5_buoys_u[buoy][self.idate+relativedelta(hours=24):])))%360
                        self.wndir_model=self.wind_params[buoy].wnddir

                        # self.dirs[buoy]=dict(model=self.wvdir_model,buoy=self.wvdir_buoy,wind_buoy=self.wndir_buoy,wind_era=self.wndir_era5)

                        self.dirs[buoy]=dict(model=self.wvdir_model,buoy=self.wvdir_buoy,wind_buoy=self.wndir_buoy)

                print(self.hs['42058']['buoy'].sort_values(ascending=False)[:20])
                return  self.hs,self.u10,self.dirs

        def plotting_wave_integral(self,axes,idx,param,dict_var,id_buoy,flag,label,color):
                try:
                        self.ax=axes[idx]
                except:
                        self.ax=axes
                
                self.colors_dict = dict(buoy='k',ERA5='c',wind_buoy='sandybrown',wind_era='salmon')
                self.dict_vars=dict_var[param][id_buoy]
                for key in self.dict_vars.keys():
                        if 'model' in self.dict_vars.keys():
                                idx_nans=np.where(self.dict_vars['buoy'].isnull()==True)[0]
                                if len(idx_nans)>0:
                                        dates_nans=self.dict_vars['buoy'].index[idx_nans]
                                        self.dict_vars['model']=self.dict_vars['model'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                        self.dict_vars['buoy']=self.dict_vars['buoy'].drop(index=dates_nans)

                                MSE = np.sqrt(((self.dict_vars['model'].values - self.dict_vars['buoy'].values)**2).mean()) 
                                RMSE = round(MSE,2)

                                bias = np.sqrt(((self.dict_vars['model'].values - np.mean(self.dict_vars['model'].values))**2).mean()) -\
                                        np.sqrt(((self.dict_vars['buoy'].values - np.mean(self.dict_vars['buoy'].values))**2).mean())
                                bias = round(bias,2)

                                MBE = np.mean(self.dict_vars['model'].values - self.dict_vars['buoy'].values)
                                MBE = round(MBE,2)

                        if flag == 'compare':
                                if label=='winp' or label=='cos2':
                                        label='$cos^2$'
                                elif label=='winp4' or label=='cos4':
                                        label='$cos^4$'
                                elif label=='bim2' or label=='bim':
                                        label='bim'
                                elif label =='all':
                                        label='$cos^4$+bim'
                                elif label=='expB':
                                        label='expA'
                                elif label=='expC':
                                        label='expB'
                                elif label=='expD':
                                        label='expC'

                                first_line=f'{label}: RMSE={RMSE}m'
                                second_line=f'$b$={MBE}m'
                                diff_charac=first_line.find('=')-second_line.find('=')
                                total_spaces=len(second_line)+diff_charac

                                # self.ax.plot(self.dict_vars['model'],c=color,label=f'{first_line}\n{second_line.rjust(total_spaces+2)}',lw=0.9)
                                self.ax.plot(self.dict_vars['model'],c=color,label=f'{first_line} - {second_line}',lw=0.9)

                                break
                        else:

                                if key =='model':
                                        first_line=f'{label} - RMSE= {RMSE}'
                                        # print(first_line)
                                        second_line=f'$b$= {MBE}'
                                        diff_charac=first_line.find('=')-second_line.find('=')
                                        total_spaces=len(second_line)+diff_charac


                                        if label == 'all':
                                                label='$cos^4$+bim'
                                        # self.ax.plot(self.dict_vars[key],c=color,label=f'{first_line}\n{second_line.rjust(total_spaces+2)}',lw=0.9)
                                        self.ax.plot(self.dict_vars[key],c=color,label=f'{first_line} - {second_line}',lw=0.9)

                                elif 'hindcast' in key:
                                        
                                        idx_nans=np.where(self.dict_vars['buoy'].isnull()==True)[0]
                                        if len(idx_nans)>0:
                                                dates_nans=self.dict_vars['buoy'].index[idx_nans]
                                                self.dict_vars['hindcast_ERA5']=self.dict_vars['hindcast_ERA5'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                                self.dict_vars['hindcast_CSFR']=self.dict_vars['hindcast_CSFR'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                                self.dict_vars['buoy']=self.dict_vars['buoy'].drop(index=dates_nans)
                                        
                                        self.index_hindcast_era5=(self.dict_vars['hindcast_ERA5'][self.idate+relativedelta(hours=24):].index)
                                        self.index_hindcast_csfr=(self.dict_vars['hindcast_CSFR'][self.idate+relativedelta(hours=24):].index)
                                        self.index_buoy=self.dict_vars['buoy'].index-pd.Timedelta(minutes=40)
                                        self.intersec_index_era5=self.index_hindcast_era5.intersection(self.index_buoy)
                                        self.intersec_index_csfr=self.index_hindcast_csfr.intersection(self.index_buoy)
                                        self.buoy_new_era5=self.dict_vars['buoy'][self.intersec_index_era5+pd.Timedelta(minutes=40)]
                                        self.buoy_new_csfr=self.dict_vars['buoy'][self.intersec_index_csfr+pd.Timedelta(minutes=40)]
                                        self.hindcast_new_era5=self.dict_vars['hindcast_ERA5'][self.intersec_index_era5]
                                        self.hindcast_new_csfr=self.dict_vars['hindcast_CSFR'][self.intersec_index_csfr]

                                        MSE_era5 = np.sqrt(((self.hindcast_new_era5.values - self.buoy_new_era5.values)**2).mean()) 
                                        RMSE_era5 = round(MSE_era5,2)
                                        bias_era5 = np.sqrt(((self.hindcast_new_era5.values - np.mean(self.hindcast_new_era5.values))**2).mean()) -\
                                                np.sqrt(((self.buoy_new_era5.values - np.mean(self.buoy_new_era5.values))**2).mean())
                                        bias_era5 = round(bias,2)
                                        MBE_era5 = np.mean(self.hindcast_new_era5.values - self.buoy_new_era5.values)
                                        MBE_era5 = round(MBE_era5,2)

                                        MSE_csfr = np.sqrt(((self.hindcast_new_csfr.values - self.buoy_new_csfr.values)**2).mean()) 
                                        RMSE_csfr = round(MSE_csfr,2)
                                        bias_csfr = np.sqrt(((self.hindcast_new_csfr.values - np.mean(self.hindcast_new_csfr.values))**2).mean()) -\
                                                np.sqrt(((self.buoy_new_csfr.values - np.mean(self.buoy_new_csfr.values))**2).mean())
                                        bias_csfr = round(bias,2)
                                        MBE_csfr = np.mean(self.hindcast_new_csfr.values - self.buoy_new_csfr.values)
                                        MBE_csfr = round(MBE_csfr,2)

                                        name_hdcast=key[len('hindcast_'):]

                                        if name_hdcast=='ERA5':
                                                first_line=f'hindcast ({name_hdcast}) - RMSE= {RMSE_era5}'
                                                second_line=f'$b$= {MBE_era5}'
                                                self.ax.plot(self.dict_vars[key],c='gray',label=f'{first_line} - {second_line}',lw=0.8)
                                        else:
                                                first_line=f'hindcast ({name_hdcast}) - RMSE= {RMSE_csfr}'
                                                second_line=f'$b$= {MBE_csfr}'
                                                self.ax.plot(self.dict_vars[key],c='olivedrab',label=f'{first_line} - {second_line}',lw=0.8)


                                else:
                                        self.ax.plot(self.dict_vars[key],c=self.colors_dict[key],label=key,markersize=3,lw=0.9)

                if param =='hs':
                        if label =='ctrl':
                                if len(self.vbles_to_plot)>1:
                                        self.ax.plot([0],[0],c='sandybrown',label='wind buoy')
                                        self.ax.plot([0],[0],c='c',label='ERA5')
                                self.ax.text(0.97, 0.87, "a)",fontsize=17,transform=self.ax.transAxes)
                                self.ax.text(0.02, 0.87, f'Buoy:{id_buoy}', fontsize=16,transform=self.ax.transAxes)


                        self.hs_max=max([max([max(i) for i in list(bouy.values())]) for bouy in list(dict_var[param].values())])
                        self.ax.set(ylabel="$H_{s}$" +" [m]",ylim=(0,self.hs_max+0.5),yticks=range(0,int(self.hs_max)+2))
                        self.ax.tick_params(which='minor', length=3, color='k')
                        self.ax.tick_params(which='major', length=5, color='k')
                        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.57),ncol=3, fancybox=True) # This is the original
                        # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),ncol=3, fancybox=True)

                elif param =='u10':
                        if label=='ctrl':
                                self.ax.text(0.97, 0.87, "b)",fontsize=17,transform=self.ax.transAxes)
                        self.ax.set(ylabel="Wind speed [m/s]")
                        self.ax.tick_params(which='minor', length=3, color='k')
                        self.ax.tick_params(which='major', length=5, color='k')
                elif param == 'dir':
                        if label=='ctrl':
                                self.ax.text(0.97, 0.87, "c)",fontsize=17,transform=self.ax.transAxes)

                        self.ax.set(ylabel="Direction [°]")
                        self.ax.tick_params(which='minor', length=3, color='k')
                        self.ax.tick_params(which='major', length=5, color='k')
                else:
                        self.ax.set(ylabel="Wave energy $[m^{2}]$")

                if self.idate.year == 2020:
                        self.myFmt = mdates.DateFormatter('%m-%d-%y')
                        self.ax.set_xlim(dt.datetime(2020,5,1,0),dt.datetime(2020,6,2))

                else:
                        self.myFmt = mdates.DateFormatter('%d-%b')
                        self.ax.set_xlim(dt.datetime(2004,9,14,0),dt.datetime(2004,9,16,7))
                        self.ax.set_yticks(np.arange(0,13,3))
                        self.ax.set_yticklabels(np.arange(0,13,3))
                        self.ax.set_xticks([dt.datetime(2004,9,14,0),dt.datetime(2004,9,15,0),dt.datetime(2004,9,16,0)])
                        
                self.fmt_day = mdates.DayLocator()        
                self.ax.xaxis.set_major_formatter(self.myFmt)
                self.ax.xaxis.set_minor_locator(self.fmt_day)
                self.ax.grid(True,linestyle='dotted')

                return self.ax

        def setting_up_plot(self,label,color):
                self.all_data_vbles=dict(hs=self.preparing_data()[0],u10=self.preparing_data()[1],dir=self.preparing_data()[2])
                self.dict_axes={buoy:list(range(len(self.vbles_to_plot))) for buoy in self.buoys_id}
                self.figs={buoy:list(range(len(self.vbles_to_plot))) for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.rows=int(len(self.vbles_to_plot))
                        if self.rows>=1:
                                self.fig,self.axes=plt.subplots(self.rows,1,figsize=(13,self.rows*3),sharex=True)
                                # self.fig,self.axes=plt.subplots(self.rows,1,figsize=(5,self.rows*2.5))
                                if self.rows==1:
                                        # plt.suptitle(f'Time series for buoy {id_buoy}',y=1.30,fontsize=15)
                                        a=3
                                elif self.rows ==2:
                                        # plt.suptitle(f'Time series for buoy {id_buoy}',y=1.07,fontsize=15)                                        
                                        a=3
                                else:
                                        # plt.suptitle(f'Time series for buoy {id_buoy}',y=1.01,fontsize=15)                                        
                                        a=3
                        else:
                                raise ValueError('The number of subplots is not expected')
                        
                        # Loop over each variable
                        for idx,el in enumerate(self.vbles_to_plot):  
                                self.dict_axes[id_buoy][idx]=self.plotting_wave_integral(self.axes,idx,el,self.all_data_vbles,id_buoy,'anything',label,color)
                        self.figs[id_buoy]=self.fig
                        plt.subplots_adjust(hspace=0.07)
                        self.fig.savefig(f'{self.plots_path}srs_{id_buoy}.png',dpi=400,bbox_inches='tight',pad_inches=0.05)
                return self.dict_axes,self.figs
        
        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.adding_variables=dict(hs=obj2.preparing_data()[0],dir=obj2.preparing_data()[2])
                self.dict_axes={buoy:[1,2,3] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        # Loop over each variable
                        for idx,el in enumerate(self.vbles_to_plot):
                                if el != 'u10':
                                        self.dict_axes[id_buoy][idx]=self.plotting_wave_integral(dict2[id_buoy],idx,el,self.adding_variables,id_buoy,'compare',label,color)
                        
                        figs2[id_buoy].savefig(f'{self.plots_path}srs_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
