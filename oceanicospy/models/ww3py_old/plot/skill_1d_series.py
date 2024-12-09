import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import util
from . import init_custom

import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.dates as mdates
import math
import numpy as np
import os
import matplotlib.colors as colors
import datetime as dt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

class Skill_1d_series():
        def __init__(self,root_path,ini_date,end_date,metrics_1d) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/spectra_1d/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date                
                self.buoys_id = ['42058']
                # self.buoys_id = ['42057','42058','42059','42060']
                self.metrics_1d = metrics_1d
                self.type_lines=dict(RMSE='-',NRMSE='-',MBAE='-',corr='-')


        def preparing_data(self):
                # Reading model results
                self.time_ww3,self.freqs_ww3,self.spec_1d_model = util.read_data_spec_1d_stations(f'{self.data_path}ww3.2020_spec_1d.nc')

                self.dics_data={}

                for id in self.buoys_id:    
                        # Spectra from the buoy
                        self.time_buoy,self.freqs_buoy,self.spec_1d_buoy=util.read_1d_spec_buoy(id)  # reading buoy results
                        self.idx_dates_buoy=np.where((self.time_buoy<=self.edate+relativedelta(hours=1)) & (self.time_buoy>=self.idate),True,False)
                        self.time_buoy=self.time_buoy[self.idx_dates_buoy]
                        self.spec_1d_buoy=self.spec_1d_buoy[self.idx_dates_buoy,3:]

                        # Spectral from model
                        self.new_x_index=(self.time_buoy+pd.Timedelta(minutes=20))
                        self.idx_dates_model=np.array(list(map(lambda x: True if x in self.new_x_index else False,self.time_ww3)))
                        self.spec_1d_model_id=self.spec_1d_model[id]
                        self.spec_1d_model_id=self.spec_1d_model_id[self.idx_dates_model,:]

                        # Smoothing 1D spectra
                        self.spec_1d_buoy=np.array(list(map(util.moving_average_filter,self.spec_1d_buoy)))

                        # Interpolating 1D spectra
                        self.spec_1d_model_id=np.array(list(map(util.interp_1d_spectra,np.tile(self.freqs_ww3,(self.spec_1d_model_id.shape[0],1)),\
                                                self.spec_1d_model_id,np.tile(self.freqs_buoy[3:],(self.spec_1d_model_id.shape[0],1)))))

                        # Computing metrics
                        metrics=np.array(list(map(util.metrics,self.spec_1d_buoy,self.spec_1d_model_id)))

                        self.dics_data[id]=dict(RMSE=pd.Series(metrics[:,0],index=self.new_x_index[:-1]),
                                                NRMSE=pd.Series(metrics[:,1],index=self.new_x_index[:-1]),
                                                MBAE=pd.Series(metrics[:,2],index=self.new_x_index[:-1]))
                
                return self.dics_data

        def plotting_one_skill_serie(self,axes,idx,key,dict_var,id_buoy,label,compare):
                try:
                        self.ax=axes[idx]
                except:
                        self.ax=axes

                self.myFmt = mdates.DateFormatter('%m-%d-%y')
                self.fmt_day = mdates.DayLocator()

                self.dict_to_plot=dict_var[id_buoy]
                self.ax.plot(self.dict_to_plot[key],color='k',lw=1,ls=self.type_lines[key])

                # if label=='cos2':
                #         label='cos^{2}'
                # elif label=='cos4':
                #         label='cos^{4}'
                # elif label=='bim2':
                #         label='bim'
                # elif label =='all':
                #         label='cos^{4}+bim'

                if compare == True:
                        self.ax.text(0.97, 0.92, '(b)', transform=axes.transAxes, size=12)
                        if key=='corr':   
                                self.ax.fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values>=0, interpolate=True, color='darkcyan')
                                self.ax.fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values<=0, interpolate=True, color='tomato')
                                self.ax.set(ylabel=f'{key} diff')

                        else:
                                self.ax.fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values>=0, interpolate=True, color='tomato')
                                self.ax.fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values<=0, interpolate=True, color='darkcyan')
                                string1=f'{key}_{{ctrl}}'
                                string2=f'{key}_{{{label}}}'
                                if key == 'RMSE':
                                        self.ax.set(ylabel=fr"$\dfrac{{{string1}-{string2}}}{{{string1}}}$")  
                                else:
                                        self.ax.set(ylabel=fr"${string1}-{string2}$")  


                else:
                        self.ax.set(ylabel=f'{key} [$m^{2}s$]')

                self.ax.grid(True,alpha=0.5)
                self.ax.axhline(y=0, color="black", linestyle="--")
                self.ax.yaxis.set_minor_locator(MultipleLocator(0.2))
                self.ax.yaxis.set_major_locator(MultipleLocator(1))
                self.ax.xaxis.set_major_formatter(self.myFmt)
                self.ax.xaxis.set_minor_locator(self.fmt_day)
                self.ax.set_xlim(dt.datetime(2020,5,1,0),dt.datetime(2020,6,2))

                return self.ax

        def setting_up_plot(self,label,color):
                self.dict=self.preparing_data()

                self.dict_axes={buoy:list(range(len(self.metrics_1d))) for buoy in self.buoys_id}
                self.figs={}

                for id_buoy in self.buoys_id:
                        self.fig,self.ax=plt.subplots(len(self.metrics_1d),1,figsize=(11,int(len(self.metrics_1d)*3)))

                        # Loop over each type of result (buoy or simulations)
                        for idx,key in enumerate(self.metrics_1d):
                                self.dict_axes[id_buoy][idx]=self.plotting_one_skill_serie(self.ax,idx,key,self.dict,
                                                                                       id_buoy,label,False)
                                

                        # self.fig.suptitle(f'Metrics for buoy {id_buoy} [{label}]',y=0.9)
                        self.figs[id_buoy]=self.fig
                        
                        self.fig.savefig(f'{self.plots_path}Fig8_skill_ef_1d_{id_buoy}.svg',dpi=800,bbox_inches='tight',pad_inches=0.05)

        def compare_another_conf(self,obj2,label,color):
                if label=='cos2':
                        label='cos^{2}'
                elif label=='cos4':
                        label='cos^{4}'
                elif label=='bim2':
                        label='bim'
                elif label =='all':
                        label='\cos^{4}+\mathrm{bim}'

                self.dict_var1=self.preparing_data()
                self.dict_var2=obj2.preparing_data()

                self.dict_axes={buoy:list(range(len(self.metrics_1d))) for buoy in self.buoys_id}

                for id_buoy in self.buoys_id:
                        self.fig,self.ax=plt.subplots(len(self.metrics_1d),1,figsize=(11,int(len(self.metrics_1d)*3)),
                                                        sharex=True)

                        # Loop over each type of result (buoy or simulations)
                        
                        for idx,key in enumerate(self.metrics_1d):
                                if key =='RMSE':
                                        self.dict_var2[id_buoy][key]=(self.dict_var1[id_buoy][key]-self.dict_var2[id_buoy][key])/self.dict_var1[id_buoy][key]
                                        # print(self.dict_var2[id_buoy][key].nsmallest(3).index)
                                else:
                                        self.dict_var2[id_buoy][key]=self.dict_var1[id_buoy][key]-self.dict_var2[id_buoy][key]
                                self.dict_axes[id_buoy][idx]=self.plotting_one_skill_serie(self.ax,idx,key,self.dict_var2,
                                                                                 id_buoy,label,True)  
                        # self.fig.suptitle(f'Difference of RMSE [ctrl - ${label}$] from spectra 1D - buoy {id_buoy}',y=0.98)        
                        # plt.subplots_adjust(wspace=0.4,hspace=0.6)                        

                        self.fig.savefig(f'{obj2.plots_path}Fig8_skill_ef_1d_{id_buoy}.svg',dpi=800,bbox_inches='tight',pad_inches=0.05)

