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
from scipy.stats import pearsonr
import string


def metrics(measures,model):
   
    bias = np.mean(measures - model)
    bias = round(bias,2)

    MSE = np.sqrt(((measures - model)**2).mean()) 
    RMSE = round(MSE,2)

    corr, _ = pearsonr(measures,model)
    corr=round(corr,2)

    si = np.sqrt(np.mean(((model - np.mean(model)) - (measures-np.mean(measures)))**2))/(np.mean(measures))
    si = round(si,2)

    return bias,RMSE,corr,si


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def allocate_metric_numbers(ax,x,bias,RMSE,corr,si,color):
        ax.text(x,0.92, f'{bias}',transform=ax.transAxes,fontsize=11,color=color)
        ax.text(x+0.1,0.85, f'{RMSE}',transform=ax.transAxes,fontsize=11,color=color)
        ax.text(x,0.78, f'{corr}',transform=ax.transAxes,fontsize=11,color=color)
        ax.text(x,0.71, f'{si}',transform=ax.transAxes,fontsize=11,color=color)

class QQplot_hs():
        def __init__(self,root_path,ini_date,end_date,vbles_to_plot,buoys_id,locs_buoys) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/qqplots/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date
                self.buoys_id = buoys_id
                self.locs_buoys = locs_buoys
                self.vbles_to_plot = vbles_to_plot
                self.metrics_dic={buoy:{} for buoy in self.buoys_id}

        def preparing_data(self):
                time_ERA5_hcst,hs_hindcast_ERA5_buoys=util.read_hindcast('/home/fayalacruz/ww3.202005_hs.nc')
                time_CFSR_hcst,hs_hindcast_CFSR_buoys=util.read_hindcast('/home/fayalacruz/ww3.202005_hs_CSFR.nc')
  
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
                        self.hs_hindcast_CFSR=pd.Series(hs_hindcast_CFSR_buoys[buoy],index=time_CFSR_hcst)
                        
                        self.hs[buoy] = dict(model=self.hs_model,buoy=self.hs_buoy,hindcast_ERA5=self.hs_hindcast_ERA5,
                                                hindcast_CFSR=self.hs_hindcast_CFSR)   

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

                return  self.hs,self.u10,self.dirs

        def setting_up_plot(self,label,color):

                all_data_vbles=self.preparing_data()[0]
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                self.figs={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        cm = 1/2.54  # centimeters in inches
                        self.fig,self.axes=plt.subplots(2,3,figsize=(10,7),constrained_layout=True,sharex=True,sharey=True)
                        # self.fig,self.axes=plt.subplots(2,3,figsize=(15*cm,11*cm),constrained_layout=True,sharex=True,sharey=True)
                        
        #                 # Loop over each variable
                        self.dict_axes[id_buoy]=self.plotting_one_scatter(self.axes,all_data_vbles,id_buoy,label,color)
                        self.figs[id_buoy]=self.fig
                        
                        self.fig.savefig(f'{self.plots_path}Fig4_scatter_hs_{id_buoy}.pdf',dpi=700,bbox_inches='tight',pad_inches=0.05)
                return self.dict_axes,self.figs
        
        def plotting_one_scatter(self,axes,dict_var,id_buoy,label,color):
                self.dict_vars=dict_var[id_buoy]
                if 'model' in self.dict_vars.keys():
                        idx_nans=np.where(self.dict_vars['buoy'].isnull()==True)[0]
                        if len(idx_nans)>0:
                                dates_nans=self.dict_vars['buoy'].index[idx_nans]
                                self.dict_vars['model']=self.dict_vars['model'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                self.dict_vars['buoy']=self.dict_vars['buoy'].drop(index=dates_nans)

                        bias,RMSE,corr,si=metrics(self.dict_vars['buoy'].values,self.dict_vars['model'].values)
                        self.metrics_dic[id_buoy][label]=dict(b=bias,rmse=RMSE,rho=corr,SI=si)

                if 'hindcast_ERA5' in self.dict_vars.keys():
                                        
                        idx_nans=np.where(self.dict_vars['buoy'].isnull()==True)[0]
                        if len(idx_nans)>0:
                                dates_nans=self.dict_vars['buoy'].index[idx_nans]
                                self.dict_vars['hindcast_ERA5']=self.dict_vars['hindcast_ERA5'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                self.dict_vars['hindcast_CFSR']=self.dict_vars['hindcast_CFSR'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                self.dict_vars['buoy']=self.dict_vars['buoy'].drop(index=dates_nans)
                        
                        self.index_hindcast_era5=(self.dict_vars['hindcast_ERA5'][self.idate+relativedelta(hours=24):].index)
                        self.index_hindcast_CFSR=(self.dict_vars['hindcast_CFSR'][self.idate+relativedelta(hours=24):].index)
                        self.index_buoy=self.dict_vars['buoy'].index-pd.Timedelta(minutes=40)
                        self.intersec_index_era5=self.index_hindcast_era5.intersection(self.index_buoy)
                        self.intersec_index_CFSR=self.index_hindcast_CFSR.intersection(self.index_buoy)
                        self.buoy_new_era5=self.dict_vars['buoy'][self.intersec_index_era5+pd.Timedelta(minutes=40)]
                        self.buoy_new_CFSR=self.dict_vars['buoy'][self.intersec_index_CFSR+pd.Timedelta(minutes=40)]
                        self.hindcast_new_era5=self.dict_vars['hindcast_ERA5'][self.intersec_index_era5]
                        self.hindcast_new_CFSR=self.dict_vars['hindcast_CFSR'][self.intersec_index_CFSR]

                        bias_era5,RMSE_era5,corr_era5,si_era5=metrics(self.buoy_new_era5.values,self.hindcast_new_era5.values)
                        bias_CFSR,RMSE_CFSR,corr_CFSR,si_CFSR=metrics(self.buoy_new_CFSR.values,self.hindcast_new_CFSR.values)
                        self.metrics_dic[id_buoy]['hindcast ERA5']=dict(b=bias_era5,rmse=RMSE_era5,rho=corr_era5,SI=si_era5)
                        self.metrics_dic[id_buoy]['hindcast CFSR']=dict(b=bias_CFSR,rmse=RMSE_CFSR,rho=corr_CFSR,SI=si_CFSR)

                if id_buoy=='42058':
                        self.x_add=0.5
                        self.correction=0.1
                        self.y_add=1.4
                else:
                        self.x_add=0.3
                        self.correction=0.1
                        self.y_add=0

                for i in range (0,6):
                        self.ax=axes[i%2][i//2]

                        if label=='ctrl':
                                self.ax.text(0.928, 0.935, f'({string.ascii_lowercase[i]})', transform=self.ax.transAxes, size=12)

                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,color=color,label=label,alpha=0.5)
                                self.ax.text(0.02, 0.92, '$b$ =',transform=self.ax.transAxes,fontsize=11)
                                self.ax.text(0.02, 0.85, 'RMSE =',transform=self.ax.transAxes,fontsize=11)
                                self.ax.text(0.02, 0.78, '$\\rho$ =',transform=self.ax.transAxes,fontsize=11)
                                self.ax.text(0.02, 0.71, '$SI$ =',transform=self.ax.transAxes,fontsize=11)
                                allocate_metric_numbers(self.ax,0.12,bias,RMSE,corr,si,color)

                        if label=='ctrl' and i==0:
                                self.ax.scatter(self.buoy_new_era5.values,self.hindcast_new_era5.values,
                                        s=35,color='darkcyan',label='Hindcast ERA5',alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias_era5,RMSE_era5,corr_era5,si_era5,'darkcyan')
                                self.ax.set(ylabel='Simulated $H_s$ [m]')

                        elif label=='ctrl' and i==1:
                                self.ax.scatter(self.buoy_new_CFSR.values,self.hindcast_new_CFSR.values,
                                        s=35,color='mediumpurple',label='Hindcast CFSR',alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias_CFSR,RMSE_CFSR,corr_CFSR,si_CFSR,'mediumpurple')
                                self.ax.set(ylabel='Simulated $H_s$ [m]',xlabel='Observed $H_s$ [m]')

                        elif label=='cos2' and i==2:
                                label='$cos^2$'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)
                        elif label=='cos4' and i==3:
                                label='$cos^4$'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)
                                self.ax.set(xlabel='Observed $H_s$ [m]')

                        elif label=='bim' and i==4:
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)

                        elif label=='all' and i==5:
                                label='$cos^4$+bim'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)
                                self.ax.set(xlabel='Observed $H_s$ [m]')
                      
                        self.ax.legend(loc='lower right')

                        if id_buoy=='42058':
                                self.ax.set(xlim=(0,4.5),ylim=(0,4.5))
                        else:
                                self.ax.set(xlim=(0,3),ylim=(0,3))

                        self.ax.plot(np.arange(-0.5,4.3,0.1),np.arange(-0.5,4.3,0.1),'--k')
                        self.ax.set_yticks([0,1,2,3,4])
                return axes

        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.adding_variables=obj2.preparing_data()[0]
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.dict_axes[id_buoy]=self.plotting_one_scatter(dict2[id_buoy],self.adding_variables,id_buoy,label,color)
                        
                        figs2[id_buoy].savefig(f'{self.plots_path}Fig4_scatter_hs_{id_buoy}.pdf',dpi=700,bbox_inches='tight',pad_inches=0.05)