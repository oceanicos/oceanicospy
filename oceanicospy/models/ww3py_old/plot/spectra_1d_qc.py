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
import os
from scipy import interpolate
import matplotlib.colors as color 
import datetime as dt
from scipy.integrate import simpson
from numpy import trapz
from scipy.stats import pearsonr
import functools
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

def allocate_metric_numbers(ax,x,bias,RMSE,corr,si,color):
        ax.text(x,0.92, f'{bias}',transform=ax.transAxes,fontsize=11,color=color)
        ax.text(x+0.1,0.85, f'{RMSE}',transform=ax.transAxes,fontsize=11,color=color)
        ax.text(x,0.78, f'{corr}',transform=ax.transAxes,fontsize=11,color=color)
        ax.text(x,0.71, f'{si}',transform=ax.transAxes,fontsize=11,color=color)


class Spectra_1d_qc():
        def __init__(self,root_path,ini_date,buoys_id) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/spectra_1d/'
                os.system(f'mkdir -p {self.plots_path}')
                self.buoys_id = buoys_id
                self.metrics_dic={buoy:{} for buoy in self.buoys_id}

        def compute_narrownes(self,date,buoy_id):

                try:
                        # Spectra from the buoy
                        self.date_buoy=date-pd.Timedelta(minutes=20)
                        self.time_buoy,self.freqs_buoy,self.spec_1d_buoy=util.read_1d_spec_buoy(buoy_id)  # reading buoy results
                        self.idx_inidate_buoy=self.time_buoy.get_loc(self.date_buoy)
                        self.spec_to_plot_buoy=self.spec_1d_buoy[self.idx_inidate_buoy,:]
                        self.spec_to_plot_buoy=util.moving_average_filter(self.spec_to_plot_buoy,3)

                        # Spectra from the model (ww3)
                        self.idx_dateini_ww3=self.time_ww3.get_loc(date)       
                        self.spec_to_plot_model=self.spec_1d_model[buoy_id]
                        self.spec_to_plot_model=self.spec_to_plot_model[self.idx_dateini_ww3,:]

                        self.freqs_buoy_splitted=self.freqs_buoy[3:]
                        m0=trapz(self.spec_to_plot_buoy[3:],x=self.freqs_buoy_splitted)
                        efn=self.spec_to_plot_buoy/m0
                        tmminus10=trapz(self.spec_to_plot_buoy[3:]*(self.freqs_buoy_splitted**-1),x=self.freqs_buoy_splitted)/m0
                        Qc_buoy=np.nanmax(efn)/tmminus10

                        # Computing spectral narrowness Qc model
                        self.spec_interp=util.interp_1d_spectra(np.tile(self.freqs_ww3,1),self.spec_to_plot_model,np.tile(self.freqs_buoy_splitted,1))
                        m0_model=trapz(self.spec_interp,x=self.freqs_buoy_splitted)
                        efn_model=self.spec_interp/m0_model
                        tmminus10_model=trapz(self.spec_interp*(self.freqs_buoy_splitted**-1),x=self.freqs_buoy_splitted)/m0_model
                        Qc_model=np.nanmax(efn_model)/tmminus10_model
                except:
                        Qc_buoy=np.NaN
                        Qc_model=np.NaN

                return Qc_buoy,Qc_model

        def preparing_data(self):
                # Reading model results
                self.time_ww3,self.freqs_ww3,self.spec_1d_model = util.read_data_spec_1d_stations(f'{self.data_path}ww3.2020_spec_1d.nc')

                self.dics_data={}
                for buoy in self.buoys_id:
                        results=np.array(list(map(functools.partial(self.compute_narrownes,buoy_id=buoy),self.time_ww3)))
                        qcs_buoy=results[:,0]
                        qcs_buoy = qcs_buoy[~np.isnan(qcs_buoy)]
                        qcs_model=results[:,1]
                        qcs_model = qcs_model[~np.isnan(qcs_model)]

                        self.dics_data[buoy]=dict(buoy=qcs_buoy,model=qcs_model)
                return self.dics_data


        def plotting_one_scatter(self,axes,dict_var,id_buoy,label,color):
                self.dict_vars=dict_var[id_buoy]
                if 'model' in self.dict_vars.keys():
                        bias,RMSE,corr,si=metrics(self.dict_vars['buoy'],self.dict_vars['model'])
                        self.metrics_dic[id_buoy][label]=dict(b=bias,rmse=RMSE,rho=corr,SI=si)

                if id_buoy=='42058':
                        self.x_add=0.5
                        self.correction=0.1
                        self.y_add=1.4
                else:
                        self.x_add=0.3
                        self.correction=0.1
                        self.y_add=0
                for i in range (0,4):
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

                        elif label=='cos2' and i==0:
                                label='$cos^2$'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)
                                self.ax.set(ylabel='Simulated $Q_c$')

                        elif label=='cos4' and i==1:
                                label='$cos^4$'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)
                                self.ax.set(ylabel='Simulated $Q_c$',xlabel='Observed $Q_c$')

                        elif label=='bim' and i==2:
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)

                        elif label=='all' and i==3:
                                label='$cos^4$+bim'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                allocate_metric_numbers(self.ax,0.24,bias,RMSE,corr,si,color)
                                self.ax.set(xlabel='Observed $Q_c$')
                       
                        self.ax.legend(loc='lower right')

                        if id_buoy=='42058':
                                self.ax.set(xlim=(0,4.5),ylim=(0,4.5))
                        else:
                                self.ax.set(xlim=(0,3),ylim=(0,3))

                        self.ax.plot(np.arange(-0.5,4.3,0.1),np.arange(-0.5,4.3,0.1),'--k')
                        self.ax.set_yticks([0,1,2,3,4])
                return axes
        
        def setting_up_plot(self,label,color):
                all_data_vbles=self.preparing_data()
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                self.figs={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        cm = 1/2.54  # centimeters in inches
                        self.fig,self.axes=plt.subplots(2,2,figsize=(7,7),constrained_layout=True,sharex=True,sharey=True)
                        
        #                 # Loop over each variable
                        self.dict_axes[id_buoy]=self.plotting_one_scatter(self.axes,all_data_vbles,id_buoy,label,color)
                        # self.fig.suptitle(f'Comparison of simulated and observed $Q_c$ between ctrl experiment and other simulations \n Buoy {id_buoy}')

                        self.figs[id_buoy]=self.fig
                        
                        self.fig.savefig(f'{self.plots_path}Fig7_scatter_Qc_{id_buoy}.pdf',dpi=700,bbox_inches='tight',pad_inches=0.05)
                return self.dict_axes,self.figs

        
        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.adding_variables=obj2.preparing_data()
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.dict_axes[id_buoy]=self.plotting_one_scatter(dict2[id_buoy],self.adding_variables,id_buoy,label,color)
                        
                        figs2[id_buoy].savefig(f'{self.plots_path}Fig7_scatter_Qc_{id_buoy}.pdf',dpi=700,bbox_inches='tight',pad_inches=0.05)