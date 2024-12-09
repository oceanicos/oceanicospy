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
import matplotlib.colors as colors
import scipy.io
import datetime as dt


class Spectra():
        def __init__(self,root_path,ini_date,date) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/spectra_2d/'
                #if self.plots_path  # Checking folder
                os.system(f'mkdir -p {self.plots_path}')
                self.date=date
                self.buoys_id = ['42058']
                self.efth_min=1e-3 # Minimum value for wave energy
                self.dics_complete=np.arange(0,370,10)

        def preparing_data_SRA(self):
                first_spectra=scipy.io.loadmat('/home/fayalacruz/runs/ivan2004_def/info/forc/IntSRA.mat')['EFTH'][1040,:,:]
                second_spectra=scipy.io.loadmat('/home/fayalacruz/runs/ivan2004_def/info/forc/IntSRA.mat')['EFTH'][678,:,:]
                dirs=scipy.io.loadmat('/home/fayalacruz/runs/ivan2004_def/info/forc/IntSRA.mat')['thetas'][0]
                dirs=np.concatenate((dirs,np.array([360])))
                freqs=scipy.io.loadmat('/home/fayalacruz/runs/ivan2004_def/info/forc/IntSRA.mat')['freqs'][0]
                r,theta = np.meshgrid(freqs,np.radians(dirs))
                return r,theta,first_spectra,second_spectra

        def preparing_data(self):
                # Reading model results
                self.time_ww3,self.freqs_ww3,self.dics_ww3,self.spec_2d_model = util.read_data_spec_stations(f'{self.data_path}ww3.2020_spec_2d.nc')

                self.wave_params=util.read_data_int_stations(f'{self.data_path}ww3.2020_tab_params.nc')
                self.wind_params=util.read_data_extra_stations(f'{self.data_path}ww3.2020_extra_params.nc')

                self.dics_data={}
                self.r_data={}
                self.theta_data={}
                self.dirs={}

                for id in self.buoys_id:    

                        # Spectra from the buoy
                        self.date_buoy=self.date-pd.Timedelta(minutes=20)
                        self.time_buoy,self.freqs_buoy,self.dics_buoy,self.spec_2d_buoy=util.read_2d_spec_buoy(id)  # reading buoy results
                        self.r,self.theta = np.meshgrid(self.freqs_buoy,np.radians(self.dics_complete))
                        self.idx_inidate_buoy=self.time_buoy.get_loc(self.date_buoy)

                        self.spec_to_plot_buoy=self.spec_2d_buoy[self.idx_inidate_buoy,:,:]
                        self.spec_to_plot_buoy=np.concatenate((self.spec_to_plot_buoy[:,:],self.spec_to_plot_buoy[:,0].reshape(-1,1)),axis=1)
                        self.spec_to_plot_buoy=np.where(self.spec_to_plot_buoy >= self.efth_min, self.spec_to_plot_buoy,self.efth_min)

                        # Spectra from the model (ww3)
                        self.idx_dateini_ww3=self.time_ww3.get_loc(self.date)
                        self.dics_complete_ww3=np.round((np.degrees(self.dics_ww3)+180)%360,1)
                        self.dic_complete_ww3_sorted=np.concatenate((self.dics_complete_ww3[::-1][8:],self.dics_complete_ww3[::-1][:8],np.array([360]))) # 0-360
                        self.r_ww3,self.theta_ww3 = np.meshgrid (self.freqs_ww3,np.radians(self.dic_complete_ww3_sorted))
        
                        self.spec_to_plot_model=self.spec_2d_model[id]
                        self.spec_to_plot_model=self.spec_to_plot_model[self.idx_dateini_ww3,:,:]
                        self.spec_to_plot_model=np.concatenate((self.spec_to_plot_model[:,::-1][:,8:],self.spec_to_plot_model[:,::-1][:,:8]),axis=1) #0-360
                        
                        self.spec_to_plot_model=np.concatenate((self.spec_to_plot_model[:,:],self.spec_to_plot_model[:,0].reshape(-1,1)),axis=1) 

                        # Storaging spectra, freqs and dirs arrays in dictioneries
                        self.dics_data[id]=dict(buoy=self.spec_to_plot_buoy,model=self.spec_to_plot_model)
                        self.r_data[id]=dict(buoy=self.r,model=self.r_ww3)
                        self.theta_data[id]=dict(buoy=self.theta,model=self.theta_ww3)
                        
                        # Storaging wind direction and wave direction
                        self.data_buoy = util.ord_buoy_data(id)
                        self.wndir_model=self.wind_params[id].wnddir[self.date]
                        self.wvdir_model=self.wave_params[id].dirp[self.date]
                        self.wndir_buoy=self.data_buoy.wndir[self.date_buoy]
                        self.wvdir_buoy=self.data_buoy.dir[self.date_buoy]
                        self.dirs[id]=dict(wind_model=self.wndir_model,wave_model=self.wvdir_model,wind_buoy=self.wndir_buoy,wave_buoy=self.wvdir_buoy)

                return self.r_data,self.theta_data,self.dics_data,self.dirs
        
        def interp_model_data(self,rs,thetas,dicts):

                for id_buoy in self.buoys_id:
                        # Sorting the freq and dirs arrays to create the interpolation function
                        self.sorted_dics_model=thetas[id_buoy]['model'][:,0]
                        self.sorted_freqs_model=rs[id_buoy]['model'][0,:]
                        self.sorted_energy = dicts[id_buoy]['model']

                        self.interp_function = interpolate.interp2d(self.sorted_dics_model[:-1], self.sorted_freqs_model, self.sorted_energy[:,:-1], kind='cubic')  

                        # Creating the new freq and dirs arrays to interpolate
                        self.new_freqs=rs[id_buoy]['buoy'][0,3:]
                        self.new_dics=self.sorted_dics_model
                        self.new_r,self.new_theta=np.meshgrid(self.new_freqs,self.new_dics)
                        self.interp_energy=self.interp_function(self.new_dics[:-1],self.new_freqs)
                        
                        # Re-storaging the interpolated arrays in the same dictionary.
                        dicts[id_buoy]['model_inter']=self.interp_energy           
                        rs[id_buoy]['model_inter']=self.new_r
                        thetas[id_buoy]['model_inter']=self.new_theta

                return rs,thetas,dicts

        def plotting_one_spectra(self,axes,idx,r,theta,dict_var,dic_dirs,id_buoy,type,levels_ctf,label,normalized,date):

                self.ax=axes[idx]
                self.spec=dict_var[id_buoy][type]
                
                # levels_plots = [0.00,0.008,0.012,0.017,0.026,0.039,0.059,0.088,0.132,0.198,0.296,0.444,0.667,1.000]
                levels_plots = [0.005,0.008,0.012,0.017,0.026,0.039,0.059,0.088,0.132,0.198,0.296,0.444,0.667,1.000]
                cmap = plt.cm.RdYlBu_r
                self.colors = cmap(np.linspace(0, cmap.N, len(levels_plots)).astype('i'))                

                if normalized == True:
                        if type == 'difference':
                                # print(label,id_buoy,np.nanmin(self.spec),np.nanmax(self.spec))
                                self.cf=self.ax.contourf(theta,r,self.spec.T,levels=levels_ctf,cmap='RdBu_r',zorder=1)
                                if date==dt.datetime(2020,5,10,13,0,0):
                                        self.cnt=self.ax.contour(theta,r,self.spec.T,levels=[0.2],linewidths=0.5,colors='k',
                                                                 zorder=10)
                                else:
                                        self.cnt=self.ax.contour(theta,r,self.spec.T,levels=[0.1],linewidths=0.5,colors='k',
                                                                 zorder=10)


                        else:                        
                                if type=='buoy':
                                        pass
                                        # print(np.nanmin(self.spec.T))
                                self.cf=self.ax.contourf(theta,r,self.spec.T,levels=levels_plots,colors=self.colors)
                else:
                        self.cf=self.ax.contourf(theta,r,self.spec.T,levels=levels_ctf,cmap='magma_r')

                plt.setp(self.ax,theta_direction=(-1),theta_zero_location=('N'))
                self.ax.set(yticks=[0.1,0.2,0.3,0.4,0.5])
                self.ax.set_yticklabels([0.1,0.2,0.3,0.4,0.5],fontsize=8)
                self.ax.set_rlabel_position(22) 
                self.ax.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'],fontsize=8)
                self.ax.tick_params(axis='x',pad=0.1) 

                if type=='buoy':
                        self.ax.annotate(None, xy=((dic_dirs[id_buoy]['wind_buoy']/360)*(2*np.pi), 0.115),
                                xytext=(0, 0),arrowprops=dict(arrowstyle="-|>",color='k'))

                        self.ax.annotate(None, xy=((self.dirs[id_buoy]['wave_buoy']/360)*(2*np.pi), 0.115), 
                                xytext=(0, 0),arrowprops=dict(arrowstyle="-|>",color='red'))
                elif type=='model':
                        self.ax.annotate(None, xy=((dic_dirs[id_buoy]['wind_model']/360)*(2*np.pi), 0.115),
                                xytext=(0, 0),arrowprops=dict(arrowstyle="-|>",color='k'))

                        self.ax.annotate(None, xy=((self.dirs[id_buoy]['wave_model']/360)*(2*np.pi), 0.115), 
                                xytext=(0, 0),arrowprops=dict(arrowstyle="-|>",color='red'))
                else:
                        pass

                if type =='model':
                        if label=='winp':
                                label='$cos^2$'
                        elif label=='bim2':
                                label='bim'
                        elif label =='all':
                                label='$cos^4$+bim'
                        self.ax.set_title(label,fontsize=10)
                elif type=='difference':
                        if label =='all':
                                label='$cos^4$+bim'
                        self.ax.set_title(f'difference [buoy - {label}]',fontsize=10)
                else:
                        self.ax.set_title(type,fontsize=10)

                self.ax.grid(True,alpha=0.5)
                self.ax.set_rlim(0,0.3)

                return self.cf,self.ax

        def setting_up_plot(self,label):
                self.r_to_plot,self.theta_to_plot,self.dict_to_plot,self.dirs_to_plot=self.preparing_data()
                self.dict_axes={buoy:[1,2] for buoy in self.buoys_id}
                self.figs={}

                for id_buoy in self.buoys_id:
                        self.fig,self.axes=plt.subplots(1,2,subplot_kw=dict(projection='polar'))
                        plt.suptitle(f'Directional spectra - buoy {id_buoy} - {self.date}',y=0.91)

                        # Common features
                        self.cbar_ticks,self.levels_plots=util.customizing_colorbar(self.dict_to_plot[id_buoy])

                        # Loop over each type of result (buoy or simulations)
                        for idx,key in enumerate(list(self.dict_to_plot[id_buoy].keys())):  
                                self.cf_last,self.dict_axes[id_buoy][idx]=self.plotting_one_spectra(self.axes,idx,
                                                                                               self.r_to_plot[id_buoy][key],self.theta_to_plot[id_buoy][key],
                                                                                               self.dict_to_plot,self.dirs_to_plot,id_buoy,key,self.levels_plots,label,False,self.date)
                        self.figs[id_buoy]=self.fig

                        self.cbar=util.vert_colorbar(self.fig,self.axes[idx],self.cf_last,0.05,0.01,'Wave energy [m²s/deg]')
                        self.cbar.set_ticks(self.cbar_ticks)

                        plt.subplots_adjust(wspace=0.4,hspace=0.6)
                        
                        self.fig.savefig(f'{self.plots_path}Fig9_spec_{id_buoy}.svg',dpi=800,bbox_inches='tight',pad_inches=0.05)

                return self.dict_axes,self.figs
        
        def setting_up_diff_plot(self,label,normalize):
                self.r_to_plot,self.theta_to_plot,self.dict_to_plot,self.dirs_to_plot=self.preparing_data()
                self.r_to_plot,self.theta_to_plot,self.dict_to_plot=self.interp_model_data(self.r_to_plot,
                                                                                           self.theta_to_plot,self.dict_to_plot)

                self.dict_axes={buoy:[1,2,3] for buoy in self.buoys_id}
                self.figs={}

                for id_buoy in self.buoys_id:
                        # self.fig,self.axes=plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(7,2.5),layout="constrained")
                        self.fig = plt.figure(figsize=(7,2.5))
                        self.ax1=self.fig.add_subplot(131,polar=True)
                        self.ax2=self.fig.add_subplot(132,polar=True)
                        self.ax3=self.fig.add_subplot(133,polar=True)
                        self.axes=(self.ax1,self.ax2,self.ax3)

                        plt.suptitle(f'Directional spectra - buoy {id_buoy} - {self.date}',fontsize=12,y=1.03)
                        max_norm=np.max(self.dict_to_plot[id_buoy]['buoy'])

                        self.spec_buoy_smoothed=np.array(list(map(util.moving_average_filter,self.dict_to_plot[id_buoy]['buoy'].T)))
                        self.dict_to_plot[id_buoy]['buoy']=self.spec_buoy_smoothed.T

                        if normalize==True:
                                self.dict_to_plot[id_buoy]['buoy']=self.dict_to_plot[id_buoy]['buoy']/max_norm
                                self.dict_to_plot[id_buoy]['model']=self.dict_to_plot[id_buoy]['model']/max_norm
                                # self.dict_to_plot[id_buoy]['buoy']=self.dict_to_plot[id_buoy]['buoy']
                                # self.dict_to_plot[id_buoy]['model']=self.dict_to_plot[id_buoy]['model']

                                self.dict_to_plot[id_buoy]['model_inter']=self.dict_to_plot[id_buoy]['model_inter']/max_norm

                        # Common features
                        self.cbar_ticks,self.levels_plots=util.customizing_colorbar(self.dict_to_plot[id_buoy],'norm',self.date)
                        self.cbar_ticks_d,self.levels_plots_d=util.customizing_colorbar(self.dict_to_plot[id_buoy],'diff',self.date)

                        # Loop over each spectra
                        for idx,key in enumerate(list(self.dict_to_plot[id_buoy].keys())):
                                if key != 'model_inter':
                                        self.cf_last,self.dict_axes[id_buoy][idx]=self.plotting_one_spectra(self.axes,idx,
                                                                                                            self.r_to_plot[id_buoy][key],self.theta_to_plot[id_buoy][key],
                                                                                                            self.dict_to_plot,self.dirs_to_plot,id_buoy,key,self.levels_plots,label,True,self.date)

                        # Plotting the difference
                        self.dict_to_plot[id_buoy]['buoy'][self.dict_to_plot[id_buoy]['buoy']<0.005]=np.nan
                        self.dict_to_plot[id_buoy]['model'][self.dict_to_plot[id_buoy]['model']<0.005]=np.nan
                        self.dict_to_plot[id_buoy]['model_inter'][self.dict_to_plot[id_buoy]['model_inter']<0.005]=np.nan
                        self.minuend=self.dict_to_plot[id_buoy]['buoy'][3:,:-1]
                        self.subtrahend=self.dict_to_plot[id_buoy]['model_inter']

                        result=np.zeros(self.minuend.shape)

                        for freq in range(self.minuend.shape[0]):
                                for dir in range(self.minuend.shape[1]):
                                        if (np.isnan(self.minuend[freq,dir])) and (np.isnan(self.subtrahend[freq,dir])==False):
                                                result[freq,dir]=-self.subtrahend[freq,dir]
                                        elif (np.isnan(self.minuend[freq,dir])==False) and (np.isnan(self.subtrahend[freq,dir])):
                                                result[freq,dir]=self.minuend[freq,dir]
                                        else:
                                                result[freq,dir]=self.minuend[freq,dir]-self.subtrahend[freq,dir]

                        self.dict_to_plot[id_buoy]['difference'] = result
                        self.dict_to_plot[id_buoy]['difference'] = np.concatenate((self.dict_to_plot[id_buoy]['difference'][:,:],self.dict_to_plot[id_buoy]['difference'][:,0].reshape(-1,1)),axis=1)
                        # self.dict_to_plot[id_buoy]['difference'][np.abs(self.dict_to_plot[id_buoy]['difference']) < 0.0002] = np.nan


                        print(f'{id_buoy} - {label}')
                        # print('buoy',np.min(self.dict_to_plot[id_buoy]['buoy'][3:,:-1]))
                        # print('model',np.min(self.dict_to_plot[id_buoy]['model_inter']),self.dict_to_plot[id_buoy]['model_inter'].shape)

                        # print('difference',self.dict_to_plot[id_buoy]['difference'])


                        self.cf_last_d,self.dict_axes[id_buoy][2]=self.plotting_one_spectra(self.axes,2,
                                                                                               self.r_to_plot[id_buoy]['model_inter'],self.theta_to_plot[id_buoy]['model_inter'],
                                                                                               self.dict_to_plot,self.dirs_to_plot,id_buoy,'difference',self.levels_plots_d,label,True,self.date)
                        self.figs[id_buoy]=self.fig

                        # Horizontal bar
                        self.cbar=util.horizontal_colorbar(self.fig,self.axes,self.cf_last,0.1,0.02,r'Normalized  $F(f,\theta)$',2)
                        self.cbar.set_ticks(self.cbar_ticks)
                        self.cbar.ax.tick_params(labelsize=7)
                        self.cbar.set_label(r'Normalized $F(f,\theta)$',fontsize=9)

                        if np.nanmax(self.dict_to_plot[id_buoy]['model'])>1:
                                self.over_spec=np.where((self.dict_to_plot[id_buoy]['model']>=1),self.dict_to_plot[id_buoy]['model'],1)
                                self.over_spec=np.where((self.dict_to_plot[id_buoy]['model']>=0.005),self.over_spec,np.nan)

                                self.cf_ext=self.axes[1].contourf(self.theta_to_plot[id_buoy]['model'],self.r_to_plot[id_buoy]['model'],
                                                                  self.over_spec.T,levels=[1.0,1.1,1.2,1.3,1.4,1.5],
                                                                  cmap='gist_heat_r',extend='max',zorder=-3)

                                self.cax_ext= self.fig.add_axes([self.axes[1].get_position().x1+0.08,
                                                                 self.axes[0].get_position().y0-0.1,self.axes[0].get_position().width/1.5,0.02])
                                self.cbar_ext=self.fig.colorbar(self.cf_ext,cax=self.cax_ext,orientation="horizontal")
                                self.cbar_ext.ax.tick_params(labelsize=7)

                        # Vertical bar
                        self.cbar=util.vert_colorbar(self.fig,self.axes[2],self.cf_last_d,0.05,0.01,r'Normalized  difference  $F(f,\theta)$')
                        self.cbar.set_ticks(self.cbar_ticks_d) #fontsize=8
                        self.cbar.ax.tick_params(labelsize=8)
                        self.cbar.set_label(r'Normalized  difference  $F(f,\theta)$',fontsize=9)

                        plt.subplots_adjust(wspace=0.35,hspace=0.5)

                        if self.date==dt.datetime(2020,5,10,13,0,0):
                                self.fig.savefig(f'{self.plots_path}Fig9_ef_2d_{id_buoy}_{label}_{self.date.strftime("%d%m%y")}.svg',
                                                 dpi=800,bbox_inches='tight',pad_inches=0.05)
                        else:
                                self.fig.savefig(f'{self.plots_path}Fig10_ef_2d_{id_buoy}_{label}_{self.date.strftime("%d%m%y")}.svg',
                                                 dpi=800,bbox_inches='tight',pad_inches=0.05)


                        # print(label,self.date,f'total error {id_buoy}',np.sum(np.abs(self.dict_to_plot[id_buoy]['difference'])))

                return self.dict_axes,self.figs
        

        def compare_another_conf(self,obj2,labels):
                self.r_to_plot,self.theta_to_plot,self.dict_to_plot=self.preparing_data()
                self.r_to_plot2,self.theta_to_plot2,self.adding_variables=obj2.preparing_data()

                self.dict_axes={buoy:[1,2,3] for buoy in self.buoys_id}

                for id_buoy in self.buoys_id:
                        self.r_to_plot2[id_buoy].pop('buoy'), self.theta_to_plot2[id_buoy].pop('buoy'), self.adding_variables[id_buoy].pop('buoy')

                        self.r_to_plot2[id_buoy][labels[1]] = self.r_to_plot2[id_buoy].pop('model')
                        self.theta_to_plot2[id_buoy][labels[1]] = self.theta_to_plot2[id_buoy].pop('model')
                        self.adding_variables[id_buoy][labels[1]] = self.adding_variables[id_buoy].pop('model')

                        self.r_to_plot[id_buoy][labels[0]] = self.r_to_plot[id_buoy].pop('model')
                        self.theta_to_plot[id_buoy][labels[0]] = self.theta_to_plot[id_buoy].pop('model')
                        self.dict_to_plot[id_buoy][labels[0]] = self.dict_to_plot[id_buoy].pop('model')   
                        
                        self.r_to_plot[id_buoy].update(self.r_to_plot2[id_buoy])
                        self.theta_to_plot[id_buoy].update(self.theta_to_plot2[id_buoy])
                        self.dict_to_plot[id_buoy].update(self.adding_variables[id_buoy])

                        self.fig,self.axes=plt.subplots(1,3,subplot_kw=dict(projection='polar'))
                        plt.suptitle(f'Directional spectra - buoy {id_buoy} - {self.date}',y=0.75)

                        self.cbar_ticks,self.levels_plots=util.customizing_colorbar(self.dict_to_plot[id_buoy])

                        # Loop over each variable
                        for idx,key in enumerate(list(self.dict_to_plot[id_buoy].keys())):  
                                self.cf_last,self.dict_axes[id_buoy][idx]=self.plotting_one_spectra(self.axes,idx,
                                                                                               self.r_to_plot[id_buoy][key],self.theta_to_plot[id_buoy][key],
                                                                                               self.dict_to_plot,id_buoy,key,self.levels_plots,'nothing')
                        self.figs[id_buoy]=self.fig

                        self.cbar=util.vert_colorbar(self.fig,self.axes[idx],self.cf_last,0.05,0.01,'Wave energy [m²s/deg]')
                        self.cbar.set_ticks(self.cbar_ticks)
                        plt.subplots_adjust(wspace=0.7,hspace=0.6)

                        self.fig.savefig(f'{self.plots_path}Fig9_spec_{id_buoy}_{labels[1]}.svg',dpi=800,bbox_inches='tight',pad_inches=0.05)
        
        def sra_plot(self):
                r,theta,first,second=self.preparing_data_SRA()
                first=np.concatenate((first[:,:],first[:,0].reshape(-1,1)),axis=1) 
                second=np.concatenate((second[:,:],second[:,0].reshape(-1,1)),axis=1) 

                time_first = dt.datetime(2004, 9, 14, 23,00)    # 2004 9 14 22 45 00
                time_second = dt.datetime(2004, 9, 12, 14,00) 

                self.time_ww3,self.freqs_ww3,self.dics_ww3,self.spec_2d_model = util.read_data_spec_stations(f'{self.data_path}ww3.2004_spec_2d.nc')
                self.data_ounp=util.read_data_int_stations(f'{self.data_path}ww3.2004_tab_params.nc')

                self.hs_model_first=self.data_ounp['A'].hs[time_first]
                self.hs_model_second=self.data_ounp['B'].hs[time_second]

                first_model = self.spec_2d_model['A']
                second_model = self.spec_2d_model['B']

                self.idx_time_first=self.time_ww3.get_loc(time_first)
                self.idx_time_sec=self.time_ww3.get_loc(time_second)

                first_model_spec=first_model[self.idx_time_first,:,:]
                second_model_spec=second_model[self.idx_time_sec,:,:]

                self.r_ww3,self.theta_ww3 = np.meshgrid (self.freqs_ww3,self.dics_ww3)
        
                # first_model=np.concatenate((first_model[:,::-1][:,8:],first_model[:,::-1][:,:8]),axis=1) #0-360
                # first_model=np.concatenate((first_model[:,:],first_model[:,0].reshape(-1,1)),axis=1) 

                # second_model=np.concatenate((second_model[:,::-1][:,8:],second_model[:,::-1][:,:8]),axis=1) #0-360
                # second_model=np.concatenate((second_model[:,:],second_model[:,0].reshape(-1,1)),axis=1) 

                # First plot from Liu et al.2017 OM
                self.fig1,self.axes=plt.subplots(1,2,subplot_kw=dict(projection='polar'),figsize=(6,3),layout="constrained")
                self.ax1=self.axes[0]
                self.ax3=self.axes[1]
                first_model_spec_to_plot=first_model_spec/np.nanmax(first)
                first_rsa_to_plot=first/np.nanmax(first)

                levels_plots = [0.005,0.008,0.012,0.017,0.026,0.039,0.059,0.088,0.132,0.198,0.296,0.444,0.667,1.000]
                cmap = plt.cm.RdYlBu_r
                self.colors = cmap(np.linspace(0, cmap.N, len(levels_plots)).astype('i'))                
                cf1=self.ax1.contourf(theta,r,first_rsa_to_plot.T,levels=levels_plots,colors=self.colors)
                cf3=self.ax3.contourf(self.theta_ww3,self.r_ww3,first_model_spec_to_plot.T,levels=levels_plots,colors=self.colors)
                self.ax1.set_rticks([0.03,0.06,0.09,0.12,0.15,0.18],labels=['',0.06,'',0.12,'',0.18],fontsize=8)
                self.ax1.set_rlim(0,0.18)
                self.ax1.set_title('SRA')
                self.ax3.set_rticks([0.03,0.06,0.09,0.12,0.15,0.18],labels=['',0.06,'',0.12,'',0.18],fontsize=8)
                self.ax3.set_rlim(0,0.18)
                self.ax3.set_title('ST6')
                # plt.setp(self.ax1,theta_direction=(1),theta_zero_location=('E'))
                self.ax1.set_xticklabels(['SE','S','SW','W','NW','N','NE','E'][::-1],fontsize=8)
                plt.setp(self.ax3,theta_direction=(-1),theta_zero_location=('N'))
                self.ax3.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'],fontsize=8)

                self.cbar_ticks=levels_plots

                self.cbar=util.horizontal_colorbar(self.fig1,self.axes,cf3,0.2,0.02,r'Normalized  $F(f,\theta)$',2)
                self.cbar.set_ticks(self.cbar_ticks)
                self.cbar.ax.tick_params(labelsize=7)
                self.cbar.set_label(r'Normalized $F(f,\theta)$',fontsize=9)

                self.over_spec=np.where((first_model_spec_to_plot>=1),first_model_spec_to_plot,1)
                self.over_spec=np.where((first_model_spec_to_plot>=0.005),self.over_spec,np.nan)

                self.cf_ext=self.ax3.contourf(self.theta_ww3,self.r_ww3,self.over_spec.T,
                                              levels=[1.0,1.1,1.2,1.3,1.4,1.5],
                                              cmap='gist_heat_r',extend='max',zorder=-3)

                self.cax_ext= self.fig1.add_axes([self.axes[1].get_position().x1+0.08,
                                                        self.axes[0].get_position().y0-0.2,self.axes[0].get_position().width/1.5,0.02])
                self.cbar_ext=self.fig1.colorbar(self.cf_ext,cax=self.cax_ext,orientation="horizontal")
                self.cbar_ext.ax.tick_params(labelsize=7)

                self.fig1.savefig(f'{self.plots_path}spec_SRA_1.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)


                # Second plot from Liu et al.2017 OM
                self.fig2,self.axes=plt.subplots(1,2,subplot_kw=dict(projection='polar'),figsize=(6,3),layout="constrained")
                self.ax2=self.axes[0]
                self.ax4=self.axes[1]
                second_rsa_to_plot=second/np.nanmax(second)
                second_model_spec_to_plot=second_model_spec/np.nanmax(second)
                cf2=self.ax2.contourf(theta,r,second_rsa_to_plot.T,levels=levels_plots,colors=self.colors)
                cf4=self.ax4.contourf(self.theta_ww3,self.r_ww3,second_model_spec_to_plot.T,levels=levels_plots,colors=self.colors,
                                      interpolarion='nearest')
                self.ax2.set_rticks([0.03,0.06,0.09,0.12,0.15,0.18],labels=['',0.06,'',0.12,'',0.18],fontsize=8)
                self.ax2.set_rlim(0,0.18)
                self.ax2.set_title('SRA')
                self.ax4.set_rticks([0.03,0.06,0.09,0.12,0.15,0.18],labels=['',0.06,'',0.12,'',0.18],fontsize=8)
                self.ax4.set_rlim(0,0.18)
                self.ax4.set_title('ST6')
                

                plt.setp(self.ax4,theta_direction=(-1),theta_zero_location=('N'))

                self.cbar=util.horizontal_colorbar(self.fig2,self.axes,cf4,0.2,0.02,r'Normalized  $F(f,\theta)$',2)
                self.cbar.set_ticks(self.cbar_ticks)
                self.cbar.ax.tick_params(labelsize=7)
                self.cbar.set_label(r'Normalized $F(f,\theta)$',fontsize=9)

                self.over_spec_sec=np.where((second_model_spec_to_plot>=1),second_model_spec_to_plot,1)
                self.over_spec_sec=np.where((second_model_spec_to_plot>=0.005),self.over_spec_sec,np.nan)

                self.cf_ext=self.ax4.contourf(self.theta_ww3,self.r_ww3,self.over_spec_sec.T,
                                              levels=[1.0,1.1,1.2,1.3,1.4,1.5],
                                              cmap='gist_heat_r',extend='max',zorder=-3)

                self.cax_ext= self.fig2.add_axes([self.axes[1].get_position().x1+0.08,
                                                        self.axes[0].get_position().y0-0.2,self.axes[0].get_position().width/1.5,0.02])
                self.cbar_ext=self.fig2.colorbar(self.cf_ext,cax=self.cax_ext,orientation="horizontal")
                self.cbar_ext.ax.tick_params(labelsize=7)

                self.fig2.savefig(f'{self.plots_path}spec_SRA_2.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
