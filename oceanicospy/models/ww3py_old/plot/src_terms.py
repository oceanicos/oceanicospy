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
from scipy import integrate
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import itertools
from matplotlib.ticker import FormatStrFormatter

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

class Src_terms():
        def __init__(self,root_path,ini_date,date) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/src/'
                os.system(f'mkdir -p {self.plots_path}')
                self.date=date
                self.buoys_id = ['42057','42058','42059','42060']
                self.efth_min = 1e-3 # Minimum value for wave energy
                self.dics_complete=np.arange(0,370,10)
                self.locs_buoys = {'42057':(16.908,-81.422+360),'42058':(14.394,-74.816+360),'42059':(15.300,-67.483+360),
                                        '42060':(16.434,-63.329+360)}

        def preparing_data(self,term):
                
                # Reading model results
                self.time_1d,self.freqs_1d,self.src_1d = util.read_data_src_stations(f'{self.data_path}ww3.2020_src_1d.nc',
                                                                                     '1d',term)
                self.time_2d,self.freqs_2d,self.dics_2d,self.src_2d = util.read_data_src_stations(f'{self.data_path}ww3.2020_src_2d.nc',
                                                                                                  '2d',term)
                
                self.time_ww3,self.freqs_ww3,self.dics_ww3,self.spec_2d_model = util.read_data_spec_stations(f'{self.data_path}ww3.2020_spec_2d.nc')

                if term=='sin':
                        self.time,self.lons_2d,self.lats_2d,self.utaw,self.vtaw = util.read_spatial_data(f'{self.data_path}ww3.2020.nc')
                        self.coordinates = np.array(list(itertools.product(self.lats_2d, self.lons_2d)))
                        self.idx_buoys=util.closest_node(self.coordinates,list(self.locs_buoys.values()))
                        self.coordinates_closer={key:self.coordinates[ele] for key,ele in zip(self.locs_buoys.keys(),self.idx_buoys)}

                self.wave_params=util.read_data_int_stations(f'{self.data_path}ww3.2020_tab_params.nc')
                self.wind_params=util.read_data_extra_stations(f'{self.data_path}ww3.2020_extra_params.nc')

                self.dics_data={}
                self.r_data={}
                self.theta_data={}
                self.dirs={}
                self.taws={}                
                self.taw_series={}                

                for id in self.buoys_id:    

                        # Spectra from the model (ww3)
                        self.idx_dateini_ww3=self.time_2d.get_loc(self.date)

                        # self.dics_complete_ww3=np.concatenate((np.arange(90,-10,-10),np.arange(350,80,-10)))
                        # self.dics_complete_ww3=(self.dics_complete_ww3+180)%360
                        # self.r_ww3,self.theta_ww3 = np.meshgrid(self.freqs_2d,np.deg2rad(self.dics_complete_ww3))

                        self.dics_complete_ww3=np.round((np.degrees(self.dics_2d)+180)%360,1)
                        self.dic_complete_ww3_sorted=np.concatenate((self.dics_complete_ww3[::-1][8:],self.dics_complete_ww3[::-1][:8],
                                                                     np.array([360]))) # 0-360
                        self.r_ww3,self.theta_ww3 = np.meshgrid (self.freqs_2d,np.radians(self.dic_complete_ww3_sorted))

        
                        self.src_to_plot_1d=self.src_1d[id]
                        self.src_to_plot_1d=self.src_to_plot_1d[self.idx_dateini_ww3,:]

                        self.src_to_plot_2d=self.src_2d[id]
                        self.src_to_plot_2d=self.src_to_plot_2d[self.idx_dateini_ww3,:,:]
                        self.src_to_plot_2d=np.concatenate((self.src_to_plot_2d[:,::-1][:,8:],self.src_to_plot_2d[:,::-1][:,:8]),axis=1) #0-360

                        self.src_to_plot_2d=np.concatenate((self.src_to_plot_2d[:,:],self.src_to_plot_2d[:,0].reshape(-1,1)),axis=1)

                        # Sorting and arranging the full spectra
                        self.dics_complete_ww3=np.round((np.degrees(self.dics_ww3)+180)%360,1)
                        self.dic_complete_ww3_sorted=np.concatenate((self.dics_complete_ww3[::-1][8:],self.dics_complete_ww3[::-1][:8],np.array([360]))) # 0-360
                        self.r_ww3,self.theta_ww3 = np.meshgrid (self.freqs_ww3,np.radians(self.dic_complete_ww3_sorted))
        
                        self.spec_to_plot_model=self.spec_2d_model[id]
                        self.spec_to_plot_model=self.spec_to_plot_model[self.idx_dateini_ww3,:,:]
                        self.spec_to_plot_model=np.concatenate((self.spec_to_plot_model[:,::-1][:,8:], 
                                                                self.spec_to_plot_model[:,::-1][:,:8]),axis=1) #0-360
                        self.spec_to_plot_model=np.concatenate((self.spec_to_plot_model[:,:],self.spec_to_plot_model[:,0].reshape(-1,1)),axis=1) 

                        # Storaging spectra, freqs and dirs arrays in dictioneries
                        self.dics_data[id]=dict(oned=self.src_to_plot_1d,twod=self.src_to_plot_2d,full_twod=self.spec_to_plot_model)
                        self.r_data[id]=dict(oned=self.r_ww3,twod=self.r_ww3)
                        self.theta_data[id]=dict(twod=self.theta_ww3)
                       
                        self.wndir_model=self.wind_params[id].wnddir[self.date]
                        self.wvdir_model=self.wave_params[id].dirp[self.date]
                        self.dirs[id]=dict(wind_model=self.wndir_model,wave_model=self.wvdir_model)

                        # Computing taw for each buoy

                        if term=='sin':
                                idx_lon=np.where(self.lons_2d==self.coordinates_closer[id][1])[0]
                                idx_lat=np.where(self.lats_2d==self.coordinates_closer[id][0])[0]
                                self.taw=np.sqrt((self.utaw[self.idx_dateini_ww3,idx_lat,idx_lon]**2)+\
                                                (self.vtaw[self.idx_dateini_ww3,idx_lat,idx_lon]**2))

                                self.taw_serie=pd.Series(index=self.time_2d,data=np.sqrt((self.utaw[:,idx_lat,idx_lon]**2)+\
                                                (self.vtaw[:,idx_lat,idx_lon]**2)).ravel())
                                self.taw_series[id]=self.taw_serie
                                self.taws[id]=dict(taw_model=self.taw)
                
                return self.r_data,self.theta_data,self.dics_data,self.dirs,self.taws,self.taw_series                                
        
        def plotting_one_2d_src(self,axes,idx,r,theta,dict_var,id_buoy,label,levels_ctf,dirs,term):

                ax=axes[idx]
                self.spec=dict_var[id_buoy]['twod']


                if term=='sin':
                        self.cf=ax.contourf(theta,r,self.spec.T,levels=levels_ctf,cmap='Spectral_r',extend='min')
                else:
                        self.cf=ax.contourf(theta,r,self.spec.T,levels=levels_ctf,cmap='Spectral')                        

                if label=='ctrl':
                        ax.plot(theta[:,0], np.tile([r[0,12]],len(theta[:,0])),ls='dotted',color='k',zorder=1,lw=1.5)
                        ax.plot(theta[:,0], np.tile([r[0,19]],len(theta[:,0])),ls='dashdot',color='k',zorder=1,lw=1.5)
                        ax.plot(theta[:,0], np.tile([r[0,23]],len(theta[:,0])),ls='dashed',color='k',zorder=1,lw=1.5)


                else:
                        ax.plot(theta[:,0], np.tile([r[0,12]],len(theta[:,0])),ls='dotted',color='darkorange',zorder=1,lw=1.5)                        
                        ax.plot(theta[:,0], np.tile([r[0,19]],len(theta[:,0])),ls='dashdot',color='darkorange',zorder=1,lw=1.5)
                        ax.plot(theta[:,0], np.tile([r[0,23]],len(theta[:,0])),ls='dashed',color='darkorange',zorder=1,lw=1.5)


                ax.annotate(None, xy=((dirs[id_buoy]['wind_model']/360)*(2*np.pi), 0.115),
                             xytext=(0, 0),arrowprops=dict(arrowstyle="-|>",color='k'))

                # ax.annotate(None, xy=((dirs[id_buoy]['wave_model']/360)*(2*np.pi), 0.115), 
                #             xytext=(0, 0),arrowprops=dict(arrowstyle="-|>",color='red'))

                plt.setp(ax,theta_direction=(-1),theta_zero_location=('N'))
                ax.set(yticks=[0.1,0.2,0.3,0.4,0.5])
                ax.set_yticklabels([0.1,0.2,0.3,0.4,0.5],fontsize=12)
                ax.set_rlabel_position(22) 
                ax.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'],fontsize=12) 
                if label=='winp':
                        ax.set_title(r'$cos^{2}$',fontsize=13)
                elif label=='winp4':
                        ax.set_title(r'$cos^{4}$',fontsize=13)
                elif label=='expe' or label=='bim':
                        ax.set_title('bimodal',fontsize=13)
                else:
                        ax.set_title(label,fontsize=13)

                ax.grid(True,alpha=0.5)
                ax.set_rlim(0,0.5)
                return self.cf,ax
        
        def plotting_dist_dir_src(self,axes,idx,r,theta,dict_var_l,dict_var_r,dirs_l,id_buoy,label_l,label_r,term):

                ax=axes[idx]
                self.spec_l=dict_var_l[id_buoy]['twod']
                self.spec_r=dict_var_r[id_buoy]['twod']

                if idx==2:
                        ax.plot(np.degrees(theta[:,0]),self.spec_l[12,:],label=label_l,ls='dotted',color='k',lw=1.5)
                        ax.plot(np.degrees(theta[:,0]),self.spec_r[12,:],label=label_r,ls='dotted',color='darkorange',lw=1.5)
                        ax.set_title(f'freq={truncate_float(r[0,12],2)} Hz',loc='right',fontsize=8)
                        ax.text(370, 0.30*np.nanmin(self.spec_r[12,:]), 'a)', fontsize=10, fontweight='bold', va='bottom', ha='right')
                        ax.tick_params(axis='x',which='both',labelbottom=False)
                        ax.ticklabel_format(useMathText=True)
                elif idx ==3:
                        ax.plot(np.degrees(theta[:,0]),self.spec_l[19,:],label=label_l,ls='dashdot',color='k',lw=1.5)
                        ax.plot(np.degrees(theta[:,0]),self.spec_r[19,:],label=label_r,ls='dashdot',color='darkorange',lw=1.5)
                        ax.set_title(f'freq={truncate_float(r[0,19],2)} Hz',loc='right',fontsize=8)
                        if term=='sin':
                                ax.text(370, 0.68*np.nanmax(self.spec_r[19,:]), 'b)', fontsize=10, fontweight='bold', va='bottom', ha='right')
                        else:
                                ax.text(370, 0.30*np.nanmin(self.spec_r[19,:]), 'b)', fontsize=10, fontweight='bold', va='bottom', ha='right')

                        ax.tick_params(axis='x',which='both',labelbottom=False)
                        ax.ticklabel_format(useMathText=True)
                        if term=='sds':         
                                ax.set_ylabel(r'$S_{ds}$'+' [m²Hz/deg]', fontsize=12)
                        else:
                                ax.set_ylabel(r'$S_{in}$'+' [m²Hz/deg]', fontsize=12)
                else:
                        ax.plot(np.degrees(theta[:,0]),self.spec_l[23,:],label=label_l,ls='dashed',color='k',lw=1.5)
                        ax.plot(np.degrees(theta[:,0]),self.spec_r[23,:],label=label_r,ls='dashed',color='darkorange',lw=1.5)
                        ax.ticklabel_format(useMathText=True)
                        if term=='sin':
                                ax.text(370, 0.68*np.nanmax(self.spec_r[23,:]), 'c)', fontsize=10, fontweight='bold', va='bottom', ha='right')
                        else:
                                ax.text(370, 0.30*np.nanmin(self.spec_r[23,:]), 'c)', fontsize=10, fontweight='bold', va='bottom', ha='right')

                        ax.set_title(f'freq={truncate_float(r[0,23],2)} Hz',loc='right',fontsize=8)
                        if term=='sin':
                                final_arrow=ax.get_ylim()[1]/3
                                origin_arrow=ax.get_ylim()[0]         
                        else:
                                final_arrow=ax.get_ylim()[0]/3
                                origin_arrow=ax.get_ylim()[1]
                        ax.annotate(None, xy=((dirs_l[id_buoy]['wind_model'], final_arrow)),
                                xytext=(dirs_l[id_buoy]['wind_model'],origin_arrow),arrowprops=dict(arrowstyle="-|>",color='k'))

                        ax.set_xlabel('Direction [°]', fontsize=12)


                ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
                ax.grid(True,alpha=0.5)

                if idx==2:
                        print('Buoy: ', id_buoy)
                        # int_sin_ctrl_f1=integrate.simpson(self.spec_l[:,:-1],dx=np.radians(10),axis=1)
                        # int_sin_ctrl_f1_2d=integrate.simpson(int_sin_ctrl_f1,x=r[0,:])
                        # print('integral over dirs - ctrl:', int_sin_ctrl_f1_2d)
                        # # int_sin_cos2_f1 =integrate.cumulative_trapezoid(self.spec_r[:,:],dx=np.radians(10))[-1]
                        # int_sin_cos2_f1 =integrate.simpson(self.spec_r[:,:],dx=np.radians(10),axis=1)
                        # int_sin_cos2_f1_2d=integrate.simpson(int_sin_cos2_f1,x=r[0,:])
                        # print('integral over dirs - cos2:', int_sin_cos2_f1_2d,'\n')
                return ax
        
        def setting_up_plot_2d(self,term,label_l,obj_r,label_r):
                self.r_to_plot_l,self.theta_to_plot_l,self.dict_to_plot_l,dirs_l,taw_l,serie_taw_l=self.preparing_data(term)
                self.r_to_plot_r,self.theta_to_plot_r,self.dict_to_plot_r,dirs_r,taw_r,serie_taw_r=obj_r.preparing_data(term)

                self.dict_axes={buoy:[1,2,3,4,5] for buoy in self.buoys_id}
                self.figs={}

                for id_buoy in self.buoys_id:
                        self.fig=plt.figure(figsize=(12,3))
                        gs = GridSpec(3,3)
                        ax1 = self.fig.add_subplot(gs[:,0],projection='polar')
                        ax2 = self.fig.add_subplot(gs[:,1],projection='polar')
                        ax3 = self.fig.add_subplot(gs[0,2])
                        ax4 = self.fig.add_subplot(gs[1,2])
                        ax5 = self.fig.add_subplot(gs[2,2])


                        self.axes=(ax1,ax2,ax3,ax4,ax5)

                        if term =='sin':
                                plt.suptitle('$S_{in}$ '+f'directional spectra - buoy {id_buoy} - {self.date}',y=1.13,fontsize=15)
                        else:
                                plt.suptitle('$S_{ds}$ '+f'directional spectra - buoy {id_buoy} - {self.date}',y=1.13,fontsize=15)                                


                        # self.dict_to_plot_l[id_buoy]['twod']=self.dict_to_plot_l[id_buoy]['twod']/self.dict_to_plot_l[id_buoy]['full_twod']
                        # self.dict_to_plot_r[id_buoy]['twod']=self.dict_to_plot_r[id_buoy]['twod']/self.dict_to_plot_r[id_buoy]['full_twod']

                        # Common features
                        self.cbar_ticks,self.levels_plots=util.customizing_colorbar_v2(self.dict_to_plot_l[id_buoy]['twod'],
                                                                                       self.dict_to_plot_r[id_buoy]['twod'],term)


                        # Loop over each type of result (buoy or simulations)
                        self.cf_last,self.dict_axes[id_buoy][0]=self.plotting_one_2d_src(self.axes,0,self.r_to_plot_l[id_buoy]['twod'],
                                                                                               self.theta_to_plot_l[id_buoy]['twod'],
                                                                                               self.dict_to_plot_l,id_buoy,label_l,
                                                                                               self.levels_plots,dirs_l,term)
                        self.cf_last,self.dict_axes[id_buoy][1]=self.plotting_one_2d_src(self.axes,1,self.r_to_plot_r[id_buoy]['twod'],
                                                                                               self.theta_to_plot_r[id_buoy]['twod'],
                                                                                               self.dict_to_plot_r,id_buoy,label_r,
                                                                                               self.levels_plots,dirs_r,term)
                        self.dict_axes[id_buoy][2]=self.plotting_dist_dir_src(self.axes,2,self.r_to_plot_r[id_buoy]['twod'],
                                                                                               self.theta_to_plot_r[id_buoy]['twod'],
                                                                                               self.dict_to_plot_l,self.dict_to_plot_r,
                                                                                               dirs_l,id_buoy,label_l,label_r,term)
                        self.dict_axes[id_buoy][3]=self.plotting_dist_dir_src(self.axes,3,self.r_to_plot_r[id_buoy]['twod'],
                                                                                               self.theta_to_plot_r[id_buoy]['twod'],
                                                                                               self.dict_to_plot_l,self.dict_to_plot_r,
                                                                                               dirs_l,id_buoy,label_l,label_r,term)
                        self.dict_axes[id_buoy][4]=self.plotting_dist_dir_src(self.axes,4,self.r_to_plot_r[id_buoy]['twod'],
                                                                                               self.theta_to_plot_r[id_buoy]['twod'],
                                                                                               self.dict_to_plot_l,self.dict_to_plot_r,
                                                                                               dirs_l,id_buoy,label_l,label_r,term)

                        self.axes[3].sharex(self.axes[2])
                        self.axes[4].sharex(self.axes[2])

                        self.cbar=util.horizontal_colorbar(self.fig,(ax1,ax3),self.cf_last,0.15,0.04,'Spectral density [m²/deg]',2)

                        formatter = ticker.ScalarFormatter(useMathText=True)
                        formatter.set_powerlimits((-3, 3))  # Set the range for scientific notation
                        self.cbar.formatter = formatter
                        self.cbar.update_ticks()
                        self.cbar.set_ticks(self.cbar_ticks)

                        plt.subplots_adjust(wspace=0.4,hspace=0.6)
                        
                        self.fig.savefig(f'{obj_r.plots_path}src_{term}_2d_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
                        self.figs[id_buoy]=self.fig

                        if term =='sin':
                                self.fig2,ax1=plt.subplots(1,1,figsize=(12,3))
                                ax1.plot(serie_taw_l[id_buoy],label='ctrl',lw=1)
                                ax1.plot(serie_taw_r[id_buoy],label=r'$cos^{4}$',lw=1)
                                ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
                                ax1.legend()
                                ax1.grid(True,linestyle='dotted')
                                ax1.set(ylabel=r'$\tau_{w}$'+' [m²/s²]')
                                ax1.set_title(r'Time series of $\tau_{w}$ for the closest location to buoy '+f'{id_buoy}')
                                myFmt = mdates.DateFormatter('%m-%d-%y')
                                fmt_day = mdates.DayLocator()
                                ax1.xaxis.set_major_formatter(myFmt)
                                ax1.xaxis.set_minor_locator(fmt_day)

                                self.fig2.savefig(f'{obj_r.plots_path}taw_series_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

                return self.dict_axes,self.figs

        def compare_another_conf(self,obj2,dict2,figs2,label,color,freq_along):
                self.r_to_plot2,self.theta_to_plot2,self.dict_to_plot2=obj2.preparing_data()
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                self.figs2={}

                for id_buoy in self.buoys_id:
                        # Loop over each variable
                        self.dict_axes[id_buoy][0]=self.plotting_one_1d_src(dict2[id_buoy][0],self.r_to_plot2[id_buoy]['unod'],self.theta_to_plot2[id_buoy],
                                                                                self.dict_to_plot2,id_buoy,
                                                                                label,color,freq_along)                        

                        if freq_along==True:
                                figs2[id_buoy].savefig(f'{self.plots_path}src_1d_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
                        else:
                                figs2[id_buoy].savefig(f'{self.plots_path}src_1d_{id_buoy}_dirs.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

                # for id in self.buoys_id:    
                #         data_y = data_spectra[id]
                #         fig,ax1=plt.subplots(1,1,subplot_kw=dict(projection='polar'))
                #         plt.setp(ax1,rmin=0,rmax=1,theta_direction=(-1),theta_zero_location=('N'))
                #         norm = colors.TwoSlopeNorm(vmin=np.nanmin(data_y[50,:,:].T), vcenter=0,vmax=np.nanmax(data_y[50,:,:].T))

                #         cf=ax1.contourf(self.theta,self.r,data_y[50,:,:].T,levels=50,norm=norm,cmap='seismic')
                #         ax1.set(title=f'Snl 2D - buoy {id} - {time[50]}',xticks=(np.radians(np.arange(0,360,90))))