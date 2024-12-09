import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from dateutil.relativedelta import relativedelta
import pandas as pd
import math 
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.tri as tri
#import wavespectra


import util
from . import soft_topo_cmap
from . import init_custom

import warnings
warnings.filterwarnings("ignore")

class domain():
        def __init__(self,root_path,ini_date) -> None:
              self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
              self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/'
              self.bthm_path = f'{root_path}/info/plots/gebco_2021_n32.0_s8.5_w-98.0_e-58.0.nc'

        def ploting(self):
                self.bat_data=xr.open_dataset(self.bthm_path)
                self.lat_bat=self.bat_data.lat
                self.lon_bat=self.bat_data.lon
                self.elevation_bat=self.bat_data.elevation

                self.norm_topo = soft_topo_cmap.FixPointNormalize(sealevel=0)
                cm = 1/2.54  # centimeters in inches

                self.fig,self.ax1 = plt.subplots(1,1,figsize=(8,6),subplot_kw={'projection': ccrs.PlateCarree()})
                self.land = cartopy.feature.NaturalEarthFeature('physical', 'land', \
                        scale='50m', edgecolor='k', facecolor=cfeature.COLORS['land'])
                self.ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
                        scale='50m', edgecolor='none', facecolor=cfeature.COLORS['water'])

                self.ax1.add_feature(self.land, facecolor='gray',alpha=0.5)
                self.ax1.add_feature(self.ocean, facecolor='gainsboro',alpha=0.6)

                cf=self.ax1.contourf(self.lon_bat[::15],self.lat_bat[::15],self.elevation_bat[::15,::15],30,transform=ccrs.PlateCarree(),\
                                norm=self.norm_topo,cmap='soft_topo',extend='both',zorder=3)

                self.ax1.set(xlabel='Longitude',ylabel='Latitude',xticks=(np.arange(-98,-55,6).tolist()), \
                                yticks=(np.arange(6,35,6).tolist()),ylim=(5.5,35),xlim=(-101,-55),frame_on=False)
                lon_formatter = LongitudeFormatter(number_format='g',
                                                degree_symbol='°')
                lat_formatter = LatitudeFormatter(number_format='g',
                                                degree_symbol='°')
                self.ax1.xaxis.set_major_formatter(lon_formatter)
                self.ax1.yaxis.set_major_formatter(lat_formatter)
                self.ax1.coastlines(resolution='10m',zorder=3)

                self.ax1.add_patch(mpatches.Rectangle(xy=[-98, 8.5], width=40, height=23.5,
                                                fc='none',ec='k',ls='--',lw=1,alpha=0.8,
                                                transform=ccrs.PlateCarree(),zorder=4))    

                self.cax = self.fig.add_axes([self.ax1.get_position().x1+0.02,self.ax1.get_position().y0,\
                                        0.015,self.ax1.get_position().height])
                self.cbar=plt.colorbar(cf,cax=self.cax,orientation="vertical",pad=0.12)
                self.cbar.ax.set_ylabel('Bathymetry and orography [m]',rotation=90,labelpad=0.45)
                return self.fig,self.ax1

        def adding_buoys(self):  # It could be a decorator
                self.fig,self.ax1=self.ploting()
                self.locs_buoys = {'42057':(16.908,-81.422),'42058':(14.394,-74.816),'42059':(15.300,-67.483)}
                for buoy in self.locs_buoys.keys():
                        self.ax1.plot(self.locs_buoys[buoy][1],self.locs_buoys[buoy][0],'^',color='r',markersize=6,zorder=5)
                        self.ax1.text(x=self.locs_buoys[buoy][1]-3.4,y=self.locs_buoys[buoy][0]-0.3,s=buoy,transform=ccrs.PlateCarree(),\
                                        fontsize=10)
                return self.fig,self.ax1

        def sav_plot(self):
                self.fig,self.ax1=self.adding_buoys()
                self.fig.savefig(f'{self.plots_path}Fig3_domain.pdf',dpi=700,bbox_inches='tight',pad_inches=0.05)
                plt.show()

class results():
        def __init__(self,root_path,ini_date,end_date) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/'
                self.idate=ini_date
                self.edate=end_date
                
        def qqplot(self):
                data_stations=util.read_data_int_stations(f'{self.data_path}ww3.2020_tab.nc')
                self.buoys_id = ['42057','42058','42059','42060']
                for id in self.buoys_id:    
                        data_x = util.ord_buoy_data(id).hs[self.idate:self.edate+relativedelta(hours=1)]
                        new_x_index=(data_x.index-pd.Timedelta(minutes=40))
                        data_y = data_stations[id].hs[new_x_index]
                        fig,ax1=plt.subplots(1,1)
                        try:
                                ax1.scatter(data_x.values,data_y.values,c='k')
                                sup_lim=math.ceil(max(max(data_x.values),max(data_y.values)))
                                ax1.plot([0,sup_lim],[0,sup_lim],'--',c='k')
                        except:
                                print('doesnt match size')
                        ax1.set(xlabel="Hs [m] - Buoy",ylabel="Hs [m] - Model",xlim=(0,sup_lim),ylim=(0,sup_lim),\
                                        title=f"QQ Plot Hs [m]: Boya {id} - Model")
                        plt.savefig(f'{self.plots_path}qqplot_{id}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)