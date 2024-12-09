import matplotlib.pyplot as plt
import numpy as np
import math
import datetime as dt

def vert_colorbar(fig,ax,cf,pad,width,label):
        cax= fig.add_axes([ax.get_position().x1+pad,ax.get_position().y0,
                                        width,ax.get_position().height])
        cbar=plt.colorbar(cf,cax=cax,orientation="vertical")
        cbar.set_label(label)
        return cbar

def horizontal_colorbar(fig,axes,cf,pad,width,label,join):
        if join==1:
                cax= fig.add_axes([axes[0].get_position().x0,axes[0].get_position().y0-pad,
                                                axes[0].get_position().width,width])
        elif join ==2:
                cax= fig.add_axes([axes[0].get_position().x0,axes[0].get_position().y0-pad,
                                                axes[0].get_position().width+1.4*axes[1].get_position().width,width])

        cbar=plt.colorbar(cf,cax=cax,orientation="horizontal")
        cbar.set_label(label)
        return cbar


def customizing_colorbar(dict_to_plot,label,date):
        max_both=max([np.max(dict_to_plot[key]) for key in dict_to_plot.keys()])
        max_final_both=(max_both-(max_both%0.25))+0.25
        levels_plots=np.linspace(0,max_final_both+0.25,50)
        if label == 'standard':
                if max_final_both>1 and max_final_both<8:
                        cbar_ticks=np.arange(0,max_final_both+0.5,0.5)
                if max_final_both>=8:
                        cbar_ticks=np.arange(0,max_final_both,2)
                else:
                        cbar_ticks=np.arange(0,max_final_both+0.25,0.25)
        elif label == 'norm':
                # levels_plots = [0.00,0.008,0.012,0.017,0.026,0.039,0.059,0.088,0.132,0.198,\
                # 0.296,0.444,0.667,1.000]
                levels_plots = [0.005,0.008,0.012,0.017,0.026,0.039,0.059,0.088,0.132,0.198,\
                0.296,0.444,0.667,1.000]

                cbar_ticks=levels_plots
        elif label =='diff':
                # cbar_ticks=np.arange(-0.7,0.8,0.1)
                # levels_plots=np.arange(-0.7,0.75,0.05)

                if date==dt.datetime(2020,5,10,13,0,0):
                        cbar_ticks=np.arange(-0.4,0.5,0.1)
                        levels_plots=np.arange(-0.4,0.5,0.1)
                else:
                        cbar_ticks=np.arange(-0.6,0.7,0.1)
                        levels_plots=np.arange(-0.6,0.7,0.1)

                # print(cbar_ticks)
                # print(levels_plots)
      
        return cbar_ticks,levels_plots

def customizing_colorbar_v2(x,y,term):
        if term =='sds':
                max_both=max([np.max(x),np.max(y)])
                min_both=min([np.min(x),np.min(y)])
                magnitude_order_min=math.floor(math.log10(abs(min_both)))
                lim_wo_min_magnitude=np.round(min_both/(1*(10**magnitude_order_min)),1)

                nearest_min_05=round((lim_wo_min_magnitude*2)/2)*(10**(magnitude_order_min))

                if nearest_min_05>min_both:
                        nearest_min_05=nearest_min_05-(0.5*(10**(magnitude_order_min)))
                levels_plots=np.linspace(nearest_min_05,0,40)

                if lim_wo_min_magnitude<5:
                        interval_cbar=0.5*(10**(magnitude_order_min))
                else:
                        interval_cbar=1*(10**(magnitude_order_min))
                cbar_ticks=np.arange(nearest_min_05,0+(interval_cbar/2),
                                     interval_cbar)
                
        else:
                max_both=max([np.max(x),np.max(y)])
                min_both=min([np.min(x),np.min(y)])
                magnitude_order_min=math.floor(math.log10(abs(min_both)))
                magnitude_order_max=math.floor(math.log10(abs(max_both)))
                lim_wo_min_magnitude=np.round(min_both/(1*(10**magnitude_order_min)),1)
                lim_wo_max_magnitude=np.round(max_both/(1*(10**magnitude_order_max)),1)

                nearest_max_05=round((lim_wo_max_magnitude*2)/2)*(10**(magnitude_order_max))
                nearest_min_05=round((lim_wo_min_magnitude*2)/2)*(10**(magnitude_order_min))

                if nearest_max_05<max_both:
                        nearest_max_05=nearest_max_05+(0.5*(10**(magnitude_order_max)))
                levels_plots=np.linspace(0,nearest_max_05,40)
                if lim_wo_max_magnitude<5:
                        interval_cbar=0.5*(10**(magnitude_order_max))
                else:
                        interval_cbar=1*(10**(magnitude_order_max))
                cbar_ticks=np.arange(0,nearest_max_05+(interval_cbar/2),
                                     interval_cbar)

        return cbar_ticks,levels_plots