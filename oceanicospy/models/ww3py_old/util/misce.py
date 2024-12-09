import numpy as np
from scipy.stats import pearsonr
from scipy import interpolate
import pandas as pd

# from sklearn.metrics import mean_absolute_error as mae

def moving_average_filter(signal, window_size=3):
    window = np.ones(window_size) / window_size
    filtered_signal = np.convolve(signal, window, mode='same')
    return filtered_signal

def metrics(buoy,model):
    if np.any(np.isnan(buoy)) == True:
        idx_nans=~np.isnan(buoy)
        buoy=buoy[idx_nans]
        model=model[idx_nans]
    
    RMSE_no_round = np.sqrt(((buoy - model)**2).mean())
    RMSE = round(RMSE_no_round,2) 
    NRMSE = round(RMSE_no_round/(np.max(buoy)-np.min(buoy)),2)

    bias = np.sqrt(((buoy - np.mean(buoy))**2).mean()) - \
            np.sqrt(((model - np.mean(model))**2).mean())
    bias = round(bias,2)

    # is it useful compare MBE and

    MBE = np.mean(np.abs(buoy - model))
    MBE = round(MBE,2)

    corr, _ = pearsonr(buoy,model)
    corr=round(corr,2)

    # MABE = mae(buoy,model)
    # MABE=round(MABE,2)

    return RMSE,NRMSE,MBE

def interp_1d_spectra(freq_original,spec_original,freq_new):
    f=interpolate.interp1d(freq_original,spec_original)
    spec_to_plot_model_inter=f(freq_new)
    return spec_to_plot_model_inter

def closest_node(nodes,nodes_target):
    nodes = np.asarray(nodes)
    indx=np.empty(len(nodes_target),dtype=np.int)
    for idx,node in enumerate(nodes_target):
      dist_2 = np.sum((nodes - node)**2, axis=1)
      series=pd.Series(dist_2)
      series=series.sort_values(ascending=True)
      indx[idx]=series.index[0]
    return indx