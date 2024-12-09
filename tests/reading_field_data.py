from datetime import datetime, timedelta
import os

from oceanicospy.observations import Awac,AQUAlogger
from oceanicospy.plots import *

out_dir = f'../tests/{os.path.basename(__file__)[:-3]}'    # Output directory
# Create output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

campaigns_directories=['/homes/medellin/ffayalac/data/AWAC_1000_2018/','/homes/medellin/ffayalac/data/AWAC_1000_Ago2023/MEDICIONES/',
                        '/homes/medellin/ffayalac/data/AWAC_1000_Feb2024/']

sampling_data_awac_2018=dict(anchoring_depth=11.6,sensor_height=1.20,sampling_freq=2,
                            start_time=datetime(2018,5,5,10,0,0),end_time=datetime(2018,5,10,7,0,0))
sampling_data_awac_2023=dict(anchoring_depth=11.6,sensor_height=1.20,sampling_freq=2,
                            start_time=datetime(2023,8,19,11,0,0),end_time=datetime(2023,9,1,8,0,0,0))
sampling_data_awac_2024=dict(anchoring_depth=11.6,sensor_height=1.20,sampling_freq=2,
                            start_time=datetime(2024,2,29,12,0,0),end_time=datetime(2024,3,15,16,0,0))

sampling_data_list=[sampling_data_awac_2018,sampling_data_awac_2023,sampling_data_awac_2024]
metadata_list=['awac_1000_2018','awac_1000_2023','awac_1000_2024']

for idx,path_folder in enumerate(campaigns_directories):
    print(metadata_list[idx])
    AWAC_measurements=Awac(path_folder,sampling_data_list[idx],metadata_list[idx])
    records=AWAC_measurements.getting_clean_records()
    spectra_puv,params_puv=AWAC_measurements.spectra_from_puv(records)
    spectra_surflevel,params_surflevel=AWAC_measurements.spectra_from_fft(records)
    params_zerocross=AWAC_measurements.params_from_zero_crossing(records)

    fig_params_hs = plot_wave_parameters(params_puv,out_dir,metadata_list[idx],label_comparison='puv')
    plot_wave_parameters(params_surflevel,out_dir,metadata_list[idx],fig_comparison=fig_params_hs,label_comparison='surf')
    plot_wave_parameters(params_zerocross,out_dir,metadata_list[idx],["H1/3"],fig_comparison=fig_params_hs,label_comparison='zerocross')

    fig_params_t = plot_wave_parameters(params_puv,out_dir,metadata_list[idx],["Tp","Tm01","Tm02"],ylabel='Period [s]',label_comparison='puv')
    plot_wave_parameters(params_surflevel,out_dir,metadata_list[idx],["Tp","Tm01","Tm02"],ylabel='Period [s]',fig_comparison=fig_params_t,label_comparison='surf')
    plot_wave_parameters(params_zerocross,out_dir,metadata_list[idx],["Tmean"],ylabel='Period [s]',fig_comparison=fig_params_t,label_comparison='zerocross')

    plot_1d_wave_spectra(spectra_puv,out_dir,metadata_list[idx],label_comparison='puv')
    plot_1d_wave_spectra(spectra_surflevel,out_dir,metadata_list[idx],label_comparison='surf_level')