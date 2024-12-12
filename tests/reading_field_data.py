from datetime import datetime, timedelta
from pathlib import Path
import os

from oceanicospy.observations import Awac,AQUAlogger
from oceanicospy.analysis import spectral,temporal
from oceanicospy.plots import *

# Output directory
out_dir = f'../tests/{Path(__file__).stem}'

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

for idx,path_folder in enumerate(campaigns_directories[:1]):
    print(metadata_list[idx])
    AWAC_measurements = Awac(path_folder,sampling_data_list[idx])
    wave_records = AWAC_measurements.get_clean_wave_records()
    spectra_puv,params_puv = spectral.spectra_from_puv(wave_records,sampling_data_list[idx])
    spectra_surflevel,params_surflevel = spectral.spectra_from_fft(wave_records,sampling_data_list[idx])
    params_zerocross = temporal.params_from_zero_crossing(wave_records,sampling_data_list[idx])

    fig_params_hs = plot_wave_parameters(params_puv,out_dir,metadata_list[idx],label_comparison='puv')
    plot_wave_parameters(params_surflevel,out_dir,metadata_list[idx],fig_comparison=fig_params_hs,label_comparison='surf')
    plot_wave_parameters(params_zerocross,out_dir,metadata_list[idx],["H1/3"],fig_comparison=fig_params_hs,label_comparison='zerocross')

    fig_params_t = plot_wave_parameters(params_puv,out_dir,metadata_list[idx],["Tp","Tm01","Tm02"],ylabel='Period [s]',label_comparison='puv')
    plot_wave_parameters(params_surflevel,out_dir,metadata_list[idx],["Tp","Tm01","Tm02"],ylabel='Period [s]',fig_comparison=fig_params_t,label_comparison='surf')
    plot_wave_parameters(params_zerocross,out_dir,metadata_list[idx],["Tmean"],ylabel='Period [s]',fig_comparison=fig_params_t,label_comparison='zerocross')

    plot_1d_wave_spectra(spectra_puv,out_dir,metadata_list[idx],label_comparison='puv')
    plot_1d_wave_spectra(spectra_surflevel,out_dir,metadata_list[idx],label_comparison='surf_level')

    current_records = AWAC_measurements.read_currents_records()

campaigns_directories=['/homes/medellin/ffayalac/data/LR1_LittleReef-out/']

sampling_data_LR1_2019=dict(anchoring_depth=11.6,sensor_height=1.20,sampling_freq=1,
                            start_time=datetime(2019,11,16,13,0,0),end_time=datetime(2019,11,20,8,0,0)-timedelta(minutes=1))

sampling_data_list=[sampling_data_LR1_2019]
metadata_list=['aqualogger_LR1_2019']

for idx,path_folder in enumerate(campaigns_directories):
    print(metadata_list[idx])
    AQUAlogger_measurements=AQUAlogger(path_folder,sampling_data_list[idx])
    records=AQUAlogger_measurements.get_clean_records()
    spectra_surflevel,params_surflevel=spectral.spectra_from_fft(records,sampling_data_list[idx])
    params_zerocross=temporal.params_from_zero_crossing(records,sampling_data_list[idx])

    fig_params_hs = plot_wave_parameters(params_surflevel,out_dir,metadata_list[idx],label_comparison='surflevel')
    plot_wave_parameters(params_zerocross,out_dir,metadata_list[idx],["H1/3"],fig_comparison=fig_params_hs,label_comparison='zerocross')

    fig_params_t = plot_wave_parameters(params_surflevel,out_dir,metadata_list[idx],["Tp","Tm01","Tm02"],ylabel='Period [s]',label_comparison='surflevel')
    plot_wave_parameters(params_zerocross,out_dir,metadata_list[idx],["Tmean"],ylabel='Period [s]',fig_comparison=fig_params_t,label_comparison='zerocross')

    plot_1d_wave_spectra(spectra_surflevel,out_dir,metadata_list[idx],label_comparison='surf_level')
