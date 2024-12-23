import os
import xarray as xr
import numpy as np

from pathlib import Path

from oceanicospy.analysis import climatology,extremes

out_dir = f'../tests/{Path(__file__).stem}'

# Create output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Test climatology
datapath = '../data/nc_data/wave_data.nc'
dataset = xr.load_dataset(datapath)
annual_cycle_hs = climatology.compute_annual_cycle(dataset.swh[:,0,0], dataset.valid_time.values)

# Test extremes
hs_value_200 = extremes.POT_method([0.005], dataset.swh[:,0,0], np.percentile(dataset.swh[:,0,0],95), 3)