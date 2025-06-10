import cdsapi

c = cdsapi.Client()

def getting_wind_data(lat_ll,lon_ll,meshes_x_wind, meshes_y_wind,dx_wind,dy_wind,data_path):
    """
    Downloads ERA5 data for the specified parameters and saves it to a NetCDF file.
    """
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': '2025',
            'month': '05',
            'day': [str(i) for i in range(4,21)],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                lat_ll+(meshes_y_wind*dy_wind), lon_ll-360, lat_ll,
                lon_ll-360+(meshes_x_wind*dx_wind),
            ],
            'grid': [0.025, 0.025],
        },
        f'{data_path}.nc')

def getting_wave_data(lat_ll,lon_ll,meshes_x_wind, meshes_y_wind,dx_wind,dy_wind,data_path):
    """
    Downloads ERA5 data for the specified parameters and saves it to a NetCDF file.
    """
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
               'mean_wave_direction','peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell',
            ],
            'year': '2025',
            'month': '05',
            'day': [str(i) for i in range(4,21)],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                lat_ll+(meshes_y_wind*dy_wind), lon_ll-360, lat_ll,
                lon_ll-360+(meshes_x_wind*dx_wind),
            ],
            'grid': [0.025, 0.025],
        },
        f'{data_path}.nc')