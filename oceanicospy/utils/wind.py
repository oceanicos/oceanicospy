from pathlib import Path

from ..downloads import ERA5Downloader, CMDSDownloader


def download_era5_winds(wind_info, ini_date, end_date, difference_to_UTC, filepath, format_localtime):
    """
    Download ERA5 wind data for the specified region and time period.

    Initializes an :class:`ERA5Downloader` with the 10-m wind components,
    downloads the data, and converts it to local time.

    Parameters
    ----------
    wind_info : dict
        Spatial wind domain configuration.  Required keys:

        - ``lon_ll_corner_wind`` : float — western boundary (degrees).
        - ``lat_ll_corner_wind`` : float — southern boundary (degrees).
        - ``nx_wind`` : int — number of grid cells in the x direction.
        - ``ny_wind`` : int — number of grid cells in the y direction.
        - ``dx_wind`` : float — grid spacing in x (degrees).
        - ``dy_wind`` : float — grid spacing in y (degrees).

    ini_date : datetime
        Simulation start date in local time.
    end_date : datetime
        Simulation end date in local time.
    difference_to_UTC : int
        Time difference to UTC in hours for local time conversion.
    filepath : str or Path
        Full path (including filename) where the downloaded file will be saved.
    format_localtime : bool
        If True, formats the time in local time. Defaults to False.
    """
    filepath = Path(filepath)
    ERA5download_obj = ERA5Downloader(
        variables=['10m_u_component_of_wind', '10m_v_component_of_wind'],
        lon_min=wind_info['lon_ll_corner_wind'],
        lon_max=wind_info['lon_ll_corner_wind'] + (wind_info['nx_wind'] * wind_info['dx_wind']),
        lat_min=wind_info['lat_ll_corner_wind'],
        lat_max=wind_info['lat_ll_corner_wind'] + (wind_info['ny_wind'] * wind_info['dy_wind']),
        start_datetime_local=ini_date,
        end_datetime_local=end_date,
        utc_offset_hours=difference_to_UTC,
        output_path=filepath.parent,
        output_filename=filepath.name,
    )
    filepath_utc = ERA5download_obj.download()
    print("\t ERA5 wind data downloaded successfully")

    if format_localtime:
        filepath_localtime = ERA5download_obj.format_to_localtime()
        return filepath_localtime
    return filepath_utc

def download_cmds_winds(wind_info, ini_date, end_date, difference_to_UTC, filepath, format_localtime):
    """
    Download CMDS wind data for the specified region and time period.

    Initializes a :class:`CMDSDownloader` with the 10-m wind components,
    downloads the data, and converts it to local time.

    Parameters
    ----------
    wind_info : dict
        Spatial wind domain configuration.  Required keys:

        - ``lon_ll_corner_wind`` : float — western boundary (degrees).
        - ``lat_ll_corner_wind`` : float — southern boundary (degrees).
        - ``nx_wind`` : int — number of grid cells in the x direction.
        - ``ny_wind`` : int — number of grid cells in the y direction.
        - ``dx_wind`` : float — grid spacing in x (degrees).
        - ``dy_wind`` : float — grid spacing in y (degrees).

    ini_date : datetime
        Simulation start date in local time.
    end_date : datetime
        Simulation end date in local time.
    difference_to_UTC : int
        Time difference to UTC in hours for local time conversion.
    filepath : str or Path
        Full path (including filename) where the downloaded file will be saved.
    format_localtime : bool
        If True, formats the time in local time. Defaults to False.
    """
    filepath = Path(filepath)
    CMDSdownload_obj = CMDSDownloader.for_winds(
        lon_min=wind_info['lon_ll_corner_wind'],
        lon_max=wind_info['lon_ll_corner_wind'] + (wind_info['nx_wind'] * wind_info['dx_wind']),
        lat_min=wind_info['lat_ll_corner_wind'],
        lat_max=wind_info['lat_ll_corner_wind'] + (wind_info['ny_wind'] * wind_info['dy_wind']),
        start_datetime_local=ini_date,
        end_datetime_local=end_date,
        utc_offset_hours=difference_to_UTC,
        output_path=filepath.parent,
        output_filename=filepath.name,
    )
    filepath_utc = CMDSdownload_obj.download()
    print("\t CMDS wind data downloaded successfully")
    if format_localtime:
        filepath_localtime = CMDSdownload_obj.format_to_localtime()
        return filepath_localtime
    return filepath_utc
