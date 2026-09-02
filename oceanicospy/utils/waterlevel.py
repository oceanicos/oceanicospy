from pathlib import Path

import pandas as pd

from ..downloads import UHSLCDownloader


def download_uhslc_waterlevel(station_id, ini_date, end_date, filepath, utc_offset_hours=-5):
    """
    Download UHSLC hourly sea-level data and return the cleaned DataFrame.

    Instantiates a :class:`~oceanicospy.downloads.UHSLCDownloader`, fetches
    the CSV from the UHSLC server, writes it to *filepath*, and returns the
    cleaned series.

    Parameters
    ----------
    station_id : str
        UHSLC station code (e.g. ``"057"``).
    ini_date : datetime
        Simulation start date in local time.
    end_date : datetime
        Simulation end date in local time.
    filepath : str or Path
        Full path (including filename) where the raw CSV will be saved.
    utc_offset_hours : float, optional
        Local-time offset from UTC in hours (local minus UTC).
        Defaults to ``-5``.

    Returns
    -------
    pandas.DataFrame
        Cleaned water-level DataFrame with a datetime index and a
        ``depth[m]`` column.
    """
    filepath = Path(filepath)
    downloader = UHSLCDownloader(
        station_id=station_id,
        output_path=filepath.parent,
        output_filename=filepath.name,
        start_datetime_local=ini_date,
        end_datetime_local=end_date,
        utc_offset_hours=utc_offset_hours,
    )
    downloader.download()
    df_clean = downloader.clean_data()
    print('\t UHSLC water level data was successfully downloaded')
    return df_clean


def load_uhslc_waterlevel(station_id, filepath):
    """
    Load and clean an already-downloaded UHSLC CSV file.

    Parameters
    ----------
    station_id : str
        UHSLC station code (e.g. ``"057"``).
    filepath : str or Path
        Full path to the previously downloaded ``h<station_id>.csv`` file.

    Returns
    -------
    pandas.DataFrame
        Cleaned water-level DataFrame with a datetime index and a
        ``depth[m]`` column.
    """
    filepath = Path(filepath)
    reader = UHSLCDownloader(
        station_id=station_id,
        output_path=filepath.parent,
        output_filename=filepath.name,
    )
    reader.last_result_path = filepath
    return reader.clean_data()
