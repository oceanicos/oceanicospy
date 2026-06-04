import pandas as pd
import glob
import os

from .pressure_sensor_base import BaseLogger

class RBR(BaseLogger):
    """
    A sensor-specific reader for RBR pressure logger files.

    Inherits from :class:`BaseLogger` and implements file parsing methods
    specific to the RBR tab/comma-separated text format, including column
    renaming, unit conversion, and hour-based burst identification.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the ``*_data.txt`` files.
    sampling_data : dict
        Dictionary containing information on device installation, including:
        
        - ``anchoring_depth``: Depth at which the device is anchored (in meters).
        - ``sensor_height``: Height of the sensor above the seabed (in meters).
        - ``sampling_freq``: Sampling frequency of the device (in Hz).
        - ``burst_length_s``: Length of each burst in seconds.
        - ``start_time``: Start time for filtering records (as a datetime object).
        - ``end_time``: End time for filtering records (as a datetime object).
    filename : str, optional
        Specific filename to read within the directory. If not provided and multiple files are found, an error will be raised.

    Notes
    -----
    Only files matching the ``*_data.txt`` pattern are recognized. Pressure
    values are converted from dbar (as exported by RBR instruments) to bar
    during column standardization.
    """

    @property
    def _file_pattern(self):
        return '*_data.txt'

    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Reads the RBR ``*_data.txt`` file into a raw DataFrame.

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame with columns as exported by the RBR instrument,
            with no index or unit conversion applied at this stage.

        Raises
        ------
        FileNotFoundError
            If the records file cannot be located via :meth:`_get_records_file`.
        """

        filepath = self._get_records_file()
        df = pd.read_csv(filepath)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names, units, and sets the datetime index.

        Drops the ``Sea pressure`` column if present, renames ``Pressure``
        and ``Depth`` to follow the ``variable[unit]`` convention, converts
        pressure from dbar to bar, and sets the ``Time`` column as a
        datetime index.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by :meth:`_load_raw_dataframe`.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by datetime with standardized column names
            and pressure values expressed in bar.
        """
        df = df.drop(columns=['Sea pressure'], errors='ignore')
        df = df.rename(columns={'Pressure': 'pressure[bar]', 'Depth': 'depth[m]'})
        df['pressure[bar]'] = df['pressure[bar]'] / 10  # dbar to bar
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.rename(columns={'Time': 'date'})
        df = df.set_index('date')
        return df

    def _assign_burst_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a burst identifier to each record by flooring timestamps to the hour.

        Unlike marker-based column in AQUAlogger files, RBR files do not contain explicit burst
        delimiters. Burst boundaries are instead inferred by grouping records
        that share the same floored hour, with IDs starting at 1.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with a datetime index.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a ``burstId`` integer column added, where each
            unique floored hour maps to a sequential burst number starting at 1.
        
        Notes
        -----
        Assuming that bursts are defined by hour, that is the sea state is expected to change at least every hour.
        """
        
        df['burstId'] = pd.factorize(df.index.floor('h'))[0] + 1
        return df
