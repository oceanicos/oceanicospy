import glob
import pandas as pd
import os

from .pressure_sensor_base import BaseLogger

class AQUAlogger(BaseLogger):
    """
    A sensor-specific reader for AQUAlogger files.

    Inherits from ``BaseLogger`` and implements methods specific to AQUAlogger file formats.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the .csv files.
    sampling_data : dict
        Dictionary containing information on device deployment, including:
        
        - ``anchoring_depth``: The depth at which the sensor is anchored (in meters).
        - ``sensor_height``: The height of the sensor above the anchoring point (in meters).
        - ``sampling_freq``: The frequency at which the sensor records data (in Hz).
        - ``burst_length_s``: The length of each burst of data recording (in seconds).
        - ``start_time``: The start time for data analysis (as a datetime object or string).
        - ``end_time``: The end time for data analysis (as a datetime object or string).

    Notes
    -----
    The following models are supported: AQUAlogger520PT5 and AQUAlogger520P4 

    """

    @property
    def _file_pattern(self):
        return '*.csv'        
    
    def _load_raw_dataframe(self):
        """
        Reads the AQUAlogger ``.csv`` file into a raw DataFrame.

        Parses the ``HEADING`` and units rows to construct column names
        following the ``variable[unit]`` convention. Columns without a
        recognized unit are labeled as ``variable[raw]``.

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame with column names derived from the header and
            units rows. No index is set at this stage.

        Raises
        ------
        FileNotFoundError
            If the records file cannot be located via :meth:`_get_records_file`.
        StopIteration
            If no ``HEADING`` row is found in the file.
        """

        filepath = self._get_records_file()
        with open(filepath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if line.startswith("HEADING"):
                    line_header_number = lineno
                    line_units = next(f)
                    break
    
        header = line.strip().split(",")
        units = line_units.strip().split(",")

        header_names = [h.strip().lower() for h in header]
        units_names = [u.strip().lower() for u in units]

        # Playing with the header and units to create column names
        col_names = []
        for i in range(len(header_names)):
            if units_names[i] == 'units':
                col_name = 'units'
            elif units_names[i] == 'timecode':
                col_name = 'date'
            elif units_names[i] == 'raw':
                col_name = header_names[i]+f'[{units_names[i]}]'
            elif header_names[i] == '':
                col_name = header_names[i-1]+f'[{units_names[i]}]'
            else:
                pass
            col_names.append(col_name)
        
        df = pd.read_csv(filepath, names=col_names, header=line_header_number, encoding='latin-1')
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names and sets the datetime index.

        Drops columns containing raw or unitless data, coerces the ``date``
        column to datetime, and sets it as the DataFrame index.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by :meth:`_load_raw_dataframe`.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by datetime, retaining only columns with
            recognized physical units (e.g., ``pressure[bar]``).
        """        
        raw_columns = [col for col in df.columns if 'raw' in col.lower() or '[]' in col]
        df = df.drop(columns=raw_columns)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')        
        df = df.set_index('date')
        return df

    def _assign_burst_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a burst identifier to each record based on ``BURSTSTART`` markers.

        Uses the cumulative count of ``BURSTSTART`` entries in the ``units``
        column to label each record with its corresponding burst number.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing a ``units`` column with ``BURSTSTART`` markers.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a ``burstId`` integer column added and the
            ``units`` column removed.
        """
        df['burstId'] = (df['units'] == 'BURSTSTART').cumsum()
        return df.drop(columns=['units'])