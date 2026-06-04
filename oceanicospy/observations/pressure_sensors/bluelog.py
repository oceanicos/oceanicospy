from datetime import datetime
import glob
import os
import pandas as pd

from .pressure_sensor_base import BaseLogger

class Bluelog(BaseLogger):
    """
    A sensor-specific reader for Bluelog pressure logger files.

    Inherits from :class:`BaseLogger` and implements file parsing methods
    specific to the Bluelog CSV format, which includes a comment-style
    metadata header terminated by a ``---`` separator, followed by
    comma-separated data records.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the Bluelog ``.csv`` file.
    sampling_data : dict
        Dictionary containing deployment metadata. Must include:
        
        - ``anchoring_depth`` (float): Depth at which the sensor is anchored.
        - ``sensor_height`` (float): Height of the sensor above the bottom.
        - ``sampling_freq`` (float): Sampling frequency of the measurements.
        - ``burst_length_s`` (float): Duration of each burst in seconds.
    filename : str, optional
        Name of the target ``.csv`` file. If not provided, the first
        ``.csv`` file found in ``directory_path`` is used.

    Notes
    -----
    The Bluelog CSV file contains a metadata block at the top, including
    a ``# Configured Start Time:`` entry formatted as ``%Y%m%d%H%M``,
    followed by a ``---`` separator that marks the beginning of the data
    section. Pressure values are exported in dbar and converted to bar
    during column standardization.
    """

    @property
    def _file_pattern(self):
        return '*.csv'
    
    def _load_raw_dataframe(self):
        """
        Reads the Bluelog ``.csv`` file into a raw DataFrame.

        Scans the file line by line to extract the configured start time
        from the ``# Configured Start Time:`` metadata entry, then locates
        the ``---`` separator to determine where the data section begins.
        The data is then loaded into a DataFrame starting from that line.

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame with columns as exported by the Bluelog instrument,
            with no index or unit conversion applied at this stage.

        Raises
        ------
        FileNotFoundError
            If the records file cannot be located via :meth:`_get_records_file`.
        ValueError
            If the ``---`` data separator is not found in the file.
        UnboundLocalError
            If the ``---`` separator is found but the configured start time
            line has not been encountered before it.
        """
        filepath = self._get_records_file()
        # Read file content line by line
        with open(filepath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                # Find the '---' line that separates header from data
                if line.strip() == "---":
                    line_header_number = lineno + 1
                    break

        if line_header_number is None:
            raise ValueError("Configured start time or data section not properly found.")

        # Load the data into a DataFrame, skipping lines before the header
        df = pd.read_csv(filepath, skiprows=line_header_number-1)
        return df

    def _standardize_columns(self, df):
        """
        Standardizes column names, units, and sets the datetime index.

        Normalizes column names to lowercase, replaces parentheses with
        brackets to follow the ``variable[unit]`` convention, renames the
        timestamp column to ``date``, converts pressure from dbar to bar,
        and sets ``date`` as a datetime index.

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
        new_columns = {}
        
        for col in df.columns:
            col_lower = col.lower()
            col_lower = col_lower.replace("(", "[")  
            col_lower = col_lower.replace(")", "]")  
            col_lower = col_lower.strip() # Remove spaces for easier matching

            if 'timestamp' in col_lower:
                new_columns[col] = 'date'
            # Rename pressure[dbar] → pressure [bar]
            elif 'pressure' in col_lower and 'dbar' in col_lower:
                new_columns[col] = 'pressure[bar]'
            else:
                new_columns[col] = col_lower  # keep as is
        
        # Apply renaming
        df = df.rename(columns=new_columns)
        
        # Convert units if pressure column exists
        if 'pressure[bar]' in df.columns:
            df['pressure[bar]'] = df['pressure[bar]'] / 10.0
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')
        return df
    
    def _assign_burst_id(self, df):
        """
        Assigns a burst identifier to each record by flooring timestamps to the hour.

        Burst boundaries are inferred by grouping records that share the same
        floored hour, with IDs starting at 1. This is the same strategy used
        by :class:`RBR`, as Bluelog files also lack explicit burst markers.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with a datetime index.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a ``burstId`` integer column added, where each
            unique floored hour maps to a sequential burst number starting at 1.
        """
        df['burstId'] = pd.factorize(df.index.floor('h'))[0] + 1
        return df

