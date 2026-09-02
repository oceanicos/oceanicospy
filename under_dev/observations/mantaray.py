import pandas as pd
import glob
import os

from .pressure_sensor_base import BaseLogger

class Mantaray(BaseLogger):
    """
    A class to handle reading and processing Mantaray ADCP data from .txt files.
    Standardized similarly to RBR and BlueLog.

    Notes
    -----
    16-Jul-2025 : Created - Daniela Rosero
    """

    def _get_records_file(self):
        """
        Finds the first .txt file in the directory.

        Returns
        -------
        str : Path to the first .txt file found.
        """
        files = glob.glob(os.path.join(self.directory_path, '*.txt'))
        if not files:
            raise FileNotFoundError("No .txt file found in the specified directory.")
        return files[0]

    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Loads the raw Mantaray data into a pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            Raw data from the Mantaray file.
        """
        filepath = self._get_records_file()
        df = pd.read_csv(
            filepath,
            sep=',',
            encoding='latin1',   # ⬅️ Cambiar de utf-8 a latin1
            skiprows=2,
            skip_blank_lines=True,
            on_bad_lines='skip'
        )

        return df


    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names and units to be consistent with other sensors.

        Returns
        -------
        df : pandas.DataFrame
            Standardized dataframe with datetime and renamed columns.
        """
        # Rename columns to standard English format
        column_map = {
            'Date and Time': 'Time',
            'Level (m)': 'level[m]',
            'Velocity (m/s)': 'velocity[m/s]',
            'Temperature (°C)': 'temperature[C]',
            'Flow (l/s)': 'flow[l/s]'
        }
        df.rename(columns=column_map, inplace=True)

        # Convert Time to datetime
        df['Time'] = pd.to_datetime(df['Time'], dayfirst=True, errors='coerce')

        return df

    def _assign_burst_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a burstId based on hourly intervals.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with a new column 'burstId'.
        """
        df['burstId'] = pd.factorize(df['Time'].dt.floor('h'))[0] + 1
        return df
