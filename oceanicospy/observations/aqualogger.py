import glob
import pandas as pd
import os
from datetime import  timedelta
from scipy.signal import detrend

from oceanicospy.utils import constants

class AQUAlogger():
    """
    A class to handle reading and processing the data files recorded by AQUAlogger. 

    Notes 
    -----
    10-Dec-2024 : Origination - Franklin Ayala

    """
    def __init__(self,directory_path: str,sampling_data: dict) -> None:
        """
        Initializes the AQUAlogger class with the given directory path, sampling data.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the .hdr and .wad files.
        sampling_data : dict
            Dictionary containing the information about the device installation
        """
        self.directory_path = directory_path
        self.sampling_data = sampling_data

    @property
    def first_record_time(self) -> pd.Timestamp:
        filepath = self._get_csv_file()
        df = self._read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')
        return df.index.min()

    @property
    def last_record_time(self) -> pd.Timestamp:
        filepath = self._get_csv_file()
        df = self._read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')
        return df.index.max()
    
    def get_raw_records(self) -> pd.DataFrame:
        """
        Reads the .csv file from the device to create a DataFrame containing data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the raw data indexed by timestamp.
        """

        filepath = self._get_csv_file()
        df = self._read_csv(filepath)
        df = self._parse_dates_and_trim(df)
        return df

    def get_clean_records(self, detrend: bool = True)-> pd.DataFrame:
        """
        Processes the raw data by grouping the series per each burst

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame with bursts identified by a 'burstId' column.
        """

        self.clean_data = self.get_raw_records()
        self.clean_data = self._compute_depth_from_pressure(self.clean_data)

        self.clean_data['burstId'] = (self.clean_data['UNITS'] == 'BURSTSTART').cumsum()
        self.clean_data = self.clean_data.drop(['UNITS'],axis=1)   

        # Compute the surface level
        self.clean_data['eta[m]'] = self.clean_data.groupby('burstId')['eta'].transform(lambda x: x - x.mean())

        if detrend:
            self.clean_data['eta[m]'] = self.clean_data.groupby('burstId')['eta[m]'].transform(lambda x: detrend(x.values, type='linear'))
        return self.clean_data

    def _get_csv_file(self) -> str:
        """Returns the first .csv file found in the directory. Raises if none are found."""

        files = glob.glob(os.path.join(self.directory_path, '*.csv'))
        if not files:
            raise FileNotFoundError("No .csv file found in the specified directory.")
        return files[0]

    def _read_csv(self,filepath:str) -> pd.DataFrame:
        if self.sampling_data.get('temperature', False):
            columns = ['UNITS', 'date', 'Raw1', 'temperature', 'Raw2', 'pressure[bar]', 'Raw3', 'depth[m]', 'nan']
            drop_cols = ['Raw1', 'Raw2', 'Raw3', 'nan']
        else:
            columns = ['UNITS', 'date', 'Raw1', 'pressure[bar]', 'Raw2', 'depth[m]', 'nan']
            drop_cols = ['Raw1', 'Raw2', 'nan']

        df = pd.read_csv(filepath, names=columns, header=21, encoding='latin-1')
        df = df.drop(columns=drop_cols)
        return df

    def _parse_dates_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')

        try:
            start = pd.to_datetime(self.sampling_data['start_time'])
            end = pd.to_datetime(self.sampling_data['end_time'])
        except KeyError:
            raise KeyError("Missing 'start_time' or 'end_time' in sampling_data.")
        except Exception as e:
            raise ValueError(f"Invalid time format in 'sampling_data': {e}")

        return df[start:end]

    def _compute_depth_from_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        df['depth_aux[m]'] = ((df['pressure[bar]'] - constants.ATM_PRESSURE_BAR) * 1e5) / (constants.WATER_DENSITY * constants.GRAVITY)

        if (df['depth_aux[m]'] - df['depth[m]']).abs().max() <= 0.1:
            df = df.drop(columns=['depth_aux[m]'])
        return df