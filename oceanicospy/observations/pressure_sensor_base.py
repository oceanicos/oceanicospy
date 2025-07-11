from abc import ABC, abstractmethod
import pandas as pd
from scipy.signal import detrend
from oceanicospy.utils import constants


class BaseLogger(ABC):
    """
    A class to handle reading and processing the data files recorded by any pressure sensor. 

    Notes 
    -----
    10-Dec-2024 : Origination - Franklin Ayala

    """
    def __init__(self, directory_path: str, sampling_data: dict):
        """
        Initializes the BaseLogger class with the given directory path, sampling data.

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
        return self.get_raw_records().index.min()

    @property
    def last_record_time(self) -> pd.Timestamp:
        try:
            time=self._load_raw_dataframe()['date']
        except:
            time=self._load_raw_dataframe()['Time']

        return time.values[-1]

    def get_raw_records(self) -> pd.DataFrame:
        """
        Reads the records file from the device to create a DataFrame containing data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the raw data indexed by timestamp.
        """
        df = self._load_raw_dataframe()
        df = self._parse_dates_and_trim(df)
        df = self._standardize_columns(df)
        return df

    def get_clean_records(self, detrended: bool = True) -> pd.DataFrame:
        """
        Processes the raw data by grouping the series per each burst

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame with bursts identified by a 'burstId' column.
        """

        df = self.get_raw_records()
        df = self._compute_depth_from_pressure(df)
        df = self._assign_burst_id(df)
        df['eta[m]'] = df.groupby('burstId')['depth[m]'].transform(lambda x: x - x.mean())

        if detrended:
            df['eta[m]'] = df.groupby('burstId')['eta[m]'].transform(lambda x: detrend(x.values, type='linear'))

        return df

    def _parse_dates_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.rename(columns={'Time': 'date'})
        
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

    @abstractmethod
    def _get_records_file(self) -> pd.DataFrame:
        """Returns the first records file found in the directory. Raises if none are found."""
        pass

    @abstractmethod
    def _load_raw_dataframe(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _assign_burst_id(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
