from abc import ABC, abstractmethod
import numpy as np
import os
import glob as glob
import pandas as pd
from scipy.signal import detrend

from oceanicospy.utils import constants

class BaseLogger(ABC):
    """
    Abstract base class for field pressure sensor data loggers.

    Provides a common interface for reading, standardising, and burst-segmenting
    raw pressure records collected by submerged oceanographic instruments in the
    field.  Subclasses implement the instrument-specific file parsing by
    overriding :meth:`_load_raw_dataframe`, :meth:`_standardize_columns`, and
    :meth:`_assign_burst_id`.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the pressure sensor data files.
    sampling_data : dict
        Dictionary containing information on device installation, including:
        
        - ``anchoring_depth``: The depth at which the sensor is anchored (in meters).
        - ``sensor_height``: The height of the sensor above the anchoring point (in meters).
        - ``sampling_freq``: The frequency at which the sensor records data (in Hz).
        - ``burst_length_s``: The length of each burst of data recording (in seconds).
        - ``start_time``: The start time for data analysis (as a datetime object or string).
        - ``end_time``: The end time for data analysis (as a datetime object or string).
    filename: str, optional
        The name of the file containing the records. If not provided, the first record file found in the directory will be used.

    Notes
    -----
    **Development history**

    - 23-Sep-2025 : Origination - Franklin Ayala
    - 12-Oct-2025 : Adding AQUAlogger logic - Juan Diego Toro
    - 10-Dec-2025 : Bluelog support - Daniela Rosero 
    
    """
    def __init__(self, directory_path: str, sampling_data: dict, filename: str = None):

        self.directory_path = directory_path
        self.sampling_data = sampling_data
        self.filename = filename

    @property
    def _file_pattern(self):
        """
        Glob pattern used to locate the records file in ``directory_path``.

        Returns 
        -------
        str
            A glob-compatible pattern string (e.g., ``*.csv``, ``*_data.txt``).

        Notes
        -----
        Subclasses must override this property to define the file pattern
        specific to their instrument format.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must define '_file_pattern'.")

    @property
    def first_record_time(self):
        """
        Returns the timestamp of the first record in the dataset.

        Returns
        -------
        pandas.Timestamp
            The timestamp of the first available record.
        """
        df = self._load_raw_dataframe()
        df = self._standardize_columns(df)
        return pd.to_datetime(df.index.values[0])
    
    @property
    def last_record_time(self):
        """
        Returns the timestamp of the last record in the dataset.

        Returns
        -------
        pandas.Timestamp
            The timestamp of the last available record.
        """
        df = self._load_raw_dataframe()
        df = self._standardize_columns(df)
        return pd.to_datetime(df.index.values[-1])

    def _get_records_file(self) -> str:
        """
        Locates and returns the path to the records file in ``directory_path``.

        Uses the glob pattern defined by :attr:`_file_pattern` to search for
        matching files. If a single file is found it is returned directly. If
        multiple files are found, ``filename`` must be specified to disambiguate.

        Returns
        -------
        str
            Absolute path to the records file.

        Raises
        ------
        FileNotFoundError
            If no files matching :attr:`_file_pattern` are found in
            ``directory_path``, or if ``filename`` is specified but does
            not match any file found.
        ValueError
            If multiple matching files are found and ``filename`` is not
            specified to disambiguate.
        """
        extension = os.path.splitext(self._file_pattern)[-1]
        files = glob.glob(os.path.join(self.directory_path, self._file_pattern))
        if not files:
            raise FileNotFoundError(f"No '{self._file_pattern}' file found in the specified directory.")
        if len(files) == 1:
            return files[0]
        if not self.filename:
            raise ValueError(f"Multiple '{self._file_pattern}' files found. Please specify the filename to use.")
        if not self.filename.endswith(extension):
            self.filename += extension
        matching_files = [f for f in files if os.path.basename(f) == self.filename]
        if not matching_files:
            raise FileNotFoundError(f"No file named '{self.filename}' found in the directory.")
        return matching_files[0]

    def get_raw_records(self) -> pd.DataFrame:
        """
        Reads the records file from the device to create a DataFrame containing data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the raw data indexed by timestamp.
        
        Notes
        -----
        The records are standardized to have a datetime index and columns following the convention variable[units] (e.g., pressure[bar], depth[m]).
        The data is trimmed to include only records between the start and end record times provided.
        """
        df = self._load_raw_dataframe()
        return df

    def get_clean_records(self, detrended: bool = True) -> pd.DataFrame:
        """
        Processes the raw data into clean, burst-segmented data with surface elevation.

        Parameters
        ----------
        detrended : bool, optional
            If True, applies a linear detrending to the depth data within each burst. Default is ``True``.

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame with bursts identified by a ``'burstId'`` column.
        """

        df = self.get_raw_records()
        df = self._standardize_columns(df)
        df = self._parse_dates_and_trim(df)
        df = self._compute_depth_from_pressure(df)
        df = self._assign_burst_id(df)
        df['eta[m]'] = df.groupby('burstId')['depth[m]'].transform(lambda x: x - x.mean())

        if detrended:
            df['eta[m]'] = df.groupby('burstId')['eta[m]'].transform(lambda x: detrend(x.values, type='linear'))

        return df

    def _parse_dates_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses the start and end timmes to datetime and trims the DataFrame to include only records between the first and last submerged record times.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be processed.
        
        Returns
        -------
        pandas.DataFrame
            The processed DataFrame with datetime index and trimmed to submerged records.
            
        Raises
        ------
        ValueError
            If the time format in 'sampling_data' is invalid.
        """

        self.start_dt = self.sampling_data.get('start_time',None)
        self.end_dt = self.sampling_data.get('end_time',None)

        if self.start_dt is None and self.end_dt is None:
            return df
        try:
            start = pd.to_datetime(self.start_dt)
            end = pd.to_datetime(self.end_dt)
        except KeyError as e:
            raise ValueError(f"Missing start or end time in 'sampling_data': {e}")
        except Exception as e:
            raise ValueError(f"Invalid time format in 'sampling_data' and cannot be parsed as datetime: {e}")

        return df[start:end]

    def _compute_depth_from_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes depth from pressure using the hydrostatic formula

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the pressure data.
        
        Returns
        -------
        pandas.DataFrame
            The DataFrame with an added 'depth[m]' column computed from pressure if it doesn't exist, and with an 
            auxiliary 'depth_aux[m]' column if 'depth[m]' already exists for comparison and their difference is over 0.1 m.

        Notes
        -----
        The depth is computed using the formula: $depth = (pressure - atmospheric pressure) / (density * gravity)$
        """

        depth_computed = ((df['pressure[bar]'] - constants.ATM_PRESSURE_BAR) * 1e5) / (constants.WATER_DENSITY * constants.GRAVITY)
        if 'depth[m]' not in df.columns:
            df['depth[m]'] = depth_computed
        else:
            df['depth_aux[m]'] = depth_computed     
            if (df['depth_aux[m]'] - df['depth[m]']).abs().max() <= 0.1:
                df = df.drop(columns=['depth_aux[m]'])
        return df

    @abstractmethod
    def _load_raw_dataframe(self) -> pd.DataFrame:
        """Abstract method to be implemented by subclasses to specify how to read the records file into a raw DataFrame.

        Returns
        -------
        pandas.DataFrame
            The raw DataFrame read from the records file.
        """
        pass

    @abstractmethod
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to be implemented by subclasses to specify how to standardize the columns of the raw DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The raw DataFrame to be standardized.
        
        Returns
        -------
        pandas.DataFrame
            The standardized DataFrame with a datetime index and columns following the convention variable[units].

        
        Notes
        -----
        Each column is expected to be renamed to follow the convention variable[units], 
        where 'variable' is a lowercase name of the measured variable (e.g., pressure, depth) and 
        'units' is the corresponding unit of measurement (e.g., bar, m). 
        The index of the DataFrame should be set to a datetime index based on the timestamp column in the raw data.
        """
        pass

    @abstractmethod
    def _assign_burst_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to be implemented by subclasses to specify how to assign burst IDs to the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to which burst IDs will be assigned.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with an added 'burstId' column that identifies the burst to which each record belongs.
        """
        pass