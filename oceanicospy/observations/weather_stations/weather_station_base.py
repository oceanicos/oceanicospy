from abc import ABC, abstractmethod

class WeatherStationBase(ABC):
    """
    Abstract base class for weather station data ingestion and processing.

    Provides a common interface for loading, cleaning, and accessing weather
    records from different station models. Subclasses must implement the abstract 
    methods to handle source-specific logic.

    Parameters
    ----------
    filepath : str
        Path to the raw data file for the weather station.

    Notes
    -----
    **Development history**

    - 19-Mar-2025 : initial version - Franklin Ayala
    - 16-Jul-2025 : adding weathersens model - Daniela Rosero
    """
    def __init__(self,filepath: str):
        self.filepath = filepath

    def get_raw_records(self):
        """
        Load and return the raw records from the data source.

        Delegates to the subclass-implemented ``_load_raw_dataframe`` method
        to read data in its original, unprocessed form.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the raw records as read from the source file.
        """
        df = self._load_raw_dataframe()
        return df

    def get_clean_records(self):
        """
        Load and return cleaned records with standardized column names.

        Retrieves raw records via ``get_raw_records``, applies
        column standardization and applies wind direction conversion where required.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by ``date`` (``datetime64[ns]``) with
            standardized column names following the ``variable[unit]``
            convention (e.g. ``air_temp[C]``, ``wind_speed[m/s]``),
            ready for downstream analysis.
        """
        df = self.get_raw_records()
        df = self._standardize_columns(df)
        df = self._compute_direction_degrees(df)
        return df

    @abstractmethod
    def _load_raw_dataframe(self):
        """
        Load the raw DataFrame from the station's data source.

        Subclasses must implement this method to handle the specific file
        format or connection protocol of their weather station model.

        Returns
        -------
        pandas.DataFrame
            The raw, unprocessed data as read from ``self.filepath``.

        Raises
        ------
        FileNotFoundError
            If ``self.filepath`` does not point to a valid file.
        ValueError
            If the file content cannot be parsed into a DataFrame.
        """
        pass

    @abstractmethod
    def _standardize_columns(self, df):
        """
        Rename and reformat columns to a shared standard schema.

        Subclasses must implement this method to map station-specific
        column names and units to the project-wide naming convention.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by ``_load_raw_dataframe``.

        Returns
        -------
        pandas.DataFrame
            DataFrame with standardized column names, dtypes, and index.

        """
        pass

    @abstractmethod
    def _compute_direction_degrees(self, df):
        """
        Compute wind direction in degrees from raw directional data.

        Subclasses must implement this method when the station records wind
        direction as cardinal strings, encoded integers, or any non-degree
        representation that must be converted to decimal degrees (0–360).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing raw wind direction data in a
            station-specific format.

        Returns
        -------
        pandas.DataFrame
            DataFrame with an added or updated column holding wind
            direction values expressed in decimal degrees.

        Notes
        -----
        Implementations should handle ``NaN`` values gracefully and document
        the source column and output column names used.
        """
        pass