from abc import ABC, abstractmethod
import pandas as pd


class CTDBase(ABC):
    """
    Abstract base class for CTD (Conductivity, Temperature, Depth) devices.

    Provides a common interface for loading and processing CTD cast data from
    different instrument formats. Metadata is read directly from the data file,
    requiring only the filepath as a constructor parameter.

    Parameters
    ----------
    filepath : str
        Path to the CTD data file.

    Notes
    -----
    **Development history**

    24-Mar-2026 : Origination - Franklin Ayala
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    @property
    def metadata(self) -> dict:
        """
        Parsed metadata extracted from the data file.

        Returns
        -------
        dict
            Key-value pairs of metadata fields as found in the file.
        """
        return self._parse_metadata()

    def get_raw_records(self) -> pd.DataFrame:
        """
        Load and return the raw records from the data file.

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame as read from the file, without column standardization.
        """
        return self._load_raw_dataframe()

    def get_clean_records(self) -> pd.DataFrame:
        """
        Load and return cleaned records with standardized column names and units.

        Delegates to ``_load_raw_dataframe`` and ``_standardize_columns``.
        Columns follow the ``variable[unit]`` naming convention
        (e.g., ``temperature[C]``, ``pressure[dbar]``).

        Returns
        -------
        pandas.DataFrame
            DataFrame with standardized column names and units.
        """
        df = self._load_raw_dataframe()
        return self._standardize_columns(df)

    @abstractmethod
    def _parse_metadata(self) -> dict:
        """
        Parse and return metadata from the data file.

        Subclasses must implement this method to extract instrument-specific
        header information (e.g., device ID, cast time, GPS coordinates,
        calibration dates).

        Returns
        -------
        dict
            Dictionary of metadata key-value pairs extracted from the file.
        """
        pass

    @abstractmethod
    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Read the data file into a raw DataFrame.

        Subclasses must implement this method to handle the specific file
        format of their CTD instrument.

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame with original column names and values as read
            from the file.

        Raises
        ------
        FileNotFoundError
            If ``self.filepath`` does not point to a valid file.
        ValueError
            If the file content cannot be parsed into a DataFrame.
        """
        pass

    @abstractmethod
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename and reformat columns to the project-wide standard schema.

        Subclasses must implement this method to map instrument-specific
        column names and units to the ``variable[unit]`` convention.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by ``_load_raw_dataframe``.

        Returns
        -------
        pandas.DataFrame
            DataFrame with standardized column names following the
            ``variable[unit]`` convention (e.g., ``temperature[C]``,
            ``pressure[dbar]``, ``salinity[PSS]``).
        """
        pass
