import glob
import os
import pandas as pd

from .ctd_base import CTDBase

class CastawayCTD(CTDBase):
    """
    Reader for CastAway CTD per-cast files (YSI/Xylem format).

    Supports two file variants:

    - **With embedded header** (``has_header=True``): the file begins with a
      ``%``-prefixed metadata block (device ID, cast time, GPS coordinates,
      calibration dates, etc.) followed by a column header row and data rows.
      Metadata is parsed directly from the file.

    - **Without embedded header** (``has_header=False``, default): the file
      contains only a column header row and data rows. Cast metadata is read
      from a multi-cast summary CSV located in the same directory, matched by
      file stem against the ``File name`` column.

    Parameters
    ----------
    filepath : str
        Path to the per-cast CastAway CTD ``.csv`` file.
    has_header : bool, optional
        ``True`` if the file contains a ``%``-prefixed metadata block.
        ``False`` (default) to read metadata from the auxiliary summary CSV.

    Notes
    -----
    Expected per-cast file structure (``has_header=False``):

    - First row is the column header (no ``%`` metadata block).

    Expected per-cast file structure (``has_header=True``):

    - Lines beginning with ``%`` contain metadata as ``% Key,Value`` pairs.
    - An empty ``%`` line acts as a separator before the data block.
    - The first non-``%`` line is the column header row.

    Data columns (both variants): Pressure (Decibar), Depth (Meter),
    Temperature (Celsius), Conductivity (MicroSiemens per Centimeter),
    Specific conductance (MicroSiemens per Centimeter), Salinity (Practical
    Salinity Scale), Sound velocity (Meters per Second), Density (Kilograms
    per Cubic Meter).

    The auxiliary summary file (``has_header=False``) must have a ``File name``
    column whose values match per-cast file stems, and cast time columns in
    Excel serial date format (days since 1899-12-30).

    The cleaned DataFrame is indexed by ``depth[m]``, reflecting the vertical
    profile nature of a CTD cast.
    """

    _COLUMN_MAP = {
        'Pressure (Decibar)': 'pressure[dbar]',
        'Depth (Meter)': 'depth[m]',
        'Temperature (Celsius)': 'temperature[C]',
        'Conductivity (MicroSiemens per Centimeter)': 'conductivity[uS/cm]',
        'Specific conductance (MicroSiemens per Centimeter)': 'specific_conductance[uS/cm]',
        'Salinity (Practical Salinity Scale)': 'salinity[PSS]',
        'Sound velocity (Meters per Second)': 'sound_velocity[m/s]',
        'Density (Kilograms per Cubic Meter)': 'density[kg/m3]',
    }

    _EXCEL_ORIGIN = pd.Timestamp('1899-12-30')

    def __init__(self, filepath: str, has_header: bool = False):
        super().__init__(filepath)
        self.has_header = has_header

    @property
    def cast_time(self) -> pd.Timestamp:
        """
        UTC timestamp of the cast start.

        When ``has_header=True`` the value is an ISO datetime string read
        directly from the embedded metadata block.  When ``has_header=False``
        the value is an Excel serial date (days since 1899-12-30) read from
        the auxiliary summary file.

        Returns
        -------
        pandas.Timestamp
            Cast start time in UTC, or ``NaT`` if no matching value is found.
        """
        serial = self.metadata.get('Cast time (UTC)')
        if serial is None:
            return pd.NaT
        if self.has_header:
            return pd.to_datetime(serial, errors='coerce')
        return self._EXCEL_ORIGIN + pd.to_timedelta(float(serial), unit='D')

    def _parse_metadata(self) -> dict:
        """
        Parse cast metadata from the file or from a summary CSV.

        When ``has_header=True``, reads all ``%``-prefixed lines in
        ``self.filepath`` and returns key-value pairs extracted from
        ``% Key,Value`` entries.

        When ``has_header=False``, scans every CSV in the same directory
        (excluding ``self.filepath``) for one that contains a ``File name``
        column, then returns the row whose ``File name`` matches the stem of
        ``self.filepath``.

        Returns
        -------
        dict
            Metadata fields (e.g. ``'Device'``, ``'Cast time (UTC)'``,
            ``'Start latitude'``, ``'Cast duration (Seconds)'``).
            Returns an empty dict if no metadata can be found.
        """
        if self.has_header:
            return self._parse_metadata_from_file()
        return self._parse_metadata_from_summary()

    def _parse_metadata_from_file(self) -> dict:
        """Read ``%``-prefixed header lines from the cast file itself."""
        metadata = {}
        with open(self.filepath, 'r') as f:
            for line in f:
                if not line.startswith('%'):
                    break
                content = line[1:].strip()
                if ',' in content:
                    key, value = content.split(',', 1)
                    metadata[key.strip()] = value.strip()
        return metadata

    def _parse_metadata_from_summary(self) -> dict:
        """Read cast metadata from a multi-cast summary CSV in the same directory."""
        directory = os.path.dirname(self.filepath)
        stem = os.path.splitext(os.path.basename(self.filepath))[0]

        for csv_path in glob.glob(os.path.join(directory, '*.csv')):
            if os.path.abspath(csv_path) == os.path.abspath(self.filepath):
                continue
            try:
                header = pd.read_csv(csv_path, nrows=0)
                if 'File name' not in header.columns:
                    continue
                summary = pd.read_csv(csv_path)
                match = summary[summary['File name'] == stem]
                if not match.empty:
                    return match.iloc[0].to_dict()
            except Exception:
                continue

        return {}

    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Load the raw data block into a DataFrame.

        When ``has_header=True``, counts the ``%``-prefixed lines and skips
        them before reading the CSV data block.  When ``has_header=False``,
        reads the file directly (first row is the column header).

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame with original column names and all data rows.
        """
        if self.has_header:
            with open(self.filepath, 'r') as f:
                header_lines = sum(1 for line in f if line.startswith('%'))
            return pd.read_csv(self.filepath, skiprows=header_lines)
        return pd.read_csv(self.filepath)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to ``variable[unit]`` convention and set depth index.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by ``_load_raw_dataframe``.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by ``depth[m]`` with standardized column names.
        """
        df = df.rename(columns=self._COLUMN_MAP)
        df = df.set_index('depth[m]')
        return df
