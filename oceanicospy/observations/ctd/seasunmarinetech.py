import re
import pandas as pd

from .ctd_base import CTDBase


class SeaSunMarineTechCTD(CTDBase):
    """
    Reader for Sea-Sun Marine Technology CTD TOB files (SST SDA format).

    Handles ``.TOB`` files produced by the SST Standard Data Acquisition
    (SDA) software. The file begins with a multi-line ASCII header containing
    the software name, config path, cast datetime, per-channel calibration
    coefficients, the source SRD path, and record count. Column labels are
    embedded as ``;``-prefixed comment lines immediately before the data block.

    Parameters
    ----------
    filepath : str
        Path to the CTD ``.TOB`` file.

    Notes
    -----
    Expected file structure:

    - Line 1: SDA software name and version.
    - Line 2: Project config path (``.SPJ``).
    - Line 3: Cast datetime (e.g. ``Sunday, January 26, 2025 06:12:50 PM``).
    - Blank padding lines.
    - Calibration lines beginning with a three-digit channel index and sensor
      model (e.g. ``001 CTM966 001  P  Press  dbar  ...``).
    - SRD source file path.
    - ``Lines : N`` record count.
    - ``;``-prefixed column name and unit lines.
    - Data rows: fixed-width, space-delimited, leading dataset index column.

    Data columns: Datasets (dropped), Press [dbar], Temp [°C], Cond [mS/cm],
    Turb [FTU], SALIN [PSU], IntD [Excel serial date], IntT [Excel serial
    time].

    The cleaned DataFrame is indexed by ``pressure[dbar]``. Columns follow the
    ``variable[unit]`` convention (e.g. ``temperature[C]``,
    ``conductivity[mS/cm]``).

    24-Mar-2026 : Origination - Franklin Ayala
    """

    _COLUMN_MAP = {
        'Press': 'pressure[dbar]',
        'Temp': 'temperature[C]',
        'Cond': 'conductivity[mS/cm]',
        'Turb': 'turbidity[FTU]',
        'SALIN': 'salinity[PSU]',
        'IntD': 'date_serial',
        'IntT': 'time_serial',
    }

    _EXCEL_ORIGIN = pd.Timestamp('1899-12-30')

    @property
    def cast_time(self) -> pd.Timestamp:
        """
        Timestamp of the cast start, parsed from the file header.

        Returns
        -------
        pandas.Timestamp
            Cast start time, or ``NaT`` if not present in metadata.
        """
        raw = self.metadata.get('cast_datetime')
        if not raw:
            return pd.NaT
        return pd.to_datetime(raw, format='%A, %B %d, %Y %I:%M:%S %p', errors='coerce')

    def _parse_metadata(self) -> dict:
        """
        Parse the TOB header into a metadata dictionary.

        Extracts software name, config path, cast datetime, per-sensor
        calibration lines, SRD source path, and record count.

        Returns
        -------
        dict
            Metadata fields:
            ``'software'``, ``'config_path'``, ``'cast_datetime'``,
            ``'srd_path'``, ``'n_records'``, ``'calibrations'``.
        """
        metadata: dict = {}
        calibrations: list[str] = []

        with open(self.filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        if len(lines) >= 1:
            metadata['software'] = lines[0].strip()
        if len(lines) >= 2:
            metadata['config_path'] = lines[1].strip()
        if len(lines) >= 3:
            metadata['cast_datetime'] = lines[2].strip()

        for line in lines[3:]:
            stripped = line.strip()
            if re.match(r'^\d{3}\s+CTM\d+', stripped):
                calibrations.append(stripped)
            elif '.SRD' in stripped and not stripped.startswith(';'):
                metadata['srd_path'] = stripped
            elif stripped.startswith('Lines'):
                match = re.search(r'Lines\s*:\s*(\d+)', stripped)
                if match:
                    metadata['n_records'] = int(match.group(1))

        metadata['calibrations'] = calibrations
        return metadata

    def _parse_column_names(self) -> list[str]:
        """
        Extract column names from the ``;``-prefixed header comment lines.

        The first ``;`` line that contains ``Datasets`` holds the column names.

        Returns
        -------
        list of str
            Column name tokens in order.
        """
        with open(self.filepath, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith(';') and 'Datasets' in line:
                    return line[1:].split()
        return []

    def _count_header_lines(self) -> int:
        """
        Count the number of header lines before the first data row.

        The data block begins on the line immediately after the last
        ``;``-prefixed comment line.

        Returns
        -------
        int
            Number of lines to skip when reading the data block.
        """
        last_semicolon = 0
        with open(self.filepath, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                if line.startswith(';'):
                    last_semicolon = i
        return last_semicolon + 1

    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Read the data block into a DataFrame.

        Skips all header and comment lines, assigns column names extracted
        from the ``;``-prefixed label lines, and reads the fixed-width
        space-delimited records.

        Returns
        -------
        pandas.DataFrame
            Raw per-sample records with original column names as exported
            by the SDA software.
        """
        columns = self._parse_column_names()
        skiprows = self._count_header_lines()
        return pd.read_csv(
            self.filepath,
            sep=r'\s+',
            skiprows=skiprows,
            header=None,
            names=columns,
            encoding='latin-1',
        )

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to ``variable[unit]`` convention and set pressure index.

        Drops the leading ``Datasets`` index column, renames remaining columns
        using ``_COLUMN_MAP``, and sets ``pressure[dbar]`` as the DataFrame
        index.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by ``_load_raw_dataframe``.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by ``pressure[dbar]`` with standardized column
            names.
        """
        df = df.drop(columns=['Datasets'], errors='ignore')
        df = df.rename(columns=self._COLUMN_MAP)
        df = df.set_index('pressure[dbar]')
        return df
