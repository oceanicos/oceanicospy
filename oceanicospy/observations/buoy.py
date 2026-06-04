import pandas as pd
from datetime import timezone, timedelta

class WaveBuoy():
    """
    Read and process Spotter buoy data exported as CSV files.

    Supports two CSV export formats produced by Sofar Ocean and AQUAlink
    platforms. The source format is detected automatically from column names.
    Both formats are normalized to a common datetime-indexed DataFrame with
    standardized physical-unit column names (e.g. ``hs[m]``, ``tp[s]``).

    Parameters
    ----------
    filepath : str
        Path to the Spotter CSV file. Both Sofar (epoch-based) and AQUAlink
        (ISO 8601 timestamp) exports are accepted.
    hours_from_utc : int, optional
        UTC offset in hours for the target local time zone. Negative values
        are west of UTC, positive values are east. Default is ``-5``
        (Colombia Standard Time, UTC-5).

    Notes
    -----
    The two supported source formats are identified by a sentinel column:

    * **Sofar** – contains an ``'Epoch Time'`` column (Unix timestamp, seconds).
    * **AQUAlink** – contains a ``'timestamp'`` column (ISO 8601 string, UTC).

    **Development history**
    
    - 10-Sep-2025 : Origination - Daniela Rosero
    - 05-Mar-2026 : Refactoring - Franklin Ayala

    """

    def __init__(self, filepath: str, sampling_data: dict, hours_from_utc: int = -5):
        self.filepath = filepath
        self.sampling_data = sampling_data
        self.hours_from_utc = hours_from_utc

    def _detect_source(self, df: pd.DataFrame) -> str:
        """
        Identify the CSV export format from sentinel column names.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame loaded directly from the CSV file.

        Returns
        -------
        str
            ``'sofar'`` when an ``'Epoch Time'`` column is present;
            ``'aqualink'`` when a ``'timestamp'`` column is present.

        Raises
        ------
        ValueError
            If neither ``'Epoch Time'`` nor ``'timestamp'`` is found among
            the DataFrame columns.
        """
        if 'Epoch Time' in df.columns:
            return 'sofar'
        elif 'timestamp' in df.columns:
            return 'aqualink'
        else:
            raise ValueError("CSV file source is not recognized. Expected 'Epoch Time' or 'timestamp' column.")

    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Load ``self.filepath`` into a DataFrame without any transformation.

        Returns
        -------
        pd.DataFrame
            Unmodified contents of the CSV file, with default pandas column
            dtypes and a default integer index.
        """
        df = pd.read_csv(self.filepath)
        return df

    def _process_format_sofar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a Sofar-format CSV export into a datetime-indexed DataFrame.

        Rows with non-numeric ``'Epoch Time'`` values (e.g. header repetitions)
        are dropped. The epoch is converted to naive local datetime using
        ``self.hours_from_utc``, and the resulting ``'date'`` column is set as
        the index in ascending chronological order.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame containing at least an ``'Epoch Time'`` column with
            Unix timestamps in seconds (UTC).

        Returns
        -------
        pd.DataFrame
            DataFrame sorted chronologically and indexed by a tz-naive
            ``datetime64[ns]`` column named ``'date'``.
        """
        df = df[pd.to_numeric(df['Epoch Time'], errors='coerce').notnull()].copy()

        utc_minus_5 = timezone(timedelta(hours=self.hours_from_utc))

        # Convert to datetime in local time
        df['date'] = pd.to_datetime(df['Epoch Time'], unit='s', utc=True) \
                        .dt.tz_convert(utc_minus_5) \
                        .dt.tz_localize(None)

        # Sort chronologically for correct slicing by date
        df = df.sort_values('date')

        # Set index
        df = df.set_index('date')
        return df

    def _process_format_aqualink(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process an AQUAlink-format CSV export into a datetime-indexed DataFrame.

        The ISO 8601 ``'timestamp'`` column (UTC) is converted to naive local
        datetime using ``self.hours_from_utc``. If the original data are in
        descending order (most-recent first, as AQUAlink exports), the rows are
        sorted into ascending chronological order. The original ``'timestamp'``
        column is removed and replaced by a ``'date'`` index.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame containing at least a ``'timestamp'`` column with
            ISO 8601 datetime strings in UTC.

        Returns
        -------
        pd.DataFrame
            DataFrame sorted chronologically and indexed by a tz-naive
            ``datetime64[ns]`` column named ``'date'``. The original
            ``'timestamp'`` column is dropped.
        """
        utc_minus_5 = timezone(timedelta(hours=self.hours_from_utc))

        # Convert ISO timestamp to datetime in local time
        df['date'] = pd.to_datetime(df['timestamp'], utc=True) \
                        .dt.tz_convert(utc_minus_5) \
                        .dt.tz_localize(None)

        # Reverse sort if necessary (most recent first)
        if df['date'].is_monotonic_decreasing:
            df = df.sort_values('date')

        df = df.drop(columns=['timestamp'], errors='ignore')
        # Set index
        df = df.set_index('date')

        return df

    def get_raw_records(self) -> pd.DataFrame:
        """
        Return the unprocessed contents of the CSV file as a DataFrame.

        No column renaming, timezone conversion, or row filtering is applied.
        Use this method to inspect the original data before any transformations.
        For analysis-ready output, prefer :meth:`get_clean_records`.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame with default integer index and original column names
            exactly as they appear in the file.
        """
        df = self._load_raw_dataframe()
        return df
    
    def get_clean_records(self) -> pd.DataFrame:
        """
        Public method to get the cleaned records from the Spotter CSV file, with standardized columns and burst IDs.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame ready for analysis.
        """
        df = self.get_raw_records()
        file_format = self._detect_source(df)

        if file_format == 'sofar':
            df =  self._process_format_sofar(df)
        else:
            df = self._process_format_aqualink(df)

        df = self._standardize_columns(df)
        df = self._parse_dates_and_trim(df)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames and converts Spotter-specific columns to a standardized format.
        Includes wave, wind, and temperature variables from Spotter buoys.
        """

        # Manual rename for known fields from both formats
        rename_map = {
            'Significant Wave Height (m)': 'hs[m]',
            'Peak Period (s)': 'tp[s]',
            'Mean Period (s)': 'tm[s]',
            'Peak Direction (deg)': 'wave_peak_dir[°]',
            'Mean Direction (deg)': 'wave_mean_dir[°]',
            'Peak Directional Spread (deg)': 'wave_peak_dir_spread[°]',
            'Mean Directional Spread (deg)':'wave_mean_dir_spread[°]',
            'significant_wave_height_spotter': 'hs[m]',
            'wave_mean_period_spotter': 'tm[s]',
            'wave_mean_direction_spotter': 'wave_mean_dir[°]',
            'wind_speed_spotter': 'wind_speed[m/s]',
            'wind_direction_spotter': 'wind_dir[°]',
            'top_temperature_spotter': 'top_temp[C]',
            'bottom_temperature_spotter': 'bottom_temp[C]'
        }

        # Apply renaming where applicable
        df = df.rename(columns=rename_map)

        # Detect all columns that contain "spotter" (renamed or not) for conversion
        parameter_cols = rename_map.values()  # Get the standardized column names

        for col in parameter_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaNs in the core spotter wave variables, if they exist
        key_wave_vars = ['hs[m]', 'tp[s]', 'wave_mean_dir[°]']
        vars_present = [v for v in key_wave_vars if v in df.columns]
        if vars_present:
            df = df.dropna(subset=vars_present, how='any')

        return df

    def _parse_dates_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trims the DataFrame to the time range specified in sampling_data.
        """
        start = pd.to_datetime(self.sampling_data['start_time'])
        end = pd.to_datetime(self.sampling_data['end_time'])

        df_trimmed = df[start:end]

        if df_trimmed.empty:
            raise ValueError(
                f"No data found in the range {start} to {end}. "
                f"Available data is from {df.index.min()} to {df.index.max()}"
            )

        return df_trimmed