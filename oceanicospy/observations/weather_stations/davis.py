import numpy as np
import pandas as pd
from .weather_station_base import WeatherStationBase

pd.set_option('future.no_silent_downcasting', True)

class DavisVantagePro(WeatherStationBase):
    """
    A sensor-specific reader for Davis Vantage Pro weather station files.

    Inherits from :class:`BaseWeatherStation` and implements file parsing methods
    specific to the Davis Vantage Pro comma-separated text format, including column
    renaming, unit conversion, and timestamp parsing.

    Parameters
    ----------
    filepath : str
        Path to the file containing the raw Davis Vantage Pro weather station data.

    Notes
    -----
    The raw file format uses single-character tokens for AM/PM (``'a'``/``'p'``)
    and ``'---'`` as a sentinel for missing values. Both are normalized during
    loading and cleaning respectively.
    """

    def _load_raw_dataframe(self):
        """
        Reads weather station records from a specified file and processes the data into a pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame containing the processed weather station data

        Notes
        -----
        - The function assumes that the first two lines of the file are headers or metadata and skips them.
        - ``'a'`` and ``'p'`` tokens in the ``AM/PM`` column are replaced with ``'AM'`` and ``'PM'`` respectively.
        """
        
        with open(self.filepath, 'r') as file:
            data = file.read().split('\n')[2:-1]
        
        processed_data = [(' '.join(line.split())).split(' ') for line in data]
        
        columns = ['Date', 'time', 'AM/PM', 'Out', 'Temp1', 'Temp2', 'Hum', 'Pt.', 'Speed', 'Dir1', 'Run', 'Speed2', 'Dir2', 
                   'Chill', 'Index1', 'Index2', 'Bar', 'Rain', 'Rate', 'D-D1', 'D-D2', 'Temp4', 'Hum2', 'Dew', 'Heat', 'EMC', 
                   'Density', 'Samp', 'Tx', 'Recept', 'Int.']
        
        df = pd.DataFrame(processed_data, columns=columns)
        df['AM/PM'] = df['AM/PM'].replace({'a': 'AM', 'p': 'PM'})
        
        return df

    def _standardize_columns(self, df):
        """
        Clean and standardize the raw DataFrame for analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by ``_load_raw_dataframe``, with
            string-typed columns and ``'---'`` as the missing-value token.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame indexed by ``date`` (``datetime64[ns]``) with
            standardized column names following the ``variable[unit]``
            convention. Only the following columns are retained (when present):
            ``rain[mm]``, ``air_temp[C]``, ``air_humidity[%]``,
            ``pressure[hPa]``, ``wind_speed[m/s]``, ``wind_direction[°]``.

        Notes
        -----
        Timestamp parsing uses the format ``'%m/%d/%y %I:%M %p'``, which
        matches the Davis Vantage Pro 12-hour clock export (e.g.
        ``'01/15/24 02:30 PM'``). Files with a different clock format will
        raise a ``ValueError`` at the ``pd.to_datetime`` call.
        Wind direction is retained as cardinal strings at this stage and
        converted to decimal degrees by ``_compute_direction_degrees``.
        """
        rename_map = {
            'Rain':  'rain[mm]',
            'Temp4':   'air_temp[C]',
            'Hum2':   'air_humidity[%]',
            'Bar':   'pressure[hPa]',
            'Speed': 'wind_speed[m/s]',
            'Dir1':  'wind_direction[°]',
        }

        df.replace('---', np.nan, inplace=True)

        df['date'] = pd.to_datetime(
            df['Date'] + ' ' + df['time'] + ' ' + df['AM/PM'],
            format='%m/%d/%y %I:%M %p'
        )
        df = df.drop(columns=['Date', 'time', 'AM/PM'])
        df = df.set_index('date')

        df = df.rename(columns=rename_map)
        keep = [c for c in rename_map.values() if c in df.columns]
        df = df[keep]

        numeric_cols = [c for c in keep if c != 'wind_direction[°]']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        return df

    def _compute_direction_degrees(self, df):
        """
        Convert cardinal wind direction labels to decimal degrees.

        Maps the string values in the ``wind_direction[°]`` column to their
        corresponding compass bearings using an 8-point rose.
        Any value not present in the mapping (including ``NaN``) produces
        ``NaN`` in the output.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing a ``wind_direction[°]`` column with cardinal
            direction strings (e.g. ``'N'``, ``'SW'``).

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with ``wind_direction[°]`` converted to
            decimal degrees (``float64``), where 0° = North, increasing
            clockwise.
        """
        direction_mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        df['wind_direction[°]'] = df['wind_direction[°]'].map(direction_mapping)
        return df