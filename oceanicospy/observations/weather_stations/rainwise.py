import pandas as pd
from .weather_station_base import WeatherStationBase

class Rainwise(WeatherStationBase):
    """
    A sensor-specific reader for Rainwise weather station files.

    Inherits from :class:`WeatherStationBase` and implements file parsing methods
    specific to the Rainwise comma-separated text format, including column
    renaming and timestamp parsing.

    Parameters
    ----------
    filepath : str
        Path to the ``.csv`` file exported by the Rainwise station software.
    """

    def _load_raw_dataframe(self):
        """
        Read raw records from the Rainwise CSV file into a DataFrame.

        Reads the file at ``self.filepath`` using the default ``pandas``
        CSV parser. No column renaming, type casting, or missing-value
        handling is applied at this stage.

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame with columns and dtypes as inferred directly
            by ``pd.read_csv`` from the Rainwise export file.
        """
        df = pd.read_csv(self.filepath)
        return df

    def _standardize_columns(self, df):
        """
        Rename columns, parse timestamps, and cast numerics to float.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame as returned by ``_load_raw_dataframe``, with
            Rainwise-native column names and mixed-type values.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame indexed by ``date`` (``datetime64[ns]``) with
            standardized column names following the ``variable[unit]``
            convention. Only the following columns are retained (when present):
            ``rain[mm]``, ``air_temp[C]``, ``air_humidity[%]``,
            ``pressure[hPa]``, ``wind_speed[m/s]``, ``wind_direction[°]``,
            ``solar_radiation[W/m2]``.
        """
        rename_map = {
            'Interval Precip':      'rain[mm]',
            'Temp Avg':             'air_temp[C]',
            'Hum Avg':              'air_humidity[%]',
            'Baro Avg':             'pressure[hPa]',
            'Windspeed':            'wind_speed[m/s]',
            'Wind Direction':       'wind_direction[°]',
            'Solar Radiation Avg':  'solar_radiation[W/m2]',
        }

        df['date'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.drop(columns=['Time'])
        df = df.set_index('date')

        df = df.rename(columns=rename_map)
        keep = [c for c in rename_map.values() if c in df.columns]
        df = df[keep]

        df[keep] = df[keep].apply(pd.to_numeric, errors='coerce')

        return df
    
    def _compute_direction_degrees(self, df):
        """
        Pass through the DataFrame unchanged.

        Rainwise exports ``wind_direction[°]`` already in decimal degrees
        (0–360), so no conversion is required.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing a ``wind_direction[°]`` column with wind
            direction values already in decimal degrees.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame, unmodified.
        """
        return df